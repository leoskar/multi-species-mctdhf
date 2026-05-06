"""Optimise (alpha, beta) for the four-anharmonic-wells setup so the lowest
many-body eigenstates are close to 4-qubit Hartree products.

Strategy: scipy Nelder-Mead on the 8-vector [alpha_0..alpha_3, beta_0..beta_3]
(in GHz), with warm-starting across evaluations — the previously converged
50-state manifold is re-used as the initial guess for the next ITE run.

Run with:
    test/venv/bin/python test/optimize_qubit_params.py
"""

from __future__ import annotations

import os
import sys
import time
from itertools import product

import numpy as np
import jax.numpy as jnp
from opt_einsum import contract
import pybobyqa

from mctdhf.integrators import get_cmf_integrator

# Make `mctdhf` and `utils` importable regardless of cwd.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from mctdhf import MCTDHF_n_species_DVR, SineDVR, get_cmf_half_step_integrator
from utils import (
    anharm_pot,
    iter_bitstrings,
    match_configs,
    qubit_corner_fidelities,
)

# ── Fixed geometry (independent of alpha, beta) ───────────────────────────────

# Unit conventions mirror test/four_anharmonic_wells.ipynb (pint-based).
import pint

_ureg = pint.UnitRegistry()
_typical_freq = 10 * _ureg.GHz
_omega = 2 * np.pi * _typical_freq
_energy_to_frequency = 1 / _ureg.planck_constant
_energy_unit = _ureg.hbar * _omega
_length_unit = np.sqrt(_ureg.hbar / (_ureg.m_e * _omega))

TYPICAL_FREQ_GHZ = 10.0
LENGTH_UNIT_UM = _length_unit.to(_ureg.micrometer).magnitude
KAPPA = (
    _length_unit * _ureg.m_e * _ureg.e**2
    / (4 * _ureg.pi * _ureg.epsilon_0 * _ureg.hbar**2)
).to(_ureg.dimensionless).magnitude

NUM_WELLS = 4
NUM_DVR = 64
NUM_MCTDHF_ORBITALS = 4
NUM_STATES = 50

WELL_CENTERS_UM = np.array([-1.5, -0.5, 0.5, 1.5])
WELL_WIDTHS_UM = np.array([1.0, 1.0, 1.0, 1.0])

WELL_CENTERS = WELL_CENTERS_UM / LENGTH_UNIT_UM
WELL_WIDTHS = WELL_WIDTHS_UM / LENGTH_UNIT_UM

# Alpha/beta bounds in GHz.
ALPHA_BOUNDS_GHZ = (5.0, 15.0)
BETA_BOUNDS_GHZ = (0.02, 0.4)

CONFIGS = iter_bitstrings(NUM_WELLS, 2)  # 16 four-qubit configs


def _ghz_to_dimensionless(x_ghz):
    return np.asarray(x_ghz) / TYPICAL_FREQ_GHZ


def _coulomb(x1, x2):
    return KAPPA / jnp.sqrt((x1 - x2) ** 2)


def _build_dvr_bases():
    return [
        SineDVR(
            x_min=centre - 0.5 * width,
            x_max=centre + 0.5 * width,
            num_points=NUM_DVR,
        )
        for centre, width in zip(WELL_CENTERS, WELL_WIDTHS)
    ]


def _build_g_dict(dvr_bases):
    g_dict = {}
    for s1 in range(NUM_WELLS):
        for s2 in range(s1 + 1, NUM_WELLS):
            g = dvr_bases[s1].get_g_cross_species(dvr_bases[s2], _coulomb)
            g_dict[(s1, s2)] = jnp.array(g, dtype=jnp.float32)
    return g_dict


DVR_BASES = _build_dvr_bases()
G_DICT = _build_g_dict(DVR_BASES)


def _build_h_list(alphas_dim, betas_dim):
    h_list = []
    for s, (a, b, centre) in enumerate(zip(alphas_dim, betas_dim, WELL_CENTERS)):
        pot = anharm_pot(a, b, centre)
        h = DVR_BASES[s].get_h(potential=pot)
        h_list.append(jnp.array(h, dtype=jnp.float32))
    return h_list


# ── Rayleigh–Ritz post-diagonalisation ────────────────────────────────────────


def _rayleigh_ritz(mctdhf, C, bs, t=0.0):
    """Diagonalise H on the span of the converged states.

    With `c_propagator="sil"` (or "cn"), state-averaged ITE only
    orthonormalises the propagated set; it does not diagonalise H within that
    subspace, so individual rows of C are not eigenstates. We rebuild
    H_{IJ} = ⟨Ψ_I|H|Ψ_J⟩ on the (num_states)-dim subspace and diagonalise so
    each row becomes a true eigenstate of H on that span. Orbitals are
    unchanged — the rotation is purely in CI space. The propagated states are
    exactly orthonormal coming out of `mctdhf.integrate(..., normalize=True)`
    (sequential QR on `C_flat.T`), so this is a standard (not generalised)
    Hermitian eigenproblem.

    With `c_propagator="diagonalize"`, every CI step already replaces C by the
    lowest-N eigenvectors of H_CI(b_t), so at convergence this RR call is a
    no-op modulo row sign/ordering. We still run it to (a) recover the
    per-state eigenvalues (the integrator does not surface them) and (b)
    repin near-degenerate blocks that FP noise can leave as a mixed
    orthonormal basis.
    """
    h_current = mctdhf._get_h_current(t)
    h_MO_list = [
        mctdhf.transform_h(h_current[s], bs[s])[1]
        for s in range(mctdhf.num_species)
    ]
    g_MO_dict, _ = mctdhf._transform_all_g(bs, full=False)

    sigma = mctdhf.dCI.get_sigma(C, h_MO_list, g_MO_dict)

    S = C.shape[0]
    C_flat = jnp.reshape(C, (S, -1))
    sigma_flat = jnp.reshape(sigma, (S, -1))

    H_mat = C_flat.conj() @ sigma_flat.T
    H_mat = 0.5 * (H_mat + H_mat.conj().T)

    evals, evecs = jnp.linalg.eigh(H_mat)
    C_new = contract("Jk,J...->k...", evecs, C)
    return C_new, np.real(evals)


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate_params(
    alphas_ghz,
    betas_ghz,
    y_prev=None,
    num_steps=2000,
    dt=1e-2,
    save_every=100,
    verbose=False,
    config_weights=None,
):
    """Run ITE at (alpha, beta), return qubit fidelities and the final state.

    When `y_prev` is given, it is used as the initial guess (warm start) with
    noise_C = 0. For cold starts (y_prev=None) we use the default single-particle
    product-state construction plus noise_C=1e-3, matching the notebook.
    """
    alphas_dim = _ghz_to_dimensionless(alphas_ghz)
    betas_dim = _ghz_to_dimensionless(betas_ghz)

    h_list = _build_h_list(alphas_dim, betas_dim)

    mctdhf = MCTDHF_n_species_DVR(
        h_list=h_list,
        g_dict=G_DICT,
        num_particles=[1] * NUM_WELLS,
        num_mctdhf_orbitals=[NUM_MCTDHF_ORBITALS] * NUM_WELLS,
        num_spatial_orbitals=[NUM_DVR] * NUM_WELLS,
        num_states=NUM_STATES,
        imag_time=True,
        state_weights=np.concatenate([np.ones(1), 0.5*np.ones(4), np.zeros(NUM_STATES-5)])
    )

    integrator = get_cmf_integrator(
        mctdhf, dt, c_propagator="diagonalize", b_propagator="sil"
    )

    if y_prev is None:
        ts, ys = mctdhf.integrate(
            integrator=integrator,
            num_steps=num_steps,
            t_init=0.0,
            normalize=True,
            noise_bs=0,
            noise_C=1e-3,
            save_every=save_every,
        )
    else:
        C_prev, bs_prev = mctdhf._y_to_Cb(y_prev)
        ts, ys = mctdhf.integrate(
            integrator=integrator,
            num_steps=num_steps,
            t_init=0.0,
            bs_init_list=bs_prev,
            C_init=C_prev,
            normalize=True,
            noise_bs=0.0,
            noise_C=0.0,
            save_every=save_every,
        )

    y_final = ys[-1]
    C_final, bs_final = mctdhf._y_to_Cb(y_final)

    # Rayleigh–Ritz to get energies
    C_final, energies = _rayleigh_ritz(mctdhf, C_final, bs_final)
    y_final = mctdhf._Cb_to_y(C_final, bs_final)

    # State-averaged NO rotation: per-species, eigh of the sum-over-states
    # 1-RDM and sort by descending occupation. After this the leading 2 columns
    # of each b_s are the qubit |0>_s, |1>_s modes (most-occupied across the
    # 50 propagated states), so the qubit fidelity is just the (0/1)^K corner
    # of |C[k, ...]|^2.
    y_final = mctdhf.rotate_to_natural_orbitals(y_final)
    C_final, bs_final = mctdhf._y_to_Cb(y_final)

    F = qubit_corner_fidelities(C_final, NUM_WELLS, n_levels=2)
    matched_idx, fidelities, loss = match_configs(F, weights=config_weights)

    if verbose:
        print(f"  loss={loss:.4f}  min_fid={fidelities.min():.4f}")

    return dict(
        loss=loss,
        fidelities=fidelities,
        matched_idx=matched_idx,
        F=np.asarray(F),
        energies=energies,
        y_final=y_final,
        mctdhf=mctdhf,
    )


# ── Optimisation ──────────────────────────────────────────────────────────────


def optimize(
    x0=None,
    num_steps_cold=1000,
    num_steps_warm=200,
    maxiter=200,
    rhobeg=0.1,
    rhoend=1e-3,
    log_path=None,
    config_weights=None,
):
    """Run PyBOBYQA over [alpha_0..alpha_3, beta_0..beta_3] in GHz.

    Trust-region derivative-free optimizer with quadratic local models.
    Variables are internally rescaled to [0, 1] using the bounds
    (`scaling_within_bounds=True`), so `rhobeg` and `rhoend` are interpreted
    in normalized space — i.e. as fractions of each parameter's bound range.
    Defaults: `rhoend=1e-3` ≈ 10 MHz on α (range 10 GHz) and ≈ 0.4 MHz on β
    (range 0.38 GHz); `rhobeg=0.1` is 10% of each range. Warm-starts ITE from
    the previous converged state. Bounds are enforced natively by PyBOBYQA.
    """
    if x0 is None:
        x0 = np.array([7.0, 9.0, 11.0, 13.0, 0.25, 0.25, 0.25, 0.25])

    lb = np.array(
        [ALPHA_BOUNDS_GHZ[0]] * NUM_WELLS + [BETA_BOUNDS_GHZ[0]] * NUM_WELLS
    )
    ub = np.array(
        [ALPHA_BOUNDS_GHZ[1]] * NUM_WELLS + [BETA_BOUNDS_GHZ[1]] * NUM_WELLS
    )

    state = dict(
        y_prev=None,
        best_loss=np.inf,
        best_x=None,
        trajectory=[],
        eval_idx=0,
    )

    def loss_fn(x):
        state["eval_idx"] += 1
        i = state["eval_idx"]
        t0 = time.time()

        num_steps = num_steps_cold if state["y_prev"] is None else num_steps_warm
        result = evaluate_params(
            alphas_ghz=x[:NUM_WELLS],
            betas_ghz=x[NUM_WELLS:],
            y_prev=state["y_prev"],
            num_steps=num_steps,
            config_weights=config_weights,
        )
        loss = result["loss"]
        min_fid = float(result["fidelities"].min())
        wall = time.time() - t0

        worst_idx = int(np.argmin(result["fidelities"]))
        worst_cfg = CONFIGS[worst_idx]
        print(
            f"[{i:03d}] α={np.round(x[:4], 3)}  β={np.round(x[4:], 3)}  "
            f"loss={loss:.4f}  min_fid={min_fid:.4f} (cfg {worst_cfg})  "
            f"warm={state['y_prev'] is not None}  {wall:.1f}s",
            flush=True,
        )

        state["y_prev"] = result["y_final"]

        if loss < state["best_loss"]:
            state["best_loss"] = loss
            state["best_x"] = x.copy()
            state["best_result"] = result

        state["trajectory"].append(
            dict(
                eval=i,
                x=x.copy(),
                loss=loss,
                min_fid=min_fid,
                fidelities=result["fidelities"].copy(),
                matched_idx=result["matched_idx"].copy(),
                energies=result["energies"].copy(),
                wall=wall,
                oob=False,
            )
        )
        return loss

    print("Starting PyBOBYQA optimisation ...")
    print(f"  x0 (GHz) = α={x0[:4]}, β={x0[4:]}")
    print(f"  num_steps cold/warm = {num_steps_cold}/{num_steps_warm}")
    print(f"  maxfun={maxiter}, rhobeg={rhobeg}, rhoend={rhoend}")
    t_start = time.time()
    soln = pybobyqa.solve(
        loss_fn,
        x0,
        bounds=(lb, ub),
        rhobeg=rhobeg,
        rhoend=rhoend,
        maxfun=maxiter,
        scaling_within_bounds=True,
        print_progress=False,
        seek_global_minimum=True,
    )
    wall_total = time.time() - t_start
    print(f"\nOptimisation finished in {wall_total:.1f}s ({state['eval_idx']} evals).")
    print(f"Solver flag: {soln.flag} ({soln.msg})")
    print(f"Best loss = {state['best_loss']:.4f}")
    print(
        f"Best α (GHz) = {state['best_x'][:4]}\nBest β (GHz) = {state['best_x'][4:]}"
    )

    if log_path is not None:
        _save_trajectory(log_path, state, soln, wall_total)

    return state, soln


def _save_trajectory(log_path, state, soln, wall_total):
    """Save optimisation trajectory and best result to an .npz archive."""
    traj = state["trajectory"]
    n = len(traj)
    xs = np.stack([t["x"] for t in traj])
    losses = np.array([t["loss"] for t in traj])
    min_fids = np.array([t["min_fid"] for t in traj])
    walls = np.array([t["wall"] for t in traj])
    oob = np.array([t["oob"] for t in traj])

    fid_rows = [t for t in traj if not t["oob"]]
    fid_xs = (
        np.stack([t["x"] for t in fid_rows]) if fid_rows else np.zeros((0, 8))
    )
    fid_matrix = (
        np.stack([t["fidelities"] for t in fid_rows])
        if fid_rows
        else np.zeros((0, len(CONFIGS)))
    )
    matched_matrix = (
        np.stack([t["matched_idx"] for t in fid_rows])
        if fid_rows
        else np.zeros((0, len(CONFIGS)), dtype=int)
    )
    energies_matrix = (
        np.stack([t["energies"] for t in fid_rows])
        if fid_rows
        else np.zeros((0, NUM_STATES))
    )

    np.savez(
        log_path,
        xs=xs,
        losses=losses,
        min_fids=min_fids,
        walls=walls,
        oob=oob,
        best_x=state["best_x"],
        best_loss=state["best_loss"],
        solver_x=soln.x,
        solver_fun=soln.f,
        solver_flag=soln.flag,
        solver_msg=soln.msg,
        wall_total=wall_total,
        configs=np.array(CONFIGS),
        fid_xs=fid_xs,
        fidelities=fid_matrix,
        matched_idx=matched_matrix,
        energies=energies_matrix,
    )
    print(f"Trajectory saved to {log_path}")


if __name__ == "__main__":

    name = "_" + sys.argv[1] if len(sys.argv) > 1 else ""

    out_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(out_dir, "optimization_trajectories/" + name + ".npz")
    config_weights = w = np.array([3.0, 3.,3.,3.,3.,  1.,1.,1.,1.,1.,1.,  0.3,0.3,0.3,0.3,  0.3])
    state, soln = optimize(log_path=log_path, config_weights=config_weights)
