"""Record reference trajectories of the pre-refactor integrators.

Run ONCE, on the tree as it stands before the ML-MCTDHF refactors of
``SemiDirectCI_n_species`` and ``integrators``::

    python -m tests._make_reference        # from the package root

It writes ``tests/reference/*.npz`` on CPU and ``tests/reference_gpu/*.npz``
on GPU (see :func:`reference_dir` — XLA backends do not share floating-point
associativity, so a recording pins only its own backend).
``tests/test_refactor_pins.py`` then replays exactly the same configurations
and asserts bit-identity, so the kernel extractions cannot silently reorder
arithmetic.

Recording a backend AFTER the refactors would make the pin vacuous. To add a
backend later, run it on a checkout of the pre-refactor version (v0.7.3,
commit ``abd0605`` of the development repository, which is not part of this
repository's history) and move the result back:

    cp tests/_make_reference.py tests/__init__.py /tmp/baseline/tests/
    (cd /tmp/baseline && python -m tests._make_reference)
    cp -r /tmp/baseline/tests/reference_gpu tests/

The system is a toy: three species on a shared sine-DVR grid with
N = [1, 2, 1], M = [3, 4, 3], intra-species interaction on the two-particle
species and inter-species interaction on every pair.

The module also exposes the builders (:func:`build_solver`,
:func:`initial_state`, :func:`reference_cases`) so the pin test constructs
bit-identically the same inputs.
"""

import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from mctdhf.MCTDHF_n_species import MCTDHF_n_species_DVR
from mctdhf.integrators import (
    get_adaptive_cmf_half_step_integrator,
    get_cmf_half_step_integrator,
    get_cmf_integrator,
    get_cmf_midpoint_integrator,
)
from mctdhf.sine_dvr import SineDVR, harmonic_potential, softened_coulomb

_HERE = os.path.dirname(os.path.abspath(__file__))


def reference_dir(backend=None):
    """Where the pinned trajectories for the CURRENT backend live.

    Floating-point associativity differs between XLA backends, so a CPU
    recording cannot pin a GPU run and vice versa. Each backend gets its own
    directory (``reference`` for CPU, ``reference_gpu`` for GPU) and
    ``test_refactor_pins.py`` skips with a clear message when the one it
    needs has not been recorded.
    """
    if backend is None:
        backend = jax.default_backend()
    name = "reference" if backend == "cpu" else f"reference_{backend}"
    return os.path.join(_HERE, name)


# Kept as a module attribute for convenience; prefer reference_dir().
REFERENCE_DIR = reference_dir()

# ── The toy system ────────────────────────────────────────────────────────

NUM_PARTICLES = [1, 2, 1]
NUM_ORBITALS = [3, 4, 3]
NUM_GRID = 8
NUM_STATES = 2
STATE_WEIGHTS = [0.7, 0.3]
DT = 0.02
NUM_STEPS = 3


def build_solver(imag_time, e_shift=0.0, ci_sparse=True):
    """The three-species DVR toy solver."""
    dvr = SineDVR(x_min=-4.0, x_max=4.0, num_points=NUM_GRID)

    h0 = dvr.get_h(potential=harmonic_potential(omega=1.0, center=-1.5))
    h1 = dvr.get_h(potential=harmonic_potential(omega=1.2, center=0.0))
    h2 = dvr.get_h(potential=harmonic_potential(omega=0.9, center=1.5))

    g_intra = dvr.get_g(softened_coulomb(kappa=0.8, a=0.5))
    g_inter = dvr.get_g(softened_coulomb(kappa=0.5, a=0.7))

    g_dict = {
        (1, 1): g_intra,  # only species 1 has N > 1
        (0, 1): g_inter,
        (0, 2): g_inter,
        (1, 2): g_inter,
    }

    return MCTDHF_n_species_DVR(
        h_list=[h0, h1, h2],
        g_dict=g_dict,
        num_particles=NUM_PARTICLES,
        num_mctdhf_orbitals=NUM_ORBITALS,
        num_spatial_orbitals=[NUM_GRID] * 3,
        num_states=NUM_STATES,
        imag_time=imag_time,
        state_weights=STATE_WEIGHTS,
        e_shift=e_shift,
        ci_sparse=ci_sparse,
    )


def initial_state(mctdh, seed=20260921):
    """A fixed, reproducible (C, bs) pair for the toy solver."""
    rng = np.random.default_rng(seed)

    bs = []
    for s in range(mctdh.num_species):
        shape = (mctdh.num_spatial_orbitals[s], mctdh.num_mctdhf_orbitals[s])
        b = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        b, _ = np.linalg.qr(b)
        bs.append(jnp.asarray(b, dtype=jnp.complex128))

    c_shape = (mctdh.num_states,) + tuple(mctdh.num_dets)
    C = rng.standard_normal(c_shape) + 1j * rng.standard_normal(c_shape)
    C = jnp.asarray(C, dtype=jnp.complex128)
    C = mctdh.ortonorm_C(C)

    return C, bs


def random_ci_inputs(mctdh, seed=13579):
    """Random MO integrals and a random CI tensor for the CI-op / RDM pins."""
    rng = np.random.default_rng(seed)

    h_list = []
    for s in range(mctdh.num_species):
        m = mctdh.num_mctdhf_orbitals[s]
        a = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
        h_list.append(jnp.asarray(a + a.conj().T, dtype=jnp.complex128))

    g_dict = {}
    for (s1, s2) in [(1, 1), (0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]:
        m1 = mctdh.num_mctdhf_orbitals[s1]
        m2 = mctdh.num_mctdhf_orbitals[s2]
        shape = (m1, m2, m1, m2)
        g = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        g_dict[(s1, s2)] = jnp.asarray(g, dtype=jnp.complex128)

    c_shape = (mctdh.num_states,) + tuple(mctdh.num_dets)
    C = rng.standard_normal(c_shape) + 1j * rng.standard_normal(c_shape)
    C = jnp.asarray(C, dtype=jnp.complex128)

    return h_list, g_dict, C


# ── The recorded configurations ───────────────────────────────────────────
#
# Each entry is (name, factory_name, imag_time, e_shift, c_propagator,
# b_propagator, c_kwargs, b_kwargs).

_FACTORIES = {
    "basic": get_cmf_integrator,
    "midpoint": get_cmf_midpoint_integrator,
    "half": get_cmf_half_step_integrator,
}


def reference_cases():
    """The integrator configurations recorded by this script."""
    cases = []
    for factory in ("basic", "midpoint"):
        for imag_time in (False, True):
            cases.append(
                (
                    f"{factory}_{'ite' if imag_time else 'rte'}_sil_diag",
                    factory,
                    imag_time,
                    0.0,
                    "sil",
                    "diagonalize",
                    {"krylov_dim": 6},
                    {},
                )
            )
            cases.append(
                (
                    f"{factory}_{'ite' if imag_time else 'rte'}_cn_dop",
                    factory,
                    imag_time,
                    -1.5,
                    "cn",
                    "dop",
                    {"solver_tol": 1e-10, "max_solver_iter": 40},
                    {"n_substeps": 2, "n_orbital_substeps": 2},
                )
            )
            cases.append(
                (
                    f"{factory}_{'ite' if imag_time else 'rte'}_diag_cn",
                    factory,
                    imag_time,
                    0.0,
                    "diagonalize",
                    "cn",
                    {},
                    {"solver_tol": 1e-10, "max_solver_iter": 40,
                     "n_orbital_substeps": 3},
                )
            )
    # The half-step integrator is not an ML target, but its call sites move
    # in the refactor, so pin it too.
    cases.append(
        ("half_rte_sil_sil", "half", False, 0.0, "sil", "sil",
         {"krylov_dim": 6}, {"krylov_dim": 8})
    )
    cases.append(
        ("half_ite_cn_cn", "half", True, 0.0, "cn", "cn",
         {"solver_tol": 1e-10, "max_solver_iter": 40},
         {"solver_tol": 1e-10, "max_solver_iter": 40, "n_orbital_substeps": 2})
    )
    return cases


def run_case(case):
    """Propagate one configuration; returns the stacked trajectory."""
    (_name, factory, imag_time, e_shift, c_prop, b_prop, c_kwargs, b_kwargs) = case

    mctdh = build_solver(imag_time, e_shift=e_shift)
    C, bs = initial_state(mctdh)
    y = mctdh._Cb_to_y(C, bs)

    integrator = _FACTORIES[factory](
        mctdh, DT, c_propagator=c_prop, b_propagator=b_prop,
        c_kwargs=c_kwargs, b_kwargs=b_kwargs,
    )

    ys = [np.asarray(y)]
    t = 0.0
    for _ in range(NUM_STEPS):
        t, y = integrator(t, y)
        ys.append(np.asarray(y))
    return np.stack(ys)


def run_adaptive():
    """One adaptive half-step; returns (y_new, err_sil, err_split)."""
    mctdh = build_solver(imag_time=False)
    C, bs = initial_state(mctdh)
    y = mctdh._Cb_to_y(C, bs)
    step = get_adaptive_cmf_half_step_integrator(
        mctdh, sil_krylov_dim=6, sil_b_krylov_dim=8
    )
    t = 0.0
    out = {}
    for i in range(NUM_STEPS):
        t, y, err_sil, err_split = step(t, y, DT)
        out[f"y_{i}"] = np.asarray(y)
        out[f"err_sil_{i}"] = np.asarray(err_sil)
        out[f"err_split_{i}"] = np.asarray(err_split)
    return out


def run_ci_ops(ci_sparse):
    """precompute_ci_ops / apply_ci_ops / RDM outputs on random inputs."""
    mctdh = build_solver(imag_time=False, ci_sparse=ci_sparse)
    h_list, g_dict, C = random_ci_inputs(mctdh)

    ops_diag, ops_pair = mctdh.dCI.precompute_ci_ops(h_list, g_dict)
    out = {}
    for s, op in enumerate(ops_diag):
        out[f"ops_diag_{s}"] = np.asarray(op)
    for (s1, s2), op in sorted(ops_pair.items()):
        out[f"ops_pair_{s1}_{s2}"] = np.asarray(op)

    out["sigma"] = np.asarray(mctdh.dCI.apply_ci_ops((ops_diag, ops_pair), C))

    for s, D in enumerate(mctdh.dCI.get_1p_RDMs(C)):
        out[f"D_{s}"] = np.asarray(D)
    for (s1, s2), d in sorted(mctdh.dCI.get_2p_RDMs(C).items()):
        out[f"d_{s1}_{s2}"] = np.asarray(d)

    w = mctdh.state_weights
    for s, D in enumerate(mctdh.dCI.get_1p_RDMs(C, weights=w)):
        out[f"Dw_{s}"] = np.asarray(D)
    for (s1, s2), d in sorted(mctdh.dCI.get_2p_RDMs(C, weights=w).items()):
        out[f"dw_{s1}_{s2}"] = np.asarray(d)

    return out


def main():
    out_dir = reference_dir()
    os.makedirs(out_dir, exist_ok=True)
    print(f"backend={jax.default_backend()} devices={jax.devices()}")
    print(f"writing to {out_dir}")

    for case in reference_cases():
        name = case[0]
        traj = run_case(case)
        path = os.path.join(out_dir, f"traj_{name}.npz")
        np.savez_compressed(path, ys=traj)
        print(f"wrote {path}  {traj.shape} {traj.dtype}")

    path = os.path.join(out_dir, "adaptive.npz")
    np.savez_compressed(path, **run_adaptive())
    print(f"wrote {path}")

    for ci_sparse in (True, False):
        tag = "sparse" if ci_sparse else "dense"
        path = os.path.join(out_dir, f"ci_ops_{tag}.npz")
        np.savez_compressed(path, **run_ci_ops(ci_sparse))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
