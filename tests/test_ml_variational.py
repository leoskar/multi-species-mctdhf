"""Variational behaviour of ML-MCTDHF at TRUNCATED rank.

Three things are checked, none of which the full-rank exactness tests can see:

* **dt ladder in real time** — per-state norm drift and energy drift must
  strictly decrease as dt is halved at fixed total propagation time. No
  convergence ORDER is asserted: the basic Lie–Trotter split shows none
  worth pinning. ``n_orbital_substeps`` and ``n_spf_substeps`` are
  held fixed down the ladder, so the frozen-projector substeps ``dt/n``
  shrink with dt as well.
* **imaginary time** — the energy must be non-increasing once the first few
  transient steps are past.
* **gauge** — ``make_regauge_ml`` must leave every physical observable alone
  while restoring ``B†B = 1`` and ``b†b = 1``.
* **convergence in n_s** — the ML observables must approach the standard
  MCTDHF ones as ``n_s → D_s``, and the error must track the SPF population
  that the truncation discards.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from mctdhf.MCTDHF_n_species import MCTDHF_n_species_DVR, pack_Cb_to_y
from mctdhf.MLMCTDHF_n_species import MLMCTDHF_n_species_DVR, make_regauge_ml
from mctdhf.integrators import get_cmf_midpoint_integrator
from mctdhf.integrators_ml import (
    get_ml_cmf_integrator,
    get_ml_cmf_midpoint_integrator,
)
from mctdhf.sine_dvr import SineDVR, harmonic_potential, softened_coulomb

NUM_PARTICLES = [2, 2]
NUM_ORBITALS = [4, 4]        # D_s = 6 per species
NUM_GRID = 10


def ingredients(coupling=2.0):
    """Two two-particle species with a STRONG inter-species interaction.

    Strong coupling is the point: it is what populates the SPF spectrum and
    makes the ``n_s`` truncation bite, so the convergence test has something
    to converge.
    """
    dvr = SineDVR(x_min=-4.0, x_max=4.0, num_points=NUM_GRID)
    h_list = [
        dvr.get_h(potential=harmonic_potential(omega=1.0, center=-1.0)),
        dvr.get_h(potential=harmonic_potential(omega=1.1, center=1.0)),
    ]
    g_dict = {
        (0, 0): dvr.get_g(softened_coulomb(kappa=0.7, a=0.5)),
        (1, 1): dvr.get_g(softened_coulomb(kappa=0.7, a=0.5)),
        (0, 1): dvr.get_g(softened_coulomb(kappa=coupling, a=0.8)),
    }
    return h_list, g_dict


def make_ml(num_spfs, imag_time=False, num_states=1, state_weights=None,
            coupling=2.0):
    h_list, g_dict = ingredients(coupling)
    return MLMCTDHF_n_species_DVR(
        h_list=h_list, g_dict=g_dict, num_particles=NUM_PARTICLES,
        num_mctdhf_orbitals=NUM_ORBITALS,
        num_spatial_orbitals=[NUM_GRID] * 2, num_states=num_states,
        imag_time=imag_time, state_weights=state_weights,
        num_spfs=list(num_spfs),
    )


def make_std(imag_time=False, num_states=1, state_weights=None, coupling=2.0):
    h_list, g_dict = ingredients(coupling)
    return MCTDHF_n_species_DVR(
        h_list=h_list, g_dict=g_dict, num_particles=NUM_PARTICLES,
        num_mctdhf_orbitals=NUM_ORBITALS,
        num_spatial_orbitals=[NUM_GRID] * 2, num_states=num_states,
        imag_time=imag_time, state_weights=state_weights,
    )


def relaxed_standard_state(seed=99, num_states=1, steps=120, dt=0.05):
    """A physically sensible starting point: an ITE-relaxed standard state."""
    std = make_std(imag_time=True, num_states=num_states)
    rng = np.random.default_rng(seed)
    bs = []
    for s in range(std.num_species):
        shape = (std.num_spatial_orbitals[s], std.num_mctdhf_orbitals[s])
        b = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        b, _ = np.linalg.qr(b)
        bs.append(jnp.asarray(b, dtype=jnp.complex128))
    c_shape = (num_states,) + tuple(std.num_dets)
    C = jnp.asarray(rng.standard_normal(c_shape)
                    + 1j * rng.standard_normal(c_shape))
    C = std.ortonorm_C(C)

    step = get_cmf_midpoint_integrator(
        std, dt, c_propagator="sil", b_propagator="diagonalize",
        c_kwargs={"krylov_dim": 10})
    y = pack_Cb_to_y(C, bs)
    t = 0.0
    for _ in range(steps):
        t, y = step(t, y)
        Ck, bsk = std._y_to_Cb(y)
        y = pack_Cb_to_y(std.ortonorm_C(Ck),
                         [std.ortonorm_b(b) for b in bsk])
    return std, std._y_to_Cb(y)


# ── real-time dt ladder ───────────────────────────────────────────────────


def _drifts(ml, step_factory, y0, T, dt, n_sub):
    """Propagate to T and return (max norm drift, max energy drift)."""
    step = step_factory(
        ml, dt, a_propagator="sil", B_propagator="diagonalize",
        b_propagator="diagonalize", a_kwargs={"krylov_dim": 10},
        B_kwargs={"n_spf_substeps": n_sub},
        b_kwargs={"n_orbital_substeps": n_sub},
    )
    num_steps = int(round(T / dt))
    n0 = np.real(np.diag(np.asarray(ml.calculate_overlap(y0, y0))))
    E0 = np.real(np.asarray(ml.calculate_energy(0.0, y0)))

    t, y = 0.0, y0
    for _ in range(num_steps):
        t, y = step(t, y)
    n1 = np.real(np.diag(np.asarray(ml.calculate_overlap(y, y))))
    E1 = np.real(np.asarray(ml.calculate_energy(t, y)))
    return float(np.max(np.abs(n1 - n0))), float(np.max(np.abs(E1 - E0)))


@pytest.mark.parametrize(
    "factory,name",
    [(get_ml_cmf_integrator, "basic"),
     (get_ml_cmf_midpoint_integrator, "midpoint")],
    ids=["basic", "midpoint"],
)
def test_rte_drifts_decrease_down_the_dt_ladder(factory, name):
    std, (C, bs) = relaxed_standard_state()
    ml = make_ml([3, 3])
    A, Bs, bs_ml, _disc = ml.from_mctdhf(C, bs)
    y0 = ml._ABb_to_y(A, Bs, bs_ml)

    T = 0.4
    results = []
    for dt in (0.1, 0.05, 0.025):
        results.append(_drifts(ml, factory, y0, T, dt, n_sub=2))

    norms = [r[0] for r in results]
    energies = [r[1] for r in results]
    assert norms[0] > norms[1] > norms[2], (name, "norm drift", norms)
    assert energies[0] > energies[1] > energies[2], (name, "energy drift",
                                                     energies)


def test_ite_energy_is_non_increasing():
    ml = make_ml([3, 3], imag_time=True)
    step = get_ml_cmf_integrator(
        ml, 0.02, a_propagator="sil", B_propagator="diagonalize",
        b_propagator="diagonalize", a_kwargs={"krylov_dim": 10})

    ts, ys = ml.integrate(step, 60, normalize=True, save_every=1)
    energies = np.array(
        [float(np.real(ml.calculate_energy(float(t), y))[0])
         for t, y in zip(np.asarray(ts), np.asarray(ys))]
    )
    # Skip the first few transient steps.
    tail = energies[5:]
    diffs = np.diff(tail)
    assert np.all(diffs <= 1e-8), (
        f"energy increased in ITE: max rise {diffs.max():.3e}"
    )
    assert tail[-1] < tail[0]


# ── gauge ─────────────────────────────────────────────────────────────────


def test_regauge_is_an_exact_gauge_transform():
    """``make_regauge_ml`` restores orthonormality without touching Ψ.

    The overlap is an EXACT invariant: the Slater-determinant overlap
    ``det(O[occ_I, occ_J])`` is valid for any two orbital sets, orthonormal
    or not, so a gauge change cannot alter it.

    ``calculate_energy`` is an exact invariant only once the orbitals are
    orthonormal — the Slater–Condon assembly it uses assumes that — so it is
    checked at 1e-12 on an orthonormal state, and, on a drifted one, against
    the size of the drift.
    """
    std, (C, bs) = relaxed_standard_state()
    ml = make_ml([4, 4])
    A, Bs, bs_ml, _disc = ml.from_mctdhf(C, bs)
    y_ref = ml._ABb_to_y(A, Bs, bs_ml)
    regauge = make_regauge_ml(ml)

    # (a) orthonormal input: everything is invariant to machine precision
    y_gauged = regauge(y_ref)
    np.testing.assert_allclose(
        np.asarray(ml.calculate_energy(0.0, y_gauged)),
        np.asarray(ml.calculate_energy(0.0, y_ref)),
        rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(ml.calculate_overlap(y_gauged, y_ref)),
        np.asarray(ml.calculate_overlap(y_ref, y_ref)),
        rtol=0, atol=1e-12)

    # (b) drifted input: the overlap is still exactly invariant, and
    #     orthonormality is restored.
    rng = np.random.default_rng(3)
    delta = 1e-7
    A_d, Bs_d, bs_d = ml._y_to_ABb(y_ref)
    Bs_d = [B + delta * jnp.asarray(rng.standard_normal(B.shape))
            for B in Bs_d]
    bs_d = [b + delta * jnp.asarray(rng.standard_normal(b.shape))
            for b in bs_d]
    y_drift = ml._ABb_to_y(A_d, Bs_d, bs_d)

    ovl_before = np.asarray(ml.calculate_overlap(y_drift, y_ref))
    y_fixed = regauge(y_drift)
    ovl_after = np.asarray(ml.calculate_overlap(y_fixed, y_ref))
    np.testing.assert_allclose(ovl_after, ovl_before, rtol=0, atol=1e-12)

    A_f, Bs_f, bs_f = ml._y_to_ABb(y_fixed)
    for s in range(ml.num_species):
        np.testing.assert_allclose(
            np.asarray(Bs_f[s].conj().T @ Bs_f[s]),
            np.eye(ml.num_spfs[s]), rtol=0, atol=1e-12, err_msg=f"B†B [{s}]")
        np.testing.assert_allclose(
            np.asarray(bs_f[s].conj().T @ bs_f[s]),
            np.eye(ml.num_mctdhf_orbitals[s]), rtol=0, atol=1e-12,
            err_msg=f"b†b [{s}]")

    # The energy is only a formal invariant here: the Slater-Condon assembly
    # assumes orthonormal orbitals, so on a drifted state it can move, but
    # only by the size of the defect.
    dE = abs(float(np.real(ml.calculate_energy(0.0, y_fixed))[0])
             - float(np.real(ml.calculate_energy(0.0, y_drift))[0]))
    assert dE < 1e-5, f"energy moved by {dE:.3e}, far more than the {delta:g} defect"


def test_rotate_to_natural_spfs_is_an_exact_gauge_transform():
    std, (C, bs) = relaxed_standard_state()
    ml = make_ml([4, 4])
    A, Bs, bs_ml, _disc = ml.from_mctdhf(C, bs)
    y = ml._ABb_to_y(A, Bs, bs_ml)

    y_nat = ml.rotate_to_natural_spfs(y)
    np.testing.assert_allclose(
        np.asarray(ml.calculate_energy(0.0, y_nat)),
        np.asarray(ml.calculate_energy(0.0, y)), rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(ml.calculate_overlap(y_nat, y)),
        np.asarray(ml.calculate_overlap(y, y)), rtol=0, atol=1e-12)

    A_n, Bs_n, _ = ml._y_to_ABb(y_nat)
    for s in range(ml.num_species):
        np.testing.assert_allclose(
            np.asarray(Bs_n[s].conj().T @ Bs_n[s]),
            np.eye(ml.num_spfs[s]), rtol=0, atol=1e-11)
        rho = np.asarray(ml.dCI._compute_rho_ci(A_n, s, ml.state_weights))
        off = rho - np.diag(np.diag(rho))
        assert np.max(np.abs(off)) < 1e-11, f"rho[{s}] not diagonal"
        d = np.real(np.diag(rho))
        assert np.all(np.diff(d) <= 1e-12), f"rho[{s}] not descending: {d}"


# ── convergence in n_s ────────────────────────────────────────────────────


def test_observables_converge_as_n_s_approaches_D_s():
    """ML → standard MCTDHF as the SPF basis is completed.

    The reference is a standard-MCTDHF real-time trajectory from the same
    initial state; the ML trajectories are run with the same integrator
    settings at n_s = 2, 3, 4, 6 (D_s = 6). Energy, particle density and the
    overlap with the reference must all improve monotonically, and the error
    must sit below the SPF population the truncation throws away.

    The overlap error is measured against the reference's OWN norm, not
    against 1: real-time CMF does not conserve the norm exactly, and that
    drift is common to both trajectories, so comparing to 1 would report it
    as ML truncation error.
    """
    std, (C, bs) = relaxed_standard_state()
    std_rte = make_std(imag_time=False)
    y_std = pack_Cb_to_y(C, bs)

    dt, num_steps = 0.05, 8
    ref_step = get_cmf_midpoint_integrator(
        std_rte, dt, c_propagator="sil", b_propagator="diagonalize",
        c_kwargs={"krylov_dim": 10})
    t, y_ref = 0.0, y_std
    for _ in range(num_steps):
        t, y_ref = ref_step(t, y_ref)
    C_ref, bs_ref = std_rte._y_to_Cb(y_ref)
    E_ref = float(np.real(std_rte.calculate_energy(t, y_ref))[0])
    dens_ref = [np.asarray(d) for d in
                std_rte.calculate_particle_densities(C_ref, bs_ref)]
    # The reference's own norm, which CMF real-time propagation does not
    # conserve exactly. Subtracting it is what makes the overlap error
    # measure ML-vs-standard disagreement rather than the shared norm drift.
    S_self = complex(np.asarray(std_rte.calculate_overlap(y_ref, y_ref))[0, 0])

    errors, discarded_weights = [], []
    for n in (2, 3, 4, 6):
        ml = make_ml([n, n])
        A, Bs, bs_ml, disc = ml.from_mctdhf(C, bs)
        y0 = ml._ABb_to_y(A, Bs, bs_ml)
        discarded_weights.append(float(np.real(np.max(jnp.stack(disc)))))

        step = get_ml_cmf_midpoint_integrator(
            ml, dt, a_propagator="sil", B_propagator="diagonalize",
            b_propagator="diagonalize", a_kwargs={"krylov_dim": 10})
        tt, y = 0.0, y0
        for _ in range(num_steps):
            tt, y = step(tt, y)

        A_e, Bs_e, bs_e = ml._y_to_ABb(y)
        E = float(np.real(ml.calculate_energy(tt, y))[0])
        dens = [np.asarray(d) for d in
                ml.calculate_particle_densities(A_e, Bs_e, bs_e)]
        # overlap of the ML state with the standard reference trajectory
        ovl = complex(np.asarray(
            std_rte.calculate_overlap(ml.to_mctdhf_y(y), y_ref))[0, 0])

        dens_err = max(float(np.max(np.abs(d - dr)))
                       for d, dr in zip(dens, dens_ref))
        errors.append((abs(E - E_ref), dens_err, abs(ovl - S_self)))

    e_err = [e[0] for e in errors]
    d_err = [e[1] for e in errors]
    o_err = [e[2] for e in errors]

    # Full rank must reproduce the standard trajectory.
    assert e_err[-1] < 1e-9, f"energy at n = D: {e_err[-1]:.3e}"
    assert d_err[-1] < 1e-9, f"density at n = D: {d_err[-1]:.3e}"
    assert o_err[-1] < 1e-10, f"overlap at n = D: {o_err[-1]:.3e}"

    # Monotone improvement with n_s.
    for name, series in (("energy", e_err), ("density", d_err),
                         ("overlap", o_err)):
        assert all(a > b for a, b in zip(series, series[1:])), \
            f"{name} error not monotone in n_s: {series}"

    # The error tracks the discarded SPF weight.
    assert all(a >= b for a, b in
               zip(discarded_weights, discarded_weights[1:])), \
        f"discarded weights not monotone: {discarded_weights}"
    assert discarded_weights[-1] < 1e-12
    for oe, dw in zip(o_err[:-1], discarded_weights[:-1]):
        assert oe < 50 * dw + 1e-9, (o_err, discarded_weights)


def test_spf_populations_match_the_discarded_weight():
    """``spf_populations`` is the convergence monitor for ``n_s``.

    Truncating a full-rank state to n SPFs must discard exactly the weight
    sitting in the populations below the cut.
    """
    std, (C, bs) = relaxed_standard_state()
    C = std.normalize_C(C)

    ml_full = make_ml([6, 6])
    A_f, Bs_f, bs_f, _ = ml_full.from_mctdhf(C, bs)
    pops = [np.asarray(p) for p in
            ml_full.spf_populations(ml_full._ABb_to_y(A_f, Bs_f, bs_f))]

    for n in (2, 3, 4):
        ml = make_ml([n, n])
        _A, _Bs, _bs, disc = ml.from_mctdhf(C, bs)
        for s in range(ml.num_species):
            expected = float(np.sum(pops[s][n:]) / np.sum(pops[s]))
            np.testing.assert_allclose(float(np.real(disc[s])), expected,
                                       rtol=0, atol=1e-10,
                                       err_msg=f"n={n} species {s}")


# ── mixed-precision CN in real time ───────────────────────────────────────
#
# ``cn_mixed`` is the defect-form CN of ``_cn_defect_linear``: the residual in
# complex128, the BiCGSTAB solve for the increment in complex64. In real time
# the increment is O(dt) every step, so -- unlike ITE, where it vanishes at the
# fixed point -- the inner tolerance does show up in the trajectory. Measured
# on CPU (T = 0.4, dt = 0.05): max|y_mixed - y_cn| = 2.8e-8 at tol 1e-4 and
# 3.2e-10 at tol 1e-6, against a time-step error of 3.6e-5.


def _mixed_cn_state():
    std, (C, bs) = relaxed_standard_state()
    ml = make_ml([3, 3])
    A, Bs, bs_ml, _disc = ml.from_mctdhf(C, bs)
    return ml, ml._ABb_to_y(A, Bs, bs_ml)


def _propagate_cn(ml, y0, dt, T, prop, tol, lo_dtype=None, renorm=False):
    kw = {"solver_tol": tol, "max_solver_iter": 200}
    if lo_dtype is not None:
        kw["lo_dtype"] = lo_dtype
    step = get_ml_cmf_midpoint_integrator(
        ml, dt, a_propagator="diagonalize", B_propagator=prop,
        b_propagator=prop, B_kwargs=kw, b_kwargs=kw, renorm=renorm)
    t, y = 0.0, y0
    for _ in range(int(round(T / dt))):
        t, y = step(t, y)
    return y


def test_cn_mixed_matches_cn_in_real_time():
    """``cn_mixed`` on both the SPF and orbital layers tracks x64 ``cn`` to
    about the inner tolerance, and tightening the tolerance tightens it."""
    ml, y0 = _mixed_cn_state()
    dt, T = 0.05, 0.4
    y_cn = _propagate_cn(ml, y0, dt, T, "cn", 1e-12)
    errs = []
    for tol in (1e-4, 1e-6):
        y_mx = _propagate_cn(ml, y0, dt, T, "cn_mixed", tol)
        assert y_mx.dtype == jnp.complex128
        err = float(jnp.max(jnp.abs(y_mx - y_cn)))
        assert err < tol, f"tol {tol:g}: max|Δy| = {err:.3e}"
        errs.append(err)
    assert errs[1] < errs[0], errs

    # Precision is the only difference: both see the same time-step error.
    y_ref = _propagate_cn(ml, y0, dt / 4, T, "cn", 1e-12)
    e_cn = float(jnp.max(jnp.abs(y_cn - y_ref)))
    e_mx = float(jnp.max(jnp.abs(
        _propagate_cn(ml, y0, dt, T, "cn_mixed", 1e-6) - y_ref)))
    assert abs(e_mx - e_cn) < 1e-2 * e_cn, (e_cn, e_mx)


def test_cn_mixed_error_shrinks_down_the_dt_ladder():
    ml, y0 = _mixed_cn_state()
    T = 0.4
    y_ref = _propagate_cn(ml, y0, 0.0125, T, "cn", 1e-12)
    errs = [float(jnp.max(jnp.abs(
                _propagate_cn(ml, y0, dt, T, "cn_mixed", 1e-6) - y_ref)))
            for dt in (0.1, 0.05, 0.025)]
    assert errs[0] > errs[1] > errs[2], errs
    assert errs[0] > 3 * errs[2], errs


def test_cn_mixed_dtype_guard():
    """The trace-time guard: a non-lower ``lo_dtype`` and a low-precision
    RHS that promotes back to complex128 must both raise."""
    ml, y0 = _mixed_cn_state()
    for prop_B, prop_b in (("cn_mixed", "cn"), ("cn", "cn_mixed")):
        step = get_ml_cmf_midpoint_integrator(
            ml, 0.05, a_propagator="diagonalize", B_propagator=prop_B,
            b_propagator=prop_b,
            B_kwargs={"lo_dtype": jnp.complex128} if prop_B == "cn_mixed"
            else None,
            b_kwargs={"lo_dtype": jnp.complex128} if prop_b == "cn_mixed"
            else None)
        with pytest.raises(TypeError, match="not lower precision"):
            step(0.0, y0)

    from mctdhf.integrators import _cn_defect_linear
    L64 = jnp.asarray(np.diag(np.arange(1.0, 5.0)), dtype=jnp.float64)
    x0 = jnp.ones(4, dtype=jnp.complex128)

    def rhs_hi(x):
        return -1j * (L64 @ x)

    # a float64 constant left un-lowered promotes the "low" matvec
    with pytest.raises(TypeError, match="low-precision RHS returns"):
        _cn_defect_linear(rhs_hi, rhs_hi, x0, 0.1, 1e-6, 50, jnp.complex64)
    # an x64 state is required on the high-precision side
    with pytest.raises(TypeError, match="high-precision RHS returns"):
        _cn_defect_linear(lambda x: rhs_hi(x).astype(jnp.complex64),
                          rhs_hi, x0, 0.1, 1e-6, 50, jnp.complex64)
    # the lowered RHS passes and gives the x64 CN step to ~float32 accuracy
    L32 = L64.astype(jnp.float32)
    x1 = _cn_defect_linear(rhs_hi, lambda x: -1j * (L32 @ x), x0, 0.1,
                           1e-7, 50, jnp.complex64)
    lam = np.arange(1.0, 5.0)
    exact_cn = (1 - 0.05j * lam) / (1 + 0.05j * lam)
    np.testing.assert_allclose(np.asarray(x1), exact_cn, rtol=0, atol=1e-6)


def test_renorm_keeps_layers_orthonormal_without_costing_accuracy():
    """``renorm=True`` (Löwdin after every step) holds the orbitals and SPFs
    orthonormal and the norm at 1, and is no less accurate than the plain
    step against a fine reference (the defect it removes is a rescaling)."""
    ml, y0 = _mixed_cn_state()
    dt, T = 0.05, 0.4

    def ortho_err(y):
        _A, Bs, bs = ml._y_to_ABb(y)
        return max(float(jnp.max(jnp.abs(X.conj().T @ X - jnp.eye(X.shape[1]))))
                   for X in list(Bs) + list(bs))

    def normalised_overlap(y1, y2):
        S = np.diag(np.asarray(ml.calculate_overlap(y1, y2)))
        n1 = np.abs(np.diag(np.asarray(ml.calculate_overlap(y1, y1))))
        n2 = np.abs(np.diag(np.asarray(ml.calculate_overlap(y2, y2))))
        return S / np.sqrt(n1 * n2)

    y_rn = _propagate_cn(ml, y0, dt, T, "cn", 1e-12, renorm=True)
    y_pl = _propagate_cn(ml, y0, dt, T, "cn", 1e-12)
    assert ortho_err(y_rn) < 1e-12, ortho_err(y_rn)
    # the top layer is unitary: with orthonormal layers the norm is the
    # initial one (the fixture's own norm is 1 only to ~1e-7)
    n0 = np.abs(np.diag(np.asarray(ml.calculate_overlap(y0, y0))))
    norms = np.abs(np.diag(np.asarray(ml.calculate_overlap(y_rn, y_rn))))
    assert np.max(np.abs(norms - n0)) < 1e-12, (norms, n0)

    y_ref = _propagate_cn(ml, y0, dt / 8, T, "cn", 1e-12, renorm=True)
    err_rn = float(np.max(np.abs(normalised_overlap(y_ref, y_rn) - 1)))
    err_pl = float(np.max(np.abs(normalised_overlap(y_ref, y_pl) - 1)))
    assert err_rn < 1.5 * err_pl + 1e-12, (err_rn, err_pl)

    # renorm=False is the plain step (default argument, same trajectory)
    step = get_ml_cmf_midpoint_integrator(
        ml, dt, a_propagator="diagonalize", B_propagator="cn",
        b_propagator="cn", B_kwargs={"solver_tol": 1e-12,
                                     "max_solver_iter": 200},
        b_kwargs={"solver_tol": 1e-12, "max_solver_iter": 200})
    t, y = 0.0, y0
    for _ in range(int(round(T / dt))):
        t, y = step(t, y)
    assert bool(jnp.all(y == y_pl))
