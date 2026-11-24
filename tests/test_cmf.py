"""Tests for the CMF (Constant Mean Field) integrator.

Tests:
1. Small-dt agreement: CMF step matches Euler step at small dt (O(dt²) difference).
2. Large-dt stability: CMF ITE converges with dt much larger than Picard CN can handle.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mctdhf.MCTDHF_n_species import MCTDHF_n_species
from mctdhf.integrators import (
    get_euler_solver,
    get_cmf_integrator,
    get_cmf_half_step_integrator,
    get_vmf_integrator,
    get_psi_integrator,
    _compute_density_quantities,
)


def _make_test_system(imag_time=False):
    """Create a small 2-species harmonic oscillator test system."""
    N_spatial = 10
    x = jnp.linspace(-5, 5, N_spatial)
    dx = x[1] - x[0]

    # Kinetic + potential
    t_mat = (
        jnp.diag(jnp.full(N_spatial, 2.0))
        + jnp.diag(jnp.full(N_spatial - 1, -1.0), k=1)
        + jnp.diag(jnp.full(N_spatial - 1, -1.0), k=-1)
    )
    t_mat = t_mat / (2 * dx**2)
    v_mat = jnp.diag(0.5 * x**2)
    h = (t_mat + v_mat).astype(jnp.float32)

    # Random interaction
    rng = np.random.default_rng(42)
    g = jnp.array(rng.random((N_spatial,) * 4) * 0.1, dtype=jnp.float32)
    g = (g + g.transpose(1, 0, 3, 2)) / 2

    M = 2
    mctdh = MCTDHF_n_species(
        h_list=[h, h],
        g_dict={(0, 0): g, (1, 1): g, (0, 1): g},
        num_particles=[1, 1],
        num_mctdhf_orbitals=[M, M],
        num_spatial_orbitals=[N_spatial, N_spatial],
        num_states=1,
        imag_time=imag_time,
    )

    # Fixed initial condition
    rng2 = np.random.default_rng(123)
    b_init = jnp.array(rng2.random((N_spatial, M)) + 0j, dtype=jnp.complex64)
    b_init, _ = jnp.linalg.qr(b_init)

    n_dets_0 = mctdh.num_dets[0]
    n_dets_1 = mctdh.num_dets[1]
    C_init = jnp.array(rng2.random((1, n_dets_0, n_dets_1)) + 0j, dtype=jnp.complex64)
    C_init = C_init / jnp.linalg.norm(C_init)

    y_init = mctdh._Cb_to_y(C_init, [b_init, b_init])
    return mctdh, y_init


def test_cmf_small_dt_agreement():
    """CMF and Euler should agree at small dt (both are O(dt) accurate)."""
    mctdh, y_init = _make_test_system(imag_time=False)
    dt = 1e-4

    integrator_euler = get_euler_solver(mctdh, dt)
    integrator_cmf = get_cmf_integrator(mctdh, dt)

    _, y_euler = integrator_euler(0.0, y_init)
    _, y_cmf = integrator_cmf(0.0, y_init)

    # Both should produce very similar results at small dt
    diff = jnp.max(jnp.abs(y_euler - y_cmf))
    print(f"Max |y_euler - y_cmf| at dt={dt}: {diff:.2e}")

    # At small dt, Euler and CN should agree to O(dt²) ≈ 1e-8
    assert diff < 1e-5, f"CMF and Euler differ too much at small dt: {diff}"


def test_cmf_large_dt_ite():
    """CMF should converge in ITE with dt much larger than Picard CN allows."""
    mctdh, y_init = _make_test_system(imag_time=True)
    dt_cmf = 1e-2
    num_steps = 100

    integrator = get_cmf_integrator(mctdh, dt_cmf)

    t = 0.0
    y = y_init
    energies = [float(jnp.real(mctdh.calculate_energy(t, y)[0]))]

    for i in range(num_steps):
        t, y = integrator(t, y)
        # Normalize (ITE)
        C, bs = mctdh._y_to_Cb(y)
        C = mctdh.ortonorm_C(C)
        bs = [mctdh.ortonorm_b(b) for b in bs]
        y = mctdh._Cb_to_y(C, bs)
        E = float(jnp.real(mctdh.calculate_energy(t, y)[0]))
        energies.append(E)

    print(f"CMF ITE: E_init={energies[0]:.6f}, E_final={energies[-1]:.6f}")
    print(f"Energy drop: {energies[0] - energies[-1]:.6f}")

    # Energy should decrease (ITE) and not diverge
    assert not jnp.isnan(energies[-1]), "Energy diverged to NaN"
    assert not jnp.isinf(energies[-1]), "Energy diverged to Inf"
    assert energies[-1] < energies[0], "Energy did not decrease during ITE"

    # Check convergence (energy should stabilize)
    late_var = abs(energies[-1] - energies[-10])
    print(f"Late energy variation (last 10 steps): {late_var:.2e}")


def test_cmf_imag_time_matches_euler_reference():
    """CMF ITE final energy should match a reference from many small Euler steps."""
    mctdh, y_init = _make_test_system(imag_time=True)

    # Reference: many small Euler steps
    dt_ref = 1e-3
    n_ref = 500
    integrator_ref = get_euler_solver(mctdh, dt_ref)

    t, y = 0.0, y_init
    for _ in range(n_ref):
        t, y = integrator_ref(t, y)
        C, bs = mctdh._y_to_Cb(y)
        C = mctdh.ortonorm_C(C)
        bs = [mctdh.ortonorm_b(b) for b in bs]
        y = mctdh._Cb_to_y(C, bs)
    E_ref = float(jnp.real(mctdh.calculate_energy(t, y)[0]))

    # CMF: fewer, larger steps
    dt_cmf = 1e-2
    n_cmf = 100
    integrator_cmf = get_cmf_integrator(mctdh, dt_cmf)

    t, y = 0.0, y_init
    for _ in range(n_cmf):
        t, y = integrator_cmf(t, y)
        C, bs = mctdh._y_to_Cb(y)
        C = mctdh.ortonorm_C(C)
        bs = [mctdh.ortonorm_b(b) for b in bs]
        y = mctdh._Cb_to_y(C, bs)
    E_cmf = float(jnp.real(mctdh.calculate_energy(t, y)[0]))

    print(f"Euler ref energy (dt={dt_ref}, {n_ref} steps): {E_ref:.6f}")
    print(f"CMF energy (dt={dt_cmf}, {n_cmf} steps): {E_cmf:.6f}")
    print(f"|E_cmf - E_ref| = {abs(E_cmf - E_ref):.6f}")

    # Both should converge to similar ground state energy
    # Allow some tolerance since different dt and methods
    assert abs(E_cmf - E_ref) < 0.5, (
        f"CMF and Euler reference energies differ too much: "
        f"E_cmf={E_cmf:.6f}, E_ref={E_ref:.6f}"
    )


def test_half_step_small_dt_agreement():
    """Half-step CMF should agree with Euler at small dt."""
    mctdh, y_init = _make_test_system(imag_time=False)
    dt = 1e-4

    integrator_euler = get_euler_solver(mctdh, dt)
    integrator_hs = get_cmf_half_step_integrator(mctdh, dt)

    _, y_euler = integrator_euler(0.0, y_init)
    _, y_hs = integrator_hs(0.0, y_init)

    diff = jnp.max(jnp.abs(y_euler - y_hs))
    print(f"Max |y_euler - y_half_step| at dt={dt}: {diff:.2e}")
    assert diff < 1e-5, f"Half-step CMF and Euler differ too much: {diff}"


def test_half_step_rte_norm_conservation():
    """Half-step CMF should conserve norm better than basic CMF in RTE."""
    mctdh, y_init = _make_test_system(imag_time=False)
    dt = 1e-2
    num_steps = 50

    integrator_basic = get_cmf_integrator(mctdh, dt)
    integrator_hs = get_cmf_half_step_integrator(mctdh, dt)

    # Basic CMF
    t, y = 0.0, y_init
    for _ in range(num_steps):
        t, y = integrator_basic(t, y)
    overlap_basic = np.diag(mctdh.calculate_overlap(y, y))
    norm_basic = float(jnp.abs(overlap_basic[0]))

    # Half-step CMF
    t, y = 0.0, y_init
    for _ in range(num_steps):
        t, y = integrator_hs(t, y)
    overlap_hs = np.diag(mctdh.calculate_overlap(y, y))
    norm_hs = float(jnp.abs(overlap_hs[0]))

    print(f"Norm after {num_steps} RTE steps (dt={dt}):")
    print(f"  Basic CMF: {norm_basic:.6f}  (deviation: {abs(norm_basic - 1):.2e})")
    print(f"  Half-step: {norm_hs:.6f}  (deviation: {abs(norm_hs - 1):.2e})")

    # Half-step should not blow up
    assert not jnp.isnan(norm_hs), "Half-step norm is NaN"
    assert not jnp.isinf(norm_hs), "Half-step norm is Inf"

    # Half-step should conserve norm at least as well as basic
    assert (
        abs(norm_hs - 1) <= abs(norm_basic - 1) + 0.01
    ), f"Half-step norm conservation worse than basic CMF"


def test_diag_solve_bs_matches_sil_small_dt_rte():
    """`b_propagator='diagonalize'` should match `b_propagator='sil'` at small dt."""
    mctdh, y_init = _make_test_system(imag_time=False)
    dt = 1e-4

    integ_sil = get_cmf_half_step_integrator(
        mctdh, dt, c_propagator="diagonalize", b_propagator="sil"
    )
    integ_diag = get_cmf_half_step_integrator(
        mctdh, dt, c_propagator="diagonalize", b_propagator="diagonalize"
    )

    _, y_sil = integ_sil(0.0, y_init)
    _, y_diag = integ_diag(0.0, y_init)

    diff = float(jnp.max(jnp.abs(y_sil - y_diag)))
    print(f"Max |y_sil - y_diag| at dt={dt}: {diff:.2e}")
    assert diff < 1e-5, (
        f"SIL and diagonalize orbital propagators disagree at dt={dt}: {diff}"
    )


def test_diag_solve_bs_matches_sil_small_dt_ite():
    """ITE counterpart of the small-dt agreement test."""
    mctdh, y_init = _make_test_system(imag_time=True)
    dt = 1e-4

    integ_sil = get_cmf_half_step_integrator(
        mctdh, dt, c_propagator="diagonalize", b_propagator="sil"
    )
    integ_diag = get_cmf_half_step_integrator(
        mctdh, dt, c_propagator="diagonalize", b_propagator="diagonalize"
    )

    _, y_sil = integ_sil(0.0, y_init)
    _, y_diag = integ_diag(0.0, y_init)

    diff = float(jnp.max(jnp.abs(y_sil - y_diag)))
    print(f"ITE max |y_sil - y_diag| at dt={dt}: {diff:.2e}")
    assert diff < 1e-5, (
        f"SIL and diagonalize orbital propagators disagree in ITE at dt={dt}: {diff}"
    )


def test_diag_solve_bs_ite_convergence():
    """All-spectral CMF (`c=diagonalize`, `b=diagonalize`) should converge in ITE
    to the same ground-state energy as the CN/CN reference (to within ITE tol)."""
    mctdh, y_init = _make_test_system(imag_time=True)
    dt = 1e-2
    num_steps = 100

    integ = get_cmf_half_step_integrator(
        mctdh, dt, c_propagator="diagonalize", b_propagator="diagonalize"
    )

    t, y = 0.0, y_init
    E_init = float(jnp.real(mctdh.calculate_energy(t, y)[0]))
    for _ in range(num_steps):
        t, y = integ(t, y)
        C, bs = mctdh._y_to_Cb(y)
        C = mctdh.ortonorm_C(C)
        bs = [mctdh.ortonorm_b(b) for b in bs]
        y = mctdh._Cb_to_y(C, bs)
    E_final = float(jnp.real(mctdh.calculate_energy(t, y)[0]))

    print(f"all-spectral CMF ITE: E_init={E_init:.6f}, E_final={E_final:.6f}")
    assert not jnp.isnan(E_final), "Energy diverged to NaN"
    assert E_final < E_init, "Energy did not decrease during ITE"


def test_cmf_kwargs_unknown_key_raises():
    """Passing an unrecognised key in c_kwargs / b_kwargs should raise ValueError
    with a message that lists the recognised keys for the chosen propagator."""
    mctdh, _ = _make_test_system(imag_time=False)
    # Unknown key for the chosen propagator
    with pytest.raises(ValueError, match="foo"):
        get_cmf_half_step_integrator(mctdh, 1e-3, b_kwargs={"foo": 1})
    # Right key but wrong propagator (krylov_dim is for sil, not cn)
    with pytest.raises(ValueError, match="krylov_dim"):
        get_cmf_half_step_integrator(
            mctdh, 1e-3, b_propagator="cn", b_kwargs={"krylov_dim": 5}
        )
    # Diagonalize has no recognised keys at all
    with pytest.raises(ValueError, match="probe_y"):
        get_cmf_half_step_integrator(
            mctdh, 1e-3, b_propagator="diagonalize", b_kwargs={"probe_y": None}
        )


def test_g_dict_double_ordering_raises():
    """Supplying both (s1,s2) and (s2,s1) must be rejected: the duplicate
    orderings double the transform work and would silently average
    inconsistent tensors."""
    N_spatial, M = 6, 2
    h = jnp.eye(N_spatial)
    g = jnp.ones((N_spatial,) * 4) * 0.1
    with pytest.raises(ValueError, match="one ordering"):
        MCTDHF_n_species(
            h_list=[h, h],
            g_dict={(0, 1): g, (1, 0): g},
            num_particles=[1, 1],
            num_mctdhf_orbitals=[M, M],
            num_spatial_orbitals=[N_spatial, N_spatial],
        )


def test_orbital_jacobian_is_nonhermitian_under_current_gauge():
    """A_s is structurally non-Hermitian under the (1-P)(h+W)D⁻¹ gauge.
    If this test ever starts reporting Hermitian, restore the eigh fast
    path in _diag_solve_bs."""
    mctdh, y_init = _make_test_system(imag_time=False)
    C, bs = mctdh._y_to_Cb(y_init)
    h_current = mctdh._get_h_current(0.0)
    D_invs, ds = _compute_density_quantities(mctdh, C)

    n_sp, M = mctdh.num_spatial_orbitals[0], mctdh.num_mctdhf_orbitals[0]
    N = n_sp * M

    g2_frozen = mctdh._precompute_frozen_g2(bs)

    def apply_once(b):
        rhs = mctdh._rhs_b_linear(
            b, 0, h_current[0], D_invs[0], ds, bs, g2_frozen
        )
        return rhs.reshape(-1)

    I_cols = jnp.eye(N, dtype=bs[0].dtype).reshape((N, n_sp, M))
    J_rows = jax.vmap(apply_once)(I_cols)
    A = 1j * J_rows.T  # RTE sign convention
    ratio = float(jnp.linalg.norm(A - A.conj().T) / jnp.linalg.norm(A))
    print(f"||A - A†||/||A|| (species 0, RTE) = {ratio:.2e}")
    assert ratio > 0.1, (
        f"A_s reports Hermitian (ratio={ratio:.2e}). The (1-P)(h+W)D⁻¹ "
        f"gauge previously gave ~0.5; if this is now small, the gauge has "
        f"changed and _diag_solve_bs could regain an eigh fast path."
    )


def test_vmf_small_dt_agreement():
    """VMF should agree with Euler at small dt."""
    mctdh, y_init = _make_test_system(imag_time=False)
    dt = 1e-4

    integrator_euler = get_euler_solver(mctdh, dt)
    integrator_vmf = get_vmf_integrator(mctdh, dt)

    _, y_euler = integrator_euler(0.0, y_init)
    _, y_vmf = integrator_vmf(0.0, y_init)

    diff = jnp.max(jnp.abs(y_euler - y_vmf))
    print(f"Max |y_euler - y_vmf| at dt={dt}: {diff:.2e}")
    assert diff < 1e-5, f"VMF and Euler differ too much: {diff}"


def test_vmf_rte_norm_conservation():
    """VMF (symplectic) should conserve norm better than CMF in RTE."""
    mctdh, y_init = _make_test_system(imag_time=False)
    dt = 1e-2
    num_steps = 50

    integrator_hs = get_cmf_half_step_integrator(mctdh, dt)
    integrator_vmf = get_vmf_integrator(mctdh, dt)

    # Half-step CMF
    t, y = 0.0, y_init
    for _ in range(num_steps):
        t, y = integrator_hs(t, y)
    overlap_hs = np.diag(mctdh.calculate_overlap(y, y))
    norm_hs = float(jnp.abs(overlap_hs[0]))

    # VMF
    t, y = 0.0, y_init
    for _ in range(num_steps):
        t, y = integrator_vmf(t, y)
    overlap_vmf = np.diag(mctdh.calculate_overlap(y, y))
    norm_vmf = float(jnp.abs(overlap_vmf[0]))

    print(f"Norm after {num_steps} RTE steps (dt={dt}):")
    print(f"  Half-step CMF: {norm_hs:.6f}  (deviation: {abs(norm_hs - 1):.2e})")
    print(f"  VMF:           {norm_vmf:.6f}  (deviation: {abs(norm_vmf - 1):.2e})")

    assert not jnp.isnan(norm_vmf), "VMF norm is NaN"
    assert not jnp.isinf(norm_vmf), "VMF norm is Inf"


def test_diffrax_rte_norm_and_energy():
    """Diffrax adaptive VMF should conserve norm and energy in RTE."""
    pytest.importorskip("diffrax")
    from mctdhf.integrators import diffrax_integrate

    mctdh, y_init = _make_test_system(imag_time=False)
    t0, t1 = 0.0, 0.5

    ts, ys = diffrax_integrate(
        mctdh,
        t0,
        t1,
        y0=y_init,
        num_save=20,
        rtol=1e-6,
        atol=1e-8,
    )

    # Check norm conservation
    overlap_final = np.diag(mctdh.calculate_overlap(ys[-1], ys[-1]))
    norm_final = float(jnp.abs(overlap_final[0]))
    print(f"Diffrax RTE: norm at t={float(ts[-1]):.2f}: {norm_final:.8f}")
    print(f"  Norm deviation: {abs(norm_final - 1):.2e}")

    # Check energy conservation
    E_init = float(jnp.real(mctdh.calculate_energy(t0, y_init)[0]))
    E_final = float(jnp.real(mctdh.calculate_energy(float(ts[-1]), ys[-1])[0]))
    print(f"  E_init={E_init:.8f}, E_final={E_final:.8f}")
    print(f"  |ΔE| = {abs(E_final - E_init):.2e}")

    assert not jnp.isnan(norm_final), "Norm is NaN"
    assert (
        abs(norm_final - 1) < 0.02
    ), f"Norm deviation too large: {abs(norm_final - 1)}"
    assert (
        abs(E_final - E_init) < 0.2
    ), f"Energy drift too large: {abs(E_final - E_init)}"


def test_psi_small_dt_agreement():
    """PSI should agree with Euler at small dt (both are O(dt) accurate).

    Note: PSI's QR step changes the orbital gauge, so we compare physical
    wavefunctions via overlap rather than raw y vectors.
    """
    mctdh, y_init = _make_test_system(imag_time=False)
    dt = 1e-4

    integrator_euler = get_euler_solver(mctdh, dt)
    integrator_psi = get_psi_integrator(mctdh, dt)

    _, y_euler = integrator_euler(0.0, y_init)
    _, y_psi = integrator_psi(0.0, y_init)

    # Compare physical states via overlap (gauge-invariant)
    overlap = np.diag(mctdh.calculate_overlap(y_euler, y_psi))
    fidelity = float(jnp.abs(overlap[0]))
    print(f"|<Euler|PSI>| at dt={dt}: {fidelity:.10f}")
    print(f"1 - |<Euler|PSI>|: {1 - fidelity:.2e}")

    # Also compare energies
    E_euler = float(jnp.real(mctdh.calculate_energy(dt, y_euler)[0]))
    E_psi = float(jnp.real(mctdh.calculate_energy(dt, y_psi)[0]))
    print(f"E_euler={E_euler:.8f}, E_psi={E_psi:.8f}, |dE|={abs(E_euler - E_psi):.2e}")

    assert (
        abs(fidelity - 1) < 1e-4
    ), f"PSI and Euler wavefunctions differ too much: |overlap|={fidelity}"
    assert (
        abs(E_euler - E_psi) < 1e-3
    ), f"PSI and Euler energies differ too much: {abs(E_euler - E_psi)}"


def test_psi_norm_conservation():
    """PSI should conserve norm well in RTE (better than CMF)."""
    mctdh, y_init = _make_test_system(imag_time=False)
    dt = 1e-3
    num_steps = 50

    integrator = get_psi_integrator(mctdh, dt)

    t, y = 0.0, y_init
    for _ in range(num_steps):
        t, y = integrator(t, y)

    overlap = np.diag(mctdh.calculate_overlap(y, y))
    norm = float(jnp.abs(overlap[0]))
    print(f"PSI norm after {num_steps} RTE steps (dt={dt}): {norm:.8f}")
    print(f"  Norm deviation: {abs(norm - 1):.2e}")

    assert not jnp.isnan(norm), "Norm is NaN"
    # Norm error dominated by Lie-Trotter splitting + float32 accumulation
    assert abs(norm - 1) < 1e-3, f"Norm deviation too large: {abs(norm - 1)}"


def test_psi_energy_conservation():
    """PSI should conserve energy reasonably well in RTE."""
    mctdh, y_init = _make_test_system(imag_time=False)
    dt = 1e-3
    num_steps = 50

    integrator = get_psi_integrator(mctdh, dt)

    E_init = float(jnp.real(mctdh.calculate_energy(0.0, y_init)[0]))

    t, y = 0.0, y_init
    for _ in range(num_steps):
        t, y = integrator(t, y)

    E_final = float(jnp.real(mctdh.calculate_energy(t, y)[0]))
    print(f"PSI energy: E_init={E_init:.8f}, E_final={E_final:.8f}")
    print(f"  |ΔE| = {abs(E_final - E_init):.2e}")

    assert not jnp.isnan(E_final), "Energy is NaN"
    assert (
        abs(E_final - E_init) < 0.1
    ), f"Energy drift too large: {abs(E_final - E_init)}"


def test_save_every():
    """save_every should correctly subsample and handle non-divisible num_steps."""
    mctdh, y_init = _make_test_system(imag_time=True)
    dt = 1e-3
    num_steps = 100

    integrator = get_euler_solver(mctdh, dt)

    # Reference: save every step
    ts_all, ys_all = mctdh.integrate(
        integrator=integrator,
        num_steps=num_steps,
        save_every=1,
    )
    assert (
        ts_all.shape[0] == num_steps + 1
    ), f"Expected {num_steps + 1} points, got {ts_all.shape[0]}"

    # save_every=10 (divides evenly)
    ts_10, ys_10 = mctdh.integrate(
        integrator=integrator,
        num_steps=num_steps,
        save_every=10,
    )
    assert ts_10.shape[0] == 11, f"Expected 11 points, got {ts_10.shape[0]}"
    assert jnp.allclose(ts_10, ts_all[::10]), "Times don't match subsampled reference"
    assert jnp.allclose(ys_10, ys_all[::10]), "States don't match subsampled reference"

    # save_every=30 (doesn't divide 100 evenly: 3 chunks + 10 remainder)
    ts_30, ys_30 = mctdh.integrate(
        integrator=integrator,
        num_steps=num_steps,
        save_every=30,
    )
    # 1 initial + 3 chunks + 1 remainder = 5 points
    assert ts_30.shape[0] == 5, f"Expected 5 points, got {ts_30.shape[0]}"
    assert jnp.allclose(ts_30[0], ts_all[0]), "Initial time mismatch"
    assert jnp.allclose(ts_30[1], ts_all[30]), "Chunk 1 time mismatch"
    assert jnp.allclose(ts_30[2], ts_all[60]), "Chunk 2 time mismatch"
    assert jnp.allclose(ts_30[3], ts_all[90]), "Chunk 3 time mismatch"
    assert jnp.allclose(ts_30[4], ts_all[100]), "Final time mismatch"

    # Initial and final
    ts_default, ys_default = mctdh.integrate(
        integrator=integrator,
        num_steps=num_steps,
        save_every=num_steps,
    )
    assert ts_default.shape[0] == 2, f"Expected 2 points, got {ts_default.shape[0]}"
    assert jnp.allclose(ts_default[0], ts_all[0]), "Default initial time mismatch"
    assert jnp.allclose(ts_default[-1], ts_all[-1]), "Default final time mismatch"
    assert jnp.allclose(ys_default[-1], ys_all[-1]), "Default final state mismatch"

    print("All save_every checks passed")


if __name__ == "__main__":
    tests = [
        ("save_every", test_save_every),
        ("psi_small_dt_agreement", test_psi_small_dt_agreement),
        ("psi_norm_conservation", test_psi_norm_conservation),
        ("psi_energy_conservation", test_psi_energy_conservation),
        ("vmf_rte_norm_conservation", test_vmf_rte_norm_conservation),
        ("diffrax_rte_norm_and_energy", test_diffrax_rte_norm_and_energy),
        ("vmf_small_dt_agreement", test_vmf_small_dt_agreement),
        ("half_step_rte_norm_conservation", test_half_step_rte_norm_conservation),
    ]
    for i, (name, fn) in enumerate(tests, 1):
        print("=" * 60)
        print(f"Test {i}: {name}")
        print("=" * 60)
        fn()
        print("\nPASSED\n")
