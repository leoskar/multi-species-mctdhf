import jax.numpy as jnp
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import pytest
from mctdhf.MCTDHF_direct_jax import MCTDHF
from mctdhf.MCTDHF_n_species import MCTDHF_n_species
from mctdhf.integrators import (
    get_euler_solver,
    get_cmf_integrator,
)

def test_regression_n_species():
    """Compare N=2 species calculation with original alpha/beta implementation."""
    
    # 1. System Setup: Harmonic Oscillator
    num_spatial = 10
    x = jnp.linspace(-5, 5, num_spatial)
    dx = x[1] - x[0]
    
    # Kinetic energy
    t_mat = jnp.diag(jnp.full(num_spatial, 2.0)) + \
            jnp.diag(jnp.full(num_spatial-1, -1.0), k=1) + \
            jnp.diag(jnp.full(num_spatial-1, -1.0), k=-1)
    t_mat = t_mat / (2 * dx**2)
    
    # Potential energy
    v_mat = jnp.diag(0.5 * x**2)
    h = t_mat + v_mat
    
    # Interaction: delta function approx
    # g_{ijkl} = \int dx dy \phi_i(x) \phi_j(y) V(x-y) \phi_k(x) \phi_l(y)
    # Contact interaction V(x-y) = g0 * delta(x-y)
    # g_{ijkl} = g0 * \int dx \phi_i(x) \phi_j(x) \phi_k(x) \phi_l(x)
    # In DVR: g_{ijkl} = g0/dx * delta_{ik} delta_{jl} delta_{ij} ?
    # Diagonal approximation in DVR: g_{ijkl} = V_{ij} delta_{ik} delta_{jl}
    # For contact: V_{ii} = 1/dx?
    
    # Just use a simple random 4-tensor or diagonal approx to match
    # For DVR, g is diagonal g_ij = V(x_i, x_j) ?
    # Let's use a simpler random g for regression testing to avoid physics nuances
    rng = np.random.default_rng(42)
    g = jnp.array(rng.random((num_spatial,)*4) * 0.1)
    # Symmetrize g
    g = (g + g.transpose(1,0,3,2)) / 2 # particle exchange?
    # Actually, just use whatever random tensor, consistency matters.
    
    # System params
    M = 2 # MCTDHF orbitals
    N_alpha = 1
    N_beta = 1
    
    # 2. Setup Original MCTDHF
    mctdhf_old = MCTDHF(
        h=h,
        ht=lambda t: 0*h,
        g=g,
        num_mctdhf_orbitals=M,
        num_spatial_orbitals=num_spatial,
        num_alpha_electrons=N_alpha,
        num_beta_electrons=N_beta,
        num_states=1
    )
    
    # 3. Setup New MCTDHF_n_species
    h_list = [h, h]
    g_dict = {
        (0,0): g, # alpha-alpha
        (1,1): g, # beta-beta
        (0,1): g  # alpha-beta
    }
    
    mctdhf_new = MCTDHF_n_species(
        h_list=h_list,
        g_dict=g_dict,
        num_particles=[N_alpha, N_beta],
        num_mctdhf_orbitals=[M, M],
        num_spatial_orbitals=[num_spatial, num_spatial],
        num_states=1
    )
    
    # 4. Initial Condition (Same for both)
    # Old: b_alpha_init, b_beta_init, C_init
    rng = np.random.default_rng(123)
    b_init = jnp.array(rng.random((num_spatial, M)) + 0j)
    b_init, _ = jnp.linalg.qr(b_init) # Orthonormal
    
    n_dets_alpha = mctdhf_old.dCI.E_pq_alpha.shape[0]
    n_dets_beta = mctdhf_old.dCI.E_pq_beta.shape[0]
    C_init = jnp.array(rng.random((1, n_dets_alpha, n_dets_beta)) + 0j)
    C_init = C_init / jnp.linalg.norm(C_init)
    
    # 5. Integrate one step
    # Use Euler to verify derivatives match
    solver = get_euler_solver
    dt = 0.001
    
    integrator_old = solver(mctdhf_old, dt)
    integrator_new = solver(mctdhf_new, dt)
    
    y_old_packed = mctdhf_old._Cb_to_y(C_init, b_init, b_init)
    y_new_packed = mctdhf_new._Cb_to_y(C_init, [b_init, b_init])
    
    t_old, y_old_next = integrator_old(0.0, y_old_packed)
    t_new, y_new_next = integrator_new(0.0, y_new_packed)
    
    # 6. Compare Results
    C_old, bs_alpha_old, bs_beta_old = mctdhf_old._y_to_Cb(y_old_next)
    C_new, bs_new_list = mctdhf_new._y_to_Cb(y_new_next)
    bs_alpha_new, bs_beta_new = bs_new_list
    
    # Explicitly check derivatives (from one step difference)
    # C
    diff_C = jnp.linalg.norm(C_old - C_new)
    print(f"Difference in C: {diff_C}")
    
    # b_alpha
    diff_ba = jnp.linalg.norm(bs_alpha_old - bs_alpha_new)
    print(f"Difference in b_alpha: {diff_ba}")
    
    # b_beta
    diff_bb = jnp.linalg.norm(bs_beta_old - bs_beta_new)
    print(f"Difference in b_beta: {diff_bb}")
    
    assert diff_C < 1e-5
    assert diff_ba < 5e-5
    assert diff_bb < 5e-5

    # Verify new methods
    # Verify new methods
    print("Verifying calculate_energy...")
    E = mctdhf_new.calculate_energy(0.0, y_new_next)
    print(f"Energy: {E}")
    assert not jnp.isnan(E)
    
    print("Verifying calculate_particle_densities...")
    C_final, bs_final = mctdhf_new._y_to_Cb(y_new_next)
    rhos = mctdhf_new.calculate_particle_densities(C_final, bs_final)
    assert len(rhos) == 2
    assert len(rhos) == 2
    print("Densities calculated successfully.")
    
    print("Verifying calculate_overlap...")
    
    # Check b_init orthonormality
    print("Checking b_init orthonormality...")
    b0 = bs_new_list[0] # Should be normalized
    S_b0 = b0.T.conj() @ b0
    diff_eye = jnp.linalg.norm(S_b0 - jnp.eye(M))
    print(f"b_init deviation from I: {diff_eye}")
    
    # Check C_init norm
    print(f"C_init norm: {jnp.linalg.norm(C_new)}")

    # Check overlap of initial normalized state
    ov_init = mctdhf_new.calculate_overlap(y_new_packed, y_new_packed)
    print(f"Self Overlap (Init): {ov_init}")
    assert jnp.abs(ov_init[0] - 1.0) < 1e-3
    
    # Check overlap of time evolved state (Euler drift expected)
    ov_self = mctdhf_new.calculate_overlap(y_new_next, y_new_next)
    print(f"Self Overlap (Next): {ov_self}")
    # Drift 3e-4 is acceptable for Euler
    assert jnp.abs(ov_self[0] - 1.0) < 1e-3

    ov_cross = mctdhf_new.calculate_overlap(y_new_packed, y_new_next) # Init vs Next
    print(f"Cross Overlap: {ov_cross}")
    assert jnp.abs(ov_cross[0]) <= 1.001

def _make_small_mctdhf(imag_time):
    """Small 2-species MCTDHF instance for diagonalize-propagator tests."""
    num_spatial = 8
    x = jnp.linspace(-4, 4, num_spatial)
    dx = x[1] - x[0]
    t_mat = (
        jnp.diag(jnp.full(num_spatial, 2.0))
        + jnp.diag(jnp.full(num_spatial - 1, -1.0), k=1)
        + jnp.diag(jnp.full(num_spatial - 1, -1.0), k=-1)
    ) / (2 * dx**2)
    h = t_mat + jnp.diag(0.5 * x**2)

    rng = np.random.default_rng(0)
    g = jnp.array(rng.random((num_spatial,) * 4) * 0.05)
    g = (g + g.transpose(1, 0, 3, 2)) / 2

    M = 3
    mctdhf = MCTDHF_n_species(
        h_list=[h, h],
        g_dict={(0, 0): g, (1, 1): g, (0, 1): g},
        num_particles=[1, 1],
        num_mctdhf_orbitals=[M, M],
        num_spatial_orbitals=[num_spatial, num_spatial],
        num_states=2,
        imag_time=imag_time,
    )

    b_init = jnp.array(rng.random((num_spatial, M)) + 0j)
    b_init, _ = jnp.linalg.qr(b_init)
    n_dets = mctdhf.num_dets
    C_init = jnp.array(rng.random((mctdhf.num_states, n_dets[0], n_dets[1])) + 0j)
    C_init, _ = jnp.linalg.qr(C_init.reshape(mctdhf.num_states, -1).T)
    C_init = C_init.T.reshape((mctdhf.num_states, n_dets[0], n_dets[1]))

    return mctdhf, C_init, [b_init, b_init]


def test_diagonalize_matches_or_beats_sil_energy():
    """In ITE, one CMF step with c_propagator='diagonalize' must give a
    state-averaged energy ≤ what SIL produces from the same starting point.

    Diagonalize returns the exact lowest-N CI eigenvectors given the
    orbitals at t+dt/2; SIL returns an exp(-dt H)-shrunk version of the
    input C. After QR re-orthonormalisation (matching the integrate()
    normalize step) the diagonalize result is variationally optimal in CI.
    """
    mctdhf, C_init, bs_init = _make_small_mctdhf(imag_time=True)
    y_init = mctdhf._Cb_to_y(C_init, bs_init)
    dt = 0.05

    integ_sil = get_cmf_integrator(mctdhf, dt, c_propagator="sil", b_propagator="sil")
    integ_diag = get_cmf_integrator(mctdhf, dt, c_propagator="diagonalize", b_propagator="sil")

    _, y_sil = integ_sil(0.0, y_init)
    _, y_diag = integ_diag(0.0, y_init)

    # Re-orthonormalise C as the integrate() driver does after each step.
    C_sil, bs_sil = mctdhf._y_to_Cb(y_sil)
    C_sil = mctdhf.ortonorm_C(C_sil)
    C_diag, bs_diag = mctdhf._y_to_Cb(y_diag)
    C_diag = mctdhf.ortonorm_C(C_diag)

    y_sil = mctdhf._Cb_to_y(C_sil, bs_sil)
    y_diag = mctdhf._Cb_to_y(C_diag, bs_diag)

    E_sil = np.real(np.array(mctdhf.calculate_energy(0.0, y_sil)))
    E_diag = np.real(np.array(mctdhf.calculate_energy(0.0, y_diag)))

    print(f"E_sil  = {E_sil}")
    print(f"E_diag = {E_diag}")

    # Sum across states is the state-averaged objective being minimised.
    assert E_diag.sum() <= E_sil.sum() + 1e-6


def test_diagonalize_rte_matches_sil():
    """In RTE, c_propagator='diagonalize' is exact propagation. A single
    SIL step with a sufficiently large Krylov dimension must agree with
    diagonalize to high precision on a small CI dimension."""
    mctdhf, C_init, bs_init = _make_small_mctdhf(imag_time=False)
    y_init = mctdhf._Cb_to_y(C_init, bs_init)
    dt = 0.02

    # CI dim is num_dets[0] * num_dets[1] = 3*3 = 9, so Krylov dim 9 is exact.
    integ_sil = get_cmf_integrator(
        mctdhf, dt,
        c_propagator="sil", b_propagator="sil",
        c_kwargs={"krylov_dim": 9}, b_kwargs={"krylov_dim": 9},
    )
    integ_diag = get_cmf_integrator(mctdhf, dt, c_propagator="diagonalize", b_propagator="sil")

    _, y_sil = integ_sil(0.0, y_init)
    _, y_diag = integ_diag(0.0, y_init)

    C_sil, _ = mctdhf._y_to_Cb(y_sil)
    C_diag, _ = mctdhf._y_to_Cb(y_diag)

    diff = jnp.linalg.norm(C_sil - C_diag) / jnp.linalg.norm(C_sil)
    print(f"||C_sil - C_diag|| / ||C_sil|| = {diff}")
    # complex64 floor — both methods are exact in arithmetic; FP noise gives ~1e-4.
    assert diff < 1e-3


def test_diagonalize_rejects_unknown_propagator():
    mctdhf, _, _ = _make_small_mctdhf(imag_time=True)
    try:
        get_cmf_integrator(mctdhf, 0.01, c_propagator="bogus")
    except ValueError as e:
        assert "diagonalize" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown c_propagator")


if __name__ == "__main__":
    test_regression_n_species()
    test_diagonalize_matches_or_beats_sil_energy()
    test_diagonalize_rte_matches_sil()
    test_diagonalize_rejects_unknown_propagator()
