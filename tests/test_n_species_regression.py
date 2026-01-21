import jax.numpy as jnp
import numpy as np
# import pytest
from mctdhf.MCTDHF_direct_jax import MCTDHF
from mctdhf.MCTDHF_n_species import MCTDHF_n_species
from mctdhf.integrators import get_euler_solver

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

if __name__ == "__main__":
    test_regression_n_species()
