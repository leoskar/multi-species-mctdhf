import jax
import jax.numpy as jnp
import numpy as np
from mctdhf.MCTDHF_direct_jax import MCTDHF
from mctdhf.MCTDHF_n_species import MCTDHF_n_species
from mctdhf.integrators import get_euler_solver

def test_m1_ite():
    # System: 1D Harmonic Oscillator, 2 species (alpha, beta), M=1 orbital each.
    # N=1 particle per species.
    # Result should be Ground State of 2-particle HO with interaction.
    
    N_spatial = 20
    x = jnp.linspace(-5, 5, N_spatial)
    dx = x[1] - x[0]
    
    # Kinetic
    t_mat = jnp.diag(jnp.full(N_spatial, 2.0)) + \
            jnp.diag(jnp.full(N_spatial-1, -1.0), k=1) + \
            jnp.diag(jnp.full(N_spatial-1, -1.0), k=-1)
    t_mat = t_mat / (2 * dx**2)
    
    # Potential
    v_mat = jnp.diag(0.5 * x**2)
    h = t_mat + v_mat
    
    # Interaction: Contact g * delta(x-y) -> diagonal g_ij = V_ii delta_ij
    # Let's use strong interaction to highlight differences
    g0 = 5.0
    # Diagonal tensor approximation for contact
    g_diag = jnp.zeros((N_spatial, N_spatial, N_spatial, N_spatial))
    # g_ijkl = V_ij delta_ik delta_jl
    # For contact x=y: V_ii implies i=j.
    # So g_iiii = V_ii.
    indices = jnp.arange(N_spatial)
    g_diag = g_diag.at[indices, indices, indices, indices].set(g0)
    
    # Or just use a separable Gaussian interaction to avoid delta issues
    # V(x,y) = g0 * exp(-(x-y)^2)
    # g_ijkl = int dx dy phi_i(x) phi_j(y) V ... phi_k(x) phi_l(y)
    # DVR: phi_i(x) ~ delta(x-xi).
    # g_ijkl = V(xi, xj) delta_ik delta_jl
    X, Y = jnp.meshgrid(x, x, indexing='ij')
    V_int = g0 * jnp.exp(-(X-Y)**2)
    g = jnp.zeros((N_spatial, N_spatial, N_spatial, N_spatial))
    # g[i, j, k, l] -> i,j bra. k,l ket.
    # g[i, j, i, j] = V_int[i, j]
    for i in range(N_spatial):
        for j in range(N_spatial):
            g = g.at[i, j, i, j].set(V_int[i, j])
            
    # Symmetrize g? V(x,y) symmetric. V_ij = V_ji.
    # g_ji ji = V_ji = V_ij = g_ij ij.
    
    # Setup Old
    mctdhf_old = MCTDHF(
        h=h, ht=lambda t: 0*h, g=g,
        num_mctdhf_orbitals=1,
        num_spatial_orbitals=N_spatial,
        num_alpha_electrons=1,
        num_beta_electrons=1,
        num_states=1,
        imag_time=True
    )
    
    # Setup New
    g_dict = {
        (0,0): g, (1,1): g, (0,1): g
        # Two species (alpha, beta) with one electron each: only the
        # alpha-beta interaction acts.
    }
    # The two-spin code applies its single g to the alpha-beta pair.
    
    mctdhf_new = MCTDHF_n_species(
        h_list=[h, h],
        g_dict={(0, 1): g},  # the alpha-beta interaction, as in the two-spin code
        num_particles=[1, 1],
        num_mctdhf_orbitals=[1, 1],
        num_spatial_orbitals=[N_spatial, N_spatial],
        num_states=1,
        imag_time=True
    )
    
    # Initial Condition - Complex to test transform_g conjugation
    rng = np.random.default_rng(42)
    b_init = jnp.array(rng.random((N_spatial, 1)) + 1j * rng.random((N_spatial, 1)))
    b_init = b_init / jnp.linalg.norm(b_init)
    
    C_init = jnp.array([[[1.0+0j]]]) # Single det, Coeff 1.
    
    # Pack
    y_old = mctdhf_old._Cb_to_y(C_init, b_init, b_init)
    y_new = mctdhf_new._Cb_to_y(C_init, [b_init, b_init])
    
    # Integrate
    solver = get_euler_solver
    dt = 0.01
    steps = 100
    
    integrator_old = solver(mctdhf_old, dt)
    integrator_new = solver(mctdhf_new, dt)
    
    y_next_old = y_old
    y_next_new = y_new
    
    print("Starting Integration...")
    for i in range(steps):
        t = i*dt
        _, y_next_old = integrator_old(t, y_next_old)
        _, y_next_new = integrator_new(t, y_next_new)
        
        # Re-orthonormalise the orbitals after each Euler step (ITE).
        
        # Normalize Old
        C_o, ba_o, bb_o = mctdhf_old._y_to_Cb(y_next_old)
        ba_o, _ = jnp.linalg.qr(ba_o); bb_o, _ = jnp.linalg.qr(bb_o)
        y_next_old = mctdhf_old._Cb_to_y(C_o, ba_o, bb_o)
        
        # Normalize New
        C_n, bs_n = mctdhf_new._y_to_Cb(y_next_new)
        bs_new_norm = []
        for b in bs_n:
            u, _, vh = jnp.linalg.svd(b, full_matrices=False)
            bs_new_norm.append(u @ vh)
        y_next_new = mctdhf_new._Cb_to_y(C_n, bs_new_norm)
        
    print("Integration Done.")
    
    # Compare B densities
    C_o, ba_o, bb_o = mctdhf_old._y_to_Cb(y_next_old)
    C_n, bs_n = mctdhf_new._y_to_Cb(y_next_new)
    
    rho_o = jnp.abs(ba_o)**2
    rho_n = jnp.abs(bs_n[0])**2
    
    diff = jnp.linalg.norm(rho_o.flatten() - rho_n.flatten())
    print(f"Density Difference: {diff}")
    
    print(f"Old b[0]: {ba_o[0,0]}")
    print(f"New b[0]: {bs_n[0][0,0]}")

if __name__ == "__main__":
    test_m1_ite()
