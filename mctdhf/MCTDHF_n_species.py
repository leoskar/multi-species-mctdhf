import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from opt_einsum import contract
from functools import partial
import numpy as np

from .SemiDirectCI_n_species import SemiDirectCI_n_species
from .integrators import get_crank_nicolson_fixed_iter

class MCTDHF_n_species:
    """Class for running MCTDHF calculations with N distinguishable species.
    
    Attributes:
        h_list: List of 1-body integrals (arrays or callables)
        g_dict: Dict of 2-body integrals {(s1,s2): array}
        num_species: Number of species
    """

    def __init__(
        self,
        h_list,         # List of N arrays/callables
        g_dict,         # Dict of {(s1,s2): array}
        num_particles,  # List of N ints
        num_mctdhf_orbitals, # List of N ints
        num_spatial_orbitals, # List of N ints
        num_states=1,
        imag_time=False,
    ):
        self.num_species = len(num_particles)
        self.num_particles = num_particles
        self.num_mctdhf_orbitals = num_mctdhf_orbitals
        self.num_spatial_orbitals = num_spatial_orbitals
        
        # Handle h inputs (callable or array)
        self.h_is_callable = [callable(h) for h in h_list]
        self.h_list_init = h_list
        
        # Handle g inputs
        self.g_dict = g_dict
        
        # Initialize CI solver
        self.dCI = SemiDirectCI_n_species(num_mctdhf_orbitals, num_particles)
        self.num_dets = self.dCI.num_dets
        self.num_states = num_states
        self.imag_time = imag_time

    def _Cb_to_y(self, C, bs_list):
        """Pack C and list of b matrices into flat vector y."""
        C_flat = C.reshape(-1)
        bs_flat = [b.reshape(-1) for b in bs_list]
        return jnp.concatenate([C_flat] + bs_flat)

    def _y_to_Cb(self, y):
        """Unpack y into C and list of b matrices."""
        # Unpack C
        c_size = np.prod(self.num_dets) * self.num_states
        C_flat = y[:c_size]
        C_shape = (self.num_states,) + tuple(self.num_dets)
        C = C_flat.reshape(C_shape)
        
        # Unpack bs
        start = c_size
        bs_list = []
        for s in range(self.num_species):
            b_size = self.num_spatial_orbitals[s] * self.num_mctdhf_orbitals[s]
            b_flat = y[start : start + b_size]
            b_shape = (self.num_spatial_orbitals[s], self.num_mctdhf_orbitals[s])
            bs_list.append(b_flat.reshape(b_shape))
            start += b_size
            
        return C, bs_list

    @partial(jit, static_argnums=0)
    def transform_h(self, h, b):
        """Transform the single particle electron integrals to the MCTDHF basis.
        
        Returns:
            h_1: <i|h|m> (spatial, mo)
            h_2: <n|h|m> (mo, mo)
            h_3: <i|P h P|m> (spatial, mo) -> Projected h
        """
        h_1 = contract("jm, ij -> im", b, h, backend="jax")
        h_2 = contract("in, im -> nm", b.conj(), h_1, backend="jax")
        h_3 = contract("in, nm -> im", b, h_2, backend="jax")
        return h_1, h_2, h_3

    @partial(jit, static_argnums=0)
    def transform_g(self, g, b_sigma, b_tau):
        """Transform two-particle integrals: <sigma tau | V | sigma tau>.
        
        Args:
            g: Spatial integrals g_{ijkl} = <ij|kl>.
            b_sigma: Orbitals for particle 1 (indices i, k)
            b_tau: Orbitals for particle 2 (indices j, l)
            
        Returns:
            g_3: Mean-field tensor for sigma (one spatial index, three MO indices)
            g_4: Fully transformed MO integrals
            g_5: Back-transformed mean-field tensor (projected W)
        """
        # Transform j (tau) and l (tau): g_{ijkl} -> g_{iqks}
        g_2 = contract("jq, ls, ijkl -> iqks", b_tau.conj(), b_tau, g, backend="jax")
        
        # Transform k (sigma): g_{iqks} -> g_{iqrs}
        # This is the W tensor (Mean field) for sigma
        g_3 = contract("kr, iqks -> iqrs", b_sigma, g_2, backend="jax")
        
        # Transform i (sigma): g_{iqrs} -> g_{pqrs}
        g_4 = contract("ip, iqrs -> pqrs", b_sigma.conj(), g_3, backend="jax")
        
        # Back-transform i (sigma): g_{pqrs} -> g_{iqrs} (Expected value of P W)
        g_5 = contract("ip, pqrs -> iqrs", b_sigma, g_4, backend="jax")
        
        return g_3, g_4, g_5

    def _transpose_g(self, g):
        """Transpose g integrals for indirect interactions.
        
        For 4D integrals: (1, 0, 3, 2) swaps particle 1 <-> particle 2.
        Override in DVR subclass for 2D case.
        """
        return jnp.transpose(g, (1, 0, 3, 2))

    @partial(jit, static_argnums=0)
    def calculate_energy(self, t, y):
        """Calculate the total energy of the system at time t."""
        C, bs = self._y_to_Cb(y)
        
        # 1. Evaluate integrals
        h_current = []
        for s in range(self.num_species):
            if self.h_is_callable[s]:
                h_current.append(self.h_list_init[s](t))
            else:
                h_current.append(self.h_list_init[s])
                
        # 2. Transform integrals
        h_MO_list = []
        g_MO_dict = {}
        
        for s in range(self.num_species):
            _, h_2, _ = self.transform_h(h_current[s], bs[s])
            h_MO_list.append(h_2)
            
        for s1 in range(self.num_species):
            for s2 in range(self.num_species):
                if (s1, s2) in self.g_dict:
                    _, g_4, _ = self.transform_g(
                        self.g_dict[(s1, s2)], 
                        bs[s1], bs[s2]
                    )
                    g_MO_dict[(s1, s2)] = g_4
                elif (s2, s1) in self.g_dict:
                    # Indirect interaction (s2 -> s1)
                    g_spatial_T = self._transpose_g(self.g_dict[(s2, s1)])
                    _, g_4, _ = self.transform_g(
                        g_spatial_T, 
                        bs[s1], bs[s2]
                    )
                    g_MO_dict[(s1, s2)] = g_4
                    
        # 3. Calculate Energy
        return self.dCI.calculate_energy(C, h_MO_list, g_MO_dict)

    @partial(jit, static_argnums=0)
    def calculate_particle_densities(self, C, bs_list, spf_list=None):
        """Calculate particle densities for all species.
        
        Args:
            C: CI coefficients
            bs_list: List of orbital coefficients (MOs)
            spf_list: Optional list of grid representations (Single Particle Functions).
                      If provided, returns density on grid. Else, returns density matrix in spatial basis.
                      
        Returns:
            List of densities [rho_0, rho_1, ...]
        """
        Ds = self.dCI.get_1p_RDMs(C)
        densities = []
        
        for s in range(self.num_species):
            D_s = Ds[s] # (batch, mo, mo) - Preserve batch dimension
            b_s = bs_list[s]
            
            if spf_list is not None and len(spf_list) > s:
                spf = spf_list[s]
                rho = contract("ix,in,snm,jm,jx->sx", spf, b_s.conj(), D_s, b_s, spf, backend="jax")
                densities.append(rho)
            else:
                # Return density matrix in spatial orbital basis?
                # rho_{ij} = sum_{mn} D_{mn} b_{im}^* b_{jn}
                # rho_spatial = b @ D @ b^dag
                # rho_{Bij} = sum_{mn} D_{Bnm} b_{im}^* b_{jn}
                # b_s.conj(): (i, n) -> in (using n for row index of D)
                # D_s: (s, n, m) -> snm
                # b_s: (j, m) -> jm (using m for col index of D)
                rho_spatial = contract("in, snm, jm -> sij", b_s.conj(), D_s, b_s, backend="jax")
                densities.append(rho_spatial)
                
        return densities

    @partial(jit, static_argnums=0)
    def calculate_overlap(self, y1, y2):
        """Calculate the overlap <Psi1|Psi2> between two states."""
        C1, bs1 = self._y_to_Cb(y1)
        C2, bs2 = self._y_to_Cb(y2)
        
        # Build S matrices for each species: S_matrix[s]_{IJ} = <D_I|D_J>
        S_matrices = []
        for s in range(self.num_species):
             b1 = bs1[s]
             b2 = bs2[s]
             # I is boolean mask of occupied orbitals (n_dets, n_mo)
             I = jnp.array(self.dCI.I[s]) 
             n_dets = self.num_dets[s]
             n_elec = self.num_particles[s]
             
             # Overlap between MOs: O_ij = <phi1_i | phi2_j>
             O = contract("ki, kj -> ij", b1.conj(), b2, backend="jax")
             
             # Construct determinant overlap matrix elements
             # We want to extract the submatrix of O corresponding to occupied orbitals for each (I, J) pair
             # Mask: M_{abij} = I_{ai} * I_{bj} (1 if i in a AND j in b)
             Mask = contract("ai, bj -> abij", I, I, backend="jax")
             
             # Values: V_{abij} = I_{ai} * I_{bj} * O_{ij}
             Val = contract("ai, bj, ij -> abij", I, I, O, backend="jax")
             
             # Flatten and extract using mask
             # We assume strictly n_elec orbitals per determinant
             expected_size = n_dets * n_dets * n_elec * n_elec
             indices = jnp.nonzero(Mask.reshape(-1), size=expected_size)
             
             elems = Val.reshape(-1)[indices]
             
             # Reshape to (n_dets, n_dets, n_elec, n_elec)
             mat = elems.reshape(n_dets, n_dets, n_elec, n_elec)
             
             # Compute determinant of the overlap submatrix
             S_det = jnp.linalg.det(mat)
             S_matrices.append(S_det)
             
        # Contract CI coefficients with determinant overlaps
        # C1: (states, a, b, c...)
        # C2: (states, x, y, z...)
        # S0: (a, x)
        # S1: (b, y)
        
        # Dynamic einsum string generation
        c_indices = [chr(97 + i) for i in range(self.num_species)] # a, b, c...
        c2_indices = [chr(122 - i) for i in range(self.num_species)][::-1] # z, y, x... (reverse to match order?)
        # Let's use separate chars clearly.
        # C1 indices: a, b, c
        # C2 indices: x, y, z
        
        c1_chars = []
        c2_chars = []
        s_contractions = []
        
        start_char = 97 # 'a'
        for s in range(self.num_species):
            c1_chars.append(chr(start_char + s))
            c2_chars.append(chr(start_char + s + self.num_species))
            
            # S matrix connects s-th index of C1 and C2
            s_contractions.append(f"{c1_chars[-1]}{c2_chars[-1]}")
            
        c1_str = "".join(c1_chars)
        c2_str = "".join(c2_chars)
        s_str = ",".join(s_contractions)
        
        # s index for states (batch) is first
        # C1: s...
        # C2: s...
        # Output: s (vector of overlaps per state)
        
        einsum_str = f"s{c1_str},s{c2_str},{s_str}->s"
        
        return contract(einsum_str, C1.conj(), C2, *S_matrices, backend="jax")

    @partial(jit, static_argnums=0)
    def __call__(self, t, y):
        """Right hand side of the equation system (time derivative of coefficients)."""
        C, bs = self._y_to_Cb(y)

        # 1. Evaluate time-dependent integrals
        h_current = []
        for s in range(self.num_species):
            if self.h_is_callable[s]:
                h_current.append(self.h_list_init[s](t))
            else:
                h_current.append(self.h_list_init[s])
                
        # 2. Transform integrals and Accumulate Mean Fields
        h_MO_list = []      # List of (n_mo, n_mo)
        g_MO_dict = {}      # Dict {(s1,s2): (n_mo...)}
        
        h_list_transformed = [] # List of (h_1, h_2, h_3) tuples
        
        g_mix_tensors = {}      # Dict {(s_target, s_source): (g_3, g_5)}
        
        # Transform h
        for s in range(self.num_species):
            h_1, h_2, h_3 = self.transform_h(h_current[s], bs[s])
            h_list_transformed.append((h_1, h_2, h_3))
            h_MO_list.append(h_2)
            
        # Loop over all pairs (s1, s2) for 2-body interactions
        for s1 in range(self.num_species):
            for s2 in range(self.num_species):
                if (s1, s2) in self.g_dict:
                    # Calculate interaction <s1 s2 | V | s1 s2>
                    # g_3 is mean-field acting on s1 (due to s2 density)
                    g_3, g_4, g_5 = self.transform_g(
                        self.g_dict[(s1, s2)], 
                        bs[s1], bs[s2]
                    )
                    
                    g_mix_tensors[(s1, s2)] = (g_3, g_5)
                    g_MO_dict[(s1, s2)] = g_4
                elif (s2, s1) in self.g_dict:
                    # Indirect interaction (s2 -> s1) using transposed spatial integrals
                    # to obtain mean-field on s1 correctly.
                    g_spatial_T = self._transpose_g(self.g_dict[(s2, s1)])
                    
                    g_3, g_4, g_5 = self.transform_g(
                        g_spatial_T, 
                        bs[s1], bs[s2]
                    )
                    
                    g_mix_tensors[(s1, s2)] = (g_3, g_5)
                    g_MO_dict[(s1, s2)] = g_4
        
        # 3. Calculate Sigma and Energy (for C EOM)
        sigma = self.dCI.get_sigma(C, h_MO_list, g_MO_dict)
        
        # 4. Calculate RDMs (for b EOM)
        Ds = self.dCI.get_1p_RDMs(C) # List of (batch, mo, mo)
        ds = self.dCI.get_2p_RDMs(C) # Dict {(s1,s2): (batch, p,q,r,s)}
        
        # Sum over batch dimension (state average for mean field)
        Ds = [jnp.sum(D, axis=0) for D in Ds]
        ds = {k: jnp.sum(v, axis=0) for k, v in ds.items()}
        
        # 5. Compute orbital time derivatives
        b_dots = []
        for s in range(self.num_species):
            # Use pinv for stability with potentially singular density matrices
            D_inv = jnp.linalg.pinv(Ds[s], rcond=1e-6)
            
            # Retrieve h components
            h_1, _, h_3 = h_list_transformed[s]
            
            # Start with h contribution: (1-P) h b = h_1 - h_3
            # We assume S=I, so Sh_1 = h_1
            term_h = h_1 - h_3
            
            # Mean field contribution
            term_g = jnp.zeros_like(term_h)
            
            for k in range(self.num_species):
                if (s, k) in g_mix_tensors:
                    g_3, g_5 = g_mix_tensors[(s, k)]
                    d_sk = ds[(s, k)]
                    
                    # Calculate (1-P) W = g_3 - g_5
                    diff_g = g_3 - g_5
                    
                    # Contract in one step: D_inv @ d @ (g3 - g5)
                    # D_inv: [i, n] -> np (using p for summation)
                    # d_sk: [n, q, r, s] -> pqrs (using p for D_inv summation)
                    # diff_g: [i, q, r, s] -> iqrs
                    # Result: [i, n] -> in (orbital derivative)
                    # "np, pqrs, iqrs -> in"
                    term_g += contract("np, pqrs, iqrs -> in", D_inv, d_sk, diff_g, backend="jax")
            
            b_dot_s = term_h + term_g
            
            if self.imag_time:
                b_dots.append(-b_dot_s)
            else:
                b_dots.append(-1j * b_dot_s)
 
        # 6. Coefficient time derivative
        if self.imag_time:
            C_dot = -sigma
        else:
            C_dot = -1j * sigma
            
        return self._Cb_to_y(C_dot, b_dots)

    @partial(jit, static_argnums=0)
    def ortonorm_b(self, b):
        """Gram-Schmidt orthonormalization of b."""
        # Assuming S=I for now as typical in this code
        for i in range(b.shape[1]):
            orto_adjustment = 0
            for j in range(0, i):
                # <b_i|b_j> projection
                overlap = contract("i,i", b[:, i].conj(), b[:, j], backend="jax")
                orto_adjustment += b[:, j] * overlap

            b = b.at[:, i].add(-orto_adjustment)
            norm = jnp.sqrt(contract("i,i", b[:, i].conj(), b[:, i], backend="jax"))
            b = b.at[:, i].divide(norm)
        return b

    @partial(jit, static_argnums=0)
    def ortonorm_C(self, C):
        """Gram-Schmidt orthonormalization of C."""
        # Reshape to (states, dets) for GS across states
        # C shape: (states, dets_alpha, dets_beta...)
        original_shape = C.shape
        states = C.shape[0]
        C_flat = C.reshape(states, -1)
        
        for i in range(states):
            orto_adjustment = 0
            for j in range(0, i):
                 overlap = contract("i,i", C_flat[i].conj(), C_flat[j], backend="jax")
                 orto_adjustment += overlap * C_flat[j]

            C_flat = C_flat.at[i].add(-orto_adjustment)
            norm = jnp.sqrt(contract("i,i", C_flat[i].conj(), C_flat[i], backend="jax"))
            C_flat = C_flat.at[i].divide(norm)
            
        return C_flat.reshape(original_shape)

    def compute_subsystem_character(self, C):
        """Compute subsystem excitation character for each state.
        
        Returns a matrix of shape (num_states, num_species) where each entry
        indicates the "average excitation level" of that subsystem for that state.
        For each state s and species k, this is computed as the weighted average
        of the orbital index: sum_n n * D[s,n,n] where D is the 1-particle RDM.
        
        Higher values indicate higher excitation levels in that subsystem.
        """
        Ds = self.dCI.get_1p_RDMs(C)  # List of (num_states, n_orb, n_orb)
        
        num_states = C.shape[0]
        character = jnp.zeros((num_states, self.num_species))
        
        for s in range(self.num_species):
            D_s = Ds[s]  # (num_states, n_orb, n_orb)
            n_orb = D_s.shape[-1]
            # Compute average orbital index (excitation level)
            # orbital_weight[n] = n (0 = ground, 1 = first excited, etc.)
            orbital_weights = jnp.arange(n_orb, dtype=jnp.float32)
            # D[state, n, n] is the occupation of orbital n in that state
            for state_idx in range(num_states):
                diag = jnp.diag(D_s[state_idx])  # Orbital occupations
                # Weighted average of orbital index
                avg_orbital = jnp.sum(orbital_weights * jnp.abs(diag)) / jnp.sum(jnp.abs(diag))
                character = character.at[state_idx, s].set(avg_orbital)
        
        return character

    def localize_states(self, y, energy_tol=1e-6, max_sweeps=20, tol=1e-8, verbose=False):
        """Localize degenerate states to Hartree products (post-processing).
        
        States are ordered by energy. Within degenerate subspaces (energies 
        within energy_tol), states are rotated to maximize subsystem purities,
        producing localized product-like states instead of entangled superpositions.
        
        Uses Jacobi rotations with analytically computed optimal angles based
        on subsystem purity Tr(ρ_s²) as the localization metric.
        
        Parameters
        ----------
        y : array
            The packed state vector from ITE.
        energy_tol : float
            Energy tolerance for grouping degenerate states.
        max_sweeps : int
            Maximum number of Jacobi sweeps per degenerate subspace.
        tol : float
            Convergence tolerance for rotation angles (radians).
        verbose : bool
            If True, print progress information.
            
        Returns
        -------
        y_localized : array
            Packed state vector with localized states, ordered by energy.
        character : array
            Subsystem character matrix (num_states, num_species).
        """
        C, bs = self._y_to_Cb(y)
        num_states = C.shape[0]
        
        if num_states <= 1:
            character = self.compute_subsystem_character(C)
            return y, character
        
        # --- Step 1: Compute per-state energies and sort ---
        energies = jnp.real(self.calculate_energy(0.0, y))
        energy_order = jnp.argsort(energies)
        C = C[energy_order]
        energies = energies[energy_order]
        
        if verbose:
            print(f"  State energies: {[f'{e:.6f}' for e in energies]}")
        
        # --- Step 2: Group degenerate states ---
        groups = []
        current_group = [0]
        for i in range(1, num_states):
            if jnp.abs(energies[i] - energies[current_group[0]]) < energy_tol:
                current_group.append(i)
            else:
                groups.append(current_group)
                current_group = [i]
        groups.append(current_group)
        
        if verbose:
            for g_idx, group in enumerate(groups):
                E_avg = float(jnp.mean(energies[jnp.array(group)]))
                print(f"  Group {g_idx}: states {group}, E ≈ {E_avg:.6f}, size={len(group)}")
        
        # --- Step 3: Localize within each degenerate subspace ---
        C_shape = C.shape
        
        for group in groups:
            if len(group) <= 1:
                continue
            
            group_indices = jnp.array(group)
            C_sub = C[group_indices]  # (group_size, n_det_0, ...)
            n_sub = len(group)
            
            initial_purity = self._compute_total_purity(C_sub)
            if verbose:
                print(f"\n  Localizing group {group} (purity={initial_purity:.6f})...")
            
            # Jacobi sweeps
            for sweep in range(max_sweeps):
                max_rotation = 0.0
                
                for ii in range(n_sub):
                    for jj in range(ii + 1, n_sub):
                        # Compute optimal rotation angle analytically
                        theta_opt = self._compute_optimal_angle(C_sub, ii, jj)
                        
                        if jnp.abs(theta_opt) > tol:
                            max_rotation = max(max_rotation, float(jnp.abs(theta_opt)))
                            
                            # Apply rotation
                            cos_t = jnp.cos(theta_opt)
                            sin_t = jnp.sin(theta_opt)
                            C_i_new = cos_t * C_sub[ii] + sin_t * C_sub[jj]
                            C_j_new = -sin_t * C_sub[ii] + cos_t * C_sub[jj]
                            C_sub = C_sub.at[ii].set(C_i_new)
                            C_sub = C_sub.at[jj].set(C_j_new)
                
                if verbose:
                    purity = self._compute_total_purity(C_sub)
                    print(f"    Sweep {sweep+1}: max_rotation={max_rotation:.6f}, purity={purity:.6f}")
                
                if max_rotation < tol:
                    if verbose:
                        print(f"    Converged after {sweep+1} sweeps")
                    break
            
            # Write back
            C = C.at[group_indices].set(C_sub)
            
            if verbose:
                final_purity = self._compute_total_purity(C_sub)
                print(f"    Purity: {initial_purity:.6f} → {final_purity:.6f}")
        
        # --- Step 4: Return ---
        y_localized = self._Cb_to_y(C, bs)
        character = self.compute_subsystem_character(C)
        
        return y_localized, character
    
    def _compute_total_purity(self, C_sub):
        """Compute total subsystem purity: Σ_α Σ_s Tr(ρ_s[α]²).
        
        For a Hartree product state, each purity = 1 (maximum).
        For an entangled state, purity < 1.
        """
        n_sub = C_sub.shape[0]
        total_purity = 0.0
        
        for alpha in range(n_sub):
            for s in range(self.num_species):
                rho_s = self._subsystem_rdm(C_sub[alpha], s)
                total_purity += float(jnp.real(jnp.trace(rho_s @ rho_s)))
        
        return total_purity
    
    def _subsystem_rdm(self, C_single, s):
        """Compute reduced density matrix of species s for a single state.
        
        ρ_s = Tr_{not s}(|ψ⟩⟨ψ|) in the CI determinant space.
        
        Parameters
        ----------
        C_single : array
            CI tensor for a single state, shape (n_det_0, ..., n_det_{K-1}).
        s : int
            Species index.
            
        Returns
        -------
        rho : array
            Reduced density matrix, shape (n_det_s, n_det_s).
        """
        # Move species s axis to front, flatten the rest
        C_moved = jnp.moveaxis(C_single, s, 0)
        n_det_s = C_moved.shape[0]
        M = C_moved.reshape(n_det_s, -1)
        
        # ρ_s = M @ M†
        rho = M @ M.conj().T
        return rho
    
    def _compute_optimal_angle(self, C_sub, ii, jj):
        """Compute optimal Jacobi rotation angle for states ii, jj.
        
        Maximizes total purity Σ_s [Tr(ρ'_s[i]²) + Tr(ρ'_s[j]²)]
        using the analytical formula θ_opt = atan2(2E, D) / 4.
        """
        D_total = 0.0
        E_total = 0.0
        
        for s in range(self.num_species):
            # Compute subsystem density matrices and transition density
            rho_i = self._subsystem_rdm(C_sub[ii], s)
            rho_j = self._subsystem_rdm(C_sub[jj], s)
            
            # Transition density: T = M_i @ M_j†
            C_i_moved = jnp.moveaxis(C_sub[ii], s, 0)
            C_j_moved = jnp.moveaxis(C_sub[jj], s, 0)
            n_det_s = C_i_moved.shape[0]
            M_i = C_i_moved.reshape(n_det_s, -1)
            M_j = C_j_moved.reshape(n_det_s, -1)
            T = M_i @ M_j.conj().T
            
            # Symmetrized transition: R = T + T†
            R = T + T.conj().T
            
            # Compute traces (all real for Hermitian matrices)
            t_AA = jnp.real(jnp.trace(rho_i @ rho_i))
            t_BB = jnp.real(jnp.trace(rho_j @ rho_j))
            t_AB = jnp.real(jnp.trace(rho_i @ rho_j))
            t_RR = jnp.real(jnp.trace(R @ R))
            t_AR = jnp.real(jnp.trace(rho_i @ R))
            t_BR = jnp.real(jnp.trace(rho_j @ R))
            
            D_s = t_AA + t_BB - t_RR - 2 * t_AB
            E_s = t_AR - t_BR
            
            D_total += float(D_s)
            E_total += float(E_s)
        
        # Analytical optimal angle
        theta_opt = jnp.arctan2(2 * E_total, D_total) / 4
        
        return theta_opt

    def integrate(
        self,
        integrator,
        num_steps,
        t_init=0,
        bs_init_list=None,
        C_init=None,
        normalize=None,
    ):
        """Run the simulation."""
        
        # Validate bs_init_list shapes if provided
        if bs_init_list is not None:
            if len(bs_init_list) != self.num_species:
                raise ValueError(
                    f"bs_init_list must have {self.num_species} elements, got {len(bs_init_list)}"
                )
            for s, b in enumerate(bs_init_list):
                expected_shape = (self.num_spatial_orbitals[s], self.num_mctdhf_orbitals[s])
                if b.shape != expected_shape:
                    raise ValueError(
                        f"bs_init_list[{s}] has shape {b.shape}, expected {expected_shape}"
                    )
        
        # Initialization
        if bs_init_list is None:
            bs_init_list = []
            for s in range(self.num_species):
                rng = np.random.default_rng(42) # Use same seed for all species to ensure symmetry if dimensions match
                r = rng.random((self.num_spatial_orbitals[s], self.num_mctdhf_orbitals[s]))
                u, _, vh = np.linalg.svd(r, full_matrices=False)
                bs_init_list.append(jnp.array((u @ vh), jnp.complex64))
        
        if C_init is None:
             rng = np.random.default_rng(42)
             dummy_dims = (self.num_states,) + tuple(self.num_dets)
             r = rng.random(dummy_dims)
             C_init = jnp.array(r, jnp.complex64)
             C_init = C_init / jnp.linalg.norm(C_init)

        y_init = self._Cb_to_y(C_init, bs_init_list)
        
        t = t_init
        y = y_init
        
        do_normalize = normalize or self.imag_time
        
        def step_fn(carry, _):
            t, y = carry
            t, y = integrator(t, y)
            
            if do_normalize:
                C, bs = self._y_to_Cb(y)
                C = self.ortonorm_C(C)
                # Stack bs and vmap orthonormalization (assumes same shape)
                # bs_stacked = jnp.stack(bs, axis=0)
                # bs_stacked = jax.vmap(self.ortonorm_b)(bs_stacked)
                # bs_new = [bs_stacked[s] for s in range(self.num_species)]

                # In case the shapes are not the same, use a loop
                bs_new = [self.ortonorm_b(b) for b in bs]
                
                y = self._Cb_to_y(C, bs_new)
            
            return (t, y), (t, y)
        
        init_carry = (t, y)
        _, (ts_arr, ys_arr) = jax.lax.scan(step_fn, init_carry, None, length=num_steps)
        
        # Prepend initial values
        ts = jnp.concatenate([jnp.array([t_init]), ts_arr])
        ys = jnp.concatenate([y_init[None, :], ys_arr])
        
        return ts, ys


class MCTDHF_n_species_DVR(MCTDHF_n_species):
    """MCTDHF_n_species variant for DVR basis.
    
    In the DVR basis, the 2-body integrals g are diagonal in spatial
    coordinates, i.e. g is 2D with g_{ij} = g(x_i, x_j) instead of 4D.
    """
    
    @partial(jit, static_argnums=0)
    def transform_g(self, g, b_sigma, b_tau):
        """Transform two-particle integrals for DVR basis.
        
        For DVR basis, g is diagonal: g_{ij} = g(x_i, x_j) (2D array).
        
        Args:
            g: Spatial integrals g_{ij} (2D, diagonal in real space)
            b_sigma: Orbitals for particle 1 (spatial, mo)
            b_tau: Orbitals for particle 2 (spatial, mo)
            
        Returns:
            g_3: Mean-field tensor for sigma (one spatial index, three MO indices)
            g_4: Fully transformed MO integrals
            g_5: Back-transformed mean-field tensor (projected W)
        """
        # Transform j (tau) and l->s (tau): g_{ij} -> g_{iqs}
        # Note: For sinc basis, j and l share the same spatial index in g
        g_2 = contract("jq, js, ij -> iqs", b_tau.conj(), b_tau, g, backend="jax")
        
        # Transform k->r (sigma) to add r index: g_{iqs} -> g_{iqrs}
        # Note: For sinc basis, i and k share the same spatial index
        g_3 = contract("ir, iqs -> iqrs", b_sigma, g_2, backend="jax")
        
        # Transform i (sigma): g_{iqrs} -> g_{pqrs}
        g_4 = contract("ip, iqrs -> pqrs", b_sigma.conj(), g_3, backend="jax")
        
        # Back-transform i (sigma): g_{pqrs} -> g_{iqrs}
        g_5 = contract("ip, pqrs -> iqrs", b_sigma, g_4, backend="jax")
        
        return g_3, g_4, g_5

    def _transpose_g(self, g):
        """Transpose g integrals for DVR basis (2D)."""
        return jnp.transpose(g, (1, 0))
