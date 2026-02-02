import numpy as np
from opt_einsum import contract
from jax import jit
import jax.numpy as jnp
from functools import partial
from itertools import combinations

class SemiDirectCI_n_species:
    """Class for running SemiDirectCI calculations with N distinguishable species.

    Generalizes the 2-species (alpha/beta) implementation to N species.
    """

    def __init__(
        self,
        num_spatial_orbitals,  # int or list of ints
        num_particles,         # list of ints [N_0, N_1, ..., N_{k-1}]
    ):
        self.num_species = len(num_particles)
        self.num_particles = num_particles
        
        # Handle uniform or species-specific spatial orbitals
        if isinstance(num_spatial_orbitals, int):
            self.num_spatial_orbitals = [num_spatial_orbitals] * self.num_species
        else:
            self.num_spatial_orbitals = num_spatial_orbitals
            
        assert len(self.num_spatial_orbitals) == self.num_species
        
        # Initialize storage for each species
        self.I = []      # Slater determinants
        self.W = []      # Address weights
        self.num_dets = []
        self.E_pq = []   # 1-body excitations [species]
        self.E_pqrs = [] # 2-body excitations [species]
        
        # Precompute data for each species
        for s in range(self.num_species):
            n_orb = self.num_spatial_orbitals[s]
            n_part = self.num_particles[s]
            
            # Slater determinants
            I_s = self.get_slater_dets(n_orb, n_part)
            self.I.append(I_s)
            self.num_dets.append(I_s.shape[0])
            
            # Address weights
            W_s = self.get_address_weights(n_orb, n_part)
            self.W.append(W_s)
            
            # Excitation operators
            self.E_pq.append(self._calculate_Epq(n_orb, self.num_dets[s], I_s, W_s))
            self.E_pqrs.append(self._calculate_Epqrs(n_orb, self.num_dets[s], I_s, W_s))

    @staticmethod
    def get_slater_dets(num_orbitals, num_electrons):
        """Generate a list of all Slater determinants. Order agrees with the address function."""
        # Reuse logic from original class
        dets = []
        if num_electrons == 0:
            dets.append(np.zeros(num_orbitals))
        elif num_electrons == num_orbitals:
            dets.append(np.ones(num_orbitals))
        else:
            for d in SemiDirectCI_n_species.get_slater_dets(num_orbitals - 1, num_electrons):
                dets.append(np.concatenate((d, [0])))
            for d in SemiDirectCI_n_species.get_slater_dets(num_orbitals - 1, num_electrons - 1):
                dets.append(np.concatenate((d, [1])))
        return np.array(dets, dtype=bool)

    @staticmethod
    def get_address_weights(num_orbitals, num_electrons):
        """Calculate node weights, see Fig. 19 and equation 3.70 in Hochstuhl et al."""
        W = np.zeros((num_orbitals + 1, num_electrons + 1))
        W[:, 0] = 1
        for m in range(1, num_orbitals + 1):
            for k in range(1, num_electrons + 1):
                W[m, k] = W[m - 1, k] + W[m - 1, k - 1]
        return W

    def address(self, n, num_spatial_orbitals, W):
        """Use node weights to calculate address."""
        res = 0
        for m in range(num_spatial_orbitals):
            res += n[m] * W[m, int(np.sum(n[: m + 1]))]
        return int(res)

    def single_exc(self, p, q, n):
        """Evaluate a single particle excitation a^+_pa_q."""
        n_pq = np.copy(n)
        n_pq[q] = 0
        if n_pq[p] == 1:
            return 0, 0
        n_pq[p] = 1
        gamma = (-1) ** (np.sum(n[:q]) + np.sum(n_pq[:p]))
        return gamma, n_pq

    def double_exc(self, p, q, s, r, n):
        """Evaluate a two particle excitation a^+_pa^+_qa_sa_r."""
        if s == r or p == q:
            return 0, 0
        n_pqsr = np.copy(n)
        n_pqsr[r] = 0
        gamma = np.sum(n_pqsr[:r])
        n_pqsr[s] = 0
        gamma += np.sum(n_pqsr[:s])
        if n_pqsr[q] == 1 or n_pqsr[p] == 1:
            return 0, 0
        n_pqsr[q] = 1
        gamma += np.sum(n_pqsr[:q])
        n_pqsr[p] = 1
        gamma += np.sum(n_pqsr[:p])
        return (-1) ** gamma, n_pqsr

    def _calculate_Epq(self, num_spatial_orbitals, num_dets, spin_strings, W):
        """Calculate matrix of single particle excitations."""
        E_pq = np.zeros((num_dets,) * 2 + (num_spatial_orbitals,) * 2)
        for J in spin_strings:
            j = self.address(J, num_spatial_orbitals, W)
            occ = np.flatnonzero(J)
            for p in range(num_spatial_orbitals):
                for q in occ:
                    xi, I = self.single_exc(p, q, J)
                    if xi != 0:
                        i = self.address(I, num_spatial_orbitals, W)
                        E_pq[i, j, p, q] += xi
        return E_pq

    def _calculate_Epqrs(self, num_spatial_orbitals, num_dets, spin_strings, W):
        """Calculate matrix of two particle excitations."""
        E_pqrs = np.zeros((num_dets,) * 2 + (num_spatial_orbitals,) * 4)
        for J in spin_strings:
            j = self.address(J, num_spatial_orbitals, W)
            occ = np.flatnonzero(J)
            for p in range(num_spatial_orbitals):
                for r in occ:
                    for q in range(num_spatial_orbitals):
                        for s in occ:
                            xi, I = self.double_exc(p, q, s, r, J)
                            if xi != 0:
                                i = self.address(I, num_spatial_orbitals, W)
                                E_pqrs[i, j, p, q, r, s] += xi
        return E_pqrs

    # --- JAX-Compatible Methods ---

    @partial(jit, static_argnums=0)
    def get_sigma(self, C, h_list, g_dict):
        """Calculate sigma = H|Psi> for N species.
        
        Args:
            C: CI coefficients tensor (batch, n_dets_0, n_dets_1, ...)
            h_list: List of 1-body integrals [h_0, h_1, ...]
            g_dict: Dict of 2-body integrals {(s1,s2): g_tensor}
            
        Returns:
            Sigma tensor of same shape as C.
        """
        sigma = jnp.zeros_like(C)
        
        # 1. One-body terms: sum_s H_s
        for s in range(self.num_species):
            sigma = sigma + self._contract_1body(C, h_list[s], s)
            
        # 2. Two-body terms: 0.5 * sum_{s1, s2} V_{s1,s2}
        # Note: g_dict contains transformed integrals g_{pqrs}
        
        # Iterate over all pairs (s1, s2)
        # We assume g_dict has keys for all valid interactions
        for (s1, s2), g_int in g_dict.items():
            if s1 == s2:
                # Intra-species: factor 0.5 (double counting in sum)
                factor = 0.5 
                term = self._contract_2body_intra(C, g_int, s1)
            else:
                # Inter-species: check if BOTH orderings are in g_dict
                # If both (s1,s2) and (s2,s1) are present, use factor 0.5 to avoid double counting
                # If only one is present, use factor 1.0 (to account for missing symmetric term)
                if (s2, s1) in g_dict:
                    factor = 0.5  # Both orderings present, avoid double counting
                else:
                    factor = 1.0  # Only one ordering, need full contribution
                term = self._contract_2body_inter(C, g_int, s1, s2)
            
            sigma = sigma + factor * term
            
        return sigma

    def _contract_1body(self, C, h, s):
        """Compute sigma contribution from 1-body operator on species s.
        
        Op: E_pq[s] * h_pq
        Contraction: Sum_{p,q} Sum_{I_s} E_{J_s, I_s, p, q} * h_{p,q} * C_{..., I_s, ...}
        
        """
        
        # Build einsum string for contracting Species-specific operator with C
        # C indices: a, b, c, ...
        c_indices = [chr(97 + i) for i in range(self.num_species)]
        
        # E[s] indices: c'_s (output), c_s (input), p, q
        # Use 'x' for output determinant index
        target_idx_char = c_indices[s]
        output_idx_char = 'x'
        
        # Contract E and h first: sigma_1body_op = E_{J,I,p,q} * h_{p,q} -> Op_{J,I}
        Op_s = contract("ijpq, pq -> ij", self.E_pq[s], h, backend="jax")
        
        # Now contract Op_s with C
        op_str = output_idx_char + target_idx_char
        c_str = "".join(c_indices)
        
        out_indices = list(c_indices)
        out_indices[s] = output_idx_char
        out_str = "".join(out_indices)
        
        # Equation: "ij, ...j... -> ...i..."
        einsum_str = f"{op_str},...{c_str}->...{out_str}"
        
        return contract(einsum_str, Op_s, C, backend="jax")

    def _contract_2body_intra(self, C, g, s):
        """Compute sigma contribution from intra-species 2-body operator."""
        # E_pqrs[s]: [Out, In, p, q, r, s]
        # g: [p, q, r, s]
        
        # Pre-contract E and g -> Op_{J,I}
        Op_s = contract("ijpqrs, pqrs -> ij", self.E_pqrs[s], g, backend="jax")
        
        # Contract with C (same as 1-body)
        c_indices = [chr(97 + i) for i in range(self.num_species)]
        target_idx_char = c_indices[s]
        output_idx_char = 'x'
        
        op_str = output_idx_char + target_idx_char
        c_str = "".join(c_indices)
        out_indices = list(c_indices)
        out_indices[s] = output_idx_char
        out_str = "".join(out_indices)
        
        einsum_str = f"{op_str},...{c_str}->...{out_str}"
        return contract(einsum_str, Op_s, C, backend="jax")

    def _contract_2body_inter(self, C, g, s1, s2):
        """Compute sigma contribution from inter-species 2-body operator.
        
        Op = g_{pqrs} * E_{p,r}^(s1) * E_{q,s}^(s2)
        Indices: g corresponds to <pq|rs>
        """
        
        # 1. Contract g with E^(s1)
        # E^(s1): [Out1, In1, p, r]
        # g: [p, q, r, s]
        # Result T1: [Out1, In1, q, s]
        # "ijpr, pqrs -> ijqs"
        T1 = contract("ijpr, pqrs -> ijqs", self.E_pq[s1], g, backend="jax")
        
        # 2. Contract T1 with E^(s2)
        # E^(s2): [Out2, In2, q, s]
        # T1: [Out1, In1, q, s]
        # Result Op_12: [Out1, In1, Out2, In2]
        # "ijqs, klqs -> ijkl"
        Op_12 = contract("ijqs, klqs -> ijkl", T1, self.E_pq[s2], backend="jax")
        
        # 3. Contract with C
        # Op_12 indices: x (out1), y (in1), z (out2), w (in2)
        # C indices: a, ..., y (at s1), ..., w (at s2), ...
        # Output: a, ..., x (at s1), ..., z (at s2), ...
        
        c_indices = [chr(97 + i) for i in range(self.num_species)]
        in1_char = c_indices[s1]
        in2_char = c_indices[s2]
        
        out1_char = 'A'
        out2_char = 'B'
        
        op_str = out1_char + in1_char + out2_char + in2_char
        c_str = "".join(c_indices)
        
        out_indices = list(c_indices)
        out_indices[s1] = out1_char
        out_indices[s2] = out2_char
        out_str = "".join(out_indices)
        
        einsum_str = f"{op_str},...{c_str}->...{out_str}"
        return contract(einsum_str, Op_12, C, backend="jax")

    @partial(jit, static_argnums=0)
    def get_1p_RDMs(self, C):
        """Calculate 1-particle RDMs for all species.
        
        Returns:
            List of RDMs [D_0, D_1, ..., D_{N-1}]
            D_s shape: (n_orb_s, n_orb_s)
        """
        Ds = []
        for s in range(self.num_species):
            # Formula: D_{pq}^{(s)} = <Psi | E_{pq}^{(s)} | Psi>
            # = Sum_{Out, In} E_{Out, In, p, q}^{(s)} * Rho_{In, Out}^{(s)}
            # Where Rho_{In, Out}^{(s)} is the reduced density matrix of the CI vector
            # obtained by tracing out all other species.
            
            # 1. Compute Reduced CI matrix Rho_CI^{(s)}_{In, Out}
            # Contract C.conj() * C over all indices EXCEPT s
            # C*: [a, ..., x (s), ...]
            # C:  [a, ..., y (s), ...]
            # Out: [x, y]
            
            c_indices = [chr(97 + i) for i in range(self.num_species)]
            c_conj_indices = list(c_indices)
            c_conj_indices[s] = 'x' # conjugage index for s
            c_indices[s] = 'y'      # normal index for s
            
            c_conj_str = "".join(c_conj_indices)
            c_str = "".join(c_indices)
            
            # Sum over all shared indices (all except x and y)
            out_str = '...xy' # x=bra, y=ket, dots=batch
            
            rho_ci_str = f"...{c_conj_str},...{c_str}->{out_str}"
            rho_ci = contract(rho_ci_str, C.conj(), C, backend="jax")
            
            # 2. Contract with E_{pq}^{(s)}
            # E: [Out (x), In (y), p, q]
            # Rho: [..., x, y] 
            # Output: (batch, p, q)
            
            D_s = contract("ijpq, ...ij -> ...pq", self.E_pq[s], rho_ci, backend="jax")
            Ds.append(D_s)
            
        return Ds

    @partial(jit, static_argnums=0)
    def get_2p_RDMs(self, C):
        """Calculate 2-particle RDMs for all species pairs.
        
        Returns:
            Dict {(s1, s2): d_tensor}
        """
        ds = {}
        
        # Loop over all pairs
        for s1 in range(self.num_species):
            for s2 in range(self.num_species):
                if s1 == s2:
                    # Intra-species 2-RDM
                    # d_{pqrs} = <Psi | E_{pqrs} | Psi>
                    # Similar to 1-RDM logic
                    c_indices = [chr(97 + i) for i in range(self.num_species)]
                    c_conj_indices = list(c_indices)
                    c_conj_indices[s1] = 'x'
                    c_indices[s1] = 'y'
                    
                    rho_ci = contract(
                        f"...{''.join(c_conj_indices)},...{''.join(c_indices)}->...xy", 
                        C.conj(), C, backend="jax"
                    )
                    
                    d_intra = contract("ijpqrs, ...ij -> ...pqrs", self.E_pqrs[s1], rho_ci, backend="jax")
                    ds[(s1, s1)] = d_intra
                    
                else:
                    # Inter-species 2-RDM
                    # d_{pqrs} = <Psi | E_{pr}^{(s1)} E_{qs}^{(s2)} | Psi>
                    # Contract over all indices excep s1 and s2
                    
                    idx_conj = [chr(97 + i) for i in range(self.num_species)]
                    idx_norm = [chr(97 + i) for i in range(self.num_species)]
                    
                    idx_conj[s1] = 'x' # s1 out
                    idx_conj[s2] = 'z' # s2 out
                    
                    idx_norm[s1] = 'y' # s1 in
                    idx_norm[s2] = 'w' # s2 in
                    
                    # Reduced Rho_CI for pair (s1, s2): indices [x, z, y, w] (out1, out2, in1, in2)
                    rho_pair_str = f"...{''.join(idx_conj)},...{''.join(idx_norm)}->...xzyw"
                    rho_pair = contract(rho_pair_str, C.conj(), C, backend="jax")
                    
                    # Contract with Excitations
                    # E1: [x, y, p, r]
                    # E2: [z, w, q, s]
                    # Rho: [..., x, z, y, w]
                    
                    # "xzyw, xypr, zwqs -> pqrs"
                    d_inter = contract(
                        "...xzyw, xypr, zwqs -> ...pqrs", 
                        rho_pair, 
                        self.E_pq[s1], 
                        self.E_pq[s2], 
                        backend="jax"
                    )
                    ds[(s1, s2)] = d_inter
                    
        return ds

    @partial(jit, static_argnums=0)
    def calculate_energy(self, C, element_h, element_g):
        """Calculate energy given transformed integrals."""
        # element_h: List [h0, h1, ...]
        # element_g: Dict {(s1,s2): g}
        
        Ds = self.get_1p_RDMs(C)
        ds = self.get_2p_RDMs(C)
        
        E = 0.0
        
        # 1-body energy
        for s, h in enumerate(element_h):
            E += contract("...pq, pq -> ...", Ds[s], h, backend="jax")
            
        # 2-body energy
        for (s1, s2), g in element_g.items():
            # Same factor logic as get_sigma
            if s1 == s2:
                factor = 0.5  # Intra-species: avoid double counting
            else:
                # Check if both orderings are present
                if (s2, s1) in element_g:
                    factor = 0.5  # Both orderings present
                else:
                    factor = 1.0  # Only one ordering
            d = ds[(s1, s2)]
            # Contract g * d
            # Both are (p,q,r,s) tensors, d has batch
            E += factor * contract("pqrs, ...pqrs -> ...", g, d, backend="jax")
            
        return E
