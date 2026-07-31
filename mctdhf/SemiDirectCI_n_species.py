"""Semi-Direct Configuration Interaction for N distinguishable species.

Implements the Semi-Direct CI approach from Hochstuhl et al. for an arbitrary
number of distinguishable particle species. Excitation operators (E_pq, E_pqrs)
are precomputed as dense tensors at initialization, then contracted with the
CI coefficient tensor at runtime via opt_einsum.

Reference:
    Hochstuhl, Bonitz — "Time-dependent multiconfiguration methods for the
    numerical simulation of photoionization processes of many-electron atoms"
"""

import numpy as np
from opt_einsum import contract
from jax import jit
import jax.numpy as jnp
from functools import partial
from itertools import combinations


class SemiDirectCI_n_species:
    """Configuration Interaction solver for N distinguishable species.

    Generalizes the 2-species (alpha/beta) implementation to N species,
    each with its own set of Slater determinants and excitation operators.

    Parameters
    ----------
    num_spatial_orbitals : int or list of int
        Number of spatial (MCTDHF) orbitals per species. If a single int,
        the same value is used for all species.
    num_particles : list of int
        Number of particles per species [N_0, N_1, ..., N_{K-1}].

    Attributes
    ----------
    E_pq : list of jnp.ndarray
        Single-excitation tensors E^(s)_{IJ,pq} for each species s.
    E_pqrs : list of jnp.ndarray
        Double-excitation tensors E^(s)_{IJ,pqrs} for each species s.
    num_dets : list of int
        Number of Slater determinants per species.
    """

    def __init__(
        self,
        num_spatial_orbitals,  # int or list of ints
        num_particles,  # list of ints [N_0, N_1, ..., N_{k-1}]
    ):
        self.num_species = len(num_particles)
        self.num_particles = num_particles

        if isinstance(num_spatial_orbitals, int):
            self.num_spatial_orbitals = [num_spatial_orbitals] * self.num_species
        else:
            self.num_spatial_orbitals = num_spatial_orbitals

        assert len(self.num_spatial_orbitals) == self.num_species

        # Precompute determinant data and excitation operators per species
        self.I = []  # Slater determinants (boolean occupation vectors)
        self.W = []  # Vertex weights for address computation
        self.num_dets = []
        self.E_pq = []  # 1-body excitation tensors
        self.E_pqrs = []  # 2-body excitation tensors

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

        # Convert to JAX arrays once — avoids repeated implicit transfers in
        # JIT-compiled methods that use opt_einsum with backend="jax"
        self.E_pq = [jnp.array(e, dtype=jnp.float32) for e in self.E_pq]
        self.E_pqrs = [jnp.array(e, dtype=jnp.float32) for e in self.E_pqrs]

    # ── Determinant enumeration and addressing ────────────────────────────

    @staticmethod
    def get_slater_dets(num_orbitals, num_electrons):
        """Generate all Slater determinants as boolean occupation vectors.

        The ordering agrees with the vertex-weight address function.
        """
        dets = []
        if num_electrons == 0:
            dets.append(np.zeros(num_orbitals))
        elif num_electrons == num_orbitals:
            dets.append(np.ones(num_orbitals))
        else:
            for d in SemiDirectCI_n_species.get_slater_dets(
                num_orbitals - 1, num_electrons
            ):
                dets.append(np.concatenate((d, [0])))
            for d in SemiDirectCI_n_species.get_slater_dets(
                num_orbitals - 1, num_electrons - 1
            ):
                dets.append(np.concatenate((d, [1])))
        return np.array(dets, dtype=bool)

    @staticmethod
    def get_address_weights(num_orbitals, num_electrons):
        """Compute vertex weights for determinant addressing.

        See Fig. 19 and Eq. 3.70 in Hochstuhl et al.
        """
        W = np.zeros((num_orbitals + 1, num_electrons + 1))
        W[:, 0] = 1
        for m in range(1, num_orbitals + 1):
            for k in range(1, num_electrons + 1):
                W[m, k] = W[m - 1, k] + W[m - 1, k - 1]
        return W

    def address(self, n, num_spatial_orbitals, W):
        """Map an occupation vector to its integer address via vertex weights."""
        res = 0
        for m in range(num_spatial_orbitals):
            res += n[m] * W[m, int(np.sum(n[: m + 1]))]
        return int(res)

    # ── Second-quantized excitation evaluation ────────────────────────────

    def single_exc(self, p, q, n):
        """Evaluate single excitation a†_p a_q on determinant n.

        Returns (phase, new_det) or (0, 0) if the excitation is forbidden.
        """
        n_pq = np.copy(n)
        n_pq[q] = 0
        if n_pq[p] == 1:
            return 0, 0
        n_pq[p] = 1
        gamma = (-1) ** (np.sum(n[:q]) + np.sum(n_pq[:p]))
        return gamma, n_pq

    def double_exc(self, p, q, s, r, n):
        """Evaluate double excitation a†_p a†_q a_s a_r on determinant n.

        Returns (phase, new_det) or (0, 0) if the excitation is forbidden.
        """
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

    # ── Excitation tensor construction (numpy, done once at init) ─────────

    def _calculate_Epq(self, num_spatial_orbitals, num_dets, spin_strings, W):
        """Build the 1-body excitation tensor E_{IJ,pq}."""
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
        """Build the 2-body excitation tensor E_{IJ,pqrs}."""
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

    # ── Einsum index helpers ──────────────────────────────────────────────

    def _species_indices(self, s, out_char="x"):
        """Build einsum index strings for contracting an operator on species s.

        Returns
        -------
        c_str : str
            Input CI index string, e.g. "abc" for 3 species.
        out_str : str
            Output CI index string with species s replaced by out_char.
        op_str : str
            Operator index string "{out_char}{in_char}".
        """
        c_indices = [chr(97 + i) for i in range(self.num_species)]
        in_char = c_indices[s]
        op_str = out_char + in_char
        c_str = "".join(c_indices)
        out_indices = list(c_indices)
        out_indices[s] = out_char
        out_str = "".join(out_indices)
        return c_str, out_str, op_str

    # ── JAX-compiled CI operations ────────────────────────────────────────

    @partial(jit, static_argnums=0)
    def precompute_ci_ops(self, h_list, g_dict):
        """Contract the MO integrals with the excitation tensors once.

        The E_pq·h and E_pqrs·g precontractions (and the inter-species
        T1/Op_12 build) depend only on the integrals, not on C. When the
        integrals are frozen — as they are across every Lanczos/Krylov
        matvec of a CMF substep — this is loop-invariant work, so callers
        that apply H|C⟩ repeatedly should call this once and then
        :meth:`apply_ci_ops` per matvec.

        Parameters
        ----------
        h_list : list of jnp.ndarray
            Transformed 1-body integrals [h_0, h_1, ...], each (n_mo, n_mo).
        g_dict : dict
            Transformed 2-body integrals {(s1, s2): g_pqrs}.

        Returns
        -------
        ops : (list, dict)
            ops_diag[s] is the (n_dets_s, n_dets_s) determinant-space
            operator acting on species s alone: E_pq·h_s plus, when an
            intra-species interaction is present, 0.5·E_pqrs·g_ss.
            ops_pair[(s1, s2)] is the (D_1, D_1, D_2, D_2) inter-species
            operator with the counting factor folded in.
        """
        ops_diag = []
        for s in range(self.num_species):
            Op_s = contract("ijpq, pq -> ij", self.E_pq[s], h_list[s], backend="jax")
            if (s, s) in g_dict:
                # Intra-species: factor 0.5 (double counting in sum)
                Op_s = Op_s + 0.5 * contract(
                    "ijpqrs, pqrs -> ij", self.E_pqrs[s], g_dict[(s, s)], backend="jax"
                )
            ops_diag.append(Op_s)

        ops_pair = {}
        for (s1, s2), g_int in g_dict.items():
            if s1 == s2:
                continue
            # If both (s1,s2) and (s2,s1) present → 0.5 each; else → 1.0
            factor = 0.5 if (s2, s1) in g_dict else 1.0
            T1 = contract("ijpr, pqrs -> ijqs", self.E_pq[s1], g_int, backend="jax")
            Op_12 = contract("ijqs, klqs -> ijkl", T1, self.E_pq[s2], backend="jax")
            ops_pair[(s1, s2)] = factor * Op_12

        return ops_diag, ops_pair

    @partial(jit, static_argnums=0)
    def apply_ci_ops(self, ops, C):
        """Apply precomputed determinant-space operators: σ = H|C⟩.

        `ops` is the (ops_diag, ops_pair) pair from :meth:`precompute_ci_ops`.
        Linear in C; cost is independent of the number of MO orbitals.
        """
        ops_diag, ops_pair = ops
        sigma = jnp.zeros_like(C)

        for s in range(self.num_species):
            c_str, out_str, op_str = self._species_indices(s)
            sigma = sigma + contract(
                f"{op_str},...{c_str}->...{out_str}", ops_diag[s], C, backend="jax"
            )

        for (s1, s2), Op_12 in ops_pair.items():
            c_indices = [chr(97 + i) for i in range(self.num_species)]
            op_str = "A" + c_indices[s1] + "B" + c_indices[s2]
            c_str = "".join(c_indices)
            out_indices = list(c_indices)
            out_indices[s1] = "A"
            out_indices[s2] = "B"
            out_str = "".join(out_indices)
            sigma = sigma + contract(
                f"{op_str},...{c_str}->...{out_str}", Op_12, C, backend="jax"
            )

        return sigma

    @partial(jit, static_argnums=0)
    def get_sigma(self, C, h_list, g_dict):
        """Compute σ = H|Ψ⟩ for N species.

        One-shot convenience wrapper: precompute_ci_ops + apply_ci_ops.
        Callers that apply H repeatedly with frozen integrals (SIL/CN
        matvecs) should precompute once instead.

        Parameters
        ----------
        C : jnp.ndarray
            CI coefficient tensor, shape (num_states, n_dets_0, n_dets_1, ...).
        h_list : list of jnp.ndarray
            Transformed 1-body integrals [h_0, h_1, ...], each (n_mo, n_mo).
        g_dict : dict
            Transformed 2-body integrals {(s1, s2): g_pqrs}.

        Returns
        -------
        sigma : jnp.ndarray
            Same shape as C.
        """
        return self.apply_ci_ops(self.precompute_ci_ops(h_list, g_dict), C)

    # ── Reduced Density Matrices ──────────────────────────────────────────

    def _compute_rho_ci(self, C, s):
        """Compute the reduced CI density matrix for species s.

        Contracts C* and C over all species indices except s, yielding
        ρ^CI_{xy} = Σ_{other} C*_{...x...} C_{...y...}.

        Returns shape (..., n_det_s, n_det_s) where ... is the batch dimension.
        """
        c_indices = [chr(97 + i) for i in range(self.num_species)]
        c_conj_indices = list(c_indices)
        c_conj_indices[s] = "x"  # conjugate index
        c_indices[s] = "y"

        rho_ci_str = f"...{''.join(c_conj_indices)},...{''.join(c_indices)}->...xy"
        return contract(rho_ci_str, C.conj(), C, backend="jax")

    @partial(jit, static_argnums=0)
    def get_1p_RDMs(self, C):
        """Compute 1-particle reduced density matrices for all species.

        Returns
        -------
        list of jnp.ndarray
            D_s of shape (num_states, n_orb_s, n_orb_s) for each species s.
            D^(s)_{pq} = ⟨Ψ| E^(s)_{pq} |Ψ⟩.
        """
        Ds = []
        for s in range(self.num_species):
            rho_ci = self._compute_rho_ci(C, s)
            D_s = contract("ijpq, ...ij -> ...pq", self.E_pq[s], rho_ci, backend="jax")
            Ds.append(D_s)
        return Ds

    @partial(jit, static_argnums=0)
    def get_2p_RDMs(self, C):
        """Compute 2-particle reduced density matrices for all species pairs.

        Returns
        -------
        dict of {(s1, s2): jnp.ndarray}
            d^(s1,s2)_{pqrs} = ⟨Ψ| E^(s1)_{pqrs} |Ψ⟩  (intra-species)
            d^(s1,s2)_{pqrs} = ⟨Ψ| E^(s1)_{pr} E^(s2)_{qs} |Ψ⟩  (inter-species)
        """
        ds = {}

        # Intra-species: reuse shared rho_ci helper
        for s1 in range(self.num_species):
            rho_ci = self._compute_rho_ci(C, s1)
            ds[(s1, s1)] = contract(
                "ijpqrs, ...ij -> ...pqrs",
                self.E_pqrs[s1],
                rho_ci,
                backend="jax",
            )

        # Inter-species: compute s1 < s2 only. Excitation operators of
        # different species commute, so the reversed pair is a pure index
        # permutation: d^(s2,s1)_{pqrs} = ⟨E^(s2)_{pr} E^(s1)_{qs}⟩
        #                              = d^(s1,s2)_{qpsr}.
        for s1 in range(self.num_species):
            for s2 in range(s1 + 1, self.num_species):
                # Contract over all indices except s1 and s2
                idx_conj = [chr(97 + i) for i in range(self.num_species)]
                idx_norm = [chr(97 + i) for i in range(self.num_species)]
                idx_conj[s1], idx_conj[s2] = "x", "z"
                idx_norm[s1], idx_norm[s2] = "y", "w"

                rho_pair_str = (
                    f"...{''.join(idx_conj)},...{''.join(idx_norm)}->...xzyw"
                )
                rho_pair = contract(rho_pair_str, C.conj(), C, backend="jax")

                d_inter = contract(
                    "...xzyw, xypr, zwqs -> ...pqrs",
                    rho_pair,
                    self.E_pq[s1],
                    self.E_pq[s2],
                    backend="jax",
                )
                ds[(s1, s2)] = d_inter
                ds[(s2, s1)] = jnp.einsum("...pqrs->...qpsr", d_inter)

        return ds

    # ── Energy calculation ────────────────────────────────────────────────

    @partial(jit, static_argnums=0)
    def calculate_energy(self, C, element_h, element_g):
        """Compute total energy from transformed integrals.

        Parameters
        ----------
        C : jnp.ndarray
            CI coefficients.
        element_h : list of jnp.ndarray
            Transformed 1-body integrals per species.
        element_g : dict of {(s1, s2): jnp.ndarray}
            Transformed 2-body integrals.
        """
        Ds = self.get_1p_RDMs(C)
        ds = self.get_2p_RDMs(C)

        E = 0.0

        # 1-body energy: Σ_s Tr(D_s · h_s)
        for s, h in enumerate(element_h):
            E += contract("...pq, pq -> ...", Ds[s], h, backend="jax")

        # 2-body energy (with correct counting factors)
        for (s1, s2), g in element_g.items():
            if s1 == s2:
                factor = 0.5  # Intra-species: avoid double counting
            else:
                factor = 0.5 if (s2, s1) in element_g else 1.0
            d = ds[(s1, s2)]
            E += factor * contract("pqrs, ...pqrs -> ...", g, d, backend="jax")

        return E
