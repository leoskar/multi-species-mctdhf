"""Multi-Configuration Time-Dependent Hartree for N distinguishable species.

Implements the MCTDHF equations of motion for an arbitrary number of
distinguishable particle species, using the Semi-Direct CI approach for
the configuration interaction part.

The state is represented as a packed vector y = [C_flat, b_0_flat, ..., b_{K-1}_flat]
containing the CI coefficients C and the orbital coefficient matrices b_s.
"""

import jax
import jax.numpy as jnp
from jax import jit
from opt_einsum import contract
from functools import partial
import numpy as np
import itertools

from .SemiDirectCI_n_species import SemiDirectCI_n_species
from .integrators import get_crank_nicolson_fixed_iter


class MCTDHF_n_species:
    """MCTDHF solver for N distinguishable species.

    Parameters
    ----------
    h_list : list of ndarray or callable
        1-body Hamiltonians per species. Each element is either a matrix
        (n_spatial, n_spatial) or a callable t → matrix for time-dependent h.
    g_dict : dict of {(int, int): ndarray}
        2-body interaction integrals. Key (s1, s2) maps to the spatial
        integral tensor ⟨s1 s2|V|s1 s2⟩. Only one ordering per pair is
        needed; the transpose is computed automatically.
    num_particles : list of int
        Number of particles per species.
    num_mctdhf_orbitals : list of int
        Number of MCTDHF (active) orbitals per species.
    num_spatial_orbitals : list of int
        Number of spatial basis functions (grid points) per species.
    num_states : int
        Number of electronic states to propagate simultaneously.
    imag_time : bool
        If True, propagate in imaginary time (for ground-state search).
    """

    def __init__(
        self,
        h_list,
        g_dict,
        num_particles,
        num_mctdhf_orbitals,
        num_spatial_orbitals,
        num_states=1,
        imag_time=False,
    ):
        err_msg = "Input lists must have the same length (number of species)."
        np.testing.assert_equal(len(num_particles), len(num_mctdhf_orbitals), err_msg)
        np.testing.assert_equal(len(num_particles), len(num_spatial_orbitals), err_msg)
        np.testing.assert_equal(len(num_particles), len(h_list), err_msg)

        self.num_species = len(num_particles)
        self.num_particles = num_particles
        self.num_mctdhf_orbitals = num_mctdhf_orbitals
        self.num_spatial_orbitals = num_spatial_orbitals

        self.h_is_callable = [callable(h) for h in h_list]
        self.h_list = h_list
        self.g_dict = g_dict

        self.dCI = SemiDirectCI_n_species(num_mctdhf_orbitals, num_particles)
        self.num_dets = self.dCI.num_dets
        self.num_states = num_states
        self.imag_time = imag_time

    # ── State vector packing / unpacking ──────────────────────────────────

    def _Cb_to_y(self, C, bs_list):
        """Pack CI coefficients C and orbital matrices bs into flat vector y."""
        C_flat = C.reshape(-1)
        bs_flat = [b.reshape(-1) for b in bs_list]
        return jnp.concatenate([C_flat] + bs_flat)

    def _y_to_Cb(self, y):
        """Unpack flat vector y into CI coefficients C and orbital matrices bs."""
        c_size = np.prod(self.num_dets) * self.num_states
        C = y[:c_size].reshape((self.num_states,) + tuple(self.num_dets))

        start = c_size
        bs_list = []
        for s in range(self.num_species):
            b_size = self.num_spatial_orbitals[s] * self.num_mctdhf_orbitals[s]
            b_shape = (self.num_spatial_orbitals[s], self.num_mctdhf_orbitals[s])
            bs_list.append(y[start : start + b_size].reshape(b_shape))
            start += b_size

        return C, bs_list

    # ── Integral helpers ──────────────────────────────────────────────────

    def _get_h_current(self, t):
        """Evaluate 1-body integrals at time t (handles callable and static h)."""
        return [
            self.h_list[s](t) if self.h_is_callable[s] else self.h_list[s]
            for s in range(self.num_species)
        ]

    def _transform_all_g(self, bs, full=False):
        """Transform all 2-body integrals to the MO basis.

        Parameters
        ----------
        bs : list of jnp.ndarray
            Orbital coefficient matrices.
        full : bool
            If True, return (g_3, g_4, g_5) per pair (needed for EOM).
            If False, return only g_4 per pair (sufficient for energy).

        Returns
        -------
        g_MO_dict : dict
            Fully transformed MO integrals {(s1, s2): g_4}.
        g_mix_tensors : dict or None
            Mean-field tensors {(s1, s2): (g_3, g_5)} if full=True, else None.
        """
        g_MO_dict = {}
        g_mix_tensors = {} if full else None

        for s1 in range(self.num_species):
            for s2 in range(self.num_species):
                if (s1, s2) in self.g_dict:
                    g_spatial = self.g_dict[(s1, s2)]
                elif (s2, s1) in self.g_dict:
                    g_spatial = self._transpose_g(self.g_dict[(s2, s1)])
                else:
                    continue

                g_3, g_4, g_5 = self.transform_g(g_spatial, bs[s1], bs[s2])
                g_MO_dict[(s1, s2)] = g_4
                if full:
                    g_mix_tensors[(s1, s2)] = (g_3, g_5)

        return g_MO_dict, g_mix_tensors

    # ── Integral transformations ──────────────────────────────────────────

    @partial(jit, static_argnums=0)
    def transform_h(self, h, b):
        """Transform 1-body integrals to the MCTDHF MO basis.

        Returns
        -------
        h_1 : jnp.ndarray
            Half-transformed ⟨i|h|m⟩ (spatial × MO).
        h_2 : jnp.ndarray
            Fully transformed ⟨n|h|m⟩ (MO × MO).
        h_3 : jnp.ndarray
            Projected ⟨i|P h P|m⟩ (spatial × MO).
        """
        h_1 = contract("jm, ij -> im", b, h, backend="jax")
        h_2 = contract("in, im -> nm", b.conj(), h_1, backend="jax")
        h_3 = contract("in, nm -> im", b, h_2, backend="jax")
        return h_1, h_2, h_3

    @partial(jit, static_argnums=0)
    def transform_g(self, g, b_sigma, b_tau):
        """Transform 2-body integrals ⟨σ τ|V|σ τ⟩ to the MO basis.

        Parameters
        ----------
        g : jnp.ndarray
            Spatial integrals g_{ijkl} = ⟨ij|kl⟩.
        b_sigma : jnp.ndarray
            Orbitals for particle 1 (indices i, k).
        b_tau : jnp.ndarray
            Orbitals for particle 2 (indices j, l).

        Returns
        -------
        g_3 : jnp.ndarray
            Mean-field tensor W (one spatial index, three MO indices).
        g_4 : jnp.ndarray
            Fully transformed MO integrals.
        g_5 : jnp.ndarray
            Projected mean-field ⟨i|P W P|m⟩.
        """
        # Transform tau indices: g_{ijkl} → g_{iqks}
        g_2 = contract("jq, ls, ijkl -> iqks", b_tau.conj(), b_tau, g, backend="jax")
        # Transform sigma k → r: g_{iqks} → g_{iqrs}  (mean field W)
        g_3 = contract("kr, iqks -> iqrs", b_sigma, g_2, backend="jax")
        # Transform sigma i → p: g_{iqrs} → g_{pqrs}
        g_4 = contract("ip, iqrs -> pqrs", b_sigma.conj(), g_3, backend="jax")
        # Back-transform: g_{pqrs} → g_{iqrs}  (projected W)
        g_5 = contract("ip, pqrs -> iqrs", b_sigma, g_4, backend="jax")
        return g_3, g_4, g_5

    def _transpose_g(self, g):
        """Transpose 2-body integrals to swap particle 1 ↔ particle 2.

        For 4D integrals: (i,j,k,l) → (j,i,l,k).
        Override in DVR subclass for the 2D diagonal case.
        """
        return jnp.transpose(g, (1, 0, 3, 2))

    # ── Energy calculation ────────────────────────────────────────────────

    @partial(jit, static_argnums=0)
    def calculate_energy(self, t, y):
        """Calculate total energy at time t."""
        C, bs = self._y_to_Cb(y)

        h_current = self._get_h_current(t)
        h_MO_list = [self.transform_h(h_current[s], bs[s])[1] for s in range(self.num_species)]
        g_MO_dict, _ = self._transform_all_g(bs, full=False)

        return self.dCI.calculate_energy(C, h_MO_list, g_MO_dict)

    # ── Observables ───────────────────────────────────────────────────────

    @partial(jit, static_argnums=0)
    def calculate_particle_densities(self, C, bs_list, spf_list=None):
        """Calculate particle densities for all species.

        Parameters
        ----------
        C : jnp.ndarray
            CI coefficients.
        bs_list : list of jnp.ndarray
            Orbital coefficient matrices.
        spf_list : list of jnp.ndarray, optional
            Single-particle functions on the grid. If provided, returns
            density on the grid; otherwise returns the density matrix in
            the spatial orbital basis.

        Returns
        -------
        list of jnp.ndarray
            Density for each species.
        """
        Ds = self.dCI.get_1p_RDMs(C)
        densities = []

        for s in range(self.num_species):
            D_s = Ds[s]   # (batch, mo, mo)
            b_s = bs_list[s]

            if spf_list is not None and len(spf_list) > s:
                # Density on grid: ρ(x) = Σ_{nm} D_{nm} φ*_n(x) φ_m(x)
                spf = spf_list[s]
                rho = contract("ix,in,snm,jm,jx->sx", spf, b_s.conj(), D_s, b_s, spf, backend="jax")
            else:
                # Density matrix in spatial basis: ρ_{ij} = Σ_{nm} D_{nm} b*_{in} b_{jm}
                rho = contract("in, snm, jm -> sij", b_s.conj(), D_s, b_s, backend="jax")
            densities.append(rho)
        return densities

    @partial(jit, static_argnums=0)
    def calculate_overlap(self, y1, y2):
        """Calculate overlap ⟨Ψ₁|Ψ₂⟩ between two multi-species states.

        The overlap factorizes over species: for each species s, we compute the
        determinant overlap matrix S^s_{IJ} = det(O^s_{occ_I, occ_J}) where
        O^s_{ij} = ⟨φ^1_i|φ^2_j⟩ is the MO overlap. The full overlap is then
        ⟨Ψ₁|Ψ₂⟩ = Σ_{I,J,...} C₁*_{I,J,...} · C₂_{I',J',...} · S⁰_{II'} · S¹_{JJ'} · ...

        Parameters
        ----------
        y1, y2 : jnp.ndarray
            Packed state vectors.

        Returns
        -------
        jnp.ndarray
            Overlap per state, shape (num_states,).
        """
        C1, bs1 = self._y_to_Cb(y1)
        C2, bs2 = self._y_to_Cb(y2)

        # Build determinant overlap matrices S_{IJ} = det(⟨φ^1_i|φ^2_j⟩) for each species
        S_matrices = []
        for s in range(self.num_species):
            b1, b2 = bs1[s], bs2[s]
            I = jnp.array(self.dCI.I[s])
            n_dets = self.num_dets[s]
            n_elec = self.num_particles[s]

            # MO overlap O_{ij} = ⟨φ^1_i|φ^2_j⟩
            O = contract("ki, kj -> ij", b1.conj(), b2, backend="jax")

            # Extract occupied submatrices O[occ_I, occ_J] for each det pair (I, J)
            Mask = contract("ai, bj -> abij", I, I, backend="jax")
            Val = contract("ai, bj, ij -> abij", I, I, O, backend="jax")

            expected_size = n_dets * n_dets * n_elec * n_elec
            indices = jnp.nonzero(Mask.reshape(-1), size=expected_size)
            elems = Val.reshape(-1)[indices]

            mat = elems.reshape(n_dets, n_dets, n_elec, n_elec)
            S_matrices.append(jnp.linalg.det(mat))

        # Contract: ⟨Ψ₁|Ψ₂⟩ = C₁*_{s,a,b,...} · C₂_{s,d,e,...} · S⁰_{ad} · S¹_{be} · ...
        K = self.num_species
        c1 = [chr(97 + i) for i in range(K)]          # a, b, c, ...
        c2 = [chr(97 + K + i) for i in range(K)]      # d, e, f, ...
        s_pairs = ",".join(f"{c1[i]}{c2[i]}" for i in range(K))
        einsum_str = f"s{''.join(c1)},s{''.join(c2)},{s_pairs}->s"

        return contract(einsum_str, C1.conj(), C2, *S_matrices, backend="jax")

    # ── Equations of motion ───────────────────────────────────────────────

    @partial(jit, static_argnums=0)
    def __call__(self, t, y):
        """Evaluate the right-hand side dy/dt of the MCTDHF equations of motion."""
        C, bs = self._y_to_Cb(y)

        # 1. Evaluate and transform integrals
        h_current = self._get_h_current(t)

        h_MO_list = []
        h_transformed = []
        for s in range(self.num_species):
            h_1, h_2, h_3 = self.transform_h(h_current[s], bs[s])
            h_transformed.append((h_1, h_2, h_3))
            h_MO_list.append(h_2)

        g_MO_dict, g_mix_tensors = self._transform_all_g(bs, full=True)

        # 2. CI coefficients: dC/dt = ∓ H|Ψ⟩
        sigma = self.dCI.get_sigma(C, h_MO_list, g_MO_dict)

        # 3. Orbital coefficients: db_s/dt = ∓ D⁻¹ [(1-P)h + Σ_k D⁻¹ d_{sk} (1-P)W_k]
        Ds = self.dCI.get_1p_RDMs(C)
        ds = self.dCI.get_2p_RDMs(C)

        # State-average RDMs (sum over batch dimension)
        Ds = [jnp.sum(D, axis=0) for D in Ds]
        ds = {k: jnp.sum(v, axis=0) for k, v in ds.items()}

        b_dots = []
        for s in range(self.num_species):
            # Use pinv for stability with potentially singular density matrices
            D_inv = jnp.linalg.pinv(Ds[s], rcond=1e-4, hermitian=True)
            h_1, _, h_3 = h_transformed[s]

            # (1-P)h contribution
            b_dot_s = h_1 - h_3

            # Mean-field contribution: Σ_k D⁻¹ d_{sk} (1-P) W_k
            for k in range(self.num_species):
                if (s, k) in g_mix_tensors:
                    g_3, g_5 = g_mix_tensors[(s, k)]
                    d_sk = ds[(s, k)]
                    b_dot_s = b_dot_s + contract(
                        "np, pqrs, iqrs -> in", D_inv, d_sk, g_3 - g_5, backend="jax"
                    )

            if self.imag_time:
                b_dots.append(-b_dot_s)
            else:
                b_dots.append(-1j * b_dot_s)

        C_dot = -sigma if self.imag_time else -1j * sigma
        return self._Cb_to_y(C_dot, b_dots)

    # ── Orthonormalization ────────────────────────────────────────────────

    @partial(jit, static_argnums=0)
    def ortonorm_b(self, b):
        """Orthonormalize orbital columns via QR with positive-diagonal convention."""
        Q, R = jnp.linalg.qr(b)
        signs = jnp.sign(jnp.diag(R))
        signs = jnp.where(signs == 0, 1.0, signs)
        return Q * signs[None, :]

    @partial(jit, static_argnums=0)
    def ortonorm_C(self, C):
        """Orthonormalize states in C via QR with positive-diagonal convention."""
        original_shape = C.shape
        C_flat = C.reshape(C.shape[0], -1)
        # QR on the transpose: columns = state vectors
        Q, R = jnp.linalg.qr(C_flat.T)
        signs = jnp.sign(jnp.diag(R))
        signs = jnp.where(signs == 0, 1.0, signs)
        return ((Q * signs[None, :]).T).reshape(original_shape)

    # ── State analysis (post-processing) ──────────────────────────────────

    def compute_subsystem_character(self, C):
        """Compute subsystem excitation character for each state.

        Returns a matrix of shape (num_states, num_species) where entry [α, s]
        is the weighted-average orbital index for state α in species s:
        ⟨n⟩_s = Σ_n n·|D_{nn}| / Σ_n |D_{nn}|.

        Higher values indicate higher excitation levels in that subsystem.
        """
        Ds = self.dCI.get_1p_RDMs(C)  # List of (num_states, n_orb, n_orb)
        num_states = C.shape[0]
        character = jnp.zeros((num_states, self.num_species))

        for s in range(self.num_species):
            D_s = Ds[s]
            n_orb = D_s.shape[-1]
            orbital_weights = jnp.arange(n_orb, dtype=jnp.float32)

            # Vectorized over states
            diags = jnp.diagonal(D_s, axis1=-2, axis2=-1)  # (num_states, n_orb)
            weights = jnp.abs(diags)
            avg_orbital = jnp.sum(orbital_weights * weights, axis=-1) / jnp.sum(weights, axis=-1)
            character = character.at[:, s].set(avg_orbital)

        return character

    # ── State localization ────────────────────────────────────────────────

    def localize_states(self, y, energy_tol=1e-6, max_sweeps=20, tol=1e-8, verbose=False):
        """Localize degenerate states to Hartree products (post-processing).

        Within degenerate subspaces (energies within energy_tol), states are
        rotated via Jacobi sweeps to maximize subsystem purities Tr(ρ_s²),
        producing localized product-like states instead of entangled
        superpositions.

        Parameters
        ----------
        y : jnp.ndarray
            Packed state vector from ITE.
        energy_tol : float
            Energy tolerance for grouping degenerate states.
        max_sweeps : int
            Maximum Jacobi sweeps per degenerate subspace.
        tol : float
            Convergence tolerance for rotation angles (radians).
        verbose : bool
            Print progress information.

        Returns
        -------
        y_localized : jnp.ndarray
            Packed state vector with localized, energy-ordered states.
        character : jnp.ndarray
            Subsystem character matrix (num_states, num_species).
        groups : list of list of int
            Degenerate subspace groupings.
        """
        C, bs = self._y_to_Cb(y)
        num_states = C.shape[0]

        if num_states <= 1:
            character = self.compute_subsystem_character(C)
            return y, character, [[0]]

        # Step 1: Sort states by energy
        energies = jnp.real(self.calculate_energy(0.0, y))
        energy_order = jnp.argsort(energies)
        C = C[energy_order]
        energies = energies[energy_order]

        if verbose:
            print(f"  State energies: {[f'{e:.6f}' for e in energies]}")

        # Step 2: Group degenerate states
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

        # Step 3: Jacobi localization within each degenerate subspace
        for group in groups:
            if len(group) <= 1:
                continue

            group_indices = jnp.array(group)
            C_sub = C[group_indices]
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

            C = C.at[group_indices].set(C_sub)

            if verbose:
                final_purity = self._compute_total_purity(C_sub)
                print(f"    Purity: {initial_purity:.6f} → {final_purity:.6f}")

        y_localized = self._Cb_to_y(C, bs)
        character = self.compute_subsystem_character(C)
        return y_localized, character, groups

    def _compute_total_purity(self, C_sub):
        """Compute total subsystem purity Σ_α Σ_s Tr(ρ_s[α]²).

        A purity of 1 per (state, species) indicates a Hartree product;
        values < 1 indicate entanglement.
        """
        n_sub = C_sub.shape[0]
        total_purity = jnp.zeros(())

        for alpha in range(n_sub):
            for s in range(self.num_species):
                rho_s = self._subsystem_rdm(C_sub[alpha], s)
                total_purity = total_purity + jnp.real(jnp.trace(rho_s @ rho_s))

        return total_purity

    def _subsystem_rdm(self, C_single, s):
        """Compute reduced density matrix of species s for a single state.

        ρ_s = Tr_{not s}(|ψ⟩⟨ψ|) in the CI determinant basis.

        Parameters
        ----------
        C_single : jnp.ndarray
            CI tensor for a single state, shape (n_det_0, ..., n_det_{K-1}).
        s : int
            Species index.

        Returns
        -------
        rho : jnp.ndarray
            Reduced density matrix, shape (n_det_s, n_det_s).
        """
        C_moved = jnp.moveaxis(C_single, s, 0)
        n_det_s = C_moved.shape[0]
        M = C_moved.reshape(n_det_s, -1)
        return M @ M.conj().T

    def _compute_optimal_angle(self, C_sub, ii, jj):
        """Compute optimal Jacobi rotation angle for states ii and jj.

        Maximizes total purity Σ_s [Tr(ρ'_s[i]²) + Tr(ρ'_s[j]²)]
        using θ_opt = atan2(2E, D) / 4.
        """
        D_total = jnp.zeros(())
        E_total = jnp.zeros(())

        for s in range(self.num_species):
            rho_i = self._subsystem_rdm(C_sub[ii], s)
            rho_j = self._subsystem_rdm(C_sub[jj], s)

            # Transition density T = M_i @ M_j†
            C_i_moved = jnp.moveaxis(C_sub[ii], s, 0)
            C_j_moved = jnp.moveaxis(C_sub[jj], s, 0)
            n_det_s = C_i_moved.shape[0]
            M_i = C_i_moved.reshape(n_det_s, -1)
            M_j = C_j_moved.reshape(n_det_s, -1)
            T = M_i @ M_j.conj().T

            # Symmetrized transition: R = T + T†
            R = T + T.conj().T

            t_AA = jnp.real(jnp.trace(rho_i @ rho_i))
            t_BB = jnp.real(jnp.trace(rho_j @ rho_j))
            t_AB = jnp.real(jnp.trace(rho_i @ rho_j))
            t_RR = jnp.real(jnp.trace(R @ R))
            t_AR = jnp.real(jnp.trace(rho_i @ R))
            t_BR = jnp.real(jnp.trace(rho_j @ R))

            D_total = D_total + (t_AA + t_BB - t_RR - 2 * t_AB)
            E_total = E_total + (t_AR - t_BR)

        return jnp.arctan2(2 * E_total, D_total) / 4

    # ── Integration ───────────────────────────────────────────────────────

    def integrate(
        self,
        integrator,
        num_steps,
        t_init=0,
        bs_init_list=None,
        C_init=None,
        normalize=None,
        noise_C=1e-2,
        noise_bs=0.0,
    ):
        """Run the MCTDHF time propagation.

        Parameters
        ----------
        integrator : callable
            Integration step function (t, y) → (t_new, y_new).
        num_steps : int
            Number of time steps.
        t_init : float
            Initial time.
        bs_init_list : list of jnp.ndarray, optional
            Initial orbitals. If None, uses lowest eigenstates of h.
        C_init : jnp.ndarray, optional
            Initial CI coefficients. If None, uses energy-ordered product states.
        normalize : bool, optional
            Force normalization each step. Defaults to True for imaginary time.
        noise_C : float
            Magnitude of random perturbation added to C_init. Set to 0 for
            no perturbation. Applied to both default and user-provided C_init.
        noise_bs : float
            Magnitude of random perturbation added to each b in bs_init_list.
            Set to 0 for no perturbation. Applied to both default and
            user-provided bs_init_list.

        Returns
        -------
        ts : jnp.ndarray
            Time points (num_steps + 1,).
        ys : jnp.ndarray
            State vectors (num_steps + 1, dim_y).
        """
        # Validate shapes
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

        # Default initialization: lowest eigenstates of single-particle Hamiltonians
        evals_list = []
        if bs_init_list is None:
            bs_init_list = []
            for s, h in enumerate(self._get_h_current(t_init)):
                eigvals, eigvecs = np.linalg.eigh(np.array(h))
                evals_list.append(eigvals)
                b_init = eigvecs[:, :self.num_mctdhf_orbitals[s]]
                bs_init_list.append(jnp.array(b_init, dtype=jnp.complex64))
                print(f"Species {s}: lowest eigenvalues = {eigvals[:self.num_mctdhf_orbitals[s]]}")

        if C_init is None:
            # Product states ordered by non-interacting energy
            ranges = [range(n) for n in self.num_mctdhf_orbitals]
            product_states = []
            for indices in itertools.product(*ranges):
                E_nonint = sum(evals_list[s][n] for s, n in enumerate(indices))
                product_states.append((E_nonint, indices))
            product_states.sort(key=lambda x: x[0])

            C_init = np.zeros((self.num_states,) + tuple(self.num_mctdhf_orbitals))
            for state_idx in range(min(self.num_states, len(product_states))):
                _, indices = product_states[state_idx]
                C_init[(state_idx,) + indices] = 1.0
            C_init = jnp.array(C_init, dtype=jnp.complex64)

        # Add noise to initial guesses and re-orthonormalize
        rng = np.random.default_rng(42)
        if noise_C > 0:
            C_init = C_init + noise_C * jnp.array(rng.standard_normal(C_init.shape), dtype=jnp.complex64)
            C_init = self.ortonorm_C(C_init)
        if noise_bs > 0:
            bs_init_list = [
                self.ortonorm_b(b + noise_bs * jnp.array(rng.standard_normal(b.shape), dtype=jnp.complex64))
                for b in bs_init_list
            ]

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
                bs_new = [self.ortonorm_b(b) for b in bs]
                y = self._Cb_to_y(C, bs_new)
            return (t, y), (t, y)

        _, (ts_arr, ys_arr) = jax.lax.scan(step_fn, (t, y), None, length=num_steps)

        # Prepend initial values
        ts = jnp.concatenate([jnp.array([t_init]), ts_arr])
        ys = jnp.concatenate([y_init[None, :], ys_arr])
        return ts, ys


# ── DVR specialization ────────────────────────────────────────────────────

class MCTDHF_n_species_DVR(MCTDHF_n_species):
    """MCTDHF variant for DVR (Discrete Variable Representation) basis.

    In DVR, the 2-body integrals g are diagonal in spatial coordinates:
    g_{ij} = V(x_i, x_j) · dx_i · dx_j (2D array instead of 4D).
    """

    @partial(jit, static_argnums=0)
    def transform_g(self, g, b_sigma, b_tau):
        """Transform 2-body integrals for DVR basis (diagonal spatial integrals).

        Parameters
        ----------
        g : jnp.ndarray
            Spatial integrals g_{ij} (2D, diagonal in real space).
        b_sigma, b_tau : jnp.ndarray
            Orbital matrices for particle 1 and 2.

        Returns
        -------
        g_3, g_4, g_5 : jnp.ndarray
            Mean-field, fully transformed, and projected tensors.
        """
        # Transform tau: g_{ij} → g_{iqs}  (j and l share same spatial index in DVR)
        g_2 = contract("jq, js, ij -> iqs", b_tau.conj(), b_tau, g, backend="jax")
        # Transform sigma k → r: (i and k share same spatial index in DVR)
        g_3 = contract("ir, iqs -> iqrs", b_sigma, g_2, backend="jax")
        # Transform sigma i → p
        g_4 = contract("ip, iqrs -> pqrs", b_sigma.conj(), g_3, backend="jax")
        # Back-transform
        g_5 = contract("ip, pqrs -> iqrs", b_sigma, g_4, backend="jax")
        return g_3, g_4, g_5

    def _transpose_g(self, g):
        """Transpose 2D DVR integrals: (i, j) → (j, i)."""
        return jnp.transpose(g, (1, 0))
