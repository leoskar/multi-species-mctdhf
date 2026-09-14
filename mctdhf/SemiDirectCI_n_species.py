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
import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
from itertools import combinations


def _segment_sum(data, ids, num_segments):
    """``segment_sum`` that never scatters a complex operand.

    XLA's GPU scatter codegen handles complex types badly: excessive per-thread
    local memory at complex128, crashes at complex64. Summing real and
    imaginary parts as one real scatter is exact, since segment_sum is linear.
    """
    if not jnp.iscomplexobj(data):
        return jax.ops.segment_sum(
            data, ids, num_segments=num_segments, indices_are_sorted=True)
    stacked = jnp.stack([data.real, data.imag], axis=-1)
    out = jax.ops.segment_sum(
        stacked, ids, num_segments=num_segments, indices_are_sorted=True)
    return (out[..., 0] + 1j * out[..., 1]).astype(data.dtype)


class _SparseExc:
    """Nonzero entries of an excitation tensor E_{ij,o} in coordinate form.

    ``E_pq`` and ``E_pqrs`` are built by visiting each allowed excitation once
    and writing a single ±1, so their entries lie in {0, ±1} and their nonzero
    counts are exactly

        nnz(E_pq)   = C(m,N) · N      · (m−N+1)
        nnz(E_pqrs) = C(m,N) · N(N−1) · (m−N+2)(m−N+1)

    At (m, N) = (9, 3) that is 1 764 and 28 224 against dense sizes of 5.4e5
    and 4.6e7 — densities of 3e-3 and 6e-4. Storing the dense tensors therefore
    spends ~99.9 % of both memory and every contraction on multiplying by zero:
    measured, the dense route costs 741 MB of complex128 working set and makes
    a single CMF step 29× more expensive at m = 9 than at m = 8, against a 3.6×
    prediction from operation counts.

    Held here as three flat arrays — ``det`` (the flattened (i, j) determinant
    pair), ``orb`` (the flattened orbital multi-index), and ``val`` (the ±1) —
    in two sort orders, so that both consumers can pass
    ``indices_are_sorted=True`` to ``segment_sum`` and be lowered to a
    contiguous segmented reduction rather than a general atomic scatter:

      ``*_d``  sorted by ``det``  — for contractions that SCATTER onto (i, j)
                                    and gather on the orbital index
                                    (``precompute_ci_ops``)
      ``*_o``  sorted by ``orb``  — for contractions that SCATTER onto the
                                    orbital index and gather on (i, j)
                                    (the RDMs)

    ``val`` is float32, matching the dtype the dense tensors were stored in, so
    promotion against complex128 integrals is unchanged. float32 is exact here
    (entries are ±1) — asserted in the Part 2 dtype guard.
    """

    __slots__ = ("n_dets", "n_orb", "k", "nnz",
                 "det_d", "orb_d", "val_d", "orb_split_d",
                 "det_o", "orb_o", "val_o")

    def __init__(self, det, orb, val, n_dets, n_orb, k):
        self.n_dets, self.n_orb, self.k = int(n_dets), int(n_orb), int(k)
        self.nnz = int(det.size)
        od = np.argsort(det, kind="stable")
        oo = np.argsort(orb, kind="stable")
        self.det_d = jnp.asarray(det[od])
        self.orb_d = jnp.asarray(orb[od])
        self.val_d = jnp.asarray(val[od], dtype=jnp.float32)
        self.det_o = jnp.asarray(det[oo])
        self.orb_o = jnp.asarray(orb[oo])
        self.val_o = jnp.asarray(val[oo], dtype=jnp.float32)
        # orbital multi-index split back into its k components, det-sorted;
        # needed where a contraction keeps some orbital indices free
        # (the inter-species T1 build).
        rem, comps = orb[od], []
        for _ in range(k):
            comps.append(rem % n_orb)
            rem = rem // n_orb
        self.orb_split_d = [jnp.asarray(c) for c in reversed(comps)]

    @property
    def n_seg_det(self):
        return self.n_dets * self.n_dets

    @property
    def n_seg_orb(self):
        return self.n_orb ** self.k

    def to_dense(self, dtype=jnp.float32):
        """Materialise the dense tensor. For tests and diagnostics only —
        this is exactly the object the sparse form exists to avoid."""
        shape = (self.n_dets,) * 2 + (self.n_orb,) * self.k
        flat = np.zeros(self.n_dets * self.n_dets * self.n_orb ** self.k)
        np.add.at(flat,
                  np.asarray(self.det_d) * (self.n_orb ** self.k)
                  + np.asarray(self.orb_d),
                  np.asarray(self.val_d, dtype=np.float64))
        return jnp.asarray(flat.reshape(shape), dtype=dtype)

    # ── the two contraction patterns ──────────────────────────────────────

    def contract_orb(self, t):
        """Σ_o E_{ij,o} t_o  →  (n_dets, n_dets).  Gathers t, scatters on (i,j)."""
        contrib = self.val_d * t.reshape(-1)[self.orb_d]
        return _segment_sum(
            contrib, self.det_d, self.n_seg_det).reshape(
                self.n_dets, self.n_dets)

    def contract_det(self, rho):
        """Σ_ij E_{ij,o} rho_{...ij}  →  (..., n_orb^k).  Gathers rho, scatters on o.

        The batch axes of ``rho`` are carried through; segment_sum reduces axis
        0, so the nnz axis is moved to the front and back again."""
        rf = rho.reshape(rho.shape[:-2] + (self.n_dets * self.n_dets,))
        contrib = rf[..., self.det_o] * self.val_o
        moved = jnp.moveaxis(contrib, -1, 0)
        out = _segment_sum(moved, self.orb_o, self.n_seg_orb)
        return jnp.moveaxis(out, 0, -1)


class _DenseExcView:
    """Backwards-compatible ``dci.E_pq[s]`` / ``dci.E_pqrs[s]`` accessor.

    Materialises one species' dense tensor on demand and does NOT cache it —
    caching would reinstate exactly the memory the sparse storage removes.
    Present for the external consumers that legitimately want the dense object
    (the C1 operator-identity test, dtype guards); nothing on the hot path uses
    it."""

    __slots__ = ("_sp",)

    def __init__(self, sparse_list):
        self._sp = sparse_list

    def __getitem__(self, s):
        return self._sp[s].to_dense()

    def __len__(self):
        return len(self._sp)

    def __iter__(self):
        return (sp.to_dense() for sp in self._sp)


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
        sparse=True,
    ):
        """
        sparse : bool
            Store the excitation tensors in coordinate form and contract them
            with ``segment_sum`` (default), rather than as dense arrays
            contracted with einsum.

            Sparse is faster wherever the CI space is not tiny — measured
            5.9x at (m, N) = (8, 3) and 39.5x at (9, 3), and it removes the
            dense tensor's memory wall entirely (1.15 GB at (10, 3), 16 GB at
            (12, 3)) along with the 29x per-step cliff that wall caused.

            It is marginally SLOWER at very small CI dimension, where the fixed
            gather/scatter cost does not shrink with n_dets: measured 1.13x
            (+132 us/step) at CI dim 6, 1.07x at CI dim 16, and ~1.00x at
            CI dim 256-625. ``sparse=False`` selects the original dense path,
            unchanged, for reproducing pre-2026-08-31 results bit-for-bit or if
            a small configuration is ever step-rate critical.
        """
        self.sparse = bool(sparse)
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
        # Excitation operators, stored sparsely (see _SparseExc). The dense
        # tensors are never allocated: at (m, N) = (10, 3) the dense float64
        # E_pqrs is 1.15 GB holding 51 840 nonzeros.
        self._sp_pq = []    # 1-body excitation entries, per species
        self._sp_pqrs = []  # 2-body excitation entries, per species

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
            self._sp_pq.append(
                self._build_Epq_sparse(n_orb, self.num_dets[s], I_s, W_s))
            self._sp_pqrs.append(
                self._build_Epqrs_sparse(n_orb, self.num_dets[s], I_s, W_s))

        if not self.sparse:
            # Dense path: materialise the same tensors the pre-sparse code
            # built (to_dense is pinned against it elementwise) and use the
            # original einsum contractions below.
            self._dn_pq = [sp.to_dense() for sp in self._sp_pq]
            self._dn_pqrs = [sp.to_dense() for sp in self._sp_pqrs]

    # ── Backwards-compatible dense views ──────────────────────────────────
    #
    # `dci.E_pq[s]` / `dci.E_pqrs[s]` still return the dense float32 tensors
    # they always did, materialised on demand. Nothing on the hot path uses
    # them; they exist for the external consumers that want the dense object
    # (the C1 operator-identity test, dtype guards, diagnostics).

    @property
    def E_pq(self):
        return _DenseExcView(self._sp_pq)

    @property
    def E_pqrs(self):
        return _DenseExcView(self._sp_pqrs)

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
        """Map an occupation vector to its integer address via vertex weights.

        The running occupation count replaces the per-orbital ``np.sum(n[:m+1])``
        of the original, which made a single address evaluation O(m^2) with a
        fresh temporary per orbital. Identical result, O(m) and allocation-free.
        """
        res = 0
        cnt = 0
        for m in range(num_spatial_orbitals):
            if n[m]:
                cnt += 1
                res += W[m, cnt]
        return int(res)

    # ── Second-quantized excitation evaluation ────────────────────────────

    def single_exc(self, p, q, n):
        """Evaluate single excitation a†_p a_q on determinant n.

        Returns (phase, new_det) or (0, 0) if the excitation is forbidden.

        The forbidden case is tested before the copy, so the array is allocated
        only when the excitation actually survives. The bitmask twin used by the
        tensor builders is :meth:`_single_exc_mask`; the two are pinned against
        each other in the C1 unit checks.
        """
        if p != q and n[p]:
            return 0, 0
        n_pq = np.copy(n)
        n_pq[q] = 0
        n_pq[p] = 1
        gamma = (-1) ** (np.sum(n[:q]) + np.sum(n_pq[:p]))
        return gamma, n_pq

    # ── Bitmask twins, used by the tensor builders ────────────────────────
    #
    # A determinant is an int whose bit k is set iff orbital k is occupied.
    # `(mask & ((1 << k) - 1)).bit_count()` replaces `np.sum(n[:k])`: the
    # occupation-parity counts that set the fermionic phase become single
    # machine instructions instead of a slice, a temporary and a reduction.
    # Semantics are a line-by-line transcription of the array versions above,
    # including which intermediate each parity is counted on.

    @staticmethod
    def _single_exc_mask(p, q, m):
        """a†_p a_q on bitmask m -> (phase, new_mask); phase 0 if forbidden."""
        below_q = (m & ((1 << q) - 1)).bit_count()
        m1 = m & ~(1 << q)
        if (m1 >> p) & 1:
            return 0, 0
        below_p = (m1 & ((1 << p) - 1)).bit_count()
        return (1 if (below_q + below_p) % 2 == 0 else -1), m1 | (1 << p)

    @staticmethod
    def _double_exc_mask(p, q, s, r, m):
        """a†_p a†_q a_s a_r on bitmask m -> (phase, new_mask)."""
        if s == r or p == q:
            return 0, 0
        gamma = (m & ((1 << r) - 1)).bit_count()
        m1 = m & ~(1 << r)
        gamma += (m1 & ((1 << s) - 1)).bit_count()
        m2 = m1 & ~(1 << s)
        if ((m2 >> q) & 1) or ((m2 >> p) & 1):
            return 0, 0
        gamma += (m2 & ((1 << q) - 1)).bit_count()
        m3 = m2 | (1 << q)
        gamma += (m3 & ((1 << p) - 1)).bit_count()
        return (1 if gamma % 2 == 0 else -1), m3 | (1 << p)

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

    # ── Sparse excitation-tensor construction (numpy, done once at init) ──

    def _det_masks(self, spin_strings, n_orb, W):
        """(list of bitmasks in address order, mask -> address dict).

        The address function is evaluated once per determinant here rather than
        once per excitation inside the builders below, turning an O(m) lookup in
        the innermost loop into a dict hit."""
        masks = [0] * len(spin_strings)
        addr = {}
        for J in spin_strings:
            m = int(np.dot(np.asarray(J, dtype=np.int64),
                           1 << np.arange(n_orb, dtype=np.int64)))
            j = self.address(J, n_orb, W)
            masks[j] = m
            addr[m] = j
        return masks, addr

    def _build_Epq_sparse(self, n_orb, num_dets, spin_strings, W):
        """Nonzero entries of E_{IJ,pq}, in coordinate form."""
        masks, addr = self._det_masks(spin_strings, n_orb, W)
        det, orb, val = [], [], []
        for j, mJ in enumerate(masks):
            occ = [k for k in range(n_orb) if (mJ >> k) & 1]
            for p in range(n_orb):
                for q in occ:
                    xi, mI = self._single_exc_mask(p, q, mJ)
                    if xi:
                        det.append(addr[mI] * num_dets + j)
                        orb.append(p * n_orb + q)
                        val.append(xi)
        return _SparseExc(np.asarray(det, dtype=np.int32),
                          np.asarray(orb, dtype=np.int32),
                          np.asarray(val, dtype=np.float64),
                          num_dets, n_orb, 2)

    def _build_Epqrs_sparse(self, n_orb, num_dets, spin_strings, W):
        """Nonzero entries of E_{IJ,pqrs}, in coordinate form."""
        masks, addr = self._det_masks(spin_strings, n_orb, W)
        det, orb, val = [], [], []
        n2, n3 = n_orb * n_orb, n_orb ** 3
        for j, mJ in enumerate(masks):
            occ = [k for k in range(n_orb) if (mJ >> k) & 1]
            for p in range(n_orb):
                for r in occ:
                    for q in range(n_orb):
                        for s in occ:
                            xi, mI = self._double_exc_mask(p, q, s, r, mJ)
                            if xi:
                                det.append(addr[mI] * num_dets + j)
                                orb.append(p * n3 + q * n2 + r * n_orb + s)
                                val.append(xi)
        return _SparseExc(np.asarray(det, dtype=np.int32),
                          np.asarray(orb, dtype=np.int32),
                          np.asarray(val, dtype=np.float64),
                          num_dets, n_orb, 4)

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
            # Sparse: gather the integral at each nonzero excitation and
            # scatter-add onto (i, j). Equivalent to
            #   contract("ijpq, pq -> ij", E_pq[s], h_list[s])
            # over ~1e-3 of the terms; validated bit-identical.
            if self.sparse:
                Op_s = self._sp_pq[s].contract_orb(h_list[s])
            else:
                Op_s = contract("ijpq, pq -> ij", self._dn_pq[s], h_list[s],
                                backend="jax")
            if (s, s) in g_dict:
                # Intra-species: factor 0.5 (double counting in sum)
                if self.sparse:
                    Op_s = Op_s + 0.5 * self._sp_pqrs[s].contract_orb(
                        g_dict[(s, s)])
                else:
                    Op_s = Op_s + 0.5 * contract(
                        "ijpqrs, pqrs -> ij", self._dn_pqrs[s],
                        g_dict[(s, s)], backend="jax")
            ops_diag.append(Op_s)

        ops_pair = {}
        for (s1, s2), g_int in g_dict.items():
            if s1 == s2:
                continue
            # If both (s1,s2) and (s2,s1) present → 0.5 each; else → 1.0
            factor = 0.5 if (s2, s1) in g_dict else 1.0
            # Sparse both halves. E_pq[s1]'s two orbital indices play the
            # roles (p, r) here, so the g slice g[p, :, r, :] is gathered per
            # nonzero and scatter-added onto (i, j) with (q, s) left free:
            #   T1 = contract("ijpr, pqrs -> ijqs", E_pq[s1], g_int)
            if not self.sparse:
                T1 = contract("ijpr, pqrs -> ijqs", self._dn_pq[s1], g_int,
                              backend="jax")
                ops_pair[(s1, s2)] = factor * contract(
                    "ijqs, klqs -> ijkl", T1, self._dn_pq[s2], backend="jax")
                continue
            sp1, sp2 = self._sp_pq[s1], self._sp_pq[s2]
            p_idx, r_idx = sp1.orb_split_d
            n1, n2 = sp1.n_dets, sp2.n_dets
            m2 = sp2.n_orb
            g_sl = g_int[p_idx, :, r_idx, :]                     # (nnz1, m2, m2)
            T1 = _segment_sum(sp1.val_d[:, None, None] * g_sl, sp1.det_d,
                              n1 * n1)                           # (n1^2, m2, m2)
            #   Op_12 = contract("ijqs, klqs -> ijkl", T1, E_pq[s2])
            cols = T1.reshape(n1 * n1, m2 * m2)[:, sp2.orb_d]    # (n1^2, nnz2)
            Op_12 = _segment_sum((cols * sp2.val_d[None, :]).T, sp2.det_d,
                                 n2 * n2)                        # (n2^2, n1^2)
            Op_12 = Op_12.T.reshape(n1, n1, n2, n2)
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

    def _compute_rho_ci(self, C, s, weights=None):
        """Compute the reduced CI density matrix for species s.

        Contracts C* and C over all species indices except s, yielding
        ρ^CI_{xy} = Σ_{other} C*_{...x...} C_{...y...}.

        Returns shape (..., n_det_s, n_det_s) where ... is the batch dimension.
        With `weights`, the batch (state) axis is contracted against them here
        instead, giving the state-averaged ρ directly with no batch axis.
        """
        c_indices = [chr(97 + i) for i in range(self.num_species)]
        c_conj_indices = list(c_indices)
        c_conj_indices[s] = "x"  # conjugate index
        c_indices[s] = "y"

        if weights is not None:
            rho_ci_str = (f"S,S{''.join(c_conj_indices)},"
                          f"S{''.join(c_indices)}->xy")
            return contract(rho_ci_str, weights, C.conj(), C, backend="jax")
        rho_ci_str = f"...{''.join(c_conj_indices)},...{''.join(c_indices)}->...xy"
        return contract(rho_ci_str, C.conj(), C, backend="jax")

    @partial(jit, static_argnums=0)
    def get_1p_RDMs(self, C, weights=None):
        """Compute 1-particle reduced density matrices for all species.

        Returns
        -------
        list of jnp.ndarray
            D_s of shape (num_states, n_orb_s, n_orb_s) for each species s.
            D^(s)_{pq} = ⟨Ψ| E^(s)_{pq} |Ψ⟩. With `weights`, the
            state-averaged D_s of shape (n_orb_s, n_orb_s) instead.
        """
        Ds = []
        for s in range(self.num_species):
            rho_ci = self._compute_rho_ci(C, s, weights)
            if self.sparse:
                m = self.num_spatial_orbitals[s]
                D_s = self._sp_pq[s].contract_det(rho_ci).reshape(
                    rho_ci.shape[:-2] + (m, m))
            else:
                D_s = contract("ijpq, ...ij -> ...pq", self._dn_pq[s], rho_ci,
                               backend="jax")
            Ds.append(D_s)
        return Ds

    @partial(jit, static_argnums=0)
    def get_2p_RDMs(self, C, weights=None):
        """Compute 2-particle reduced density matrices for all species pairs.

        Returns
        -------
        dict of {(s1, s2): jnp.ndarray}
            d^(s1,s2)_{pqrs} = ⟨Ψ| E^(s1)_{pqrs} |Ψ⟩  (intra-species)
            d^(s1,s2)_{pqrs} = ⟨Ψ| E^(s1)_{pr} E^(s2)_{qs} |Ψ⟩  (inter-species)
            With `weights`, the state-averaged RDMs, with no batch axis.

        Averaging over states before the excitation contraction is exact (the
        contraction is linear in ρ) and keeps the batch axis out of it.
        """
        ds = {}

        # Intra-species: reuse shared rho_ci helper
        for s1 in range(self.num_species):
            rho_ci = self._compute_rho_ci(C, s1, weights)
            if self.sparse:
                m = self.num_spatial_orbitals[s1]
                ds[(s1, s1)] = self._sp_pqrs[s1].contract_det(rho_ci).reshape(
                    rho_ci.shape[:-2] + (m, m, m, m))
            else:
                ds[(s1, s1)] = contract(
                    "ijpqrs, ...ij -> ...pqrs", self._dn_pqrs[s1], rho_ci,
                    backend="jax")

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

                if weights is not None:
                    rho_pair_str = (f"S,S{''.join(idx_conj)},"
                                    f"S{''.join(idx_norm)}->xzyw")
                    rho_pair = contract(rho_pair_str, weights, C.conj(), C,
                                        backend="jax")
                else:
                    rho_pair_str = (
                        f"...{''.join(idx_conj)},...{''.join(idx_norm)}->...xzyw"
                    )
                    rho_pair = contract(rho_pair_str, C.conj(), C,
                                        backend="jax")

                # Sparse, in two segment_sums:
                #   B[..., pr, zw] = sum_xy rho[..., xy, zw] E1[xy, pr]
                #   d[..., pr, qs] = sum_zw B[..., pr, zw] E2[zw, qs]
                # then (p, r, q, s) -> (p, q, r, s).
                if not self.sparse:
                    d_inter = contract(
                        "...xzyw, xypr, zwqs -> ...pqrs", rho_pair,
                        self._dn_pq[s1], self._dn_pq[s2], backend="jax")
                    ds[(s1, s2)] = d_inter
                    ds[(s2, s1)] = jnp.einsum("...pqrs->...qpsr", d_inter)
                    continue
                sp1, sp2 = self._sp_pq[s1], self._sp_pq[s2]
                n1, n2 = sp1.n_dets, sp2.n_dets
                m1, m2 = sp1.n_orb, sp2.n_orb
                rp = jnp.einsum("...xzyw->...xyzw", rho_pair).reshape(
                    rho_pair.shape[:-4] + (n1 * n1, n2 * n2))
                B = _segment_sum(
                    jnp.moveaxis(rp[..., sp1.det_o, :]
                                 * sp1.val_o[:, None], -2, 0),
                    sp1.orb_o, m1 * m1)                  # (m1^2, ..., n2^2)
                d_pr_qs = _segment_sum(
                    jnp.moveaxis(B[..., sp2.det_o] * sp2.val_o, -1, 0),
                    sp2.orb_o, m2 * m2)                  # (m2^2, m1^2, ...)
                d_pr_qs = jnp.moveaxis(jnp.moveaxis(d_pr_qs, 0, -1), 0, -2)
                d_inter = jnp.einsum(
                    "...prqs->...pqrs",
                    d_pr_qs.reshape(d_pr_qs.shape[:-2] + (m1, m1, m2, m2)))
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
