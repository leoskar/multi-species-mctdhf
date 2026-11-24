"""Layer operators for multi-layer (ML) MCTDHF with species as nodes.

Extends :class:`~mctdhf.SemiDirectCI_n_species.SemiDirectCI_n_species` with
the quantities of the middle layer: a set of ``n_s`` many-body single-particle
functions (SPFs) per species, expanded in that species' Slater determinants,

    |φ^s_j⟩ = Σ_I B_s[I, j] |Φ^s_I(b_s)⟩,     B_s† B_s = 1,

and a top tensor over SPFs rather than determinants,

    |Ψ_α⟩ = Σ_j A[α, j_0 … j_{K-1}] Π_s |φ^s_{j_s}⟩.

Standard MCTDHF is the special case ``n_s = D_s``, ``B_s = 1``, ``A = C``;
every method here reduces to its determinant-space counterpart there, which is
the main correctness test (``tests/test_ml_layers.py``).

Index conventions — these must match the base class exactly:

* ``E^s_pq[I, I'] = ⟨I| a†_p a_q |I'⟩`` with the BRA determinant first. The
  same holds for every transition/density object below: the conjugated index
  comes first.
* ``g4^{sk}[p, q, r, t] = ⟨p q|V|r t⟩`` with ``p, r`` on species s and
  ``q, t`` on species k, so the pair operator is
  ``V_sk = Σ g4[p,q,r,t] E^s_pr E^k_qt``.
* Pair densities are in "xzyw" order: the two CONJUGATED indices (x on the
  first species, z on the second) come before the two unconjugated ones
  (y, w). This is the order :meth:`_compute_rho_pair` produces.
* ``_transform_all_g`` returns BOTH orderings ``(s,k)`` and ``(k,s)`` of every
  interacting pair, so the top-layer pair operators carry factor 0.5 each,
  exactly as ``precompute_ci_ops`` does. The mean field on species s, by
  contrast, carries factor 1 when built from ``g_dict[(s,k)]``: the two
  half-weighted orderings ``0.5 V_sk + 0.5 V_ks`` are the same operator and
  add up to one.

Every method has a dense branch for ``sparse=False``, mirroring the base
class. The ML operands are ``nnz × n²`` rather than ``nnz × D²``, so the
sparse path should be comfortable even on GPU, but the fallback is cheap
insurance.
"""

from functools import partial

import jax.numpy as jnp
from jax import jit
from opt_einsum import contract

from .SemiDirectCI_n_species import (
    SemiDirectCI_n_species,
    _inter_rdm_transpose,
    _segment_sum,
)


class SemiDirectCI_ML(SemiDirectCI_n_species):
    """Semi-direct CI with an SPF layer between the top tensor and the
    determinants.

    Constructed exactly like the base class; the SPF count ``n_s`` is not
    stored here but passed in through the ``B_s`` arrays, so one instance
    serves any truncation.
    """

    # ── Layer-2 transition densities ──────────────────────────────────────

    @partial(jit, static_argnums=(0, 1))
    def spf_transition_ops(self, s, B_s):
        """``e^s[p, q, j, j'] = ⟨φ^s_j| a†_p a_q |φ^s_{j'}⟩``.

        Parameters
        ----------
        s : int
            Species index (static).
        B_s : jnp.ndarray
            SPF coefficients, shape ``(D_s, n_s)``.

        Returns
        -------
        jnp.ndarray
            Shape ``(M_s, M_s, n_s, n_s)``; the conjugated SPF index ``j``
            comes first, matching ``E_pq``'s bra-first convention.

        Cost is ``nnz(E_pq)·n_s²`` in the sparse branch — the operand that
        replaces the ``nnz × D²`` one of standard MCTDHF.
        """
        m = self.num_spatial_orbitals[s]
        if not self.sparse:
            return contract("ijpq, ix, jy -> pqxy", self._dn_pq[s],
                            B_s.conj(), B_s, backend="jax")
        sp = self._sp_pq[s]
        bra = B_s.conj()[sp.bra_o]                   # (nnz, n)
        ket = B_s[sp.ket_o]                          # (nnz, n)
        contrib = sp.val_o[:, None, None] * bra[:, :, None] * ket[:, None, :]
        out = _segment_sum(contrib, sp.orb_o, m * m)  # (m*m, n, n)
        return out.reshape((m, m) + out.shape[1:])

    @partial(jit, static_argnums=(0, 1))
    def ci_density_from_spf(self, s, B_s, rho_s):
        """Lift a node density from SPF space to determinant space.

        ``ρ^{CI,s} = conj(B_s) ρ^s B_sᵀ``, i.e.
        ``ρ^{CI,s}[I, I'] = Σ_{jj'} conj(B_s[I,j]) ρ^s[j,j'] B_s[I',j']``,
        with the conjugated index first on both sides. Any leading batch
        (state) axes of ``rho_s`` are carried through.

        This is the object the determinant-space contractions
        (``contract_det``) consume, so the intra-species RDMs of the ML state
        are computed by exactly the base-class code path. Cost ``D_s² n_s``.
        """
        return contract("Ix, ...xy, Jy -> ...IJ", B_s.conj(), rho_s, B_s,
                        backend="jax")

    # ── Reduced density matrices of the ML state ──────────────────────────

    @partial(jit, static_argnums=0)
    def ml_1p_RDMs(self, A, Bs, weights=None):
        """1-particle RDMs ``D^s_{pq} = ⟨Ψ| E^s_{pq} |Ψ⟩`` of the ML state.

        Identical to ``get_1p_RDMs`` applied to the expanded CI tensor
        ``C = A ×_s B_s``, but never forms ``C``: the node density is built
        in SPF space and lifted to determinant space per species.

        With ``weights`` the state axis is contracted here, giving the
        state-averaged ``D^s`` with no batch axis; without it, one ``D^s``
        per state. ``Tr D^s = N_s`` for a normalised state.
        """
        Ds = []
        for s in range(self.num_species):
            rho_s = self._compute_rho_ci(A, s, weights)
            rho_ci = self.ci_density_from_spf(s, Bs[s], rho_s)
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
    def ml_2p_RDMs(self, A, Bs, weights=None, es=None):
        """2-particle RDMs of the ML state, both orderings per pair.

        ``d^{(s,s)}_{pqrt} = ⟨E^s_{pqrt}⟩`` (intra-species) and
        ``d^{(s,k)}_{pqrt} = ⟨E^s_{pr} E^k_{qt}⟩`` (inter-species).

        The inter-species block is contracted entirely in SPF space:

            d^{(s,k)}[p,q,r,t] = Σ_{xzyw} ρ^{sk}[x,z,y,w]
                                          e^s[p,r,x,y] e^k[q,t,z,w]

        The determinant-space pair density ``ρ^{CI,(s,k)}`` is never built —
        that object is ``D_s² D_k²``, the very wall the SPF layer removes.

        ``es`` may carry the precomputed ``e^s`` list (from
        :meth:`spf_transition_ops`) to avoid rebuilding it.
        """
        if es is None:
            es = [self.spf_transition_ops(s, Bs[s])
                  for s in range(self.num_species)]

        ds = {}

        # Intra-species: lift the node density and reuse the base-class path.
        for s in range(self.num_species):
            rho_s = self._compute_rho_ci(A, s, weights)
            rho_ci = self.ci_density_from_spf(s, Bs[s], rho_s)
            if self.sparse:
                m = self.num_spatial_orbitals[s]
                ds[(s, s)] = self._sp_pqrs[s].contract_det(rho_ci).reshape(
                    rho_ci.shape[:-2] + (m, m, m, m))
            else:
                ds[(s, s)] = contract(
                    "ijpqrs, ...ij -> ...pqrs", self._dn_pqrs[s], rho_ci,
                    backend="jax")

        # Inter-species: two einsums in SPF space, s1 < s2 only.
        for s1 in range(self.num_species):
            for s2 in range(s1 + 1, self.num_species):
                rho_pair = self._compute_rho_pair(A, s1, s2, weights)
                # T[..., x, y, q, t] = Σ_{zw} ρ[..., x, z, y, w] e2[q, t, z, w]
                T = contract("...xzyw, qtzw -> ...xyqt", rho_pair, es[s2],
                             backend="jax")
                # d[..., p, q, r, t] = Σ_{xy} T[..., x, y, q, t] e1[p, r, x, y]
                d_inter = contract("...xyqt, prxy -> ...pqrt", T, es[s1],
                                   backend="jax")
                ds[(s1, s2)] = d_inter
                ds[(s2, s1)] = _inter_rdm_transpose(d_inter)

        return ds

    # ── Layer operators ───────────────────────────────────────────────────

    @partial(jit, static_argnums=0)
    def ml_species_ops(self, h_list, g_dict):
        """``Op_s`` — the single-species determinant-space operators.

        Alias of :meth:`precompute_species_ops`; named for symmetry with the
        other ``ml_*`` entry points. ``Op_s[I, I'] = ⟨I| h_s + 0.5 g_ss |I'⟩``,
        bra index first.
        """
        return self.precompute_species_ops(h_list, g_dict)

    @partial(jit, static_argnums=0)
    def ml_top_ops(self, h_list, g_dict, Bs, es=None):
        """The top-layer Hamiltonian in the ``(ops_diag, ops_pair)`` format.

        Returns exactly what :meth:`precompute_ci_ops` returns, but in SPF
        space, so ``apply_ci_ops`` propagates ``A`` unchanged:

            ops_diag[s]      = h̃^s     = B_s† Op_s B_s        (n_s, n_s)
            ops_pair[(s,k)]  = Ṽ^{sk}  = factor · Σ g4[p,q,r,t]
                                          e^s[p,r,·,·] e^k[q,t,·,·]
                                                         (n_s, n_s, n_k, n_k)

        The counting factors follow :meth:`precompute_ci_ops` verbatim: 0.5
        for each ordering of an inter-species pair when both are present in
        ``g_dict`` (which ``_transform_all_g`` always supplies), 1.0
        otherwise; the intra-species 0.5 is already inside ``Op_s``.

        At ``B_s = 1`` this returns ``precompute_ci_ops(h_list, g_dict)``.
        """
        if es is None:
            es = [self.spf_transition_ops(s, Bs[s])
                  for s in range(self.num_species)]

        Ops = self.precompute_species_ops(h_list, g_dict)
        ops_diag = [
            contract("Ix, IJ, Jy -> xy", Bs[s].conj(), Ops[s], Bs[s],
                     backend="jax")
            for s in range(self.num_species)
        ]

        ops_pair = {}
        for (s1, s2), g_int in g_dict.items():
            if s1 == s2:
                continue
            factor = 0.5 if (s2, s1) in g_dict else 1.0
            ops_pair[(s1, s2)] = factor * contract(
                "pqrt, prab, qtcd -> abcd", g_int, es[s1], es[s2],
                backend="jax")

        return ops_diag, ops_pair

    @partial(jit, static_argnums=0)
    def ml_get_sigma(self, A, top_ops):
        """``σ = H_top |A⟩``. Thin alias of :meth:`apply_ci_ops`.

        ``apply_ci_ops``'s einsum strings depend only on ``num_species``, so
        it applies unchanged to the SPF-space tensor.
        """
        return self.apply_ci_ops(top_ops, A)

    # ── Mean field of the SPF (middle) layer ──────────────────────────────

    @partial(jit, static_argnums=(0, 4))
    def ml_meanfield_W(self, A, Bs, g_dict, s, es=None, weights=None,
                       rho_pairs=None):
        """Mean-field coefficients ``W^{sk}`` acting on species s.

        For every partner ``k ≠ s`` with ``(s, k)`` in ``g_dict``:

            η^{sk}[m, m', q, t] = Σ_{z,w} ρ^{sk}[m, z, m', w] e^k[q, t, z, w]
            W^{sk}[m, m', p, r] = Σ_{q,t} g4^{sk}[p, q, r, t] η^{sk}[m,m',q,t]

        so that ``⟨Ψ^{(s)}_m| V_sk |Ψ^{(s)}_{m'}⟩ = Σ_{pr} W^{sk}[m,m',p,r]
        E^s_{pr}`` as an operator on species s, with ``Ψ^{(s)}`` the
        single-hole functions of the top tensor.

        Factor 1, NOT 0.5: ``0.5 V_sk + 0.5 V_ks`` is one copy of the same
        operator (see the module docstring), matching how ``_rhs_b_linear``
        uses ``g_mix_tensors[(s, k)]`` in the orbital equation.

        ``es`` are the partner SPFs' transition densities, FROZEN at the
        substep start by the caller — the analogue of the frozen partner
        orbitals of ``_rhs_b_linear``. ``weights`` state-averages the node
        pair density.

        ``rho_pairs`` optionally carries the node pair densities
        ``{(s,k): ρ^{sk}}`` computed once for the whole CMF step (they are
        frozen there, so recomputing them per SPF substep only repeats work).
        When omitted they are rebuilt from ``A``.

        Returns ``{k: W}`` with ``W`` of shape ``(n_s, n_s, M_s, M_s)``.
        """
        if es is None:
            es = [self.spf_transition_ops(k, Bs[k])
                  for k in range(self.num_species)]

        W_dict = {}
        for k in range(self.num_species):
            if k == s or (s, k) not in g_dict:
                continue
            if rho_pairs is not None and (s, k) in rho_pairs:
                rho_pair = rho_pairs[(s, k)]
            else:
                rho_pair = self._compute_rho_pair(A, s, k, weights)
            eta = contract("mzow, qtzw -> moqt", rho_pair, es[k],
                           backend="jax")
            W_dict[k] = contract("pqrt, moqt -> mopr", g_dict[(s, k)], eta,
                                 backend="jax")
        return W_dict

    @partial(jit, static_argnums=(0, 1))
    def apply_meanfield_B(self, s, B_s, W_dict):
        """Apply the SPF mean field to ``B_s``: ``U_s``.

            U_s[I, m] = Σ_{k≠s} Σ_{I'pr m'} E^s_{I I', pr}
                                 W^{sk}[m, m', p, r] B_s[I', m']

        Sparse implementation: gather ``W[:, :, p, r]`` and ``B[I', :]`` at
        every nonzero of ``E_pq``, contract the ``m'`` index there, then
        scatter-add onto the BRA determinant ``I``. ``bra_d`` is
        non-decreasing because ``det_d`` is sorted, so the scatter lowers to
        a contiguous segmented reduction. Cost ``nnz(E_pq)·n_s²``.

        Returns ``(D_s, n_s)``; zeros when ``W_dict`` is empty (a species
        with no interacting partner).
        """
        D_s = self.num_dets[s]
        n_s = B_s.shape[1]

        if not W_dict:
            return jnp.zeros((D_s, n_s), dtype=B_s.dtype)

        if not self.sparse:
            E = self._dn_pq[s]
            U = None
            for W in W_dict.values():
                term = contract("IJpr, mopr, Jo -> Im", E, W, B_s,
                                backend="jax")
                U = term if U is None else U + term
            return U

        sp = self._sp_pq[s]
        p_idx, r_idx = sp.orb_split_d
        B_ket = B_s[sp.ket_d, :]                        # (nnz, n_s)
        U = None
        for W in W_dict.values():
            W_nz = jnp.moveaxis(W[:, :, p_idx, r_idx], -1, 0)  # (nnz, n, n)
            contrib = contract("kmn, kn -> km", W_nz, B_ket, backend="jax")
            contrib = contrib * sp.val_d[:, None]
            term = _segment_sum(contrib, sp.bra_d, D_s)        # (D_s, n_s)
            U = term if U is None else U + term
        return U
