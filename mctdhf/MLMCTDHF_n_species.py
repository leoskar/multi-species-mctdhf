"""Multi-layer MCTDHF (ML-MCTDHF) for N distinguishable species.

Three layers, one node per species (Wang & Thoss 2003; Manthe 2008; Cao et
al. 2013 for mixtures; RMP 92, 011001 §V.D):

    layer 1   A   : (num_states, n_0, …, n_{K-1})   top tensor over SPFs
    layer 2   B_s : (D_s, n_s)                      many-body SPFs in dets
    layer 3   b_s : (N_spatial_s, M_s)              time-dependent orbitals

with

    |φ^s_j⟩ = Σ_I B_s[I, j] |Φ^s_I(b_s)⟩,
    |Ψ_α⟩   = Σ_j A[α, j_0 … j_{K-1}] Π_s |φ^s_{j_s}⟩.

The layer-2 truncation ``n_s ≤ D_s`` is what removes the CI wall: the top
tensor is ``num_states · Π n_s`` instead of ``num_states · Π D_s``, and the
inter-species pair operator is ``n_s² n_k²`` instead of ``D_s² D_k²``.
Standard MCTDHF is recovered EXACTLY at ``n_s = D_s``, ``B_s = 1``, ``A = C``
— there ``(1 − P_f) ≡ 0`` in the B equation, so ``Ḃ_s = 0`` and the solve is
skipped outright.

Gauges (zero constraint operator on all three layers): ``b_s† ḃ_s = 0`` and
``B_s† Ḃ_s = 0``. Because the orbital gauge makes ``⟨Φ_I|Φ̇_J⟩ = 0``, the SPF
gauge reduces to the latter alone (Wang–Thoss Eq. 2.30b).

Packed state::

    y = [A.flat, B_0.flat, …, B_{K-1}.flat, b_0.flat, …, b_{K-1}.flat]

``_y_to_Cb`` deliberately RAISES on these classes: an ML state has no CI
tensor, and silently unpacking it with the standard layout would produce
garbage. Use :meth:`MLMCTDHF_n_species.expand_C` or
:meth:`MLMCTDHF_n_species.to_mctdhf_y` instead.

Integrators live in :mod:`mctdhf.integrators_ml` (basic and midpoint CMF
only). The coupled RHS (``__call__``), the IMEX split and the adaptive
integrator are NOT implemented for ML and raise through ``_y_to_Cb``.
"""

import itertools
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from opt_einsum import contract
from scipy.special import comb

from . import postprocessing
from .MCTDHF_n_species import (
    MCTDHF_n_species,
    MCTDHF_n_species_DVR,
    _regularized_inverse,
    pack_Cb_to_y,
)
from .SemiDirectCI_ml import SemiDirectCI_ML


# ── Packing ───────────────────────────────────────────────────────────────


def pack_ABb_to_y(A, Bs, bs_list):
    """Pack ``(A, Bs, bs)`` into the flat ML state vector."""
    flats = [A.reshape(-1)]
    flats += [B.reshape(-1) for B in Bs]
    flats += [b.reshape(-1) for b in bs_list]
    return jnp.concatenate(flats)


def unpack_y_to_ABb(
    y,
    num_states,
    num_particles,
    num_spatial_orbitals,
    num_mctdhf_orbitals,
    num_spfs,
):
    """Unpack the flat ML state vector into ``(A, Bs, bs)``.

    ``D_s`` is derived from ``num_particles`` and ``num_mctdhf_orbitals`` as
    the binomial coefficient per species, matching
    :class:`~mctdhf.SemiDirectCI_n_species.SemiDirectCI_n_species`.
    """
    num_species = len(num_spatial_orbitals)
    num_dets = tuple(
        comb(num_mctdhf_orbitals[s], num_particles[s], exact=True)
        for s in range(num_species)
    )

    a_size = int(np.prod(num_spfs)) * num_states
    A = y[:a_size].reshape((num_states,) + tuple(num_spfs))

    start = a_size
    Bs = []
    for s in range(num_species):
        shape = (num_dets[s], num_spfs[s])
        size = shape[0] * shape[1]
        Bs.append(y[start:start + size].reshape(shape))
        start += size

    bs_list = []
    for s in range(num_species):
        shape = (num_spatial_orbitals[s], num_mctdhf_orbitals[s])
        size = shape[0] * shape[1]
        bs_list.append(y[start:start + size].reshape(shape))
        start += size

    return A, Bs, bs_list


# ── The ML mixin ──────────────────────────────────────────────────────────


class _MLMixin:
    """Everything ML-specific, independent of the base class's DVR flavour.

    Mixed in front of :class:`~mctdhf.MCTDHF_n_species.MCTDHF_n_species` and
    its DVR subclass. All the layer-3 machinery — ``transform_h``,
    ``transform_g``, ``_transform_all_g``, ``_precompute_frozen_g2``,
    ``_rhs_b_linear``, ``_regularized_inverse``, the DVR overrides — is
    inherited and used verbatim: the orbital variation at fixed ``(A, B)``
    sees Ψ only through its 1- and 2-RDMs, and those are supplied by
    :meth:`_compute_full_RDMs`.
    """

    _ci_class = SemiDirectCI_ML

    def __init__(self, *args, num_spfs=None, eps_spf=None, **kwargs):
        if num_spfs is None:
            raise TypeError(
                "num_spfs is required: give the number of many-body SPFs per "
                "species, 1 <= n_s <= D_s. Pass num_spfs=num_dets for the "
                "standard-MCTDHF limit."
            )
        super().__init__(*args, **kwargs)

        if len(num_spfs) != self.num_species:
            raise ValueError(
                f"num_spfs must have {self.num_species} entries, "
                f"got {len(num_spfs)}"
            )
        for s, n in enumerate(num_spfs):
            if int(n) != n or not (1 <= n <= self.num_dets[s]):
                raise ValueError(
                    f"num_spfs[{s}] = {n} is out of range: need an integer "
                    f"with 1 <= n_s <= D_s = {self.num_dets[s]}"
                )
        self.num_spfs = tuple(int(n) for n in num_spfs)

        # Regularisation of the SPF node density. Defaults to the orbital eps.
        self.eps_spf = self.eps if eps_spf is None else float(eps_spf)

        self.a_shape = (self.num_states,) + self.num_spfs
        self.B_shapes = [
            (self.num_dets[s], self.num_spfs[s])
            for s in range(self.num_species)
        ]
        self.b_shapes = [
            (self.num_spatial_orbitals[s], self.num_mctdhf_orbitals[s])
            for s in range(self.num_species)
        ]
        # Species whose SPF basis is complete: (1 - P_f) = 0, so B is exactly
        # stationary and the solve is skipped.
        self.spf_is_full = tuple(
            self.num_spfs[s] == self.num_dets[s]
            for s in range(self.num_species)
        )

    # ── State vector packing / unpacking ──────────────────────────────────

    def _ABb_to_y(self, A, Bs, bs_list):
        return pack_ABb_to_y(A, Bs, bs_list)

    def _y_to_ABb(self, y):
        return unpack_y_to_ABb(
            y,
            num_states=self.num_states,
            num_particles=self.num_particles,
            num_spatial_orbitals=self.num_spatial_orbitals,
            num_mctdhf_orbitals=self.num_mctdhf_orbitals,
            num_spfs=self.num_spfs,
        )

    def _y_to_Cb(self, y):
        raise TypeError(
            "An ML-MCTDHF state has no CI tensor: y = [A, B_0…, b_0…], not "
            "[C, b_0…]. Use _y_to_ABb(y), or expand_C(A, Bs) / "
            "to_mctdhf_y(y) to get the standard-MCTDHF state. Methods that "
            "need the coupled RHS (__call__, _rhs_implicit/_rhs_explicit, "
            "integrate_adaptive, diffrax) are not implemented for ML."
        )

    # ── Conversions to and from standard MCTDHF ───────────────────────────

    @partial(jit, static_argnums=0)
    def expand_C(self, A, Bs):
        """``C[α, I_0…] = Σ_j A[α, j_0…] Π_s B_s[I_s, j_s]``.

        The ML state written in the standard determinant basis. Only for
        tests, overlaps with standard-MCTDHF states and post-processing
        through the old code paths — it is exactly the object the SPF layer
        exists to avoid building on the hot path.
        """
        C = A
        for s, B in enumerate(Bs):
            C = jnp.tensordot(C, B, axes=([s + 1], [1]))
            C = jnp.moveaxis(C, -1, s + 1)
        return C

    def to_mctdhf_y(self, y_ml):
        """Convert an ML state to a standard-MCTDHF packed state.

        The result is the packed state of a companion
        :class:`~mctdhf.MCTDHF_n_species.MCTDHF_n_species` (or its DVR
        subclass) with the same ``num_particles``, ``num_mctdhf_orbitals``,
        ``num_spatial_orbitals`` and ``num_states``.
        """
        A, Bs, bs = self._y_to_ABb(y_ml)
        return pack_Cb_to_y(self.expand_C(A, Bs), bs)

    def from_mctdhf(self, C, bs, num_spfs=None, weights=None):
        """Truncated HOSVD of a standard-MCTDHF state onto the SPF layer.

        ``B_s`` is the set of ``n_s`` dominant natural SPFs of species s: the
        top-``n_s`` eigenvectors of ``conj(ρ^{CI,s})``, where ``ρ^{CI,s}`` is
        the state-averaged reduced CI density
        ``ρ[x, y] = Σ_α w_α Σ_other C*[…x…] C[…y…]`` (conjugated index
        first). The conjugation matters: the weight kept by a column ``v`` is
        ``v† conj(ρ) v``, so it is ``conj(ρ)``, not ``ρ``, that must be
        diagonalised.

        ``A = C ×_s B_s†`` is then applied sequentially, one species at a
        time, so the full ``C`` is contracted away species by species.

        Parameters
        ----------
        C : jnp.ndarray
            Standard CI tensor, ``(num_states, D_0, …)``.
        bs : list of jnp.ndarray
            Orbitals; passed through unchanged (layer 3 is untouched).
        num_spfs : sequence of int, optional
            Must equal this solver's ``num_spfs``; present so the call reads
            explicitly. Defaults to ``self.num_spfs``.
        weights : array, optional
            State-averaging weights. Defaults to ``self.state_weights``.

        Returns
        -------
        A, Bs, bs, discarded : tuple
            ``discarded[s]`` is the relative weight thrown away on species s,
            ``1 − (Σ kept eigenvalues) / (Σ all eigenvalues)``. For a
            normalised, state-averaged ``C`` the denominator is 1, so this is
            ``1 − Σ kept eigenvalues``.
        """
        if num_spfs is None:
            num_spfs = self.num_spfs
        if tuple(int(n) for n in num_spfs) != self.num_spfs:
            raise ValueError(
                f"num_spfs {tuple(num_spfs)} does not match this solver's "
                f"{self.num_spfs}; A would not fit a_shape."
            )
        if weights is None:
            weights = self.state_weights

        A = C
        Bs = []
        discarded = []
        for s in range(self.num_species):
            rho = self.dCI._compute_rho_ci(C, s, weights)
            # Kept weight of a column v is v† conj(rho) v (see docstring).
            evals, evecs = jnp.linalg.eigh(jnp.conj(rho))
            n_s = self.num_spfs[s]
            # eigh is ascending; take the n_s largest.
            B_s = evecs[:, ::-1][:, :n_s]
            Bs.append(B_s)
            lam = jnp.real(evals)[::-1]
            total = jnp.sum(lam)
            kept = jnp.sum(lam[:n_s])
            safe_total = jnp.where(total > 0, total, 1.0)
            discarded.append(1.0 - kept / safe_total)
            A = jnp.tensordot(A, B_s.conj(), axes=([s + 1], [0]))
            A = jnp.moveaxis(A, -1, s + 1)

        return A, Bs, list(bs), discarded

    def from_species_vectors(self, vecs_per_species, A):
        """Build the SPF layer from given determinant-space vectors.

        ``vecs_per_species[s]`` is a ``(D_s, n_s)`` matrix whose columns span
        the species-s SPF space — the register's "logical basis" of frozen
        per-well states, for instance. The columns need not be orthonormal:
        they are Löwdin-orthonormalised, ``B_s = V_s S_s^{-1/2}`` with
        ``S_s = V_s† V_s``, which is the orthonormalisation closest to the
        input in the least-squares sense and does not privilege any column.

        ``A`` is interpreted in the GIVEN (non-orthonormal) basis and is
        transformed along, ``A ← A ×_s S_s^{1/2}``, so the physical state is
        unchanged. With already-orthonormal input, ``S^{1/2} = 1`` and ``A``
        passes through.

        Returns ``(A, Bs)``.
        """
        Bs = []
        A_new = A
        for s, V in enumerate(vecs_per_species):
            V = jnp.asarray(V)
            if V.shape != self.B_shapes[s]:
                raise ValueError(
                    f"vecs_per_species[{s}] has shape {V.shape}, expected "
                    f"{self.B_shapes[s]}"
                )
            S = V.conj().T @ V
            lam, U = jnp.linalg.eigh(S)
            lam = jnp.maximum(jnp.real(lam), 0.0)
            safe_lam = jnp.where(lam > 0, lam, 1.0)
            sqrt_S = (U * jnp.sqrt(lam)[None, :]) @ U.conj().T
            inv_sqrt = jnp.where(lam > 0, 1.0 / jnp.sqrt(safe_lam), 0.0)
            inv_sqrt_S = (U * inv_sqrt[None, :]) @ U.conj().T
            Bs.append(V @ inv_sqrt_S.astype(V.dtype))
            # A[…m…] ← Σ_j S^{1/2}[m, j] A[…j…]
            A_new = jnp.tensordot(A_new, sqrt_S.astype(A_new.dtype),
                                  axes=([s + 1], [1]))
            A_new = jnp.moveaxis(A_new, -1, s + 1)
        return A_new, Bs

    # ── Integral / layer-operator wrappers ────────────────────────────────

    def _as_h_current(self, t_or_h_current):
        """Accept either a time or an already-evaluated ``h_current`` list."""
        if isinstance(t_or_h_current, (list, tuple)):
            return list(t_or_h_current)
        return self._get_h_current(t_or_h_current)

    def _ml_integrals(self, t_or_h_current, bs):
        """``(h_MO_list, g_MO_dict)`` at the given orbitals.

        ``g_MO_dict`` carries BOTH orderings of every interacting pair, as
        ``_transform_all_g`` always does; the 0.5 counting factors downstream
        rely on that.
        """
        h_current = self._as_h_current(t_or_h_current)
        h_MO_list = [
            self.transform_h(h_current[s], bs[s])[1]
            for s in range(self.num_species)
        ]
        g_MO_dict, _ = self._transform_all_g(bs, full=False)
        return h_MO_list, g_MO_dict

    def _top_ops(self, t_or_h_current, Bs, bs, es=None):
        """Top-layer Hamiltonian in ``(ops_diag, ops_pair)`` form."""
        h_MO_list, g_MO_dict = self._ml_integrals(t_or_h_current, bs)
        return self.dCI.ml_top_ops(h_MO_list, g_MO_dict, Bs, es=es)

    def _species_ops(self, t_or_h_current, bs):
        """The per-species determinant-space operators ``Op_s``."""
        h_MO_list, g_MO_dict = self._ml_integrals(t_or_h_current, bs)
        return self.dCI.ml_species_ops(h_MO_list, g_MO_dict)

    # ── Energy ────────────────────────────────────────────────────────────

    @partial(jit, static_argnums=0)
    def calculate_energy(self, t, y):
        """Total energy per state, ``E_α = ⟨A_α|H_top|A_α⟩ + e_shift``.

        Uses the same contraction path as the top-layer RHS, so it costs one
        ``apply_ci_ops`` and no RDM construction. ``h`` carries the ``-e_shift``
        constant (through ``transform_h``), which is added back here.
        """
        A, Bs, bs = self._y_to_ABb(y)
        top_ops = self._top_ops(t, Bs, bs)
        sigma = self.dCI.ml_get_sigma(A, top_ops)
        axes = tuple(range(1, A.ndim))
        return jnp.sum(A.conj() * sigma, axis=axes) + self.e_shift

    def suggest_e_shift(self, t, y):
        """Mean UNSHIFTED total energy over states — a good ``e_shift``.

        Inlines the one-body MO transform instead of routing through the
        ``@jit``'d ``transform_h``, which Python-branches on whether the shift
        constant is zero: tracing it with a temporary ``e_shift = 0`` would
        cache a no-shift version and poison every later call. Same caveat and
        same workaround as the base class.
        """
        A, Bs, bs = self._y_to_ABb(y)
        h_current = self._get_h_current(t)

        h_MO_list = []
        for s in range(self.num_species):
            b = bs[s]
            h_1 = contract("jm, ij -> im", b, h_current[s], backend="jax")
            h_2 = contract("in, im -> nm", b.conj(), h_1, backend="jax")
            h_MO_list.append(h_2)

        g_MO_dict, _ = self._transform_all_g(bs, full=False)
        top_ops = self.dCI.ml_top_ops(h_MO_list, g_MO_dict, Bs)
        sigma = self.dCI.ml_get_sigma(A, top_ops)
        axes = tuple(range(1, A.ndim))
        E = jnp.sum(A.conj() * sigma, axis=axes)
        return float(jnp.real(jnp.mean(E)))

    # ── Equations of motion, layer by layer ───────────────────────────────

    def _rhs_A_linear(self, A, top_ops):
        """Top layer: ``Ȧ = ∓ H_top A`` (``-`` in ITE, ``-i`` in RTE).

        Linear in A with the layer operators frozen, exactly like
        ``_rhs_C_linear`` is in the CI layer.
        """
        sigma = self.dCI.apply_ci_ops(top_ops, A)
        return -sigma if self.imag_time else -1j * sigma

    def _rhs_B_linear(self, B_s, s, Op_s, rho_inv_s, W_dict_s, B_frozen_s):
        """Middle layer: SPF RHS with the projector and densities frozen.

            Z_s = (1 − P_f)(Op_s B_s) + (1 − P_f) U_s · conj(ρ⁻¹_s)
            Ḃ_s = ∓ Z_s

        with ``U_s`` the SPF mean field (:meth:`SemiDirectCI_ML.apply_meanfield_B`)
        and ``P_f = B_f B_f†`` the projector onto the SPF space frozen at the
        substep start, applied in factored form.

        Two things the derivation fixes and the code must honour:

        * ``ρ⁻¹`` multiplies ONLY the mean-field term. The ``Op_s`` term comes
          with a ``ρ ρ⁻¹`` that cancels exactly, so applying the regularised
          inverse there would inject an O(eps) error — the same reason
          ``_rhs_b_linear`` does not apply ``D⁻¹`` to its one-body term.
        * it is ``(ρ⁻¹)ᵀ``, not ``ρ⁻¹``, that multiplies from the right;
          ``ρ⁻¹`` is Hermitian, so that is ``conj(ρ⁻¹)``.

        Linear in ``B_s``.
        """
        def project_out(X):
            return X - B_frozen_s @ (B_frozen_s.conj().T @ X)

        Z = project_out(Op_s @ B_s)

        if W_dict_s:
            U = self.dCI.apply_meanfield_B(s, B_s, W_dict_s)
            Z = Z + project_out(U) @ jnp.conj(rho_inv_s)

        return -Z if self.imag_time else -1j * Z

    # ── Densities ─────────────────────────────────────────────────────────

    def _compute_node_densities(self, A):
        """State-averaged SPF node densities of the top tensor.

        Returns ``(rho_invs, rho_pairs)``:

        * ``rho_invs[s]`` — the regularised inverse of
          ``ρ^s[j,j'] = Σ_α w_α Σ_other A*[…j…] A[…j'…]`` (``eps_spf``);
        * ``rho_pairs[(s,k)]`` — ``ρ^{sk}[x,z,y,w]`` in "xzyw" order, for
          every interacting pair, both orderings.

        Both stay FROZEN for the whole CMF step: only the partner SPFs (and
        with them ``e^k``) are re-frozen per SPF substep.
        """
        w = self.state_weights
        rho_invs = []
        for s in range(self.num_species):
            rho_s = self.dCI._compute_rho_ci(A, s, w)
            rho_invs.append(_regularized_inverse(self.eps_spf, rho_s))

        rho_pairs = {}
        for s in range(self.num_species):
            for k in range(self.num_species):
                if s == k or (s, k) in rho_pairs:
                    continue
                if not self._pair_interacts(s, k):
                    continue
                rho_pairs[(s, k)] = self.dCI._compute_rho_pair(A, s, k, w)
                rho_pairs[(k, s)] = jnp.einsum(
                    "xzyw->zxwy", rho_pairs[(s, k)]
                )
        return rho_invs, rho_pairs

    def _pair_interacts(self, s, k):
        """True if species s and k have an interaction in either ordering."""
        return (s, k) in self.g_dict or (k, s) in self.g_dict

    def _compute_full_RDMs(self, A, Bs, es=None):
        """State-averaged 1- and 2-RDMs of the full Ψ, for the orbital step.

        The ML replacement of ``integrators._compute_density_quantities``:
        the orbital equation is unchanged from standard MCTDHF, it just needs
        the RDMs of the ML state. Returns ``(D_invs, ds)``.
        """
        w = self.state_weights
        Ds = self.dCI.ml_1p_RDMs(A, Bs, weights=w)
        ds = self.dCI.ml_2p_RDMs(A, Bs, weights=w, es=es)
        D_invs = [self._invert_density_matrix(D) for D in Ds]
        return D_invs, ds

    # ── Normalisation ─────────────────────────────────────────────────────

    @partial(jit, static_argnums=0)
    def ortonorm_B(self, B):
        """Orthonormalise SPF columns via QR with positive-diagonal convention.

        NOTE this changes the state unless A is compensated (it is the
        ``ortonorm_b`` analogue, not a gauge transform). Used only as the
        per-step clean-up inside :meth:`integrate`, where the drift being
        removed is roundoff.
        """
        return self.ortonorm_b(B)

    @partial(jit, static_argnums=0)
    def ortonorm_A(self, A):
        """Orthonormalise the states of the top tensor (QR, positive diag)."""
        return self.ortonorm_C(A)

    @partial(jit, static_argnums=0)
    def normalize_A(self, A):
        """Normalise the states of the top tensor without orthogonalising."""
        return self.normalize_C(A)

    @partial(jit, static_argnums=0)
    def normalize_B(self, B):
        """Normalise SPF columns without orthogonalising."""
        return self.normalize_b(B)

    def _make_norm_fn(self, norm_mode):
        """The per-step renormalisation for the ML layout."""
        if norm_mode == "none":
            return None

        def norm_fn(y):
            A, Bs, bs = self._y_to_ABb(y)
            if norm_mode == "orthonormalize":
                A = self.ortonorm_A(A)
                Bs_new = [self.ortonorm_B(B) for B in Bs]
                bs_new = [self.ortonorm_b(b) for b in bs]
            else:
                A = self.normalize_A(A)
                Bs_new = [self.normalize_B(B) for B in Bs]
                bs_new = [self.normalize_b(b) for b in bs]
            return self._ABb_to_y(A, Bs_new, bs_new)

        return norm_fn

    # ── Default initial state ─────────────────────────────────────────────

    def _default_orbitals(self, t_init):
        """Lowest eigenvectors of ``h_s`` — the base class's default."""
        bs_init_list, evals_list = [], []
        for s, h in enumerate(self._get_h_current(t_init)):
            eigvals, eigvecs = np.linalg.eigh(np.array(h))
            evals_list.append(eigvals)
            b_init = eigvecs[:, : self.num_mctdhf_orbitals[s]]
            bs_init_list.append(jnp.array(b_init, dtype=jnp.complex64))
            print(
                f"Species {s}: lowest eigenvalues = "
                f"{eigvals[:self.num_mctdhf_orbitals[s]]}"
            )
        return bs_init_list, evals_list

    def _default_spfs(self, t_init, bs_init_list):
        """Lowest-``n_s`` eigenvectors of the bare ``Op_s``.

        ``Op_s`` is the species-s determinant-space Hamiltonian at the initial
        orbitals — one-body plus intra-species interaction, WITHOUT any
        mean field from the other species. The one-body integrals are
        transformed here inline and UNSHIFTED: ``e_shift`` has not been
        resolved yet at this point in ``integrate``, and a uniform shift only
        moves the eigenvalues, never the eigenvectors.

        Returns ``(Bs_init, spf_evals)``; the eigenvalues order the default
        top tensor's product states.
        """
        h_current = self._get_h_current(t_init)
        h_MO_list = []
        for s in range(self.num_species):
            b = bs_init_list[s]
            h_1 = contract("jm, ij -> im", b, h_current[s], backend="jax")
            h_2 = contract("in, im -> nm", b.conj(), h_1, backend="jax")
            h_MO_list.append(h_2)
        g_MO_dict, _ = self._transform_all_g(bs_init_list, full=False)

        Ops = self.dCI.ml_species_ops(h_MO_list, g_MO_dict)

        Bs_init, spf_evals = [], []
        for s in range(self.num_species):
            Op = np.array(Ops[s])
            Op = 0.5 * (Op + Op.conj().T)
            evals, evecs = np.linalg.eigh(Op)
            n_s = self.num_spfs[s]
            spf_evals.append(evals)
            Bs_init.append(
                jnp.array(evecs[:, :n_s], dtype=bs_init_list[s].dtype)
            )
            print(f"Species {s}: lowest SPF eigenvalues = {evals[:n_s]}")
        return Bs_init, spf_evals

    def _default_top_tensor(self, spf_evals, dtype=jnp.complex64):
        """Energy-ordered product states over the SPF indices."""
        ranges = [range(n) for n in self.num_spfs]
        product_states = []
        for indices in itertools.product(*ranges):
            E_nonint = sum(spf_evals[s][j] for s, j in enumerate(indices))
            product_states.append((E_nonint, indices))
        product_states.sort(key=lambda x: x[0])

        A_init = np.zeros(self.a_shape)
        for state_idx in range(min(self.num_states, len(product_states))):
            _, indices = product_states[state_idx]
            A_init[(state_idx,) + indices] = 1.0
        return jnp.array(A_init, dtype=dtype)

    # ── Integration ───────────────────────────────────────────────────────

    def integrate(
        self,
        integrator,
        num_steps,
        t_init=0,
        bs_init_list=None,
        A_init=None,
        Bs_init=None,
        normalize=None,
        noise_A=0.0,
        noise_B=0.0,
        noise_bs=0.0,
        save_every=1,
    ):
        """Run the ML-MCTDHF time propagation.

        Same semantics as :meth:`MCTDHF_n_species.integrate`, with the CI
        tensor replaced by the two upper layers: ``A_init``/``noise_A`` play
        the role of ``C_init``/``noise_C``, and ``Bs_init``/``noise_B`` are
        new.

        Defaults, when an initial layer is not supplied:

        * orbitals — lowest eigenvectors of ``h_s``, as in the base class;
        * SPFs — lowest-``n_s`` eigenvectors of the bare ``Op_s``
          (:meth:`_default_spfs`);
        * top tensor — product states ordered by the sum of those SPF
          eigenvalues.

        Returns ``(ts, ys)`` with the initial state prepended.
        """
        if bs_init_list is not None:
            if len(bs_init_list) != self.num_species:
                raise ValueError(
                    f"bs_init_list must have {self.num_species} elements, "
                    f"got {len(bs_init_list)}"
                )
            for s, b in enumerate(bs_init_list):
                if b.shape != self.b_shapes[s]:
                    raise ValueError(
                        f"bs_init_list[{s}] has shape {b.shape}, expected "
                        f"{self.b_shapes[s]}"
                    )
        if Bs_init is not None:
            if len(Bs_init) != self.num_species:
                raise ValueError(
                    f"Bs_init must have {self.num_species} elements, "
                    f"got {len(Bs_init)}"
                )
            for s, B in enumerate(Bs_init):
                if B.shape != self.B_shapes[s]:
                    raise ValueError(
                        f"Bs_init[{s}] has shape {B.shape}, expected "
                        f"{self.B_shapes[s]}"
                    )
        if A_init is not None and A_init.shape != self.a_shape:
            raise ValueError(
                f"A_init has shape {A_init.shape}, expected {self.a_shape}"
            )

        if bs_init_list is None:
            bs_init_list, _ = self._default_orbitals(t_init)

        spf_evals = None
        if Bs_init is None or A_init is None:
            Bs_default, spf_evals = self._default_spfs(t_init, bs_init_list)
            if Bs_init is None:
                Bs_init = Bs_default
        if A_init is None:
            A_init = self._default_top_tensor(spf_evals, dtype=Bs_init[0].dtype)

        # Noise on the initial guesses, then re-orthonormalise. Same rng
        # stream and ordering as the base class.
        rng = np.random.default_rng(42)
        if noise_A > 0:
            A_init = A_init + noise_A * jnp.array(
                rng.standard_normal(A_init.shape), dtype=A_init.dtype
            )
            A_init = self.ortonorm_A(A_init)
        if noise_B > 0:
            Bs_init = [
                self.ortonorm_B(
                    B + noise_B * jnp.array(
                        rng.standard_normal(B.shape), dtype=B.dtype)
                )
                for B in Bs_init
            ]
        if noise_bs > 0:
            bs_init_list = [
                self.ortonorm_b(
                    b + noise_bs * jnp.array(
                        rng.standard_normal(b.shape), dtype=b.dtype)
                )
                for b in bs_init_list
            ]

        y_init = self._ABb_to_y(A_init, Bs_init, bs_init_list)
        self._resolve_auto_e_shift(t_init, y_init)

        norm_mode = self._resolve_norm_mode(normalize)
        return self._scan_integrate(
            integrator, y_init, t_init, num_steps, save_every,
            self._make_norm_fn(norm_mode),
        )

    # ── Observables ───────────────────────────────────────────────────────

    @partial(jit, static_argnums=0)
    def get_1p_RDMs(self, y):
        """Per-state 1-RDMs ``D^s`` of the ML state, shape ``(num_states, M, M)``."""
        A, Bs, _bs = self._y_to_ABb(y)
        return self.dCI.ml_1p_RDMs(A, Bs)

    def calculate_particle_densities(self, A, Bs, bs_list, spf_list=None):
        """Particle densities per state and species.

        Shares
        :func:`mctdhf.postprocessing.particle_densities_from_rdms` with the
        standard solver; only the 1-RDMs come from ``(A, B)`` instead of a CI
        tensor.
        """
        Ds = self.dCI.ml_1p_RDMs(A, Bs)
        return postprocessing.particle_densities_from_rdms(Ds, bs_list, spf_list)

    @partial(jit, static_argnums=0)
    def calculate_overlap(self, y1, y2):
        """``⟨Ψ₁|Ψ₂⟩`` per state between two ML states.

        The determinant overlap matrices are the base class's
        (:meth:`_det_overlap_matrices`); they are then sandwiched between the
        two SPF matrices,

            S̃^s = B1_s† S^s B2_s,          (n_s, n_s)

        and contracted with the two top tensors by the same state einsum the
        base class uses. Cost is the determinant overlap build plus
        ``Π n_s`` — the ``Π D_s`` contraction never happens.
        """
        A1, Bs1, bs1 = self._y_to_ABb(y1)
        A2, Bs2, bs2 = self._y_to_ABb(y2)

        S_det = self._det_overlap_matrices(bs1, bs2)
        S_spf = [
            contract("Im, IJ, Jn -> mn", Bs1[s].conj(), S_det[s], Bs2[s],
                     backend="jax")
            for s in range(self.num_species)
        ]

        K = self.num_species
        c1 = [chr(97 + i) for i in range(K)]
        c2 = [chr(97 + K + i) for i in range(K)]
        s_pairs = ",".join(f"{c1[i]}{c2[i]}" for i in range(K))
        einsum_str = f"s{''.join(c1)},t{''.join(c2)},{s_pairs}->st"

        return contract(einsum_str, A1.conj(), A2, *S_spf, backend="jax")

    # ── Diagnostics ───────────────────────────────────────────────────────

    def spf_populations(self, y):
        """Eigenvalues of the state-averaged SPF node density ``ρ^s``.

        Descending. The smallest of these is the convergence monitor for
        ``n_s``: the weight sitting in the last SPF is what a further
        truncation would throw away.
        """
        A, _Bs, _bs = self._y_to_ABb(y)
        out = []
        for s in range(self.num_species):
            rho = self.dCI._compute_rho_ci(A, s, self.state_weights)
            out.append(jnp.real(jnp.linalg.eigvalsh(rho))[::-1])
        return out

    def natural_occupations(self, y):
        """Eigenvalues of the state-averaged 1-RDM ``D^s``. Descending.

        The convergence monitor for the orbital budget ``M_s``; they sum to
        ``N_s``.
        """
        A, Bs, _bs = self._y_to_ABb(y)
        Ds = self.dCI.ml_1p_RDMs(A, Bs, weights=self.state_weights)
        return [jnp.real(jnp.linalg.eigvalsh(D))[::-1] for D in Ds]

    # ── Gauge transforms ──────────────────────────────────────────────────

    @partial(jit, static_argnums=0)
    def rotate_to_natural_spfs(self, y):
        """Rotate the SPF layer into its natural basis. Exact gauge transform.

        Diagonalises the state-averaged node density ``ρ^s = U Λ U†`` and
        rotates ``B_s ← B_s conj(U)``, ``A ← A ×_s U`` (i.e.
        ``A[…m…] ← Σ_j U[j,m] A[…j…]``), which leaves Ψ, the energy and every
        overlap unchanged while making ``ρ^s`` diagonal with descending
        occupations.

        The conjugation is not cosmetic: ``ρ^s`` is stored with the
        CONJUGATED index first, so the transformation that diagonalises it is
        built from ``conj(U)``, not ``U``.
        """
        A, Bs, bs = self._y_to_ABb(y)
        Bs_new = []
        for s in range(self.num_species):
            rho = self.dCI._compute_rho_ci(A, s, self.state_weights)
            _evals, U = jnp.linalg.eigh(rho)
            U = U[:, ::-1]                      # descending occupations
            Bs_new.append(Bs[s] @ jnp.conj(U))
            A = jnp.tensordot(A, U, axes=([s + 1], [0]))
            A = jnp.moveaxis(A, -1, s + 1)
        return self._ABb_to_y(A, Bs_new, bs)


# ── Concrete solvers ──────────────────────────────────────────────────────


class MLMCTDHF_n_species(_MLMixin, MCTDHF_n_species):
    """ML-MCTDHF solver for N distinguishable species (4-index integrals).

    Constructor arguments are those of
    :class:`~mctdhf.MCTDHF_n_species.MCTDHF_n_species` plus:

    num_spfs : list of int
        Number of many-body SPFs per species, ``1 ≤ n_s ≤ D_s``. Keyword-only.
        ``num_spfs = num_dets`` reproduces standard MCTDHF exactly.
    eps_spf : float, optional
        Tikhonov regularisation for the SPF node-density inverse. Defaults to
        ``eps``. Keyword-only.
    """


class MLMCTDHF_n_species_DVR(_MLMixin, MCTDHF_n_species_DVR):
    """ML-MCTDHF with diagonal DVR 2-body integrals.

    Same constructor extension as :class:`MLMCTDHF_n_species`; the DVR
    specialisations of the integral transforms are inherited unchanged.
    """


# ── Exact gauge re-orthonormalisation ─────────────────────────────────────


def make_regauge_ml(solver):
    """Exact re-orthonormalisation of the ML state's orbitals and SPFs.

    A GAUGE change: Ψ, its energy and every overlap are unchanged; norms are
    not forced. This is not ``ortonorm_b``/``ortonorm_B``, which replace a
    layer by its Q factor WITHOUT compensating the layer above and therefore
    do change the state.

    Two stages per species:

    1. ``b_s = Q R`` (positive-diagonal convention, so Q and R are unique and
       runs are reproducible). Writing the old orbitals in terms of the new
       ones, ``φ_m = Σ_k R[k,m] ψ_k``, a Slater determinant of N old orbitals
       expands as ``|Φ_I⟩ = Σ_K det(R[occ_K, occ_I]) |Ψ_K⟩``, so the SPF
       coefficients transform with the N-th COMPOUND MATRIX (exterior power)
       of R: ``B_s ← M_s B_s`` with ``M_s[K,I] = det(R[occ_K, occ_I])``. At
       N = 1 this reduces to ``M = R``. (In standard MCTDHF the same M acts on
       the CI tensor; here it lands on B instead.)

    2. ``M_s B_s`` is no longer orthonormal, so ``B_s = Q_B R_B`` (positive
       diagonal again), ``B_s ← Q_B`` and ``A ← A ×_s R_B``, i.e.
       ``A[…j…] ← Σ_{j'} R_B[j,j'] A[…j'…]``.

    The occupation lists are read out of ``dCI.I[s]`` once here, at closure
    build time — they are numpy bool arrays and ``np.flatnonzero`` cannot run
    inside the jitted function.
    """
    occs = []
    for s in range(solver.num_species):
        dets = np.asarray(solver.dCI.I[s])
        occs.append(jnp.asarray(np.array([np.flatnonzero(d) for d in dets])))

    @jax.jit
    def regauge(y):
        A, Bs, bs = solver._y_to_ABb(y)
        bs_new, Bs_new = [], []
        for s, b in enumerate(bs):
            Q, R = jnp.linalg.qr(b)
            sgn = jnp.sign(jnp.real(jnp.diag(R)))
            sgn = jnp.where(sgn == 0, 1.0, sgn).astype(R.dtype)
            Q = Q * sgn[None, :]          # Q' = Q S
            R = sgn[:, None] * R          # R' = S R, so Q'R' = QR = b exactly
            bs_new.append(Q)

            occ = occs[s]                                     # (n_dets, N)
            # sub[K, I, a, b] = R[occ_K[a], occ_I[b]]
            sub = R[occ[:, None, :, None], occ[None, :, None, :]]
            M = jnp.linalg.det(sub)                           # (n_dets, n_dets)
            B = M @ Bs[s]

            Q_B, R_B = jnp.linalg.qr(B)
            sgn_B = jnp.sign(jnp.real(jnp.diag(R_B)))
            sgn_B = jnp.where(sgn_B == 0, 1.0, sgn_B).astype(R_B.dtype)
            Q_B = Q_B * sgn_B[None, :]
            R_B = sgn_B[:, None] * R_B
            Bs_new.append(Q_B)

            # A[…j…] ← Σ_{j'} R_B[j, j'] A[…j'…]
            A = jnp.tensordot(A, R_B, axes=([s + 1], [1]))
            A = jnp.moveaxis(A, -1, s + 1)

        return solver._ABb_to_y(A, Bs_new, bs_new)

    return regauge
