"""State-averaged ML-MCTDHF imaginary-time relaxation for an outer
optimisation loop over the one-body Hamiltonian ``h_list``.

The ML counterpart of :mod:`mctdhf.MCTDHF_n_species_ITE_opt`. Build it once
for a fixed system (interaction, particle numbers, orbital and SPF budgets)
and call :meth:`MLMCTDHF_n_species_ITE_opt.integrate` with a different
``h_list`` for each operating point. ``h_list`` is a TRACED argument of every
jitted entry point, so a new operating point does not recompile.

What one iteration does
-----------------------

It is the improved-relaxation step of ``register.relax`` in the paper's
three-electron simulation code (``split="strang"``, ``precision="mixed"``)
with its roots selected by identity, and it is the one-electron-per-well
construction of the paper's supplementary material one layer up:

1. rotate the SPF layer into its natural basis under the state-averaging
   weights (``_regauge_natural``), so SPF index ``k`` of species ``s`` means
   level ``k`` of that species;
2. build the top-layer Hamiltonian at the current ``(B, b)`` densely
   (``_dense_top``) and diagonalise it COMPLETELY;
3. take the roots at the frozen energy ranks ``cols`` as the new top tensor
   ``A``, one row per relaxed configuration;
4. node densities, then the SPF and orbital layers as ``B/2 . b . B/2``,
   each a defect-form Crank-Nicolson solve (``_cn_defect``).

The top layer is REPLACED, not propagated, so ``dt`` enters only the SPF
and orbital solves.

The weights go by IDENTITY
--------------------------

``relax_configs`` are product configurations (one level per species) and
``relax_weights`` their state-averaging weights. The default is the paper's
set: the ground configuration at weight 1 and the single excitations at 1/2,
with every other configuration at 0 and therefore not propagated at all.
Row ``a`` of ``A`` is always the root matched to ``relax_configs[a]``, so a
weight is attached to a configuration and not to an energy rank. That
matters wherever an intruder sits between the configurations in energy: with
three electrons per well, energy rank 4 is a double excitation of the lowest
well, not ``|0001>``.

The same weights fix the gauge. With (1, 1/2 x 4) every species ends up with
``rho^s = (5/6, 1/6, ~0, ...)``, which is non-degenerate on the two levels
the qubit corner reads. The ordering of the empty levels ``>= 2`` is
arbitrary, but nothing here reads them.

Matching and the ``lax.scan``
-----------------------------

The Hungarian assignment is SciPy and cannot run inside a scan. So
:meth:`integrate` runs chunks of ``chunk`` iterations, each ONE jitted
``lax.scan`` with the rank set ``cols`` frozen, and re-matches on the host
between chunks against the fixed one-hot reference of ``relax_configs``.

If the end-of-chunk match moves any weight onto a different root, the chunk
is thrown away and redone from its start state with per-iteration matching,
which is the ``register.relax`` loop. A swap between two roots of EQUAL
weight is not a change: the densities, the only place the weights enter, do
not see it.

Every evaluation runs a fixed number of iterations. It has no ``|dE|`` stop,
because a stalled relaxation reads as a converged one. :meth:`relax_to_tol` is
the tolerance-stopped variant for cold starts and reference runs.

Precision
---------

``precision="mixed"`` forms the CN residual in float64 and solves for the
INCREMENT in float32 BiCGSTAB. At a fixed point the float64 residual is
zero, so the fixed point is the float64 one whatever the inner tolerance,
and the inner tolerance only sets the rate. ``"x64d"`` is the same defect
form solved in float64. The dense top block, its ``eigh``, the regauge and
all integral transforms stay float64 in both. Requires ``jax_enable_x64``.

Provenance
----------

``_regauge_natural``, ``_dense_top``, ``_config_reference`` and
``_additivity`` are ported from the simulation code of the four-well paper
(``three_electrons/system/register.py``); ``_lower`` and ``_cn_defect`` from
``three_electrons/system/mixed.py``; ``_B_half`` and ``_b_full`` from the same
module term for term, with ``ml._get_h_current(0.0)`` replaced by the traced
``h_list``; ``_qubit_corner`` from ``utils.qubit_corner_fidelities`` of the
one-electron simulations. They are copies so that this module does not depend
on the simulation code. None of the helpers carries its own ``@jit`` (nested
jits change the GPU results in the last bits); only the entry points built in
``__init__`` are jitted.
"""

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
from opt_einsum import contract
from scipy.optimize import linear_sum_assignment

from .MLMCTDHF_n_species import MLMCTDHF_n_species_DVR


# ── Configurations ────────────────────────────────────────────────────────


def register_configs(K, n_levels=2):
    """The ``n_levels**K`` register configurations in the paper's order.

    Sorted by total excitation, then with EARLIER species carrying the
    excitation first:

        |0000>, |1000>, |0100>, |0010>, |0001>, |1100>, |1010>, ...

    This is ``utils.iter_bitstrings`` / ``_energy_order_perm`` of the
    one-electron simulation code, the column order its loss modules index
    into (stage 2's ``A, B = 1, 2`` means |1000>, |0100>).

    ⚠ ``register.iter_bitstrings`` of the three-electron simulation code sorts by
    ``(sum(c), c)`` instead, which puts |0001> first among the singles —
    the reverse — despite its docstring. Do not substitute it here.
    """
    configs = list(product(range(n_levels), repeat=K))
    return sorted(configs, key=lambda c: (sum(c), tuple(-x for x in c)))


def _config_reference(configs, num_spfs, dtype=np.complex128):
    """One-hot top tensors, row ``a`` = ``configs[a]`` in the current SPF basis.

    Ported from ``register.config_reference``, generalised to per-species
    SPF counts.
    """
    A = np.zeros((len(configs),) + tuple(num_spfs), dtype=dtype)
    for a, c in enumerate(configs):
        A[(a,) + tuple(c)] = 1.0
    return A


def _additivity(evals, configs):
    """Detunings, additivity residual and the ``healthy`` verdict.

    Ported from ``register.additivity``. ``evals[a]`` is the energy of the
    root matched to ``configs[a]``. ``Delta_s`` is fitted from the single
    excitations; the residual is the largest departure of any configuration
    from ``E(0) + sum_s n_s Delta_s`` (the static ZZ scale). ``healthy``
    requires every detuning positive and the ground configuration lowest.

    It catches a COLLAPSED register, which purity cannot. It does not catch
    a mislabelled level (seen on the four-well register at ``n_s = 6``); only comparing
    ``sum E`` against another setting does.
    """
    idx = {tuple(c): a for a, c in enumerate(configs)}
    K = len(configs[0])
    zero = tuple([0] * K)
    e = np.asarray(evals)
    e0 = e[idx[zero]]
    det = np.array([e[idx[tuple(1 if i == s else 0 for i in range(K))]] - e0
                    for s in range(K)])
    resid = max(abs(e[idx[tuple(c)]] - e0 - float(np.dot(det, c)))
                for c in configs)
    healthy = bool(det.min() > 0.0 and np.argmin(e) == idx[zero])
    return det, float(resid), healthy


def _qubit_corner(A, K, n_levels=2):
    """``F[k, n] = |A[k, n_0, ..., n_{K-1}]|^2``, ``n_s < n_levels``.

    A copy of ``utils.qubit_corner_fidelities`` of the one-electron simulation code,
    applied to the ML top tensor. The columns come out in
    :func:`register_configs` order. It presumes the natural-SPF gauge, where
    the leading ``n_levels`` SPFs of each species are the qubit levels.
    """
    sl = (slice(None),) + (slice(0, n_levels),) * K
    F_flat = jnp.reshape(jnp.abs(A[sl]) ** 2, (A.shape[0], n_levels ** K))
    rowmajor = list(product(range(n_levels), repeat=K))
    perm = sorted(range(len(rowmajor)),
                  key=lambda i: (sum(rowmajor[i]),
                                 tuple(-c for c in rowmajor[i])))
    return F_flat[:, jnp.array(perm)]


# ── Layer helpers (plain functions, traced inside the entry points) ────────


def _regauge_natural(ml, A, Bs, weights):
    """Rotate the SPF layer into the natural basis of the weighted node
    density, occupations descending. Exact: Psi is unchanged.

    Ported from ``register.regauge_natural``. ``rho^s`` is stored with the
    CONJUGATED index first, so the SPF transform is ``conj(U)``.
    """
    Bs_new = []
    for s in range(ml.num_species):
        rho = ml.dCI._compute_rho_ci(A, s, weights)
        _evals, U = jnp.linalg.eigh(rho)
        U = U[:, ::-1]
        Bs_new.append(Bs[s] @ jnp.conj(U))
        A = jnp.tensordot(A, U, axes=([s + 1], [0]))
        A = jnp.moveaxis(A, -1, s + 1)
    return A, Bs_new


def _dense_top(ml, top_ops, a_shape):
    """The top-layer Hamiltonian as a dense Hermitian matrix.

    Ported from ``register.dense_top``: the top operator applied to the
    identity in SPF-product space. The symmetrisation is cosmetic (the raw
    block is Hermitian to ~4e-17 on the four-well register).
    """
    dim = int(np.prod(a_shape[1:]))
    eye = jnp.eye(dim, dtype=jnp.complex128).reshape((dim,) + tuple(a_shape[1:]))
    H = ml.dCI.apply_ci_ops(top_ops, eye).reshape(dim, dim)
    return 0.5 * (H + H.conj().T)


def _top_matrix(ml, Bs, bs, h_list, a_shape):
    h_MO, g_MO = ml._ml_integrals(h_list, bs)
    es = [ml.dCI.spf_transition_ops(s, Bs[s]) for s in range(ml.num_species)]
    return _dense_top(ml, ml.dCI.ml_top_ops(h_MO, g_MO, Bs, es=es), a_shape)


def _lower(tree):
    """complex128 -> complex64 and float64 -> float32 on every leaf; integer
    leaves (index tables) pass through. Ported from ``mixed.lower``."""
    def cast(x):
        if not hasattr(x, "dtype"):
            return x
        if x.dtype == jnp.complex128:
            return x.astype(jnp.complex64)
        if x.dtype == jnp.float64:
            return x.astype(jnp.float32)
        return x
    return jax.tree_util.tree_map(cast, tree)


def _cn_defect(rhs_hi, rhs_lo, x0, dt, tol, maxiter, lo_dtype):
    """CN step ``x1 = x0 + d``, ``(I - dt/2 L_lo) d = dt L_hi x0``.

    Ported from ``mixed.cn_defect``. Algebraically ``_cn_linear``'s
    ``(I - dt/2 L) x1 = (I + dt/2 L) x0``, written for the increment so the
    residual (float64) and the solve (``lo_dtype``) can differ. BiCGSTAB's
    error is relative to ``d``, which vanishes at a fixed point, so the
    inner ``tol`` never moves the fixed point.
    """
    shape = x0.shape
    # Trace-time guard: one float64 leaf left in the frozen operators would
    # promote every matvec back to complex128 and "mixed" would silently be
    # x64 under another name.
    probe = jax.eval_shape(rhs_lo, jax.ShapeDtypeStruct(shape, lo_dtype))
    if probe.dtype != lo_dtype:
        raise TypeError(f"low-precision RHS returns {probe.dtype}, "
                        f"expected {lo_dtype}")
    r = dt * rhs_hi(x0)

    def matvec(d_flat):
        d = d_flat.reshape(shape)
        return (d - (dt / 2) * rhs_lo(d)).reshape(-1)

    r_lo = r.astype(lo_dtype).reshape(-1)
    d_flat, _ = jax.scipy.sparse.linalg.bicgstab(
        matvec, r_lo, x0=r_lo, tol=tol, maxiter=maxiter)
    return x0 + d_flat.reshape(shape).astype(x0.dtype)


def _B_half(ml, A, Bs, bs, h_list, rho_invs, rho_pairs, dt, tol, maxiter, lo):
    """SPF layer over ``dt``: one frozen-projector substep, defect-form CN.

    ``mixed.B_half`` term for term (itself ``_solve_Bs_subcycled`` with one
    substep), with ``h_list`` passed in instead of read from the solver.
    """
    h_MO, g_MO = ml._ml_integrals(h_list, bs)
    Ops = ml.dCI.ml_species_ops(h_MO, g_MO)
    w = ml.state_weights
    es = [ml.dCI.spf_transition_ops(s, Bs[s]) for s in range(ml.num_species)]

    out = list(Bs)
    for s in range(ml.num_species):
        if ml.spf_is_full[s]:
            continue
        W = ml.dCI.ml_meanfield_W(A, Bs, g_MO, s, es=es, weights=w,
                                  rho_pairs=rho_pairs)
        frozen = (Ops[s], rho_invs[s], W, Bs[s])
        frozen_lo = _lower(frozen) if lo else frozen

        def rhs_hi(X, _s=s, _f=frozen):
            return ml._rhs_B_linear(X, _s, *_f)

        def rhs_lo(X, _s=s, _f=frozen_lo):
            return ml._rhs_B_linear(X, _s, *_f)

        out[s] = _cn_defect(rhs_hi, rhs_lo, Bs[s], dt, tol, maxiter,
                            jnp.complex64 if lo else Bs[s].dtype)
    return [ml.ortonorm_B(B) for B in out]


def _b_full(ml, A, Bs, bs, h_list, dt, tol, maxiter, lo):
    """Orbital layer over ``dt``: one frozen-projector substep, defect CN.

    ``mixed.b_full`` term for term (``_cn_solve_bs`` with one substep), with
    ``h_list`` passed in.
    """
    es = [ml.dCI.spf_transition_ops(s, Bs[s]) for s in range(ml.num_species)]
    D_invs, ds = ml._compute_full_RDMs(A, Bs, es)
    g2 = ml._precompute_frozen_g2(bs)

    out = list(bs)
    for s in range(ml.num_species):
        frozen = (h_list[s], D_invs[s], ds, bs, g2)
        frozen_lo = _lower(frozen) if lo else frozen

        def rhs_hi(x, _s=s, _f=frozen):
            return ml._rhs_b_linear(x, _s, *_f)

        def rhs_lo(x, _s=s, _f=frozen_lo):
            return ml._rhs_b_linear(x, _s, *_f)

        out[s] = _cn_defect(rhs_hi, rhs_lo, bs[s], dt, tol, maxiter,
                            jnp.complex64 if lo else bs[s].dtype)
    return [ml.ortonorm_b(b) for b in out]


# ── Solver ────────────────────────────────────────────────────────────────


class MLMCTDHF_n_species_ITE_opt:
    """ML-MCTDHF state-averaged relaxation with a traced ``h_list``.

    Parameters
    ----------
    g_dict : dict
        DVR two-body integrals ``{(s, k): g}``, as for
        :class:`~mctdhf.MLMCTDHF_n_species.MLMCTDHF_n_species_DVR`.
    num_particles, num_mctdhf_orbitals, num_spatial_orbitals, num_spfs : list
        Per species.
    relax_configs : list of tuple, optional
        The configurations that are relaxed and carry weight, one level per
        species. Default: the ground configuration and the single
        excitations of the two-level register, in :func:`register_configs`
        order.
    relax_weights : array, optional
        Their state-averaging weights, normalised here. Default: 1 for the
        ground configuration and 1/2 for each single excitation. These also
        fix the natural-SPF gauge.
    all_configs : list of tuple, optional
        The configurations :meth:`analyse` matches and reports. Default: the
        two-level register in :func:`register_configs` order.
    dt : float
        Imaginary-time step of the SPF and orbital solves.
    precision : {"mixed", "x64d"}
        Defect-form CN solved in float32 or float64 (module docstring).
    inner_tol, inner_maxiter : float, int
        BiCGSTAB tolerance (relative, on the increment) and iteration cap.
    chunk : int
        Iterations per jitted ``lax.scan``; roots are re-matched between
        chunks.
    eps : float
        Tikhonov regularisation of the orbital and SPF density inverses.
    h_placeholder : list of arrays, optional
        The ``h_list`` the wrapped solver is built with. It is never read on
        any path of this class; the default is NaN so that any accidental
        use poisons the result instead of passing silently. The ``traced``
        validation passes real values here to show the traced ``h_list`` is
        the one used.
    """

    def __init__(self, g_dict, num_particles, num_mctdhf_orbitals,
                 num_spatial_orbitals, num_spfs, relax_configs=None,
                 relax_weights=None, all_configs=None, dt=2e-2,
                 precision="mixed", inner_tol=1e-2, inner_maxiter=200,
                 chunk=50, eps=1e-6, h_placeholder=None):
        if not jax.config.jax_enable_x64:
            raise ValueError(
                "MLMCTDHF_n_species_ITE_opt needs jax_enable_x64=True: the "
                "residuals, the top block and its eigh are float64. Set "
                "jax.config.update('jax_enable_x64', True) before creating "
                "any array.")
        if precision not in ("mixed", "x64d"):
            raise ValueError(
                f"precision must be 'mixed' or 'x64d', got {precision!r}")

        K = len(num_particles)
        self.num_species = K
        self.num_spfs = tuple(int(n) for n in num_spfs)
        if len(self.num_spfs) != K:
            raise ValueError(f"num_spfs needs {K} entries")

        if all_configs is None:
            all_configs = register_configs(K, 2)
        if relax_configs is None:
            relax_configs = [c for c in register_configs(K, 2)
                             if sum(c) <= 1]
        relax_configs = [tuple(int(v) for v in c) for c in relax_configs]
        all_configs = [tuple(int(v) for v in c) for c in all_configs]
        if relax_weights is None:
            if any(sum(c) > 1 for c in relax_configs):
                raise ValueError(
                    "relax_weights has a default only for the ground "
                    "configuration and single excitations; pass it "
                    "explicitly for any other relax_configs")
            relax_weights = [1.0 if sum(c) == 0 else 0.5
                             for c in relax_configs]
        w = np.asarray(relax_weights, dtype=np.float64)
        if w.shape != (len(relax_configs),) or np.any(w <= 0):
            raise ValueError(
                "relax_weights must be positive, one per relax config; a "
                "zero-weight configuration should simply not be relaxed")
        for c in relax_configs + all_configs:
            if len(c) != K or any(not 0 <= c[s] < self.num_spfs[s]
                                  for s in range(K)):
                raise ValueError(f"configuration {c} does not fit "
                                 f"num_spfs = {self.num_spfs}")

        self.relax_configs = relax_configs
        self.all_configs = all_configs
        self.relax_weights = w / w.sum()
        self.num_states = len(relax_configs)
        self.dt = float(dt)
        self.precision = precision
        self.inner_tol = float(inner_tol)
        self.inner_maxiter = int(inner_maxiter)
        self.chunk = int(chunk)
        self.eps = float(eps)

        if h_placeholder is None:
            h_placeholder = [jnp.full((n, n), jnp.nan, dtype=jnp.complex128)
                             for n in num_spatial_orbitals]
        self.ml = MLMCTDHF_n_species_DVR(
            h_list=list(h_placeholder), g_dict=g_dict,
            num_particles=list(num_particles),
            num_mctdhf_orbitals=list(num_mctdhf_orbitals),
            num_spatial_orbitals=list(num_spatial_orbitals),
            num_spfs=list(self.num_spfs), num_states=self.num_states,
            imag_time=True, eps=self.eps, state_weights=self.relax_weights,
            e_shift=0.0,
        )
        # The base class stores its weights in float32, so 1/3 and 1/6 would
        # enter every density rounded at 3e-8. They are read at trace time,
        # before anything below is traced, so replacing them here is safe.
        self.ml.state_weights = jnp.asarray(self.relax_weights)
        self._w = jnp.asarray(self.relax_weights)

        self.a_shape = self.ml.a_shape
        self.dim = int(np.prod(self.num_spfs))
        self._flat_relax = np.array([np.ravel_multi_index(c, self.num_spfs)
                                     for c in relax_configs])
        self._flat_all = np.array([np.ravel_multi_index(c, self.num_spfs)
                                   for c in all_configs])
        # Roots of EQUAL weight may trade places without that being a change
        # of the relaxed functional (module docstring).
        self._weight_groups = [np.nonzero(np.isclose(self.relax_weights, v))[0]
                               for v in np.unique(np.round(self.relax_weights,
                                                           12))]
        self._build_entry_points()

    # ── jitted entry points ───────────────────────────────────────────────

    def _build_entry_points(self):
        ml, w, a_shape = self.ml, self._w, self.a_shape
        dt, tol, maxit = self.dt, self.inner_tol, self.inner_maxiter
        lo = self.precision == "mixed"
        flat_relax = jnp.asarray(self._flat_relax)
        S = self.num_states

        def front(A, Bs, bs, h):
            A, Bs = _regauge_natural(ml, A, Bs, w)
            evals, evecs = jnp.linalg.eigh(_top_matrix(ml, Bs, bs, h, a_shape))
            return Bs, evals, evecs

        def back(evecs, cols, Bs, bs, h):
            A = evecs[:, cols].T.reshape((S,) + a_shape[1:])
            rho_invs, rho_pairs = ml._compute_node_densities(A)
            Bs_h = _B_half(ml, A, Bs, bs, h, rho_invs, rho_pairs, dt / 2,
                           tol, maxit, lo)
            bs_new = _b_full(ml, A, Bs_h, bs, h, dt, tol, maxit, lo)
            Bs_new = _B_half(ml, A, Bs_h, bs_new, h, rho_invs, rho_pairs,
                             dt / 2, tol, maxit, lo)
            return A, Bs_new, bs_new

        def chunk(A, Bs, bs, h, cols, length):
            def body(carry, _):
                A, Bs, bs = carry
                Bs, evals, evecs = front(A, Bs, bs, h)
                A, Bs, bs = back(evecs, cols, Bs, bs, h)
                return (A, Bs, bs), jnp.sum(evals[cols])
            (A, Bs, bs), e = jax.lax.scan(body, (A, Bs, bs), None,
                                          length=length)
            return A, Bs, bs, e

        def front_match(A, Bs, bs, h):
            Bs, evals, evecs = front(A, Bs, bs, h)
            return Bs, evals, evecs, jnp.abs(evecs[flat_relax, :]) ** 2

        def analyse_dev(A, Bs, bs, h):
            A_rg, Bs_rg = _regauge_natural(ml, A, Bs, w)
            evals, evecs = jnp.linalg.eigh(
                _top_matrix(ml, Bs_rg, bs, h, a_shape))
            occ = [jnp.linalg.eigvalsh(ml.dCI._compute_rho_ci(A_rg, s, w))[::-1]
                   for s in range(ml.num_species)]
            return A_rg, Bs_rg, evals, evecs, occ

        self._chunk = jax.jit(chunk, static_argnames=("length",))
        self._front_match = jax.jit(front_match)
        self._back = jax.jit(back)
        self._analyse_dev = jax.jit(analyse_dev)

    def compile_counts(self):
        """Number of compiled variants per entry point (for the ``traced``
        validation: a new ``h_list`` must not add one)."""
        return {name: getattr(self, name)._cache_size()
                for name in ("_chunk", "_front_match", "_back",
                             "_analyse_dev")}

    # ── host-side helpers ─────────────────────────────────────────────────

    @staticmethod
    def as_h(h_list):
        return tuple(jnp.asarray(h, dtype=jnp.complex128) for h in h_list)

    def _assign(self, ovl):
        """Hungarian: relax config ``a`` -> energy rank ``cols[a]``."""
        ovl = np.asarray(ovl)
        rows, cols = linear_sum_assignment(-ovl)
        cols = cols[np.argsort(rows)]
        return cols, ovl[np.arange(len(cols)), cols]

    def _same_weighted_roots(self, cols_a, cols_b):
        return all(set(cols_a[g]) == set(cols_b[g])
                   for g in self._weight_groups)

    def match(self, state, h_list, configs=None):
        """Regauge, diagonalise, and Hungarian-match ``configs`` (default
        ``relax_configs``) onto the roots via the fixed one-hot reference.

        Returns ``dict(ranks, evals, overlaps, evals_all)``: ``ranks[a]`` is
        the energy rank of the root matched to ``configs[a]``, ``overlaps[a]``
        its population on that configuration.
        """
        h = self.as_h(h_list)
        A, Bs, bs = state
        if configs is None:
            _Bs, evals, _evecs, ovl = self._front_match(A, Bs, bs, h)
        else:
            flat = np.array([np.ravel_multi_index(tuple(c), self.num_spfs)
                             for c in configs])
            _Bs, evals, evecs, _ = self._front_match(A, Bs, bs, h)
            ovl = jnp.abs(evecs[jnp.asarray(flat), :]) ** 2
        cols, pop = self._assign(ovl)
        evals = np.asarray(evals)
        return dict(ranks=cols, evals=evals[cols], overlaps=pop,
                    evals_all=evals)

    def _matched_steps(self, state, h, n):
        """``n`` iterations with per-iteration matching (the
        ``register.relax`` loop). Returns the state, per-step sum E and the
        per-step ranks."""
        A, Bs, bs = state
        es, ranks = [], []
        for _ in range(n):
            Bs, evals, evecs, ovl = self._front_match(A, Bs, bs, h)
            cols, _ = self._assign(ovl)
            A, Bs, bs = self._back(evecs, jnp.asarray(cols), Bs, bs, h)
            es.append(float(np.sum(np.asarray(evals)[cols])))
            ranks.append(cols)
        return (A, Bs, bs), np.array(es), ranks

    # ── integration ───────────────────────────────────────────────────────

    def integrate(self, num_steps, h_list, state):
        """Run exactly ``num_steps`` iterations at ``h_list`` from ``state``.

        ``state = (A, Bs, bs)``. Chunked as described in the module
        docstring. Returns ``(state, e_sum, info)``: ``e_sum[i]`` is the sum
        of the relaxed roots' energies at iteration ``i`` (before its layer
        step), and ``info`` holds ``fallback_chunks`` (indices of chunks
        redone with per-iteration matching), ``last_chunk_fallback``,
        ``dE_last_chunk`` (the change of ``e_sum`` across the last chunk)
        and the final ``ranks``.
        """
        h = self.as_h(h_list)
        A, Bs, bs = state
        cols = self.match((A, Bs, bs), h)["ranks"]
        e_all, fallback, done, k = [], [], 0, 0
        last_len = 0
        while done < num_steps:
            L = min(self.chunk, num_steps - done)
            A1, Bs1, bs1, e = self._chunk(A, Bs, bs, h, jnp.asarray(cols),
                                          length=L)
            cols_end = self.match((A1, Bs1, bs1), h)["ranks"]
            e = np.asarray(e)
            if not self._same_weighted_roots(cols, cols_end):
                (A1, Bs1, bs1), e, _r = self._matched_steps((A, Bs, bs), h, L)
                cols_end = self.match((A1, Bs1, bs1), h)["ranks"]
                fallback.append(k)
            A, Bs, bs, cols = A1, Bs1, bs1, cols_end
            e_all.append(e)
            done += L
            last_len = L
            k += 1
        e_sum = np.concatenate(e_all) if e_all else np.zeros(0)
        # Change across the last chunk: from the iteration before it, or from
        # its own first iteration when it is the only chunk.
        if len(e_sum) == 0:
            dE = float("nan")
        else:
            ref = e_sum[-1 - last_len] if len(e_sum) > last_len else e_sum[0]
            dE = float(abs(e_sum[-1] - ref))
        info = dict(fallback_chunks=fallback,
                    last_chunk_fallback=bool(fallback and fallback[-1] == k - 1),
                    dE_last_chunk=dE, ranks=cols, n_chunks=k)
        return (A, Bs, bs), e_sum, info

    def relax_to_tol(self, h_list, state, tol=1e-9, max_steps=20000,
                     verbose=False):
        """Relax until ``|dE| < tol`` between the last two iterations of a
        chunk, or ``max_steps``. Stops only at chunk boundaries.

        ⚠ ``sum E ~ 1.3e4`` for five states has a float64 ulp of 1.8e-12, so
        ``tol`` much below ~1e-10 terminates on rounding, not convergence.
        """
        e_all, fallback, steps, converged = [], [], 0, False
        info = {}
        while steps < max_steps:
            state, e, info = self.integrate(self.chunk, h_list, state)
            e_all.append(e)
            fallback += [steps // self.chunk] if info["fallback_chunks"] else []
            steps += len(e)
            if not np.all(np.isfinite(e)):
                break
            if verbose:
                print(f"    it {steps:6d}  sum E = {e[-1]:.10f}  "
                      f"dE = {e[-1] - e[-2]:+.3e}", flush=True)
            if abs(e[-1] - e[-2]) < tol:
                converged = True
                break
        e_sum = np.concatenate(e_all) if e_all else np.zeros(0)
        info = dict(info, converged=converged, steps=steps,
                    fallback_chunks=fallback)
        return state, e_sum, info

    # ── analysis ──────────────────────────────────────────────────────────

    def analyse(self, state, h_list, integrate_info=None):
        """Full diagnosis of a relaxed state at ``h_list``.

        Regauge, diagonalise the top block completely, then Hungarian-match
        ``all_configs`` onto the roots. Returns a dict with

        * ``F_matched`` (``n_cfg x n_cfg``) — row ``i`` is the root matched to
          ``all_configs[i]``, column ``j`` its population on
          ``all_configs[j]``; ``E_matched`` their energies;
        * ``F_energy`` / ``E_all`` — the same for EVERY root, energy order;
        * ``ranks_all``, ``ranks_relax``, ``e_relax`` (the sum E of the
          relaxed configurations from their own match);
        * ``detunings``, ``additivity``, ``healthy`` (:func:`_additivity`);
        * ``occupations`` — weighted node-density spectra per species, ideal
          (5/6, 1/6, 0, ...) for the default weights — and ``leakage``, the
          occupation beyond the first two levels;
        * ``dE_last_chunk`` / ``last_chunk_fallback`` from ``integrate_info``;
        * ``state_regauged`` — ``(A, Bs, bs)`` in the natural gauge.
        """
        h = self.as_h(h_list)
        A, Bs, bs = state
        A_rg, Bs_rg, evals, evecs, occ = self._analyse_dev(A, Bs, bs, h)
        evals = np.asarray(evals)
        F_energy = np.asarray(jnp.abs(evecs[jnp.asarray(self._flat_all), :])
                              ** 2).T
        rows, cols = linear_sum_assignment(-F_energy.T)
        ranks_all = cols[np.argsort(rows)]
        F_matched = F_energy[ranks_all, :]
        E_matched = evals[ranks_all]

        ovl_relax = np.asarray(jnp.abs(evecs[jnp.asarray(self._flat_relax), :])
                               ** 2)
        ranks_relax, pop_relax = self._assign(ovl_relax)

        det, resid, healthy = _additivity(E_matched, self.all_configs)
        occ = np.array([np.asarray(o) for o in occ])
        out = dict(
            F_matched=F_matched, E_matched=E_matched,
            F_energy=F_energy, E_all=evals, ranks_all=ranks_all,
            ranks_relax=ranks_relax, pop_relax=pop_relax,
            e_relax=float(np.sum(evals[ranks_relax])),
            detunings=det, additivity=resid, healthy=healthy,
            occupations=occ, leakage=occ[:, 2:].sum(axis=1),
            state_regauged=(A_rg, Bs_rg, bs),
        )
        if integrate_info is not None:
            out["dE_last_chunk"] = integrate_info.get("dE_last_chunk")
            out["last_chunk_fallback"] = integrate_info.get(
                "last_chunk_fallback")
        return out

    # ── initial state ─────────────────────────────────────────────────────

    def default_state(self, bs, h_list):
        """``(A, Bs, bs)`` from given orbitals.

        ``Bs``: lowest ``n_s`` eigenvectors of each species' bare
        determinant-space Hamiltonian at ``bs`` (``_default_spfs`` with the
        given ``h_list`` instead of the solver's, unshifted). ``A``: the
        one-hot reference of ``relax_configs``.
        """
        ml = self.ml
        bs = [jnp.asarray(b, dtype=jnp.complex128) for b in bs]
        h = self.as_h(h_list)
        h_MO = []
        for s in range(ml.num_species):
            h_1 = contract("jm, ij -> im", bs[s], h[s], backend="jax")
            h_MO.append(contract("in, im -> nm", bs[s].conj(), h_1,
                                 backend="jax"))
        g_MO, _ = ml._transform_all_g(bs, full=False)
        Ops = ml.dCI.ml_species_ops(h_MO, g_MO)
        Bs = []
        for s in range(ml.num_species):
            Op = np.asarray(Ops[s])
            _e, v = np.linalg.eigh(0.5 * (Op + Op.conj().T))
            Bs.append(jnp.asarray(v[:, :self.num_spfs[s]],
                                  dtype=jnp.complex128))
        A = jnp.asarray(_config_reference(self.relax_configs, self.num_spfs))
        return A, Bs, bs
