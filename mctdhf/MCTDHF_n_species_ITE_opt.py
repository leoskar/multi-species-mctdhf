"""Specialized MCTDHF solver for state-averaged imaginary-time evolution
with a DVR basis, treating the 1-body Hamiltonian `h_list` as a traced JIT
argument.

This is a stripped-down companion to `MCTDHF_n_species` / `MCTDHF_n_species_DVR`
purpose-built for an outer optimization loop that varies only the 1-body
Hamiltonian between iterations (e.g. `test/optimize_qubit_params.py`). The
general module bakes `h_list` into the JIT trace via `self.h_list`, forcing
a full recompilation each time the optimizer asks for a new evaluation.

Hard-wired choices:

* DVR basis (2-D `g_dict` form, diagonal in real space)
* CMF integrator with `c_propagator="diagonalize"` and `b_propagator="sil"`
* Imaginary time only
* Time-independent `h_list` (no callable-h support)

Pattern of use::

    mctdhf = MCTDHF_n_species_ITE_opt(
        g_dict=g_dict, num_particles=..., num_mctdhf_orbitals=...,
        num_spatial_orbitals=..., num_states=..., state_weights=...,
    )
    ts, ys = mctdhf.integrate(num_steps, dt, h_list_A)
    ts, ys = mctdhf.integrate(num_steps, dt, h_list_B)   # NO recompile

The CMF step, `_diag_solve_C`, `_sil_solve_bs`, and
`_compute_density_quantities` mirror the implementations in
``mctdhf/integrators.py`` (lines 374, 421, 583, 603) but with the imag-time
branches inlined and `h_list` flowing through as a traced argument.
"""

import itertools
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from opt_einsum import contract

from . import postprocessing
from .SemiDirectCI_n_species import SemiDirectCI_n_species


# ── CMF helper functions (ITE-only adaptations of integrators.py helpers) ────


def _diag_solve_C(mctdh, h_MO_list, g_MO_dict, c_shape):
    """ITE C step: replace C with the lowest ``num_states`` eigenvectors of H_CI.

    Mirrors ``_diag_solve_C`` in ``mctdhf/integrators.py:374`` (imag-time branch
    only; dt is irrelevant for ITE in this branch).
    """
    num_states = c_shape[0]
    det_shape = c_shape[1:]
    n_dets_tot = 1
    for n in det_shape:
        n_dets_tot *= int(n)

    eye = jnp.eye(n_dets_tot, dtype=jnp.complex64).reshape((n_dets_tot,) + det_shape)
    sigma = mctdh.dCI.get_sigma(eye, h_MO_list, g_MO_dict)
    H_CI = sigma.reshape(n_dets_tot, n_dets_tot)
    H_CI = 0.5 * (H_CI + H_CI.conj().T)
    _, eigvecs = jnp.linalg.eigh(H_CI)

    return eigvecs[:, :num_states].T.reshape(c_shape)


def _sil_solve_bs(mctdh, bs_start, h_list, D_invs, ds, bs_frozen, P_frozens,
                  dt, b_shapes, krylov_dim):
    """ITE SIL orbital propagation with full re-orthogonalisation.

    Mirrors ``_sil_solve_bs`` in ``mctdhf/integrators.py:421`` (imag-time
    branch only). `h_list[s]` is the spatial 1-body matrix for species s
    and is passed straight through as `h_current_s` to `_rhs_b_linear`.
    """
    bs_new = list(bs_start)
    for s in range(mctdh.num_species):
        b_s = bs_start[s]
        b_flat = b_s.reshape(-1)
        dim = b_flat.shape[0]
        norm_b = jnp.linalg.norm(b_flat)
        safe_norm = jnp.where(norm_b > 0, norm_b, 1.0)

        def apply_A(v_flat, _s=s):
            b = v_flat.reshape(b_shapes[_s])
            rhs = mctdh._rhs_b_linear(
                b, _s, h_list[_s], D_invs[_s], ds, bs_frozen, P_frozens[_s]
            )
            # _rhs_b_linear returns -A·b in ITE; negate to get the Hermitian A·b.
            return (-rhs).reshape(-1)

        v0 = b_flat / safe_norm
        V_init = jnp.zeros((krylov_dim, dim), dtype=b_flat.dtype)
        vec_dtype = b_flat.dtype
        real_dtype = b_flat.real.dtype

        def lanczos_step(carry, j):
            V, v_prev, v_curr, beta_prev = carry
            V = V.at[j].set(v_curr)
            w = apply_A(v_curr).astype(vec_dtype)
            alpha = jnp.real(jnp.dot(v_curr.conj(), w)).astype(real_dtype)
            w = w - alpha * v_curr - beta_prev * v_prev
            overlaps = jnp.dot(V.conj(), w)
            w = w - jnp.dot(overlaps, V)
            beta = jnp.sqrt(jnp.real(jnp.dot(w.conj(), w))).astype(real_dtype)
            safe_beta = jnp.where(beta > 1e-15, beta, 1.0)
            v_next = w / safe_beta
            return (V, v_curr, v_next, beta), (alpha, beta)

        init = (V_init, jnp.zeros_like(v0), v0, jnp.array(0.0, dtype=real_dtype))
        (V_final, _, _, _), (alphas, betas) = jax.lax.scan(
            lanczos_step, init, jnp.arange(krylov_dim), length=krylov_dim
        )

        T = jnp.diag(alphas) + jnp.diag(betas[:-1], 1) + jnp.diag(betas[:-1], -1)
        expT = jax.scipy.linalg.expm(-dt * T)
        e1 = jnp.zeros(krylov_dim, dtype=expT.dtype).at[0].set(1.0)
        coeffs = expT @ e1

        b_new_flat = safe_norm * jnp.dot(coeffs, V_final)
        b_new_flat = jnp.where(norm_b > 0, b_new_flat, b_flat)
        bs_new[s] = b_new_flat.reshape(b_shapes[s])

    return bs_new


def _compute_density_quantities(mctdh, C):
    """State-averaged 1-RDM inverses and 2-RDMs.

    Mirrors ``_compute_density_quantities`` in ``mctdhf/integrators.py:583``.
    """
    Ds = mctdh.dCI.get_1p_RDMs(C)
    ds = mctdh.dCI.get_2p_RDMs(C)
    Ds = [mctdh._state_average(D) for D in Ds]
    ds = {k: mctdh._state_average(v) for k, v in ds.items()}
    D_invs = [mctdh._invert_density_matrix(Ds[s]) for s in range(mctdh.num_species)]
    return D_invs, ds


def _make_cmf_step(mctdh, dt, sil_b_krylov_dim):
    """Build the CMF step function.

    The returned ``step(t, y, h_list) -> (t_new, y_new)`` is JIT-compiled
    with `h_list` as a traced argument, so calling it with different
    `h_list` values does not retrigger compilation. Algorithm matches
    ``get_cmf_integrator(mctdh, dt, c_propagator="diagonalize",
    b_propagator="sil")`` in ``mctdhf/integrators.py:603``.
    """
    c_shape = (mctdh.num_states,) + tuple(mctdh.num_dets)
    b_shapes = [
        (mctdh.num_spatial_orbitals[s], mctdh.num_mctdhf_orbitals[s])
        for s in range(mctdh.num_species)
    ]

    @jax.jit
    def step(t, y, h_list):
        C, bs = mctdh._y_to_Cb(y)

        # transform_h applies the constant energy shift (H - e_shift) to the
        # MO integrals; the orbital sub-problem below uses the spatial h_list
        # directly and stays unshifted, keeping the shift an exact phase.
        h_MO_list = [
            mctdh.transform_h(h_list[s], bs[s])[1]
            for s in range(mctdh.num_species)
        ]
        g_MO_dict, _ = mctdh._transform_all_g(bs)

        C_new = _diag_solve_C(mctdh, h_MO_list, g_MO_dict, c_shape)

        D_invs, ds = _compute_density_quantities(mctdh, C_new)
        P_frozens = [bs[s] @ bs[s].conj().T for s in range(mctdh.num_species)]

        bs_new = _sil_solve_bs(
            mctdh, bs, h_list, D_invs, ds, bs, P_frozens,
            dt, b_shapes, sil_b_krylov_dim,
        )

        return t + dt, mctdh._Cb_to_y(C_new, bs_new)

    return step


# ── Solver class ─────────────────────────────────────────────────────────────


class MCTDHF_n_species_ITE_opt:
    """State-averaged MCTDHF imaginary-time solver for DVR basis.

    Specialised for an outer optimisation loop that varies only `h_list`:
    construct once with the static configuration, then call
    :meth:`integrate` any number of times with different `h_list` values.
    The JIT'd step compiles on the first call and cache-hits afterwards
    because `h_list` is threaded as a traced argument.

    Parameters mirror the relevant subset of :class:`MCTDHF_n_species`.
    `imag_time` and the DVR basis are hard-wired.
    """

    imag_time = True  # Used by reused helpers from `mctdhf/integrators.py`.

    def __init__(
        self,
        g_dict,
        num_particles,
        num_mctdhf_orbitals,
        num_spatial_orbitals,
        num_states,
        state_weights=None,
        eps=1e-6,
        e_shift=0.0,
    ):
        err = "Input lists must have the same length (number of species)."
        np.testing.assert_equal(len(num_particles), len(num_mctdhf_orbitals), err)
        np.testing.assert_equal(len(num_particles), len(num_spatial_orbitals), err)

        self.num_species = len(num_particles)
        self.num_particles = num_particles
        self.num_mctdhf_orbitals = num_mctdhf_orbitals
        self.num_spatial_orbitals = num_spatial_orbitals

        self.g_dict = g_dict
        self.dCI = SemiDirectCI_n_species(num_mctdhf_orbitals, num_particles)
        self.num_dets = self.dCI.num_dets
        self.num_states = num_states
        self.eps = eps

        if state_weights is None:
            w = jnp.ones(num_states, dtype=jnp.float32)
        else:
            w = jnp.array(state_weights, dtype=jnp.float32)
            if w.shape != (num_states,):
                raise ValueError(
                    f"state_weights must have shape ({num_states},), got {w.shape}"
                )
        self.state_weights = w / jnp.sum(w)

        # Constant Hamiltonian energy shift (H - e_shift), realized as a
        # per-species c*I term on the 1-body integrals. Pure global phase;
        # observables unchanged, calculate_energy adds it back. Default 0.0
        # is a no-op. See MCTDHF_n_species for details.
        self.e_shift = float(e_shift)

        # Caches keyed by (dt, save_every, sil_b_krylov_dim) and (dt, sil_b_krylov_dim).
        self._step_cache = {}
        self._run_cache = {}

    # ── State vector packing ──────────────────────────────────────────────

    def _Cb_to_y(self, C, bs_list):
        C_flat = C.reshape(-1)
        bs_flat = [b.reshape(-1) for b in bs_list]
        return jnp.concatenate([C_flat] + bs_flat)

    def _y_to_Cb(self, y):
        c_size = int(np.prod(self.num_dets) * self.num_states)
        C = y[:c_size].reshape((self.num_states,) + tuple(self.num_dets))
        start = c_size
        bs_list = []
        for s in range(self.num_species):
            b_size = self.num_spatial_orbitals[s] * self.num_mctdhf_orbitals[s]
            b_shape = (self.num_spatial_orbitals[s], self.num_mctdhf_orbitals[s])
            bs_list.append(y[start : start + b_size].reshape(b_shape))
            start += b_size
        return C, bs_list

    def _state_average(self, tensor):
        return contract("s,s...->...", self.state_weights, tensor, backend="jax")

    def _h_shift_constant(self):
        """Uniform per-particle constant c = -e_shift / N_total added to the MO
        integral h_2 inside transform_h, shifting the total Hamiltonian by
        -e_shift (sum_s c*N_s = -e_shift)."""
        return -self.e_shift / sum(self.num_particles)

    # ── Integral transforms (DVR g) ───────────────────────────────────────

    @partial(jit, static_argnums=0)
    def transform_h(self, h, b):
        """1-body integral transform (basis is irrelevant — h is always 2-D).

        Applies the constant energy shift (H - e_shift) by adding c*I_MO to the
        fully transformed h_2 only. This is exact regardless of orbital
        orthonormality; h_1/h_3 and the orbital sub-problem stay unshifted.
        """
        h_1 = contract("jm, ij -> im", b, h, backend="jax")
        h_2 = contract("in, im -> nm", b.conj(), h_1, backend="jax")
        h_3 = contract("in, nm -> im", b, h_2, backend="jax")
        c = self._h_shift_constant()
        if c != 0.0:
            h_2 = h_2 + c * jnp.eye(h_2.shape[0], dtype=h_2.dtype)
        return h_1, h_2, h_3

    @partial(jit, static_argnums=0)
    def transform_g(self, g, b_sigma, b_tau):
        """DVR 2-body integral transform — `g` is 2-D (diagonal in real space)."""
        g_2 = contract("jq, js, ij -> iqs", b_tau.conj(), b_tau, g, backend="jax")
        g_3 = contract("ir, iqs -> iqrs", b_sigma, g_2, backend="jax")
        g_4 = contract("ip, iqrs -> pqrs", b_sigma.conj(), g_3, backend="jax")
        g_5 = contract("ip, pqrs -> iqrs", b_sigma, g_4, backend="jax")
        return g_3, g_4, g_5

    def _transpose_g(self, g):
        """DVR transpose: swap species roles, (i, j) → (j, i)."""
        return jnp.transpose(g, (1, 0))

    def _transform_all_g(self, bs):
        """Fully transformed MO 2-body integrals per species pair."""
        g_MO_dict = {}
        for s1 in range(self.num_species):
            for s2 in range(self.num_species):
                if (s1, s2) in self.g_dict:
                    g_spatial = self.g_dict[(s1, s2)]
                elif (s2, s1) in self.g_dict:
                    g_spatial = self._transpose_g(self.g_dict[(s2, s1)])
                else:
                    continue
                _, g_4, _ = self.transform_g(g_spatial, bs[s1], bs[s2])
                g_MO_dict[(s1, s2)] = g_4
        return g_MO_dict, None

    # ── RHS pieces used by the CMF step ───────────────────────────────────

    def _rhs_b_linear(self, b_s, s, h_current_s, D_inv, ds, bs_frozen, P_frozen):
        """Orbital RHS for ITE with frozen P and density matrices.

        Returns -A·b (imag-time convention). Mirrors
        ``MCTDHF_n_species._rhs_b_linear`` (line 323) with the imag-time
        branch fixed.
        """
        h_1 = contract("ij, jm -> im", h_current_s, b_s, backend="jax")
        h_3 = P_frozen @ h_1
        b_dot = h_1 - h_3

        for k in range(self.num_species):
            if (s, k) in self.g_dict:
                g_spatial = self.g_dict[(s, k)]
            elif (k, s) in self.g_dict:
                g_spatial = self._transpose_g(self.g_dict[(k, s)])
            else:
                continue

            g_3, _, _ = self.transform_g(g_spatial, b_s, bs_frozen[k])
            g_5 = contract("ij, jqrs -> iqrs", P_frozen, g_3, backend="jax")
            d_sk = ds[(s, k)]
            b_dot = b_dot + contract(
                "np, pqrs, iqrs -> in", D_inv, d_sk, g_3 - g_5, backend="jax"
            )

        return -b_dot

    @partial(jit, static_argnums=0)
    def _invert_density_matrix(self, D):
        """Tikhonov-regularised inverse via complex128 eigh."""
        D_64 = D.astype(jnp.complex128)
        eigvals_64, eigvecs_64 = jnp.linalg.eigh(D_64)
        D_inv_64 = (eigvecs_64 / (eigvals_64 + self.eps)) @ eigvecs_64.conj().T
        return D_inv_64.astype(jnp.complex64)

    # ── Energy ────────────────────────────────────────────────────────────

    @partial(jit, static_argnums=0)
    def calculate_energy(self, y, h_list):
        """Total state-averaged energy. `h_list` is a traced argument."""
        C, bs = self._y_to_Cb(y)
        h_MO_list = [
            self.transform_h(h_list[s], bs[s])[1]
            for s in range(self.num_species)
        ]
        g_MO_dict, _ = self._transform_all_g(bs)
        # transform_h shifts the MO integrals by -e_shift; add it back here.
        return self.dCI.calculate_energy(C, h_MO_list, g_MO_dict) + self.e_shift

    # ── Orthonormalisation (always applied for ITE) ───────────────────────

    @partial(jit, static_argnums=0)
    def ortonorm_b(self, b):
        Q, R = jnp.linalg.qr(b)
        signs = jnp.sign(jnp.diag(R))
        signs = jnp.where(signs == 0, 1.0, signs)
        return Q * signs[None, :]

    @partial(jit, static_argnums=0)
    def ortonorm_C(self, C):
        original_shape = C.shape
        C_flat = C.reshape(C.shape[0], -1)
        Q, R = jnp.linalg.qr(C_flat.T)
        signs = jnp.sign(jnp.diag(R))
        signs = jnp.where(signs == 0, 1.0, signs)
        return ((Q * signs[None, :]).T).reshape(original_shape)

    # ── Integration ───────────────────────────────────────────────────────

    def _build_run(self, num_steps, dt, save_every, sil_b_krylov_dim):
        """Build a JIT'd integration function for fixed (num_steps, dt, ...).

        Returns ``run(t_init, y_init, h_list) -> (ts, ys)``. Different
        `h_list` arrays of the same shape/dtype reuse the same compiled HLO.
        """
        step = self._step_cache.get((dt, sil_b_krylov_dim))
        if step is None:
            step = _make_cmf_step(self, dt, sil_b_krylov_dim)
            self._step_cache[(dt, sil_b_krylov_dim)] = step

        num_chunks = num_steps // save_every
        remainder = num_steps % save_every

        def step_fn(carry, _):
            t, y, h_list = carry
            t_new, y_new = step(t, y, h_list)
            C, bs = self._y_to_Cb(y_new)
            C = self.ortonorm_C(C)
            bs_new = [self.ortonorm_b(b) for b in bs]
            y_new = self._Cb_to_y(C, bs_new)
            return (t_new, y_new, h_list), None

        def chunk_fn(carry, _):
            carry, _ = jax.lax.scan(step_fn, carry, None, length=save_every)
            t, y, _ = carry
            return carry, (t, y)

        @jax.jit
        def run(t_init, y_init, h_list):
            init = (t_init, y_init, h_list)

            if num_chunks > 0:
                (t, y, _), (ts_arr, ys_arr) = jax.lax.scan(
                    chunk_fn, init, None, length=num_chunks
                )
            else:
                t, y = t_init, y_init
                ts_arr = jnp.zeros((0,), dtype=t_init.dtype)
                ys_arr = jnp.zeros((0,) + y_init.shape, dtype=y_init.dtype)

            if remainder > 0:
                (t, y, _), _ = jax.lax.scan(
                    step_fn, (t, y, h_list), None, length=remainder
                )
                ts_extra = t[None]
                ys_extra = y[None]
            else:
                ts_extra = jnp.zeros((0,), dtype=t_init.dtype)
                ys_extra = jnp.zeros((0,) + y_init.shape, dtype=y_init.dtype)

            ts = jnp.concatenate([t_init[None], ts_arr, ts_extra])
            ys = jnp.concatenate([y_init[None, ...], ys_arr, ys_extra])
            return ts, ys

        return run

    def integrate(
        self,
        num_steps,
        dt,
        h_list,
        t_init=0.0,
        C_init=None,
        bs_init_list=None,
        sil_b_krylov_dim=20,
        save_every=1,
        noise_C=0.0,
        noise_bs=0.0,
    ):
        """Run state-averaged ITE for ``num_steps`` of size ``dt``.

        Parameters
        ----------
        num_steps : int
            Number of time steps. Static — different values build different
            cached compilations.
        dt : float
            Time step. Static.
        h_list : sequence of array-like
            1-body Hamiltonians per species, shape `(N_s, N_s)` each.
            **Traced** — varying it across calls does not retrigger
            compilation.
        t_init : float
        C_init, bs_init_list : optional
            Warm-start arrays. If None, `bs` defaults to lowest eigenvectors
            of each `h_list[s]`, and `C` defaults to energy-ordered product
            states (same convention as the general
            :meth:`MCTDHF_n_species.integrate`).
        sil_b_krylov_dim : int
            Krylov dimension for the orbital SIL substep.
        save_every : int
            Save every N steps. Static.
        noise_C, noise_bs : float
            Optional Gaussian perturbations on the initial guesses (re-
            orthonormalised after).

        Returns
        -------
        ts, ys : jnp.ndarray
            Saved times and packed state vectors.
        """
        h_list_jnp = tuple(jnp.asarray(h, dtype=jnp.float32) for h in h_list)

        if bs_init_list is not None and len(bs_init_list) != self.num_species:
            raise ValueError(
                f"bs_init_list must have {self.num_species} elements, got {len(bs_init_list)}"
            )

        evals_list = []
        if bs_init_list is None:
            bs_init_list = []
            for s in range(self.num_species):
                eigvals, eigvecs = np.linalg.eigh(np.asarray(h_list_jnp[s]))
                evals_list.append(eigvals)
                b_init = eigvecs[:, : self.num_mctdhf_orbitals[s]]
                bs_init_list.append(jnp.array(b_init, dtype=jnp.complex64))

        if C_init is None:
            if not evals_list:
                for s in range(self.num_species):
                    eigvals, _ = np.linalg.eigh(np.asarray(h_list_jnp[s]))
                    evals_list.append(eigvals)
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

        rng = np.random.default_rng(42)
        if noise_C > 0:
            C_init = C_init + noise_C * jnp.array(
                rng.standard_normal(C_init.shape), dtype=jnp.complex64
            )
            C_init = self.ortonorm_C(C_init)
        if noise_bs > 0:
            bs_init_list = [
                self.ortonorm_b(
                    b + noise_bs * jnp.array(rng.standard_normal(b.shape),
                                              dtype=jnp.complex64)
                )
                for b in bs_init_list
            ]

        y_init = self._Cb_to_y(C_init, bs_init_list)
        t_init_jnp = jnp.asarray(t_init, dtype=jnp.float32)

        cache_key = (int(num_steps), float(dt), int(save_every), int(sil_b_krylov_dim))
        run = self._run_cache.get(cache_key)
        if run is None:
            run = self._build_run(num_steps, dt, save_every, sil_b_krylov_dim)
            self._run_cache[cache_key] = run

        return run(t_init_jnp, y_init, h_list_jnp)

    # ── Post-processing delegators ────────────────────────────────────────

    def rotate_to_natural_orbitals(self, y):
        """See :func:`mctdhf.postprocessing.rotate_to_natural_orbitals`."""
        return postprocessing.rotate_to_natural_orbitals(self, y)
