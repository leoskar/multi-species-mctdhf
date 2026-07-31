"""Species-batched CMF half-step integrator (SIL/SIL) for the DVR basis.

Parallel `_opt`-style companion to ``get_cmf_half_step_integrator`` in
``mctdhf/integrators.py``. The core integrator runs one Short-Iterative-Lanczos
orbital recursion per species in a Python loop, which unrolls into K
back-to-back sequences of small kernels — on a GPU each is too small to
saturate the device, so K species cost ~K× the launch latency. This module
collapses those K recursions into a single ``jax.vmap``'d recursion over a
stacked species axis, batches the K density-matrix inverses into one call, and
folds in the loop-invariant ``g_2`` precompute (the frozen-tau half of the
2-body transform, constant across all Lanczos steps of a substep).

Applicability (asserted at construction):

* DVR basis (``MCTDHF_n_species_DVR``): 2-body ``g`` is a 2-D matrix.
* Uniform shapes: every species shares the same ``(num_spatial, num_mctdhf)``.
  This is what makes the ``vmap`` batch axis well-defined; unequal shapes would
  need padding, which is out of scope here.
* SIL for both the CI and orbital substeps.

Intra-species 2-body terms ARE supported: the CMF substep freezes the tau
orbital for every partner k including k == s, so the diagonal g_2[s,s] is built
from the frozen b_s and is loop-invariant just like the off-diagonal blocks.

Physics matches ``get_cmf_half_step_integrator(mctdh, dt, c_propagator="sil",
b_propagator="sil")`` term for term; only the execution is reshaped. Verified
against it in ``tests`` / scratchpad on the four-well system.
"""

import jax
import jax.numpy as jnp
from opt_einsum import contract

from .MCTDHF_n_species import _regularized_inverse, MCTDHF_n_species_DVR
from .integrators import _sil_solve_C


# ── Per-substep batched preparation ──────────────────────────────────────────


def _build_g2_all(mctdh, bs_frozen, dtype):
    """Frozen-tau half of the DVR 2-body transform, for every (s, k) pair.

    g2_all[s, k, i, q, s'] = Σ_j b_f[k]*[j,q] g_{sk}[i,j] b_f[k][j,s']
    (the ``contract("jq, js, ij -> iqs", ...)`` of the DVR ``transform_g``,
    which depends only on the frozen partner orbital and the static integral).

    Non-interacting pairs — and, when present, would-be entries with no g —
    are left as zeros, so the mean-field contribution through them vanishes
    identically (g_3 and g_5 are built from g2 and are then zero). The
    diagonal k == s (intra-species) block is filled whenever (s, s) is in
    g_dict, from the frozen b_s.
    """
    K = mctdh.num_species
    N = mctdh.num_spatial_orbitals[0]
    M = mctdh.num_mctdhf_orbitals[0]

    rows = []
    for s in range(K):
        row = []
        for k in range(K):
            if (s, k) in mctdh.g_dict:
                g_sk = mctdh.g_dict[(s, k)]
            elif (k, s) in mctdh.g_dict:
                g_sk = mctdh._transpose_g(mctdh.g_dict[(k, s)])
            else:
                row.append(jnp.zeros((N, M, M), dtype=dtype))
                continue
            g2 = mctdh._transform_g_tau(g_sk, bs_frozen[k])
            row.append(g2.astype(dtype))
        rows.append(jnp.stack(row))
    return jnp.stack(rows)  # (K, K, N, M, M)


def _build_d_all(mctdh, ds, dtype):
    """State-averaged 2-RDMs stacked over ordered pairs: d_all[s, k] = ds[(s,k)].

    Shape (K, K, M, M, M, M). ds already contains every (s, k) key (incl. the
    diagonal). Entries whose g2 block is zero contribute nothing regardless of
    their value, so no gating on g_dict is needed here.
    """
    K = mctdh.num_species
    rows = []
    for s in range(K):
        rows.append(jnp.stack([ds[(s, k)].astype(dtype) for k in range(K)]))
    return jnp.stack(rows)  # (K, K, M, M, M, M)


def _batched_density(mctdh, C, dtype):
    """Stacked, state-averaged D⁻¹ (K, M, M) and 2-RDM tensor (K, K, M, M, M, M).

    The K density-matrix inverses are computed in one ``vmap`` over the
    Cholesky solve instead of K separate ``eigh``/Cholesky launches.
    """
    Ds = mctdh.dCI.get_1p_RDMs(C)
    ds = mctdh.dCI.get_2p_RDMs(C)
    Ds = [mctdh._state_average(D) for D in Ds]
    ds = {k: mctdh._state_average(v) for k, v in ds.items()}

    D_stack = jnp.stack(Ds)  # (K, M, M)
    eps = mctdh.eps
    Dinv_all = jax.vmap(lambda D: _regularized_inverse(eps, D))(D_stack)
    Dinv_all = Dinv_all.astype(dtype)

    d_all = _build_d_all(mctdh, ds, dtype)
    return Dinv_all, d_all


# ── Batched SIL orbital solve ────────────────────────────────────────────────


def _sil_solve_bs_batched(
    bs_start_stack,   # (K, N, M) — Krylov start (may differ from frozen)
    h_all,            # (K, N, N)
    bf_all,           # (K, N, M) — frozen orbitals (projector + g2 source)
    Dinv_all,         # (K, M, M)
    g2_all,           # (K, K, N, M, M)
    d_all,            # (K, K, M, M, M, M)
    dt,
    imag_time,        # static
    krylov_dim,       # static
):
    """Propagate every species' orbital block by exp(∓ dt A_s) via one vmapped
    Lanczos recursion. Mirrors ``integrators._sil_solve_bs`` per species.

    A_s is applied by ``apply_A`` below, which reproduces the (factored-P) DVR
    ``_rhs_b_linear`` with the frozen-tau g2 precomputed: the varying Krylov
    vector enters only as the sigma orbital.
    """

    def one_species(b_start, h_s, bf_s, Dinv_s, g2_s, d_s):
        N, M = b_start.shape
        b_flat = b_start.reshape(-1)
        dim = b_flat.shape[0]
        norm_b = jnp.linalg.norm(b_flat)
        safe_norm = jnp.where(norm_b > 0, norm_b, 1.0)

        vec_dtype = b_flat.dtype
        real_dtype = b_flat.real.dtype

        def apply_A(v_flat):
            b = v_flat.reshape(N, M)
            # 1-body (1-P) h b, projector in factored form
            h1 = h_s @ b
            h3 = bf_s @ (bf_s.conj().T @ h1)
            b_dot = h1 - h3
            # 2-body mean field, summed over partners k (leading axis of g2/d).
            # g_3[k,i,q,r,s'] = Σ_? b[i,r] g2_s[k,i,q,s']
            g3 = jnp.einsum("ir, kiqs -> kiqrs", b, g2_s)
            # (1-P) applied on the spatial index i, factored through bf_s
            proj = jnp.einsum("jn, kjqrs -> knqrs", bf_s.conj(), g3)
            g5 = jnp.einsum("jn, knqrs -> kjqrs", bf_s, proj)
            contrib = jnp.einsum(
                "np, kpqrs, kiqrs -> in", Dinv_s, d_s, g3 - g5
            )
            b_dot = b_dot + contrib
            # apply_A must return A·b (Hermitian-operator convention): the
            # ∓ sign/-1j of _rhs_b_linear is applied via expT below, so here
            # we return the raw b_dot = A·b for both ITE and RTE.
            return b_dot.reshape(-1)

        v0 = b_flat / safe_norm
        V_init = jnp.zeros((krylov_dim, dim), dtype=vec_dtype)

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
        if imag_time:
            expT = jax.scipy.linalg.expm(-dt * T)
        else:
            expT = jax.scipy.linalg.expm(-1j * dt * T)

        e1 = jnp.zeros(krylov_dim, dtype=expT.dtype).at[0].set(1.0)
        coeffs = expT @ e1
        sil_error = jnp.abs(betas[-1] * coeffs[-1])

        b_new_flat = safe_norm * jnp.dot(coeffs, V_final)
        b_new_flat = jnp.where(norm_b > 0, b_new_flat, b_flat)
        return b_new_flat.reshape(N, M), sil_error

    b_new_all, errs = jax.vmap(one_species)(
        bs_start_stack, h_all, bf_all, Dinv_all, g2_all, d_all
    )
    return b_new_all, jnp.max(errs)


# ── Integrator ───────────────────────────────────────────────────────────────


def get_cmf_half_step_integrator_batched(
    mctdh,
    dt,
    sil_c_krylov_dim=10,
    sil_b_krylov_dim=20,
):
    """Species-batched CMF half-step integrator, SIL for both substeps.

    Drop-in for ``get_cmf_half_step_integrator(mctdh, dt, c_propagator="sil",
    b_propagator="sil", c_kwargs={"krylov_dim": sil_c_krylov_dim},
    b_kwargs={"krylov_dim": sil_b_krylov_dim})`` on a DVR system with uniform
    per-species shapes. Same predictor/corrector algorithm and the same SIL
    truncation, with the per-species orbital loop and the K density inverses
    batched.

    Returns ``integrate(t, y) -> (t_new, y_new)``.
    """
    if not isinstance(mctdh, MCTDHF_n_species_DVR):
        raise TypeError(
            "batched integrator requires MCTDHF_n_species_DVR (DVR 2-body form)."
        )
    N0 = mctdh.num_spatial_orbitals[0]
    M0 = mctdh.num_mctdhf_orbitals[0]
    if any(n != N0 for n in mctdh.num_spatial_orbitals) or any(
        m != M0 for m in mctdh.num_mctdhf_orbitals
    ):
        raise ValueError(
            "batched integrator requires uniform (num_spatial, num_mctdhf) "
            f"across species; got spatial={mctdh.num_spatial_orbitals}, "
            f"mctdhf={mctdh.num_mctdhf_orbitals}."
        )

    imag_time = mctdh.imag_time
    K = mctdh.num_species
    c_shape = (mctdh.num_states,) + tuple(mctdh.num_dets)
    dt_half = dt / 2

    @jax.jit
    def integrate(t, y):
        C, bs = mctdh._y_to_Cb(y)
        state_dtype = bs[0].dtype

        h_t = mctdh._get_h_current(t)
        h_half = mctdh._get_h_current(t + dt_half)
        h_all_t = jnp.stack([jnp.asarray(h, dtype=state_dtype) for h in h_t])
        h_all_half = jnp.stack(
            [jnp.asarray(h, dtype=state_dtype) for h in h_half]
        )

        # ── Predictor: half-step with mean fields at t ──────────────────
        h_MO_t = [mctdh.transform_h(h_t[s], bs[s])[1] for s in range(K)]
        g_MO_t, _ = mctdh._transform_all_g(bs, full=False)
        C_half, _ = _sil_solve_C(
            mctdh, C, h_MO_t, g_MO_t, dt_half, c_shape, sil_c_krylov_dim
        )

        Dinv_t, d_t = _batched_density(mctdh, C, state_dtype)
        g2_t = _build_g2_all(mctdh, bs, state_dtype)
        bf_t = jnp.stack(bs)

        bs_half_stack, _ = _sil_solve_bs_batched(
            jnp.stack(bs), h_all_t, bf_t, Dinv_t, g2_t, d_t,
            dt_half, imag_time, sil_b_krylov_dim,
        )
        bs_half = [bs_half_stack[s] for s in range(K)]

        # ── Corrector: full step with midpoint mean fields ──────────────
        Dinv_h, d_h = _batched_density(mctdh, C_half, state_dtype)
        g2_h = _build_g2_all(mctdh, bs_half, state_dtype)
        bf_h = bs_half_stack  # frozen = b(t+dt/2)

        bs_new_stack, _ = _sil_solve_bs_batched(
            jnp.stack(bs), h_all_half, bf_h, Dinv_h, g2_h, d_h,
            dt, imag_time, sil_b_krylov_dim,
        )
        bs_new = [bs_new_stack[s] for s in range(K)]

        h_new = mctdh._get_h_current(t + dt)
        h_MO_new = [mctdh.transform_h(h_new[s], bs_new[s])[1] for s in range(K)]
        g_MO_new, _ = mctdh._transform_all_g(bs_new, full=False)
        C_new, _ = _sil_solve_C(
            mctdh, C_half, h_MO_new, g_MO_new, dt_half, c_shape, sil_c_krylov_dim
        )

        return t + dt, mctdh._Cb_to_y(C_new, bs_new)

    return integrate
