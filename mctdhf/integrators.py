"""JAX-compatible integrators for MCTDHF.

These integrators are designed to work with newer JAX versions that have
stricter autodiff requirements.
"""

import jax
import jax.numpy as jnp
from functools import partial
from opt_einsum import contract


def get_crank_nicolson_solver(rhs, dt, max_iter=100, tol=1e-6):
    """Create a Crank-Nicolson integrator using fixed-point iteration.

    This implementation avoids jax.scipy.sparse.linalg.cg which has
    compatibility issues with newer JAX versions when used with complex
    nested operations.

    Args:
        rhs: Right-hand side function rhs(t, y)
        dt: Time step
        max_iter: Maximum iterations for fixed-point solver
        tol: Convergence tolerance

    Returns:
        integrate: Function integrate(t, y) -> (t_new, y_new)
    """

    @jax.jit
    def integrate(t, y):
        # Crank-Nicolson: y_{n+1} = y_n + dt/2 * (f(t_n, y_n) + f(t_{n+1}, y_{n+1}))
        # Rearranged: y_{n+1} - dt/2 * f(t_{n+1}, y_{n+1}) = y_n + dt/2 * f(t_n, y_n)

        t_new = t + dt

        # Compute the explicit part (known)
        f_n = rhs(t, y)
        b = y + (dt / 2) * f_n

        # Fixed-point iteration to solve: x = b + dt/2 * f(t_new, x)
        # Using simple iteration (Picard iteration)
        def body_fn(carry):
            x_old, _, iter_count = carry
            f_new = rhs(t_new, x_old)
            x_new = b + (dt / 2) * f_new
            diff = jnp.max(jnp.abs(x_new - x_old))
            return x_new, diff, iter_count + 1

        def cond_fn(carry):
            _, diff, iter_count = carry
            return (diff > tol) & (iter_count < max_iter)

        # Initial guess: explicit Euler step
        x_init = b + (dt / 2) * rhs(t_new, y)

        # Fixed-point iteration using lax.while_loop
        x_final, _, _ = jax.lax.while_loop(cond_fn, body_fn, (x_init, jnp.inf, 0))

        return t_new, x_final

    return integrate


def get_crank_nicolson_fixed_iter(rhs, dt, num_iter=3):
    """Create a Crank-Nicolson integrator with fixed number of iterations.

    This is simpler and avoids while_loop convergence checks.
    Uses a fixed number of Picard iterations.

    Args:
        rhs: Right-hand side function rhs(t, y)
        dt: Time step
        num_iter: Number of fixed-point iterations (default: 3)

    Returns:
        integrate: Function integrate(t, y) -> (t_new, y_new)
    """

    @jax.jit
    def integrate(t, y):
        t_new = t + dt

        # Compute the explicit part
        f_n = rhs(t, y)
        b = y + (dt / 2) * f_n

        # Fixed number of Picard iterations
        def iter_fn(x, _):
            f_new = rhs(t_new, x)
            return b + (dt / 2) * f_new, None

        # Start with explicit Euler guess
        x = b + (dt / 2) * rhs(t_new, y)

        # Apply fixed number of iterations using lax.scan
        x_final, _ = jax.lax.scan(iter_fn, x, None, length=num_iter)

        return t_new, x_final

    return integrate


def get_euler_solver(rhs, dt):
    """Simple explicit Euler integrator.

    Args:
        rhs: Right-hand side function rhs(t, y)
        dt: Time step

    Returns:
        integrate: Function integrate(t, y) -> (t_new, y_new)
    """

    @jax.jit
    def integrate(t, y):
        return t + dt, y + dt * rhs(t, y)

    return integrate


def get_rk4_solver(rhs, dt):
    """4th-order Runge-Kutta integrator.

    Args:
        rhs: Right-hand side function rhs(t, y)
        dt: Time step

    Returns:
        integrate: Function integrate(t, y) -> (t_new, y_new)
    """

    @jax.jit
    def integrate(t, y):
        k1 = rhs(t, y)
        k2 = rhs(t + dt / 2, y + dt / 2 * k1)
        k3 = rhs(t + dt / 2, y + dt / 2 * k2)
        k4 = rhs(t + dt, y + dt * k3)

        return t + dt, y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return integrate


# ── Generic single-block kernels ──────────────────────────────────────────
#
# Every CMF sub-solver below is the same handful of linear-ODE propagators
# applied to one block of the state (the CI tensor, one species' orbital
# matrix, and — in ML-MCTDHF — one species' SPF matrix). The propagators are
# factored out here so the ML integrators reuse them verbatim rather than
# growing a second copy; the sub-solvers are thin loops over these.
#
# Each kernel takes the block's linear RHS as a callable and the block itself,
# and returns the propagated block with the same shape and dtype.


def _cn_linear(rhs_fn, x0, dt, tol, maxiter):
    """Crank–Nicolson step of the linear ODE ``ẋ = rhs_fn(x)``.

    Solves ``x - dt/2 · rhs_fn(x) = x0 + dt/2 · rhs_fn(x0)`` with BiCGSTAB,
    starting from the explicit-Euler guess ``x0 + dt · rhs_fn(x0)``.

    Not unitary when the generator is non-Hermitian (the Cayley map only is
    for skew-Hermitian generators), so the caller owns re-gauging.
    """
    shape = x0.shape
    dot_0 = rhs_fn(x0)
    b_vec = x0 + (dt / 2) * dot_0

    def matvec(x_flat):
        x = x_flat.reshape(shape)
        fx = rhs_fn(x)
        return (x - (dt / 2) * fx).reshape(-1)

    guess = (x0 + dt * dot_0).reshape(-1)
    x_new_flat, _ = jax.scipy.sparse.linalg.bicgstab(
        matvec,
        b_vec.reshape(-1),
        x0=guess,
        tol=tol,
        maxiter=maxiter,
    )
    return x_new_flat.reshape(shape)


def _lower(tree):
    """complex128 -> complex64 and float64 -> float32 on every leaf; integer
    leaves (index tables) and non-array leaves pass through.

    Same as ``MLMCTDHF_n_species_ITE_opt._lower``.
    """
    def cast(x):
        if not hasattr(x, "dtype"):
            return x
        if x.dtype == jnp.complex128:
            return x.astype(jnp.complex64)
        if x.dtype == jnp.float64:
            return x.astype(jnp.float32)
        return x
    return jax.tree_util.tree_map(cast, tree)


def _cn_defect_linear(rhs_hi, rhs_lo, x0, dt, tol, maxiter, lo_dtype,
                      project=None):
    """Mixed-precision Crank–Nicolson step of ``ẋ = rhs_fn(x)``, defect form.

    ``x1 = x0 + d`` with ``(I - dt/2 · L_lo) d = dt · L_hi x0``: algebraically
    ``_cn_linear``'s ``(I - dt/2 L) x1 = (I + dt/2 L) x0``, written for the
    increment so the residual (``rhs_hi``, at ``x0.dtype``) and the BiCGSTAB
    solve (``rhs_lo``, at ``lo_dtype``) can run at different precisions.
    Same as ``MLMCTDHF_n_species_ITE_opt._cn_defect``.

    Like ``_cn_linear``, the sign convention (``-A·x`` in ITE, ``-i·A·x`` in
    RTE) lives in the RHS callables, so real and imaginary time go through
    the same code. The inner ``tol`` is relative to ``d``: in imaginary time
    ``d → 0`` at a fixed point and the tolerance only sets the rate; in real
    time ``d`` is ``O(dt)`` every step, so the per-step error is roughly
    ``max(tol, eps(lo_dtype)) · |d|``.

    Trace-time guards, because a silent promotion would make "mixed" x64
    under another name (one float64 leaf left in the frozen operators
    promotes every matvec back to complex128), and a non-lower ``lo_dtype``
    would make it a slower copy of ``_cn_linear``:

    * ``lo_dtype`` must be strictly narrower than ``x0.dtype``;
    * ``rhs_hi`` must return ``x0.dtype``;
    * ``rhs_lo`` and the solver matvec must return ``lo_dtype``.

    ``project`` (optional, applied to ``d`` at the state precision) is the
    frozen ``(1 - P_f)``. The exact CN increment lies in its range, since
    every term of the orbital and SPF RHS does; the low-precision solve
    applies ``P_f`` only to ``eps(lo_dtype)`` relative accuracy, and with a
    stiff frozen operator (``D^-1`` ~ 1/eps on weakly occupied orbitals) that
    leaks an in-span component into ``d`` which breaks orthonormality at
    FIRST order and compounds step to step (measured on the four-well MF
    layout: the active orbitals' ``|b^dagger b - 1|`` grew 3-5x per step and
    reached NaN within 10 steps). Re-projecting ``d`` with the high-precision
    projector restores the exact structure; unlike a trailing
    orthonormalisation it does not touch the in-span part of the state.
    """
    shape = x0.shape
    lo_dtype = jnp.dtype(lo_dtype)
    if lo_dtype.itemsize >= x0.dtype.itemsize:
        raise TypeError(f"lo_dtype {lo_dtype} is not lower precision than "
                        f"the state dtype {x0.dtype}")
    probe_hi = jax.eval_shape(rhs_hi, jax.ShapeDtypeStruct(shape, x0.dtype))
    if probe_hi.dtype != x0.dtype:
        raise TypeError(f"high-precision RHS returns {probe_hi.dtype}, "
                        f"expected {x0.dtype}")
    probe = jax.eval_shape(rhs_lo, jax.ShapeDtypeStruct(shape, lo_dtype))
    if probe.dtype != lo_dtype:
        raise TypeError(f"low-precision RHS returns {probe.dtype}, "
                        f"expected {lo_dtype}")
    r = dt * rhs_hi(x0)

    def matvec(d_flat):
        d = d_flat.reshape(shape)
        return (d - (dt / 2) * rhs_lo(d)).reshape(-1)

    probe_mv = jax.eval_shape(matvec, jax.ShapeDtypeStruct((x0.size,), lo_dtype))
    if probe_mv.dtype != lo_dtype:
        raise TypeError(f"low-precision matvec returns {probe_mv.dtype}, "
                        f"expected {lo_dtype} (is dt a float64 array?)")

    r_lo = r.astype(lo_dtype).reshape(-1)
    d_flat, _ = jax.scipy.sparse.linalg.bicgstab(
        matvec, r_lo, x0=r_lo, tol=tol, maxiter=maxiter)
    d = d_flat.reshape(shape).astype(x0.dtype)
    if project is not None:
        d = project(d)
    return x0 + d


def _expm_dense_linear(rhs_fn, x0, dt, imag_time, compute_x64=False):
    """Exact step of ``ẋ = rhs_fn(x)`` via a dense matrix exponential.

    ``rhs_fn`` is assumed linear in ``x`` and to carry the solver's sign
    convention, i.e. it returns ``-A·x`` (ITE) or ``-i·A·x`` (RTE). The dense
    generator is built by ``vmap``-ing ``rhs_fn`` over the standard basis of
    the flattened block, then ``expm`` is applied to ``x0``.

    ``compute_x64`` promotes only the precision-critical part — the
    non-Hermitian ``expm`` and its application — to complex128, leaving the
    (much larger) basis-vector build in the native bulk dtype.

    Exact in the linearised substep; cost is ``prod(x0.shape)`` RHS
    evaluations plus one dense ``expm``, so it is the right choice only while
    that dimension is small (a few hundred).
    """
    shape = x0.shape
    x_flat = x0.reshape(-1)
    n = x_flat.shape[0]

    def apply_once(x):
        return rhs_fn(x).reshape(-1)

    I_cols = jnp.eye(n, dtype=x0.dtype).reshape((n,) + shape)
    J_rows = jax.vmap(apply_once)(I_cols)   # row k = J · e_k
    J_mat = J_rows.T                         # columns = J · e_k → J as matrix

    # Extract A from the rhs sign convention; form the expm argument.
    if imag_time:
        arg = -dt * (-J_mat)
    else:
        arg = -1j * dt * (1j * J_mat)

    if compute_x64:
        expA = jax.scipy.linalg.expm(arg.astype(jnp.complex128))
        x_new_flat = expA @ x_flat.astype(jnp.complex128)
    else:
        expA = jax.scipy.linalg.expm(arg)
        x_new_flat = expA @ x_flat

    return x_new_flat.reshape(shape).astype(x0.dtype)


def _dop_linear(rhs_fn, x0, dt, n_substeps=1):
    """Fixed-step Dormand–Prince 5(4) step of ``ẋ = rhs_fn(x)``.

    The CMF sub-problems have a FROZEN generator over the substep, so they
    are autonomous linear ODEs — the case an explicit Runge–Kutta method
    handles best. Makes no Hermiticity assumption, builds no matrix and
    solves no linear system; being explicit it is stability-limited by
    ``dt · ρ(A) ≲ 3``, which ``n_substeps`` relieves.
    """
    h = dt / n_substeps

    def one_substep(x, _):
        ks = []
        for i in range(7):
            xi = x
            for j, aij in enumerate(_DP_A[i]):
                if aij != 0.0:
                    xi = xi + h * aij * ks[j]
            ks.append(rhs_fn(xi))
        out = x
        for coef, k in zip(_DP_B5, ks):
            if coef != 0.0:
                out = out + h * coef * k
        return out, None

    x_out, _ = jax.lax.scan(one_substep, x0, None, length=n_substeps)
    return x_out


def _cn_solve_C(
    mctdh, C, h_MO_list, g_MO_dict, dt_sub, c_shape, solver_tol, max_solver_iter
):
    """CN-solve the CI sub-problem (linear in C, orbitals frozen).

    Solves: x - dt_sub/2 · f_C(x) = C + dt_sub/2 · f_C(C).
    """
    # Integrals are frozen for the whole solve: contract them with the
    # excitation tensors once, apply the small determinant-space operators
    # in every matvec.
    ops = mctdh.dCI.precompute_ci_ops(h_MO_list, g_MO_dict)

    def rhs_C(x):
        sigma = mctdh.dCI.apply_ci_ops(ops, x)
        return -sigma if mctdh.imag_time else -1j * sigma

    return _cn_linear(rhs_C, C, dt_sub, solver_tol, max_solver_iter)


def _cn_solve_bs(
    mctdh,
    bs_start,
    h_current,
    D_invs,
    ds,
    bs_frozen,
    g2_frozen,
    dt_sub,
    b_shapes,
    solver_tol,
    max_solver_iter,
):
    """CN-solve all orbital sub-problems (linear in b_s, P and CI frozen).

    Solves per species: x - dt_sub/2 · f_b(x) = b_s + dt_sub/2 · f_b(b_s).
    `g2_frozen` is the per-substep _precompute_frozen_g2(bs_frozen) result.
    """
    bs_new = list(bs_start)
    for s in range(mctdh.num_species):

        def rhs_b(x, _s=s):
            return mctdh._rhs_b_linear(
                x, _s, h_current[_s], D_invs[_s], ds, bs_frozen, g2_frozen
            )

        bs_new[s] = _cn_linear(
            rhs_b, bs_start[s], dt_sub, solver_tol, max_solver_iter
        )
    return bs_new


def _cn_mixed_solve_bs(
    mctdh,
    bs_start,
    h_current,
    D_invs,
    ds,
    bs_frozen,
    g2_frozen,
    dt_sub,
    b_shapes,
    solver_tol,
    max_solver_iter,
    lo_dtype,
):
    """Mixed-precision (defect-form) CN solve of all orbital sub-problems.

    As ``_cn_solve_bs``, with the residual at the state precision and the
    BiCGSTAB solve on the frozen tuple ``(h, D_inv, ds, bs, g2)`` lowered to
    ``lo_dtype`` (``_cn_defect_linear``). No trailing orthonormalisation: in
    real time that is not gauge-exact; the caller owns re-gauging.
    """
    bs_new = list(bs_start)
    for s in range(mctdh.num_species):
        frozen = (h_current[s], D_invs[s], ds, bs_frozen, g2_frozen)
        frozen_lo = _lower(frozen)

        def rhs_hi(x, _s=s, _f=frozen):
            return mctdh._rhs_b_linear(x, _s, *_f)

        def rhs_lo(x, _s=s, _f=frozen_lo):
            return mctdh._rhs_b_linear(x, _s, *_f)

        def project(X, _bf=bs_frozen[s]):
            return X - _bf @ (_bf.conj().T @ X)

        bs_new[s] = _cn_defect_linear(
            rhs_hi, rhs_lo, bs_start[s], dt_sub, solver_tol, max_solver_iter,
            lo_dtype, project=project,
        )
    return bs_new


def _sil_expm_coeffs(alphas, betas, dt, imag_time, vec_dtype, sign=-1.0):
    """First column of exp(sign*i*dt*T) (RTE) / exp(sign*dt*T) (ITE), T the
    Lanczos tridiagonal. T and the exponential are formed in double regardless
    of the vector dtype: in complex64 the k x k expm is non-unitary by ~5e-7
    per step, a norm bias that dominates the CI propagation at large CI blocks
    (measured on the three-electron four-well system). The coefficients are cast back to `vec_dtype`
    for the reconstruction with the stored basis."""
    f64 = jax.dtypes.canonicalize_dtype(jnp.float64)
    a = alphas.astype(f64)
    b = betas[:-1].astype(f64)
    T = jnp.diag(a) + jnp.diag(b, 1) + jnp.diag(b, -1)
    if imag_time:
        expT = jax.scipy.linalg.expm(sign * dt * T)
    else:
        expT = jax.scipy.linalg.expm(sign * 1j * dt * T)
    return expT[:, 0].astype(vec_dtype)


def _lanczos_basis(apply_H, v0, krylov_dim):
    """Lanczos recursion with full re-orthogonalization.

    Returns ``(V, alphas, betas)``: the orthonormal Krylov basis as a
    ``(krylov_dim, dim)`` matrix and the tridiagonal coefficients.

    ``V`` is carried in the scan state so each step can orthogonalize against
    every previously computed basis vector — essential in float32, where
    classical Lanczos loses orthogonality within a few iterations. Rows beyond
    the current step are still zero, so their dot products vanish and no
    masking is needed.
    """
    dim = v0.shape[0]
    vec_dtype = v0.dtype
    real_dtype = v0.real.dtype
    V_init = jnp.zeros((krylov_dim, dim), dtype=vec_dtype)

    def lanczos_step(carry, j):
        V, v_prev, v_curr, beta_prev = carry
        # Store v_curr as basis vector j (at step start, not end,
        # to avoid overwriting row 0 on the last iteration)
        V = V.at[j].set(v_curr)
        w = apply_H(v_curr).astype(vec_dtype)
        alpha = jnp.real(jnp.dot(v_curr.conj(), w)).astype(real_dtype)
        w = w - alpha * v_curr - beta_prev * v_prev
        overlaps = jnp.dot(V.conj(), w)  # (krylov_dim,)
        w = w - jnp.dot(overlaps, V)  # (dim,)
        beta = jnp.sqrt(jnp.real(jnp.dot(w.conj(), w))).astype(real_dtype)
        safe_beta = jnp.where(beta > 1e-15, beta, 1.0)
        v_next = w / safe_beta
        return (V, v_curr, v_next, beta), (alpha, beta)

    init = (V_init, jnp.zeros_like(v0), v0, jnp.array(0.0, dtype=real_dtype))
    (V_final, _, _, _), (alphas, betas) = jax.lax.scan(
        lanczos_step, init, jnp.arange(krylov_dim), length=krylov_dim
    )
    return V_final, alphas, betas


def _lanczos_expm(apply_H, v_flat, dt, imag_time, krylov_dim=10, sign=-1.0):
    """Short Iterative Lanczos (SIL) propagation of one flat block.

    Computes ``exp(sign·i·dt·H)·v`` (RTE) / ``exp(sign·dt·H)·v`` (ITE) by
    building a Krylov subspace with :func:`_lanczos_basis` and exponentiating
    the small tridiagonal matrix.

    ``apply_H`` maps a flat vector to a flat vector and must be the HERMITIAN
    generator, not the signed RHS: callers extract it from ``_rhs_*_linear``
    by negating (ITE) or multiplying by ``1j`` (RTE).

    Returns ``(v_new, sil_error)`` with ``sil_error`` the relative Saad bound
    ``|beta_last · coeff_last|``.
    """
    norm_v = jnp.linalg.norm(v_flat)
    safe_norm = jnp.where(norm_v > 0, norm_v, 1.0)
    v0 = v_flat / safe_norm

    V_final, alphas, betas = _lanczos_basis(apply_H, v0, krylov_dim)

    # T and the exponential are formed in double regardless of vector dtype
    # (see _sil_expm_coeffs).
    coeffs = _sil_expm_coeffs(alphas, betas, dt, imag_time, v_flat.dtype,
                              sign=sign)

    # SIL Krylov truncation error (Saad bound), relative
    sil_error = jnp.abs(betas[-1] * coeffs[-1])

    # Reconstruct: v_new = ||v|| * V^T @ coeffs (V is orthonormal)
    v_new = safe_norm * jnp.dot(coeffs, V_final)
    v_new = jnp.where(norm_v > 0, v_new, v_flat)
    return v_new, sil_error


def _lanczos_expm_bidi(apply_H, v_flat, dt, imag_time, krylov_dim=10):
    """Forward and backward SIL propagation from ONE Lanczos decomposition.

    Returns ``(v_fwd, v_bwd, sil_error_fwd)`` where ``v_fwd`` uses
    ``sign = -1`` and ``v_bwd`` uses ``sign = +1``. The extra cost over
    :func:`_lanczos_expm` is one additional small ``krylov_dim²`` matrix
    exponential; the backward vector is what the adaptive integrator compares
    against the input to estimate the CMF splitting error.
    """
    norm_v = jnp.linalg.norm(v_flat)
    safe_norm = jnp.where(norm_v > 0, norm_v, 1.0)
    v0 = v_flat / safe_norm

    V_final, alphas, betas = _lanczos_basis(apply_H, v0, krylov_dim)

    coeffs_fwd = _sil_expm_coeffs(alphas, betas, dt, imag_time,
                                  v_flat.dtype, sign=-1.0)
    coeffs_bwd = _sil_expm_coeffs(alphas, betas, dt, imag_time,
                                  v_flat.dtype, sign=+1.0)

    sil_error = jnp.abs(betas[-1] * coeffs_fwd[-1])

    v_fwd = safe_norm * jnp.dot(coeffs_fwd, V_final)
    v_fwd = jnp.where(norm_v > 0, v_fwd, v_flat)

    v_bwd = safe_norm * jnp.dot(coeffs_bwd, V_final)
    v_bwd = jnp.where(norm_v > 0, v_bwd, v_flat)

    return v_fwd, v_bwd, sil_error


def _sil_solve_C(mctdh, C, h_MO_list, g_MO_dict, dt, c_shape, krylov_dim=10):
    """Short Iterative Lanczos (SIL) propagation of CI coefficients.

    Computes C(t+dt) = exp(dt*A)*C(t) where A is the CI Hamiltonian operator,
    by building a Krylov subspace via the Lanczos algorithm and exponentiating
    the resulting small tridiagonal matrix.

    SIL is particularly efficient for the CI propagation because:
    - The CI Hamiltonian is Hermitian, so Lanczos produces a real tridiagonal
    - A small Krylov dimension (5-15) suffices for typical MCTDH time steps
    - The matrix exponential is unitary (norm-preserving) for RTE

    Uses full re-orthogonalization (V carried in scan state) for numerical
    stability in float32/complex64 arithmetic.
    """
    # Integrals are frozen across the Lanczos recursion: precontract them
    # with the excitation tensors once (E_pq·h, E_pqrs·g, T1/Op_12) so each
    # Lanczos step only applies the small determinant-space operators.
    # XLA does NOT hoist this out of the scan by itself (verified 2026-07).
    ops = mctdh.dCI.precompute_ci_ops(h_MO_list, g_MO_dict)

    # Lanczos on H (Hermitian): sigma = H*C
    def apply_H(v_flat):
        return mctdh.dCI.apply_ci_ops(ops, v_flat.reshape(c_shape)).reshape(-1)

    c_new_flat, sil_error = _lanczos_expm(
        apply_H, C.reshape(-1), dt, mctdh.imag_time, krylov_dim
    )
    return c_new_flat.reshape(c_shape), sil_error


def _sil_solve_C_bidi(mctdh, C, h_MO_list, g_MO_dict, dt, c_shape, krylov_dim=10):
    """Bidirectional SIL propagation: forward and backward from a single Lanczos decomposition.

    Computes both C_fwd = exp(-i*dt*H)*C and C_bwd = exp(+i*dt*H)*C using
    the same Krylov subspace. The extra cost over _sil_solve_C is one additional
    small (krylov_dim x krylov_dim) matrix exponential.

    Returns:
        (C_fwd, C_bwd, sil_error_fwd)
    """
    # Loop-invariant integral precontraction (see _sil_solve_C)
    ops = mctdh.dCI.precompute_ci_ops(h_MO_list, g_MO_dict)

    def apply_H(v_flat):
        return mctdh.dCI.apply_ci_ops(ops, v_flat.reshape(c_shape)).reshape(-1)

    c_fwd_flat, c_bwd_flat, sil_error = _lanczos_expm_bidi(
        apply_H, C.reshape(-1), dt, mctdh.imag_time, krylov_dim
    )
    return c_fwd_flat.reshape(c_shape), c_bwd_flat.reshape(c_shape), sil_error


def _ci_apply_fn(mctdh, h_MO_list, g_MO_dict):
    """``σ = H|C⟩`` as a callable, for ``_diag_solve_C``.

    Accepts tensors of shape ``(..., n_dets_0, ...)`` — any leading batch
    shape — which is what lets ``_diag_solve_C`` build the dense CI
    Hamiltonian by applying it to a batch of unit vectors.

    This wraps ``get_sigma`` rather than hoisting the integral
    precontraction out of it: ``_diag_solve_C`` applies H exactly once, so
    there is nothing to hoist, and splitting ``get_sigma`` into its two
    jitted halves changes the HLO the GPU backend fuses, which moves the last
    bit of a propagated trajectory (``tests/test_refactor_pins.py``).
    """
    return lambda X: mctdh.dCI.get_sigma(X, h_MO_list, g_MO_dict)


def _diag_solve_C(apply_H, C, dt, imag_time, c_shape):
    """Exact top-block propagator for the linearised CMF substep.

    ``apply_H`` applies the (Hermitian) top-block Hamiltonian to a tensor of
    shape ``(..., *c_shape[1:])``; the dense matrix is recovered by applying
    it to the identity. Standard MCTDHF passes ``_ci_apply_fn`` (determinant
    space); ML-MCTDHF passes ``apply_ci_ops(top_ops, ·)`` in SPF space, where
    ``prod(num_spfs)`` is small enough that this is the cheap choice.

    ITE: replace C by the lowest ``num_states`` eigenvectors of H_CI via a
    dense eigendecomposition. The input C is used only as a dtype/shape
    template — the result is the exact state-averaged MCSCF CI update given
    the current orbitals (so the dt argument is irrelevant in this branch).

    RTE: form ``expm(-i*dt*H_CI) @ C`` via a dense matrix exponential. This
    is autodiff-clean: unlike the eigendecomposition route
    (``U @ diag(exp(-i*dt*λ)) @ U†``), ``expm``'s reverse-mode VJP is built
    from matmuls and a linear solve — it has no ``1/(λ_i-λ_j)`` terms, so it
    does not produce NaN gradients when H_CI has degenerate eigenvalues
    (verified). Forward value and unitarity match the eigh route
    to machine precision at the dt·‖H_CI‖ of a typical step, and expm is the
    faster of the two on GPU for the forward pass.

    Cheap when prod(num_dets) is small (≲ a few thousand). For larger CI
    dimensions prefer SIL.

    Returns ``(C_new, error)`` to match the ``_sil_solve_C`` signature; the
    error estimate is identically zero (exact in the linearised substep).
    """
    num_states = c_shape[0]
    det_shape = c_shape[1:]
    n_dets_tot = 1
    for n in det_shape:
        n_dets_tot *= int(n)

    # Build the dense CI Hamiltonian by applying H to the identity in the
    # determinant (or SPF) basis. apply_H uses '...' einsum strings so any
    # leading batch shape is accepted.
    eye = jnp.eye(n_dets_tot, dtype=C.dtype).reshape((n_dets_tot,) + det_shape)
    sigma = apply_H(eye)
    H_CI = sigma.reshape(n_dets_tot, n_dets_tot)
    # Hermitise to clean up floating-point noise.
    H_CI = 0.5 * (H_CI + H_CI.conj().T)

    if imag_time:
        # Lowest num_states eigenvectors as the new state-averaged CI vectors.
        # eigvecs columns -> states; transpose so the leading axis is `state`.
        #
        # NOTE the returned eigenvectors carry an ARBITRARY phase per state:
        # eigh pins it by the backend's internal normalisation, and that
        # choice moves discontinuously under a perturbation of the input far
        # below the eigenvalue gap (~1e-8 rad for a 1e-13 input change, on
        # cuSOLVER). This is harmless in a relaxation — a global phase per
        # state leaves the energy, the densities and every |overlap|
        # unchanged — but it means the coefficients themselves must never be
        # compared elementwise between two implementations or two backends.
        # tests/test_ml_exactness.py removes the phase before comparing.
        _, eigvecs = jnp.linalg.eigh(H_CI)  # ascending eigenvalues
        C_new = eigvecs[:, :num_states].T.reshape(c_shape)
    else:
        # Exact unitary propagation via the matrix exponential. expm keeps
        # the reverse pass NaN-free at degenerate H_CI eigenvalues; the eigh
        # route's eigenvector VJP would divide by (λ_i - λ_j). The
        # contraction `C_flat @ U` reproduces the eigh route's
        # `C_flat @ V @ diag(exp(-i dt λ)) @ V†` exactly — note NO transpose
        # on U: H_CI is complex-Hermitian (not symmetric), so `C_flat @ U.T`
        # would propagate with the conjugated Hamiltonian.
        C_flat = C.reshape(num_states, n_dets_tot)
        U = jax.scipy.linalg.expm(-1j * dt * H_CI)
        C_new = (C_flat @ U).reshape(c_shape)

    return C_new, jnp.zeros((), dtype=H_CI.real.dtype)


def _sil_solve_bs(
    mctdh,
    bs_start,
    h_current,
    D_invs,
    ds,
    bs_frozen,
    g2_frozen,
    dt,
    b_shapes,
    krylov_dim=20,
):
    """SIL (Short Iterative Lanczos) propagation of orbital matrices.

    For each species s, vectorizes b_s and applies the Lanczos algorithm
    to the linear orbital Hamiltonian operator A (where db/dt = -iAb for RTE),
    then computes b(t+dt) = exp(-i*dt*A) * b(t).

    This handles stiff DVR grids (large kinetic energy eigenvalues) naturally
    because the Krylov subspace captures the extreme eigenvalues and the matrix
    exponential gives the exact phase for all captured modes.

    `g2_frozen` is the per-substep _precompute_frozen_g2(bs_frozen) result —
    loop-invariant across the Lanczos matvecs, so it is built once by the
    caller rather than inside apply_A.
    """
    bs_new = list(bs_start)
    sil_errors = []
    for s in range(mctdh.num_species):
        b_s = bs_start[s]

        def apply_A(v_flat, _s=s):
            """Apply orbital Hamiltonian A such that db/dt = -iAb (RTE)."""
            b = v_flat.reshape(b_shapes[_s])
            rhs = mctdh._rhs_b_linear(
                b, _s, h_current[_s], D_invs[_s], ds, bs_frozen, g2_frozen
            )
            # _rhs_b_linear returns -1j*A*b (RTE) or -A*b (ITE)
            # We need A*b for the Lanczos (Hermitian operator)
            if mctdh.imag_time:
                return (-rhs).reshape(-1)
            else:
                return (1j * rhs).reshape(-1)

        b_new_flat, sil_error_s = _lanczos_expm(
            apply_A, b_s.reshape(-1), dt, mctdh.imag_time, krylov_dim
        )
        sil_errors.append(sil_error_s)
        bs_new[s] = b_new_flat.reshape(b_shapes[s])

    return bs_new, jnp.max(jnp.array(sil_errors))


def _diag_solve_bs(
    mctdh,
    bs_start,
    h_current,
    D_invs,
    ds,
    bs_frozen,
    g2_frozen,
    dt,
    b_shapes,
    compute_x64=False,
):
    """Exact orbital propagator via dense matrix exponential of A_s.

    When ``compute_x64`` is set, the dense A_s build (the basis-vector
    matvecs) stays in the native bulk dtype, but the precision-critical step
    — the non-Hermitian matrix exponential and its application to b — is
    promoted to complex128, then cast back. This stabilises float32 CMF runs
    without paying complex128 cost throughout the build.

    For each species s, builds A_s as a dense (N_s, N_s) matrix where
    N_s = n_spatial_s * M_s by applying _rhs_b_linear to the standard
    basis vectors of the flat orbital space, then propagates exactly:

        b_new_flat = jax.scipy.linalg.expm(factor · dt · A_s) @ b_flat

    with `factor = -1.0` (ITE) or `-1j` (RTE).

    The propagator is exact in the linearised (frozen-coefficient) CMF
    substep ODE. Sign convention follows _sil_solve_bs: extract A_s from
    _rhs_b_linear by negating (ITE) or multiplying by 1j (RTE). Under the
    current (1-P)·(h+W)·D⁻¹ MCTDHF gauge A_s is non-Hermitian (asymmetry
    ratio ~0.5 on the standard test system), so eigh is not applicable;
    expm handles the general case at modest extra cost.

    Cheap when n_spatial · n_orb is small (≲ a few hundred per species);
    for larger orbital dimensions prefer SIL or CN.

    Returns (bs_new, zero_error) to match _sil_solve_bs's tuple shape.
    """
    bs_new = list(bs_start)
    err_dtype = bs_start[0].real.dtype

    for s in range(mctdh.num_species):

        def rhs_b(b, _s=s):
            return mctdh._rhs_b_linear(
                b, _s, h_current[_s], D_invs[_s], ds, bs_frozen, g2_frozen
            )

        bs_new[s] = _expm_dense_linear(
            rhs_b, bs_start[s], dt, mctdh.imag_time, compute_x64=compute_x64
        )

    return bs_new, jnp.zeros((), dtype=err_dtype)


def _bs_modified_midpoint(rhs_fn, y0, dt, n_steps):
    """Modified midpoint method with n_steps subdivisions and smoothing.

    Computes an O(h^2) approximation to y(t+dt) where h = dt/n_steps.
    Used as the base method for Bulirsch-Stoer extrapolation.
    """
    h = dt / n_steps

    z_prev = y0
    z_curr = y0 + h * rhs_fn(y0)

    def body(_, carry):
        z_prev, z_curr = carry
        z_next = z_prev + 2 * h * rhs_fn(z_curr)
        return (z_curr, z_next)

    z_prev, z_curr = jax.lax.fori_loop(1, n_steps, body, (z_prev, z_curr))

    # Smoothing step
    return 0.5 * (z_curr + z_prev + h * rhs_fn(z_curr))


# Dormand-Prince 5(4) tableau, used by _dop_solve_bs.
_DP_C = (0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0)
_DP_A = (
    (),
    (1/5,),
    (3/40, 9/40),
    (44/45, -56/15, 32/9),
    (19372/6561, -25360/2187, 64448/6561, -212/729),
    (9017/3168, -355/33, 46732/5247, 49/176, -5103/18656),
    (35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84),
)
_DP_B5 = (35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0)


def _dop_solve_bs(
    mctdh,
    bs_start,
    h_current,
    D_invs,
    ds,
    bs_frozen,
    g2_frozen,
    dt,
    b_shapes,
    n_substeps=1,
):
    """Orbital substep by fixed-step Dormand-Prince 5(4).

    The CMF orbital sub-problem db/dt = A_s b has A_s FROZEN over the substep,
    so it is an autonomous linear ODE — the case an explicit Runge-Kutta method
    handles best. Cost per species is ``7 * n_substeps`` evaluations of
    ``_rhs_b_linear``, against:

      * ``_diag_solve_bs``: ``n_spatial * norb`` evaluations to build A_s
        densely (384 at norb = 6, n_spatial = 64) plus a dense expm — exact in
        the linearised substep, but the most expensive orbital solver by far;
      * ``_cn_solve_bs``: one BiCGSTAB solve — cheap, but the Cayley map is
        NOT unitary when A_s is non-Hermitian (which it is, under the
        (1-P)(h+W)D^-1 gauge), so b^H b drifts and regauge becomes mandatory;
        and for stiff modes its amplification factor tends to -1 rather than
        tracking exp, i.e. it aliases rather than diverging — a wrong answer
        that looks stable.

    DOP makes no Hermiticity assumption, builds no matrix and solves no linear
    system. Being explicit it is stability-limited by ``dt * rho(A_s) <~ 3``, so
    ``n_substeps`` splits the substep when the frozen generator is stiff; that
    is the same device ``_cn_solve_bs`` uses via ``n_orbital_substeps`` in the
    half-step integrator.

    Returns ``(bs_new, zero_error)`` to match the ``_sil_solve_bs`` /
    ``_diag_solve_bs`` tuple shape.
    """
    bs_new = []
    for s, b_s in enumerate(bs_start):

        def rhs(b, _s=s):
            return mctdh._rhs_b_linear(
                b, _s, h_current[_s], D_invs[_s], ds, bs_frozen, g2_frozen
            )

        bs_new.append(_dop_linear(rhs, b_s, dt, n_substeps))
    return bs_new, jnp.zeros((), dtype=bs_start[0].real.dtype)


def _bs_linear(rhs_fn, x0, dt, bs_sequence=(2, 4, 6, 8, 12, 16)):
    """Bulirsch–Stoer step of ``ẋ = rhs_fn(x)``.

    Modified midpoint at each subdivision of ``bs_sequence``, then polynomial
    (Neville) Richardson extrapolation in ``h²``, reaching
    ``O(h^{2·len(sequence)})`` for relatively few RHS evaluations.
    """
    T_list = []
    h_sq = []
    for n in bs_sequence:
        T_list.append(_bs_modified_midpoint(rhs_fn, x0, dt, n))
        h_sq.append((dt / n) ** 2)

    # T_list[i] has error O(h_i^2); extrapolation removes successive powers
    # of h^2.
    for k in range(1, len(bs_sequence)):
        for i in range(len(bs_sequence) - 1, k - 1, -1):
            ratio = h_sq[i - k] / h_sq[i]
            T_list[i] = T_list[i] + (T_list[i] - T_list[i - 1]) / (ratio - 1)

    return T_list[-1]


def _bs_solve_bs(
    mctdh,
    bs_start,
    h_current,
    D_invs,
    ds,
    bs_frozen,
    g2_frozen,
    dt,
    b_shapes,
    bs_sequence=(2, 4, 6, 8, 12, 16),
):
    """Bulirsch-Stoer solve for all orbital sub-problems.

    Uses the modified midpoint method with Richardson extrapolation
    (polynomial, Neville's algorithm) to achieve high-order accuracy.
    The Bulirsch-Stoer method is well-suited for the orbital propagation
    because it achieves high accuracy with relatively few RHS evaluations
    compared to fixed-order methods.
    """
    bs_new = list(bs_start)
    for s in range(mctdh.num_species):

        def rhs_b(b, _s=s):
            return mctdh._rhs_b_linear(
                b, _s, h_current[_s], D_invs[_s], ds, bs_frozen, g2_frozen
            )

        bs_new[s] = _bs_linear(rhs_b, bs_start[s], dt, bs_sequence)
    return bs_new


def _compute_density_quantities(mctdh, C):
    """Compute state-averaged RDMs and regularized inverse density matrices.

    Returns (D_invs, ds) where D_invs is a list of inverse 1-RDMs per species
    and ds is a dict of state-averaged 2-RDMs.
    """
    # Average over states inside the RDM build: exact (the excitation
    # contraction is linear in ρ) and keeps the state axis out of it.
    w = mctdh.state_weights
    Ds = mctdh.dCI.get_1p_RDMs(C, weights=w)
    ds = mctdh.dCI.get_2p_RDMs(C, weights=w)

    D_invs = []
    for s in range(mctdh.num_species):
        D_inv = mctdh._invert_density_matrix(Ds[s])
        D_invs.append(D_inv)

    return D_invs, ds


def _validate_kwargs(kwargs, recognised, dict_name, propagator_name):
    """Raise ValueError if any key in `kwargs` is not in `recognised`.

    The error message lists the recognised keys for the given propagator
    so typos are easy to fix.
    """
    extras = set(kwargs) - set(recognised)
    if extras:
        raise ValueError(
            f"{dict_name} contains unrecognised key(s) {sorted(extras)} for "
            f"propagator '{propagator_name}'. Recognised keys: "
            f"{sorted(recognised) or 'none'}."
        )


_C_KWARG_KEYS = {
    "cn": ("solver_tol", "max_solver_iter"),
    "sil": ("krylov_dim",),
    "diagonalize": (),
}
# `n_orbital_substeps` sub-cycles the orbital step of either CMF integrator,
# re-freezing the projector and the mean-field partner orbitals every
# dt/n while the CI step stays at dt (see _solve_bs_subcycled).
_B_KWARG_KEYS = {
    "cn": ("solver_tol", "max_solver_iter", "n_orbital_substeps"),
    "sil": ("krylov_dim", "n_orbital_substeps"),
    "bs": ("bs_sequence", "n_orbital_substeps"),
    "diagonalize": ("orbital_x64", "n_orbital_substeps"),
    "dop": ("n_substeps", "n_orbital_substeps"),
    # Mixed-precision defect-form CN (_cn_defect_linear). Its default
    # solver_tol is 1e-6, not 1e-8: the solve runs at lo_dtype.
    "cn_mixed": ("solver_tol", "max_solver_iter", "lo_dtype",
                 "n_orbital_substeps"),
}
_B_KWARG_KEYS_CMF = _B_KWARG_KEYS
_B_KWARG_KEYS_HALF = _B_KWARG_KEYS
_B_PROPAGATORS = ("cn", "bs", "sil", "diagonalize", "dop", "cn_mixed")
_CN_MIXED_DEFAULT_TOL = 1e-6
_CN_MIXED_DEFAULT_LO_DTYPE = jnp.complex64
_C_PROPAGATORS = ("cn", "sil", "diagonalize")


def _cmf_options(c_propagator, b_propagator, c_kwargs, b_kwargs):
    """Validate and default the sub-solver options of both CMF integrators."""
    if c_propagator not in _C_PROPAGATORS:
        raise ValueError(
            f"Unknown c_propagator '{c_propagator}'. Use one of {_C_PROPAGATORS}."
        )
    if b_propagator not in _B_PROPAGATORS:
        raise ValueError(
            f"Unknown b_propagator '{b_propagator}'. Use one of {_B_PROPAGATORS}."
        )
    c_kwargs = dict(c_kwargs or {})
    b_kwargs = dict(b_kwargs or {})
    _validate_kwargs(c_kwargs, _C_KWARG_KEYS[c_propagator], "c_kwargs", c_propagator)
    _validate_kwargs(b_kwargs, _B_KWARG_KEYS[b_propagator], "b_kwargs", b_propagator)
    c_opts = {
        "solver_tol": c_kwargs.get("solver_tol", 1e-8),
        "max_solver_iter": c_kwargs.get("max_solver_iter", 100),
        "krylov_dim": c_kwargs.get("krylov_dim", 10),
    }
    b_opts = {
        "solver_tol": b_kwargs.get("solver_tol", 1e-8),
        "max_solver_iter": b_kwargs.get("max_solver_iter", 100),
        "krylov_dim": b_kwargs.get("krylov_dim", 20),
        "n_substeps": b_kwargs.get("n_substeps", 1),
        "bs_sequence": b_kwargs.get("bs_sequence", (2, 4, 6, 8, 12, 16)),
        "orbital_x64": b_kwargs.get("orbital_x64", False),
        "n_orbital_substeps": int(b_kwargs.get("n_orbital_substeps", 1)),
    }
    if b_propagator == "cn_mixed":
        b_opts["solver_tol"] = b_kwargs.get("solver_tol",
                                            _CN_MIXED_DEFAULT_TOL)
        b_opts["lo_dtype"] = b_kwargs.get("lo_dtype",
                                          _CN_MIXED_DEFAULT_LO_DTYPE)
    return c_opts, b_opts


def _solve_C(mctdh, c_propagator, C, h_MO_list, g_MO_dict, dt, c_shape, opts):
    """One linearised CI substep (orbitals frozen) with the chosen propagator."""
    if c_propagator == "sil":
        return _sil_solve_C(
            mctdh, C, h_MO_list, g_MO_dict, dt, c_shape, opts["krylov_dim"]
        )[0]
    if c_propagator == "diagonalize":
        return _diag_solve_C(
            _ci_apply_fn(mctdh, h_MO_list, g_MO_dict), C, dt,
            mctdh.imag_time, c_shape,
        )[0]
    return _cn_solve_C(
        mctdh, C, h_MO_list, g_MO_dict, dt, c_shape,
        opts["solver_tol"], opts["max_solver_iter"],
    )


def _solve_bs(
    mctdh, b_propagator, bs_start, h_current, D_invs, ds, bs_frozen, g2_frozen,
    dt, b_shapes, opts,
):
    """One linearised orbital substep (projector, partner orbitals and density
    matrices frozen) with the chosen propagator."""
    args = (mctdh, bs_start, h_current, D_invs, ds, bs_frozen, g2_frozen, dt, b_shapes)
    if b_propagator == "sil":
        return _sil_solve_bs(*args, opts["krylov_dim"])[0]
    if b_propagator == "diagonalize":
        return _diag_solve_bs(*args, compute_x64=opts["orbital_x64"])[0]
    if b_propagator == "dop":
        return _dop_solve_bs(*args, opts["n_substeps"])[0]
    if b_propagator == "bs":
        return _bs_solve_bs(*args, opts["bs_sequence"])
    if b_propagator == "cn_mixed":
        return _cn_mixed_solve_bs(*args, opts["solver_tol"],
                                  opts["max_solver_iter"], opts["lo_dtype"])
    return _cn_solve_bs(*args, opts["solver_tol"], opts["max_solver_iter"])


def _solve_bs_subcycled(
    mctdh, b_propagator, bs_start, h_current, D_invs, ds, dt, b_shapes, opts
):
    """Orbital step over dt in `n_orbital_substeps` substeps.

    The density matrices stay frozen over the whole CMF step; the projector
    (1-P) and the mean-field partner orbitals are re-frozen at the start of
    every substep. On stiff orbital equations (weakly occupied orbitals, DVR
    kinetic energy) the frozen projector is the dominant CMF error — measured
    on the four-well N=3 system the whole-step error depends only on
    dt/n_orbital_substeps, not on how the CI step is coupled — so this buys
    a larger CI step for the cost of extra orbital solves, which are cheap
    when the CI block dominates. n = 1 is the plain frozen-projector substep.
    """
    n_sub = opts["n_orbital_substeps"]
    bs = bs_start
    for _ in range(n_sub):
        g2 = mctdh._precompute_frozen_g2(bs)
        bs = _solve_bs(
            mctdh, b_propagator, bs, h_current, D_invs, ds, bs, g2,
            dt / n_sub, b_shapes, opts,
        )
    return bs


def get_cmf_integrator(
    mctdh,
    dt,
    c_propagator="cn",
    b_propagator="cn",
    c_kwargs=None,
    b_kwargs=None,
):
    """Constant Mean Field (CMF) integrator, first-order (Lie-Trotter) split.

    Decouples CI and orbital propagation. Each sub-problem becomes linear
    in its own variable with the other frozen, and any of CN/SIL/
    diagonalize can solve it.

    This allows dt 100-1000× larger than Picard-iteration CN because
    the decoupled sub-problems are genuinely linear and can be solved
    to machine precision.

    Step: CI over dt with the integrals at b(t) (one-body operator at
    t+dt/2), then the orbitals over dt with the density matrices from
    C(t+dt). See get_cmf_half_step_integrator for the second-order layout.

    Args:
        mctdh: MCTDHF_n_species instance.
        dt: Time step.
        c_propagator: Sub-integrator for CI coefficients. One of:
            'cn'          - Crank-Nicolson with BiCGSTAB (default)
            'sil'         - Short Iterative Lanczos (Krylov + matrix exp)
            'diagonalize' - Dense eigh of H_CI. Exact MCSCF-style update in
                            ITE (replaces C by the lowest num_states
                            eigenvectors, independent of dt) and exact
                            propagation in RTE. Cheap when prod(num_dets)
                            is small.
        b_propagator: Sub-integrator for orbitals. One of:
            'cn'          - Crank-Nicolson with BiCGSTAB (default)
            'sil'         - Short Iterative Lanczos. Assumes a Hermitian
                            generator, which the (1-P)(h+W)D⁻¹ gauge is not.
            'diagonalize' - Dense matrix exponential of A_s (the linearised
                            substep operator). Exact in the linearised
                            ODE. Cheap when n_spatial · n_orb is small.
            'dop'         - fixed-step Dormand-Prince 5(4)
            'bs'          - Bulirsch-Stoer (modified midpoint + Richardson)
        c_kwargs: Dict of solver-specific options for the chosen
            c_propagator. Recognised keys (defaults in parens):
              cn:           solver_tol (1e-8), max_solver_iter (100)
              sil:          krylov_dim (10)
              diagonalize:  (none)
        b_kwargs: Dict of solver-specific options for the chosen
            b_propagator. Recognised keys (defaults in parens):
              cn:           solver_tol (1e-8), max_solver_iter (100)
              sil:          krylov_dim (20)
              diagonalize:  orbital_x64 (False)
              dop:          n_substeps (1)
              bs:           bs_sequence ((2,4,6,8,12,16))
              all:          n_orbital_substeps (1) — sub-cycle the orbital
                            step with the projector re-frozen every dt/n
                            (see _solve_bs_subcycled)

    Returns:
        integrate: Function integrate(t, y) -> (t_new, y_new).
    """
    c_opts, b_opts = _cmf_options(c_propagator, b_propagator, c_kwargs, b_kwargs)
    c_shape = (mctdh.num_states,) + tuple(mctdh.num_dets)
    b_shapes = [
        (mctdh.num_spatial_orbitals[s], mctdh.num_mctdhf_orbitals[s])
        for s in range(mctdh.num_species)
    ]

    @jax.jit
    def integrate(t, y):
        C, bs = mctdh._y_to_Cb(y)

        # 1. Transform integrals at midpoint time for second-order accuracy
        h_current = mctdh._get_h_current(t + dt / 2)
        h_MO_list = [
            mctdh.transform_h(h_current[s], bs[s])[1] for s in range(mctdh.num_species)
        ]
        g_MO_dict, _ = mctdh._transform_all_g(bs, full=False)

        # 2. Solve CI (orbitals frozen)
        C_new = _solve_C(mctdh, c_propagator, C, h_MO_list, g_MO_dict, dt,
                         c_shape, c_opts)

        # 3. Update density matrices from new C
        D_invs, ds = _compute_density_quantities(mctdh, C_new)

        # 4. Solve orbitals (projector frozen from bs, re-frozen per substep)
        bs_new = _solve_bs_subcycled(
            mctdh, b_propagator, bs, h_current, D_invs, ds, dt, b_shapes, b_opts
        )

        return t + dt, mctdh._Cb_to_y(C_new, bs_new)

    return integrate


def get_cmf_half_step_integrator(
    mctdh,
    dt,
    c_propagator="cn",
    b_propagator="cn",
    c_kwargs=None,
    b_kwargs=None,
):
    """Improved CMF integrator with half-step predictor-corrector.

    Uses a midpoint scheme to reduce the CMF splitting error from O(dt²)
    to O(dt³) per step, which is critical for real-time evolution stability
    at moderate dt.

    Algorithm:
        1. Predictor: propagate C and b by dt/2 using mean fields at t
        2. Re-evaluate mean fields at the midpoint (t + dt/2)
        3. Corrector: propagate C and b from t by full dt using midpoint
           mean fields

    This doubles the computational cost per step compared to basic CMF,
    but typically allows 5-10× larger dt in RTE, for a net speedup.

    Caution: the corrector freezes the projector and mean field at the
    PREDICTED b(t+dt/2). On stiff orbital equations (DVR grids, e.g.
    the three-electron four-well system) that feedback is a linear instability at every dt;
    get_cmf_midpoint_integrator is the same layout without the predictor.

    Args:
        mctdh: MCTDHF_n_species instance.
        dt: Time step.
        c_propagator: Sub-integrator for CI coefficients. One of:
            'cn'          - Crank-Nicolson with BiCGSTAB (default)
            'sil'         - Short Iterative Lanczos
            'diagonalize' - Dense eigh of H_CI; exact in the linearised
                            substep.
        b_propagator: Sub-integrator for orbitals. One of:
            'cn'          - Crank-Nicolson with BiCGSTAB (default)
            'bs'          - Bulirsch-Stoer (modified midpoint + Richardson)
            'sil'         - Short Iterative Lanczos
            'diagonalize' - Dense matrix exponential of A_s; exact in the
                            linearised substep, autodiff-clean.
        c_kwargs: Dict of solver-specific options for the chosen
            c_propagator. Recognised keys (defaults in parens):
              cn:           solver_tol (1e-8), max_solver_iter (100)
              sil:          krylov_dim (10)
              diagonalize:  (none)
        b_kwargs: Dict of solver-specific options for the chosen
            b_propagator. Recognised keys (defaults in parens):
              cn:           solver_tol (1e-8), max_solver_iter (100),
                            n_orbital_substeps (1)
              sil:          krylov_dim (20)
              bs:           bs_sequence ((2,4,6,8,12,16))
              diagonalize:  (none)

    Returns:
        integrate: Function integrate(t, y) -> (t_new, y_new).
    """
    if c_propagator not in ("cn", "sil", "diagonalize"):
        raise ValueError(
            f"Unknown c_propagator '{c_propagator}'. "
            f"Use 'cn', 'sil', or 'diagonalize'."
        )
    if b_propagator not in ("cn", "bs", "sil", "diagonalize", "dop"):
        raise ValueError(
            f"Unknown b_propagator '{b_propagator}'. "
            f"Use 'cn', 'bs', 'sil', or 'diagonalize'."
        )

    c_kwargs = dict(c_kwargs or {})
    b_kwargs = dict(b_kwargs or {})
    _validate_kwargs(c_kwargs, _C_KWARG_KEYS[c_propagator], "c_kwargs", c_propagator)
    _validate_kwargs(b_kwargs, _B_KWARG_KEYS_HALF[b_propagator], "b_kwargs", b_propagator)

    c_solver_tol = c_kwargs.get("solver_tol", 1e-8)
    c_max_iter = c_kwargs.get("max_solver_iter", 100)
    c_krylov_dim = c_kwargs.get("krylov_dim", 10)

    b_solver_tol = b_kwargs.get("solver_tol", 1e-8)
    b_max_iter = b_kwargs.get("max_solver_iter", 100)
    b_krylov_dim = b_kwargs.get("krylov_dim", 20)
    b_n_substeps = b_kwargs.get("n_substeps", 1)
    n_sub = b_kwargs.get("n_orbital_substeps", 1)
    bs_sequence = b_kwargs.get("bs_sequence", (2, 4, 6, 8, 12, 16))
    b_orbital_x64 = b_kwargs.get("orbital_x64", False)

    c_shape = (mctdh.num_states,) + tuple(mctdh.num_dets)
    b_shapes = [
        (mctdh.num_spatial_orbitals[s], mctdh.num_mctdhf_orbitals[s])
        for s in range(mctdh.num_species)
    ]
    dt_half = dt / 2

    @jax.jit
    def integrate(t, y):
        C, bs = mctdh._y_to_Cb(y)

        # ── Predictor: half-step with mean fields at t ──────────────────
        h_current_t = mctdh._get_h_current(t)
        h_current_half = mctdh._get_h_current(t + dt_half)

        # Transform integrals at b(t)
        h_MO_list_t = [
            mctdh.transform_h(h_current_t[s], bs[s])[1]
            for s in range(mctdh.num_species)
        ]
        g_MO_dict_t, _ = mctdh._transform_all_g(bs, full=False)

        # CI half-step: C(t) → C(t+dt/2)
        if c_propagator == "sil":
            C_half, _ = _sil_solve_C(
                mctdh, C, h_MO_list_t, g_MO_dict_t, dt_half, c_shape, c_krylov_dim
            )
        elif c_propagator == "diagonalize":
            C_half, _ = _diag_solve_C(
                _ci_apply_fn(mctdh, h_MO_list_t, g_MO_dict_t), C, dt_half,
                mctdh.imag_time, c_shape,
            )
        else:
            C_half = _cn_solve_C(
                mctdh,
                C,
                h_MO_list_t,
                g_MO_dict_t,
                dt_half,
                c_shape,
                c_solver_tol,
                c_max_iter,
            )

        # Density matrices from C(t)
        D_invs_t, ds_t = _compute_density_quantities(mctdh, C)

        # Orbital half-step: b(t) → b(t+dt/2) (frozen projector from bs)
        g2_t = mctdh._precompute_frozen_g2(bs)
        if b_propagator == "sil":
            bs_half, _ = _sil_solve_bs(
                mctdh,
                bs,
                h_current_t,
                D_invs_t,
                ds_t,
                bs,
                g2_t,
                dt_half,
                b_shapes,
                b_krylov_dim,
            )
        elif b_propagator == "bs":
            bs_half = _bs_solve_bs(
                mctdh,
                bs,
                h_current_t,
                D_invs_t,
                ds_t,
                bs,
                g2_t,
                dt_half,
                b_shapes,
                bs_sequence,
            )
        elif b_propagator == "diagonalize":
            bs_half, _ = _diag_solve_bs(
                mctdh,
                bs,
                h_current_t,
                D_invs_t,
                ds_t,
                bs,
                g2_t,
                dt_half,
                b_shapes,
                compute_x64=b_orbital_x64,
            )
        elif b_propagator == "dop":
            bs_half, _ = _dop_solve_bs(
                mctdh,
                bs,
                h_current_t,
                D_invs_t,
                ds_t,
                bs,
                g2_t,
                dt_half,
                b_shapes,
                b_n_substeps,
            )
        else:
            dt_orb_pred = dt_half / n_sub
            bs_half = bs
            for _ in range(n_sub):
                bs_half = _cn_solve_bs(
                    mctdh,
                    bs_half,
                    h_current_t,
                    D_invs_t,
                    ds_t,
                    bs,
                    g2_t,
                    dt_orb_pred,
                    b_shapes,
                    b_solver_tol,
                    b_max_iter,
                )

        # ── Corrector: full step with midpoint mean fields ──────────────

        # Density matrices from C(t+dt/2)
        D_invs_half, ds_half = _compute_density_quantities(mctdh, C_half)

        # Orbital full step: b(t) → b(t+dt) using midpoint mean fields
        # (frozen projector from bs_half)
        g2_half = mctdh._precompute_frozen_g2(bs_half)
        if b_propagator == "sil":
            bs_new, _ = _sil_solve_bs(
                mctdh,
                bs,
                h_current_half,
                D_invs_half,
                ds_half,
                bs_half,
                g2_half,
                dt,
                b_shapes,
                b_krylov_dim,
            )
        elif b_propagator == "bs":
            bs_new = _bs_solve_bs(
                mctdh,
                bs,
                h_current_half,
                D_invs_half,
                ds_half,
                bs_half,
                g2_half,
                dt,
                b_shapes,
                bs_sequence,
            )
        elif b_propagator == "diagonalize":
            bs_new, _ = _diag_solve_bs(
                mctdh,
                bs,
                h_current_half,
                D_invs_half,
                ds_half,
                bs_half,
                g2_half,
                dt,
                b_shapes,
                compute_x64=b_orbital_x64,
            )
        elif b_propagator == "dop":
            bs_new, _ = _dop_solve_bs(
                mctdh,
                bs,
                h_current_half,
                D_invs_half,
                ds_half,
                bs_half,
                g2_half,
                dt,
                b_shapes,
                b_n_substeps,
            )
        else:
            dt_orb_corr = dt / n_sub
            bs_new = bs
            for _ in range(n_sub):
                bs_new = _cn_solve_bs(
                    mctdh,
                    bs_new,
                    h_current_half,
                    D_invs_half,
                    ds_half,
                    bs_half,
                    g2_half,
                    dt_orb_corr,
                    b_shapes,
                    b_solver_tol,
                    b_max_iter,
                )

        h_current_new = mctdh._get_h_current(t + dt)

        # Transform integrals at b(t+dt)
        h_MO_list_new = [
            mctdh.transform_h(h_current_new[s], bs_new[s])[1]
            for s in range(mctdh.num_species)
        ]
        g_MO_dict_new, _ = mctdh._transform_all_g(bs_new, full=False)

        # CI full step: C(t) → C(t+dt) using midpoint MO integrals
        if c_propagator == "sil":
            C_new, _ = _sil_solve_C(
                mctdh,
                C_half,
                h_MO_list_new,
                g_MO_dict_new,
                dt_half,
                c_shape,
                c_krylov_dim,
            )
        elif c_propagator == "diagonalize":
            C_new, _ = _diag_solve_C(
                _ci_apply_fn(mctdh, h_MO_list_new, g_MO_dict_new),
                C_half,
                dt_half,
                mctdh.imag_time,
                c_shape,
            )
        else:
            C_new = _cn_solve_C(
                mctdh,
                C_half,
                h_MO_list_new,
                g_MO_dict_new,
                dt_half,
                c_shape,
                c_solver_tol,
                c_max_iter,
            )

        return t + dt, mctdh._Cb_to_y(C_new, bs_new)

    return integrate



def get_cmf_midpoint_integrator(
    mctdh,
    dt,
    c_propagator="cn",
    b_propagator="cn",
    c_kwargs=None,
    b_kwargs=None,
):
    """Second-order (midpoint) CMF integrator.

    Algorithm (the Beck-Meyer second-order CMF layout):
        1. CI half-step C(t) -> C(t+dt/2) with the integrals at b(t)
        2. Orbital full step b(t) -> b(t+dt) with the density matrices from
           C(t+dt/2) and the one-body operator at t+dt/2; projector and
           mean-field partner orbitals frozen at b(t) (sub-cycled if
           n_orbital_substeps > 1)
        3. CI half-step C(t+dt/2) -> C(t+dt) with the integrals at b(t+dt)

    Unlike get_cmf_half_step_integrator there is NO orbital predictor: that
    integrator freezes the projector and mean field at a predicted b(t+dt/2),
    and feeding a predicted orbital back through (1-P) with the stiff one-body
    operator is a linear instability at every dt (observed for
    the three-electron four-well system). Cost: two CI half-steps and one orbital step, i.e. ~2x the
    basic integrator when the CI step dominates.

    Args: as get_cmf_integrator.

    Returns:
        integrate: Function integrate(t, y) -> (t_new, y_new).
    """
    c_opts, b_opts = _cmf_options(c_propagator, b_propagator, c_kwargs, b_kwargs)
    c_shape = (mctdh.num_states,) + tuple(mctdh.num_dets)
    b_shapes = [
        (mctdh.num_spatial_orbitals[s], mctdh.num_mctdhf_orbitals[s])
        for s in range(mctdh.num_species)
    ]
    dt_half = dt / 2

    @jax.jit
    def integrate(t, y):
        C, bs = mctdh._y_to_Cb(y)

        # 1. CI half-step with the integrals at b(t)
        h_current_t = mctdh._get_h_current(t)
        h_MO_list_t = [
            mctdh.transform_h(h_current_t[s], bs[s])[1]
            for s in range(mctdh.num_species)
        ]
        g_MO_dict_t, _ = mctdh._transform_all_g(bs, full=False)
        C_half = _solve_C(mctdh, c_propagator, C, h_MO_list_t, g_MO_dict_t,
                          dt_half, c_shape, c_opts)

        # 2. Orbital full step: midpoint density matrices, projector at b(t)
        D_invs_half, ds_half = _compute_density_quantities(mctdh, C_half)
        h_current_half = mctdh._get_h_current(t + dt_half)
        bs_new = _solve_bs_subcycled(
            mctdh, b_propagator, bs, h_current_half, D_invs_half, ds_half, dt,
            b_shapes, b_opts,
        )

        # 3. CI half-step with the integrals at b(t+dt)
        h_current_new = mctdh._get_h_current(t + dt)
        h_MO_list_new = [
            mctdh.transform_h(h_current_new[s], bs_new[s])[1]
            for s in range(mctdh.num_species)
        ]
        g_MO_dict_new, _ = mctdh._transform_all_g(bs_new, full=False)
        C_new = _solve_C(mctdh, c_propagator, C_half, h_MO_list_new,
                         g_MO_dict_new, dt_half, c_shape, c_opts)

        return t + dt, mctdh._Cb_to_y(C_new, bs_new)

    return integrate


def get_adaptive_cmf_half_step_integrator(
    mctdh,
    sil_krylov_dim=10,
    sil_b_krylov_dim=20,
):
    """Adaptive-step CMF half-step integrator with SIL sub-propagators.

    Returns a function adaptive_step(t, y, dt) that performs one CMF half-step
    and returns the new state along with error estimates for step size control.

    Uses the same algorithm as get_cmf_half_step_integrator with c_propagator='sil'
    and b_propagator='sil', but dt is a runtime argument (not closure-captured)
    so it can be varied by jax.lax.while_loop.

    Error estimates:
    - err_sil: max SIL Krylov truncation error from corrector propagations
    - err_split: CMF splitting error from forward/backward comparison at midpoint

    Args:
        mctdh: MCTDHF_n_species instance.
        sil_krylov_dim: Krylov subspace dimension for CI SIL (default 10).
        sil_b_krylov_dim: Krylov subspace dimension for orbital SIL (default 20).

    Returns:
        adaptive_step: Function (t, y, dt) -> (t_new, y_new, err_sil, err_split).
    """
    c_shape = (mctdh.num_states,) + tuple(mctdh.num_dets)
    b_shapes = [
        (mctdh.num_spatial_orbitals[s], mctdh.num_mctdhf_orbitals[s])
        for s in range(mctdh.num_species)
    ]

    @jax.jit
    def adaptive_step(t, y, dt):
        dt_half = dt / 2
        C, bs = mctdh._y_to_Cb(y)

        # ── Predictor: half-step with mean fields at t ──────────────────
        h_current_t0 = mctdh._get_h_current(t)
        h_current_half = mctdh._get_h_current(t + dt_half)

        # Transform integrals at b(t)
        h_MO_list_t0 = [
            mctdh.transform_h(h_current_t0[s], bs[s])[1]
            for s in range(mctdh.num_species)
        ]
        g_MO_dict_t0, _ = mctdh._transform_all_g(bs, full=False)

        # CI half-step: C(t) → C(t+dt/2)
        C_half, _ = _sil_solve_C(
            mctdh, C, h_MO_list_t0, g_MO_dict_t0, dt_half, c_shape, sil_krylov_dim
        )

        # Density matrices from C(t)
        D_invs_t0, ds_t0 = _compute_density_quantities(mctdh, C)

        # Orbital half-step: b(t) → b(t+dt/2) (frozen projector from bs)
        g2_t0 = mctdh._precompute_frozen_g2(bs)
        bs_half, _ = _sil_solve_bs(
            mctdh,
            bs,
            h_current_t0,
            D_invs_t0,
            ds_t0,
            bs,
            g2_t0,
            dt_half,
            b_shapes,
            sil_b_krylov_dim,
        )

        # ── Corrector: full step with midpoint mean fields ──────────────
        # Density matrices from C(t+dt/2)
        D_invs_half, ds_half = _compute_density_quantities(mctdh, C_half)

        # Orbital full step: b(t) → b(t+dt) using midpoint mean fields
        # (frozen projector from bs_half)
        g2_half = mctdh._precompute_frozen_g2(bs_half)
        bs_t1, err_sil_b = _sil_solve_bs(
            mctdh,
            bs,
            h_current_half,
            D_invs_half,
            ds_half,
            bs_half,
            g2_half,
            dt,
            b_shapes,
            sil_b_krylov_dim,
        )

        h_current_t1 = mctdh._get_h_current(t + dt)

        # Transform integrals at b(t+dt)
        h_MO_list_t1 = [
            mctdh.transform_h(h_current_t1[s], bs_t1[s])[1]
            for s in range(mctdh.num_species)
        ]
        g_MO_dict_t1, _ = mctdh._transform_all_g(bs_t1, full=False)

        # CI corrector + backward propagation from a single Lanczos decomposition.
        # C_t1 = exp(-i*dt/2*H_end)*C_half  (forward, the corrector result)
        # C_back = exp(+i*dt/2*H_end)*C_half (backward, for splitting error)
        C_t1, C_back, err_sil_C = _sil_solve_C_bidi(
            mctdh,
            C_half,
            h_MO_list_t1,
            g_MO_dict_t1,
            dt_half,
            c_shape,
            sil_krylov_dim,
        )

        # ── Error estimates ─────────────────────────────────────────────
        err_sil = jnp.maximum(err_sil_C, err_sil_b)

        # CMF splitting error: ||C(t) - C_back|| / ||C(t)||
        norm_C = jnp.linalg.norm(C.reshape(-1))
        safe_norm_C = jnp.where(norm_C > 0, norm_C, 1.0)
        err_split = jnp.linalg.norm((C - C_back).reshape(-1)) / safe_norm_C

        y_new = mctdh._Cb_to_y(C_t1, bs_t1)
        return t + dt, y_new, err_sil, err_split

    return adaptive_step


def get_vmf_integrator(
    mctdh, dt, num_newton=3, min_solver_tol=1e-8, max_solver_tol=1e-2, max_solver_iter=50, precondition=True
):
    """Variable Mean Field integrator using implicit midpoint + Newton-Krylov.

    Uses the implicit midpoint rule, which is symplectic (preserves norm
    in RTE) and A-stable (handles stiffness). The nonlinear implicit
    equation is solved with Newton-Krylov iterations using JAX's forward-
    mode AD (jvp) for matrix-free Jacobian-vector products.

    Unlike the CMF integrators, this has NO splitting error — mean fields
    are always evaluated at the current state. The implicit midpoint rule
    also preserves quadratic invariants (norm, energy), making it ideal
    for real-time propagation.

    Trade-off: each step is more expensive than CMF (Newton iterations
    with Krylov solves), but the absence of splitting error and symplectic
    structure often allow significantly larger dt.

    The optional block-diagonal preconditioner targets the orbital one-body
    Jacobian block via precomputed eigh(h_list[s]). This is the dominant
    source of stiffness (DVR kinetic energy spectrum) and the preconditioner
    clusters eigenvalues of M⁻¹·A near 1, significantly reducing BiCGSTAB
    iteration counts and enabling larger time steps. The CI block uses
    identity (small and not stiff). For time-dependent h, the preconditioner
    stays anchored to the static reference — still effective if the driving
    is perturbative relative to T+V_static.

    Args:
        mctdh: MCTDHF_n_species instance. Its __call__ method is used as
               the full (coupled) RHS.
        dt: Time step.
        num_newton: Number of Newton iterations per step.
        solver_tol: BiCGSTAB convergence tolerance for Newton linear systems.
        max_solver_iter: Maximum BiCGSTAB iterations per Newton step.
        precondition: If True (default), use a block-diagonal preconditioner
            for the orbital one-body Jacobian block. The preconditioner is
            built from precomputed eigendecompositions of h_list[s] and
            applies M⁻¹ = (I + α·h_s)⁻¹ spectrally per species. Set to
            False to recover the original unpreconditioned BiCGSTAB.

    Returns:
        integrate: Function integrate(t, y) -> (t_new, y_new).
    """
    t_half = dt / 2

    # ── One-time eigendecomposition for the preconditioner ────────────
    if precondition:
        alpha = (dt / 2) if mctdh.imag_time else (1j * dt / 2)
        eig_data = []  # list of (Q_s, inv_factors_s) per species
        h_static = mctdh._get_h_current(0.0)
        for s in range(mctdh.num_species):
            h_s = jnp.asarray(h_static[s])                # (N_s, N_s)
            D_s, Q_s = jnp.linalg.eigh(h_s)               # D_s real, Q_s unitary
            inv_factors = 1.0 / (1.0 + alpha * D_s)        # (N_s,)
            eig_data.append((Q_s, inv_factors))
    else:
        eig_data = None

    @jax.jit
    def integrate(t, y):
        t_mid = t + t_half

        # Initial guess: explicit midpoint predictor (order 2).
        # Gives O(dt³) error in z₀ vs Euler's O(dt²), which can save a
        # Newton iteration at larger dt.
        f0 = mctdh(t, y)
        y_half_pred = y + t_half * f0
        f_half = mctdh(t_mid, y_half_pred)
        z = y + dt * f_half

        # Newton iterations to solve the implicit midpoint equation:
        #   G(z) = z - y - dt * f(t_mid, (z + y)/2) = 0
        #
        # Newton update: G'(z) · δz = -G(z)
        #   G'(z) · v = v - (dt/2) · J_f(y_mid) · v
        # where J_f · v is computed via jax.jvp.
        #
        # Eisenstat–Walker adaptive tolerance: early Newton steps have
        # large residuals and don't need tight BiCGSTAB solves. We set
        # tol_k = max(solver_tol, min(0.5, ‖G(z_k)‖/‖G(z_0)‖)).
        def newton_step(carry, _):
            z, res_norm_0 = carry
            y_mid = (z + y) / 2

            # Linearize: compute f_mid AND cache intermediates for JVP.
            # Each BiCGSTAB matvec then only computes the tangent (~1x RHS)
            # instead of recomputing both primals and tangents (~2x RHS).
            f_mid, jvp_fn = jax.linearize(
                lambda yy: mctdh(t_mid, yy), y_mid
            )
            residual = z - y - dt * f_mid

            # Adaptive tolerance (Eisenstat–Walker)
            res_norm = jnp.linalg.norm(residual)
            # First iteration: res_norm_0 is 0 → initialize it
            res_norm_0 = jnp.where(res_norm_0 == 0.0, res_norm, res_norm_0)
            adaptive_tol = jnp.maximum(
                min_solver_tol,
                jnp.minimum(max_solver_tol, res_norm / res_norm_0),
            )

            def matvec(v):
                return v - t_half * jvp_fn(v)

            # Initial guess: M⁻¹·(-residual) ≈ A⁻¹·(-residual) when the
            # preconditioner is available; -residual otherwise.
            neg_res = -residual
            x0 = _apply_M_inv(neg_res) if eig_data is not None else neg_res

            dz, _ = jax.scipy.sparse.linalg.bicgstab(
                matvec,
                neg_res,
                x0=x0,
                tol=adaptive_tol,
                maxiter=max_solver_iter,
                M=_apply_M_inv if eig_data is not None else None,
            )
            return (z + dz, res_norm_0), None

        (z, _), _ = jax.lax.scan(
            newton_step, (z, jnp.array(0.0, dtype=z.dtype)), None, length=num_newton
        )

        return t + dt, z

    # ── Preconditioner: block-diagonal M⁻¹ ───────────────────────────
    # Identity on the CI block; spectral inverse (I + α·h_s)⁻¹ on each
    # orbital block b_s. Cost per application: K × 2 × N²·M flops +
    # K × N·M complex divides — negligible vs one JVP.
    def _apply_M_inv(v):
        v_C, v_bs = mctdh._y_to_Cb(v)
        v_bs_new = []
        for s, (Q_s, inv_factors) in enumerate(eig_data):
            # v_bs[s] shape: (N_s, M_s) — apply M⁻¹ to each orbital column.
            u = Q_s.conj().T @ v_bs[s]          # rotate to eigenbasis  (N_s, M_s)
            u = inv_factors[:, None] * u         # diagonal scale per eigenmode
            u = Q_s @ u                          # back to DVR basis
            v_bs_new.append(u)
        return mctdh._Cb_to_y(v_C, v_bs_new)

    return integrate


def get_psi_integrator(mctdh, dt):
    """Projector-Splitting Integrator (PSI) for 1-particle-per-species MCTDHF.

    Implements a first-order Lie-Trotter splitting that bypasses the density
    matrix inverse D⁻¹. Provides exact norm conservation (up to time
    discretization) via QR decomposition.

    Constraints (enforced at construction):
      - num_states == 1.
      - num_particles[s] == 1 for every species s.
      - Square-matricization rank constraint: M_s == ∏_{k≠s} M_k for every
        species s. This is stricter than the standard PSI condition
        (M_s ≤ ∏_{k≠s} M_k); it is required here because the QR step
        replaces b_s by Q and the C-slice along mode s by R in one shot,
        which only closes when K_(s) is square (so R is M_s × M_s).

    The rank constraint has only trivial solutions for N ≥ 3 species:
    setting M_s == ∏_{k≠s} M_k for every s collapses (by substitution) to
    M_s == 1 for all s. So in practice this implementation is usable
    only for N == 2 species (any equal M_0 == M_1) or for the trivial
    N ≥ 3 case with all M_s == 1.

    Args:
        mctdh: MCTDHF_n_species instance (1 particle per species).
        dt: Time step.

    Returns:
        integrate: Function integrate(t, y) -> (t_new, y_new).
    """
    import numpy as np

    N = mctdh.num_species
    Ms = list(mctdh.num_mctdhf_orbitals)

    # Validate constraints
    if mctdh.num_states != 1:
        raise ValueError(f"PSI requires num_states=1, got {mctdh.num_states}")
    for s in range(N):
        if mctdh.num_particles[s] != 1:
            raise ValueError(
                f"PSI requires 1 particle per species, "
                f"species {s} has {mctdh.num_particles[s]}"
            )

    # Validate rank constraint: ∏_{k≠s} M_k == M_s for all s
    for s in range(N):
        D = int(np.prod([Ms[k] for k in range(N) if k != s]))
        if D != Ms[s]:
            raise ValueError(
                f"PSI rank constraint violated for species {s}: "
                f"product of other M_k = {D} != M_{s} = {Ms[s]}"
            )

    N_spatials = list(mctdh.num_spatial_orbitals)

    @jax.jit
    def integrate(t, y):
        C, bs = mctdh._y_to_Cb(y)
        h_current = mctdh._get_h_current(t)

        for s in range(N):
            # ── K-step (forward, +dt) ────────────────────────────────
            # Form K_s by contracting C with b_s along mode s.
            # C shape: (1, M_0, ..., M_{N-1})
            # b_s shape: (N_spatial_s, M_s)
            # K_s shape: (1, M_0, ..., M_{s-1}, N_spatial_s, M_{s+1}, ..., M_{N-1})
            idx = [chr(98 + k) for k in range(N)]  # b, c, d, ...
            c_str = "a" + "".join(idx)

            new_char = chr(98 + N)  # next unused
            k_idx = list(idx)
            k_idx[s] = new_char
            k_str = "a" + "".join(k_idx)
            b_str = f"{new_char}{idx[s]}"
            einsum_form_K = f"{b_str},{c_str}->{k_str}"
            K_s = contract(einsum_form_K, bs[s], C, backend="jax")

            # Precompute integrals once (depend on bs, not K_s)
            K_integrals = mctdh._precompute_K_integrals(s, bs, h_current)

            # Propagate K_s forward by dt (RK4)
            k1 = mctdh._apply_K_hamiltonian(K_s, s, *K_integrals)
            k2 = mctdh._apply_K_hamiltonian(K_s + dt / 2 * k1, s, *K_integrals)
            k3 = mctdh._apply_K_hamiltonian(K_s + dt / 2 * k2, s, *K_integrals)
            k4 = mctdh._apply_K_hamiltonian(K_s + dt * k3, s, *K_integrals)
            K_s_new = K_s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

            # QR decompose: reshape K to (N_spatial_s, D) where D = ∏_{k≠s} M_k
            # The state axis and all non-s mode axes are flattened into D.
            # First, move the spatial axis (position s+1) to position 1:
            K_shape = K_s_new.shape  # (1, M_0, ..., N_spatial_s, ..., M_{N-1})
            # Axes: 0=state, 1..s=modes 0..s-1, s+1=spatial, s+2..N=modes s+1..N-1
            spatial_ax = s + 1
            # Move spatial to front (after state), then flatten the rest
            perm = [spatial_ax] + [i for i in range(0, N + 1) if i != spatial_ax]
            K_perm = jnp.transpose(K_s_new, perm)
            # K_perm shape: (1, N_spatial_s, M_0, ..., M_{s-1}, M_{s+1}, ..., M_{N-1})
            D = int(np.prod([K_perm.shape[i] for i in range(1, N + 1)]))
            K_mat = K_perm.reshape(N_spatials[s], D)

            # Thin QR
            Q, R = jnp.linalg.qr(K_mat)
            # Positive diagonal convention for gauge consistency
            signs = jnp.sign(jnp.diag(R))
            signs = jnp.where(signs == 0, 1.0, signs)
            Q = Q * signs[None, :]
            R = signs.conj()[:, None] * R

            # New b_s = Q (shape: N_spatial_s × D = N_spatial_s × M_s)
            bs[s] = Q

            # New C = R reshaped back to tensor form
            # R shape: (D, D) = (M_s, M_s) ... but we need to reshape
            # back to the permuted shape then un-permute.
            # R has shape (D, D) where D = num_states * ∏_{k≠s} M_k
            # = 1 * M_s = M_s
            # Reshape R to (1, M_0, ..., M_{s-1}, M_s_new, M_{s+1}, ..., M_{N-1})
            # where M_s_new is at position 1 in the permuted layout
            perm_shape = [K_perm.shape[i] for i in range(2, N + 1)]
            C_perm = R.reshape([Ms[s]] + [1] + perm_shape)
            # C_perm shape: (1, M_s_new, M_0, ..., M_{s-1}, M_{s+1}, ..., M_{N-1})
            # Undo the permutation: perm was [0, spatial_ax, other_axes...]
            # Inverse perm: position i in output came from perm[i] in input
            inv_perm = [0] * (N + 1)
            for i, p in enumerate(perm):
                inv_perm[p] = i
            C = jnp.transpose(C_perm, inv_perm)

            # ── S-step (backward, -dt) — for s < N-1 only ───────────
            if s < N - 1:
                # Transform integrals to new orbital basis
                h_MO_list = [
                    mctdh.transform_h(h_current[k], bs[k])[1] for k in range(N)
                ]
                g_MO_dict, _ = mctdh._transform_all_g(bs, full=False)

                # Propagate C backward by dt (RK4)
                def c_rhs(c):
                    return mctdh._rhs_C_linear(c, h_MO_list, g_MO_dict)

                k1 = c_rhs(C)
                k2 = c_rhs(C - dt / 2 * k1)
                k3 = c_rhs(C - dt / 2 * k2)
                k4 = c_rhs(C - dt * k3)
                C = C - dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return t + dt, mctdh._Cb_to_y(C, bs)

    return integrate


def diffrax_integrate(
    mctdh,
    t0,
    t1,
    y0,
    dt0=None,
    save_ts=None,
    num_save=100,
    solver=None,
    rtol=1e-6,
    atol=1e-8,
    max_steps=16**4,
):
    """Adaptive VMF integration using diffrax.

    Uses the full coupled RHS (mctdh.__call__) with an adaptive ODE solver.
    No CMF splitting error, no frozen mean fields. Step size is controlled
    automatically based on local error estimates.

    Complex state vectors are split into stacked real/imaginary float32
    parts to avoid diffrax's complex dtype limitations and to keep the
    same memory footprint as complex64.

    Best suited for real-time evolution (RTE) where:
    - Adaptive stepping efficiently handles varying dynamics
    - Norm is conserved by the equations (no normalization needed)
    - The absence of splitting error allows accurate long-time propagation

    For imaginary time evolution (ITE), use the fixed-step CMF integrators
    with explicit normalization instead.

    Args:
        mctdh: MCTDHF_n_species instance.
        t0: Start time.
        t1: End time.
        y0: Initial state vector (packed, complex64 or float32).
        dt0: Initial step size guess. Defaults to (t1-t0)/100.
        save_ts: Array of times at which to save the solution. If None,
                 saves at num_save evenly-spaced points.
        num_save: Number of evenly-spaced save points (used if save_ts is
                  None). Default 100.
        solver: diffrax solver instance, or 'explicit' (Tsit5) /
                'implicit' (Kvaerno5). Defaults to Tsit5.
        rtol: Relative tolerance for step-size control.
        atol: Absolute tolerance for step-size control.
        max_steps: Maximum number of internal steps.

    Returns:
        ts: Save times, shape (num_save,).
        ys: Complex state vectors at save times, shape (num_save, dim_y).
    """
    import diffrax

    if solver is None or solver == "explicit":
        solver = diffrax.Tsit5()
    elif solver == "implicit":
        solver = diffrax.Kvaerno5()
    elif isinstance(solver, str):
        raise ValueError("Invalid solver string. Use 'explicit' or 'implicit'.")

    if dt0 is None:
        dt0 = (t1 - t0) / 100

    is_complex = jnp.iscomplexobj(y0)
    n = y0.shape[0]

    if is_complex:
        # Split complex64 → stacked float32 [real | imag]
        y0_real = jnp.concatenate(
            [y0.real.astype(jnp.float32), y0.imag.astype(jnp.float32)]
        )

        def rhs_real(t, y_real, args):
            y_complex = y_real[:n] + 1j * y_real[n:]
            y_complex = y_complex.astype(jnp.complex64)
            dy = mctdh(t, y_complex)
            return jnp.concatenate(
                [dy.real.astype(jnp.float32), dy.imag.astype(jnp.float32)]
            )

        term = diffrax.ODETerm(rhs_real)
        y0_solve = y0_real
    else:
        term = diffrax.ODETerm(lambda t, y, args: mctdh(t, y))
        y0_solve = y0

    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

    if save_ts is None:
        save_ts = jnp.linspace(t0, t1, num_save)

    saveat = diffrax.SaveAt(ts=save_ts)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0_solve,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
        max_steps=max_steps,
    )

    if is_complex:
        # Recombine float32 → complex64
        ys = sol.ys[:, :n] + 1j * sol.ys[:, n:]
        ys = ys.astype(jnp.complex64)
    else:
        ys = sol.ys

    return sol.ts, ys


def diffrax_integrate_imex(
    mctdh,
    t0,
    t1,
    y0,
    dt0=None,
    save_ts=None,
    num_save=100,
    rtol=1e-6,
    atol=1e-8,
    max_steps=16**4,
):
    """Adaptive IMEX integration using diffrax Sil3.

    Splits the RHS into:
    - Implicit (stiff): orbital one-body (1-P)h·b — high-energy kinetic modes
    - Explicit (non-stiff): CI + orbital two-body mean-field contributions

    This treats the stiff kinetic operator implicitly (no stability restriction)
    while handling the nonlinear mean-field coupling explicitly (cheap, accurate).
    No CMF splitting error since both terms are evaluated at the current state.

    Complex states are split into real/imaginary float32 parts for diffrax
    compatibility.

    Args:
        mctdh: MCTDHF_n_species instance with _rhs_implicit and _rhs_explicit.
        t0: Start time.
        t1: End time.
        y0: Initial state vector (packed, complex64 or float32).
        dt0: Initial step size guess. Defaults to (t1-t0)/100.
        save_ts: Array of times to save. If None, num_save evenly-spaced.
        num_save: Number of save points (default 100).
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        max_steps: Maximum internal steps.

    Returns:
        ts: Save times, shape (num_save,).
        ys: Complex state vectors, shape (num_save, dim_y).
    """
    import diffrax

    if dt0 is None:
        dt0 = (t1 - t0) / 100

    is_complex = jnp.iscomplexobj(y0)
    n = y0.shape[0]

    if is_complex:
        y0_real = jnp.concatenate(
            [y0.real.astype(jnp.float32), y0.imag.astype(jnp.float32)]
        )

        def _to_complex(y_real):
            return (y_real[:n] + 1j * y_real[n:]).astype(jnp.complex64)

        def _to_real(dy):
            return jnp.concatenate(
                [dy.real.astype(jnp.float32), dy.imag.astype(jnp.float32)]
            )

        def rhs_implicit_real(t, y_real, args):
            return _to_real(mctdh._rhs_implicit(t, _to_complex(y_real)))

        def rhs_explicit_real(t, y_real, args):
            return _to_real(mctdh._rhs_explicit(t, _to_complex(y_real)))

        terms = diffrax.MultiTerm(
            diffrax.ODETerm(rhs_explicit_real),
            diffrax.ODETerm(rhs_implicit_real),
        )
        y0_solve = y0_real
    else:
        terms = diffrax.MultiTerm(
            diffrax.ODETerm(lambda t, y, args: mctdh._rhs_explicit(t, y)),
            diffrax.ODETerm(lambda t, y, args: mctdh._rhs_implicit(t, y)),
        )
        y0_solve = y0

    solver = diffrax.Sil3()
    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

    if save_ts is None:
        save_ts = jnp.linspace(t0, t1, num_save)

    saveat = diffrax.SaveAt(ts=save_ts)

    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0_solve,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
        max_steps=max_steps,
    )

    if is_complex:
        ys = (sol.ys[:, :n] + 1j * sol.ys[:, n:]).astype(jnp.complex64)
    else:
        ys = sol.ys

    return sol.ts, ys
