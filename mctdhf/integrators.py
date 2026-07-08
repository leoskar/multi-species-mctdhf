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


def _cn_solve_C(
    mctdh, C, h_MO_list, g_MO_dict, dt_sub, c_shape, solver_tol, max_solver_iter
):
    """CN-solve the CI sub-problem (linear in C, orbitals frozen).

    Solves: x - dt_sub/2 · f_C(x) = C + dt_sub/2 · f_C(C).
    """
    C_dot_0 = mctdh._rhs_C_linear(C, h_MO_list, g_MO_dict)
    b_vec = C + (dt_sub / 2) * C_dot_0

    def matvec(x_flat):
        x = x_flat.reshape(c_shape)
        fx = mctdh._rhs_C_linear(x, h_MO_list, g_MO_dict)
        return (x - (dt_sub / 2) * fx).reshape(-1)

    x0 = (C + dt_sub * C_dot_0).reshape(-1)
    C_new_flat, _ = jax.scipy.sparse.linalg.bicgstab(
        matvec,
        b_vec.reshape(-1),
        x0=x0,
        tol=solver_tol,
        maxiter=max_solver_iter,
    )
    return C_new_flat.reshape(c_shape)


def _cn_solve_bs(
    mctdh,
    bs_start,
    h_current,
    D_invs,
    ds,
    bs_frozen,
    dt_sub,
    b_shapes,
    solver_tol,
    max_solver_iter,
):
    """CN-solve all orbital sub-problems (linear in b_s, P and CI frozen).

    Solves per species: x - dt_sub/2 · f_b(x) = b_s + dt_sub/2 · f_b(b_s).
    """
    bs_new = list(bs_start)
    for s in range(mctdh.num_species):
        b_s = bs_start[s]

        b_dot_0 = mctdh._rhs_b_linear(
            b_s, s, h_current[s], D_invs[s], ds, bs_frozen
        )
        b_vec = b_s + (dt_sub / 2) * b_dot_0

        def matvec(x_flat, _s=s):
            x = x_flat.reshape(b_shapes[_s])
            fx = mctdh._rhs_b_linear(
                x, _s, h_current[_s], D_invs[_s], ds, bs_frozen
            )
            return (x - (dt_sub / 2) * fx).reshape(-1)

        x0 = (b_s + dt_sub * b_dot_0).reshape(-1)
        b_new_flat, _ = jax.scipy.sparse.linalg.bicgstab(
            matvec,
            b_vec.reshape(-1),
            x0=x0,
            tol=solver_tol,
            maxiter=max_solver_iter,
        )
        bs_new[s] = b_new_flat.reshape(b_shapes[s])
    return bs_new


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
    c_flat = C.reshape(-1)
    dim = c_flat.shape[0]
    norm_c = jnp.linalg.norm(c_flat)
    safe_norm = jnp.where(norm_c > 0, norm_c, 1.0)

    # Lanczos on H (Hermitian): sigma = H*C = dCI.get_sigma(C, ...)
    def apply_H(v_flat):
        return mctdh.dCI.get_sigma(
            v_flat.reshape(c_shape), h_MO_list, g_MO_dict
        ).reshape(-1)

    v0 = c_flat / safe_norm

    # Lanczos iteration via lax.scan with full re-orthogonalization.
    # V is carried in the scan state so that at each step we can
    # orthogonalize against all previously computed basis vectors.
    # This is essential for float32 where classical Lanczos loses
    # orthogonality within a few iterations.
    V_init = jnp.zeros((krylov_dim, dim), dtype=c_flat.dtype)

    vec_dtype = c_flat.dtype
    real_dtype = c_flat.real.dtype

    def lanczos_step(carry, j):
        V, v_prev, v_curr, beta_prev = carry
        # Store v_curr as basis vector j (at step start, not end,
        # to avoid overwriting row 0 on the last iteration)
        V = V.at[j].set(v_curr)
        w = apply_H(v_curr).astype(vec_dtype)
        alpha = jnp.real(jnp.dot(v_curr.conj(), w)).astype(real_dtype)
        w = w - alpha * v_curr - beta_prev * v_prev
        # Full re-orthogonalization against all stored basis vectors.
        # Rows beyond j are zero so the dot products vanish — no
        # masking needed, and the matmul is on a small (krylov_dim, dim)
        # matrix which stays in GPU cache.
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
    # alphas: (krylov_dim,), betas: (krylov_dim,)
    # V_final[0:krylov_dim] contains the orthonormal basis vectors

    # Build real symmetric tridiagonal T
    T = jnp.diag(alphas) + jnp.diag(betas[:-1], 1) + jnp.diag(betas[:-1], -1)

    # Time propagation: exp(-i*dt*H) for RTE, exp(-dt*H) for ITE
    if mctdh.imag_time:
        expT = jax.scipy.linalg.expm(-dt * T)
    else:
        expT = jax.scipy.linalg.expm(-1j * dt * T)

    e1 = jnp.zeros(krylov_dim, dtype=expT.dtype).at[0].set(1.0)
    coeffs = expT @ e1

    # SIL Krylov truncation error (Saad bound), relative
    sil_error = jnp.abs(betas[-1] * coeffs[-1])

    # Reconstruct: C_new = ||C|| * V^T @ coeffs (V is orthonormal)
    c_new_flat = safe_norm * jnp.dot(coeffs, V_final)
    c_new_flat = jnp.where(norm_c > 0, c_new_flat, c_flat)

    return c_new_flat.reshape(c_shape), sil_error


def _sil_solve_C_bidi(mctdh, C, h_MO_list, g_MO_dict, dt, c_shape, krylov_dim=10):
    """Bidirectional SIL propagation: forward and backward from a single Lanczos decomposition.

    Computes both C_fwd = exp(-i*dt*H)*C and C_bwd = exp(+i*dt*H)*C using
    the same Krylov subspace. The extra cost over _sil_solve_C is one additional
    small (krylov_dim x krylov_dim) matrix exponential.

    Returns:
        (C_fwd, C_bwd, sil_error_fwd)
    """
    c_flat = C.reshape(-1)
    dim = c_flat.shape[0]
    norm_c = jnp.linalg.norm(c_flat)
    safe_norm = jnp.where(norm_c > 0, norm_c, 1.0)

    def apply_H(v_flat):
        return mctdh.dCI.get_sigma(
            v_flat.reshape(c_shape), h_MO_list, g_MO_dict
        ).reshape(-1)

    v0 = c_flat / safe_norm

    V_init = jnp.zeros((krylov_dim, dim), dtype=c_flat.dtype)
    vec_dtype = c_flat.dtype
    real_dtype = c_flat.real.dtype

    def lanczos_step(carry, j):
        V, v_prev, v_curr, beta_prev = carry
        V = V.at[j].set(v_curr)
        w = apply_H(v_curr).astype(vec_dtype)
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

    # Diagonalize T and exponentiate eigenvalues for both directions
    # Forward: exp(-i*dt*H) for RTE, exp(-dt*H) for ITE
    if mctdh.imag_time:
        expT_fwd = jax.scipy.linalg.expm(-dt * T)
        expT_bwd = jax.scipy.linalg.expm(dt * T)
    else:
        expT_fwd = jax.scipy.linalg.expm(-1j * dt * T)
        expT_bwd = jax.scipy.linalg.expm(1j * dt * T)

    e1 = jnp.zeros(krylov_dim, dtype=expT_fwd.dtype).at[0].set(1.0)
    coeffs_fwd = expT_fwd @ e1
    coeffs_bwd = expT_bwd @ e1

    # SIL error from forward propagation
    sil_error = jnp.abs(betas[-1] * coeffs_fwd[-1])

    # Reconstruct both directions
    c_fwd_flat = safe_norm * jnp.dot(coeffs_fwd, V_final)
    c_fwd_flat = jnp.where(norm_c > 0, c_fwd_flat, c_flat)

    c_bwd_flat = safe_norm * jnp.dot(coeffs_bwd, V_final)
    c_bwd_flat = jnp.where(norm_c > 0, c_bwd_flat, c_flat)

    return c_fwd_flat.reshape(c_shape), c_bwd_flat.reshape(c_shape), sil_error


def _diag_solve_C(mctdh, C, h_MO_list, g_MO_dict, dt, c_shape):
    """Exact CI propagator for the linearised CMF substep.

    ITE: replace C by the lowest ``num_states`` eigenvectors of H_CI via a
    dense eigendecomposition. The input C is used only as a dtype/shape
    template — the result is the exact state-averaged MCSCF CI update given
    the current orbitals (so the dt argument is irrelevant in this branch).

    RTE: form ``expm(-i*dt*H_CI) @ C`` via a dense matrix exponential. This
    is autodiff-clean: unlike the eigendecomposition route
    (``U @ diag(exp(-i*dt*λ)) @ U†``), ``expm``'s reverse-mode VJP is built
    from matmuls and a linear solve — it has no ``1/(λ_i-λ_j)`` terms, so it
    does not produce NaN gradients when H_CI has degenerate eigenvalues
    (verified 2026-05-22). Forward value and unitarity match the eigh route
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

    # Build the dense CI Hamiltonian by applying get_sigma to the identity in
    # the determinant basis. dCI.get_sigma uses '...' einsum strings so any
    # leading batch shape is accepted.
    eye = jnp.eye(n_dets_tot, dtype=C.dtype).reshape((n_dets_tot,) + det_shape)
    sigma = mctdh.dCI.get_sigma(eye, h_MO_list, g_MO_dict)
    H_CI = sigma.reshape(n_dets_tot, n_dets_tot)
    # Hermitise to clean up floating-point noise.
    H_CI = 0.5 * (H_CI + H_CI.conj().T)

    if mctdh.imag_time:
        # Lowest num_states eigenvectors as the new state-averaged CI vectors.
        # eigvecs columns -> states; transpose so the leading axis is `state`.
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
    """
    bs_new = list(bs_start)
    sil_errors = []
    for s in range(mctdh.num_species):
        b_s = bs_start[s]
        b_flat = b_s.reshape(-1)
        dim = b_flat.shape[0]
        norm_b = jnp.linalg.norm(b_flat)
        safe_norm = jnp.where(norm_b > 0, norm_b, 1.0)

        def apply_A(v_flat, _s=s):
            """Apply orbital Hamiltonian A such that db/dt = -iAb (RTE)."""
            b = v_flat.reshape(b_shapes[_s])
            rhs = mctdh._rhs_b_linear(
                b, _s, h_current[_s], D_invs[_s], ds, bs_frozen
            )
            # _rhs_b_linear returns -1j*A*b (RTE) or -A*b (ITE)
            # We need A*b for the Lanczos (Hermitian operator)
            if mctdh.imag_time:
                return (-rhs).reshape(-1)
            else:
                return (1j * rhs).reshape(-1)

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
            # Full re-orthogonalization
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

        if mctdh.imag_time:
            expT = jax.scipy.linalg.expm(-dt * T)
        else:
            expT = jax.scipy.linalg.expm(-1j * dt * T)

        e1 = jnp.zeros(krylov_dim, dtype=expT.dtype).at[0].set(1.0)
        coeffs = expT @ e1

        # SIL Krylov truncation error (Saad bound), relative
        sil_error_s = jnp.abs(betas[-1] * coeffs[-1])
        sil_errors.append(sil_error_s)

        b_new_flat = safe_norm * jnp.dot(coeffs, V_final)
        b_new_flat = jnp.where(norm_b > 0, b_new_flat, b_flat)
        bs_new[s] = b_new_flat.reshape(b_shapes[s])

    return bs_new, jnp.max(jnp.array(sil_errors))


def _diag_solve_bs(
    mctdh,
    bs_start,
    h_current,
    D_invs,
    ds,
    bs_frozen,
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
    out_dtype = bs_start[0].dtype

    for s in range(mctdh.num_species):
        b_s = bs_start[s]
        b_flat = b_s.reshape(-1)
        N_s = b_shapes[s][0] * b_shapes[s][1]

        def apply_once(b, _s=s):
            rhs = mctdh._rhs_b_linear(
                b, _s, h_current[_s], D_invs[_s], ds, bs_frozen
            )
            return rhs.reshape(-1)

        # Build dense linear operator by vmapping apply_once over standard
        # basis vectors of the flat (n_spatial · M) orbital space. This stays
        # in the native bulk dtype even when compute_x64 is set.
        I_cols = jnp.eye(N_s, dtype=b_s.dtype).reshape((N_s,) + b_shapes[s])
        J_rows = jax.vmap(apply_once)(I_cols)   # row k = J · e_k
        J_mat = J_rows.T                         # columns = J · e_k → J as matrix

        # Extract A_s from _rhs_b_linear sign convention; form the expm argument.
        if mctdh.imag_time:
            arg = -dt * (-J_mat)
        else:
            arg = -1j * dt * (1j * J_mat)

        # The non-Hermitian expm is the precision-critical step: optionally
        # promote just it (and the application to b) to complex128.
        if compute_x64:
            expA = jax.scipy.linalg.expm(arg.astype(jnp.complex128))
            b_new_flat = expA @ b_flat.astype(jnp.complex128)
        else:
            expA = jax.scipy.linalg.expm(arg)
            b_new_flat = expA @ b_flat

        bs_new[s] = b_new_flat.reshape(b_shapes[s]).astype(out_dtype)

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


def _bs_solve_bs(
    mctdh,
    bs_start,
    h_current,
    D_invs,
    ds,
    bs_frozen,
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
        b_s = bs_start[s]

        def rhs_b(b, _s=s):
            return mctdh._rhs_b_linear(
                b, _s, h_current[_s], D_invs[_s], ds, bs_frozen
            )

        # Compute modified midpoint approximations for each n in sequence
        T_list = []
        h_sq = []
        for n in bs_sequence:
            T_n = _bs_modified_midpoint(rhs_b, b_s, dt, n)
            T_list.append(T_n)
            h_sq.append((dt / n) ** 2)

        # Polynomial extrapolation (Neville's algorithm)
        # T_list[i] has error O(h_i^2), extrapolation removes successive
        # powers of h^2 to reach O(h^{2*len(sequence)})
        for k in range(1, len(bs_sequence)):
            for i in range(len(bs_sequence) - 1, k - 1, -1):
                ratio = h_sq[i - k] / h_sq[i]
                T_list[i] = T_list[i] + (T_list[i] - T_list[i - 1]) / (ratio - 1)

        bs_new[s] = T_list[-1]
    return bs_new


def _compute_density_quantities(mctdh, C):
    """Compute state-averaged RDMs and regularized inverse density matrices.

    Returns (D_invs, ds) where D_invs is a list of inverse 1-RDMs per species
    and ds is a dict of state-averaged 2-RDMs.
    """
    Ds = mctdh.dCI.get_1p_RDMs(C)
    ds = mctdh.dCI.get_2p_RDMs(C)

    Ds = [mctdh._state_average(D) for D in Ds]
    ds = {k: mctdh._state_average(v) for k, v in ds.items()}

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
_B_KWARG_KEYS_CMF = {
    "cn": ("solver_tol", "max_solver_iter"),
    "sil": ("krylov_dim",),
    "diagonalize": (),
}
_B_KWARG_KEYS_HALF = {
    "cn": ("solver_tol", "max_solver_iter", "n_orbital_substeps"),
    "sil": ("krylov_dim",),
    "bs": ("bs_sequence",),
    "diagonalize": ("orbital_x64",),
}


def get_cmf_integrator(
    mctdh,
    dt,
    c_propagator="cn",
    b_propagator="cn",
    c_kwargs=None,
    b_kwargs=None,
):
    """Constant Mean Field (CMF) integrator.

    Decouples CI and orbital propagation. Each sub-problem becomes linear
    in its own variable with the other frozen, and any of CN/SIL/
    diagonalize can solve it.

    This allows dt 100-1000× larger than Picard-iteration CN because
    the decoupled sub-problems are genuinely linear and can be solved
    to machine precision.

    Best suited for imaginary time evolution (ITE) where the dissipative
    dynamics are tolerant of the first-order CMF splitting error.

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
            'sil'         - Short Iterative Lanczos. Recommended for large
                            DVR grids where dt * max(h) >> 1.
            'diagonalize' - Dense matrix exponential of A_s (the linearised
                            substep operator). Exact in the linearised
                            ODE. Cheap when n_spatial · n_orb is small
                            (≲ a few hundred per species). Autodiff-clean.
        c_kwargs: Dict of solver-specific options for the chosen
            c_propagator. Recognised keys (defaults in parens):
              cn:           solver_tol (1e-8), max_solver_iter (100)
              sil:          krylov_dim (10)
              diagonalize:  (none)
        b_kwargs: Dict of solver-specific options for the chosen
            b_propagator. Recognised keys (defaults in parens):
              cn:           solver_tol (1e-8), max_solver_iter (100)
              sil:          krylov_dim (20)
              diagonalize:  (none)

    Returns:
        integrate: Function integrate(t, y) -> (t_new, y_new).
    """
    if c_propagator not in ("cn", "sil", "diagonalize"):
        raise ValueError(
            f"Unknown c_propagator '{c_propagator}'. "
            f"Use 'cn', 'sil', or 'diagonalize'."
        )
    if b_propagator not in ("cn", "sil", "diagonalize"):
        raise ValueError(
            f"Unknown b_propagator '{b_propagator}'. "
            f"Use 'cn', 'sil', or 'diagonalize'."
        )

    c_kwargs = dict(c_kwargs or {})
    b_kwargs = dict(b_kwargs or {})
    _validate_kwargs(c_kwargs, _C_KWARG_KEYS[c_propagator], "c_kwargs", c_propagator)
    _validate_kwargs(b_kwargs, _B_KWARG_KEYS_CMF[b_propagator], "b_kwargs", b_propagator)

    c_solver_tol = c_kwargs.get("solver_tol", 1e-8)
    c_max_iter = c_kwargs.get("max_solver_iter", 100)
    c_krylov_dim = c_kwargs.get("krylov_dim", 10)

    b_solver_tol = b_kwargs.get("solver_tol", 1e-8)
    b_max_iter = b_kwargs.get("max_solver_iter", 100)
    b_krylov_dim = b_kwargs.get("krylov_dim", 20)

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
        if c_propagator == "sil":
            C_new, _ = _sil_solve_C(
                mctdh, C, h_MO_list, g_MO_dict, dt, c_shape, c_krylov_dim
            )
        elif c_propagator == "diagonalize":
            C_new, _ = _diag_solve_C(
                mctdh, C, h_MO_list, g_MO_dict, dt, c_shape
            )
        else:
            C_new = _cn_solve_C(
                mctdh, C, h_MO_list, g_MO_dict, dt, c_shape, c_solver_tol, c_max_iter
            )

        # 3. Update density matrices from new C
        D_invs, ds = _compute_density_quantities(mctdh, C_new)

        # 4. Solve orbitals per species (frozen projector taken from bs)
        if b_propagator == "sil":
            bs_new, _ = _sil_solve_bs(
                mctdh,
                bs,
                h_current,
                D_invs,
                ds,
                bs,
                dt,
                b_shapes,
                b_krylov_dim,
            )
        elif b_propagator == "diagonalize":
            bs_new, _ = _diag_solve_bs(
                mctdh,
                bs,
                h_current,
                D_invs,
                ds,
                bs,
                dt,
                b_shapes,
            )
        else:
            bs_new = _cn_solve_bs(
                mctdh,
                bs,
                h_current,
                D_invs,
                ds,
                bs,
                dt,
                b_shapes,
                b_solver_tol,
                b_max_iter,
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
    if b_propagator not in ("cn", "bs", "sil", "diagonalize"):
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
                mctdh, C, h_MO_list_t, g_MO_dict_t, dt_half, c_shape
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
        if b_propagator == "sil":
            bs_half, _ = _sil_solve_bs(
                mctdh,
                bs,
                h_current_t,
                D_invs_t,
                ds_t,
                bs,
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
                dt_half,
                b_shapes,
                compute_x64=b_orbital_x64,
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
        if b_propagator == "sil":
            bs_new, _ = _sil_solve_bs(
                mctdh,
                bs,
                h_current_half,
                D_invs_half,
                ds_half,
                bs_half,
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
                dt,
                b_shapes,
                compute_x64=b_orbital_x64,
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
                mctdh,
                C_half,
                h_MO_list_new,
                g_MO_dict_new,
                dt_half,
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
        bs_half, _ = _sil_solve_bs(
            mctdh,
            bs,
            h_current_t0,
            D_invs_t0,
            ds_t0,
            bs,
            dt_half,
            b_shapes,
            sil_b_krylov_dim,
        )

        # ── Corrector: full step with midpoint mean fields ──────────────
        # Density matrices from C(t+dt/2)
        D_invs_half, ds_half = _compute_density_quantities(mctdh, C_half)

        # Orbital full step: b(t) → b(t+dt) using midpoint mean fields
        # (frozen projector from bs_half)
        bs_t1, err_sil_b = _sil_solve_bs(
            mctdh,
            bs,
            h_current_half,
            D_invs_half,
            ds_half,
            bs_half,
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
