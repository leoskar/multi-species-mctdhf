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
    P_frozens,
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
            b_s, s, h_current[s], D_invs[s], ds, bs_frozen, P_frozens[s]
        )
        b_vec = b_s + (dt_sub / 2) * b_dot_0

        def matvec(x_flat, _s=s):
            x = x_flat.reshape(b_shapes[_s])
            fx = mctdh._rhs_b_linear(
                x, _s, h_current[_s], D_invs[_s], ds, bs_frozen, P_frozens[_s]
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

    def lanczos_step(carry, j):
        V, v_prev, v_curr, beta_prev = carry
        # Store v_curr as basis vector j (at step start, not end,
        # to avoid overwriting row 0 on the last iteration)
        V = V.at[j].set(v_curr)
        w = apply_H(v_curr)
        alpha = jnp.real(jnp.dot(v_curr.conj(), w))
        w = w - alpha * v_curr - beta_prev * v_prev
        # Full re-orthogonalization against all stored basis vectors.
        # Rows beyond j are zero so the dot products vanish — no
        # masking needed, and the matmul is on a small (krylov_dim, dim)
        # matrix which stays in GPU cache.
        overlaps = jnp.dot(V.conj(), w)  # (krylov_dim,)
        w = w - jnp.dot(overlaps, V)  # (dim,)
        beta = jnp.sqrt(jnp.real(jnp.dot(w.conj(), w)))
        safe_beta = jnp.where(beta > 1e-15, beta, 1.0)
        v_next = w / safe_beta
        return (V, v_curr, v_next, beta), (alpha, beta)

    init = (V_init, jnp.zeros_like(v0), v0, jnp.array(0.0, dtype=c_flat.real.dtype))
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

    # Reconstruct: C_new = ||C|| * V^T @ coeffs (V is orthonormal)
    c_new_flat = safe_norm * jnp.dot(coeffs, V_final)
    c_new_flat = jnp.where(norm_c > 0, c_new_flat, c_flat)

    return c_new_flat.reshape(c_shape)


def _sil_solve_bs(
    mctdh,
    bs_start,
    h_current,
    D_invs,
    ds,
    bs_frozen,
    P_frozens,
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
                b, _s, h_current[_s], D_invs[_s], ds, bs_frozen, P_frozens[_s]
            )
            # _rhs_b_linear returns -1j*A*b (RTE) or -A*b (ITE)
            # We need A*b for the Lanczos (Hermitian operator)
            if mctdh.imag_time:
                return (-rhs).reshape(-1)
            else:
                return (1j * rhs).reshape(-1)

        v0 = b_flat / safe_norm

        V_init = jnp.zeros((krylov_dim, dim), dtype=b_flat.dtype)

        def lanczos_step(carry, j):
            V, v_prev, v_curr, beta_prev = carry
            V = V.at[j].set(v_curr)
            w = apply_A(v_curr)
            alpha = jnp.real(jnp.dot(v_curr.conj(), w))
            w = w - alpha * v_curr - beta_prev * v_prev
            # Full re-orthogonalization
            overlaps = jnp.dot(V.conj(), w)
            w = w - jnp.dot(overlaps, V)
            beta = jnp.sqrt(jnp.real(jnp.dot(w.conj(), w)))
            safe_beta = jnp.where(beta > 1e-15, beta, 1.0)
            v_next = w / safe_beta
            return (V, v_curr, v_next, beta), (alpha, beta)

        init = (V_init, jnp.zeros_like(v0), v0, jnp.array(0.0, dtype=b_flat.real.dtype))
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

        b_new_flat = safe_norm * jnp.dot(coeffs, V_final)
        b_new_flat = jnp.where(norm_b > 0, b_new_flat, b_flat)
        bs_new[s] = b_new_flat.reshape(b_shapes[s])

    return bs_new


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
    P_frozens,
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
                b, _s, h_current[_s], D_invs[_s], ds, bs_frozen, P_frozens[_s]
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


def get_cmf_integrator(mctdh, dt, solver_tol=1e-8, max_solver_iter=100):
    """Constant Mean Field (CMF) integrator using CN + BiCGSTAB.

    Decouples the CI and orbital propagation, making each sub-problem
    linear. Uses Crank-Nicolson time discretization with BiCGSTAB
    iterative solver for the implicit linear systems.

    This allows dt 100-1000× larger than Picard-iteration CN because
    the decoupled sub-problems are genuinely linear and can be solved
    to machine precision.

    Best suited for imaginary time evolution (ITE) where the dissipative
    dynamics are tolerant of the first-order CMF splitting error.

    Args:
        mctdh: MCTDHF_n_species instance.
        dt: Time step.
        solver_tol: Convergence tolerance for BiCGSTAB.
        max_solver_iter: Maximum BiCGSTAB iterations.

    Returns:
        integrate: Function integrate(t, y) -> (t_new, y_new).
    """
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

        # 2. CN-solve CI (orbitals frozen)
        C_new = _cn_solve_C(
            mctdh, C, h_MO_list, g_MO_dict, dt, c_shape, solver_tol, max_solver_iter
        )

        # 3. Update density matrices from new C
        D_invs, ds = _compute_density_quantities(mctdh, C_new)
        P_frozens = [bs[s] @ bs[s].conj().T for s in range(mctdh.num_species)]

        # 4. CN-solve orbitals per species
        bs_new = _cn_solve_bs(
            mctdh,
            bs,
            h_current,
            D_invs,
            ds,
            bs,
            P_frozens,
            dt,
            b_shapes,
            solver_tol,
            max_solver_iter,
        )

        return t + dt, mctdh._Cb_to_y(C_new, bs_new)

    return integrate


def get_cmf_half_step_integrator(
    mctdh,
    dt,
    solver_tol=1e-8,
    max_solver_iter=100,
    n_orbital_substeps=1,
    c_propagator="cn",
    b_propagator="cn",
    sil_krylov_dim=10,
    sil_b_krylov_dim=20,
    bs_sequence=(2, 4, 6, 8, 12, 16),
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
        solver_tol: Convergence tolerance for BiCGSTAB (CN only).
        max_solver_iter: Maximum BiCGSTAB iterations (CN only).
        n_orbital_substeps: Number of sub-steps for the orbital CN solves.
            The mean fields (density matrices, projectors, MO integrals)
            remain frozen; only the orbital time integration is refined.
            Useful when the orbital EOM is stiffer than the CI EOM (e.g.
            large spatial grids). Default 1 (no sub-stepping). Only used
            when b_propagator='cn'.
        c_propagator: Sub-integrator for CI coefficients. Options:
            'cn' - Crank-Nicolson with BiCGSTAB (default)
            'sil' - Short Iterative Lanczos (Krylov subspace + matrix exp)
        b_propagator: Sub-integrator for orbitals. Options:
            'cn' - Crank-Nicolson with BiCGSTAB (default)
            'bs' - Bulirsch-Stoer (modified midpoint + Richardson extrapolation)
            'sil' - Short Iterative Lanczos (Krylov subspace + matrix exp).
                Recommended for large DVR grids where dt * max(h) >> 1.
        sil_krylov_dim: Krylov subspace dimension for CI SIL (default 10).
        sil_b_krylov_dim: Krylov subspace dimension for orbital SIL (default 20).
            Needs to be larger than sil_krylov_dim because the orbital
            Hamiltonian in the spatial basis is much stiffer than the CI
            Hamiltonian in the MO basis.
        bs_sequence: Subdivision sequence for Bulirsch-Stoer extrapolation.
            Default (2, 4, 6, 8, 12, 16).

    Returns:
        integrate: Function integrate(t, y) -> (t_new, y_new).
    """
    if c_propagator not in ("cn", "sil"):
        raise ValueError(f"Unknown c_propagator '{c_propagator}'. Use 'cn' or 'sil'.")
    if b_propagator not in ("cn", "bs", "sil"):
        raise ValueError(
            f"Unknown b_propagator '{b_propagator}'. Use 'cn', 'bs', or 'sil'."
        )
    c_shape = (mctdh.num_states,) + tuple(mctdh.num_dets)
    b_shapes = [
        (mctdh.num_spatial_orbitals[s], mctdh.num_mctdhf_orbitals[s])
        for s in range(mctdh.num_species)
    ]
    dt_half = dt / 2
    n_sub = n_orbital_substeps

    # Claude Code implementation
    @jax.jit
    def integrate_cc(t, y):
        C, bs = mctdh._y_to_Cb(y)

        # ── Predictor: half-step with mean fields at t ──────────────────
        h_current_t = mctdh._get_h_current(t)
        h_current_mid = mctdh._get_h_current(t + dt_half)

        # Transform integrals at b(t)
        h_MO_list_t = [
            mctdh.transform_h(h_current_t[s], bs[s])[1]
            for s in range(mctdh.num_species)
        ]
        g_MO_dict_t, _ = mctdh._transform_all_g(bs, full=False)

        # CI half-step: C(t) → C(t+dt/2)
        if c_propagator == "sil":
            C_half = _sil_solve_C(
                mctdh, C, h_MO_list_t, g_MO_dict_t, dt_half, c_shape, sil_krylov_dim
            )
        else:
            C_half = _cn_solve_C(
                mctdh,
                C,
                h_MO_list_t,
                g_MO_dict_t,
                dt_half,
                c_shape,
                solver_tol,
                max_solver_iter,
            )

        # Density matrices from C(t+dt/2)
        D_invs_half, ds_half = _compute_density_quantities(mctdh, C_half)
        P_frozens_t = [bs[s] @ bs[s].conj().T for s in range(mctdh.num_species)]

        # Orbital half-step: b(t) → b(t+dt/2)
        if b_propagator == "sil":
            bs_half = _sil_solve_bs(
                mctdh,
                bs,
                h_current_t,
                D_invs_half,
                ds_half,
                bs,
                P_frozens_t,
                dt_half,
                b_shapes,
                sil_b_krylov_dim,
            )
        elif b_propagator == "bs":
            bs_half = _bs_solve_bs(
                mctdh,
                bs,
                h_current_t,
                D_invs_half,
                ds_half,
                bs,
                P_frozens_t,
                dt_half,
                b_shapes,
                bs_sequence,
            )
        else:
            dt_orb_pred = dt_half / n_sub
            bs_half = bs
            for _ in range(n_sub):
                bs_half = _cn_solve_bs(
                    mctdh,
                    bs_half,
                    h_current_t,
                    D_invs_half,
                    ds_half,
                    bs,
                    P_frozens_t,
                    dt_orb_pred,
                    b_shapes,
                    solver_tol,
                    max_solver_iter,
                )

        # ── Corrector: full step with midpoint mean fields ──────────────
        # Re-transform integrals at b(t+dt/2)
        # Use h at t+dt/2 for the corrector (midpoint-centered)
        h_MO_list_mid = [
            mctdh.transform_h(h_current_mid[s], bs_half[s])[1]
            for s in range(mctdh.num_species)
        ]
        g_MO_dict_mid, _ = mctdh._transform_all_g(bs_half, full=False)

        # CI full step: C(t) → C(t+dt) using midpoint MO integrals
        if c_propagator == "sil":
            C_new = _sil_solve_C(
                mctdh, C, h_MO_list_mid, g_MO_dict_mid, dt, c_shape, sil_krylov_dim
            )
        else:
            C_new = _cn_solve_C(
                mctdh,
                C,
                h_MO_list_mid,
                g_MO_dict_mid,
                dt,
                c_shape,
                solver_tol,
                max_solver_iter,
            )

        # Density matrices from C(t+dt)
        D_invs_new, ds_new = _compute_density_quantities(mctdh, C_new)
        P_frozens_mid = [
            bs_half[s] @ bs_half[s].conj().T for s in range(mctdh.num_species)
        ]

        # Orbital full step: b(t) → b(t+dt) using midpoint mean fields
        if b_propagator == "sil":
            bs_new = _sil_solve_bs(
                mctdh,
                bs,
                h_current_mid,
                D_invs_new,
                ds_new,
                bs_half,
                P_frozens_mid,
                dt,
                b_shapes,
                sil_b_krylov_dim,
            )
        elif b_propagator == "bs":
            bs_new = _bs_solve_bs(
                mctdh,
                bs,
                h_current_mid,
                D_invs_new,
                ds_new,
                bs_half,
                P_frozens_mid,
                dt,
                b_shapes,
                bs_sequence,
            )
        else:
            dt_orb_corr = dt / n_sub
            bs_new = bs
            for _ in range(n_sub):
                bs_new = _cn_solve_bs(
                    mctdh,
                    bs_new,
                    h_current_mid,
                    D_invs_new,
                    ds_new,
                    bs_half,
                    P_frozens_mid,
                    dt_orb_corr,
                    b_shapes,
                    solver_tol,
                    max_solver_iter,
                )

        return t + dt, mctdh._Cb_to_y(C_new, bs_new)

    # Fixed (?) implementation
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
            C_half = _sil_solve_C(
                mctdh, C, h_MO_list_t, g_MO_dict_t, dt_half, c_shape, sil_krylov_dim
            )
        else:
            C_half = _cn_solve_C(
                mctdh,
                C,
                h_MO_list_t,
                g_MO_dict_t,
                dt_half,
                c_shape,
                solver_tol,
                max_solver_iter,
            )

        # Density matrices from C(t)
        D_invs_t, ds_t = _compute_density_quantities(mctdh, C)
        P_frozens_t = [bs[s] @ bs[s].conj().T for s in range(mctdh.num_species)]

        # Orbital half-step: b(t) → b(t+dt/2)
        if b_propagator == "sil":
            bs_half = _sil_solve_bs(
                mctdh,
                bs,
                h_current_t,
                D_invs_t,
                ds_t,
                bs,
                P_frozens_t,
                dt_half,
                b_shapes,
                sil_b_krylov_dim,
            )
        elif b_propagator == "bs":
            bs_half = _bs_solve_bs(
                mctdh,
                bs,
                h_current_t,
                D_invs_t,
                ds_t,
                bs,
                P_frozens_t,
                dt_half,
                b_shapes,
                bs_sequence,
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
                    P_frozens_t,
                    dt_orb_pred,
                    b_shapes,
                    solver_tol,
                    max_solver_iter,
                )

        # ── Corrector: full step with midpoint mean fields ──────────────
        # Re-transform integrals at b(t+dt/2)
        # Use h at t+dt/2 for the corrector (midpoint-centered)
        h_MO_list_half = [
            mctdh.transform_h(h_current_half[s], bs_half[s])[1]
            for s in range(mctdh.num_species)
        ]
        g_MO_dict_half, _ = mctdh._transform_all_g(bs_half, full=False)

        # Density matrices from C(t+dt/2)
        D_invs_half, ds_half = _compute_density_quantities(mctdh, C_half)
        P_frozens_half = [
            bs_half[s] @ bs_half[s].conj().T for s in range(mctdh.num_species)
        ]

        # Orbital full step: b(t) → b(t+dt) using midpoint mean fields
        if b_propagator == "sil":
            bs_new = _sil_solve_bs(
                mctdh,
                bs,
                h_current_half,
                D_invs_half,
                ds_half,
                bs_half,
                P_frozens_half,
                dt,
                b_shapes,
                sil_b_krylov_dim,
            )
        elif b_propagator == "bs":
            bs_new = _bs_solve_bs(
                mctdh,
                bs,
                h_current_half,
                D_invs_half,
                ds_half,
                bs_half,
                P_frozens_half,
                dt,
                b_shapes,
                bs_sequence,
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
                    P_frozens_half,
                    dt_orb_corr,
                    b_shapes,
                    solver_tol,
                    max_solver_iter,
                )

        h_current_new = mctdh._get_h_current(t + dt)

        # Transform integrals at b(t)
        h_MO_list_new = [
            mctdh.transform_h(h_current_new[s], bs_new[s])[1]
            for s in range(mctdh.num_species)
        ]
        g_MO_dict_new, _ = mctdh._transform_all_g(bs_new, full=False)

        # CI full step: C(t) → C(t+dt) using midpoint MO integrals
        if c_propagator == "sil":
            C_new = _sil_solve_C(
                mctdh,
                C_half,
                h_MO_list_new,
                g_MO_dict_new,
                dt_half,
                c_shape,
                sil_krylov_dim,
            )
        else:
            C_new = _cn_solve_C(
                mctdh,
                C_half,
                h_MO_list_new,
                g_MO_dict_new,
                dt_half,
                c_shape,
                solver_tol,
                max_solver_iter,
            )

        return t + dt, mctdh._Cb_to_y(C_new, bs_new)

    return integrate


def get_vmf_integrator(mctdh, dt, num_newton=3, solver_tol=1e-8, max_solver_iter=50):
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

    Args:
        mctdh: MCTDHF_n_species instance. Its __call__ method is used as
               the full (coupled) RHS.
        dt: Time step.
        num_newton: Number of Newton iterations per step.
        solver_tol: BiCGSTAB convergence tolerance for Newton linear systems.
        max_solver_iter: Maximum BiCGSTAB iterations per Newton step.

    Returns:
        integrate: Function integrate(t, y) -> (t_new, y_new).
    """
    t_half = dt / 2

    @jax.jit
    def integrate(t, y):
        t_mid = t + t_half

        # Initial guess: explicit Euler
        f0 = mctdh(t, y)
        z = y + dt * f0

        # Newton iterations to solve the implicit midpoint equation:
        #   G(z) = z - y - dt * f(t_mid, (z + y)/2) = 0
        #
        # Newton update: G'(z) · δz = -G(z)
        #   G'(z) · v = v - (dt/2) · J_f(y_mid) · v
        # where J_f · v is computed via jax.jvp.
        def newton_step(z, _):
            y_mid = (z + y) / 2
            f_mid = mctdh(t_mid, y_mid)
            residual = z - y - dt * f_mid

            def matvec(v):
                _, Jv = jax.jvp(lambda yy: mctdh(t_mid, yy), (y_mid,), (v,))
                return v - t_half * Jv

            dz, _ = jax.scipy.sparse.linalg.bicgstab(
                matvec,
                -residual,
                x0=jnp.zeros_like(residual),
                tol=solver_tol,
                maxiter=max_solver_iter,
            )
            return z + dz, None

        z, _ = jax.lax.scan(newton_step, z, None, length=num_newton)

        return t + dt, z

    return integrate


def get_psi_integrator(mctdh, dt):
    """Projector-Splitting Integrator (PSI) for 1-particle-per-species MCTDHF.

    Implements a first-order Lie-Trotter splitting that bypasses the density
    matrix inverse D⁻¹. Provides exact norm conservation (up to time
    discretization) via QR decomposition.

    Requires num_states=1 and all M_s equal (Tucker rank constraint).

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
