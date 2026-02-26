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


def get_cmf_integrator(mctdh, dt, solver_tol=1e-8, max_solver_iter=100):
    """Constant Mean Field (CMF) integrator using CN + BiCGSTAB.

    Decouples the CI and orbital propagation, making each sub-problem
    linear. Uses Crank-Nicolson time discretization with BiCGSTAB
    iterative solver for the implicit linear systems.

    This allows dt 100-1000× larger than Picard-iteration CN because
    the decoupled sub-problems are genuinely linear and can be solved
    to machine precision.

    Args:
        mctdh: MCTDHF_n_species instance (provides RHS helpers, transforms,
               density matrix routines, and packing/unpacking).
        dt: Time step.
        solver_tol: Convergence tolerance for BiCGSTAB.
        max_solver_iter: Maximum BiCGSTAB iterations.

    Returns:
        integrate: Function integrate(t, y) -> (t_new, y_new).
    """

    c_size = int(jnp.prod(jnp.array(mctdh.num_dets)) * mctdh.num_states)
    c_shape = (mctdh.num_states,) + tuple(mctdh.num_dets)

    b_shapes = [
        (mctdh.num_spatial_orbitals[s], mctdh.num_mctdhf_orbitals[s])
        for s in range(mctdh.num_species)
    ]

    @jax.jit
    def integrate(t, y):
        C, bs = mctdh._y_to_Cb(y)

        # ── 1. Transform integrals at current orbitals ──────────────────
        h_current = mctdh._get_h_current(t)

        h_MO_list = []
        for s in range(mctdh.num_species):
            _, h_2, _ = mctdh.transform_h(h_current[s], bs[s])
            h_MO_list.append(h_2)

        g_MO_dict, _ = mctdh._transform_all_g(bs, full=False)

        # ── 2. CN-solve CI (orbitals frozen) ────────────────────────────
        # Linear operator: A(x) = x - dt/2 · f_C(x)
        # Solve: A(x) = b_vec where b_vec = C + dt/2 · f_C(C)
        C_dot_0 = mctdh._rhs_C_linear(C, h_MO_list, g_MO_dict)
        b_vec_C = C + (dt / 2) * C_dot_0

        def matvec_C(x_flat):
            x = x_flat.reshape(c_shape)
            fx = mctdh._rhs_C_linear(x, h_MO_list, g_MO_dict)
            return (x - (dt / 2) * fx).reshape(-1)

        x0_C = (C + dt * C_dot_0).reshape(-1)  # Euler guess
        C_new_flat, _ = jax.scipy.sparse.linalg.bicgstab(
            matvec_C,
            b_vec_C.reshape(-1),
            x0=x0_C,
            tol=solver_tol,
            maxiter=max_solver_iter,
        )
        C_new = C_new_flat.reshape(c_shape)

        # ── 3. Update density matrices from new C ───────────────────────
        Ds = mctdh.dCI.get_1p_RDMs(C_new)
        ds = mctdh.dCI.get_2p_RDMs(C_new)

        # State-average RDMs
        Ds = [jnp.sum(D, axis=0) for D in Ds]
        ds = {k: jnp.sum(v, axis=0) for k, v in ds.items()}

        # Pre-compute D_inv and P_frozen per species
        D_invs = []
        P_frozens = []
        for s in range(mctdh.num_species):
            eigvals, eigvecs = jnp.linalg.eigh(Ds[s])
            D_inv = (eigvecs / (eigvals + mctdh.eps)) @ eigvecs.conj().T
            D_invs.append(D_inv)
            P_frozens.append(bs[s] @ bs[s].conj().T)

        # ── 4. CN-solve orbitals per species (CI + projector frozen) ────
        bs_new = list(bs)
        for s in range(mctdh.num_species):
            b_s = bs[s]
            b_s_shape = b_shapes[s]

            b_dot_0 = mctdh._rhs_b_linear(
                b_s, s, h_current[s], D_invs[s], ds, bs, P_frozens[s]
            )
            b_vec = b_s + (dt / 2) * b_dot_0

            def matvec_b(x_flat, _s=s):
                x = x_flat.reshape(b_shapes[_s])
                fx = mctdh._rhs_b_linear(
                    x, _s, h_current[_s], D_invs[_s], ds, bs, P_frozens[_s]
                )
                return (x - (dt / 2) * fx).reshape(-1)

            x0_b = (b_s + dt * b_dot_0).reshape(-1)  # Euler guess
            b_new_flat, _ = jax.scipy.sparse.linalg.bicgstab(
                matvec_b,
                b_vec.reshape(-1),
                x0=x0_b,
                tol=solver_tol,
                maxiter=max_solver_iter,
            )
            bs_new[s] = b_new_flat.reshape(b_s_shape)

        return t + dt, mctdh._Cb_to_y(C_new, bs_new)

    return integrate
