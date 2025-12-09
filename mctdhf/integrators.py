"""JAX-compatible integrators for MCTDHF.

These integrators are designed to work with newer JAX versions that have
stricter autodiff requirements.
"""
import jax
import jax.numpy as jnp
from functools import partial


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
        x_final, _, _ = jax.lax.while_loop(
            cond_fn, 
            body_fn, 
            (x_init, jnp.inf, 0)
        )
        
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
        k2 = rhs(t + dt/2, y + dt/2 * k1)
        k3 = rhs(t + dt/2, y + dt/2 * k2)
        k4 = rhs(t + dt, y + dt * k3)
        
        return t + dt, y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return integrate
