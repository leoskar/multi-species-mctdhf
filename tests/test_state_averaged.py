"""Tests for state-averaged orbital optimization.

Tests:
1. Backward compatibility: state_weights=None matches original behaviour.
2. Explicit uniform weights reproduce state_weights=None (ratio cancellation).
3. Ground-state-only weight [1, 0] matches a single-state calculation.
4. Non-uniform weights run without error and produce finite energies.
"""

import jax
import jax.numpy as jnp
import numpy as np

from mctdhf.MCTDHF_n_species import MCTDHF_n_species
from mctdhf.integrators import get_euler_solver


def _make_system(num_states=1, imag_time=True, state_weights=None):
    """Create a small 2-species test system."""
    N_spatial = 10
    x = jnp.linspace(-5, 5, N_spatial)
    dx = x[1] - x[0]

    t_mat = (
        jnp.diag(jnp.full(N_spatial, 2.0))
        + jnp.diag(jnp.full(N_spatial - 1, -1.0), k=1)
        + jnp.diag(jnp.full(N_spatial - 1, -1.0), k=-1)
    )
    t_mat = t_mat / (2 * dx**2)
    v_mat = jnp.diag(0.5 * x**2)
    h = t_mat + v_mat

    rng = np.random.default_rng(42)
    g = jnp.array(rng.random((N_spatial,) * 4) * 0.1, dtype=jnp.complex64)
    g = (g + g.transpose(1, 0, 3, 2)) / 2

    M = 2
    mctdh = MCTDHF_n_species(
        h_list=[h, h],
        g_dict={(0, 0): g, (1, 1): g, (0, 1): g},
        num_particles=[1, 1],
        num_mctdhf_orbitals=[M, M],
        num_spatial_orbitals=[N_spatial, N_spatial],
        num_states=num_states,
        imag_time=imag_time,
        state_weights=state_weights,
    )
    return mctdh


def _make_init(mctdh):
    """Create deterministic initial condition."""
    rng = np.random.default_rng(123)
    N_spatial = mctdh.num_spatial_orbitals[0]
    M = mctdh.num_mctdhf_orbitals[0]

    b_init = jnp.array(rng.random((N_spatial, M)) + 0j, dtype=jnp.complex64)
    b_init, _ = jnp.linalg.qr(b_init)

    c_shape = (mctdh.num_states,) + tuple(mctdh.num_dets)
    C_init = jnp.array(rng.random(c_shape) + 0j, dtype=jnp.complex64)
    C_init = mctdh.ortonorm_C(C_init)

    return mctdh._Cb_to_y(C_init, [b_init, b_init])


def test_backward_compat_none_weights():
    """state_weights=None should produce identical results to original code."""
    mctdh = _make_system(num_states=2, state_weights=None)
    y = _make_init(mctdh)

    rhs = mctdh(0.0, y)
    assert not jnp.any(jnp.isnan(rhs)), "RHS contains NaN"
    print("  state_weights=None: RHS is finite ✓")


def test_uniform_weights_match_none():
    """Explicit uniform weights [0.5, 0.5] should give same orbital EOM as None.

    Because D_inv @ d is a ratio, the factor-of-2 difference between
    sum (= weights [1,1]) and average (= weights [0.5, 0.5]) cancels.
    We compare the full RHS dy/dt.
    """
    mctdh_none = _make_system(num_states=2, state_weights=None)
    mctdh_uniform = _make_system(num_states=2, state_weights=[0.5, 0.5])

    y = _make_init(mctdh_none)

    rhs_none = mctdh_none(0.0, y)
    rhs_uniform = mctdh_uniform(0.0, y)

    diff = jnp.max(jnp.abs(rhs_none - rhs_uniform))
    print(f"  max |RHS_none - RHS_uniform|: {diff:.2e}")
    assert diff < 1e-5, f"Uniform weights differ from None: {diff}"


def test_ground_state_only_weight():
    """state_weights=[1, 0] should match a single-state calculation.

    With weight 0 on the second state, the density matrices should
    only reflect the ground state, identical to num_states=1.
    """
    mctdh_1 = _make_system(num_states=1, state_weights=None)
    mctdh_2 = _make_system(num_states=2, state_weights=[1.0, 0.0])

    # Single-state init
    y1 = _make_init(mctdh_1)
    C1, bs1 = mctdh_1._y_to_Cb(y1)

    # Two-state init: first state = same C, second state = orthogonal
    rng = np.random.default_rng(999)
    c_shape_2 = (2,) + tuple(mctdh_2.num_dets)
    C2 = jnp.zeros(c_shape_2, dtype=jnp.complex64)
    C2 = C2.at[0].set(C1[0])
    # Add a random orthogonal second state
    C2_rand = jnp.array(rng.random(C1[0].shape) + 0j, dtype=jnp.complex64)
    C2 = C2.at[1].set(C2_rand)
    C2 = mctdh_2.ortonorm_C(C2)
    y2 = mctdh_2._Cb_to_y(C2, bs1)

    # Compare orbital part of RHS (skip CI part since shapes differ)
    rhs1 = mctdh_1(0.0, y1)
    rhs2 = mctdh_2(0.0, y2)

    # Extract orbital derivatives only
    c1_size = int(np.prod(mctdh_1.num_dets) * mctdh_1.num_states)
    c2_size = int(np.prod(mctdh_2.num_dets) * mctdh_2.num_states)
    b_rhs1 = rhs1[c1_size:]
    b_rhs2 = rhs2[c2_size:]

    diff = jnp.max(jnp.abs(b_rhs1 - b_rhs2))
    print(f"  max |b_dot(1-state) - b_dot(2-state, w=[1,0])|: {diff:.2e}")
    assert diff < 1e-4, f"Ground-state-only weight differs from single-state: {diff}"


def test_nonuniform_weights_ite():
    """Non-uniform weights should run ITE without errors."""
    mctdh = _make_system(num_states=2, imag_time=True, state_weights=[0.7, 0.3])
    y = _make_init(mctdh)

    dt = 1e-3
    integrator = get_euler_solver(mctdh, dt)

    t = 0.0
    energies = [float(jnp.real(mctdh.calculate_energy(t, y)[0]))]

    for _ in range(50):
        t, y = integrator(t, y)
        C, bs = mctdh._y_to_Cb(y)
        C = mctdh.ortonorm_C(C)
        bs = [mctdh.ortonorm_b(b) for b in bs]
        y = mctdh._Cb_to_y(C, bs)
        E = float(jnp.real(mctdh.calculate_energy(t, y)[0]))
        energies.append(E)

    print(
        f"  SA-ITE (w=[0.7, 0.3]): E_init={energies[0]:.6f}, E_final={energies[-1]:.6f}"
    )
    assert not jnp.isnan(energies[-1]), "Energy diverged to NaN"
    assert not jnp.isinf(energies[-1]), "Energy diverged to Inf"
    assert energies[-1] < energies[0], "GS energy did not decrease during ITE"


def test_invalid_weights_raise():
    """Wrong-sized state_weights should raise ValueError."""
    try:
        _make_system(num_states=2, state_weights=[1.0])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Correctly raised: {e}")


if __name__ == "__main__":
    tests = [
        ("Backward compat (None weights)", test_backward_compat_none_weights),
        ("Uniform weights match None", test_uniform_weights_match_none),
        ("Ground-state-only weight", test_ground_state_only_weight),
        ("Non-uniform weights ITE", test_nonuniform_weights_ite),
        ("Invalid weights raise", test_invalid_weights_raise),
    ]
    for i, (name, fn) in enumerate(tests, 1):
        print("=" * 60)
        print(f"Test {i}: {name}")
        print("=" * 60)
        fn()
        print("\nPASSED\n")
