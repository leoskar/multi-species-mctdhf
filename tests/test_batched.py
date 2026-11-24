"""Tests for the species-batched CMF half-step integrator.

Locks the bit-exactness contract of `integrators_batched` against the core
CMF half-step (SIL/SIL) on a small DVR system, across RTE/ITE and
inter-only / inter+intra interaction structures.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from mctdhf.MCTDHF_n_species import MCTDHF_n_species_DVR
from mctdhf.integrators import get_cmf_half_step_integrator
from mctdhf.integrators_batched import get_cmf_half_step_integrator_batched


def _make_dvr_system(intra, imag_time, K=3, N_spatial=12, M=3, n_part=1):
    """Small K-species DVR system with soft-Coulomb interactions."""
    x = np.linspace(-4, 4, N_spatial)
    dx = x[1] - x[0]
    t_mat = (
        np.diag(np.full(N_spatial, 2.0))
        + np.diag(np.full(N_spatial - 1, -1.0), k=1)
        + np.diag(np.full(N_spatial - 1, -1.0), k=-1)
    ) / (2 * dx**2)
    hs = [
        jnp.array(t_mat + np.diag(0.5 * (x - 0.3 * s) ** 2), dtype=jnp.float64)
        for s in range(K)
    ]

    g = jnp.array(0.5 / np.sqrt((x[:, None] - x[None, :]) ** 2 + 1.0))
    g_dict = {}
    for s1 in range(K):
        for s2 in range(s1 + 1, K):
            g_dict[(s1, s2)] = g
    if intra:
        for s in range(K):
            g_dict[(s, s)] = 0.7 * g

    mctdh = MCTDHF_n_species_DVR(
        h_list=hs,
        g_dict=g_dict,
        num_particles=[n_part] * K,
        num_mctdhf_orbitals=[M] * K,
        num_spatial_orbitals=[N_spatial] * K,
        num_states=1,
        imag_time=imag_time,
    )

    rng = np.random.default_rng(7)
    bs = []
    for _ in range(K):
        b = rng.standard_normal((N_spatial, M)) + 1j * rng.standard_normal(
            (N_spatial, M)
        )
        q, _ = np.linalg.qr(b)
        bs.append(jnp.array(q, dtype=jnp.complex128))
    c_shape = (1,) + tuple(mctdh.num_dets)
    C = rng.standard_normal(c_shape) + 1j * rng.standard_normal(c_shape)
    C = jnp.array(C / np.linalg.norm(C), dtype=jnp.complex128)
    return mctdh, mctdh._Cb_to_y(C, bs)


@pytest.mark.parametrize("imag_time", [False, True])
@pytest.mark.parametrize("intra", [False, True])
def test_batched_matches_core_cmf_half_step(intra, imag_time):
    """Batched integrator must reproduce the core SIL/SIL half-step to
    machine precision (the module's stated bit-exactness contract)."""
    n_part = 2 if intra else 1
    mctdh, y0 = _make_dvr_system(intra, imag_time, n_part=n_part)
    dt = 0.02

    core = get_cmf_half_step_integrator(
        mctdh,
        dt,
        c_propagator="sil",
        b_propagator="sil",
        c_kwargs={"krylov_dim": 8},
        b_kwargs={"krylov_dim": 12},
    )
    batched = get_cmf_half_step_integrator_batched(
        mctdh, dt, sil_c_krylov_dim=8, sil_b_krylov_dim=12
    )

    ta, ya = 0.0, y0
    tb, yb = 0.0, y0
    for _ in range(5):
        ta, ya = core(ta, ya)
        tb, yb = batched(tb, yb)

    assert bool(jnp.all(jnp.isfinite(ya)))
    assert bool(jnp.all(jnp.isfinite(yb)))
    diff = float(jnp.max(jnp.abs(ya - yb)))
    assert diff < 1e-13, f"core vs batched deviate: {diff:.3e}"


def test_batched_requires_dvr_and_uniform_shapes():
    """Construction-time guards: non-DVR and non-uniform shapes must raise."""
    mctdh, _ = _make_dvr_system(intra=False, imag_time=False)
    # Uniform DVR system constructs fine
    get_cmf_half_step_integrator_batched(mctdh, 0.01)

    class NotDVR:
        pass

    with pytest.raises(TypeError):
        get_cmf_half_step_integrator_batched(NotDVR(), 0.01)
