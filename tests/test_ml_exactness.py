"""ML-MCTDHF reduces to standard MCTDHF at full rank.

With ``num_spfs = num_dets`` and ``B_s = 1`` the SPF layer is the identity,
``(1 − P_f) ≡ 0`` so ``Ḃ_s = 0``, and every ML step must reproduce the
corresponding standard step. This module checks that for both ML
integrators across imaginary and real time, DVR and 4-index integrals, one
and several states, zero and nonzero ``e_shift``, and sub-cycled orbital
steps — plus the ``from_mctdhf`` / ``to_mctdhf_y`` round trip.

Agreement is not bit-for-bit: the ML top-layer operators are the same
numbers reached by a different contraction order (``B† Op B`` and the
SPF-space pair contraction), so they differ from ``precompute_ci_ops`` at
the level of floating-point associativity, and a CMF step then propagates
that.

The tolerances are therefore written in multiples of the working dtype's
epsilon rather than as bare decimals, so they mean the same thing at any
precision. Measured on this system in complex128 (eps = 2.2e-16), the
residual between the two paths after one step is ~80 eps on the orbitals
and ~90 eps on the CI block once the eigenvector phase is removed;
``_TOL_ULP`` sits an order above that.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from mctdhf.MCTDHF_n_species import (
    MCTDHF_n_species,
    MCTDHF_n_species_DVR,
    pack_Cb_to_y,
)
from mctdhf.MLMCTDHF_n_species import (
    MLMCTDHF_n_species,
    MLMCTDHF_n_species_DVR,
)
from mctdhf.integrators import get_cmf_integrator, get_cmf_midpoint_integrator
from mctdhf.integrators_ml import (
    get_ml_cmf_integrator,
    get_ml_cmf_midpoint_integrator,
)
from mctdhf.sine_dvr import SineDVR, harmonic_potential, softened_coulomb

NUM_PARTICLES = [1, 2]
NUM_ORBITALS = [3, 4]
NUM_GRID = 8
DT = 0.02

# Tolerances in epsilons of the working dtype (see the module docstring).
_EPS = float(np.finfo(np.float64).eps)
_TOL_ULP = 4096          # one propagated CMF step, both code paths
_TOL_ULP_STATIC = 64     # single evaluations: energies, RDMs, overlaps


def _atol(ulp, scale=1.0):
    return ulp * _EPS * max(float(scale), 1.0)


# ── system builders ───────────────────────────────────────────────────────


def _dvr_ingredients():
    dvr = SineDVR(x_min=-4.0, x_max=4.0, num_points=NUM_GRID)
    h_list = [
        dvr.get_h(potential=harmonic_potential(omega=1.0, center=-1.2)),
        dvr.get_h(potential=harmonic_potential(omega=1.3, center=1.0)),
    ]
    g_dict = {
        (1, 1): dvr.get_g(softened_coulomb(kappa=0.8, a=0.5)),
        (0, 1): dvr.get_g(softened_coulomb(kappa=0.5, a=0.7)),
    }
    return h_list, g_dict


def _full_ingredients():
    """4-index integrals: the DVR ones promoted to dense rank-4 tensors."""
    h_list, g2 = _dvr_ingredients()
    g_dict = {}
    for key, g in g2.items():
        n = g.shape[0]
        full = np.zeros((n, n, n, n), dtype=np.float64)
        idx = np.arange(n)
        full[idx[:, None], idx[None, :], idx[:, None], idx[None, :]] = np.asarray(g)
        g_dict[key] = jnp.asarray(full)
    return h_list, g_dict


def build_pair(dvr, imag_time, num_states, e_shift, state_weights=None):
    """A standard solver and its full-rank ML twin on the same system."""
    h_list, g_dict = _dvr_ingredients() if dvr else _full_ingredients()
    std_cls = MCTDHF_n_species_DVR if dvr else MCTDHF_n_species
    ml_cls = MLMCTDHF_n_species_DVR if dvr else MLMCTDHF_n_species

    common = dict(
        h_list=h_list,
        g_dict=g_dict,
        num_particles=NUM_PARTICLES,
        num_mctdhf_orbitals=NUM_ORBITALS,
        num_spatial_orbitals=[NUM_GRID] * 2,
        num_states=num_states,
        imag_time=imag_time,
        state_weights=state_weights,
        e_shift=e_shift,
    )
    std = std_cls(**common)
    ml = ml_cls(num_spfs=list(std.num_dets), **common)
    return std, ml


def initial_Cb(std, seed=4242):
    rng = np.random.default_rng(seed)
    bs = []
    for s in range(std.num_species):
        shape = (std.num_spatial_orbitals[s], std.num_mctdhf_orbitals[s])
        b = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        b, _ = np.linalg.qr(b)
        bs.append(jnp.asarray(b, dtype=jnp.complex128))
    c_shape = (std.num_states,) + tuple(std.num_dets)
    C = rng.standard_normal(c_shape) + 1j * rng.standard_normal(c_shape)
    C = std.ortonorm_C(jnp.asarray(C, dtype=jnp.complex128))
    return C, bs


def identity_Bs(ml):
    return [jnp.eye(D, dtype=jnp.complex128) for D in ml.num_dets]


# ── the exactness matrix ──────────────────────────────────────────────────

CONFIGS = []
for _imag in (False, True):
    for _dvr in (True, False):
        for _ns, _w in ((1, None), (3, [0.5, 0.3, 0.2])):
            for _shift in (0.0, -2.5):
                for _nsub in (1, 4):
                    CONFIGS.append((_imag, _dvr, _ns, _w, _shift, _nsub))


def _cfg_id(cfg):
    imag, dvr, ns, _w, shift, nsub = cfg
    return (f"{'ite' if imag else 'rte'}-{'dvr' if dvr else 'g4'}"
            f"-S{ns}-sh{shift:g}-sub{nsub}")


@pytest.mark.parametrize("cfg", CONFIGS, ids=_cfg_id)
@pytest.mark.parametrize("factory", ["basic", "midpoint"])
@pytest.mark.parametrize("props", [("sil", "cn"), ("cn", "diagonalize"),
                                   ("diagonalize", "dop")],
                         ids=["sil-cn", "cn-diag", "diag-dop"])
def test_full_rank_step_matches_standard(cfg, factory, props):
    imag_time, dvr, num_states, weights, e_shift, n_sub = cfg
    top_prop, b_prop = props

    std, ml = build_pair(dvr, imag_time, num_states, e_shift, weights)
    C, bs = initial_Cb(std)
    Bs = identity_Bs(ml)

    y_std = pack_Cb_to_y(C, bs)
    y_ml = ml._ABb_to_y(C, Bs, bs)

    top_kwargs = {"krylov_dim": 8} if top_prop == "sil" else (
        {"solver_tol": 1e-12, "max_solver_iter": 200} if top_prop == "cn"
        else {})
    b_kwargs = {"n_orbital_substeps": n_sub}
    if b_prop == "cn":
        b_kwargs.update(solver_tol=1e-12, max_solver_iter=200)
    elif b_prop == "dop":
        b_kwargs.update(n_substeps=2)

    if factory == "basic":
        step_std = get_cmf_integrator(
            std, DT, c_propagator=top_prop, b_propagator=b_prop,
            c_kwargs=top_kwargs, b_kwargs=b_kwargs)
        step_ml = get_ml_cmf_integrator(
            ml, DT, a_propagator=top_prop, B_propagator="diagonalize",
            b_propagator=b_prop, a_kwargs=top_kwargs, b_kwargs=b_kwargs)
    else:
        step_std = get_cmf_midpoint_integrator(
            std, DT, c_propagator=top_prop, b_propagator=b_prop,
            c_kwargs=top_kwargs, b_kwargs=b_kwargs)
        step_ml = get_ml_cmf_midpoint_integrator(
            ml, DT, a_propagator=top_prop, B_propagator="diagonalize",
            b_propagator=b_prop, a_kwargs=top_kwargs, b_kwargs=b_kwargs)

    t_std, y_std_new = step_std(0.0, y_std)
    t_ml, y_ml_new = step_ml(0.0, y_ml)

    assert t_std == t_ml

    A_new, Bs_new, bs_new = ml._y_to_ABb(y_ml_new)
    # B must be EXACTLY unchanged: the solve is skipped at full rank.
    for s in range(ml.num_species):
        np.testing.assert_array_equal(np.asarray(Bs_new[s]),
                                      np.asarray(Bs[s]))

    C_ml, bs_ml = std._y_to_Cb(jnp.asarray(ml.to_mctdhf_y(y_ml_new)))
    C_ref, bs_ref = std._y_to_Cb(y_std_new)

    for s in range(std.num_species):
        np.testing.assert_allclose(
            np.asarray(bs_ml[s]), np.asarray(bs_ref[s]),
            rtol=0, atol=_atol(_TOL_ULP), err_msg=f"orbitals[{s}]")

    Cm = np.asarray(C_ml).reshape(num_states, -1)
    Cr = np.asarray(C_ref).reshape(num_states, -1)

    if imag_time and top_prop == "diagonalize":
        # In imaginary time this propagator REPLACES the top block by the
        # lowest eigenvectors of the top Hamiltonian, and an eigenvector is
        # defined only up to a phase — `eigh` pins it by the backend's
        # internal normalisation, nothing physical. Two matrices agreeing to
        # 1e-13 can come back with per-state phases differing by ~1e-8 rad
        # (measured on cuSOLVER), which is a 1e-8 elementwise difference in
        # a state that is otherwise identical to 1e-14.
        #
        # So compare phase-invariantly: the overlap matrix must be the
        # identity (no mixing, no lost norm), and the states must agree
        # elementwise once that one phase per state is removed. This is
        # strictly stronger than an elementwise check at a loosened
        # tolerance — it still catches any real difference at _TOL_ULP.
        S = Cm.conj() @ Cr.T
        np.testing.assert_allclose(
            np.abs(S), np.eye(num_states), rtol=0, atol=_atol(_TOL_ULP),
            err_msg="top-block overlap matrix is not the identity")
        Cm = Cm * np.exp(1j * np.angle(np.diag(S)))[:, None]

    np.testing.assert_allclose(Cm, Cr, rtol=0, atol=_atol(_TOL_ULP),
                               err_msg="top block")


@pytest.mark.parametrize("imag_time", [False, True], ids=["rte", "ite"])
def test_full_rank_energy_matches_standard(imag_time):
    std, ml = build_pair(True, imag_time, 2, -1.0, [0.7, 0.3])
    C, bs = initial_Cb(std)
    y_std = pack_Cb_to_y(C, bs)
    y_ml = ml._ABb_to_y(C, identity_Bs(ml), bs)

    np.testing.assert_allclose(
        np.asarray(ml.calculate_energy(0.0, y_ml)),
        np.asarray(std.calculate_energy(0.0, y_std)),
        rtol=0, atol=_atol(_TOL_ULP_STATIC, 1e2),
    )
    assert abs(ml.suggest_e_shift(0.0, y_ml)
               - std.suggest_e_shift(0.0, y_std)) < 1e-10

    np.testing.assert_allclose(
        np.asarray(ml.calculate_overlap(y_ml, y_ml)),
        np.asarray(std.calculate_overlap(y_std, y_std)),
        rtol=0, atol=_atol(_TOL_ULP_STATIC, 1e2),
    )
    for s, (D_ml, D_std) in enumerate(
            zip(ml.get_1p_RDMs(y_ml), std.dCI.get_1p_RDMs(C))):
        np.testing.assert_allclose(np.asarray(D_ml), np.asarray(D_std),
                                   rtol=0, atol=_atol(_TOL_ULP_STATIC, 1e2),
                                   err_msg=f"D[{s}]")


# ── converters ────────────────────────────────────────────────────────────


def test_expand_C_matches_reference():
    from tests.test_ml_layers import expand_C_ref

    _std, ml = build_pair(True, False, 2, 0.0, None)
    rng = np.random.default_rng(11)
    Bs = []
    for s in range(ml.num_species):
        M = (rng.standard_normal(ml.B_shapes[s])
             + 1j * rng.standard_normal(ml.B_shapes[s]))
        Q, _ = np.linalg.qr(M)
        Bs.append(jnp.asarray(Q))
    A = jnp.asarray(rng.standard_normal(ml.a_shape)
                    + 1j * rng.standard_normal(ml.a_shape))
    np.testing.assert_allclose(np.asarray(ml.expand_C(A, Bs)),
                               expand_C_ref(A, Bs), rtol=0, atol=1e-13)


def test_from_mctdhf_round_trip_at_full_rank():
    std, ml = build_pair(True, False, 2, 0.0, [0.6, 0.4])
    C, bs = initial_Cb(std)

    A, Bs, bs_out, discarded = ml.from_mctdhf(C, bs)
    for s in range(ml.num_species):
        assert A.shape == ml.a_shape
        assert Bs[s].shape == ml.B_shapes[s]
        # B must be unitary at full rank
        np.testing.assert_allclose(
            np.asarray(Bs[s].conj().T @ Bs[s]),
            np.eye(ml.num_spfs[s]), rtol=0, atol=1e-11)
        assert float(np.real(discarded[s])) < 1e-10

    y_ml = ml._ABb_to_y(A, Bs, bs_out)
    np.testing.assert_allclose(np.asarray(ml.to_mctdhf_y(y_ml)),
                               np.asarray(pack_Cb_to_y(C, bs)),
                               rtol=0, atol=1e-11)


def test_from_mctdhf_truncation_discards_the_reported_weight():
    """The reported discarded weight is exactly the norm the truncation
    loses, when C is normalised per state and the weights are the solver's."""
    std, _ml_full = build_pair(True, False, 2, 0.0, [0.6, 0.4])
    C, bs = initial_Cb(std)
    C = std.normalize_C(C)

    _h, g = _dvr_ingredients()
    ml = MLMCTDHF_n_species_DVR(
        h_list=_h, g_dict=g, num_particles=NUM_PARTICLES,
        num_mctdhf_orbitals=NUM_ORBITALS, num_spatial_orbitals=[NUM_GRID] * 2,
        num_states=2, state_weights=[0.6, 0.4], num_spfs=[2, 3],
    )
    A, Bs, _bs, discarded = ml.from_mctdhf(C, bs)

    w = np.asarray(ml.state_weights)
    C_np = np.asarray(C)
    for s in range(ml.num_species):
        # Weight kept on species s alone: project C onto B_s and compare the
        # state-averaged norms.
        proj = np.tensordot(C_np, np.asarray(Bs[s]).conj(), axes=([s + 1], [0]))
        kept = np.sum(w * np.sum(np.abs(proj) ** 2,
                                 axis=tuple(range(1, proj.ndim))))
        total = np.sum(w * np.sum(np.abs(C_np) ** 2,
                                  axis=tuple(range(1, C_np.ndim))))
        np.testing.assert_allclose(float(np.real(discarded[s])),
                                   1.0 - kept / total, rtol=0, atol=1e-10,
                                   err_msg=f"species {s}")


def test_from_species_vectors_preserves_the_state():
    std, ml = build_pair(True, False, 2, 0.0, None)
    rng = np.random.default_rng(5)
    vecs = []
    for s in range(ml.num_species):
        V = (rng.standard_normal(ml.B_shapes[s])
             + 1j * rng.standard_normal(ml.B_shapes[s]))
        vecs.append(jnp.asarray(V))
    A = jnp.asarray(rng.standard_normal(ml.a_shape)
                    + 1j * rng.standard_normal(ml.a_shape))

    A_new, Bs = ml.from_species_vectors(vecs, A)

    for s in range(ml.num_species):
        np.testing.assert_allclose(
            np.asarray(Bs[s].conj().T @ Bs[s]),
            np.eye(ml.num_spfs[s]), rtol=0, atol=1e-11,
            err_msg=f"B[{s}] not orthonormal")

    from tests.test_ml_layers import expand_C_ref
    np.testing.assert_allclose(expand_C_ref(A_new, Bs), expand_C_ref(A, vecs),
                               rtol=0, atol=1e-10)


def test_y_to_Cb_raises_on_ml_states():
    _std, ml = build_pair(True, False, 1, 0.0, None)
    with pytest.raises(TypeError, match="no CI tensor"):
        ml._y_to_Cb(jnp.zeros(1))


def test_num_spfs_is_validated():
    h_list, g_dict = _dvr_ingredients()
    common = dict(
        h_list=h_list, g_dict=g_dict, num_particles=NUM_PARTICLES,
        num_mctdhf_orbitals=NUM_ORBITALS,
        num_spatial_orbitals=[NUM_GRID] * 2, num_states=1,
    )
    with pytest.raises(TypeError, match="num_spfs is required"):
        MLMCTDHF_n_species_DVR(**common)
    with pytest.raises(ValueError, match="out of range"):
        MLMCTDHF_n_species_DVR(num_spfs=[1, 99], **common)
    with pytest.raises(ValueError, match="out of range"):
        MLMCTDHF_n_species_DVR(num_spfs=[0, 2], **common)
    with pytest.raises(ValueError, match="must have 2 entries"):
        MLMCTDHF_n_species_DVR(num_spfs=[2], **common)


# ── integrator option handling ────────────────────────────────────────────


def test_ml_integrator_kwargs_are_validated():
    from mctdhf.integrators_ml import _SPF_KWARG_KEYS

    _std, ml = build_pair(True, False, 1, 0.0, None)

    with pytest.raises(ValueError, match="Unknown a_propagator"):
        get_ml_cmf_integrator(ml, DT, a_propagator="nope")
    with pytest.raises(ValueError, match="Unknown B_propagator"):
        get_ml_cmf_integrator(ml, DT, B_propagator="nope")
    with pytest.raises(ValueError, match="Unknown b_propagator"):
        get_ml_cmf_integrator(ml, DT, b_propagator="nope")

    with pytest.raises(ValueError, match="a_kwargs contains unrecognised"):
        get_ml_cmf_integrator(ml, DT, a_propagator="sil",
                              a_kwargs={"solver_tol": 1e-8})
    with pytest.raises(ValueError, match="B_kwargs contains unrecognised"):
        get_ml_cmf_midpoint_integrator(
            ml, DT, B_kwargs={"n_orbital_substeps": 2})
    with pytest.raises(ValueError, match="b_kwargs contains unrecognised"):
        get_ml_cmf_integrator(ml, DT, b_kwargs={"n_spf_substeps": 2})

    # The SPF layer takes the orbital keys with n_orbital_substeps renamed.
    assert "n_spf_substeps" in _SPF_KWARG_KEYS["cn"]
    assert "n_orbital_substeps" not in _SPF_KWARG_KEYS["cn"]


@pytest.mark.parametrize("n_spf_substeps", [1, 3])
def test_spf_substeps_run_and_are_stationary_at_full_rank(n_spf_substeps):
    """Sub-cycling the SPF layer must not disturb a complete SPF basis."""
    _std, ml = build_pair(True, False, 1, 0.0, None)
    C, bs = initial_Cb(ml)
    Bs = identity_Bs(ml)
    step = get_ml_cmf_integrator(
        ml, DT, a_propagator="sil", B_propagator="cn", b_propagator="cn",
        a_kwargs={"krylov_dim": 6},
        B_kwargs={"n_spf_substeps": n_spf_substeps})
    _t, y_new = step(0.0, ml._ABb_to_y(C, Bs, bs))
    _A, Bs_new, _bs = ml._y_to_ABb(y_new)
    for s in range(ml.num_species):
        np.testing.assert_array_equal(np.asarray(Bs_new[s]), np.asarray(Bs[s]))


def test_e_shift_auto_resolves_before_any_jitted_trace():
    h_list, g_dict = _dvr_ingredients()
    ml = MLMCTDHF_n_species_DVR(
        h_list=h_list, g_dict=g_dict, num_particles=NUM_PARTICLES,
        num_mctdhf_orbitals=NUM_ORBITALS,
        num_spatial_orbitals=[NUM_GRID] * 2, num_states=1,
        imag_time=True, e_shift="auto", num_spfs=[2, 3],
    )
    assert ml.e_shift == "auto"
    step = get_ml_cmf_integrator(
        ml, DT, a_propagator="sil", B_propagator="diagonalize",
        b_propagator="cn", a_kwargs={"krylov_dim": 6})
    ts, ys = ml.integrate(step, 2, normalize=True)
    assert isinstance(ml.e_shift, float)
    # The reported energy adds e_shift back, so it is the physical one.
    E = float(np.real(ml.calculate_energy(float(ts[-1]), ys[-1]))[0])
    assert np.isfinite(E)
    assert abs(E - ml.e_shift) < abs(ml.e_shift) + 10.0
