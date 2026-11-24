"""Identities of the ML-MCTDHF layer operators.

Every quantity in :mod:`mctdhf.SemiDirectCI_ml` is checked two ways:

* against the determinant-space code path applied to the EXPANDED CI tensor
  ``C = A ×_s B_s`` — the ML objects must be the same physical quantities,
  only computed without ever forming ``C``;
* against the ``B_s = 1`` limit, where the SPF layer is the identity and
  every ML operator must reduce to its standard-MCTDHF counterpart.

The mean field additionally gets a fully independent dense-numpy
construction from the expanded single-hole functions.
"""

import itertools

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from mctdhf.SemiDirectCI_ml import SemiDirectCI_ML

TOL = 1e-12


# ── helpers ───────────────────────────────────────────────────────────────


def expand_C_ref(A, Bs):
    """``C[α, I_0…] = Σ_j A[α, j_0…] Π_s B_s[I_s, j_s]`` (numpy reference)."""
    C = np.asarray(A)
    for s, B in enumerate(Bs):
        B = np.asarray(B)
        # contract axis s+1 of C with axis 1 (the SPF index) of B
        C = np.tensordot(C, B, axes=([s + 1], [1]))
        # tensordot appends the determinant axis at the end; move it back
        C = np.moveaxis(C, -1, s + 1)
    return C


def _orthonormal(rng, rows, cols):
    M = rng.standard_normal((rows, cols)) + 1j * rng.standard_normal((rows, cols))
    Q, _ = np.linalg.qr(M)
    return Q


def make_case(num_particles, num_orbitals, num_spfs, num_states, sparse,
              seed=7, full_rank=False):
    """A tiny ML system: solver, (A, Bs), random MO integrals."""
    dci = SemiDirectCI_ML(list(num_orbitals), list(num_particles), sparse=sparse)
    rng = np.random.default_rng(seed)

    if full_rank:
        num_spfs = tuple(dci.num_dets)
        Bs = [jnp.asarray(np.eye(D, dtype=np.complex128)) for D in dci.num_dets]
    else:
        Bs = [
            jnp.asarray(_orthonormal(rng, dci.num_dets[s], num_spfs[s]))
            for s in range(dci.num_species)
        ]

    a_shape = (num_states,) + tuple(num_spfs)
    A = rng.standard_normal(a_shape) + 1j * rng.standard_normal(a_shape)
    A = A / np.linalg.norm(A.reshape(num_states, -1), axis=1)[
        (slice(None),) + (None,) * len(num_spfs)
    ]
    A = jnp.asarray(A)

    h_list = []
    for s in range(dci.num_species):
        m = num_orbitals[s]
        a = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
        h_list.append(jnp.asarray(a + a.conj().T))

    g_dict = {}
    for s in range(dci.num_species):
        if num_particles[s] > 1:
            m = num_orbitals[s]
            shape = (m, m, m, m)
            g = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
            g_dict[(s, s)] = jnp.asarray(g)
    for s1 in range(dci.num_species):
        for s2 in range(s1 + 1, dci.num_species):
            m1, m2 = num_orbitals[s1], num_orbitals[s2]
            g = (rng.standard_normal((m1, m2, m1, m2))
                 + 1j * rng.standard_normal((m1, m2, m1, m2)))
            g = jnp.asarray(g)
            # both orderings, exactly as _transform_all_g supplies them
            g_dict[(s1, s2)] = g
            g_dict[(s2, s1)] = jnp.transpose(g, (1, 0, 3, 2))

    return dci, A, Bs, h_list, g_dict


CASES = [
    # (num_particles, num_orbitals, num_spfs, num_states)
    ([1, 2], [3, 4], (2, 3), 1),
    ([1, 2], [3, 4], (3, 4), 2),
    ([1, 2, 1], [2, 3, 2], (2, 2, 1), 2),
    ([2, 2], [4, 4], (3, 2), 3),
]
CASE_IDS = ["K2n23s1", "K2n34s2", "K3n221s2", "K2N2n32s3"]


def _params():
    return [
        pytest.param(c, sp, id=f"{i}-{'sparse' if sp else 'dense'}")
        for c, i in zip(CASES, CASE_IDS)
        for sp in (True, False)
    ]


# ── ML RDMs equal the standard RDMs of the expanded C ─────────────────────


@pytest.mark.parametrize("case,sparse", _params())
def test_ml_rdms_match_expanded(case, sparse):
    num_particles, num_orbitals, num_spfs, num_states = case
    dci, A, Bs, _h, _g = make_case(num_particles, num_orbitals, num_spfs,
                                   num_states, sparse)
    C = jnp.asarray(expand_C_ref(A, Bs))

    for weights in (None, jnp.asarray(
            np.linspace(0.2, 1.0, num_states) / np.sum(
                np.linspace(0.2, 1.0, num_states)))):
        Ds_ref = dci.get_1p_RDMs(C, weights=weights)
        Ds_ml = dci.ml_1p_RDMs(A, Bs, weights=weights)
        for s in range(dci.num_species):
            np.testing.assert_allclose(
                np.asarray(Ds_ml[s]), np.asarray(Ds_ref[s]),
                rtol=0, atol=TOL, err_msg=f"D[{s}]")

        ds_ref = dci.get_2p_RDMs(C, weights=weights)
        ds_ml = dci.ml_2p_RDMs(A, Bs, weights=weights)
        assert set(ds_ml) == set(ds_ref)
        for key in ds_ref:
            np.testing.assert_allclose(
                np.asarray(ds_ml[key]), np.asarray(ds_ref[key]),
                rtol=0, atol=TOL, err_msg=f"d{key}")


@pytest.mark.parametrize("case,sparse", _params())
def test_trace_of_1rdm_is_particle_number(case, sparse):
    num_particles, num_orbitals, num_spfs, num_states = case
    dci, A, Bs, _h, _g = make_case(num_particles, num_orbitals, num_spfs,
                                   num_states, sparse)
    Ds = dci.ml_1p_RDMs(A, Bs)
    for s in range(dci.num_species):
        tr = np.trace(np.asarray(Ds[s]), axis1=-2, axis2=-1)
        np.testing.assert_allclose(tr, num_particles[s] * np.ones(num_states),
                                   rtol=0, atol=1e-11)


# ── B = 1 limits ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("case,sparse", _params())
def test_spf_transition_ops_at_identity_is_Epq(case, sparse):
    num_particles, num_orbitals, num_spfs, num_states = case
    dci, A, Bs, _h, _g = make_case(num_particles, num_orbitals, num_spfs,
                                   num_states, sparse, full_rank=True)
    for s in range(dci.num_species):
        e = np.asarray(dci.spf_transition_ops(s, Bs[s]))
        E = np.asarray(dci.E_pq[s], dtype=np.float64)   # (I, J, p, q)
        np.testing.assert_allclose(e, np.einsum("ijpq->pqij", E),
                                   rtol=0, atol=TOL)


@pytest.mark.parametrize("case,sparse", _params())
def test_ml_top_ops_at_identity_equal_precompute_ci_ops(case, sparse):
    num_particles, num_orbitals, num_spfs, num_states = case
    dci, A, Bs, h_list, g_dict = make_case(num_particles, num_orbitals,
                                           num_spfs, num_states, sparse,
                                           full_rank=True)
    ref_diag, ref_pair = dci.precompute_ci_ops(h_list, g_dict)
    ml_diag, ml_pair = dci.ml_top_ops(h_list, g_dict, Bs)

    for s in range(dci.num_species):
        np.testing.assert_allclose(np.asarray(ml_diag[s]),
                                   np.asarray(ref_diag[s]),
                                   rtol=0, atol=TOL, err_msg=f"ops_diag[{s}]")
    assert set(ml_pair) == set(ref_pair)
    for key in ref_pair:
        np.testing.assert_allclose(np.asarray(ml_pair[key]),
                                   np.asarray(ref_pair[key]),
                                   rtol=0, atol=TOL, err_msg=f"ops_pair{key}")


@pytest.mark.parametrize("case,sparse", _params())
def test_meanfield_W_at_identity_matches_partial_2rdm(case, sparse):
    """At B = 1, ``W[m,m',p,r] = Σ_{qt} g4[p,q,r,t] d^{(s,k)}[m,q,m',t]``
    with the species-s determinant indices of the pair density left free."""
    num_particles, num_orbitals, num_spfs, num_states = case
    dci, A, Bs, _h, g_dict = make_case(num_particles, num_orbitals, num_spfs,
                                       num_states, sparse, full_rank=True)
    w = jnp.asarray(np.full(num_states, 1.0 / num_states))

    for s in range(dci.num_species):
        W_dict = dci.ml_meanfield_W(A, Bs, g_dict, s, weights=w)
        for k, W in W_dict.items():
            rho_pair = np.asarray(dci._compute_rho_pair(A, s, k, w))
            E_k = np.asarray(dci.E_pq[k], dtype=np.float64)
            # d_partial[m, m', q, t] = Σ_{zw} ρ[m, z, m', w] E^k[z, w, q, t]
            d_partial = np.einsum("mzow, zwqt -> moqt", rho_pair, E_k)
            W_ref = np.einsum("pqrt, moqt -> mopr",
                              np.asarray(g_dict[(s, k)]), d_partial)
            np.testing.assert_allclose(np.asarray(W), W_ref,
                                       rtol=0, atol=TOL,
                                       err_msg=f"W[{s},{k}]")


# ── energy consistency at truncated rank ──────────────────────────────────


@pytest.mark.parametrize("case,sparse", _params())
def test_top_energy_matches_expanded_ci_energy(case, sparse):
    num_particles, num_orbitals, num_spfs, num_states = case
    dci, A, Bs, h_list, g_dict = make_case(num_particles, num_orbitals,
                                           num_spfs, num_states, sparse)
    C = jnp.asarray(expand_C_ref(A, Bs))

    top_ops = dci.ml_top_ops(h_list, g_dict, Bs)
    sigma_top = dci.ml_get_sigma(A, top_ops)
    det_axes = tuple(range(1, A.ndim))
    E_ml = np.asarray(jnp.sum(A.conj() * sigma_top, axis=det_axes))

    sigma_ci = dci.get_sigma(C, h_list, g_dict)
    E_ref = np.asarray(jnp.sum(C.conj() * sigma_ci,
                               axis=tuple(range(1, C.ndim))))

    np.testing.assert_allclose(E_ml, E_ref, rtol=0, atol=1e-11)


@pytest.mark.parametrize("case,sparse", _params())
def test_top_sigma_is_the_projected_ci_sigma(case, sparse):
    """``H_top A`` must equal ``(Π_s B_s†) H C`` — the CI sigma of the
    expanded state, projected back onto the SPF basis."""
    num_particles, num_orbitals, num_spfs, num_states = case
    dci, A, Bs, h_list, g_dict = make_case(num_particles, num_orbitals,
                                           num_spfs, num_states, sparse)
    C = jnp.asarray(expand_C_ref(A, Bs))

    sigma_top = np.asarray(dci.ml_get_sigma(A, dci.ml_top_ops(h_list, g_dict, Bs)))

    sig = np.asarray(dci.get_sigma(C, h_list, g_dict))
    for s, B in enumerate(Bs):
        sig = np.tensordot(sig, np.asarray(B).conj(), axes=([s + 1], [0]))
        sig = np.moveaxis(sig, -1, s + 1)

    np.testing.assert_allclose(sigma_top, sig, rtol=0, atol=1e-11)


# ── mean field against an independent dense construction ──────────────────


def _meanfield_reference(dci, A, Bs, g_dict, s, k, weights):
    """``W^{sk}`` from the expanded single-hole functions, in dense numpy.

    Builds ``Φ[α, m, {I_l}_{l≠s}]`` — the top tensor with species s left in
    SPF space and every OTHER species expanded into its determinant basis —
    then contracts ``E^k`` between two copies of it, tracing every remaining
    species index. This reaches ``W`` by a different route than
    :meth:`ml_meanfield_W`, which never leaves SPF space.
    """
    K = dci.num_species
    Phi = np.asarray(A)
    for l in range(K):
        if l == s:
            continue
        Phi = np.tensordot(Phi, np.asarray(Bs[l]), axes=([l + 1], [1]))
        Phi = np.moveaxis(Phi, -1, l + 1)

    # letters: 'S' state, per-species axis letters for bra and ket
    bra = [chr(ord("a") + i) for i in range(K)]
    ket = list(bra)
    bra[s], ket[s] = "m", "o"          # free SPF indices on species s
    ket[k] = "W"                       # E^k connects bra[k] ── ket[k]
    E_k = np.asarray(dci.E_pq[k], dtype=np.float64)

    sub = (f"S,S{''.join(bra)},{bra[k]}{ket[k]}qt,S{''.join(ket)}->moqt")
    d_partial = np.einsum(sub, np.asarray(weights), Phi.conj(), E_k, Phi)

    return np.einsum("pqrt, moqt -> mopr",
                     np.asarray(g_dict[(s, k)]), d_partial)


@pytest.mark.parametrize("case,sparse", _params())
def test_meanfield_and_apply_against_dense_reference(case, sparse):
    num_particles, num_orbitals, num_spfs, num_states = case
    dci, A, Bs, _h, g_dict = make_case(num_particles, num_orbitals, num_spfs,
                                       num_states, sparse)
    w = np.linspace(0.3, 1.0, num_states)
    w = w / w.sum()
    wj = jnp.asarray(w)

    for s in range(dci.num_species):
        W_dict = dci.ml_meanfield_W(A, Bs, g_dict, s, weights=wj)
        partners = [k for k in range(dci.num_species)
                    if k != s and (s, k) in g_dict]
        assert sorted(W_dict) == sorted(partners)

        U_ref = np.zeros((dci.num_dets[s], num_spfs[s]), dtype=np.complex128)
        E_s = np.asarray(dci.E_pq[s], dtype=np.float64)
        for k in partners:
            W_ref = _meanfield_reference(dci, A, Bs, g_dict, s, k, w)
            np.testing.assert_allclose(np.asarray(W_dict[k]), W_ref,
                                       rtol=0, atol=1e-11,
                                       err_msg=f"W[{s},{k}]")
            U_ref += np.einsum("IJpr, mopr, Jo -> Im", E_s, W_ref,
                               np.asarray(Bs[s]))

        U = np.asarray(dci.apply_meanfield_B(s, Bs[s], W_dict))
        np.testing.assert_allclose(U, U_ref, rtol=0, atol=1e-11,
                                   err_msg=f"U[{s}]")


# ── sparse and dense branches agree ───────────────────────────────────────


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_sparse_and_dense_branches_agree(case):
    num_particles, num_orbitals, num_spfs, num_states = case
    outs = []
    for sparse in (True, False):
        dci, A, Bs, h_list, g_dict = make_case(num_particles, num_orbitals,
                                               num_spfs, num_states, sparse)
        w = jnp.asarray(np.full(num_states, 1.0 / num_states))
        es = [dci.spf_transition_ops(s, Bs[s]) for s in range(dci.num_species)]
        diag, pair = dci.ml_top_ops(h_list, g_dict, Bs, es=es)
        out = {f"e{s}": np.asarray(es[s]) for s in range(dci.num_species)}
        out.update({f"diag{s}": np.asarray(diag[s])
                    for s in range(dci.num_species)})
        out.update({f"pair{key}": np.asarray(v) for key, v in pair.items()})
        for s in range(dci.num_species):
            W_dict = dci.ml_meanfield_W(A, Bs, g_dict, s, es=es, weights=w)
            out[f"U{s}"] = np.asarray(dci.apply_meanfield_B(s, Bs[s], W_dict))
        for s, D in enumerate(dci.ml_1p_RDMs(A, Bs, weights=w)):
            out[f"D{s}"] = np.asarray(D)
        for key, d in dci.ml_2p_RDMs(A, Bs, weights=w, es=es).items():
            out[f"d{key}"] = np.asarray(d)
        outs.append(out)

    assert set(outs[0]) == set(outs[1])
    for key in sorted(outs[0]):
        np.testing.assert_allclose(outs[0][key], outs[1][key],
                                   rtol=0, atol=1e-11, err_msg=key)
