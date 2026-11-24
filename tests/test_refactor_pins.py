"""Behaviour pins for the pre-ML-MCTDHF refactors.

``tests/_make_reference.py`` recorded these outputs on the tree as it stood
before the ``SemiDirectCI_n_species`` and ``integrators``
kernel extractions. Those refactors are required to be behaviour-preserving,
so the pin is as strict as the backend allows.

**On CPU the pin is exact on the machine type that recorded it.** With
``MCTDHF_PIN_EXACT=1`` every quantity is compared with ``assert_array_equal``
in x64: any reordering of floating-point arithmetic shows up immediately, and
it does hold, also when the references are recorded from the pre-refactor
tree (v0.7.3) on a second machine and v0.8.0 is pinned against those.

**Across CPU types it is not exact**, because XLA's CPU code generation
follows the host's instruction set. Measured with the shipped CPU recordings
on a second machine (36-core AVX-512 host, JAX 0.9.1): at most 2 eps on every
trajectory, except the two ITE/``diagonalize`` cases, which differ by
1.2e4 eps on the orbitals and 1.8e3 eps on the CI block after removing the
eigenvector phases (the density-inverse amplification and the phase freedom
described below). By default the CPU is therefore compared like the GPU, with
the bound of those two cases widened to ``_TOL_ULP_XCPU``.

**On GPU exact equality is not attainable**, for two measured reasons, which
are handled separately rather than with one blunt tolerance:

1. *Last-bit noise.* Only one case, ``midpoint_ite_sil_diag``, is genuinely
   irreducible: two fresh processes running identical code differ by
   2.7e-18 relative per element, so no source change can make it
   bit-identical.

   Everything else IS bit-identical on GPU, but only in a clean process.
   XLA's compiled-code state, autotuning and allocator arena are
   process-global, so running the rest of the suite first moves these same
   computations: the ITE/``diagonalize`` cases go from exactly zero
   deviation to 2.5e-12 on the orbitals (~1e4 eps, amplified by the
   regularised density inverse). ``tests/conftest.py`` therefore runs this
   module FIRST. Do not remove that ordering to make a pin pass elsewhere —
   it is what keeps the tolerance below meaningful, rather than being raised
   by four orders to accommodate a dirty process.

   The GPU tolerances are written as multiples of the dtype's own epsilon
   rather than as bare decimals, so they mean the same thing if the suite is
   ever run in another precision. Measured on the RTX 2070 SUPER in
   complex128 (eps = 2.2e-16):

   ==========================================  ===========  ==========
   quantity                                    relative     in eps
   ==========================================  ===========  ==========
   fresh process, ITE/diagonalize              0            0
   ``midpoint_ite_sil_diag`` run to run        2.7e-18      0.01
   same module after the full suite (avoided)  2.5e-12      11000
   ==========================================  ===========  ==========

2. *An exact gauge freedom, not noise.* In imaginary time the
   ``diagonalize`` CI propagator replaces C by eigenvectors of H_CI, and an
   eigenvector's phase is fixed only by the eigensolver's internal
   normalisation. That choice moves discontinuously under input changes far
   below the eigenvalue gap, so on GPU these cases differ by a per-state
   phase of order 1e-10 rad — a 4e-12 elementwise difference in a state that
   is otherwise identical. Those cases are compared phase-invariantly: the
   orbitals elementwise, and the CI block after removing one phase per
   state. On CPU they are still compared exactly.

**What the GPU pin can and cannot catch.** It is the coarser instrument.
A genuine regression found during this work — a nested ``@jit`` that made
XLA fuse differently — also showed up at only ~1 eps, i.e. INSIDE these
tolerances. The CPU exact-equality pin is what catches that class; the GPU
pin catches gross changes and keeps the backend honest. Size at the
precision floor tells you how big a discrepancy is, not where it came from,
so distinguishing the two takes an experiment (run the recorder twice in
separate processes; run this module alone versus inside the suite), not a
threshold.

Do not regenerate a reference to make a failing pin pass, and do not loosen
a tolerance without a measurement. Backends do not share floating-point
associativity, so each has its own recording — see
:func:`tests._make_reference.reference_dir`.
"""

import os

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from tests import _make_reference as ref


REFERENCE_DIR = ref.reference_dir()

# Trajectory tolerances, in multiples of the working dtype's epsilon and
# relative to the trajectory's own scale. None means exact equality.
# See the module docstring for the measurements these come from.
#
#   _TOL_ULP       covers the 3.7 eps of process-history noise, 17x over.
#   _TOL_ULP_PHASE covers the 90 eps left after removing the eigenvector
#                  phase in the ITE/diagonalize cases, 6x over.
#   _TOL_ULP_XCPU  covers the 1.2e4 eps of the ITE/diagonalize cases between
#                  two CPU types (orbitals and phase-free CI), 5.6x over.
_EXACT_CPU = os.environ.get("MCTDHF_PIN_EXACT", "0") == "1"
_BACKEND_TOL = {"cpu": None if _EXACT_CPU else "ulp", "gpu": "ulp"}
_TOL_ULP = 64
_TOL_ULP_PHASE = 512
_TOL_ULP_XCPU = 65536


def _atol(expected, ulp):
    """`ulp` epsilons of the array's own dtype, times its scale."""
    eps = float(np.finfo(np.asarray(expected).real.dtype).eps)
    scale = max(float(np.max(np.abs(expected))), 1.0)
    return ulp * eps * scale

pytestmark = pytest.mark.skipif(
    not os.path.isdir(REFERENCE_DIR),
    reason=(f"no pinned trajectories for backend "
            f"'{jax.default_backend()}' in {REFERENCE_DIR}; record them on "
            f"the PRE-refactor tree with `python -m tests._make_reference`"),
)


def _is_exact():
    """True when this backend is pinned by exact equality."""
    return _BACKEND_TOL.get(jax.default_backend(), "ulp") is None


def _load(name):
    path = os.path.join(REFERENCE_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"missing reference file {name}")
    return np.load(path)


def _assert_close(got, expected, ulp, err_msg=""):
    """Exact when this backend is exact, else within `ulp` epsilons."""
    if _is_exact():
        np.testing.assert_array_equal(got, expected, err_msg=err_msg)
    else:
        np.testing.assert_allclose(got, expected, rtol=0,
                                   atol=_atol(expected, ulp),
                                   err_msg=err_msg)


def _assert_trajectory_phase_invariant(case, got, expected):
    """Compare an ITE/``diagonalize`` trajectory up to the CI phase gauge.

    Each snapshot is unpacked; the orbitals are compared elementwise and the
    CI block after removing the one arbitrary phase per state that the
    eigensolver leaves free. The overlap matrix is checked to be the identity
    first, so a genuine change of state — mixing, or lost norm — still fails.
    """
    imag_time, e_shift = case[2], case[3]
    solver = ref.build_solver(imag_time, e_shift=e_shift)
    num_states = solver.num_states
    xcpu = jax.default_backend() == "cpu"
    tol_b = _TOL_ULP_XCPU if xcpu else _TOL_ULP
    tol_c = _TOL_ULP_XCPU if xcpu else _TOL_ULP_PHASE

    for step in range(got.shape[0]):
        C_got, bs_got = solver._y_to_Cb(got[step])
        C_exp, bs_exp = solver._y_to_Cb(expected[step])

        for s in range(solver.num_species):
            _assert_close(np.asarray(bs_got[s]), np.asarray(bs_exp[s]),
                          tol_b, err_msg=f"step {step} orbitals[{s}]")

        g = np.asarray(C_got).reshape(num_states, -1)
        e = np.asarray(C_exp).reshape(num_states, -1)
        norms = np.linalg.norm(e, axis=1)
        norms = np.where(norms > 0, norms, 1.0)
        S = (g.conj() @ e.T) / np.outer(norms, norms)
        np.testing.assert_allclose(
            np.abs(S), np.eye(num_states), rtol=0, atol=1e-9,
            err_msg=f"step {step}: CI overlap matrix is not the identity")
        g = g * np.exp(1j * np.angle(np.diag(S)))[:, None]
        _assert_close(g, e, tol_c, err_msg=f"step {step} CI block")


@pytest.mark.parametrize("case", ref.reference_cases(), ids=lambda c: c[0])
def test_integrator_trajectory_pinned(case):
    expected = _load(f"traj_{case[0]}.npz")["ys"]
    got = ref.run_case(case)
    assert got.dtype == expected.dtype

    imag_time, c_propagator = case[2], case[4]
    if not _is_exact() and imag_time and c_propagator == "diagonalize":
        _assert_trajectory_phase_invariant(case, got, expected)
    else:
        _assert_close(got, expected, _TOL_ULP)


def test_adaptive_half_step_pinned():
    expected = _load("adaptive.npz")
    got = ref.run_adaptive()
    assert set(got) == set(expected.files)
    for key in sorted(got):
        _assert_close(got[key], expected[key], _TOL_ULP, err_msg=key)


@pytest.mark.parametrize("ci_sparse", [True, False], ids=["sparse", "dense"])
def test_ci_ops_and_rdms_bit_identical(ci_sparse):
    """The algebraic kernels are pinned EXACTLY on every backend.

    Unlike a propagated trajectory these are a single evaluation with no
    accumulation, and they reproduce bit-for-bit on GPU too — so there is no
    reason to weaken them.
    """
    tag = "sparse" if ci_sparse else "dense"
    expected = _load(f"ci_ops_{tag}.npz")
    got = ref.run_ci_ops(ci_sparse)
    assert set(got) == set(expected.files)
    for key in sorted(got):
        np.testing.assert_array_equal(got[key], expected[key], err_msg=key)
