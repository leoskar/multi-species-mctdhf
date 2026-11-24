# multi-species-mctdhf

Multi-species multiconfiguration time-dependent Hartree–Fock (MCTDHF) and its
multilayer form (ML-MCTDHF), written in [JAX](https://github.com/jax-ml/jax).

The wavefunction is a sum of products of Slater determinants, one per
**distinguishable species**. Each species has its own time-dependent orbitals and
its own primitive (sine-DVR) basis, and antisymmetrisation acts only within a
species. The code was written to simulate gates in multi-qubit hardware, where each
qubit is a separate species, but nothing in it is specific to that application.

Features:

- **Semi-direct CI.** One- and two-particle excitation operators are precomputed
  per species, and the action of the Hamiltonian on the coefficient tensor is
  evaluated directly instead of forming the Hamiltonian matrix.
- **State averaging.** Several states share one set of orbitals and are
  propagated together.
- **ML-MCTDHF.** A middle layer of many-body species basis states (SBS, called
  SPFs in the code) between the top tensor and the determinants of each species.
  This shrinks the top tensor from `prod_s C(M_s, N_s)` to `prod_s n_s` entries.
  Standard MCTDHF is recovered exactly at `n_s = C(M_s, N_s)`.
- **Integrators.** Constant-mean-field (CMF) schemes (basic, half-step and
  midpoint), RK4/VMF, a projector-splitting integrator, and adaptive variants.
  The sub-propagators are exact matrix exponentials, Lanczos, shift-invert
  Lanczos and Crank–Nicolson, the last optionally in mixed precision.
- **Imaginary-time relaxation** for an outer optimisation loop over the one-body
  Hamiltonian (`MCTDHF_n_species_ITE_opt`, `MLMCTDHF_n_species_ITE_opt`).

## Installation

Python ≥ 3.10.

```bash
pip install "multi-species-mctdhf @ git+https://github.com/leoskar/multi-species-mctdhf"
```

Optional extras: `[cuda12]` for CUDA, `[adaptive]` for the diffrax-based adaptive
solvers, `[test]` for pytest. For development, clone the repository and run
`pip install -e ".[test]"`.

## Package overview

| module | contents |
|---|---|
| `sine_dvr.py` | Sine-DVR basis: grid, kinetic energy, one- and two-body integrals (`SineDVR.get_h`, `get_g`, `get_g_cross_species`) |
| `SemiDirectCI_n_species.py` | Determinant tables, excitation operators, σ-vectors `H|Ψ⟩`, one- and two-particle RDMs |
| `MCTDHF_n_species.py` | The solver: right-hand side of the coupled coefficient and orbital equations of motion; `MCTDHF_n_species_DVR` for diagonal DVR integrals |
| `integrators.py` | Time propagation (CMF, RK4/VMF, projector splitting, adaptive) and the generic single-block kernels they share |
| `MCTDHF_n_species_ITE_opt.py` | State-averaged imaginary-time relaxation with the one-body Hamiltonian as a traced argument |
| `MLMCTDHF_n_species.py` | ML-MCTDHF solver (a mixin in front of the standard solver), layer converters, gauge transforms |
| `SemiDirectCI_ml.py` | ML layer operators and densities |
| `integrators_ml.py` | Basic and midpoint CMF integrators for the three ML layers |
| `MLMCTDHF_n_species_ITE_opt.py` | ML counterpart of `MCTDHF_n_species_ITE_opt` |
| `postprocessing.py` | State analysis: particle densities, subsystem character, localisation, rotation to natural orbitals |
| `integrators_batched.py` | Species-batched variant of the half-step CMF integrator |
| `MCTDHF_direct_jax.py`, `SemiDirectCI_jax.py`, `electron_integrals.py` | Earlier two-spin (α/β) implementation and its helper integrals |

A walk-through, from the DVR basis to an imaginary-time relaxation and the
multilayer variant, is in [`docs/guide.md`](docs/guide.md).

## Tests

```bash
pytest tests
```

The suite takes about ten minutes on CPU. `tests/test_refactor_pins.py` replays
trajectories recorded with version 0.7.3 (in `tests/reference/` for CPU and
`tests/reference_gpu/` for GPU) and checks that the single-layer solver and
integrators still reproduce them. Floating-point results differ slightly between
CPU types, so by default the comparison allows a few units in the last place
(more for two imaginary-time cases; the bounds and their measurements are in the
module docstring). On the machine type that recorded the references,
`MCTDHF_PIN_EXACT=1` enforces bit-exact equality.
`tests/test_ml_exactness.py` checks that ML-MCTDHF reduces to standard MCTDHF at
full rank. `tests/bench_ml_four_well.py` is a benchmark and is not collected by
pytest (`python -m tests.bench_ml_four_well --quick`).

## How the code was written

The package grew from a spin-unrestricted MCTDHF code written by the author and tested
against the results of J. Zanghellini, M. Kitzler, T. Brabec and A. Scrinzi,
J. Phys. B 37, 763 (2004). The extensions to several species and to the
multilayer form were written with an AI coding assistant (Claude, Anthropic)
under the author's direction and are tested against that code:
`tests/test_n_species_regression.py` compares a two-species calculation with the
spin-unrestricted code, and `tests/test_ml_exactness.py` checks that the multilayer
form reproduces standard MCTDHF at full SBS count.

## Use in publications

This code was used for

> O. Leinonen, *Multi-species MCTDHF for gate simulations in multi-qubit hardware* (2026).

The simulation scripts and data of that paper are in a separate repository,
[mctdhf-gate-simulations](https://github.com/leoskar/mctdhf-gate-simulations). The three-electron
(ML-MCTDHF) calculations used this version, 0.8.0. The one-electron calculations
used the single-layer code at version 0.7.3. The single-layer code was refactored
for 0.8.0 without changing its results: with references recorded from 0.7.3 on the
same machine, `tests/test_refactor_pins.py` passes bit-exactly.

## License

MIT, see [`LICENSE`](LICENSE).
