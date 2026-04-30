# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

N-Species MCTDHF: a JAX-based quantum dynamics framework for multi-particle systems with multiple distinguishable species. Uses semi-direct CI (precomputed excitation operators, on-the-fly HC products) rather than full Hamiltonian diagonalization.

## Commands

```bash
# Install (editable, for development)
pip install -e .

# Install with optional dependencies
pip install -e ".[adaptive]"     # diffrax adaptive ODE solvers
pip install -e ".[cuda12]"       # CUDA support

# Run all tests
pytest tests/

# Run a single test
pytest tests/test_cmf.py -v
pytest tests/test_cmf.py::test_cmf_small_dt_agreement -v
```

## Architecture

### State representation

The quantum state is a packed 1D vector: `y = [C_flat, b_0_flat, ..., b_{K-1}_flat]` where:
- **C** = CI coefficient tensor, shape `(num_states, n_dets_0, ..., n_dets_{K-1})`
- **b_s** = orbital coefficient matrix per species, shape `(num_spatial_s, num_mctdhf_s)`

Pack/unpack via `_y_to_Cb(y)` / `_Cb_to_y(C, bs)` on the solver.

### Key modules

- **`MCTDHF_n_species.py`** — Core solver class. `__call__(t, y)` computes the RHS of the equations of motion (CI + orbital). Subclass `MCTDHF_n_species_DVR` handles diagonal DVR basis integrals. Key internal methods: `transform_h`, `transform_g` (integral transforms to MO basis), `_invert_density_matrix` (Tikhonov-regularized D⁻¹).

- **`integrators.py`** — All time-propagation schemes. Main integrators:
  - `get_rk4_solver` — explicit RK4, small dt
  - `get_cmf_integrator` / `get_cmf_half_step_integrator` — Constant Mean Field (decouples CI and orbital propagation), supports large dt especially in imaginary time
  - `get_adaptive_cmf_half_step_integrator` — variable time stepping for real-time
  - `get_psi_integrator` — Projector-Splitting Integrator (1 particle/species only), exact norm conservation, avoids D⁻¹
  - SIL (Shift-Invert Lanczos) sub-solvers: `_sil_solve_C`, `_sil_solve_bs`

- **`SemiDirectCI_n_species.py`** — Precomputes Slater determinants and excitation tensors (E_pq, E_pqrs) per species. `get_sigma(C, h_MO, g_MO)` computes H|Ψ⟩. Provides 1- and 2-particle RDMs.

- **`sine_dvr.py`** — Sine-DVR (particle-in-a-box) basis. `SineDVR` class provides grid, kinetic energy, and methods `get_h(potential)` / `get_g(interaction)` for constructing spatial integrals.

### Coupling between modules

The solver (`MCTDHF_n_species`) and integrators are tightly coupled — integrators call solver methods like `_rhs_C_linear`, `_rhs_b_linear`, and integral transforms directly. Changes to the solver's internal interface require checking all integrators.

### Computational notes

- All heavy computation uses `jax.numpy` + `opt_einsum`. Einsum contraction strings are load-bearing — modify carefully.
- JAX JIT compilation is used throughout. Functions passed to integrators must be JIT-compatible (no Python side effects, dynamic shapes, etc.).
- Density matrix inversion uses Tikhonov regularization (`eps` parameter, default 1e-6) via eigendecomposition. PSI integrator avoids this entirely.
- Both real-time (`imag_time=False`) and imaginary-time evolution (`imag_time=True`, for ground-state search) are supported.
