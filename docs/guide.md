# Multi-species MCTDHF: Setup and ITE Guide

This guide covers how to set up a two-species MCTDHF calculation (e.g., spin-up and spin-down electrons) with spin-dependent Hamiltonians, using a Sine-DVR spatial basis and imaginary time evolution via the CMF integrator.


## Imports

```python
import numpy as np
import jax.numpy as jnp
from mctdhf.sine_dvr import SineDVR
from mctdhf.MCTDHF_n_species import MCTDHF_n_species_DVR
from mctdhf.integrators import get_cmf_half_step_integrator
```

Use `MCTDHF_n_species_DVR` (not the base `MCTDHF_n_species`) when working with a DVR basis. The DVR subclass exploits the fact that 2-body integrals are 2D matrices instead of 4D tensors.

## Step 1: Set Up the Sine-DVR Basis

Create a `SineDVR` instance per species. If both species share the same spatial grid (typical for spin-up/spin-down electrons), one DVR object suffices.

```python
dvr = SineDVR(x_min=-10, x_max=10, num_points=64, mass=1.0)
```

Parameters:
- `x_min`, `x_max`: domain boundaries (hard-wall boundary conditions: wavefunction is zero at the walls)
- `num_points`: number of DVR grid points (determines spatial resolution and basis size)
- `mass`: particle mass (default 1.0, i.e., atomic units)

The grid points are at `dvr.grid` (interior points of the box). Grid spacing is `dvr.dx`.

## Step 2: Build the 1-Body Hamiltonians

`dvr.get_h(potential)` returns the single-particle Hamiltonian matrix (kinetic + potential) of shape `(num_points, num_points)`.

```python
# Example: spin-dependent potentials
def V_up(x):
    return 0.5 * x**2 + 0.1 * x   # harmonic + Zeeman-like term

def V_down(x):
    return 0.5 * x**2 - 0.1 * x   # opposite sign for spin-down

h_up = dvr.get_h(potential=V_up)
h_down = dvr.get_h(potential=V_down)

h_list = [h_up, h_down]  # one per species, order defines species index
```

The potential function receives the grid array `dvr.grid` and must return an array of the same shape. The kinetic energy is computed analytically from the FBR-DVR transformation.

For time-dependent potentials, use the `time_dependent_potential` parameter instead:

```python
def V_drive(x, t):
    return 0.5 * x**2 + 0.1 * np.sin(t) * x

h_td = dvr.get_h(time_dependent_potential=V_drive)  # returns a callable h(t)
```

### Convenience potentials

The `sine_dvr` module provides factory functions:

```python
from mctdhf.sine_dvr import harmonic_potential, softened_coulomb, double_well, contact_interaction

V = harmonic_potential(omega=1.0, center=0.0)      # 0.5 * omega^2 * (x - center)^2
V = double_well(a=1.0, b=1.0, center=0.0)          # a*(x-c)^4 - b*(x-c)^2
```

## Step 3: Build the 2-Body Interaction Integrals

`dvr.get_g(interaction)` returns a 2D matrix of shape `(num_points, num_points)` for same-species interactions. The `interaction` callable takes either `(x1, x2)` or is tried as `V(|x1-x2|)`.

```python
from mctdhf.sine_dvr import softened_coulomb

coulomb = softened_coulomb(kappa=1.0, a=0.1)  # kappa / sqrt((x1-x2)^2 + a^2)
```

### Same-species interactions

For interactions within species 0 (spin-up with spin-up) and within species 1 (spin-down with spin-down):

```python
g_uu = dvr.get_g(interaction=coulomb)  # spin-up with spin-up
g_dd = dvr.get_g(interaction=coulomb)  # spin-down with spin-down (same interaction here)
```

### Cross-species interactions

For interactions between species 0 and species 1 (spin-up with spin-down). When both species use the same DVR grid, use `get_g` directly:

```python
g_ud = dvr.get_g(interaction=coulomb)  # spin-up with spin-down
```

If the species use different DVR grids, use `get_g_cross_species`:

```python
dvr_up = SineDVR(x_min=-10, x_max=10, num_points=64)
dvr_down = SineDVR(x_min=-8, x_max=8, num_points=48)
g_ud = dvr_up.get_g_cross_species(dvr_down, interaction=coulomb)
# shape: (dvr_up.num_points, dvr_down.num_points)
```

### Spin-dependent interactions

To have different interaction strengths per species pair:

```python
g_uu = dvr.get_g(interaction=softened_coulomb(kappa=1.0, a=0.1))
g_dd = dvr.get_g(interaction=softened_coulomb(kappa=1.0, a=0.1))
g_ud = dvr.get_g(interaction=softened_coulomb(kappa=0.5, a=0.1))  # weaker cross-spin
```

### Assembling g_dict

The `g_dict` maps species-pair tuples to interaction matrices. Only one ordering per pair is needed; the code computes the transpose automatically.

```python
g_dict = {
    (0, 0): g_uu,   # up-up interaction
    (1, 1): g_dd,   # down-down interaction
    (0, 1): g_ud,   # up-down interaction (code derives (1,0) by transposing)
}
```

If a pair is absent from `g_dict`, those species do not interact. For example, to have only cross-spin Coulomb repulsion (no same-spin interaction), use:

```python
g_dict = {(0, 1): g_ud}
```

## Step 4: Initialize the MCTDHF Solver

```python
N_spatial = dvr.num_points  # e.g., 64

mctdh = MCTDHF_n_species_DVR(
    h_list=h_list,                          # [h_up, h_down]
    g_dict=g_dict,                          # {(0,0): g_uu, (1,1): g_dd, (0,1): g_ud}
    num_particles=[N_up, N_down],           # e.g., [1, 1] for 2 electrons
    num_mctdhf_orbitals=[M_up, M_down],     # active orbitals per species, e.g., [4, 4]
    num_spatial_orbitals=[N_spatial, N_spatial],
    num_states=1,                           # number of states to compute simultaneously
    imag_time=True,                         # ITE for ground state search
    eps=1e-6,                               # Tikhonov regularization for density matrix inverse
)
```

Parameters:
- `num_particles`: list with number of particles per species. Species index 0 = spin-up, 1 = spin-down (by convention from `h_list` ordering).
- `num_mctdhf_orbitals`: number of active (MCTDHF) orbitals per species. More orbitals = more accurate but exponentially more determinants. The number of Slater determinants per species is C(M, N) (M choose N).
- `num_spatial_orbitals`: basis size per species (= number of DVR grid points).
- `num_states`: set > 1 for excited states. Use `state_weights` for state-averaged calculations.
- `imag_time`: `True` for imaginary time evolution (ground state search), `False` for real-time dynamics.
- `eps`: regularization for the density matrix inversion D^{-1}. Increase if you see numerical instability; decrease for higher accuracy.

## Step 5: Run Imaginary Time Evolution

### Option A: Using `integrate()` (fixed time step)

```python
dt = 0.01
num_steps = 1000

integrator = get_cmf_half_step_integrator(
    mctdh,
    dt,
    c_propagator="sil",       # SIL for CI coefficients (recommended)
    b_propagator="sil",       # SIL for orbitals (recommended for DVR)
    sil_krylov_dim=10,        # Krylov dimension for CI (increase if convergence issues)
    sil_b_krylov_dim=20,      # Krylov dimension for orbitals (larger because stiffer)
)

ts, ys = mctdh.integrate(
    integrator=integrator,
    num_steps=num_steps,
    normalize=True,       # orthonormalize C and orbitals each step (essential for ITE)
    noise_C=1e-3,         # break symmetry in initial CI coefficients
    noise_bs=1e-3,        # break symmetry in initial orbitals
    save_every=100,       # save every 100 steps to reduce memory
)
```

`integrate()` handles initialization automatically when `C_init` and `bs_init_list` are not provided:
- Orbitals `b_s` are initialized to the lowest eigenstates of each single-particle Hamiltonian `h_list[s]`.
- CI coefficients `C` are initialized to energy-ordered product states (lowest non-interacting energy first).

The `noise_C` and `noise_bs` parameters add small random perturbations to break any symmetry in the initial guess, which helps ITE converge to the true ground state rather than getting stuck in a symmetric saddle point.

`normalize=True` maps to `'orthonormalize'`, which applies QR orthonormalization to the orbital columns and CI state vectors after each step. This is essential during ITE because `e^{-Ht}` is not unitary and the state norm decays exponentially.

### Option B: Using `integrate_adaptive()` (adaptive time step)

```python
ts, ys, info = mctdh.integrate_adaptive(
    t_init=0.0,
    t_final=10.0,
    dt_init=0.01,
    num_save=100,              # number of uniformly-spaced save points
    tol=1e-5,                  # error tolerance per step
    sil_krylov_dim=10,
    sil_b_krylov_dim=20,
    normalize=True,
    noise_C=1e-3,
    noise_bs=1e-3,
)
print(f"Steps taken: {info['steps_taken']}, rejected: {info['steps_rejected']}")
```

The adaptive integrator uses a PI controller to adjust dt based on the CMF splitting error estimate. It always uses SIL sub-propagators. `tol` controls per-step relative accuracy; with complex64 arithmetic the noise floor is ~1e-7, so set `tol` well above that.

### Alternatively: manual loop

For more control (e.g., monitoring energy at each step):

```python
integrator = get_cmf_half_step_integrator(mctdh, dt, c_propagator="sil", b_propagator="sil")

y_init = mctdh._Cb_to_y(C_init, [b_up_init, b_down_init])
t, y = 0.0, y_init

for step in range(num_steps):
    t, y = integrator(t, y)

    # Orthonormalize (required for ITE)
    C, bs = mctdh._y_to_Cb(y)
    C = mctdh.ortonorm_C(C)
    bs = [mctdh.ortonorm_b(b) for b in bs]
    y = mctdh._Cb_to_y(C, bs)

    if step % 100 == 0:
        E = mctdh.calculate_energy(t, y)
        print(f"Step {step}: E = {float(jnp.real(E[0])):.8f}")
```

## Step 6: Extract Results

### Energy

`calculate_energy` returns a per-state array of shape `(num_states,)`:

```python
energies = mctdh.calculate_energy(t, ys[-1])
E_ground = float(jnp.real(energies[0]))
```

### Particle densities

```python
C, bs = mctdh._y_to_Cb(ys[-1])
spf_list = [dvr.get_spf(), dvr.get_spf()]  # one per species
densities = mctdh.calculate_particle_densities(C, bs, spf_list)

rho_up = np.array(densities[0][0])    # species 0, state 0
rho_down = np.array(densities[1][0])  # species 1, state 0

import matplotlib.pyplot as plt
plt.plot(dvr.grid, rho_up, label='spin-up')
plt.plot(dvr.grid, rho_down, label='spin-down')
plt.legend()
```

### Convergence check

Monitor energy vs. step. For converged ITE, the energy should plateau:

```python
energies = []
for i in range(len(ts)):
    E = float(jnp.real(mctdh.calculate_energy(float(ts[i]), ys[i])[0]))
    energies.append(E)
```

## Integrator Selection Summary

| Integrator | Function | Best for |
|---|---|---|
| `get_cmf_integrator` | Basic CMF (CN+BiCGSTAB) | ITE with large dt |
| `get_cmf_half_step_integrator` | Half-step CMF (predictor-corrector) | ITE and RTE, better splitting error |
| `integrate_adaptive` (method) | Adaptive half-step CMF with SIL | Long RTE runs, unknown dt |

For imaginary time evolution, `get_cmf_half_step_integrator` with `c_propagator="sil"` and `b_propagator="sil"` is recommended. The SIL propagator handles the stiffness of large DVR grids well and is norm-preserving by construction (important for diagnostics).

## Key Sizing Considerations

- `num_points` (DVR grid): determines spatial resolution. More points = better resolution but higher cost. The 2-body integrals scale as O(num_points^2) in DVR.
- `num_mctdhf_orbitals` (M): the number of active orbitals per species. The number of determinants per species is C(M, N_particles), and the CI tensor has size `num_states * prod(num_dets)`. This is the main driver of computational cost.
- `num_states`: propagating multiple states simultaneously is needed for excited states. Use `state_weights` for state-averaged orbital optimization.

## Complete Example: Two Spin Species with Spin-Dependent Hamiltonian

```python
import numpy as np
import jax.numpy as jnp
from mctdhf.sine_dvr import SineDVR, softened_coulomb
from mctdhf.MCTDHF_n_species import MCTDHF_n_species_DVR
from mctdhf.integrators import get_cmf_half_step_integrator

# 1. DVR grid
dvr = SineDVR(x_min=-10, x_max=10, num_points=64)
N = dvr.num_points

# 2. Spin-dependent 1-body Hamiltonians
B = 0.1  # magnetic field strength
h_up   = dvr.get_h(potential=lambda x: 0.5 * x**2 + B * x)
h_down = dvr.get_h(potential=lambda x: 0.5 * x**2 - B * x)

# 3. 2-body interactions
coulomb = softened_coulomb(kappa=1.0, a=0.1)
g_same  = dvr.get_g(interaction=coulomb)
g_cross = dvr.get_g(interaction=coulomb)

# 4. MCTDHF solver
mctdh = MCTDHF_n_species_DVR(
    h_list=[h_up, h_down],
    g_dict={(0, 0): g_same, (1, 1): g_same, (0, 1): g_cross},
    num_particles=[1, 1],
    num_mctdhf_orbitals=[4, 4],
    num_spatial_orbitals=[N, N],
    num_states=1,
    imag_time=True,
)

# 5. Imaginary time evolution
integrator = get_cmf_half_step_integrator(
    mctdh, dt=0.01, c_propagator="sil", b_propagator="sil",
)

ts, ys = mctdh.integrate(
    integrator=integrator,
    num_steps=1000,
    normalize=True,
    noise_C=1e-3,
    noise_bs=1e-3,
    save_every=100,
)

# 6. Results
E = float(jnp.real(mctdh.calculate_energy(float(ts[-1]), ys[-1])[0]))
print(f"Ground state energy: {E:.8f}")

C, bs = mctdh._y_to_Cb(ys[-1])
spf_list = [dvr.get_spf(), dvr.get_spf()]
densities = mctdh.calculate_particle_densities(C, bs, spf_list)
```

## ML-MCTDHF

Multi-layer MCTDHF inserts a third layer between the CI tensor and the Slater
determinants. Per species there are `n_s` many-body **single-particle
functions** (SPFs), each a vector in that species' determinant space, and the
top tensor runs over SPF indices rather than determinant indices:

```
A   : (num_states, n_0, ..., n_{K-1})   top tensor
B_s : (D_s, n_s)                        SPFs in the determinant basis
b_s : (N_spatial_s, M_s)                orbitals   (unchanged)
```

with `|phi^s_j> = sum_I B_s[I,j] |Phi^s_I(b_s)>` and
`|Psi_a> = sum_j A[a, j_0..j_{K-1}] prod_s |phi^s_{j_s}>`.

The packed state is `y = [A, B_0 ... B_{K-1}, b_0 ... b_{K-1}]`. There is no CI
tensor in it, so `_y_to_Cb` raises; use `expand_C(A, Bs)` or `to_mctdhf_y(y)`.

Why it matters: at four wells of N = 3 with `norb = 9` the CI tensor is
`16 * 84^4 = 8.0e8` coefficients and the Lanczos basis alone is 42 GB. With
`n_s = 6` the top tensor is `16 * 6^4 = 20736` and the inter-species pair
operator drops from `84^2 * 84^2` to `6^2 * 6^2`. The orbital step is
untouched, so it becomes the whole cost. Standard MCTDHF is recovered exactly
at `n_s = D_s`, where the SPF equation gives `dB/dt = 0` and the solve is
skipped.

### Setting up

```python
from mctdhf.MLMCTDHF_n_species import MLMCTDHF_n_species_DVR, make_regauge_ml
from mctdhf.integrators_ml import (
    get_ml_cmf_integrator, get_ml_cmf_midpoint_integrator,
)

ml = MLMCTDHF_n_species_DVR(
    h_list=h_list,
    g_dict=g_dict,
    num_particles=[3, 3, 3, 3],
    num_mctdhf_orbitals=[9, 9, 9, 9],
    num_spatial_orbitals=[N] * 4,
    num_states=16,
    imag_time=False,
    num_spfs=[6, 6, 6, 6],   # 1 <= n_s <= D_s = C(9, 3) = 84
    eps_spf=1e-6,            # defaults to eps
)
```

`num_spfs` and `eps_spf` are keyword-only; everything else is the standard
constructor. `MLMCTDHF_n_species` is the 4-index-integral variant.

### Propagating

```python
step = get_ml_cmf_integrator(
    ml, dt=0.01,
    a_propagator="sil",           # top layer
    B_propagator="diagonalize",   # SPF layer
    b_propagator="cn",            # orbital layer
    a_kwargs={"krylov_dim": 10},
    B_kwargs={"n_spf_substeps": 2},
    b_kwargs={"n_orbital_substeps": 4},
)

ts, ys = ml.integrate(step, num_steps=1000, normalize=True,
                      noise_A=1e-3, noise_bs=1e-3, save_every=100)
```

`get_ml_cmf_midpoint_integrator` is the second-order layout
`A/2 . B/2 . b . B/2 . A/2`; it costs one extra top and SPF half-step and
takes the same arguments. These two are the only ML integrators: adaptive
stepping, PSI and the coupled RK4/VMF routes are not implemented for ML.

Option names follow the standard integrators. `a_kwargs` takes the CI keys
(`krylov_dim` for `sil`; `solver_tol` and `max_solver_iter` for `cn`),
`b_kwargs` is exactly the orbital solver's, and `B_kwargs` is the orbital
solver's with `n_orbital_substeps` renamed `n_spf_substeps`. The SPF default
`diagonalize` is a dense matrix exponential of the `D_s * n_s` linearised
generator: exact within the substep and safe for a non-Hermitian generator,
which this one is.

Without explicit initial layers, `integrate` starts from the lowest
eigenvectors of `h_s` for the orbitals, the lowest `n_s` eigenvectors of the
bare species Hamiltonian `Op_s` for the SPFs, and energy-ordered product
states for the top tensor.

### Starting from an existing MCTDHF state

`from_mctdhf` performs a truncated higher-order SVD, keeping the `n_s`
dominant natural SPFs per species, and reports what it threw away:

```python
C, bs = mctdh._y_to_Cb(y_standard)
A, Bs, bs, discarded = ml.from_mctdhf(C, bs)
y_ml = ml._ABb_to_y(A, Bs, bs)
print("discarded weight per species:", [float(d) for d in discarded])

y_back = ml.to_mctdhf_y(y_ml)          # back to the standard layout
```

`from_species_vectors(vecs, A)` builds the SPF layer from given
determinant-space vectors instead, Loewdin-orthonormalising them and
transforming `A` so the state is unchanged. That is the route for a logical
basis of frozen per-well states.

### Monitoring convergence

```python
pops = ml.spf_populations(y)        # eigenvalues of the node density, per species
occ  = ml.natural_occupations(y)    # eigenvalues of the 1-RDM, per species
```

`spf_populations` is the convergence monitor for `n_s`: the weight in the
lowest-populated SPF is what raising `n_s` would recover, and it is the same
quantity `from_mctdhf` reports as discarded. `natural_occupations` is the
monitor for the orbital budget `M_s`, and sums to `N_s`.

### Re-gauging

Non-unitary sub-propagators (Crank-Nicolson in particular) let `b_s` and
`B_s` drift off orthonormality. `make_regauge_ml` repairs both layers exactly,
without changing the state:

```python
regauge = make_regauge_ml(ml)
y = regauge(y)
```

It QR-factorises `b_s`, pushes the triangular factor onto `B_s` through the
N-th compound matrix of `R` (the exterior power, which is what a Slater
determinant of N orbitals transforms with), then QR-factorises `B_s` and
pushes that triangular factor onto `A`. `rotate_to_natural_spfs(y)` is a
second, cheaper gauge transform that diagonalises the node density so the SPF
populations come out ordered.

### Observables

```python
E = ml.calculate_energy(t, y)                       # per state
S = ml.calculate_overlap(y1, y2)                    # per state pair
A, Bs, bs = ml._y_to_ABb(y)
dens = ml.calculate_particle_densities(A, Bs, bs, spf_list)
Ds = ml.get_1p_RDMs(y)
```

The overlap sandwiches the determinant overlap matrices between the two SPF
matrices, so it never forms the expanded CI tensor.
