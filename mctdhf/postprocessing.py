"""Post-processing utilities for N-species MCTDHF.

Free functions that operate on a converged or propagated state produced by
:class:`mctdhf.MCTDHF_n_species.MCTDHF_n_species`. These do not participate in
the time-propagation hot path; they compute observables (particle densities),
characterize states (subsystem character, localization), and perform exact
gauge transforms (rotation into the natural-orbital basis, with or without
continuity across snapshots).

Each public function takes the solver instance as its first argument so that it
can reach ``mctdh.dCI``, ``mctdh.num_species``, ``mctdh._y_to_Cb``, and
``mctdh._Cb_to_y``. The class retains thin method wrappers that delegate here.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from opt_einsum import contract
from scipy.optimize import linear_sum_assignment


# ── Shared primitives ─────────────────────────────────────────────────────────


def _sa_1rdms(mctdh, C):
    """State-averaged 1-RDMs per species as numpy arrays of shape (M_s, M_s).

    Uses a uniform sum over states (matching the orbital optimizer's
    ``_state_average`` when ``state_weights`` is ``None``).
    """
    return [
        np.array(jnp.sum(D, axis=0)).astype(
            np.complex128 if np.iscomplexobj(D) else np.float64, copy=False
        )
        for D in mctdh.dCI.get_1p_RDMs(C)
    ]


def _canonical_phase(V):
    """Fix each column's phase so its largest-|.| entry is positive real.

    Operates on a copy of the input numpy array and returns the fixed array.
    """
    V = V.copy()
    for k in range(V.shape[1]):
        ph = V[np.argmax(np.abs(V[:, k])), k]
        V[:, k] *= np.conj(ph) / np.abs(ph)
    return V


def _align_columns(V, bs_s, prev_bs_s):
    """Permute and phase ``V``'s columns so that the resulting natural orbitals
    ``(bs_s @ V)[:, k]`` track ``prev_bs_s[:, k]`` across time.

    Uses the Hungarian algorithm on ``-|⟨prev, cand⟩|`` and picks the phase that
    makes each matched overlap positive real.
    """
    M = (bs_s @ jnp.array(V, dtype=bs_s.dtype)).conj().T @ prev_bs_s
    M = np.array(M)
    row, col = linear_sum_assignment(-np.abs(M))
    perm = np.empty_like(row)
    perm[row] = col
    V = V[:, perm]
    phases = np.exp(-1j * np.angle(np.diag(M[row][:, col])))
    return V * phases[None, :]


def _apply_gauge_transform(C, bs, Us, num_species):
    """Apply ``b_s → b_s U_s`` and ``C → C ×_s U_s†``.

    Parameters
    ----------
    C : jnp.ndarray
        CI tensor, shape ``(num_states, M_0, ..., M_{K-1})``.
    bs : list of jnp.ndarray
        Orbital coefficient matrices.
    Us : list of jnp.ndarray
        Per-species unitary gauge transformations.
    num_species : int
        Number of species (length of ``bs`` and ``Us``).

    Returns
    -------
    C_new, bs_new
        Transformed CI tensor and orbital list; the physical state is unchanged.
    """
    bs_new = [bs[s] @ Us[s] for s in range(num_species)]
    C_new = C
    for s, U in enumerate(Us):
        C_new = jnp.tensordot(C_new, U.conj(), axes=([s + 1], [0]))
        # tensordot appends the new contracted axis at the end; move it back.
        nd = C_new.ndim
        perm = list(range(nd))
        perm = perm[: s + 1] + [perm[-1]] + perm[s + 1 : -1]
        C_new = jnp.transpose(C_new, perm)
    return C_new, bs_new


# ── Observables ───────────────────────────────────────────────────────────────


@partial(jit, static_argnums=(0,))
def particle_densities(mctdh, C, bs_list, spf_list=None):
    """Compute particle densities for all species.

    If ``spf_list`` is supplied, returns the density on the underlying grid,
    ``ρ(x) = Σ_{nm} D_{nm} φ*_n(x) φ_m(x)``. Otherwise returns the density
    matrix in the spatial-orbital basis.
    """
    Ds = mctdh.dCI.get_1p_RDMs(C)
    densities = []

    for s in range(mctdh.num_species):
        D_s = Ds[s]
        b_s = bs_list[s]

        if spf_list is not None and len(spf_list) > s:
            spf = spf_list[s]
            rho = contract(
                "ix,in,snm,jm,jx->sx", spf, b_s.conj(), D_s, b_s, spf, backend="jax"
            )
        else:
            rho = contract(
                "in, snm, jm -> sij", b_s.conj(), D_s, b_s, backend="jax"
            )
        densities.append(rho)
    return densities


def subsystem_character(mctdh, C):
    """Weighted-average orbital index per (state, species).

    Entry ``[α, s]`` is ``⟨n⟩_s = Σ_n n·|D_{nn}| / Σ_n |D_{nn}|``. Higher values
    indicate higher excitation levels in subsystem ``s`` for state ``α``.
    """
    Ds = mctdh.dCI.get_1p_RDMs(C)
    num_states = C.shape[0]
    character = jnp.zeros((num_states, mctdh.num_species))

    for s in range(mctdh.num_species):
        D_s = Ds[s]
        n_orb = D_s.shape[-1]
        orbital_weights = jnp.arange(n_orb, dtype=jnp.float32)

        diags = jnp.diagonal(D_s, axis1=-2, axis2=-1)
        weights = jnp.abs(diags)
        avg_orbital = jnp.sum(orbital_weights * weights, axis=-1) / jnp.sum(
            weights, axis=-1
        )
        character = character.at[:, s].set(avg_orbital)

    return character


# ── State localization ────────────────────────────────────────────────────────


def _subsystem_rdm(C_single, s):
    """Partial-trace RDM of species ``s`` for a single state (unnormalized)."""
    C_moved = jnp.moveaxis(C_single, s, 0)
    n_det_s = C_moved.shape[0]
    M = C_moved.reshape(n_det_s, -1)
    return M @ M.conj().T


def _compute_total_purity(C_sub, num_species):
    """Sum of ``Tr(ρ_s²)`` across states in ``C_sub`` and species."""
    n_sub = C_sub.shape[0]
    total_purity = jnp.zeros(())
    for alpha in range(n_sub):
        for s in range(num_species):
            rho_s = _subsystem_rdm(C_sub[alpha], s)
            total_purity = total_purity + jnp.real(jnp.trace(rho_s @ rho_s))
    return total_purity


def _compute_optimal_angle(C_sub, ii, jj, num_species):
    """Optimal Jacobi rotation angle for states ``ii``/``jj`` to maximize purity.

    ``θ_opt = atan2(2E, D) / 4`` where the coefficients aggregate subsystem
    transition traces; see Edmiston–Ruedenberg-style derivations.
    """
    D_total = jnp.zeros(())
    E_total = jnp.zeros(())

    for s in range(num_species):
        rho_i = _subsystem_rdm(C_sub[ii], s)
        rho_j = _subsystem_rdm(C_sub[jj], s)

        C_i_moved = jnp.moveaxis(C_sub[ii], s, 0)
        C_j_moved = jnp.moveaxis(C_sub[jj], s, 0)
        n_det_s = C_i_moved.shape[0]
        M_i = C_i_moved.reshape(n_det_s, -1)
        M_j = C_j_moved.reshape(n_det_s, -1)
        T = M_i @ M_j.conj().T
        R = T + T.conj().T

        t_AA = jnp.real(jnp.trace(rho_i @ rho_i))
        t_BB = jnp.real(jnp.trace(rho_j @ rho_j))
        t_AB = jnp.real(jnp.trace(rho_i @ rho_j))
        t_RR = jnp.real(jnp.trace(R @ R))
        t_AR = jnp.real(jnp.trace(rho_i @ R))
        t_BR = jnp.real(jnp.trace(rho_j @ R))

        D_total = D_total + (t_AA + t_BB - t_RR - 2 * t_AB)
        E_total = E_total + (t_AR - t_BR)

    return jnp.arctan2(2 * E_total, D_total) / 4


def localize_states(mctdh, y, energy_tol=1e-6, max_sweeps=20, tol=1e-8, verbose=False):
    """Rotate degenerate states to maximize subsystem purity (Hartree products).

    Sorts states by energy, groups those within ``energy_tol``, and runs Jacobi
    sweeps within each degenerate subspace to maximize total ``Σ_s Tr(ρ_s²)``.

    Returns
    -------
    y_localized : jnp.ndarray
        Packed state vector with energy-ordered, localized states.
    character : jnp.ndarray
        Subsystem character matrix ``(num_states, num_species)``.
    groups : list of list of int
        Degenerate-subspace groupings used for localization.
    """
    C, bs = mctdh._y_to_Cb(y)
    num_states = C.shape[0]

    if num_states <= 1:
        return y, subsystem_character(mctdh, C), [[0]]

    # Step 1: energy-sort
    energies = jnp.real(mctdh.calculate_energy(0.0, y))
    energy_order = jnp.argsort(energies)
    C = C[energy_order]
    energies = energies[energy_order]

    if verbose:
        print(f"  State energies: {[f'{e:.6f}' for e in energies]}")

    # Step 2: group degenerate states
    groups = []
    current_group = [0]
    for i in range(1, num_states):
        if jnp.abs(energies[i] - energies[current_group[0]]) < energy_tol:
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]
    groups.append(current_group)

    if verbose:
        for g_idx, group in enumerate(groups):
            E_avg = float(jnp.mean(energies[jnp.array(group)]))
            print(
                f"  Group {g_idx}: states {group}, E ≈ {E_avg:.6f}, size={len(group)}"
            )

    # Step 3: Jacobi sweeps within each degenerate subspace
    K = mctdh.num_species
    for group in groups:
        if len(group) <= 1:
            continue

        group_indices = jnp.array(group)
        C_sub = C[group_indices]
        n_sub = len(group)

        initial_purity = _compute_total_purity(C_sub, K)
        if verbose:
            print(f"\n  Localizing group {group} (purity={initial_purity:.6f})...")

        for sweep in range(max_sweeps):
            max_rotation = 0.0

            for ii in range(n_sub):
                for jj in range(ii + 1, n_sub):
                    theta_opt = _compute_optimal_angle(C_sub, ii, jj, K)

                    if jnp.abs(theta_opt) > tol:
                        max_rotation = max(max_rotation, float(jnp.abs(theta_opt)))

                        cos_t = jnp.cos(theta_opt)
                        sin_t = jnp.sin(theta_opt)
                        C_i_new = cos_t * C_sub[ii] + sin_t * C_sub[jj]
                        C_j_new = -sin_t * C_sub[ii] + cos_t * C_sub[jj]
                        C_sub = C_sub.at[ii].set(C_i_new)
                        C_sub = C_sub.at[jj].set(C_j_new)

            if verbose:
                purity = _compute_total_purity(C_sub, K)
                print(
                    f"    Sweep {sweep+1}: max_rotation={max_rotation:.6f}, purity={purity:.6f}"
                )

            if max_rotation < tol:
                if verbose:
                    print(f"    Converged after {sweep+1} sweeps")
                break

        C = C.at[group_indices].set(C_sub)

        if verbose:
            final_purity = _compute_total_purity(C_sub, K)
            print(f"    Purity: {initial_purity:.6f} → {final_purity:.6f}")

    y_localized = mctdh._Cb_to_y(C, bs)
    character = subsystem_character(mctdh, C)
    return y_localized, character, groups


# ── Natural-orbital gauge transforms ──────────────────────────────────────────


def rotate_to_natural_orbitals(mctdh, y):
    """Rotate ``(C, bs)`` into the natural-orbital gauge of the current state.

    Exact gauge transform — the physical state is preserved. Fixes each column's
    phase so the largest-|.| entry is positive real.
    """
    C, bs = mctdh._y_to_Cb(y)
    Us = []
    for D in _sa_1rdms(mctdh, C):
        w, V = np.linalg.eigh(D)
        V = _canonical_phase(V[:, np.argsort(-w)])
        Us.append(jnp.array(V, dtype=C.dtype))
    C_new, bs_new = _apply_gauge_transform(C, bs, Us, mctdh.num_species)
    return mctdh._Cb_to_y(C_new, bs_new)


def rotate_snapshots_to_NO(mctdh, ys):
    """Per-snapshot rotation into natural-orbital gauge, with orbital labels
    and phases aligned across time via the Hungarian algorithm on the orbital
    overlap matrix. The physical trajectory is unchanged."""
    ys_no = []
    prev_bs = None
    for y in ys:
        C, bs = mctdh._y_to_Cb(y)
        Us = []
        for s, D in enumerate(_sa_1rdms(mctdh, C)):
            w, V = np.linalg.eigh(D)
            V = V[:, np.argsort(-w)]
            V = (
                _canonical_phase(V) if prev_bs is None
                else _align_columns(V, bs[s], prev_bs[s])
            )
            Us.append(jnp.array(V, dtype=C.dtype))
        C_no, bs_no = _apply_gauge_transform(C, bs, Us, mctdh.num_species)
        ys_no.append(mctdh._Cb_to_y(C_no, bs_no))
        prev_bs = bs_no
    return jnp.stack(ys_no)
