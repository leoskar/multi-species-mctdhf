import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.special import hermite
from scipy.optimize import linear_sum_assignment
from math import factorial, sqrt, pi
from itertools import product

def plot_energy_convergence(mctdhf, ts, ys, title):
    """Plot per-state energy convergence during ITE."""
    step = max(1, len(ts) // 50)
    idx = list(range(0, len(ts), step))
    if idx[-1] != len(ts) - 1:
        idx.append(len(ts) - 1)
    
    times = [ts[i] for i in idx]
    energies = np.array([np.real(mctdhf.calculate_energy(ts[i], ys[i])) for i in idx])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    if energies.ndim == 2:
        for s in range(energies.shape[1]):
            ax.plot(times, energies[:, s], lw=1.5, label=f'State {s}')
        ax.legend(fontsize=9)
    else:
        ax.plot(times, energies, 'b-', lw=2)
    ax.set_xlabel('Imaginary time τ');  ax.set_ylabel('Energy')
    ax.set_title(title);  ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_densities(mctdhf, y, dvr_list, grids, labels, title,
                   max_states=6, num_plot_points=500):
    """Plot particle densities for all states and species.

    Reconstructs smooth density curves via DVR → FBR transformation of the
    MCTDHF orbitals and contraction with the 1-RDM.  The original DVR grid
    values are overlaid as dots for reference.
    """
    C, bs = mctdhf._y_to_Cb(y)
    spf_list = [jnp.array(d.get_spf()) for d in dvr_list]
    densities = mctdhf.calculate_particle_densities(C, bs, spf_list)

    # 1-RDMs for smooth reconstruction
    Ds = mctdhf.dCI.get_1p_RDMs(C)

    ns = min(C.shape[0], max_states)
    nsp = len(dvr_list)
    fig, axes = plt.subplots(ns, nsp, figsize=(5*nsp, 2.5*ns), squeeze=False)

    for s in range(nsp):
        dvr = dvr_list[s]
        rho_dvr = np.real(np.array(densities[s]))

        # --- smooth reconstruction ---
        N = dvr.num_points
        L = dvr.L
        j = np.arange(1, N + 1)
        n = np.arange(1, N + 1)
        U = np.sqrt(2.0 / (N + 1)) * np.sin(
            np.outer(j, n) * np.pi / (N + 1))

        orbitals = np.asarray(bs[s])            # (N, n_orb)
        fbr_coeffs = U.T @ orbitals             # (N, n_orb)

        x_fine = np.linspace(dvr.x_min, dvr.x_max, num_plot_points)
        phi = np.sqrt(2.0 / L) * np.sin(
            np.outer(x_fine - dvr.x_min, n) * np.pi / L)  # (npts, N)
        psi_fine = phi @ fbr_coeffs             # (npts, n_orb)

        D_s = np.array(Ds[s])                   # (batch, n_orb, n_orb)

        for st in range(ns):
            ax = axes[st, s]
            # smooth density: ρ(x) = Σ_{pq} D_{pq} φ_p*(x) φ_q(x)
            rho_smooth = np.real(np.einsum(
                'xp,pq,xq->x', psi_fine.conj(), D_s[st], psi_fine))
            ax.plot(x_fine, rho_smooth, lw=1.5)
            ax.fill_between(x_fine, rho_smooth, alpha=0.2)
            # DVR grid points as dots
            ax.plot(grids[s], rho_dvr[st], 'o', ms=4, color='C0')
            ax.set_xlabel('x');  ax.set_ylabel('ρ(x)')
            ax.set_title(f'{labels[s]}, State {st}')
            ax.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_wavefunctions(mctdhf, y, dvr_list, labels, title,
                       max_states=6, scale=1.0, num_plot_points=500):
    """Plot smooth MCTDHF orbital wavefunctions for all states and species.

    Uses the DVR → FBR reconstruction so the plotted curves are smooth
    rather than piecewise-constant on the DVR grid.

    Parameters
    ----------
    mctdhf : MCTDHF_n_species or MCTDHF_n_species_DVR
        The MCTDHF solver object.
    y : array
        State vector (from integrate output).
    dvr_list : list of SineDVR
        One DVR basis per species.
    labels : list of str
        Label for each species (used in subplot titles).
    title : str
        Overall figure title.
    max_states : int, optional
        Maximum number of states (rows) to plot.
    scale : float, optional
        Visual scaling factor for wavefunction amplitude.
    num_plot_points : int, optional
        Number of points on the fine plotting grid per species.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    C, bs = mctdhf._y_to_Cb(y)

    ns = min(C.shape[0], max_states)
    nsp = len(dvr_list)
    fig, axes = plt.subplots(ns, nsp, figsize=(5 * nsp, 2.5 * ns), squeeze=False)

    for s in range(nsp):
        dvr = dvr_list[s]
        # bs[s] shape: (num_spatial_orbitals, num_mctdhf_orbitals)
        orbitals = np.asarray(bs[s])

        N = dvr.num_points
        L = dvr.L

        # FBR-to-DVR transformation matrix
        j = np.arange(1, N + 1)
        n = np.arange(1, N + 1)
        U = np.sqrt(2.0 / (N + 1)) * np.sin(np.outer(j, n) * np.pi / (N + 1))

        # DVR → FBR coefficients
        fbr_coeffs = U.T @ orbitals  # (N, num_mctdhf_orbitals)

        # Fine grid for smooth curves
        x_fine = np.linspace(dvr.x_min, dvr.x_max, num_plot_points)

        # FBR basis on fine grid: phi_n(x) = sqrt(2/L) sin(n pi (x-x_min)/L)
        phi = np.sqrt(2.0 / L) * np.sin(
            np.outer(x_fine - dvr.x_min, n) * np.pi / L
        )  # (num_plot_points, N)

        # Smooth wavefunctions
        psi = phi @ fbr_coeffs  # (num_plot_points, num_mctdhf_orbitals)

        for st in range(ns):
            ax = axes[st, s]
            if st < psi.shape[1]:
                wf = np.real(psi[:, st]) * scale
                ax.plot(x_fine, wf, lw=1.5)
                ax.fill_between(x_fine, wf, alpha=0.3)
            ax.set_xlabel('x')
            ax.set_ylabel('ψ(x)')
            ax.set_title(f'{labels[s]}, Orbital {st}')
            ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_state_coefficients(C, state_index, title=None):
    """
    Visualizes the CI coefficients for a specific state.
    
    Args:
        C: Coefficient tensor of shape (num_states, n_dets_1, n_dets_2, ...)
        state_index: Index of the state to visualize
        title: Optional plot title
    """
    # Extract coefficients for the state
    coeffs = np.array(C[state_index])
    
    # Check dimensionality (number of species)
    ndim = coeffs.ndim
    
    # Create title if None
    if title is None:
        title = f"State {state_index} Coefficients"
        
    print(f"Visualizing {title}")
    print(f"Coefficient tensor shape: {coeffs.shape}")
    print(f"Norm: {np.linalg.norm(coeffs):.6f}")
    
    # Visualization based on dimensionality
    if ndim == 1:
        # 1 Species: Bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        indices = np.arange(len(coeffs))
        
        # Plot real and imag parts if complex
        if np.iscomplexobj(coeffs) and np.any(np.abs(np.imag(coeffs)) > 1e-10):
            ax.bar(indices - 0.2, np.real(coeffs), width=0.4, label='Real')
            ax.bar(indices + 0.2, np.imag(coeffs), width=0.4, label='Imag')
            ax.legend()
        else:
            ax.bar(indices, np.real(coeffs))
            
        ax.set_xlabel('Determinant Index')
        ax.set_ylabel('Coefficient Value')
        ax.set_title(title)
        plt.grid(True, alpha=0.3)
        plt.show()
        
    elif ndim == 2:
        # 2 Species: Heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        
        data_real = np.real(coeffs)
        vmin, vmax = data_real.min(), data_real.max()
        limit = max(abs(vmin), abs(vmax))
        vmin, vmax = -limit, limit
        
        im = ax.imshow(data_real, cmap='RdBu', origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label='Coefficient (Real Part)')
        ax.set_xlabel('Species 2 Determinant')
        ax.set_ylabel('Species 1 Determinant')
        ax.set_title(title)
        plt.show()
        
    elif ndim == 3:
        # 3 Species: Subplots for each index of Species 1
        n_subplots = coeffs.shape[0]
        # Calculate grid layout
        cols = min(n_subplots, 3) 
        rows = (n_subplots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows), constrained_layout=True)
        if rows == 1 and cols == 1: axes = [axes]
        axes = np.array(axes).flatten()
        
        # Determine global min/max for consistent colorbar
        data_real = np.real(coeffs)
        vmin, vmax = data_real.min(), data_real.max()
        limit = max(abs(vmin), abs(vmax))
        vmin, vmax = -limit, limit
        
        print(f"Color scale limits: [{vmin:.3f}, {vmax:.3f}]")
        for i in range(n_subplots):
            if i < len(axes):
                ax = axes[i]
                # Slice: coeffs[i, :, :]
                slice_data = data_real[i]
                
                im = ax.imshow(slice_data, cmap='RdBu', origin='lower', vmin=vmin, vmax=vmax)
                ax.set_xlabel('Species 3')
                ax.set_ylabel('Species 2')
                ax.set_title(f'Species 1 Index = {i}')
                plt.colorbar(im, ax=ax)
                
        # Hide unused subplots
        for i in range(n_subplots, len(axes)):
            axes[i].axis('off')
            
        fig.suptitle(title)
        plt.show()
        
    else:
        # >3D: Flatten and show top components
        print("High dimensionality > 3D. Showing top 20 coefficients.")
        flat = coeffs.flatten()
        # Get indices of top 20 by magnitude
        top_indices = np.argsort(np.abs(flat))[-20:]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(range(len(top_indices)), np.real(flat[top_indices]))
        ax.set_xticks(range(len(top_indices)))
        
        # Recover multi-index for labels
        labels = []
        for idx in top_indices:
            multi_idx = np.unravel_index(idx, coeffs.shape)
            labels.append(str(multi_idx))
            
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Coefficient (Real)')
        ax.set_title(f"{title} - Top 20 Determinants")
        plt.show()

def format_coefficient_table(C, state_index, threshold=1e-3):
    """
    Formats a table of CI coefficients for a specific state that are greater than a threshold.
    
    Args:
        C: Coefficient tensor of shape (num_states, n_dets_1, n_dets_2, ...)
        state_index: Index of the state to check
        threshold: Magnitude threshold for printing coefficients
    
    Returns:
        str: Formatted table as a string
    """
    lines = []
    
    # Extract coefficients for the state
    coeffs = np.array(C[state_index])
    
    # Get indices where magnitude > threshold, sorted by excitation number
    significant_indices = np.argwhere(np.abs(coeffs) > threshold)
    sort_key = np.array([idx.sum() for idx in significant_indices])
    significant_indices = significant_indices[np.lexsort((*significant_indices.T[::-1], sort_key))]
    
    lines.append(f"State {state_index} Coefficients (Threshold > {threshold})")
    lines.append("-" * 40)
    lines.append(f"{'Index':<10} | {'Value'}")
    lines.append("-" * 40)
    
    for idx_arr in significant_indices:
        idx = tuple(int(i) for i in idx_arr)
        val = coeffs[idx]
        idx_str = str(idx)
        
        if np.iscomplexobj(coeffs):
            val_str = f"{val:.6f}"
        else:
            val_str = f"{val.real:.6f}"
            
        lines.append(f"{idx_str:<10} | {val_str}")
    
    lines.append("-" * 40)
    lines.append(f"Norm of displayed coeffs: {np.linalg.norm(coeffs[np.abs(coeffs) > threshold]):.6f}")
    lines.append(f"Total Norm: {np.linalg.norm(coeffs):.6f}")
    
    return "\n".join(lines)


def print_coefficient_table(C, state_index, threshold=1e-3):
    """
    Prints a table of CI coefficients for a specific state that are greater than a threshold.
    
    Args:
        C: Coefficient tensor of shape (num_states, n_dets_1, n_dets_2, ...)
        state_index: Index of the state to check
        threshold: Magnitude threshold for printing coefficients
    """
    print(format_coefficient_table(C, state_index, threshold))



def plot_harmonic_oscillator(x_extent, omega, mass=1.0, n_levels=4, hbar=1.0):
    """
    Plot harmonic oscillator potential and wavefunctions.

    Parameters
    ----------
    x_extent : float
        Grid runs from -x_extent to +x_extent.
    omega : float
        Angular frequency of the oscillator.
    mass : float
        Particle mass (default 1.0).
    n_levels : int
        Number of energy levels / wavefunctions to plot.
    hbar : float
        Reduced Planck constant (default 1.0).
    """
    x = np.linspace(-x_extent, x_extent, 500)
    V = 0.5 * mass * omega**2 * x**2

    # Characteristic length scale
    xi = np.sqrt(hbar / (mass * omega))

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot potential
    ax.plot(x, V, 'k-', lw=2, label='V(x)')

    for n in range(n_levels):
        E_n = hbar * omega * (n + 0.5)

        # Normalized wavefunction
        H_n = hermite(n)
        norm = (1.0 / sqrt(2**n * factorial(n))) * (1.0 / (pi * xi**2))**0.25
        psi = norm * H_n(x / xi) * np.exp(-x**2 / (2 * xi**2))

        # Scale wavefunction for display and offset by energy
        scale = E_n * 0.6  # visual scaling factor
        ax.fill_between(x, E_n, E_n + psi * scale, alpha=0.3)
        ax.plot(x, E_n + psi * scale, lw=1.5, label=f'n={n}, E={E_n:.2f}')
        ax.axhline(E_n, color='gray', ls='--', lw=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('Energy / V(x)')
    ax.set_title(f'Harmonic Oscillator (ω = {omega})')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(-x_extent, x_extent)
    ax.set_ylim(0, hbar * omega * (n_levels + 1))
    plt.tight_layout()
    plt.show()


# ── Qubit-fidelity helpers ────────────────────────────────────────────────────


def anharm_pot(a, b, center):
    """Anharmonic well potential V(x) = 0.5 a^2 (x-c)^2 + (b a^2 / 3)(x-c)^4."""
    return lambda x: 0.5 * a**2 * (x - center) ** 2 + ((b * a**2) / 3) * (x - center) ** 4


def single_well_eigenstates(h_s, n_levels=2):
    """Lowest `n_levels` eigenvectors of a single-well Hamiltonian (DVR basis).

    Returns eigvals shape (n_levels,) and eigvecs shape (num_dvr, n_levels).
    JAX-native so this is differentiable through h_s.
    """
    w, v = jnp.linalg.eigh(h_s)
    return w[:n_levels], v[:, :n_levels]


def _energy_order_perm(K, n_levels):
    """Permutation from row-major flatten order to non-interacting energy order.

    Configs are sorted by total excitation (sum of digits) — for n_levels=2
    this is the Hamming weight — then within each shell by the tuple itself
    so that earlier species carry the excitation first. For K=4, n_levels=2
    the resulting bitstring sequence is

        |0000>, |1000>, |0100>, |0010>, |0001>,
        |1100>, |1010>, |1001>, |0110>, |0101>, |0011>,
        |1110>, |1101>, |1011>, |0111>, |1111>.
    """
    configs_rowmajor = list(product(range(n_levels), repeat=K))
    return sorted(
        range(len(configs_rowmajor)),
        key=lambda i: (sum(configs_rowmajor[i]),
                       tuple(-c for c in configs_rowmajor[i])),
    )


def qubit_corner_fidelities(C, K, n_levels=2):
    """F[k, n] = |C[k, n_0, ..., n_{K-1}]|^2 with each n_s ∈ {0, ..., n_levels-1}.

    Assumes ``C`` is in a gauge where, for each species, the leading
    ``n_levels`` orbital indices are the qubit modes (e.g. after a
    state-averaged NO rotation that sorts orbitals by descending occupation).

    The output column order matches ``iter_bitstrings(K, n_levels)``: configs
    are sorted by total excitation level (Hamming weight for n_levels=2),
    then by species-major order so earlier species carry excitation first.
    This matches the non-interacting energy ordering.

    Returns shape (num_states, n_levels**K).
    """
    sl = (slice(None),) + (slice(0, n_levels),) * K
    C_corner = C[sl]
    F_flat = jnp.reshape(jnp.abs(C_corner) ** 2,
                         (C.shape[0], n_levels ** K))
    perm = jnp.array(_energy_order_perm(K, n_levels))
    return F_flat[:, perm]


def state_averaged_qubit_basis(C_subspace, bs, dCI, n_levels=2):
    """Per-species qubit basis from natural orbitals of a state-averaged 1-RDM.

    C_subspace : (S, M_0, ..., M_{K-1}) CI tensor of the S states to average over
                 (e.g. the lowest 16 eigenstates spanning the qubit subspace).
    bs : list of orbital matrices, each (num_dvr_s, M_s).
    dCI : SemiDirectCI_n_species providing get_1p_RDMs.
    n_levels : 2 → take the 2 leading NOs per species as |0>, |1>.

    Returns a list of length K: NO_dvr_s of shape (num_dvr_s, n_levels), the
    natural orbitals expressed in the DVR basis (compatible with v_per_well in
    qubit_fidelity_matrix). Sign convention: largest |value| in each column is
    real-positive.
    """
    K = len(bs)
    Ds = dCI.get_1p_RDMs(C_subspace)  # list of (S, M_s, M_s)
    no_dvr_per_well = []
    for s in range(K):
        D_avg = jnp.mean(Ds[s], axis=0)
        D_avg = 0.5 * (D_avg + D_avg.conj().T)
        occ, V = jnp.linalg.eigh(D_avg)
        # Sort by descending occupation, take the n_levels leading NOs.
        order = jnp.argsort(-occ)[:n_levels]
        V_top = V[:, order]                       # (M_s, n_levels)
        NO_dvr = bs[s] @ V_top                    # (num_dvr_s, n_levels)
        # Sign-fix: largest-|.| entry of each column is real-positive.
        idx = jnp.argmax(jnp.abs(NO_dvr), axis=0)
        leading = NO_dvr[idx, jnp.arange(n_levels)]
        phases = jnp.where(jnp.abs(leading) > 0,
                           jnp.conj(leading) / jnp.abs(leading), 1.0)
        NO_dvr = NO_dvr * phases[None, :]
        no_dvr_per_well.append(NO_dvr)
    return no_dvr_per_well


def hartree_overlap(C_k, bs, u_list):
    """Scalar overlap <product | Psi_k> for a single state.

    C_k : CI tensor of shape (M_0, ..., M_{K-1}) for ONE state.
    bs  : list of orbital matrices, each (num_dvr_s, M_s).
    u_list : list of K vectors u_s of shape (num_dvr_s,), each being the
             single-particle target state for species s (e.g. the DVR-basis
             eigenvector v_s^{(n_s)}). The overlap is

                 <target | Psi_k> = sum_{i_0..i_{K-1}} C_k[i_0,...,i_{K-1}]
                                    * prod_s (u_s^dagger @ bs[s])[i_s].
    """
    K = len(bs)
    coeffs = [jnp.einsum('x,xm->m', jnp.conj(u_list[s]), bs[s]) for s in range(K)]
    # Build einsum: 'a,b,c,...,abc...->'
    letters = [chr(ord('a') + s) for s in range(K)]
    ci_str = ''.join(letters)
    args_str = ','.join(letters) + ',' + ci_str + '->'
    return jnp.einsum(args_str, *coeffs, C_k)


def qubit_fidelity_matrix(C, bs, v_per_well, n_levels=2):
    """Fidelity matrix F_{k,n} = |<bitstring_n | Psi_k>|^2.

    C : CI tensor of shape (num_states, M_0, ..., M_{K-1}).
    bs : list of orbital matrices, each (num_dvr_s, M_s).
    v_per_well : list of K arrays, each shape (num_dvr_s, n_levels), holding
                 the `n_levels` lowest single-well eigenvectors per species.
    n_levels : 2 for a qubit subspace (|0>, |1>). Returned matrix is
               (num_states, n_levels**K) with columns ordered by total
               excitation, then species-major (see `iter_bitstrings`).
    """
    K = len(bs)
    num_states = C.shape[0]
    configs = iter_bitstrings(K, n_levels)
    n_configs = len(configs)
    F = jnp.zeros((num_states, n_configs))
    for n_idx, config in enumerate(configs):
        u_list = [v_per_well[s][:, config[s]] for s in range(K)]
        overlaps = jax_vmap_hartree_overlap(C, bs, u_list)  # shape (num_states,)
        F = F.at[:, n_idx].set(jnp.abs(overlaps) ** 2)
    return F


def jax_vmap_hartree_overlap(C, bs, u_list):
    """Vectorised hartree_overlap over the leading state axis of C."""
    K = len(bs)
    coeffs = [jnp.einsum('x,xm->m', jnp.conj(u_list[s]), bs[s]) for s in range(K)]
    letters = [chr(ord('a') + s) for s in range(K)]
    ci_str = 'S' + ''.join(letters)
    args_str = ','.join(letters) + ',' + ci_str + '->S'
    return jnp.einsum(args_str, *coeffs, C)


def iter_bitstrings(K, n_levels=2):
    """Enumerate configurations in the same order as `qubit_fidelity_matrix`.

    Sorted by total excitation (Hamming weight for n_levels=2), then by
    species-major order so earlier species carry excitation first.
    """
    configs_rowmajor = list(product(range(n_levels), repeat=K))
    perm = _energy_order_perm(K, n_levels)
    return [configs_rowmajor[i] for i in perm]


def match_configs(F, weights=None):
    """Hungarian assignment of configs to eigenstates.

    F : (num_states, n_configs) fidelity matrix (|overlap|^2).
    weights : (n_configs,) non-negative weights, or None for uniform weights.
    Returns (matched_idx, fidelities_by_config, loss) where matched_idx[n] is
    the eigenstate chosen for config n and
    loss = sum(weights) - sum_n weights[n] * F[matched_idx[n], n].
    """
    F_np = np.asarray(F)
    n_configs = F_np.shape[1]
    if weights is None:
        weights = np.ones(n_configs)
    else:
        weights = np.asarray(weights, dtype=float)
    # linear_sum_assignment minimises cost; maximise sum(w_n * F[i_n, n]).
    row_idx, col_idx = linear_sum_assignment(-(F_np * weights[None, :]))
    matched_idx = np.empty(n_configs, dtype=int)
    matched_idx[col_idx] = row_idx
    matched_fidelities = F_np[row_idx, col_idx]
    fidelities_by_config = np.empty(n_configs)
    fidelities_by_config[col_idx] = matched_fidelities
    loss = float(weights.sum() - (weights * fidelities_by_config).sum())
    return matched_idx, fidelities_by_config, loss


def _bitstring_label(config):
    return '|' + ''.join(str(n) for n in config) + '>'


def plot_qubit_ladder(energies, matched_idx, configs, ax=None, title=None):
    """Horizontal energy level diagram coloured by matched qubit config.

    energies : (num_states,) array of eigenstate energies.
    matched_idx : (n_configs,) eigenstate index per config.
    configs : list of bitstring tuples in the same order as matched_idx.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))
    else:
        fig = ax.figure
    energies = np.asarray(energies).real
    state_to_config = {int(matched_idx[n]): configs[n] for n in range(len(configs))}
    for k, E in enumerate(energies):
        if k in state_to_config:
            cfg = state_to_config[k]
            excite = sum(cfg)
            colour = f'C{excite}'
            ax.hlines(E, 0.1, 0.9, colors=colour, lw=2)
            ax.text(0.92, E, _bitstring_label(cfg), va='center', fontsize=8, color=colour)
        else:
            ax.hlines(E, 0.1, 0.9, colors='lightgray', lw=1)
            ax.text(0.92, E, f'#{k}', va='center', fontsize=7, color='gray')
    ax.set_xlim(0, 1.3)
    ax.set_xticks([])
    ax.set_ylabel('Energy')
    if title:
        ax.set_title(title)
    return fig


def plot_fidelity_bars(fidelities_by_config, configs, ax=None, title=None):
    """Bar chart of best-matched fidelity per qubit config."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure
    labels = [_bitstring_label(c) for c in configs]
    excitations = [sum(c) for c in configs]
    colours = [f'C{e}' for e in excitations]
    xs = np.arange(len(configs))
    ax.bar(xs, fidelities_by_config, color=colours)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.axhline(0.98, color='k', ls='--', lw=0.7, alpha=0.5)
    ax.set_ylabel('|<target | Psi_matched>|²')
    ax.set_ylim(0, 1.02)
    worst = int(np.argmin(fidelities_by_config))
    ax.annotate(
        f'worst: {labels[worst]} = {fidelities_by_config[worst]:.3f}',
        xy=(worst, fidelities_by_config[worst]),
        xytext=(worst, 0.3),
        ha='center', fontsize=8,
        arrowprops=dict(arrowstyle='->', color='red', lw=0.8),
    )
    if title:
        ax.set_title(title)
    return fig