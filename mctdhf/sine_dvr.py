"""Sine-DVR (Discrete Variable Representation) basis implementation.

This module provides a Sine-DVR basis class for use with MCTDHF calculations.
The Sine-DVR basis uses particle-in-a-box eigenfunctions (sine functions) as
the underlying FBR basis, which gives exact kinetic energy matrix elements 
and diagonal potential matrix elements at the DVR grid points.

Note: This is different from sinc-DVR (Colbert & Miller) which uses sinc 
functions for an infinite domain. Sine-DVR has hard-wall boundary conditions.
"""

import numpy as np
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt


class SineDVR:
    """Sine-DVR basis for quantum dynamics calculations.
    
    The Sine-DVR uses particle-in-a-box eigenfunctions as the underlying 
    basis, which gives analytic kinetic energy matrix elements and diagonal
    potential matrix elements at the DVR grid points.
    
    Attributes
    ----------
    x_min : float
        Left boundary of the spatial domain.
    x_max : float
        Right boundary of the spatial domain.
    num_points : int
        Number of DVR grid points.
    grid : ndarray (num_points,)
        DVR grid points.
    dx : float
        Grid spacing.
    T : ndarray (num_points, num_points)
        Kinetic energy matrix elements.
    spf : ndarray (num_points, num_points)
        Single-particle functions (identity for DVR).
    
    Examples
    --------
    >>> dvr = SineDVR(x_min=-10, x_max=10, num_points=64)
    >>> h = dvr.get_h(potential=lambda x: 0.5 * x**2)
    >>> g = dvr.get_g(interaction=lambda r: 1.0/np.abs(r + 0.1))
    """
    
    def __init__(self, x_min, x_max, num_points, mass=1.0):
        """Initialize the Sine-DVR basis.
        
        Parameters
        ----------
        x_min : float
            Left boundary of the spatial domain.
        x_max : float
            Right boundary of the spatial domain.
        num_points : int
            Number of DVR grid points.
        mass : float, optional
            Particle mass (default 1.0, i.e., atomic units).
        """
        self.x_min = x_min
        self.x_max = x_max
        self.num_points = num_points
        self.mass = mass
        
        # Box length
        self.L = x_max - x_min
        
        # Grid spacing
        self.dx = self.L / (num_points + 1)
        
        # DVR grid points (interior points only)
        self.grid = x_min + self.dx * np.arange(1, num_points + 1)
        
        # Compute kinetic energy matrix elements via FBR-DVR transformation
        self.T = self._compute_kinetic_energy()
        
        # Compute first derivative matrix via FBR-DVR transformation
        self.D = self._compute_first_derivative()
        
        # For DVR, spf is identity (basis functions are delta functions at grid points)
        # But for density calculations, we need sqrt(dx) normalization
        self.spf = np.eye(num_points) / np.sqrt(self.dx)
    
    def _compute_kinetic_energy(self):
        """Compute the kinetic energy matrix elements.
        
        Uses the FBR-to-DVR transformation with particle-in-a-box eigenfunctions:
        T_DVR = U^T @ T_FBR @ U
        
        where U_jn = sqrt(2/(N+1)) * sin(n*pi*j/(N+1)) is the transformation 
        matrix and T_FBR is diagonal with eigenvalues (hbar^2/2m)*(n*pi/L)^2.
        
        Returns
        -------
        T : ndarray (num_points, num_points)
            Kinetic energy matrix in DVR representation.
        """
        N = self.num_points
        L = self.L
        
        # FBR-to-DVR transformation matrix
        # U_jn = sqrt(2/(N+1)) * sin(n*pi*j/(N+1))
        # j = 1, 2, ..., N (DVR points)
        # n = 1, 2, ..., N (FBR basis functions)
        j = np.arange(1, N + 1)
        n = np.arange(1, N + 1)
        
        # Create the transformation matrix
        U = np.sqrt(2.0 / (N + 1)) * np.sin(np.outer(j, n) * np.pi / (N + 1))
        
        # FBR kinetic energy eigenvalues: T_nn = (hbar^2/2m) * (n*pi/L)^2
        # With hbar = 1 in atomic units
        T_FBR_diag = (n * np.pi / L) ** 2 / (2 * self.mass)
        
        # Transform to DVR: T_DVR = U @ diag(T_FBR) @ U^T
        T = U @ np.diag(T_FBR_diag) @ U.T
        
        return T

    def _compute_first_derivative(self):
        """Compute the first derivative matrix elements analytically.
        
        Uses the FBR-to-DVR transformation with particle-in-a-box eigenfunctions.
        The FBR matrix elements of d/dx are:
        
            D^FBR_{nm} = 4nm / (L(n² - m²))   when n+m is odd
            D^FBR_{nm} = 0                      otherwise
        
        The DVR representation is obtained via:
            D_DVR = U @ D_FBR @ U^T
        
        Returns
        -------
        D : ndarray (num_points, num_points)
            First derivative matrix in DVR representation.
        """
        N = self.num_points
        L = self.L
        
        # FBR-to-DVR transformation matrix (same as for kinetic energy)
        j = np.arange(1, N + 1)
        n = np.arange(1, N + 1)
        U = np.sqrt(2.0 / (N + 1)) * np.sin(np.outer(j, n) * np.pi / (N + 1))
        
        # First derivative matrix in FBR basis
        # D^FBR_{nm} = 4nm / (L(n² - m²)) when n+m is odd, 0 otherwise
        nn, mm = np.meshgrid(n, n, indexing='ij')
        parity_mask = (nn + mm) % 2 == 1  # True when n+m is odd
        
        # Avoid division by zero on diagonal (n=m) and even-parity entries
        denom = np.where(parity_mask, nn**2 - mm**2, 1.0)
        D_FBR = np.where(parity_mask, 4.0 * nn * mm / (L * denom), 0.0)
        
        # Transform to DVR: D_DVR = U @ D_FBR @ U^T
        D = U @ D_FBR @ U.T
        
        return D
    
    def get_position_matrix(self):
        """Get the position operator matrix (diagonal in DVR).
        
        Returns
        -------
        X : ndarray (num_points, num_points)
            Diagonal position matrix with X_{jj} = x_j.
        """
        return np.diag(self.grid)
    
    def get_momentum_matrix(self):
        """Get the momentum operator matrix.
        
        In atomic units (ħ=1), the momentum operator is p = -i d/dx.
        The first derivative is computed analytically in the sine-DVR basis.
        
        Returns
        -------
        P : ndarray (num_points, num_points)
            Momentum matrix P = -i D where D is the first derivative matrix.
        """
        return -1j * self.D

    
    def get_potential_matrix(self, potential):
        """Get the potential energy matrix (diagonal in DVR).
        
        Parameters
        ----------
        potential : callable
            Potential function V(x) that takes grid positions.
            
        Returns
        -------
        V : ndarray (num_points, num_points)
            Diagonal potential matrix.
        """
        V_diag = potential(self.grid)
        return np.diag(V_diag)
    
    def get_h(self, potential=None, time_dependent_potential=None):
        """Get the single-particle Hamiltonian matrix elements.
        
        Parameters
        ----------
        potential : callable, optional
            Time-independent potential V(x).
        time_dependent_potential : callable, optional
            Time-dependent potential V(x, t). Should return V evaluated at x.
            If provided, returns a callable h(t) instead of a matrix.
            
        Returns
        -------
        h : ndarray or callable
            If time_dependent_potential is None: (num_points, num_points) matrix
            If time_dependent_potential is provided: callable h(t) returning matrix
        """
        # Start with kinetic energy
        h_static = self.T.copy()
        
        # Add time-independent potential if provided
        if potential is not None:
            h_static += self.get_potential_matrix(potential)
        
        # If no time-dependent potential, return static matrix
        if time_dependent_potential is None:
            return h_static
        
        # Return callable for time-dependent case
        def h_of_t(t):
            V_t = time_dependent_potential(self.grid, t)
            return h_static + np.diag(V_t)
        
        return h_of_t
    
    def get_g(self, interaction):
        """Get the 2-body interaction matrix elements (same species).
        
        For DVR, the 2-body integrals are diagonal in the combined spatial
        index, giving a 2D matrix g_{ij} = V(x_i, x_j).
        
        Parameters
        ----------
        interaction : callable
            Two-body interaction V(x1, x2) or V(r) where r = |x1 - x2|.
            The function should accept broadcasted arrays.
            
        Returns
        -------
        g : ndarray (num_points, num_points)
            2D interaction matrix (diagonal in DVR for each particle).
        """
        # Create meshgrid for all pairs
        x1, x2 = np.meshgrid(self.grid, self.grid, indexing='ij')
        
        # Evaluate interaction (try both signatures)
        try:
            g = interaction(x1, x2)
        except TypeError:
            # If interaction takes |r|, compute distance
            r = np.abs(x1 - x2)
            g = interaction(r)
        
        return g
    
    def get_g_cross_species(self, other_dvr, interaction):
        """Get 2-body interaction matrix elements between different species.
        
        For multi-species calculations where species may have different 
        spatial representations (e.g., non-overlapping spatial distributions).
        
        Parameters
        ----------
        other_dvr : SineDVR
            The DVR basis for the other species.
        interaction : callable
            Two-body interaction V(x1, x2) taking positions of both species.
            
        Returns
        -------
        g : ndarray (num_points_self, num_points_other)
            2D cross-species interaction matrix.
        """
        # Create meshgrid between the two grids
        x1, x2 = np.meshgrid(self.grid, other_dvr.grid, indexing='ij')
        
        try:
            g = interaction(x1, x2)
        except TypeError:
            r = np.abs(x1 - x2)
            g = interaction(r)
        
        return g
    
    def get_spf(self):
        """Get single-particle functions for density calculations.
        
        For DVR, these are normalized delta functions at grid points.
        The returned array can be passed to MCTDHF.calculate_particle_densities.
        
        Returns
        -------
        spf : ndarray (num_points, num_points)
            SPF matrix where spf[i, j] = delta_{ij} / sqrt(dx).
        """
        return self.spf
    
    def get_grid(self):
        """Get the DVR grid points.
        
        Returns
        -------
        grid : ndarray (num_points,)
            Grid positions.
        """
        return self.grid
    
    def plot_density(self, density, ax=None, **kwargs):
        """Plot a particle density.
        
        Parameters
        ----------
        density : ndarray (num_points,) or (batch, num_points)
            Particle density on the grid.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        **kwargs : dict
            Additional arguments passed to plt.plot.
            
        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the plot.
        """
        
        if ax is None:
            fig, ax = plt.subplots()
        
        density = np.asarray(density)
        if density.ndim == 1:
            ax.plot(self.grid, density, **kwargs)
        else:
            for i, d in enumerate(density):
                ax.plot(self.grid, d, label=f'State {i}', **kwargs)
            ax.legend()
        
        ax.set_xlabel('x')
        ax.set_ylabel('Density')
        
        return ax
    
    def plot_wavefunctions(self, eigenvectors, energies=None, states=None,
                           potential=None, scale=1.0, num_plot_points=500,
                           ax=None, **kwargs):
        """Plot wavefunctions obtained from diagonalizing the DVR Hamiltonian.
        
        Reconstructs smooth wavefunctions by transforming the DVR eigenvector
        coefficients to the FBR (particle-in-a-box) basis and evaluating
        
            psi(x) = sum_n d_n * phi_n(x)
        
        where d_n = (U^T @ c)_n are the FBR coefficients and
        phi_n(x) = sqrt(2/L) * sin(n*pi*(x - x_min)/L) are the FBR basis
        functions.
        
        Parameters
        ----------
        eigenvectors : ndarray (num_points, num_states)
            Matrix of eigenvectors as columns, e.g. from np.linalg.eigh(H).
            Each column eigenvectors[:, i] is one eigenstate.
        energies : ndarray (num_states,), optional
            Eigenvalues corresponding to each eigenvector. If provided,
            wavefunctions are vertically offset by their energy and the
            potential is drawn as a reference.
        states : list of int or int, optional
            Which states to plot (0-indexed). Default: all columns in
            eigenvectors.
        potential : callable, optional
            Potential V(x) to overlay on the plot.
        scale : float, optional
            Scaling factor for the wavefunction amplitude (default 1.0).
            Useful when plotting with energy offsets to control visual size.
        num_plot_points : int, optional
            Number of points on the fine grid for smooth plotting (default 500).
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        **kwargs : dict
            Additional keyword arguments passed to ax.plot for wavefunction
            lines.
            
        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the plot.
        """
        
        if ax is None:
            fig, ax = plt.subplots()
        
        eigenvectors = np.asarray(eigenvectors)
        
        # Determine which states to plot
        if states is None:
            n_states = eigenvectors.shape[1]
            states = list(range(n_states))
        elif isinstance(states, (int, np.integer)):
            states = [states]
        
        N = self.num_points
        L = self.L
        
        # FBR-to-DVR transformation matrix (same as used in __init__)
        j = np.arange(1, N + 1)
        n = np.arange(1, N + 1)
        U = np.sqrt(2.0 / (N + 1)) * np.sin(np.outer(j, n) * np.pi / (N + 1))
        
        # Transform DVR coefficients to FBR coefficients: d = U^T @ c
        # eigenvectors[:, i] are DVR coefficients for state i
        fbr_coeffs = U.T @ eigenvectors  # (N, num_states)
        
        # Fine grid for smooth plotting (including boundary points where psi=0)
        x_fine = np.linspace(self.x_min, self.x_max, num_plot_points)
        
        # Evaluate FBR basis functions on fine grid:
        # phi_n(x) = sqrt(2/L) * sin(n*pi*(x - x_min)/L)
        # psi(x) = sum_n d_n * phi_n(x)
        phi = np.sqrt(2.0 / L) * np.sin(
            np.outer(x_fine - self.x_min, n) * np.pi / L
        )  # (num_plot_points, N)
        
        # Reconstruct wavefunctions on fine grid
        psi = phi @ fbr_coeffs  # (num_plot_points, num_states)
        
        # Plot potential if provided
        if potential is not None:
            V = potential(x_fine)
            ax.plot(x_fine, V, 'k--', lw=1.0, label='V(x)', zorder=0)
        
        # Plot each requested state
        for i in states:
            wf = psi[:, i] * scale
            label = f'$\\psi_{{{i}}}$'
            
            if energies is not None:
                offset = energies[i]
                label += f' (E={energies[i]:.4f})'
                ax.axhline(offset, color='gray', lw=0.5, ls=':', zorder=0)
                ax.plot(x_fine, wf + offset, label=label, **kwargs)
            else:
                ax.plot(x_fine, wf, label=label, **kwargs)
        
        ax.set_xlabel('x')
        ax.set_ylabel('$\\psi(x)$' if energies is None else 'Energy')
        ax.legend()
        
        return ax
    
    def to_jax(self):
        """Convert arrays to JAX arrays for use with MCTDHF.
        
        Returns
        -------
        T, spf, grid : jnp.ndarray
            JAX versions of kinetic energy, SPF, and grid.
        """
        return jnp.array(self.T), jnp.array(self.spf), jnp.array(self.grid)


# Common interaction potentials

def softened_coulomb(kappa=1.0, a=0.1):
    """Create a softened Coulomb interaction.
    
    V(x1, x2) = kappa / sqrt((x1 - x2)^2 + a^2)
    
    Parameters
    ----------
    kappa : float
        Interaction strength.
    a : float
        Softening parameter to avoid singularity at x1 = x2.
        
    Returns
    -------
    interaction : callable
        Function V(x1, x2).
    """
    def interaction(x1, x2):
        return kappa / np.sqrt((x1 - x2) ** 2 + a ** 2)
    return interaction


def contact_interaction(g=1.0, sigma=0.1):
    """Create a contact (delta-like) interaction.
    
    V(x1, x2) = g * exp(-(x1 - x2)^2 / (2 * sigma^2)) / (sqrt(2*pi) * sigma)
    
    Parameters
    ----------
    g : float
        Interaction strength.
    sigma : float
        Width of the Gaussian approximation to delta function.
        
    Returns
    -------
    interaction : callable
        Function V(x1, x2).
    """
    def interaction(x1, x2):
        norm = 1.0 / (np.sqrt(2 * np.pi) * sigma)
        return g * norm * np.exp(-0.5 * ((x1 - x2) / sigma) ** 2)
    return interaction


def harmonic_potential(omega=1.0, center=0.0):
    """Create a harmonic oscillator potential.
    
    V(x) = 0.5 * omega^2 * (x - center)^2
    
    Parameters
    ----------
    omega : float
        Oscillator frequency.
    center : float
        Center of the potential.
        
    Returns
    -------
    potential : callable
        Function V(x).
    """
    def potential(x):
        return 0.5 * omega ** 2 * (x - center) ** 2
    return potential


def double_well(a=1.0, b=1.0, center=0.0):
    """Create a double-well potential.
    
    V(x) = a * (x - center)^4 - b * (x - center)^2
    
    Parameters
    ----------
    a : float
        Quartic coefficient.
    b : float
        Quadratic coefficient.
    center : float
        Center of the potential.
        
    Returns
    -------
    potential : callable
        Function V(x).
    """
    def potential(x):
        y = x - center
        return a * y ** 4 - b * y ** 2
    return potential


def gaussian_barrier(height=1.0, width=1.0, center=0.0):
    """Create a Gaussian barrier/well potential.
    
    V(x) = height * exp(-(x - center)^2 / (2 * width^2))
    
    Parameters
    ----------
    height : float
        Barrier height (positive) or well depth (negative).
    width : float
        Width of the Gaussian.
    center : float
        Center position.
        
    Returns
    -------
    potential : callable
        Function V(x).
    """
    def potential(x):
        return height * np.exp(-0.5 * ((x - center) / width) ** 2)
    return potential


def time_dependent_displacement(omega_drive, amplitude, base_potential):
    """Create a time-dependent displaced potential (e.g., shaking trap).
    
    V(x, t) = base_potential(x - amplitude * sin(omega_drive * t))
    
    Parameters
    ----------
    omega_drive : float
        Driving frequency.
    amplitude : float
        Displacement amplitude.
    base_potential : callable
        The static potential V(x).
        
    Returns
    -------
    potential : callable
        Function V(x, t).
    """
    def potential(x, t):
        x_shifted = x - amplitude * np.sin(omega_drive * t)
        return base_potential(x_shifted)
    return potential


def time_dependent_ramp(ramp_rate, base_potential):
    """Create a linearly ramped potential.
    
    V(x, t) = base_potential(x) * (1 + ramp_rate * t)
    
    Parameters
    ----------
    ramp_rate : float
        Rate of ramping.
    base_potential : callable
        The static potential V(x).
        
    Returns
    -------
    potential : callable
        Function V(x, t).
    """
    def potential(x, t):
        return base_potential(x) * (1 + ramp_rate * t)
    return potential
