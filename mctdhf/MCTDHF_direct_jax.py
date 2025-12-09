import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from jax import jit
from functools import partial
from scipy.special import comb
from opt_einsum import contract

from .SemiDirectCI_jax import SemiDirectCI


class MCTDHF:
    """Class for running spin-unrestricted MCTDHF calculations.

    Uses separate orbital transformation coefficients (b_alpha, b_beta) for each
    spin, following the formalism in Hochstuhl et al. EPJ ST 2014.

    Attributes
    ----------
    h_alpha : ndarray (num_spatial_orbitals,)*2
        Time-independent single-particle integrals for alpha spin.
    h_beta : ndarray (num_spatial_orbitals,)*2
        Time-independent single-particle integrals for beta spin.
    ht_alpha : function of t, returning ndarray (num_spatial_orbitals,)*2
        Time-dependent single-particle integrals for alpha spin.
    ht_beta : function of t, returning ndarray (num_spatial_orbitals,)*2
        Time-dependent single-particle integrals for beta spin.
    g : ndarray (num_spatial_orbitals,)*4
        Two-particle electron integrals.
    num_mctdhf_orbitals : int
        The number of MCTDHF orbitals per spin.
    num_spatial_orbitals: int
        The number of spatial orbitals (basis set size).
    num_alpha_electrons : int
        The number of electrons with alpha spin.
    num_beta_electrons : int
        The number of electrons with beta spin.
    imag_time : bool
        Whether to perform imaginary time evolution.
    S: ndarray (num_spatial_orbitals,)*2
        Overlap matrix for the basis functions.

    Notes
    -----
    For backward compatibility, `h` and `ht` can be provided as:
    - A single array/function: used for both spins (restricted integrals)
    - A tuple (h_alpha, h_beta): separate for each spin (unrestricted integrals)
    
    The state vector is [C, b_alpha, b_beta] where b_alpha and b_beta are the
    orbital transformation coefficients for each spin.
    """

    def __init__(
        self,
        h,
        ht,
        g,
        num_mctdhf_orbitals,
        num_spatial_orbitals,
        num_alpha_electrons,
        num_beta_electrons,
        num_states=1,
        imag_time=False,
        S=None,
    ):
        # Handle both restricted (single) and unrestricted (tuple) input
        # for single-particle integrals
        if isinstance(h, tuple):
            self.h_alpha, self.h_beta = h
        else:
            self.h_alpha = h
            self.h_beta = h

        if isinstance(ht, tuple):
            self.ht_alpha, self.ht_beta = ht
        else:
            self.ht_alpha = ht
            self.ht_beta = ht

        self.g = g
        self.num_mctdhf_orbitals = num_mctdhf_orbitals
        self.num_spatial_orbitals = num_spatial_orbitals
        self.num_alpha_electrons = num_alpha_electrons
        self.num_beta_electrons = num_beta_electrons
        self.num_states = num_states

        self.num_alpha_dets = int(comb(num_mctdhf_orbitals, num_alpha_electrons))
        self.num_beta_dets = int(comb(num_mctdhf_orbitals, num_beta_electrons))

        self.num_slater_dets = self.num_alpha_dets * self.num_beta_dets

        self.dCI = SemiDirectCI(
            self.num_mctdhf_orbitals, num_alpha_electrons, num_beta_electrons
        )

        self.imag_time = imag_time
        self.time = 1 if imag_time else 1j

        if S is None:
            self.S = np.eye(num_spatial_orbitals)
            self.S_inv = np.eye(num_spatial_orbitals)
        else:
            self.S = S
            self.S_inv = np.linalg.inv(S)

    @partial(jit, static_argnums=0)
    def transform_h(self, h, b, bc):
        """Transform the single particle electron integrals to he MCTDHF basis."""

        h_1 = contract("jm, ij -> im", b, h, backend="jax")
        h_2 = contract("in, im -> nm", bc, h_1, backend="jax")
        h_3 = contract("in, nm -> im", b, h_2, backend="jax")

        return h_1, h_2, h_3

    @partial(jit, static_argnums=0)
    def transform_g(self, g, b, bc):
        """Transform two-particle integrals using the same b for all indices."""
        g_2 = contract("jq, ls, ijkl -> iqks", bc, b, g, backend="jax")
        g_3 = contract("kr, iqks -> iqrs", b, g_2, backend="jax")
        g_4 = contract("ip, iqrs -> pqrs", bc, g_3, backend="jax")
        g_5 = contract("ip, pqrs -> iqrs", b, g_4, backend="jax")
        return g_3, g_4, g_5

    @partial(jit, static_argnums=0)
    def transform_g_spin(self, g, b_sigma, b_tau, bc_sigma, bc_tau):
        """Transform two-particle integrals with spin indices: g^{σ,τ}.
        
        Uses physicist notation: g_{ijkl} = <ij|kl> where
        - Particle 1: indices i (bra) and k (ket) -> use b_sigma
        - Particle 2: indices j (bra) and l (ket) -> use b_tau
        
        Args:
            g: Two-particle integrals in original basis (ijkl)
            b_sigma: Orbital coefficients for particle 1 (spin σ)
            b_tau: Orbital coefficients for particle 2 (spin τ)
            bc_sigma: Conjugate of b_sigma
            bc_tau: Conjugate of b_tau
            
        Returns:
            g_3, g_4, g_5: Partially and fully transformed integrals
        """
        # Transform j (tau) and l (tau): g_{ijkl} -> g_{iqks}
        g_2 = contract("jq, ls, ijkl -> iqks", bc_tau, b_tau, g, backend="jax")
        
        # Transform k (sigma): g_{iqks} -> g_{iqrs}
        g_3 = contract("kr, iqks -> iqrs", b_sigma, g_2, backend="jax")
        
        # Transform i (sigma): g_{iqrs} -> g_{pqrs}
        g_4 = contract("ip, iqrs -> pqrs", bc_sigma, g_3, backend="jax")
        
        # Back-transform for orbital evolution (using sigma)
        g_5 = contract("ip, pqrs -> iqrs", b_sigma, g_4, backend="jax")
        
        return g_3, g_4, g_5

    @partial(jit, static_argnums=0)
    def get_E(self, t, b_alpha, b_beta, C):
        """Calculate the energy with spin-specific orbital coefficients."""
        bc_alpha = b_alpha.conj()
        bc_beta = b_beta.conj()

        h_alpha = self.h_alpha + self.ht_alpha(t)
        h_beta = self.h_beta + self.ht_beta(t)

        # Transform h with spin-specific b
        _, h_2_alpha, _ = self.transform_h(h_alpha, b_alpha, bc_alpha)
        _, h_2_beta, _ = self.transform_h(h_beta, b_beta, bc_beta)

        # Transform g with spin combinations
        _, g_aa_4, _ = self.transform_g_spin(self.g, b_alpha, b_alpha, bc_alpha, bc_alpha)
        _, g_ab_4, _ = self.transform_g_spin(self.g, b_alpha, b_beta, bc_alpha, bc_beta)
        _, g_bb_4, _ = self.transform_g_spin(self.g, b_beta, b_beta, bc_beta, bc_beta)

        return self.dCI.calculate_energy(C, h_2_alpha, h_2_beta, g_aa_4, g_ab_4, g_bb_4)

    @partial(jit, static_argnums=0)
    def _Cb_to_y(self, C, b_alpha, b_beta):
        """Pack state vector: [C, b_alpha, b_beta]"""
        return jnp.concatenate([C.flatten(), b_alpha.flatten(), b_beta.flatten()])

    @partial(jit, static_argnums=0)
    def _y_to_Cb(self, y):
        """Unpack state vector into C, b_alpha, b_beta."""
        b_size = self.num_spatial_orbitals * self.num_mctdhf_orbitals
        C_size = self.num_slater_dets * self.num_states
        
        C = y[:C_size]
        b_alpha = y[C_size:C_size + b_size]
        b_beta = y[C_size + b_size:]
        
        C = jnp.reshape(C, (self.num_states, self.num_alpha_dets, self.num_beta_dets))
        b_alpha = jnp.reshape(b_alpha, (self.num_spatial_orbitals, self.num_mctdhf_orbitals))
        b_beta = jnp.reshape(b_beta, (self.num_spatial_orbitals, self.num_mctdhf_orbitals))

        return C, b_alpha, b_beta

    @partial(jit, static_argnums=0)
    def ortonorm_b(self, b):
        """Function to ortonormalise the transformation coefficents b

        The function uses the overlap algorithm from Lehtovaara et al.,
        DOI: 10.1016/j.jcp.2006.06.006
        """

        #b = jnp.divide(b, jnp.sqrt(contract("ij, ij -> j", b.conj(), b, backend="jax")))
        ## Overlap renormalisation
        # u, _, vh = jnp.linalg.svd(b, full_matrices=False)
        # return u@vh

        ## Gram-Schmidt renormalisation
        for i in range(b.shape[1]):
            orto_adjustment = 0
            for j in range(0, i):
                orto_adjustment += b[:, j] * contract(
                    "i,j,ij", b[:, i], b[:, j], self.S, backend="jax"
                )

            b = b.at[:, i].add(-orto_adjustment)
            b = b.at[:, i].divide(jnp.sqrt(contract("i,i", b[:,i].conj(), b[:,i], backend="jax")))

        return b

    @partial(jit, static_argnums=0)
    def ortonorm_C(self, C):
        """Function to ortonormalise C

        The function uses the overlap algorithm from Lehtovaara et al.,
        DOI: 10.1016/j.jcp.2006.06.006
        """

        # C = jnp.divide(
        #     C,
        #     jnp.sqrt(contract("cij, cij -> c", C.conj(), C, backend="jax")).reshape(
        #         self.num_states, 1, 1
        #     ),
        # )

        ## Overlap renormalisation
        # u, _, vh = jnp.linalg.svd(C.reshape(self.num_states, -1), full_matrices=False)
        # return jnp.reshape(u@vh, (self.num_states, self.num_alpha_dets, self.num_beta_dets))

        # Gram-Schmidt renormalisation
        for i in range(C.shape[0]):
            orto_adjustment = 0
            for j in range(0, i):
                orto_adjustment += contract("ij,ij", C[i], C[j]) * C[j]

            C = C.at[i].add(-orto_adjustment)
            C = C.at[i].divide(jnp.sqrt(contract("ij, ij", C[i].conj(), C[i])))

        return C

    @partial(jit, static_argnums=0)
    def __call__(self, t, y):
        """RHS of the MCTDHF problem (spin-unrestricted).
        
        Implements Eq. 3.98-3.99 from Hochstuhl et al.
        """
        C, b_alpha, b_beta = self._y_to_Cb(y)
        bc_alpha = b_alpha.conj()
        bc_beta = b_beta.conj()

        h_alpha = self.h_alpha + self.ht_alpha(t)
        h_beta = self.h_beta + self.ht_beta(t)

        # Transform h with spin-specific b
        h_1_alpha, h_2_alpha, h_3_alpha = self.transform_h(h_alpha, b_alpha, bc_alpha)
        h_1_beta, h_2_beta, h_3_beta = self.transform_h(h_beta, b_beta, bc_beta)

        # Transform g with all spin combinations (Eq. 3.103-3.106)
        g_aa_3, g_aa_4, g_aa_5 = self.transform_g_spin(self.g, b_alpha, b_alpha, bc_alpha, bc_alpha)
        g_ab_3, g_ab_4, g_ab_5 = self.transform_g_spin(self.g, b_alpha, b_beta, bc_alpha, bc_beta)
        g_ba_3, g_ba_4, g_ba_5 = self.transform_g_spin(self.g, b_beta, b_alpha, bc_beta, bc_alpha)
        g_bb_3, g_bb_4, g_bb_5 = self.transform_g_spin(self.g, b_beta, b_beta, bc_beta, bc_beta)

        # Get spin-resolved RDMs
        D_alpha, D_beta = self.dCI.get_1p_RDMs_spin(C)
        d_aa, d_ab, d_ba, d_bb = self.dCI.get_2p_RDMs_spin(C)

        # Alpha orbital evolution (Eq. 3.98)
        D_alpha_inv = jnp.linalg.pinv(D_alpha[0])
        Sh_1_alpha = contract("kl, lm -> km", self.S_inv, h_1_alpha, backend="jax")
        Sg_aa_3 = contract("kl, lqrs -> kqrs", self.S_inv, g_aa_3, backend="jax")
        Sg_ab_3 = contract("kl, lqrs -> kqrs", self.S_inv, g_ab_3, backend="jax")
        
        b_alpha_dot = -self.time * (
            Sh_1_alpha - h_3_alpha
            + contract("np, pqrs, iqrs -> in", D_alpha_inv, d_aa[0], Sg_aa_3 - g_aa_5, backend="jax")
            + contract("np, pqrs, iqrs -> in", D_alpha_inv, d_ab[0], Sg_ab_3 - g_ab_5, backend="jax")
        )

        # Beta orbital evolution (Eq. 3.99)
        D_beta_inv = jnp.linalg.pinv(D_beta[0])
        Sh_1_beta = contract("kl, lm -> km", self.S_inv, h_1_beta, backend="jax")
        Sg_ba_3 = contract("kl, lqrs -> kqrs", self.S_inv, g_ba_3, backend="jax")
        Sg_bb_3 = contract("kl, lqrs -> kqrs", self.S_inv, g_bb_3, backend="jax")
        
        b_beta_dot = -self.time * (
            Sh_1_beta - h_3_beta
            + contract("np, pqrs, iqrs -> in", D_beta_inv, d_ba[0], Sg_ba_3 - g_ba_5, backend="jax")
            + contract("np, pqrs, iqrs -> in", D_beta_inv, d_bb[0], Sg_bb_3 - g_bb_5, backend="jax")
        )

        # CI evolution
        sigma = self.dCI.get_sigma(C, h_2_alpha, h_2_beta, g_aa_4, g_ab_4, g_bb_4)
        C_dot = -self.time * sigma

        return self._Cb_to_y(C_dot, b_alpha_dot, b_beta_dot)

    def integrate(
        self,
        integrator,
        num_steps: int,
        t_init=0,
        b_alpha_init=None,
        b_beta_init=None,
        C_init=None,
        normalize=None,
    ):
        """Integrate the MCTDHF problem.

        Args:
            integrator: The integrator used to solve the problem.
            num_steps (int): Number of integration steps
            t_init (int, optional): Initial time. Defaults to 0.
            b_alpha_init (optional): Initial alpha orbital coefficients. Defaults to random.
            b_beta_init (optional): Initial beta orbital coefficients. Defaults to random.
            C_init (optional): Initial CI coefficients. Defaults to random.
            normalize (optional): Whether to normalize during integration. Defaults to imag_time.

        Returns:
            ts, Es, Cs, bs_alpha, bs_beta: Lists of values at each time step.
        """

        if b_alpha_init is None:
            # Generate random orthonormal vectors
            rng = np.random.default_rng()
            r = rng.random((self.num_spatial_orbitals, self.num_mctdhf_orbitals))
            u, _, vh = np.linalg.svd(r, full_matrices=False)
            b_alpha_init = jnp.array((u @ vh), jnp.complex64)

        if b_beta_init is None:
            # Default: use same orbitals as alpha (appropriate for closed-shell)
            # For open-shell systems, user can provide different b_beta_init
            b_beta_init = b_alpha_init

        if C_init is None:
            # Random initial C
            rng = np.random.default_rng()
            r = rng.random((self.num_states, self.num_alpha_dets, self.num_beta_dets))
            u, _, vh = np.linalg.svd(r, full_matrices=False)
            C_init = jnp.array((u @ vh), jnp.complex64)

        if normalize is None:
            normalize = self.imag_time

        y = self._Cb_to_y(C_init, b_alpha_init, b_beta_init)

        E_init = self.get_E(0, b_alpha_init, b_beta_init, C_init)
        Es = [E_init]
        Cs = [C_init]
        bs_alpha = [b_alpha_init]
        bs_beta = [b_beta_init]
        ts = [t_init]

        t = t_init
        for _ in range(num_steps):
            t, E, C, b_alpha, b_beta = self._step(t, y, integrator, normalize)

            Es.append(E)
            Cs.append(C)
            bs_alpha.append(b_alpha)
            bs_beta.append(b_beta)
            ts.append(t)

            y = self._Cb_to_y(C, b_alpha, b_beta)

        return np.array(ts), np.array(Es), np.array(Cs), np.array(bs_alpha), np.array(bs_beta)

    @partial(jit, static_argnums=(0, 3, 4))
    def _step(self, t, y, integrator, normalize):
        t, y = integrator(t, y)
        C, b_alpha, b_beta = self._y_to_Cb(y)

        if normalize:
            C = self.ortonorm_C(C)
            b_alpha = self.ortonorm_b(b_alpha)
            b_beta = self.ortonorm_b(b_beta)

        E = self.get_E(t, b_alpha, b_beta, C)

        return t, E, C, b_alpha, b_beta

    @partial(jit, static_argnums=0)
    def calculate_overlap(self, b1_alpha, b1_beta, C1, b2_alpha, b2_beta, C2):
        """Calculate the wave function overlap between two spin-unrestricted states.
        
        Args:
            b1_alpha, b1_beta: Orbital coefficients for state 1
            C1: CI coefficients for state 1
            b2_alpha, b2_beta: Orbital coefficients for state 2
            C2: CI coefficients for state 2
        """
        I_alpha = SemiDirectCI.get_slater_dets(
            self.num_mctdhf_orbitals, self.num_alpha_electrons
        )

        # Alpha overlap uses b_alpha from each state
        s_a = contract("ai,ki,kj,bj->abij", I_alpha, b1_alpha, b2_alpha, I_alpha, backend="jax")
        s_a = s_a[
            jnp.nonzero(s_a, size=(self.num_alpha_dets * self.num_alpha_electrons) ** 2)
        ]
        S_a = jnp.linalg.det(
            s_a.reshape((self.num_alpha_dets,) * 2 + (self.num_alpha_electrons,) * 2)
        )

        if self.num_beta_electrons == self.num_alpha_electrons:
            # Beta overlap uses b_beta from each state
            s_b = contract("ai,ki,kj,bj->abij", I_alpha, b1_beta, b2_beta, I_alpha, backend="jax")
            s_b = s_b[
                jnp.nonzero(s_b, size=(self.num_alpha_dets * self.num_alpha_electrons) ** 2)
            ]
            S_b = jnp.linalg.det(
                s_b.reshape((self.num_alpha_dets,) * 2 + (self.num_alpha_electrons,) * 2)
            )
            return contract("cik,cjl,ij,kl->c", C1.conj(), C2, S_a, S_b, backend="jax")

        I_beta = SemiDirectCI.get_slater_dets(
            self.num_mctdhf_orbitals, self.num_beta_electrons
        )

        # Beta overlap uses b_beta from each state
        s_b = contract("ai,ki,kj,bj->abij", I_beta, b1_beta, b2_beta, I_beta, backend="jax")
        s_b = s_b[
            jnp.nonzero(s_b, size=(self.num_beta_dets * self.num_beta_electrons) ** 2)
        ]
        S_b = jnp.linalg.det(
            s_b.reshape((self.num_beta_dets,) * 2 + (self.num_beta_electrons,) * 2)
        )

        return contract("cik,cjl,ij,kl->c", C1.conj(), C2, S_a, S_b, backend="jax")

    def calculate_particle_densities(self, b_alpha, b_beta, C, spf):
        """Calculate the total particle density (sum of alpha and beta contributions).
        
        Args:
            b_alpha: Alpha orbital transformation coefficients
            b_beta: Beta orbital transformation coefficients
            C: CI coefficients
            spf: Single particle functions (basis functions evaluated on grid)
            
        Returns:
            Total particle density (alpha + beta) on the grid
        """
        D_alpha, D_beta = self.dCI.get_1p_RDMs_spin(C)
        
        # Alpha contribution: transform D_alpha using b_alpha
        rho_alpha = contract("ix,in,snm,jm,jx->sx", spf, b_alpha, D_alpha, b_alpha, spf)
        
        # Beta contribution: transform D_beta using b_beta
        rho_beta = contract("ix,in,snm,jm,jx->sx", spf, b_beta, D_beta, b_beta, spf)
        
        return rho_alpha + rho_beta

    def plot_Cbe(self, ts, Cs, bs_alpha, bs_beta, Es):
        """Plot evolution of C, b, and E."""
        fig, axs = plt.subplots(1, 4, figsize=(20, 6))
        axs[0].plot(ts, np.real(Cs.reshape(Cs.shape[0], -1)))
        axs[0].set_title("Cs")
        axs[1].plot(
            ts,
            np.real(
                bs_alpha.reshape(-1, self.num_spatial_orbitals, self.num_mctdhf_orbitals)[
                    :, :, 0
                ]
            ),
        )
        axs[1].set_title("bs_alpha")
        axs[2].plot(
            ts,
            np.real(
                bs_beta.reshape(-1, self.num_spatial_orbitals, self.num_mctdhf_orbitals)[
                    :, :, 0
                ]
            ),
        )
        axs[2].set_title("bs_beta")
        axs[3].plot(ts, np.real(Es))
        axs[3].set_title("E")
        plt.show()

    def get_full_basis_1p_RDM(self, Cs, bs_alpha, bs_beta):
        """Get 1-particle RDM in full basis (returns alpha and beta separately)."""
        D_alpha, D_beta = self.dCI.get_1p_RDMs_spin(Cs)
        rho_alpha = contract("tin, tnm, tjm -> tij", bs_alpha, D_alpha, bs_alpha, backend="jax")
        rho_beta = contract("tin, tnm, tjm -> tij", bs_beta, D_beta, bs_beta, backend="jax")
        return rho_alpha, rho_beta

    def expectation_val(self, op, Cs, bs_alpha, bs_beta):
        """Calculate expectation value of one-body operator."""
        rho_alpha, rho_beta = self.get_full_basis_1p_RDM(Cs, bs_alpha, bs_beta)
        return (
            contract("pq, tqp -> t", op, rho_alpha)
            + contract("pq, tqp -> t", op, rho_beta)
        )


class MCTDHF_sinc(MCTDHF):
    """MCTDHF variant for sinc-DVR basis."""
    
    @partial(jit, static_argnums=0)
    def transform_g(self, g, b, bc):
        """Transform g for sinc basis (diagonal in spatial coordinates)."""
        g_2 = contract("jq, js, ij -> iqs", bc, b, g, backend="jax")
        g_3 = contract("ir, iqs -> iqrs", b, g_2, backend="jax")
        g_4 = contract("ip, iqrs -> pqrs", bc, g_3, backend="jax")
        g_5 = contract("ip, pqrs -> iqrs", b, g_4, backend="jax")
        return g_3, g_4, g_5

    @partial(jit, static_argnums=0)
    def transform_g_spin_sinc(self, g, b_sigma, b_tau, bc_sigma, bc_tau):
        """Transform g with spin indices for sinc basis (physicist notation).
        
        Particle 1 (i,k) uses sigma, particle 2 (j,l) uses tau.
        For sinc basis, g is diagonal: g_ij = g(x_i, x_j).
        """
        # Transform j (tau) and l (tau): g_{ij} -> g_{iqs}
        g_2 = contract("jq, js, ij -> iqs", bc_tau, b_tau, g, backend="jax")
        
        # Transform k (sigma) to add r index: g_{iqs} -> g_{iqrs}
        g_3 = contract("ir, iqs -> iqrs", b_sigma, g_2, backend="jax")
        
        # Transform i (sigma): g_{iqrs} -> g_{pqrs}
        g_4 = contract("ip, iqrs -> pqrs", bc_sigma, g_3, backend="jax")
        
        # Back-transform
        g_5 = contract("ip, pqrs -> iqrs", b_sigma, g_4, backend="jax")
        return g_3, g_4, g_5

    @partial(jit, static_argnums=0)
    def get_E(self, t, b_alpha, b_beta, C):
        """Calculate energy for sinc basis."""
        bc_alpha = b_alpha.conj()
        bc_beta = b_beta.conj()

        h_alpha = self.h_alpha + self.ht_alpha(t)
        h_beta = self.h_beta + self.ht_beta(t)

        _, h_2_alpha, _ = self.transform_h(h_alpha, b_alpha, bc_alpha)
        _, h_2_beta, _ = self.transform_h(h_beta, b_beta, bc_beta)

        _, g_aa_4, _ = self.transform_g_spin_sinc(self.g, b_alpha, b_alpha, bc_alpha, bc_alpha)
        _, g_ab_4, _ = self.transform_g_spin_sinc(self.g, b_alpha, b_beta, bc_alpha, bc_beta)
        _, g_bb_4, _ = self.transform_g_spin_sinc(self.g, b_beta, b_beta, bc_beta, bc_beta)

        return self.dCI.calculate_energy(C, h_2_alpha, h_2_beta, g_aa_4, g_ab_4, g_bb_4)

