"""CMF integrators for ML-MCTDHF (three layers, species as nodes).

Two integrators, both built from the same generic single-block kernels the
standard integrators use (:mod:`mctdhf.integrators`): the layers differ only
in which linear RHS is handed to the kernel.

    get_ml_cmf_integrator           Lie–Trotter split, A · B · b
    get_ml_cmf_midpoint_integrator  second-order, A/2 · B/2 · b · B/2 · A/2

Not implemented for ML (follow-ups, explicitly out of scope): adaptive
stepping, PSI, RK4/VMF on the full coupled RHS, the species-batched variant
and ``ITE_opt``.

What is frozen where, per CMF step:

* the node densities ``ρ^s``, ``ρ^{sk}`` — frozen for the WHOLE step, taken
  from the already-propagated top tensor;
* the SPF projector ``B_f`` and the partner SPFs (through ``e^k``) — re-frozen
  at the start of each SPF substep, the analogue of ``_solve_bs_subcycled``'s
  treatment of the orbital projector and partner orbitals;
* the orbital projector and partner orbitals — re-frozen per orbital substep,
  by the base ``_solve_bs_subcycled`` used verbatim.

A species with ``n_s == D_s`` has a complete SPF basis, so ``(1 − P_f) ≡ 0``
and ``Ḃ_s = 0`` exactly: its B solve is SKIPPED, not merely projected away.
That is what makes ML reduce to standard MCTDHF bit-for-bit in structure.

No orbital or SPF PREDICTOR is fed back through ``(1 − P)`` anywhere. The
half-step integrator's instability on stiff orbital equations comes from
exactly that feedback (observed for the three-electron four-well system), and the midpoint
layout below avoids it on both upper layers.
"""

import jax
import jax.numpy as jnp

from .integrators import (
    _B_KWARG_KEYS,
    _C_KWARG_KEYS,
    _CN_MIXED_DEFAULT_LO_DTYPE,
    _CN_MIXED_DEFAULT_TOL,
    _bs_linear,
    _cn_defect_linear,
    _cn_linear,
    _diag_solve_C,
    _dop_linear,
    _expm_dense_linear,
    _lanczos_expm,
    _lower,
    _solve_bs_subcycled,
    _cmf_options,
    _validate_kwargs,
)

_A_PROPAGATORS = ("cn", "sil", "diagonalize")
_B_PROPAGATORS = ("cn", "bs", "sil", "diagonalize", "dop", "cn_mixed")

# The SPF layer takes the same option names as the orbital layer, with
# `n_orbital_substeps` renamed to `n_spf_substeps`.
_SPF_KWARG_KEYS = {
    prop: tuple(
        "n_spf_substeps" if k == "n_orbital_substeps" else k for k in keys
    )
    for prop, keys in _B_KWARG_KEYS.items()
}


def _ml_options(a_propagator, B_propagator, b_propagator,
                a_kwargs, B_kwargs, b_kwargs):
    """Validate and default the sub-solver options of the three layers.

    The orbital layer is validated by the standard ``_cmf_options`` (its
    ``c_propagator`` slot is unused here and is passed a valid placeholder),
    so the ML integrators can never drift from the orbital solver's option
    set.
    """
    if a_propagator not in _A_PROPAGATORS:
        raise ValueError(
            f"Unknown a_propagator '{a_propagator}'. "
            f"Use one of {_A_PROPAGATORS}."
        )
    if B_propagator not in _B_PROPAGATORS:
        raise ValueError(
            f"Unknown B_propagator '{B_propagator}'. "
            f"Use one of {_B_PROPAGATORS}."
        )

    a_kwargs = dict(a_kwargs or {})
    B_kwargs = dict(B_kwargs or {})
    _validate_kwargs(a_kwargs, _C_KWARG_KEYS[a_propagator], "a_kwargs",
                     a_propagator)
    _validate_kwargs(B_kwargs, _SPF_KWARG_KEYS[B_propagator], "B_kwargs",
                     B_propagator)

    a_opts = {
        "solver_tol": a_kwargs.get("solver_tol", 1e-8),
        "max_solver_iter": a_kwargs.get("max_solver_iter", 100),
        "krylov_dim": a_kwargs.get("krylov_dim", 10),
    }
    B_opts = {
        "solver_tol": B_kwargs.get("solver_tol", 1e-8),
        "max_solver_iter": B_kwargs.get("max_solver_iter", 100),
        "krylov_dim": B_kwargs.get("krylov_dim", 20),
        "n_substeps": B_kwargs.get("n_substeps", 1),
        "bs_sequence": B_kwargs.get("bs_sequence", (2, 4, 6, 8, 12, 16)),
        "orbital_x64": B_kwargs.get("orbital_x64", False),
        "n_spf_substeps": int(B_kwargs.get("n_spf_substeps", 1)),
    }
    if B_propagator == "cn_mixed":
        B_opts["solver_tol"] = B_kwargs.get("solver_tol",
                                            _CN_MIXED_DEFAULT_TOL)
        B_opts["lo_dtype"] = B_kwargs.get("lo_dtype",
                                          _CN_MIXED_DEFAULT_LO_DTYPE)

    # Reuse the orbital layer's validation and defaults verbatim.
    _, b_opts = _cmf_options("cn", b_propagator, None, b_kwargs)
    return a_opts, B_opts, b_opts


# ── Layer solves ──────────────────────────────────────────────────────────


def _solve_A(ml, a_propagator, A, top_ops, dt, a_shape, opts):
    """One linearised top-layer substep with the chosen propagator.

    ``top_ops`` is the ``(ops_diag, ops_pair)`` pair in SPF space; the
    generator is Hermitian, so SIL applies without reservation. The top block
    is ``num_states · Π n_s``, which is small enough that 'diagonalize' is
    also cheap — and in imaginary time it is the improved-relaxation update
    (replace A by the lowest ``num_states`` eigenvectors of ``H_top``).
    """
    def apply_H(X):
        return ml.dCI.apply_ci_ops(top_ops, X)

    if a_propagator == "sil":
        A_new, _ = _lanczos_expm(
            lambda v: apply_H(v.reshape(a_shape)).reshape(-1),
            A.reshape(-1), dt, ml.imag_time, opts["krylov_dim"],
        )
        return A_new.reshape(a_shape)
    if a_propagator == "diagonalize":
        return _diag_solve_C(apply_H, A, dt, ml.imag_time, a_shape)[0]
    return _cn_linear(
        lambda X: ml._rhs_A_linear(X, top_ops), A, dt,
        opts["solver_tol"], opts["max_solver_iter"],
    )


def _solve_B_block(ml, B_propagator, rhs_fn, B_s, dt, opts, rhs_lo=None,
                   project=None):
    """One linearised SPF substep for one species.

    ``rhs_fn`` carries the solver's sign convention (it is
    ``_rhs_B_linear``), exactly as ``_rhs_b_linear`` does for the orbital
    layer.

    'diagonalize' is the default: the linearised generator is
    ``D_s · n_s``-dimensional — a few hundred at the production budgets — so
    the dense ``expm`` is cheap, exact within the substep, and makes no
    Hermiticity assumption. 'sil' is allowed but comes with the same caveat
    as for orbitals: the generator ``(1 − P)(Op + W ρ⁻¹)`` is NOT Hermitian,
    which is the assumption Lanczos is built on.

    'diagonalize' does cost ``D_s·n_s`` RHS evaluations, vmapped, to build the
    dense generator — at the largest production budgets that batch is the one
    place the SPF layer allocates something sizeable, and 'dop' or 'cn' are
    the cheaper fallbacks there.

    'cn_mixed' is the defect-form CN of ``_cn_defect_linear``: the residual
    through ``rhs_fn`` at the state precision, the solve through ``rhs_lo``,
    the same RHS on the frozen ``(Op, ρ⁻¹, W, B_f)`` lowered to
    ``opts["lo_dtype"]``, and the increment re-projected by ``project``
    (the frozen ``1 - P_f`` at the state precision; see
    ``_cn_defect_linear``). No trailing orthonormalisation.
    """
    if B_propagator == "cn_mixed":
        return _cn_defect_linear(rhs_fn, rhs_lo, B_s, dt, opts["solver_tol"],
                                 opts["max_solver_iter"], opts["lo_dtype"],
                                 project=project)
    if B_propagator == "sil":
        def apply_A(v_flat):
            rhs = rhs_fn(v_flat.reshape(B_s.shape))
            return ((-rhs) if ml.imag_time else (1j * rhs)).reshape(-1)

        B_new, _ = _lanczos_expm(
            apply_A, B_s.reshape(-1), dt, ml.imag_time, opts["krylov_dim"]
        )
        return B_new.reshape(B_s.shape)
    if B_propagator == "diagonalize":
        return _expm_dense_linear(rhs_fn, B_s, dt, ml.imag_time,
                                  compute_x64=opts["orbital_x64"])
    if B_propagator == "dop":
        return _dop_linear(rhs_fn, B_s, dt, opts["n_substeps"])
    if B_propagator == "bs":
        return _bs_linear(rhs_fn, B_s, dt, opts["bs_sequence"])
    return _cn_linear(rhs_fn, B_s, dt, opts["solver_tol"],
                      opts["max_solver_iter"])


def _solve_Bs_subcycled(ml, B_propagator, Bs_start, A, g_MO_dict, Ops,
                        rho_invs, rho_pairs, dt, opts):
    """SPF step over `dt` in `n_spf_substeps` substeps.

    The node densities stay frozen over the whole CMF step; the SPF projector
    ``B_f`` and the partner SPFs (through ``e^k``) are re-frozen at the start
    of every substep — the same structure as ``_solve_bs_subcycled`` uses for
    the orbital projector and partner orbitals. ``n = 1`` is the plain
    frozen-projector substep.

    Species with a complete SPF basis are copied through untouched.
    """
    n_sub = opts["n_spf_substeps"]
    Bs = list(Bs_start)
    w = ml.state_weights

    for _ in range(n_sub):
        B_frozen = list(Bs)
        es = [
            ml.dCI.spf_transition_ops(s, B_frozen[s])
            for s in range(ml.num_species)
        ]
        Bs_next = list(Bs)
        for s in range(ml.num_species):
            if ml.spf_is_full[s]:
                continue  # (1 - P_f) = 0 exactly; B_s is stationary
            W_dict = ml.dCI.ml_meanfield_W(
                A, B_frozen, g_MO_dict, s, es=es, weights=w,
                rho_pairs=rho_pairs,
            )

            def rhs_B(X, _s=s, _W=W_dict, _Bf=B_frozen[s]):
                return ml._rhs_B_linear(
                    X, _s, Ops[_s], rho_invs[_s], _W, _Bf
                )

            if B_propagator == "cn_mixed":
                frozen_lo = _lower(
                    (Ops[s], rho_invs[s], W_dict, B_frozen[s]))

                def rhs_B_lo(X, _s=s, _f=frozen_lo):
                    return ml._rhs_B_linear(X, _s, *_f)

                def project_B(X, _Bf=B_frozen[s]):
                    return X - _Bf @ (_Bf.conj().T @ X)

                Bs_next[s] = _solve_B_block(
                    ml, B_propagator, rhs_B, Bs[s], dt / n_sub, opts,
                    rhs_lo=rhs_B_lo, project=project_B,
                )
                continue
            Bs_next[s] = _solve_B_block(
                ml, B_propagator, rhs_B, Bs[s], dt / n_sub, opts
            )
        Bs = Bs_next
    return Bs


def _lowdin(X):
    """Symmetric (Löwdin) orthonormalisation ``X (X^dagger X)^(-1/2)``.

    Of all matrices with orthonormal columns spanning ``range(X)``, the one
    closest to ``X``: it removes only the Hermitian part of ``X^dagger X - 1``
    and does not rotate the columns among themselves.
    """
    ev, U = jnp.linalg.eigh(X.conj().T @ X)
    return X @ ((U * (ev ** -0.5)[None, :]) @ U.conj().T)


def _renormalise(ml, Bs, bs):
    """Löwdin-renormalise every orbital matrix and every non-full SPF matrix.

    The frozen-projector step of the SPF and orbital layers returns
    ``X + Δ`` with ``Δ ⟂ X``, while the exact flow is
    ``X + Δ − ½ X Δ^dagger Δ + O(dt³)``: the dropped term is a Hermitian
    in-span rescaling, O(dt²) per step, so the norm carries an O(dt) GLOBAL
    defect that the exact regauge (``make_regauge_ml``) moves into ``A`` but
    cannot remove. The symmetric orthonormalisation restores exactly that
    term. It is NOT ``ortonorm_b``/``ortonorm_B``: a QR orthonormalisation
    would also rotate within the span without compensating the layer above.

    Measured on the four-well mean-field layout of the three-electron
    simulations: without it the norm drifts 2.5–5 % over a 4 t.u. ramp at dt 1e-2
    and x64 CN collapses at dt ≥ 5e-3; with it the norm holds to 1e-12 and
    the state agrees with the unrenormalised one to the discretisation error.
    Species with a complete SPF basis keep their (stationary) ``B``.
    """
    Bs = [B if ml.spf_is_full[s] else _lowdin(B) for s, B in enumerate(Bs)]
    bs = [_lowdin(b) for b in bs]
    return Bs, bs


# ── The integrators ───────────────────────────────────────────────────────


def get_ml_cmf_integrator(
    ml,
    dt,
    a_propagator="sil",
    B_propagator="diagonalize",
    b_propagator="cn",
    a_kwargs=None,
    B_kwargs=None,
    b_kwargs=None,
    renorm=False,
):
    """Basic (Lie–Trotter) CMF integrator for ML-MCTDHF.

    One step, mirroring :func:`~mctdhf.integrators.get_cmf_integrator` with
    the SPF layer inserted between the two it already had:

    1. transform the integrals at ``b(t)``, with the one-body operator at
       ``t + dt/2``; build ``Op_s``, ``e^s`` and the top-layer operators;
    2. top layer over ``dt`` with the orbitals and SPFs frozen;
    3. node densities from the NEW top tensor;
    4. SPF layer over ``dt``, sub-cycled, projector and partner SPFs
       re-frozen per substep, species with ``n_s = D_s`` skipped;
    5. full 1- and 2-RDMs from the new top tensor and SPFs;
    6. orbital layer over ``dt`` — the base ``_solve_bs_subcycled``, unchanged.

    Parameters
    ----------
    ml : MLMCTDHF_n_species or MLMCTDHF_n_species_DVR
    dt : float
    a_propagator : {'sil', 'cn', 'diagonalize'}
        Top layer. Default 'sil'. 'diagonalize' is cheap here too
        (``Π n_s`` is small) and is the improved-relaxation update in ITE.
    B_propagator : {'diagonalize', 'cn', 'cn_mixed', 'sil', 'dop', 'bs'}
        SPF layer. Default 'diagonalize' — the generator is
        ``D_s·n_s``-dimensional, non-Hermitian, and the dense ``expm`` is
        exact within the substep. 'sil' assumes a Hermitian generator, which
        this is not.
    b_propagator : {'cn', 'cn_mixed', 'sil', 'diagonalize', 'dop', 'bs'}
        Orbital layer. Default 'cn', as in the standard integrator.
        'cn_mixed' (either layer) is defect-form CN with the solve at
        ``lo_dtype`` (default complex64; ``solver_tol`` default 1e-6).
    a_kwargs, B_kwargs, b_kwargs : dict, optional
        Solver options. ``a_kwargs`` takes the standard CI keys
        (``sil``: ``krylov_dim``; ``cn``: ``solver_tol``,
        ``max_solver_iter``; ``diagonalize``: none). ``B_kwargs`` takes the
        orbital keys with ``n_orbital_substeps`` renamed ``n_spf_substeps``.
        ``b_kwargs`` is exactly the orbital solver's, ``n_orbital_substeps``
        included.
    renorm : bool
        Löwdin-renormalise the orbitals and (non-full) SPFs at the end of
        every step (:func:`_renormalise`). Off by default; with it off the
        step is unchanged.

    Returns
    -------
    integrate : callable
        ``(t, y) -> (t_new, y_new)``, jitted, static shapes only.
    """
    a_opts, B_opts, b_opts = _ml_options(
        a_propagator, B_propagator, b_propagator, a_kwargs, B_kwargs, b_kwargs
    )
    a_shape = ml.a_shape
    b_shapes = ml.b_shapes

    @jax.jit
    def integrate(t, y):
        A, Bs, bs = ml._y_to_ABb(y)

        # 1. Integrals at b(t), one-body operator at the midpoint time
        h_current = ml._get_h_current(t + dt / 2)
        h_MO_list, g_MO_dict = ml._ml_integrals(h_current, bs)
        Ops = ml.dCI.ml_species_ops(h_MO_list, g_MO_dict)
        es = [ml.dCI.spf_transition_ops(s, Bs[s])
              for s in range(ml.num_species)]
        top_ops = ml.dCI.ml_top_ops(h_MO_list, g_MO_dict, Bs, es=es)

        # 2. Top layer (SPFs and orbitals frozen)
        A_new = _solve_A(ml, a_propagator, A, top_ops, dt, a_shape, a_opts)

        # 3. Node densities from the new top tensor, frozen from here on
        rho_invs, rho_pairs = ml._compute_node_densities(A_new)

        # 4. SPF layer (orbitals frozen; projector and partner SPFs
        #    re-frozen per substep)
        Bs_new = _solve_Bs_subcycled(
            ml, B_propagator, Bs, A_new, g_MO_dict, Ops, rho_invs, rho_pairs,
            dt, B_opts,
        )

        # 5. Full RDMs of the new (A, B)
        es_new = [ml.dCI.spf_transition_ops(s, Bs_new[s])
                  for s in range(ml.num_species)]
        D_invs, ds = ml._compute_full_RDMs(A_new, Bs_new, es_new)

        # 6. Orbital layer — the standard solver, reused unchanged
        bs_new = _solve_bs_subcycled(
            ml, b_propagator, bs, h_current, D_invs, ds, dt, b_shapes, b_opts
        )

        if renorm:
            Bs_new, bs_new = _renormalise(ml, Bs_new, bs_new)
        return t + dt, ml._ABb_to_y(A_new, Bs_new, bs_new)

    return integrate


def get_ml_cmf_midpoint_integrator(
    ml,
    dt,
    a_propagator="sil",
    B_propagator="diagonalize",
    b_propagator="cn",
    a_kwargs=None,
    B_kwargs=None,
    b_kwargs=None,
    renorm=False,
):
    """Second-order (midpoint) CMF integrator for ML-MCTDHF.

    The Beck–Meyer C-b-C layout generalised to three layers,
    ``A/2 · B/2 · b · B/2 · A/2``:

    1. top half-step with the operators at ``(B(t), b(t), h(t))``;
    2. SPF half-step: node densities from ``A(t+dt/2)``, ``Op_s`` and ``g4``
       at ``b(t)``, projector and partner SPFs at ``B(t)``;
    3. orbital full step: RDMs from ``(A(t+dt/2), B(t+dt/2))``, one-body
       operator at ``t + dt/2``, projector and partner orbitals at ``b(t)``,
       sub-cycled;
    4. SPF half-step: the SAME node densities, ``Op_s`` and ``g4`` now at
       ``b(t+dt)``, projector and partner SPFs at ``B(t+dt/2)``;
    5. top half-step with the operators at ``(B(t+dt), b(t+dt), h(t+dt))``.

    Steps 4 and 5 share one integral transform at ``(h(t+dt), b(t+dt))``,
    mirroring steps 1 and 2, which share the transform at ``(h(t), b(t))``.

    Nothing predicted is fed back through a projector: step 2 freezes on the
    input SPFs and step 4 on the propagated ones, never on an extrapolation.

    Cost: two top half-steps, two SPF half-steps and one orbital step.

    Args: as :func:`get_ml_cmf_integrator` (``renorm`` included; it acts after
    the final top half-step, which does not read ``B`` or ``b`` again).
    """
    a_opts, B_opts, b_opts = _ml_options(
        a_propagator, B_propagator, b_propagator, a_kwargs, B_kwargs, b_kwargs
    )
    a_shape = ml.a_shape
    b_shapes = ml.b_shapes
    dt_half = dt / 2

    @jax.jit
    def integrate(t, y):
        A, Bs, bs = ml._y_to_ABb(y)

        # 1. Top half-step with the operators at (B(t), b(t), h(t))
        h_current_t = ml._get_h_current(t)
        h_MO_t, g_MO_t = ml._ml_integrals(h_current_t, bs)
        Ops_t = ml.dCI.ml_species_ops(h_MO_t, g_MO_t)
        es_t = [ml.dCI.spf_transition_ops(s, Bs[s])
                for s in range(ml.num_species)]
        top_ops_t = ml.dCI.ml_top_ops(h_MO_t, g_MO_t, Bs, es=es_t)
        A_half = _solve_A(ml, a_propagator, A, top_ops_t, dt_half, a_shape,
                          a_opts)

        # Node densities from the midpoint top tensor; frozen for the rest
        # of the step, used by BOTH SPF half-steps.
        rho_invs, rho_pairs = ml._compute_node_densities(A_half)

        # 2. SPF half-step: Op/g4 at b(t), projector and partners at B(t)
        Bs_half = _solve_Bs_subcycled(
            ml, B_propagator, Bs, A_half, g_MO_t, Ops_t, rho_invs, rho_pairs,
            dt_half, B_opts,
        )

        # 3. Orbital full step with the midpoint RDMs, projector at b(t)
        es_half = [ml.dCI.spf_transition_ops(s, Bs_half[s])
                   for s in range(ml.num_species)]
        D_invs, ds = ml._compute_full_RDMs(A_half, Bs_half, es_half)
        h_current_half = ml._get_h_current(t + dt_half)
        bs_new = _solve_bs_subcycled(
            ml, b_propagator, bs, h_current_half, D_invs, ds, dt, b_shapes,
            b_opts,
        )

        # Integrals at the new orbitals and the end time, shared by 4 and 5
        h_current_new = ml._get_h_current(t + dt)
        h_MO_new, g_MO_new = ml._ml_integrals(h_current_new, bs_new)
        Ops_new = ml.dCI.ml_species_ops(h_MO_new, g_MO_new)

        # 4. SPF half-step: Op/g4 at b(t+dt), projector and partners at
        #    B(t+dt/2)
        Bs_new = _solve_Bs_subcycled(
            ml, B_propagator, Bs_half, A_half, g_MO_new, Ops_new, rho_invs,
            rho_pairs, dt_half, B_opts,
        )

        # 5. Top half-step with the operators at (B(t+dt), b(t+dt), h(t+dt))
        es_new = [ml.dCI.spf_transition_ops(s, Bs_new[s])
                  for s in range(ml.num_species)]
        top_ops_new = ml.dCI.ml_top_ops(h_MO_new, g_MO_new, Bs_new, es=es_new)
        A_new = _solve_A(ml, a_propagator, A_half, top_ops_new, dt_half,
                         a_shape, a_opts)

        if renorm:
            Bs_new, bs_new = _renormalise(ml, Bs_new, bs_new)
        return t + dt, ml._ABb_to_y(A_new, Bs_new, bs_new)

    return integrate
