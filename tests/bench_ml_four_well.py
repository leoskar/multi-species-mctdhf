"""Four-well ML-MCTDHF benchmark. NOT collected by pytest — run it directly.

    python -m tests.bench_ml_four_well --help
    python -m tests.bench_ml_four_well --quick
    python -m tests.bench_ml_four_well --norb 6 7 9 --spfs 4 6 8 --states 16

The system is a self-contained four-well model in the shape of the
three-electron system of the four-well paper: four DISTINGUISHABLE species of N = 3
particles each, one anharmonic well per species on a shared sine-DVR grid,
softened-Coulomb interaction within and between wells. It is built here
rather than imported so the benchmark runs against this package alone.

Two things are measured.

(a) **Exactness.** At ``n_s = D_s`` the ML step must reproduce the standard
    step. The 1e-10 target is a double-precision statement, so pass
    ``--x64`` to see it; in the production complex64 dtype the floor is the
    dtype's own ~1e-7.

(b) **Cost.** Seconds per step and peak memory for a ladder of orbital
    budgets ``norb`` and SPF counts ``n``, against the standard solver
    wherever the standard solver still fits, plus gate-matrix-style overlaps
    against the full-rank run at the smallest orbital budget. The ML numbers
    should be flat in ``norb`` up to the orbital step's own cost, while the
    standard ones blow up as ``C(norb, 3)^4``.

Each configuration runs in its OWN SUBPROCESS. That is what makes "peak
memory" mean anything: ``ru_maxrss`` is a per-process high-water mark that
never comes back down, so configurations sharing an interpreter would each
report the largest one seen so far. It also contains OOMs — one row dying
does not take the table with it.

Memory comes from the device allocator when JAX exposes one (GPU) and from
peak process RSS otherwise (CPU).

**A GPU trap worth knowing about.** At norb 9 with 8 SPFs the default
combination (sparse excitation tensors, ``diagonalize`` on the SPF layer)
fails on an RTX 2070 SUPER with ``RESOURCE_EXHAUSTED`` while the card is
nearly empty — 339 MB of arrays live, 7.5 GB free. It is not a shortage of
memory: the failing kernel is ``input_scatter_fusion``, and XLA's GPU
scatter codegen needs excessive per-thread local memory on large operands
(see ``_segment_sum``'s docstring). Reducing the allocator fraction,
switching allocators and disabling command buffers all leave it failing.

Either documented fallback clears it, measured on that configuration:

    --B-propagator dop     0.118 s/step,    88 MB
    --no-ci-sparse         0.174 s/step,  1486 MB

``dop`` is the better of the two here and is in fact faster than
``diagonalize`` at every SPF budget this benchmark reaches, so it is worth
considering as the production choice rather than only as a fallback.
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import time

# x64 has to be set before the first JAX array exists, so it is read out of
# argv here rather than from the parsed arguments below.
_X64 = "--x64" in sys.argv

import jax

jax.config.update("jax_enable_x64", _X64)

import jax.numpy as jnp
import numpy as np
from scipy.special import comb

from mctdhf.MCTDHF_n_species import MCTDHF_n_species_DVR, pack_Cb_to_y
from mctdhf.MLMCTDHF_n_species import MLMCTDHF_n_species_DVR
from mctdhf.integrators import get_cmf_integrator
from mctdhf.integrators_ml import get_ml_cmf_integrator
from mctdhf.sine_dvr import SineDVR, softened_coulomb

NUM_WELLS = 4
NUM_PARTICLES_PER_WELL = 3
WELL_SPACING = 6.0
GRID_HALF_WIDTH = 15.0

CDTYPE = jnp.complex128 if _X64 else jnp.complex64


# ── the model ─────────────────────────────────────────────────────────────


def four_well_ingredients(num_grid):
    """``(h_list, g_dict)`` for the four-well model.

    Each species sees a quartic anharmonic well centred on its own site; the
    wells are far enough apart that the orbitals localise, which is what
    makes the four species distinguishable.
    """
    dvr = SineDVR(x_min=-GRID_HALF_WIDTH, x_max=GRID_HALF_WIDTH,
                  num_points=num_grid)
    centres = [(i - (NUM_WELLS - 1) / 2) * WELL_SPACING
               for i in range(NUM_WELLS)]

    def well(centre):
        def V(x):
            u = x - centre
            return 0.5 * u ** 2 - 0.03 * u ** 4 + 0.002 * u ** 6
        return V

    h_list = [dvr.get_h(potential=well(c)) for c in centres]

    g = dvr.get_g(softened_coulomb(kappa=1.0, a=0.5))
    g_dict = {(s, s): g for s in range(NUM_WELLS)}
    for s in range(NUM_WELLS):
        for k in range(s + 1, NUM_WELLS):
            g_dict[(s, k)] = g
    return h_list, g_dict


def _leading_dets(norb, num_states):
    """The determinant index each state occupies in each well.

    ``idx[a][i]`` for state ``a``, well ``i``. The states are enumerated as a
    four-digit number in base ``L``, the smallest base with ``L^4 >=
    num_states``, so each well carries only ``L`` distinct determinants. That
    is the register's product-state picture, and it is what makes a small SPF
    budget the right ansatz: 16 states over 4 wells need just 2 logical
    states per well.

    Indexing the wells by bit-shifts of the state number instead would make
    the lowest well carry ``num_states`` distinct determinants, which no
    small SPF budget can represent.
    """
    n_dets = comb(norb, NUM_PARTICLES_PER_WELL, exact=True)
    base = 1
    while base ** NUM_WELLS < num_states:
        base += 1
    if base > n_dets:
        raise ValueError(
            f"norb={norb} gives only {n_dets} determinants per well, too few "
            f"to enumerate {num_states} states over {NUM_WELLS} wells")
    return [[(a // base ** i) % base for i in range(NUM_WELLS)]
            for a in range(num_states)], n_dets


def _orbitals(num_grid, norb):
    h_list, g_dict = four_well_ingredients(num_grid)
    bs = []
    for s in range(NUM_WELLS):
        _w, v = np.linalg.eigh(np.asarray(h_list[s]))
        bs.append(jnp.asarray(v[:, :norb], dtype=CDTYPE))
    return h_list, g_dict, bs


def _seed_core(norb, num_states, seed, noise):
    """The seed state, in the smallest per-well basis that carries it.

    Returns ``(columns, A_core, n_seed, n_dets)``: the determinants each well
    occupies, and the top tensor over exactly those.

    ``A_core`` is drawn ONCE at this fixed size, never at the caller's SPF
    budget. That is what makes the seed the same physical state for every
    ``n_spf`` and for the expanded standard tensor: a perturbation drawn at
    the budget's own shape would be a different random state for each row,
    and the overlap column would compare unrelated trajectories.

    The perturbation has RELATIVE norm ``noise``. A fixed per-element
    amplitude would grow with the tensor size and turn the seed into white
    noise at large orbital budgets.
    """
    idx, n_dets = _leading_dets(norb, num_states)
    columns = [sorted({row[i] for row in idx}) for i in range(NUM_WELLS)]
    n_seed = max(len(c) for c in columns)
    for i in range(NUM_WELLS):
        used = set(columns[i])
        filler = (d for d in range(n_dets) if d not in used)
        while len(columns[i]) < n_seed:
            columns[i].append(next(filler))

    rng = np.random.default_rng(seed)
    a_shape = (num_states,) + (n_seed,) * NUM_WELLS
    A = np.zeros(a_shape, dtype=np.complex128)
    for a, row in enumerate(idx):
        A[(a,) + tuple(columns[i].index(row[i])
                       for i in range(NUM_WELLS))] += 1.0

    pert = rng.standard_normal(a_shape) + 1j * rng.standard_normal(a_shape)
    per_state = np.linalg.norm(pert.reshape(num_states, -1), axis=1)
    pert = pert / per_state[(slice(None),) + (None,) * NUM_WELLS]
    A = A + noise * pert
    A = A / np.linalg.norm(A.reshape(num_states, -1), axis=1)[
        (slice(None),) + (None,) * NUM_WELLS]
    return columns, A, n_seed, n_dets


def ml_initial_state(num_grid, norb, num_states, n_spf, seed=1234,
                     noise=1e-2):
    """Seed the ML layers DIRECTLY, without ever forming the CI tensor.

    This is the whole point of the ansatz and the benchmark has to respect
    it: routing the seed through ``from_mctdhf`` would build the full
    ``num_states x D^4`` tensor — 5.9 GiB at norb 9 in complex64 — and
    exhaust the card before a single step ran, which is precisely the wall
    ML-MCTDHF exists to remove.

    Each well's SPF basis is the determinants the seed occupies, padded with
    the lowest unused ones up to ``n_spf``; the top tensor embeds
    :func:`_seed_core` with zeros on the padding. So every SPF budget and the
    expanded standard tensor all carry the SAME state, and any later
    difference between them is propagation, not initialisation.
    """
    h_list, g_dict, bs = _orbitals(num_grid, norb)
    columns, A_core, n_seed, n_dets = _seed_core(norb, num_states, seed, noise)
    if n_spf < n_seed:
        raise ValueError(
            f"num_spfs={n_spf} cannot carry this seed: {num_states} states "
            f"occupy {n_seed} determinants per well")

    Bs = []
    for i in range(NUM_WELLS):
        cols = list(columns[i])
        used = set(cols)
        filler = (d for d in range(n_dets) if d not in used)
        while len(cols) < n_spf:
            cols.append(next(filler))
        B = np.zeros((n_dets, n_spf))
        for j, d in enumerate(cols):
            B[d, j] = 1.0
        Bs.append(jnp.asarray(B, dtype=CDTYPE))

    a_shape = (num_states,) + (n_spf,) * NUM_WELLS
    A = np.zeros(a_shape, dtype=np.complex128)
    A[(slice(None),) + (slice(0, n_seed),) * NUM_WELLS] = A_core
    return h_list, g_dict, jnp.asarray(A, dtype=CDTYPE), Bs, bs


def initial_state(num_grid, norb, num_states, seed=1234, noise=1e-2):
    """The same seed, expanded into a standard CI tensor.

    Only for the standard-solver rows and the exactness check, where the
    ``D^4`` tensor is affordable by construction.
    """
    n_dets = comb(norb, NUM_PARTICLES_PER_WELL, exact=True)
    h_list, g_dict, A, Bs, bs = ml_initial_state(
        num_grid, norb, num_states, n_dets, seed=seed, noise=noise)
    C = A
    for s, B in enumerate(Bs):
        C = jnp.tensordot(C, B, axes=([s + 1], [1]))
        C = jnp.moveaxis(C, -1, s + 1)
    return h_list, g_dict, C, bs


def _common(h_list, g_dict, norb, num_grid, num_states, ci_sparse=True):
    return dict(
        h_list=h_list, g_dict=g_dict,
        num_particles=[NUM_PARTICLES_PER_WELL] * NUM_WELLS,
        num_mctdhf_orbitals=[norb] * NUM_WELLS,
        num_spatial_orbitals=[num_grid] * NUM_WELLS,
        num_states=num_states,
        ci_sparse=ci_sparse,
    )


STD_KW = dict(c_propagator="sil", b_propagator="cn",
              c_kwargs={"krylov_dim": 8})


def _ml_kw(args):
    """ML propagator choice.

    The SPF default 'diagonalize' builds a dense ``D_s * n_s`` generator by
    vmapping the RHS over its basis, which is the one place the SPF layer
    allocates something sizeable: at norb 9 with 8 SPFs that batch does not
    fit an 8 GB card. 'dop' and 'cn' build no matrix and are the fallbacks.
    """
    return dict(a_propagator="sil", B_propagator=args.B_propagator,
                b_propagator=args.b_propagator, a_kwargs={"krylov_dim": 8})


# ── measurement ───────────────────────────────────────────────────────────


def _peak_memory_mb():
    """Device allocator peak if JAX exposes one, else peak process RSS."""
    try:
        stats = jax.local_devices()[0].memory_stats()
        if stats and "peak_bytes_in_use" in stats:
            return stats["peak_bytes_in_use"] / 1e6, "device"
    except Exception:
        pass
    # Linux reports ru_maxrss in kilobytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3, "rss"


def _time_steps(step, y, num_steps):
    """Compile, then time `num_steps` steps. Returns (s/step, t_end, y_end)."""
    t1, y1 = step(0.0, y)
    jax.block_until_ready(y1)                      # compilation + first step
    start = time.perf_counter()
    t, y = t1, y1
    for _ in range(num_steps):
        t, y = step(t, y)
    jax.block_until_ready(y)
    return (time.perf_counter() - start) / num_steps, t, y


def run_standard(args, norb):
    h_list, g_dict, C, bs = initial_state(args.grid, norb, args.states)
    std = MCTDHF_n_species_DVR(**_common(h_list, g_dict, norb, args.grid,
                                         args.states, args.ci_sparse))
    step = get_cmf_integrator(std, args.dt, **STD_KW)
    sec, _t, _y = _time_steps(step, pack_Cb_to_y(C, bs), args.steps)
    mem, kind = _peak_memory_mb()
    return dict(sec=sec, mem=mem, mem_kind=kind)


def run_ml(args, norb, n, with_overlap):
    h_list, g_dict, A, Bs, bs = ml_initial_state(args.grid, norb,
                                                 args.states, n)
    common = _common(h_list, g_dict, norb, args.grid, args.states,
                     args.ci_sparse)
    ml = MLMCTDHF_n_species_DVR(num_spfs=[n] * NUM_WELLS, **common)
    step = get_ml_cmf_integrator(ml, args.dt, **_ml_kw(args))
    sec, _t, y_end = _time_steps(step, ml._ABb_to_y(A, Bs, bs), args.steps)
    mem, kind = _peak_memory_mb()

    out = dict(sec=sec, mem=mem, mem_kind=kind)

    if with_overlap:
        # Gate-matrix-style overlaps against the full-rank standard run at
        # the same orbital budget. The reference's OWN overlap matrix is
        # subtracted: real-time CMF does not conserve the norm exactly, and
        # that drift is common to both trajectories, so comparing against
        # the identity would report it as ML truncation error.
        _h, _g, C, _bs = initial_state(args.grid, norb, args.states)
        std = MCTDHF_n_species_DVR(**common)
        ref_step = get_cmf_integrator(std, args.dt, **STD_KW)
        _s, _t, y_ref = _time_steps(ref_step, pack_Cb_to_y(C, bs), args.steps)
        S = np.asarray(std.calculate_overlap(ml.to_mctdhf_y(y_end), y_ref))
        S_self = np.asarray(std.calculate_overlap(y_ref, y_ref))
        out["ovl"] = float(np.max(np.abs(S - S_self)))
    return out


def run_exactness(args, norb, num_states):
    """One ML step at n = D against one standard step.

    The orbital solver is run at a much tighter BiCGSTAB tolerance than the
    timing rows use. Both paths call the identical Crank-Nicolson kernel, but
    the ML top-layer operators reach the same numbers by a different
    contraction order and so differ in the last bit; at the default
    ``solver_tol = 1e-8`` an iterative solve turns that last bit into a 1e-8
    difference in the answer, which would mask the thing being measured.
    """
    h_list, g_dict, C, bs = initial_state(args.grid, norb, num_states)
    common = _common(h_list, g_dict, norb, args.grid, num_states,
                     args.ci_sparse)
    std = MCTDHF_n_species_DVR(**common)
    ml = MLMCTDHF_n_species_DVR(num_spfs=list(std.num_dets), **common)
    Bs = [jnp.eye(D, dtype=CDTYPE) for D in std.num_dets]

    tight = {"solver_tol": 1e-13, "max_solver_iter": 400}
    _t, y_std = get_cmf_integrator(std, args.dt, b_kwargs=tight, **STD_KW)(
        0.0, pack_Cb_to_y(C, bs))
    _t, y_ml = get_ml_cmf_integrator(ml, args.dt, b_kwargs=tight, **_ml_kw(args))(
        0.0, ml._ABb_to_y(C, Bs, bs))

    diff = float(np.max(np.abs(np.asarray(ml.to_mctdhf_y(y_ml))
                               - np.asarray(y_std))))
    scale = float(np.max(np.abs(np.asarray(y_std))))
    return dict(abs=diff, rel=diff / scale,
                dtype=str(np.asarray(y_std).dtype))


# ── subprocess driver ─────────────────────────────────────────────────────

_PASSTHROUGH = ("grid", "dt", "steps", "states")


def _child_argv(args, job):
    argv = [sys.executable, "-m", "tests.bench_ml_four_well", "--child",
            json.dumps(job)]
    for key in _PASSTHROUGH:
        argv += [f"--{key}", str(getattr(args, key))]
    argv += ["--B-propagator", args.B_propagator,
             "--b-propagator", args.b_propagator]
    if not args.ci_sparse:
        argv.append("--no-ci-sparse")
    if _X64:
        argv.append("--x64")
    return argv


def _run_child(args, job, timeout):
    """Run one configuration in a fresh interpreter; return its result dict."""
    try:
        proc = subprocess.run(
            _child_argv(args, job), capture_output=True, text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONWARNINGS": "ignore"},
        )
    except subprocess.TimeoutExpired:
        return {"note": f"timeout after {timeout:.0f}s"}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    return {"note": _explain_failure(proc)}


_BOILERPLATE = (
    "For simplicity, JAX has removed",
    "The stack trace below excludes",
    "--------------------",
)


def _explain_failure(proc):
    """The most informative line of a failed child, not just the last one.

    JAX ends a traceback with its own framework boilerplate, so the last line
    of stderr is usually "For simplicity, JAX has removed its internal
    frames" — which says nothing about what went wrong. Prefer the last line
    that looks like an exception.
    """
    lines = [ln.strip() for ln in (proc.stderr or "").splitlines() if ln.strip()]
    lines += [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    lines = [ln for ln in lines if not ln.startswith(_BOILERPLATE)]
    for ln in reversed(lines):
        if ("Error" in ln or "Exception" in ln or "error" in ln
                or "OOM" in ln or "RESOURCE_EXHAUSTED" in ln):
            return ln[:110]
    if proc.returncode and proc.returncode < 0:
        return f"killed by signal {-proc.returncode}"
    return (lines[-1] if lines else f"no result (rc={proc.returncode})")[:110]


def print_table(rows):
    head = (f"{'method':9s} {'norb':>4s} {'D':>4s} {'n':>4s} {'top dim':>11s} "
            f"{'s/step':>10s} {'peak MB':>9s} {'src':>6s} "
            f"{'|dS|':>9s}")
    print(head)
    print("-" * len(head))
    for r in rows:
        def fmt(key, spec, width):
            v = r.get(key)
            return ("—" if v is None else format(v, spec)).rjust(width)
        print(f"{r['kind']:9s} {r['norb']:4d} {r['D']:4d} {r['n']:4d} "
              f"{r['top_dim']:11d} {fmt('sec', '.4f', 10)} "
              f"{fmt('mem', '.0f', 9)} {r.get('mem_kind', ''):>6s} "
              f"{fmt('ovl', '.2e', 9)}"
              + (f"   {r['note']}" if r.get("note") else ""))


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--norb", type=int, nargs="+", default=[6, 7, 9])
    p.add_argument("--spfs", type=int, nargs="+", default=[4, 6, 8])
    p.add_argument("--states", type=int, default=16)
    p.add_argument("--grid", type=int, default=96)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--std-budget", type=int, default=4_000_000,
                   help="skip the standard solver above this CI dimension")
    p.add_argument("--timeout", type=float, default=1800.0,
                   help="per-configuration wall-clock limit, seconds")
    p.add_argument("--x64", action="store_true",
                   help="double precision (needed for the 1e-10 exactness "
                        "target; production runs are complex64)")
    p.add_argument("--B-propagator", default="diagonalize",
                   choices=("diagonalize", "cn", "sil", "dop", "bs"),
                   help="SPF-layer propagator (default diagonalize)")
    p.add_argument("--b-propagator", default="cn",
                   choices=("cn", "diagonalize", "sil", "dop", "bs"),
                   help="orbital-layer propagator (default cn)")
    p.add_argument("--no-ci-sparse", dest="ci_sparse", action="store_false",
                   help="dense excitation tensors. XLA's GPU scatter codegen "
                        "needs excessive per-thread local memory on large "
                        "operands, which can fail to launch even with the "
                        "card nearly empty; this is the documented fallback")
    p.add_argument("--skip-exactness", action="store_true")
    p.add_argument("--quick", action="store_true",
                   help="small, fast configuration for a smoke run")
    p.add_argument("--child", type=str, default=None,
                   help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    # ── child mode: run exactly one configuration and report it ──────────
    if args.child is not None:
        job = json.loads(args.child)
        if job["kind"] == "standard":
            res = run_standard(args, job["norb"])
        elif job["kind"] == "ML":
            res = run_ml(args, job["norb"], job["n"], job["overlap"])
        else:
            res = run_exactness(args, job["norb"], job["states"])
        print("RESULT " + json.dumps(res))
        return 0

    if args.quick:
        args.norb, args.spfs, args.states = [4, 5], [2, 4], 8
        args.grid, args.steps = 32, 2

    print(f"jax {jax.__version__} on {jax.devices()}  "
          f"dtype {'complex128' if _X64 else 'complex64'}")
    print(f"grid={args.grid} states={args.states} dt={args.dt} "
          f"steps={args.steps} B={args.B_propagator} b={args.b_propagator} "
          f"ci_sparse={args.ci_sparse}\n")

    if not args.skip_exactness:
        norb0 = args.norb[0]
        res = _run_child(args, {"kind": "exact", "norb": norb0,
                                "states": min(args.states, 4)}, args.timeout)
        if "abs" in res:
            print(f"(a) exactness at n = D, norb = {norb0}, "
                  f"{res['dtype']}: max |Δy| = {res['abs']:.3e} "
                  f"(relative {res['rel']:.3e})")
            if not _X64:
                print("    complex64 floor; rerun with --x64 for the 1e-10 "
                      "target.")
        else:
            print(f"(a) exactness: {res.get('note')}")
        print()

    rows = []
    norb0 = args.norb[0]
    for norb in args.norb:
        D = comb(norb, NUM_PARTICLES_PER_WELL, exact=True)
        top_dim_std = args.states * D ** NUM_WELLS

        row = dict(kind="standard", norb=norb, D=D, n=D, top_dim=top_dim_std)
        if top_dim_std <= args.std_budget:
            row.update(_run_child(args, {"kind": "standard", "norb": norb},
                                  args.timeout))
        else:
            row["note"] = "skipped (CI dim over --std-budget)"
        rows.append(row)

        for n in args.spfs:
            if n > D:
                continue
            want_ovl = (norb == norb0) and top_dim_std <= args.std_budget
            row = dict(kind="ML", norb=norb, D=D, n=n,
                       top_dim=args.states * n ** NUM_WELLS)
            row.update(_run_child(
                args, {"kind": "ML", "norb": norb, "n": n,
                       "overlap": want_ovl}, args.timeout))
            rows.append(row)

    print("(b) cost — one subprocess per row, so peak memory is per-row")
    print_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
