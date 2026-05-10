"""
Stokes solver with CG-based pressure correction (Uzawa-CG scheme).

Usage:
    python solcx.py [options]

Example:
    python solcx.py --monitor
    python solcx.py --res 8 --iter 20 --monitor
    mpirun -n 4 python solcx.py --res 8 --monitor --no-pcg-tol
"""

# ─────────────────────────────────────────────────────────
# Problem / solver parameters  (edit here before running)
# ─────────────────────────────────────────────────────────
RES           = 7       # mesh resolution exponent: nx = 2**RES
PCG_MAX_ITER  = 8       # max pressure correction iterations
RTOL          = 1e-10   # relative tolerance for all Krylov solvers
ATOL          = 1e-12   # absolute tolerance for all Krylov solvers
MAX_KRYLOV_IT = 500     # maximum Krylov iterations
GMRES_RESTART = 400     # GMRES restart parameter
PCG_TOL       = 1e-9    # stop early if ||qk||_M < PCG_TOL (set --no-pcg-tol to disable)
# ─────────────────────────────────────────────────────────

import argparse
import logging
import time

import numpy as np
from fenics import *
from dolfin import *
from mpi4py import MPI as pyMPI
from petsc4py import PETSc

logging.getLogger("FFC").setLevel(logging.WARNING)
logging.getLogger("UFL").setLevel(logging.WARNING)

# ── ANSI colour helpers ───────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    CYAN   = "\033[36m"
    GREEN  = "\033[32m"
    YELLOW = "\033[33m"
    RED    = "\033[31m"
    BLUE   = "\033[34m"
    MAGENTA= "\033[35m"
    WHITE  = "\033[97m"
    LIGHT_GREEN = "\033[92m"

def b(s):       return f"{C.BOLD}{s}{C.RESET}"
def dim(s):     return f"{C.DIM}{s}{C.RESET}"
def cyan(s):    return f"{C.CYAN}{s}{C.RESET}"
def green(s):   return f"{C.GREEN}{s}{C.RESET}"
def yellow(s):  return f"{C.YELLOW}{s}{C.RESET}"
def red(s):     return f"{C.RED}{s}{C.RESET}"
def blue(s):    return f"{C.BLUE}{s}{C.RESET}"
def magenta(s): return f"{C.MAGENTA}{s}{C.RESET}"
def lgreen(s):  return f"{C.LIGHT_GREEN}{s}{C.RESET}"
# ─────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stokes solver with CG pressure correction (FEniCS + PETSc)"
    )
    parser.add_argument(
        "--res", type=int, default=RES,
        help=f"Mesh resolution exponent: nx = 2**res (default: {RES})"
    )
    parser.add_argument(
        "--iter", type=int, default=PCG_MAX_ITER,
        help=f"Max pressure correction iterations (default: {PCG_MAX_ITER})"
    )
    parser.add_argument(
        "--output", type=str, default="./output/solcx_output.xdmf",
        help="Output XDMF file path (default: output.xdmf)"
    )
    parser.add_argument(
        "--analytical-dir", type=str, default='./analytical_solutions',
        help="Directory with analytical solution HDF5 files. "
             "Expected: u_a<nx>.h5, p_a<nx>.h5"
    )
    parser.add_argument(
        "--monitor", action="store_true", default=False,
        help="Print per-iteration convergence table"
    )
    parser.add_argument(
        "--no-output", action="store_true", default=False,
        help="Skip writing XDMF output file"
    )
    parser.add_argument(
        "--no-pcg-tol", action="store_true", default=False,
        help=f"Disable early stopping by PCG_TOL={PCG_TOL:.0e} (run all --iter iterations)"
    )
    parser.add_argument(
        "--quiet", action="store_true", default=False,
        help="Suppress per-iteration table; print only final residual and termination reason"
    )
    return parser.parse_args()


def build_mesh(comm, nx, ny):
    return RectangleMesh.create(
        comm,
        [Point(0.0, 0.0), Point(1.0, 1.0)],
        [nx, ny],
        CellType.Type.triangle,
        "left/right",
    )


def build_spaces(mesh):
    DG0 = FiniteElement("DG", mesh.ufl_cell(), 0)
    P2  = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1  = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, P2)
    Q = FunctionSpace(mesh, P1)
    Y = FunctionSpace(mesh, DG0)
    return V, Q, Y


def build_bcs(V):
    return [
        DirichletBC(V.sub(0), Constant(0.0), "near(x[0], 0.0) or near(x[0], 1.0)"),
        DirichletBC(V.sub(1), Constant(0.0), "near(x[1], 0.0) or near(x[1], 1.0)"),
    ]


def epsilon(w):
    return sym(nabla_grad(w))


def build_stokes_solver(L_, bcs, rtol, atol, max_it, gmres_restart):
    solver_s = PETScKrylovSolver("gmres")
    solver_s.set_operator(L_)

    ksp = as_backend_type(solver_s).ksp()
    ksp.setGMRESRestart(gmres_restart)
    ksp.setTolerances(rtol=rtol, atol=atol, max_it=max_it)
    ksp.setFromOptions()

    pc = ksp.getPC()
    pc.setType("hypre")
    pc.setHYPREType("boomeramg")

    return solver_s, ksp


def build_mass_solver(M_, rtol, atol, max_it):
    solver_q = PETScKrylovSolver("cg")
    solver_q.set_operator(M_)
    solver_q.parameters["error_on_nonconvergence"] = False

    ksp_q = as_backend_type(solver_q).ksp()
    ksp_q.setTolerances(rtol=rtol, atol=atol, max_it=max_it)

    pc_q = ksp_q.getPC()
    pc_q.setType("hypre")

    return solver_q


def build_w_solver(Aw, rtol, atol, max_it):
    solver_w = PETScKrylovSolver("cg", "hypre_amg")
    solver_w.parameters["maximum_iterations"] = max_it
    solver_w.parameters["absolute_tolerance"] = atol
    solver_w.parameters["relative_tolerance"] = rtol
    solver_w.parameters["error_on_nonconvergence"] = False
    solver_w.parameters["monitor_convergence"] = False
    solver_w.set_operator(Aw)
    return solver_w


def load_analytical(V, Q, nx, analytical_dir):
    u_a = Function(V)
    p_a = Function(Q)
    try:
        hdf_u = HDF5File(MPI.comm_world, f"{analytical_dir}/u_a{nx}.h5", "r")
        hdf_u.read(u_a, "/u")
        del hdf_u

        hdf_p = HDF5File(MPI.comm_world, f"{analytical_dir}/p_a{nx}.h5", "r")
        hdf_p.read(p_a, "/u")
        del hdf_p
        return u_a, p_a, True
    except Exception:
        return u_a, p_a, False


def compute_errors(u, p, u_a, p_a):
    u_err = np.sqrt(assemble(inner(u - u_a, u - u_a) * dx))
    p_err = np.sqrt(assemble((p - p_a) * (p - p_a) * dx))
    return u_err, p_err


def strip_ansi(s):
    """Return string with ANSI escape codes removed (for length calculation)."""
    import re
    return re.sub(r"\033\[[0-9;]*m", "", s)


def get_gpu_info():
    """Return (count, summary) of GPUs actually used by this process.

    Respects CUDA_VISIBLE_DEVICES: if set, only those GPU indices are counted.
    Returns (0, None) if nvidia-smi is unavailable.
    """
    import os, subprocess
    from collections import Counter
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        all_gpus = []   # list of (index, name) in physical order
        for line in out.splitlines():
            parts = line.split(",", 1)
            if len(parts) == 2:
                all_gpus.append((int(parts[0].strip()), parts[1].strip()))
        if not all_gpus:
            return 0, None

        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if cvd and cvd.lower() != "nodevfile":
            # filter to the requested indices
            requested = [int(x) for x in cvd.split(",") if x.strip().lstrip("-").isdigit()]
            index_map = {idx: name for idx, name in all_gpus}
            used = [(i, index_map[i]) for i in requested if i in index_map]
        else:
            used = all_gpus

        if not used:
            return 0, None

        counts = Counter(name for _, name in used)
        summary = ", ".join(f"{n}x {name}" for name, n in counts.items())
        return len(used), summary
    except Exception:
        return 0, None

W_IT    = 3
W_NORM  = 16
W_UERR  = 16
W_PERR  = 16
W_GMRES = 5

def fmt_norm(val, tol=None):
    """Colour-code a residual norm: green if below tol, yellow if close, red otherwise."""
    s = f"{val:.6e}"
    if tol is None:
        return cyan(s)
    if val < tol:
        return green(s)
    if val < tol * 1e2:
        return yellow(s)
    return red(s)

def pad_ansi(s, width):
    visible = len(strip_ansi(s))
    return s + " " * max(width - visible, 0)

def cell(text, width, align="right"):
    """Pad printable width while ignoring ANSI escape sequences."""
    visible = len(strip_ansi(text))
    pad = max(width - visible, 0)
    if align == "left":
        return text + " " * pad
    return " " * pad + text
    
def main():
    args = parse_args()

    set_log_active(False)
    comm = pyMPI.COMM_WORLD
    rank = comm.Get_rank()

    parameters["ghost_mode"] = "shared_vertex"

    nx = 2 ** args.res
    ny = nx
    use_pcg_tol = not args.no_pcg_tol

    if rank == 0:
        gpu_count, gpu_summary = get_gpu_info()
        KEY_W = 16
        TITLE = "  GPU Stokes Solver  (PCG)  "

        # Build rows as (key, value_str) to measure widths before printing
        rows = [
            ("Mesh",         f"{nx} × {ny}  (res={args.res})"),
            ("PCG iters",    f"{args.iter}  {'(no early stop)' if not use_pcg_tol else f'(tol={PCG_TOL:.0e})'}"),
            ("Krylov tol",   f"rtol={RTOL:.0e}  atol={ATOL:.0e}"),
            ("GMRES restart",str(GMRES_RESTART)),
            ("Output",       "disabled" if args.no_output else args.output),
            ("GPUs",         f"{gpu_count}  {gpu_summary}" if gpu_count else "none detected"),
        ]

        # W = max of title width and all row visible widths, plus 2 margin
        row_visible = lambda k, v: len(f"  {k}{' ' * (KEY_W - len(k))}{v}") + 2
        W = max(len(TITLE), max(row_visible(k, v) for k, v in rows)) + 2

        def print_row(k, v, colored_v):
            line = f"  {yellow(k)}{' ' * (KEY_W - len(k))}{C.RESET}{colored_v}"
            pad = W - len(strip_ansi(line))
            print(b(lgreen("║")) + line + " " * max(pad, 0) + b(lgreen("║")))

        print()
        print(b(lgreen("╔" + "═" * W + "╗")))
        print(b(lgreen("║")) + b(f"{TITLE:^{W}}") + b(lgreen("║")))
        print(b(lgreen("╠" + "═" * W + "╣")))
        print_row("Mesh",          rows[0][1], rows[0][1])
        print_row("PCG iters",     rows[1][1], rows[1][1])
        print_row("Krylov tol",    rows[2][1], rows[2][1])
        print_row("GMRES restart", rows[3][1], rows[3][1])
        print_row("Output",        rows[4][1], rows[4][1])
        if gpu_count:
            print_row("GPUs", rows[5][1], f"{green(str(gpu_count))}  {dim(gpu_summary)}")
        else:
            print_row("GPUs", rows[5][1], yellow("none detected"))
        print(b(lgreen("╚" + "═" * W + "╝")))
        print()

    t_start = time.time()

    # ── mesh & spaces ─────────────────────────────────────
    mesh = build_mesh(comm, nx, ny)
    V, Q, Y = build_spaces(mesh)

    u_n = Function(V);  p_n = Function(Q)
    u_  = Function(V);  p_  = Function(Q)
    qk  = Function(Q);  dk  = Function(Q)
    hk  = Function(V);  wk  = Function(Q)
    rho = Function(Y);  eta = Function(Y)

    rho_expr = Expression("sin(pi*x[1]) * cos(pi*x[0])", degree=2, pi=np.pi)
    eta_expr = Expression("(x[0] <= 0.5) ? 1.0 : 1e6", degree=0)
    eta.interpolate(eta_expr)
    rho.interpolate(rho_expr)

    # ── analytical solution (optional) ───────────────────
    has_analytical = False
    if args.analytical_dir:
        u_a, p_a, has_analytical = load_analytical(V, Q, nx, args.analytical_dir)
        if rank == 0:
            tag = green("loaded") if has_analytical else yellow("not found — skipping error calc")
            print(f"  Analytical solution : {tag}\n")

    # ── BCs & initial conditions ──────────────────────────
    bcs = build_bcs(V)
    p_.interpolate(Expression("0.0", degree=4))
    p_n.interpolate(Expression("0.0", degree=4))

    g = Constant((0.0, 1.0))
    u, v = TrialFunction(V), TestFunction(V)
    p, q = TrialFunction(Q), TestFunction(Q)

    if rank == 0:
        print(f"  {dim('Assembling matrices...')}", flush=True)

    t0 = time.time()
    L_ = assemble(inner(2.0 * eta * epsilon(u), epsilon(v)) * dx)
    G_ = assemble(-inner(p, div(v)) * dx)
    D_ = assemble(div(u) * q * dx)
    M_ = assemble(p * q * dx)
    Aw = assemble((1.0 / eta) * p * q * dx)
    t_assemble = time.time() - t0

    if rank == 0:
        print(f"  {dim('Assembly done')}  {cyan(f'{t_assemble:.2f}s')}")

    solver_s, ksp = build_stokes_solver(L_, bcs, RTOL, ATOL, MAX_KRYLOV_IT, GMRES_RESTART)
    solver_q = build_mass_solver(M_, RTOL, ATOL, MAX_KRYLOV_IT)
    solver_w = build_w_solver(Aw, RTOL, ATOL, MAX_KRYLOV_IT)

    # ── initial Stokes solve ──────────────────────────────
    if rank == 0:
        print(f"  {dim('Initial velocity guess...')}", flush=True)

    t0 = time.time()
    bs = assemble(inner(rho * g - grad(p_n), v) * dx)
    for bc in bcs:
        bc.apply(L_, bs)

    solver_s.solve(u_n.vector(), bs)
    n_iter = ksp.getIterationNumber()
    t_initial = time.time() - t0

    if rank == 0:
        print(f"  {dim('Initial solve done')}  {cyan(f'{t_initial:.2f}s')}  {dim(f'({n_iter} GMRES its)')}")

    bq = D_ * u_n.vector()
    solver_q.solve(qk.vector(), bq)

    bw = assemble(qk * q * dx)
    solver_w.solve(wk.vector(), bw)
    dk.vector()[:] = -wk.vector()[:]
    
    qk_norm0 = np.sqrt(qk.vector().inner(bw))
    u_err0, p_err0 = None, None
    if has_analytical and args.monitor:
        u_err0, p_err0 = compute_errors(u_n, p_n, u_a, p_a)
        
    # ── table header ──────────────────────────────────────
    if rank == 0:
        tol_str = f"tol={PCG_TOL:.0e}" if use_pcg_tol else "no early stop"
        print(f"\n  {b('Pressure correction')}  {dim(f'(max {args.iter} iters, {tol_str})')}\n")
    
        if not args.quiet:
            if has_analytical and args.monitor:
                hdr = (
                    "  "
                    + b(cyan(cell("it",      W_IT,    "right"))) + " │ "
                    + b(cyan(cell("‖qk‖_M",  W_NORM,  "right"))) + " │ "
                    + b(cyan(cell("u_error", W_UERR,  "right"))) + " │ "
                    + b(cyan(cell("p_error", W_PERR,  "right"))) + " │ "
                    + b(cyan(cell("GMRES",   W_GMRES, "right")))
                )
                sep = (
                    "  "
                    + "─" * W_IT    + "─┼─"
                    + "─" * W_NORM  + "─┼─"
                    + "─" * W_UERR  + "─┼─"
                    + "─" * W_PERR  + "─┼─"
                    + "─" * W_GMRES
                )
            else:
                hdr = (
                    "  "
                    + b(cyan(cell("it",     W_IT,    "right"))) + " │ "
                    + b(cyan(cell("‖qk‖_M", W_NORM,  "right"))) + " │ "
                    + b(cyan(cell("GMRES",  W_GMRES, "right")))
                )
                sep = (
                    "  "
                    + "─" * W_IT    + "─┼─"
                    + "─" * W_NORM  + "─┼─"
                    + "─" * W_GMRES
                )
    
            print(hdr)
            print(dim(sep))
            
            it0_str    = cell(b("0"), W_IT, "right")
            norm0_str  = cell(fmt_norm(qk_norm0, PCG_TOL if use_pcg_tol else None), W_NORM, "right")
            gmres0_str = cell(f"{n_iter}", W_GMRES, "right")

            if has_analytical and args.monitor:
                uerr0_str = cell(cyan(f"{u_err0:.6e}"), W_UERR, "right")
                perr0_str = cell(cyan(f"{p_err0:.6e}"), W_PERR, "right")
                print(
                    f"  {it0_str} │ {norm0_str} │ {uerr0_str} │ {perr0_str} │ {gmres0_str}"
                )
            else:
                print(
                    f"  {it0_str} │ {norm0_str} │ {gmres0_str}"
                )
                
    t_iter = time.time()
    num = qk.vector().inner(M_ * wk.vector())
    qk_norm = None
    converged = False
    converged_at = args.iter

    for ii in range(args.iter):
        bs = G_ * dk.vector()
        for bc in bcs:
            bc.apply(bs)
        solver_s.solve(hk.vector(), bs)
        n_iter = ksp.getIterationNumber()

        denom = (G_ * dk.vector()).inner(hk.vector())
        ak = num / denom

        p_.vector()[:] = p_n.vector()[:] + ak * dk.vector()[:]
        u_.vector().zero()
        u_.vector().axpy(1.0, u_n.vector())
        u_.vector().axpy(-ak, hk.vector())

        p_n.assign(p_)
        u_n.assign(u_)

        bq = D_ * u_.vector()
        solver_q.solve(qk.vector(), bq)

        bw = M_ * qk.vector()
        solver_w.solve(wk.vector(), bw)

        new_num = qk.vector().inner(M_ * wk.vector())
        bk      = new_num / num
        num     = new_num

        dk.vector()[:] = -wk.vector()[:] + bk * dk.vector()[:]

        qk_norm = np.sqrt(qk.vector().inner(bw))
        tol_for_color = PCG_TOL if use_pcg_tol else None
        if has_analytical and args.monitor:
            u_err, p_err = compute_errors(u_, p_, u_a, p_a)
        if rank == 0 and not args.quiet:
            it_str    = cell(b(f"{ii+1}"), W_IT, "right")
            norm_str  = cell(fmt_norm(qk_norm, tol_for_color), W_NORM, "right")
            gmres_str = cell(f"{n_iter}", W_GMRES, "right")
        
            if has_analytical and args.monitor:
                uerr_str = cell(cyan(f"{u_err:.6e}"), W_UERR, "right")
                perr_str = cell(cyan(f"{p_err:.6e}"), W_PERR, "right")
        
                print(
                    f"  {it_str} │ {norm_str} │ {uerr_str} │ {perr_str} │ {gmres_str}"
                )
            else:
                print(
                    f"  {it_str} │ {norm_str} │ {gmres_str}"
                )

        if use_pcg_tol and qk_norm < PCG_TOL:
            converged = True
            converged_at = ii + 1
            break
    else:
        converged = False
        converged_at = args.iter

    t_total = time.time() - t_start
    t_loop  = time.time() - t_iter

    if rank == 0:
        print()
        print(dim("  " + "─" * 48))
        if converged:
            reason = green(f"✓ Converged at iter {converged_at}  (‖qk‖_M < {PCG_TOL:.0e})")
        elif use_pcg_tol:
            reason = yellow(f"△ Max iterations reached ({args.iter})  — not converged")
        else:
            reason = dim(f"■ Completed {args.iter} iterations  (PCG_TOL disabled)")
        print(f"  {reason}")
        print(f"  {b('Final')}  ‖qk‖_M = {fmt_norm(qk_norm, tol_for_color)}")
        print(f"  {dim('Loop')}   {t_loop:.2f}s   {dim('│')}   {dim('Total')}  {t_total:.2f}s")
        print(dim("  " + "─" * 48))
        print()

    if not args.no_output:
        if rank == 0:
            print(f"  {dim('Writing')} {args.output} ...", flush=True)
        xdmf = XDMFFile(mesh.mpi_comm(), args.output)
        xdmf.write(mesh)
        xdmf.write(u_, 0.0)
        xdmf.write(p_, 0.0)
        if rank == 0:
            print(f"  {green('✓')} Output written.")


if __name__ == "__main__":
    main()
