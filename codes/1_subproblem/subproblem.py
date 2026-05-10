"""
Variable-viscosity vector PDE solver (single GMRES solve).

Problem:
    Solve
        -div(2 * eta * epsilon(u)) = -f
    on [0,1] x [0,1]
    with Dirichlet boundary condition given by the analytical solution.

Usage:
    python solcx_single.py
    python solcx_single.py --res 8
    python solcx_single.py --res 8 --monitor
    mpirun -n 4 python solcx_single.py --res 8 --monitor
"""

# ─────────────────────────────────────────────────────────
# Problem / solver parameters  (edit here before running)
# ─────────────────────────────────────────────────────────
RES            = 8       # mesh resolution exponent: nx = 2**RES
RTOL           = 1e-10   # relative tolerance
ATOL           = 1e-30   # absolute tolerance
MAX_KRYLOV_IT  = 500      # maximum Krylov iterations
GMRES_RESTART  = 400     # GMRES restart parameter
# ─────────────────────────────────────────────────────────

import argparse
import time
import subprocess
import os
from collections import Counter

import numpy as np
from fenics import *
from dolfin import *
from mpi4py import MPI as pyMPI
from petsc4py import PETSc

parameters["ghost_mode"] = "shared_vertex"


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


def strip_ansi(s):
    import re
    return re.sub(r"\033\[[0-9;]*m", "", s)


def cell(text, width, align="right"):
    visible = len(strip_ansi(text))
    pad = max(width - visible, 0)
    if align == "left":
        return text + " " * pad
    return " " * pad + text


def get_gpu_info():
    """
    Return (count, summary) of GPUs visible to this process.
    Respects CUDA_VISIBLE_DEVICES.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        all_gpus = []
        for line in out.splitlines():
            parts = line.split(",", 1)
            if len(parts) == 2:
                all_gpus.append((int(parts[0].strip()), parts[1].strip()))

        if not all_gpus:
            return 0, None

        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if cvd and cvd.lower() != "nodevfile":
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single variable-viscosity vector PDE solve (FEniCS + PETSc)"
    )
    parser.add_argument(
        "--res", type=int, default=RES,
        help=f"Mesh resolution exponent: nx = 2**res (default: {RES})"
    )
    parser.add_argument(
        "--rtol", type=float, default=RTOL,
        help=f"Relative tolerance for GMRES (default: {RTOL})"
    )
    parser.add_argument(
        "--atol", type=float, default=ATOL,
        help=f"Absolute tolerance for GMRES (default: {ATOL})"
    )
    parser.add_argument(
        "--max-it", type=int, default=MAX_KRYLOV_IT,
        help=f"Maximum GMRES iterations (default: {MAX_KRYLOV_IT})"
    )
    parser.add_argument(
        "--restart", type=int, default=GMRES_RESTART,
        help=f"GMRES restart value (default: {GMRES_RESTART})"
    )
    parser.add_argument(
        "--output", type=str, default="./output/solcx_single_output.xdmf",
        help="Output XDMF file path"
    )
    parser.add_argument(
        "--no-output", action="store_true", default=False,
        help="Skip writing XDMF output"
    )
    parser.add_argument(
        "--monitor", action="store_true", default=False,
        help="Print solver summary table"
    )
    parser.add_argument(
        "--ksp-monitor", action="store_true", default=False,
        help="Enable PETSc KSP monitor via options database"
    )
    parser.add_argument(
        "--quiet", action="store_true", default=False,
        help="Print only essential final information"
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
    P2_vec = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P2_sca = FiniteElement("Lagrange", mesh.ufl_cell(), 2)
    V = FunctionSpace(mesh, P2_vec)
    Q = FunctionSpace(mesh, P2_sca)
    return V, Q


def epsilon(w):
    return sym(nabla_grad(w))


def build_problem(V, Q):
    u = TrialFunction(V)
    v = TestFunction(V)

    uu = Function(V, name="u")
    eta = Function(Q, name="eta")
    f = Function(V, name="f")
    u_exact = Function(V, name="u_exact")

    eta_expr = Expression(
        "exp(c * x[0])",
        c=np.log(1e6),
        degree=4,
    )

    f_expr = Expression(
        (
            "2*pi*exp(c*x[0]) * sin(pi*x[1]) * (c*cos(pi*x[0]) - pi*sin(pi*x[0]))",
            "-2*pi*pi * exp(c*x[0]) * cos(pi*x[0]) * cos(pi*x[1])",
        ),
        c=np.log(1e6),
        pi=np.pi,
        degree=4,
    )

    u_expr = Expression(
        (
            "sin(pi*x[0])*sin(pi*x[1])",
            "cos(pi*x[0])*cos(pi*x[1])",
        ),
        pi=np.pi,
        degree=4,
    )

    eta.interpolate(eta_expr)
    f.interpolate(f_expr)
    u_exact.interpolate(u_expr)

    bcs = [DirichletBC(V, u_expr, "on_boundary")]

    a_form = inner(2.0 * eta * epsilon(u), epsilon(v)) * dx
    rhs_form = inner(-f, v) * dx

    return {
        "u": u,
        "v": v,
        "uu": uu,
        "eta": eta,
        "f": f,
        "u_exact": u_exact,
        "u_expr": u_expr,
        "a_form": a_form,
        "rhs_form": rhs_form,
        "bcs": bcs,
    }


def build_solver(A, rtol, atol, max_it, restart):
    solver = PETScKrylovSolver("gmres", "hypre_amg")
    solver.set_operator(A)

    ksp = as_backend_type(solver).ksp()
    ksp.setGMRESRestart(restart)
    ksp.setTolerances(rtol=rtol, atol=atol, max_it=max_it)

    pc = ksp.getPC()
    pc.setType("hypre")
    pc.setHYPREType("boomeramg")

    ksp.setFromOptions()
    return solver, ksp


def compute_l2_error(u_num, u_ex):
    return np.sqrt(assemble(inner(u_num - u_ex, u_num - u_ex) * dx))


def compute_relative_l2_error(u_num, u_ex):
    num = assemble(inner(u_num - u_ex, u_num - u_ex) * dx)
    den = assemble(inner(u_ex, u_ex) * dx)
    if den <= 0.0:
        return np.nan
    return np.sqrt(num / den)


def main():
    args = parse_args()

    set_log_active(False)
    comm = pyMPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if args.ksp_monitor:
        opts = PETSc.Options()
        opts["ksp_monitor"] = None

    nx = 2 ** args.res
    ny = nx

    if rank == 0 and not args.quiet:
        gpu_count, gpu_summary = get_gpu_info()
        KEY_W = 16
        TITLE = "  Variable-Viscosity subproblem Solver  "

        rows = [
            ("MPI ranks",      f"{size}"),
            ("Mesh",           f"{nx} × {ny}  (res={args.res})"),
            ("Krylov tol",     f"rtol={args.rtol:.0e}  atol={args.atol:.0e}"),
            ("GMRES restart",  f"{args.restart}"),
            ("Max GMRES it",   f"{args.max_it}"),
            ("Output",         "disabled" if args.no_output else args.output),
            ("GPUs",           f"{gpu_count}  {gpu_summary}" if gpu_count else "none detected"),
        ]

        row_visible = lambda k, v: len(f"  {k}{' ' * (KEY_W - len(k))}{v}") + 2
        W = max(len(TITLE), max(row_visible(k, v) for k, v in rows)) + 2

        def print_row(k, colored_v):
            line = f"  {yellow(k)}{' ' * (KEY_W - len(k))}{C.RESET}{colored_v}"
            pad = W - len(strip_ansi(line))
            print(b(lgreen("║")) + line + " " * max(pad, 0) + b(lgreen("║")))

        print()
        print(b(lgreen("╔" + "═" * W + "╗")))
        print(b(lgreen("║")) + b(f"{TITLE:^{W}}") + b(lgreen("║")))
        print(b(lgreen("╠" + "═" * W + "╣")))
        print_row("MPI ranks", rows[0][1])
        print_row("Mesh", rows[1][1])
        print_row("Krylov tol", rows[2][1])
        print_row("GMRES restart", rows[3][1])
        print_row("Max GMRES it", rows[4][1])
        print_row("Output", rows[5][1])
        if gpu_count:
            print_row("GPUs", f"{green(str(gpu_count))}  {dim(gpu_summary)}")
        else:
            print_row("GPUs", yellow("none detected"))
        print(b(lgreen("╚" + "═" * W + "╝")))
        print()

    t_total0 = time.time()

    if rank == 0 and not args.quiet:
        print(f"  {dim('Building mesh and spaces...')}", flush=True)

    t0 = time.time()
    mesh = build_mesh(comm, nx, ny)
    V, Q = build_spaces(mesh)
    t_mesh = time.time() - t0

    problem = build_problem(V, Q)

    if rank == 0 and not args.quiet:
        print(f"  {dim('Assembling system...')}", flush=True)

    t0 = time.time()
    A = assemble(problem["a_form"])
    b_vec = assemble(problem["rhs_form"])

    for bc in problem["bcs"]:
        bc.apply(A, b_vec)

    t_assemble = time.time() - t0

    solver, ksp = build_solver(
        A,
        rtol=args.rtol,
        atol=args.atol,
        max_it=args.max_it,
        restart=args.restart
    )

    if rank == 0 and not args.quiet:
        print(f"  {dim('Solving...')}", flush=True)

    t0 = time.time()
    solver.solve(problem["uu"].vector(), b_vec)
    t_solve = time.time() - t0

    gmres_it = ksp.getIterationNumber()
    converged_reason = ksp.getConvergedReason()

    u_error = compute_l2_error(problem["uu"], problem["u_exact"])
    u_rel_error = compute_relative_l2_error(problem["uu"], problem["u_exact"])

    t_total = time.time() - t_total0

    if rank == 0:
        print()
        print(f"  {b('Solve summary')}\n")

        W_KEY = 18
        W_VAL = 18

        hdr = (
            "  "
            + b(cyan(cell("quantity", W_KEY, "left"))) + " │ "
            + b(cyan(cell("value", W_VAL, "right")))
        )
        sep = (
            "  "
            + "─" * W_KEY + "─┼─" + "─" * W_VAL
        )

        print(hdr)
        print(dim(sep))

        def row(k, v):
            print(f"  {cell(k, W_KEY, 'left')} │ {cell(v, W_VAL, 'right')}")

        row("mesh build [s]", f"{t_mesh:.3f}")
        row("assembly [s]", f"{t_assemble:.3f}")
        row("solve [s]", f"{t_solve:.3f}")
        row("total [s]", f"{t_total:.3f}")
        row("GMRES iters", f"{gmres_it}")
        row("L2 error", f"{u_error:.6e}")
        row("rel L2 error", f"{u_rel_error:.6e}")

        print()
        if converged_reason > 0:
            print(f"  {green('✓ Converged')}  {dim(f'(PETSc reason = {converged_reason})')}")
        else:
            print(f"  {red('✗ Not converged')}  {dim(f'(PETSc reason = {converged_reason})')}")
        print()

    if not args.no_output:
        if rank == 0 and not args.quiet:
            print(f"  {dim('Writing')} {args.output} ...", flush=True)

        xdmf = XDMFFile(mesh.mpi_comm(), args.output)
        xdmf.parameters["flush_output"] = True
        xdmf.parameters["functions_share_mesh"] = True
        xdmf.write(mesh)
        xdmf.write(problem["uu"], 0.0)
        xdmf.write(problem["eta"], 0.0)

        if rank == 0 and not args.quiet:
            print(f"  {green('✓')} Output written.")


if __name__ == "__main__":
    main()