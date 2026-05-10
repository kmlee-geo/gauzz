"""
Blanken: 3-D Mantle-Convection Solver (Stokes + Temperature Advection-Diffusion).

Stokes equation is solved via an Uzawa-CG pressure correction loop.
Temperature is advanced with a Crank-Nicolson advection-diffusion scheme.

Usage:
    python blanken.py [options]

Example:
    python blanken.py --monitor
    python blanken.py --nx 50 --ny 20 --nz 50 --Ra 1e5 --steps 100 --monitor
    mpirun -n 4 python blanken.py --nx 100 --ny 40 --nz 100 --Ra 1e4 --steps 200
"""

NX            = 100
NY            = 40
NZ            = 100
RA            = 1e4
DT            = 0.001
NSTEPS        = 251
UZAWA_MAX     = 2000
UZAWA_TOL     = 1e-9
RTOL          = 1e-7
ATOL          = 1e-12
MAX_KRYLOV_IT = 500
GMRES_RESTART = 400
ALPHA_INIT    = 0.02

import argparse
import logging
import time
import os

logging.getLogger("FFC").setLevel(logging.WARNING)
logging.getLogger("UFL").setLevel(logging.WARNING)

import numpy as np
from dolfin import *
from mpi4py import MPI as pyMPI
from petsc4py import PETSc

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
def lgreen(s):  return f"{C.LIGHT_GREEN}{s}{C.RESET}"


def strip_ansi(s):
    import re
    return re.sub(r"\033\[[0-9;]*m", "", s)


def cell(text, width, align="right"):
    """Pad printable width while ignoring ANSI escape sequences."""
    visible = len(strip_ansi(text))
    pad = max(width - visible, 0)
    if align == "left":
        return text + " " * pad
    return " " * pad + text


def fmt_norm(val, tol=None):
    s = f"{val:.6e}"
    if tol is None:
        return cyan(s)
    if val < tol:
        return green(s)
    if val < tol * 1e2:
        return yellow(s)
    return red(s)


def get_gpu_info():
    import subprocess
    from collections import Counter
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
        description="Blanken: 3-D Mantle-Convection Solver (FEniCS + PETSc)"
    )
    parser.add_argument("--nx", type=int, default=NX,
                        help=f"Mesh cells in x (default: {NX})")
    parser.add_argument("--ny", type=int, default=NY,
                        help=f"Mesh cells in y (default: {NY})")
    parser.add_argument("--nz", type=int, default=NZ,
                        help=f"Mesh cells in z (default: {NZ})")
    parser.add_argument("--Ra", type=float, default=RA,
                        help=f"Rayleigh number (default: {RA:.0e})")
    parser.add_argument("--dt", type=float, default=DT,
                        help=f"Time step (default: {DT})")
    parser.add_argument("--steps", type=int, default=NSTEPS,
                        help=f"Number of time steps (default: {NSTEPS})")
    parser.add_argument("--uzawa-iter", type=int, default=UZAWA_MAX,
                        help=f"Max Uzawa-CG iterations (default: {UZAWA_MAX})")
    parser.add_argument("--uzawa-tol", type=float, default=UZAWA_TOL,
                        help=f"Uzawa divergence-free tolerance (default: {UZAWA_TOL:.0e})")
    parser.add_argument("--rtol", type=float, default=RTOL,
                        help=f"Krylov relative tolerance (default: {RTOL:.0e})")
    parser.add_argument("--atol", type=float, default=ATOL,
                        help=f"Krylov absolute tolerance (default: {ATOL:.0e})")
    parser.add_argument("--max-krylov", type=int, default=MAX_KRYLOV_IT,
                        help=f"Max Krylov iterations (default: {MAX_KRYLOV_IT})")
    parser.add_argument("--gmres-restart", type=int, default=GMRES_RESTART,
                        help=f"GMRES restart parameter (default: {GMRES_RESTART})")
    parser.add_argument("--output", type=str, default="./output/blanken_output.xdmf",
                        help="Output XDMF file path (default: ./output/blanken_output.xdmf)")
    parser.add_argument("--no-output", action="store_true", default=False,
                        help="Skip writing XDMF output")
    parser.add_argument("--monitor", action="store_true", default=False,
                        help="Print per-Uzawa-iteration convergence table")
    parser.add_argument("--quiet", action="store_true", default=False,
                        help="Suppress per-step table; print only summary")
    return parser.parse_args()


def epsilon(u):
    return sym(nabla_grad(u))


def build_mesh(comm, nx, ny, nz):
    return BoxMesh.create(
        comm,
        [Point(0.0, 0.0, 0.0), Point(1.0, ny / nx, 1.0)],
        [nx, ny, nz],
        CellType.Type.tetrahedron
    )


def build_spaces(mesh):
    P1  = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    B   = FiniteElement("Bubble",   mesh.ufl_cell(), mesh.topology().dim() + 1)
    P1B = VectorElement(NodalEnrichedElement(P1, B))
    P1_ = VectorElement("Lagrange", mesh.ufl_cell(), 1)

    V  = FunctionSpace(mesh, P1B)
    Q  = FunctionSpace(mesh, P1)
    Q0 = FunctionSpace(mesh, "DG", 0)
    return V, Q, Q0


def build_bcs(V, Q, ny, nx):
    Ly = ny / nx
    bcs_u = [
        DirichletBC(V.sub(0), Constant(0.0), "near(x[0], 0.0) or near(x[0], 1.0)"),
        DirichletBC(V.sub(1), Constant(0.0), f"near(x[1], 0.0) or near(x[1], {Ly})"),
        DirichletBC(V.sub(2), Constant(0.0), "near(x[2], 0.0) or near(x[2], 1.0)"),
    ]
    bcs_T = [
        DirichletBC(Q, Constant(0.0), "near(x[2], 1.0)"),
        DirichletBC(Q, Constant(1.0), "near(x[2], 0.0)"),
    ]
    return bcs_u, bcs_T


def build_stokes_solver(rtol, atol, max_it, gmres_restart):
    solver_s = PETScKrylovSolver("gmres", "hypre_amg")
    ksp = as_backend_type(solver_s).ksp()
    ksp.setGMRESRestart(gmres_restart)
    ksp.setTolerances(rtol=rtol, atol=atol, max_it=max_it)
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
    pc_q.setHYPREType("boomeramg")
    return solver_q


def build_advection_solver(rtol, atol, max_it):
    solver_ad = PETScKrylovSolver("gmres", "hypre_amg")
    solver_ad.ksp().setGMRESRestart(400)
    solver_ad.parameters["maximum_iterations"] = max_it
    solver_ad.parameters["absolute_tolerance"] = atol
    solver_ad.parameters["relative_tolerance"] = rtol
    solver_ad.parameters["error_on_nonconvergence"] = False
    solver_ad.parameters["monitor_convergence"] = False
    return solver_ad


def stokes_uzawa_solve(
    solver_s, ksp, solver_q,
    G_, D_, M_,
    u_, u_n, p_, p_n, qk, dk, hk,
    Ra, g, T_old, bcs_u,
    v_test, q_test,
    uzawa_max, uzawa_tol, monitor, rank
):
    """
    Uzawa-CG pressure correction loop.
    A_stokes must already be assembled and set as solver_s operator before calling.

    Returns
    -------
    uzawa_it   : int    number of Uzawa iterations taken
    gmres_total: int    total GMRES iterations across all Stokes solves
    wall_time  : float  total wall-clock time in seconds
    """
    t0 = time.time()
    gmres_total = 0
    uzawa_it = 0

    p_.interpolate(Expression("0.0", degree=4))
    p_n.interpolate(Expression("0.0", degree=4))

    b = assemble(inner(Ra * T_old * g - grad(p_n), v_test) * dx)
    for bc in bcs_u:
        bc.apply(b)
    solver_s.solve(u_n.vector(), b)

    bq = assemble(div(u_n) * q_test * dx)
    solver_q.solve(qk.vector(), bq)
    dk.vector().zero()
    dk.vector().axpy(-1.0, qk.vector())

    for ii in range(uzawa_max):
        uzawa_it += 1

        bs = G_ * dk.vector()
        for bc in bcs_u:
            bc.apply(bs)
        solver_s.solve(hk.vector(), bs)
        gmres_total += ksp.getIterationNumber()

        num   = qk.vector().inner(M_ * qk.vector())
        denom = (G_ * dk.vector()).inner(hk.vector())
        ak    = num / denom

        p_.vector().zero()
        p_.vector().axpy(1.0, p_n.vector())
        p_.vector().axpy(ak, dk.vector())

        u_.vector().zero()
        u_.vector().axpy(1.0, u_n.vector())
        u_.vector().axpy(-ak, hk.vector())

        bq = D_ * u_.vector()

        p_n.vector().zero()
        p_n.vector().axpy(1.0, p_.vector())
        u_n.vector().zero()
        u_n.vector().axpy(1.0, u_.vector())

        div_norm = np.sqrt(bq.inner(bq)) / max(np.sqrt(u_.vector().inner(u_.vector())), 1e-300)

        if monitor and rank == 0:
            print(f"    {dim(f'uzawa {ii+1:4d}')}  "
                  f"div_norm = {fmt_norm(div_norm, uzawa_tol)}", flush=True)

        if (ii + 1) >= 3 and div_norm < uzawa_tol:
            break

        solver_q.solve(qk.vector(), bq)

        new_num = assemble(inner(qk, qk) * dx)
        bk = new_num / num

        temp = dk.vector().copy()
        dk.vector().zero()
        dk.vector().axpy(-1.0, qk.vector())
        dk.vector().axpy(bk, temp)

    u_.vector().update_ghost_values()
    return uzawa_it, gmres_total, time.time() - t0


def advection_diffusion_step(solver_ad, velocity, T_old, T_new, bcs_T, dt, phi, psi):
    """
    Crank-Nicolson advection-diffusion step for temperature.

    Returns
    -------
    wall_time : float  wall-clock time in seconds
    """
    t0 = time.time()

    T_old.vector().update_ghost_values()
    velocity.vector().update_ghost_values()

    aa = (phi / Constant(dt)) * psi * dx \
       + Constant(0.5) * dot(grad(phi), grad(psi)) * dx \
       + Constant(0.5) * inner(velocity, grad(phi)) * psi * dx

    La = (T_old / Constant(dt)) * psi * dx \
       - Constant(0.5) * dot(grad(T_old), grad(psi)) * dx \
       - Constant(0.5) * inner(velocity, grad(T_old)) * psi * dx

    A_a = assemble(aa)
    b_a = assemble(La)
    for bc_T in bcs_T:
        bc_T.apply(A_a, b_a)

    solver_ad.solve(A_a, T_new.vector(), b_a)
    return time.time() - t0


W_STEP   = 5
W_TIME   = 10
W_VRMS   = 14
W_STOKES = 9
W_ADV    = 9
W_UZ     = 6
W_GMRES  = 7


def main():
    args = parse_args()

    set_log_active(False)
    parameters["ghost_mode"] = "shared_facet"

    comm = pyMPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        gpu_count, gpu_summary = get_gpu_info()
        KEY_W = 18
        TITLE = "  3-D Thermo-Mechanical Convection  "
        rows = [
            ("Mesh",          f"{args.nx} × {args.ny} × {args.nz}"),
            ("Ra",            f"{args.Ra:.2e}"),
            ("dt / steps",    f"{args.dt}  /  {args.steps}"),
            ("Uzawa",         f"max {args.uzawa_iter} iters  tol={args.uzawa_tol:.0e}"),
            ("Krylov tol",    f"rtol={args.rtol:.0e}  atol={args.atol:.0e}"),
            ("GMRES restart", str(args.gmres_restart)),
            ("Output",        "disabled" if args.no_output else args.output),
            ("GPUs",          f"{gpu_count}  {gpu_summary}" if gpu_count else "none detected"),
        ]
        row_vis = lambda k, v: len(f"  {k}{' ' * (KEY_W - len(k))}{v}") + 2
        W = max(len(TITLE), max(row_vis(k, v) for k, v in rows)) + 2

        def print_row(k, v, colored_v=None):
            colored_v = colored_v or v
            line = f"  {yellow(k)}{' ' * (KEY_W - len(k))}{C.RESET}{colored_v}"
            pad = W - len(strip_ansi(line))
            print(b(lgreen("║")) + line + " " * max(pad, 0) + b(lgreen("║")))

        print()
        print(b(lgreen("╔" + "═" * W + "╗")))
        print(b(lgreen("║")) + b(f"{TITLE:^{W}}") + b(lgreen("║")))
        print(b(lgreen("╠" + "═" * W + "╣")))
        print_row("Mesh",          rows[0][1])
        print_row("Ra",            rows[1][1])
        print_row("dt / steps",    rows[2][1])
        print_row("Uzawa",         rows[3][1])
        print_row("Krylov tol",    rows[4][1])
        print_row("GMRES restart", rows[5][1])
        print_row("Output",        rows[6][1])
        if gpu_count:
            print_row("GPUs", rows[7][1], f"{green(str(gpu_count))}  {dim(gpu_summary)}")
        else:
            print_row("GPUs", rows[7][1], yellow("none detected"))
        print(b(lgreen("╚" + "═" * W + "╝")))
        print()

    t_start = time.time()

    if rank == 0:
        print(f"  {dim('Building mesh and spaces...')}", flush=True)

    mesh = build_mesh(comm, args.nx, args.ny, args.nz)
    V, Q, Q0 = build_spaces(mesh)

    u_  = Function(V);  u_n = Function(V)
    p_  = Function(Q);  p_n = Function(Q)
    qk  = Function(Q);  dk  = Function(Q)
    hk  = Function(V)
    T_old = Function(Q);  T_new = Function(Q)

    T_init = Expression(
        "1.0 - x[2] + alpha * cos(pi * x[0]) * sin(pi * x[2])",
        alpha=ALPHA_INIT, degree=2
    )
    T_old.interpolate(T_init)

    bcs_u, bcs_T = build_bcs(V, Q, args.ny, args.nx)

    g = Constant((0.0, 0.0, 1.0))
    Ra = Constant(args.Ra)

    if rank == 0:
        print(f"  {dim('Assembling static matrices...')}", flush=True)

    u,   v   = TrialFunction(V), TestFunction(V)
    phi, psi = TrialFunction(Q), TestFunction(Q)

    t0 = time.time()
    G_       = assemble(inner(grad(phi), v)   * dx)
    D_       = assemble(div(u) * psi          * dx)
    M_       = assemble(phi * psi             * dx)
    M_2      = assemble(inner(u, v)           * dx)
    A_stokes = assemble(inner(2.0 * epsilon(u), epsilon(v)) * dx)
    area     = assemble(Constant(1.0) * dx(domain=p_.ufl_domain()))
    t_assemble = time.time() - t0

    for bc in bcs_u:
        bc.apply(A_stokes)

    if rank == 0:
        print(f"  {dim('Assembly done')}  {cyan(f'{t_assemble:.2f}s')}\n")

    solver_s, ksp = build_stokes_solver(args.rtol, args.atol, args.max_krylov, args.gmres_restart)
    solver_s.set_operator(A_stokes)
    solver_q = build_mass_solver(M_, args.rtol, args.atol, args.max_krylov)
    solver_ad = build_advection_solver(args.rtol, args.atol, args.max_krylov)

    if rank == 0 and not args.quiet:
        hdr = (
            "  "
            + b(cyan(cell("step",    W_STEP,  "right"))) + " │ "
            + b(cyan(cell("t",       W_TIME,  "right"))) + " │ "
            + b(cyan(cell("v_rms",   W_VRMS,  "right"))) + " │ "
            + b(cyan(cell("t_stokes",W_STOKES,"right"))) + " │ "
            + b(cyan(cell("t_adv",   W_ADV,   "right"))) + " │ "
            + b(cyan(cell("uzawa",   W_UZ,    "right"))) + " │ "
            + b(cyan(cell("GMRES",   W_GMRES, "right")))
        )
        sep = (
            "  "
            + "─" * W_STEP  + "─┼─"
            + "─" * W_TIME  + "─┼─"
            + "─" * W_VRMS  + "─┼─"
            + "─" * W_STOKES + "─┼─"
            + "─" * W_ADV   + "─┼─"
            + "─" * W_UZ    + "─┼─"
            + "─" * W_GMRES
        )
        print(hdr)
        print(dim(sep))

    t = 0.0
    v_rms_list = []
    total_stokes = 0.0
    total_adv    = 0.0

    for i in range(args.steps):
        uz_it, gmres_it, t_stokes = stokes_uzawa_solve(
            solver_s, ksp, solver_q,
            G_, D_, M_,
            u_, u_n, p_, p_n, qk, dk, hk,
            Ra, g, T_old, bcs_u,
            v, psi,
            args.uzawa_iter, args.uzawa_tol,
            args.monitor, rank
        )
        total_stokes += t_stokes

        rms_vel = float(np.sqrt(np.abs(
            u_n.vector().inner(M_2 * u_.vector()) / area
        )))
        v_rms_list.append(rms_vel)

        t_adv = advection_diffusion_step(
            solver_ad, u_, T_old, T_new, bcs_T, args.dt, phi, psi
        )
        total_adv += t_adv

        T_old.vector().zero()
        T_old.vector().axpy(1.0, T_new.vector())
        t += args.dt

        if rank == 0 and not args.quiet:
            step_s  = cell(b(f"{i+1}"),            W_STEP,   "right")
            time_s  = cell(f"{t:.5f}",              W_TIME,   "right")
            vrms_s  = cell(cyan(f"{rms_vel:.6e}"),  W_VRMS,   "right")
            stok_s  = cell(f"{t_stokes:.2f}s",      W_STOKES, "right")
            adv_s   = cell(f"{t_adv:.2f}s",         W_ADV,    "right")
            uz_s    = cell(f"{uz_it}",               W_UZ,     "right")
            gmres_s = cell(f"{gmres_it}",            W_GMRES,  "right")
            print(f"  {step_s} │ {time_s} │ {vrms_s} │ {stok_s} │ {adv_s} │ {uz_s} │ {gmres_s}")

    t_total = time.time() - t_start

    if rank == 0:
        print()
        print(dim("  " + "─" * 58))
        print(f"  {b('Steps completed')}  {args.steps}")
        if v_rms_list:
            print(f"  {b('Final v_rms')}      {cyan(f'{v_rms_list[-1]:.6e}')}")
        print(f"  {dim('Stokes total')}     {total_stokes:.2f}s")
        print(f"  {dim('Advect total')}     {total_adv:.2f}s")
        print(f"  {dim('Wall total')}       {t_total:.2f}s")
        print(dim("  " + "─" * 58))
        print()

    if not args.no_output:
        out_dir = os.path.dirname(args.output)
        if out_dir and rank == 0:
            os.makedirs(out_dir, exist_ok=True)
        comm.Barrier()
        if rank == 0:
            print(f"  {dim('Writing')} {args.output} ...", flush=True)
        xdmf = XDMFFile(mesh.mpi_comm(), args.output)
        xdmf.parameters["flush_output"] = True
        xdmf.write(mesh)
        xdmf.write(u_,    float(t))
        xdmf.write(p_,    float(t))
        xdmf.write(T_old, float(t))
        if rank == 0:
            print(f"  {green('✓')} Output written.")


if __name__ == "__main__":
    main()