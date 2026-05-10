"""
Van Keken: 2-D compositional buoyancy benchmark (Stokes + level-set advection).

Stokes equation is solved with an Uzawa-CG pressure correction loop.
The level-set field is advanced with a stabilized Crank-Nicolson advection step.

Usage:
    python vankeken.py [options]

Example:
    python vankeken.py --monitor
    python vankeken.py --nx 256 --ny 256 --dt 0.5 --final-time 500
    mpirun -n 4 python vankeken.py --nx 512 --ny 512 --quiet
"""

WIDTH         = 0.9142
HEIGHT        = 1.0
NX            = 512
NY            = 512
DT            = 0.5
FINAL_TIME    = 1500.0
MAX_STEPS     = 10000000
UZAWA_MAX     = 2000
UZAWA_TOL     = 1e-4
RTOL          = 1e-7
ATOL          = 1e-10
MAX_KRYLOV_IT = 500
GMRES_RESTART = 400
INTERFACE_AMP = 0.02
INTERFACE_Y0  = 0.2

import argparse
import gc
import logging
import os
import sys
import time

logging.getLogger("FFC").setLevel(logging.WARNING)
logging.getLogger("UFL").setLevel(logging.WARNING)

import numpy as np
from dolfin import *
from mpi4py import MPI as pyMPI


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    LIGHT_GREEN = "\033[92m"


def b(text):
    return f"{C.BOLD}{text}{C.RESET}"


def dim(text):
    return f"{C.DIM}{text}{C.RESET}"


def cyan(text):
    return f"{C.CYAN}{text}{C.RESET}"


def green(text):
    return f"{C.GREEN}{text}{C.RESET}"


def yellow(text):
    return f"{C.YELLOW}{text}{C.RESET}"


def red(text):
    return f"{C.RED}{text}{C.RESET}"


def lgreen(text):
    return f"{C.LIGHT_GREEN}{text}{C.RESET}"


def strip_ansi(text):
    import re

    return re.sub(r"\033\[[0-9;]*m", "", text)


def cell(text, width, align="right"):
    visible = len(strip_ansi(text))
    pad = max(width - visible, 0)
    if align == "left":
        return text + " " * pad
    return " " * pad + text


def fmt_norm(value, tol=None):
    text = f"{value:.6e}"
    if tol is None:
        return cyan(text)
    if value < tol:
        return green(text)
    if value < tol * 1e2:
        return yellow(text)
    return red(text)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Van Keken benchmark: 2-D Stokes + level-set advection (FEniCS + PETSc)"
    )
    parser.add_argument("--width", type=float, default=WIDTH,
                        help=f"Domain width (default: {WIDTH})")
    parser.add_argument("--height", type=float, default=HEIGHT,
                        help=f"Domain height (default: {HEIGHT})")
    parser.add_argument("--nx", type=int, default=NX,
                        help=f"Mesh cells in x (default: {NX})")
    parser.add_argument("--ny", type=int, default=NY,
                        help=f"Mesh cells in y (default: {NY})")
    parser.add_argument("--dt", type=float, default=DT,
                        help=f"Time step (default: {DT})")
    parser.add_argument("--final-time", type=float, default=FINAL_TIME,
                        help=f"Stop time (default: {FINAL_TIME})")
    parser.add_argument("--steps", type=int, default=MAX_STEPS,
                        help=f"Safety cap on time steps (default: {MAX_STEPS})")
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
    parser.add_argument("--interface-amplitude", type=float, default=INTERFACE_AMP,
                        help=f"Initial interface cosine amplitude (default: {INTERFACE_AMP})")
    parser.add_argument("--interface-offset", type=float, default=INTERFACE_Y0,
                        help=f"Initial interface offset from bottom (default: {INTERFACE_Y0})")
    parser.add_argument("--output", type=str, default="./output/vankeken_output.xdmf",
                        help="Output XDMF file path (default: ./output/vankeken_output.xdmf)")
    parser.add_argument("--no-output", action="store_true", default=False,
                        help="Skip writing XDMF output")
    parser.add_argument("--monitor", action="store_true", default=False,
                        help="Print per-Uzawa-iteration convergence table")
    parser.add_argument("--quiet", action="store_true", default=False,
                        help="Suppress per-step table; print only summary")
    return parser.parse_args()


def epsilon(u):
    return sym(nabla_grad(u))


def build_mesh(comm, nx, ny, width, height):
    return RectangleMesh.create(
        comm,
        [Point(0.0, 0.0), Point(width, height)],
        [nx, ny],
        CellType.Type.triangle,
        "left/right",
    )


def build_spaces(mesh):
    p2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    p1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    velocity_space = FunctionSpace(mesh, p2)
    scalar_space = FunctionSpace(mesh, p1)
    return velocity_space, scalar_space


def build_bcs(velocity_space, width, height):
    return [
        DirichletBC(
            velocity_space.sub(0),
            Constant(0.0),
            f"near(x[0], 0.0) or near(x[0], {width})",
        ),
        DirichletBC(
            velocity_space,
            Constant((0.0, 0.0)),
            f"near(x[1], 0.0) or near(x[1], {height})",
        ),
    ]


def build_operators(velocity_space, scalar_space, bcs_u):
    u_trial = TrialFunction(velocity_space)
    v_test = TestFunction(velocity_space)
    phi_trial = TrialFunction(scalar_space)
    psi_test = TestFunction(scalar_space)

    gradient_op = assemble(inner(grad(phi_trial), v_test) * dx)
    divergence_op = assemble(div(u_trial) * psi_test * dx)
    scalar_mass = assemble(phi_trial * psi_test * dx)
    velocity_mass = assemble(inner(u_trial, v_test) * dx)
    stokes_matrix = assemble(inner(2.0 * epsilon(u_trial), epsilon(v_test)) * dx)
    for bc_u in bcs_u:
        bc_u.apply(stokes_matrix)

    h = CellDiameter(velocity_space.mesh())

    return (
        gradient_op,
        divergence_op,
        scalar_mass,
        velocity_mass,
        stokes_matrix,
        h,
        v_test,
        psi_test,
        phi_trial,
    )


def build_stokes_solver(rtol, atol, max_it, gmres_restart):
    solver = PETScKrylovSolver("gmres", "hypre_amg")
    # Use a local ksp reference only for setup; do NOT return it separately.
    # Storing a long-lived petsc4py KSP wrapper alongside the FEniCS solver
    # causes a double-free at shutdown (both try to destroy the same PETSc KSP).
    _ksp = as_backend_type(solver).ksp()
    _ksp.setGMRESRestart(gmres_restart)
    _ksp.setTolerances(rtol=rtol, atol=atol, max_it=max_it)
    pc = _ksp.getPC()
    pc.setType("hypre")
    pc.setHYPREType("boomeramg")
    return solver


def build_mass_solver(scalar_mass, rtol, atol, max_it):
    solver = PETScKrylovSolver("cg")
    solver.set_operator(scalar_mass)
    solver.parameters["error_on_nonconvergence"] = False
    ksp = as_backend_type(solver).ksp()
    ksp.setTolerances(rtol=rtol, atol=atol, max_it=max_it)
    pc = ksp.getPC()
    pc.setType("hypre")
    pc.setHYPREType("boomeramg")
    return solver


def build_advection_solver(rtol, atol, max_it, gmres_restart):
    solver = PETScKrylovSolver("gmres", "hypre_amg")
    solver.ksp().setGMRESRestart(gmres_restart)
    solver.parameters["maximum_iterations"] = max_it
    solver.parameters["absolute_tolerance"] = atol
    solver.parameters["relative_tolerance"] = rtol
    solver.parameters["error_on_nonconvergence"] = False
    solver.parameters["monitor_convergence"] = False
    return solver


def initialize_state(velocity_space, scalar_space, width, amplitude, offset):
    u_ = Function(velocity_space)
    u_n = Function(velocity_space)
    p_ = Function(scalar_space)
    p_n = Function(scalar_space)
    qk = Function(scalar_space)
    dk = Function(scalar_space)
    hk = Function(velocity_space)
    rho = Function(scalar_space)
    phi = Function(scalar_space)
    phi_new = Function(scalar_space)

    phi_init = Expression(
        "x[1] - amp*cos(pi*x[0]/L) - y0",
        degree=2,
        pi=np.pi,
        amp=amplitude,
        y0=offset,
        L=width,
    )
    phi.interpolate(phi_init)
    level_set_filter(phi, rho)

    return u_, u_n, p_, p_n, qk, dk, hk, rho, phi, phi_new


def level_set_filter(phi, rho):
    phi_vec = phi.vector().get_local()
    rho_vec = np.where(phi_vec <= 0.0, 0.0, 1.0)
    rho.vector().set_local(rho_vec)
    rho.vector().apply("insert")
    rho.vector().update_ghost_values()
    return rho


def stokes_uzawa_solve(
    solver_s,
    solver_q,
    gradient_op,
    divergence_op,
    scalar_mass,
    velocity_mass,
    u_,
    u_n,
    p_,
    p_n,
    qk,
    dk,
    hk,
    rho,
    gravity,
    bcs_u,
    v_test,
    psi_test,
    uzawa_max,
    uzawa_tol,
    monitor,
    rank,
):
    t0 = time.time()
    gmres_total = 0
    uzawa_it = 0

    # Local ksp reference for iteration count; stays alive only for this call.
    # This avoids the double-free that occurs when a long-lived petsc4py KSP
    # wrapper coexists with the FEniCS solver that owns the same PETSc KSP.
    _ksp = as_backend_type(solver_s).ksp()

    p_.interpolate(Expression("0.0", degree=0))
    p_n.interpolate(Expression("0.0", degree=0))

    rhs = assemble(inner(rho * gravity - grad(p_n), v_test) * dx)
    for bc_u in bcs_u:
        bc_u.apply(rhs)
    solver_s.solve(u_n.vector(), rhs)
    gmres_total += _ksp.getIterationNumber()

    bq = assemble(div(u_n) * psi_test * dx)
    solver_q.solve(qk.vector(), bq)
    dk.vector().zero()
    dk.vector().axpy(-1.0, qk.vector())

    for ii in range(uzawa_max):
        uzawa_it += 1

        bs = gradient_op * dk.vector()
        for bc_u in bcs_u:
            bc_u.apply(bs)
        solver_s.solve(hk.vector(), bs)
        gmres_total += _ksp.getIterationNumber()

        num = qk.vector().inner(scalar_mass * qk.vector())
        denom = (gradient_op * dk.vector()).inner(hk.vector())
        if abs(denom) < 1e-30:
            raise RuntimeError("Uzawa breakdown: search direction denominator is zero.")

        alpha = num / denom

        p_.vector().zero()
        p_.vector().axpy(1.0, p_n.vector())
        p_.vector().axpy(alpha, dk.vector())

        u_.vector().zero()
        u_.vector().axpy(1.0, u_n.vector())
        u_.vector().axpy(-alpha, hk.vector())

        bq = divergence_op * u_.vector()

        p_n.vector().zero()
        p_n.vector().axpy(1.0, p_.vector())
        u_n.vector().zero()
        u_n.vector().axpy(1.0, u_.vector())

        div_norm = np.sqrt(max(bq.inner(bq), 0.0)) / max(
            np.sqrt(max(u_.vector().inner(velocity_mass * u_.vector()), 0.0)),
            1e-300,
        )

        if monitor and rank == 0:
            print(
                f"    {dim(f'uzawa {ii + 1:4d}')}  "
                f"div_norm = {fmt_norm(div_norm, uzawa_tol)}",
                flush=True,
            )

        if (ii + 1) >= 3 and div_norm < uzawa_tol:
            break

        solver_q.solve(qk.vector(), bq)

        new_num = qk.vector().inner(scalar_mass * qk.vector())
        beta = new_num / num

        tmp = dk.vector().copy()
        dk.vector().zero()
        dk.vector().axpy(-1.0, qk.vector())
        dk.vector().axpy(beta, tmp)

    u_.vector().update_ghost_values()
    p_.vector().update_ghost_values()
    return uzawa_it, gmres_total, time.time() - t0


def level_set_advection_step(
    h,
    solver_ad,
    velocity,
    phi_old,
    phi_new,
    dt,
    phi_trial,
    psi_test,
):
    t0 = time.time()

    velocity.vector().update_ghost_values()
    phi_old.vector().update_ghost_values()

    speed = sqrt(dot(velocity, velocity)) + Constant(1e-12)
    tau = h / (2.0 * speed)

    a_form = (phi_trial / Constant(dt)) * psi_test * dx
    a_form += Constant(0.5) * dot(velocity, grad(phi_trial)) * psi_test * dx
    a_form += tau * dot(velocity, grad(psi_test)) * (
        phi_trial / Constant(dt) + Constant(0.5) * dot(velocity, grad(phi_trial))
    ) * dx

    l_form = (phi_old / Constant(dt)) * psi_test * dx
    l_form += -Constant(0.5) * dot(velocity, grad(phi_old)) * psi_test * dx
    l_form += tau * dot(velocity, grad(psi_test)) * (
        phi_old / Constant(dt) - Constant(0.5) * dot(velocity, grad(phi_old))
    ) * dx

    A_ad = assemble(a_form)
    b_ad = assemble(l_form)

    solver_ad.set_operator(A_ad)
    solver_ad.solve(phi_new.vector(), b_ad)
    phi_new.vector().update_ghost_values()

    return time.time() - t0


W_STEP = 5
W_TIME = 10
W_VRMS = 14
W_STOKES = 9
W_ADV = 9
W_UZ = 6
W_GMRES = 7


def main():
    args = parse_args()

    if args.dt <= 0.0:
        raise ValueError("--dt must be positive.")
    if args.final_time < 0.0:
        raise ValueError("--final-time must be non-negative.")
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")

    set_log_active(False)
    parameters["ghost_mode"] = "shared_vertex"

    comm = pyMPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        key_width = 18
        title = "  2-D Rayliegh-Taylor instability Benchmark  "
        rows = [
            ("Mesh", f"{args.nx} x {args.ny}"),
            ("Domain", f"{args.width} x {args.height}"),
            ("dt / t_end", f"{args.dt}  /  {args.final_time}"),
            ("Step cap", str(args.steps)),
            ("Uzawa", f"max {args.uzawa_iter} iters  tol={args.uzawa_tol:.0e}"),
            ("Krylov tol", f"rtol={args.rtol:.0e}  atol={args.atol:.0e}"),
            ("GMRES restart", str(args.gmres_restart)),
            ("Output", "disabled" if args.no_output else args.output),
        ]
        row_vis = lambda key, value: len(f"  {key}{' ' * (key_width - len(key))}{value}") + 2
        width = max(len(title), max(row_vis(key, value) for key, value in rows)) + 2

        def print_row(key, value):
            line = f"  {yellow(key)}{' ' * (key_width - len(key))}{C.RESET}{value}"
            pad = width - len(strip_ansi(line))
            print(b(lgreen("║")) + line + " " * max(pad, 0) + b(lgreen("║")))

        print()
        print(b(lgreen("╔" + "═" * width + "╗")))
        print(b(lgreen("║")) + b(f"{title:^{width}}") + b(lgreen("║")))
        print(b(lgreen("╠" + "═" * width + "╣")))
        for key, value in rows:
            print_row(key, value)
        print(b(lgreen("╚" + "═" * width + "╝")))
        print()

    t_start = time.time()

    if rank == 0:
        print(f"  {dim('Building mesh and spaces...')}", flush=True)

    mesh = build_mesh(comm, args.nx, args.ny, args.width, args.height)
    velocity_space, scalar_space = build_spaces(mesh)
    bcs_u = build_bcs(velocity_space, args.width, args.height)
    (
        gradient_op,
        divergence_op,
        scalar_mass,
        velocity_mass,
        stokes_matrix,
        h,
        v_test,
        psi_test,
        phi_trial,
    ) = build_operators(velocity_space, scalar_space, bcs_u)
    area = assemble(Constant(1.0) * dx(domain=mesh))

    (
        u_,
        u_n,
        p_,
        p_n,
        qk,
        dk,
        hk,
        rho,
        phi,
        phi_new,
    ) = initialize_state(
        velocity_space,
        scalar_space,
        args.width,
        args.interface_amplitude,
        args.interface_offset,
    )

    solver_s = build_stokes_solver(
        args.rtol, args.atol, args.max_krylov, args.gmres_restart
    )
    solver_s.set_operator(stokes_matrix)
    solver_q = build_mass_solver(scalar_mass, args.rtol, args.atol, args.max_krylov)
    solver_ad = build_advection_solver(
        args.rtol, args.atol, args.max_krylov, args.gmres_restart
    )

    gravity = Constant((0.0, -1.0))

    if rank == 0 and not args.quiet:
        header = (
            "  "
            + b(cyan(cell("step", W_STEP))) + " | "
            + b(cyan(cell("t", W_TIME))) + " | "
            + b(cyan(cell("v_rms", W_VRMS))) + " | "
            + b(cyan(cell("t_stokes", W_STOKES))) + " | "
            + b(cyan(cell("t_adv", W_ADV))) + " | "
            + b(cyan(cell("uzawa", W_UZ))) + " | "
            + b(cyan(cell("GMRES", W_GMRES)))
        )
        sep = (
            "  "
            + "-" * W_STEP + "-+-"
            + "-" * W_TIME + "-+-"
            + "-" * W_VRMS + "-+-"
            + "-" * W_STOKES + "-+-"
            + "-" * W_ADV + "-+-"
            + "-" * W_UZ + "-+-"
            + "-" * W_GMRES
        )
        print(header)
        print(dim(sep))

    current_time = 0.0
    completed_steps = 0
    v_rms_list = []
    total_stokes = 0.0
    total_adv = 0.0

    if not args.no_output:
        out_dir = os.path.dirname(args.output)
        if out_dir and rank == 0:
            os.makedirs(out_dir, exist_ok=True)
        comm.Barrier()
        u_.rename("velocity", "velocity")
        p_.rename("pressure", "pressure")
        phi.rename("phi", "level_set")
        rho.rename("rho", "density")
        xdmf = XDMFFile(mesh.mpi_comm(), args.output)
        xdmf.parameters["flush_output"] = True
        xdmf.write(mesh)
    else:
        xdmf = None

    while completed_steps < args.steps and current_time < args.final_time - 1e-15:
        dt_step = min(args.dt, args.final_time - current_time)

        uzawa_it, gmres_it, t_stokes = stokes_uzawa_solve(
            solver_s,
            solver_q,
            gradient_op,
            divergence_op,
            scalar_mass,
            velocity_mass,
            u_,
            u_n,
            p_,
            p_n,
            qk,
            dk,
            hk,
            rho,
            gravity,
            bcs_u,
            v_test,
            psi_test,
            args.uzawa_iter,
            args.uzawa_tol,
            args.monitor,
            rank,
        )
        total_stokes += t_stokes

        rms_velocity = float(
            np.sqrt(max(abs(u_.vector().inner(velocity_mass * u_.vector()) / area), 0.0))
        )
        v_rms_list.append(rms_velocity)

        t_adv = level_set_advection_step(
            h,
            solver_ad,
            u_,
            phi,
            phi_new,
            dt_step,
            phi_trial,
            psi_test,
        )
        total_adv += t_adv

        phi.vector().zero()
        phi.vector().axpy(1.0, phi_new.vector())
        phi.vector().update_ghost_values()
        level_set_filter(phi, rho)

        current_time += dt_step
        completed_steps += 1

        if xdmf is not None and completed_steps % 100 == 0:
            xdmf.write(u_, float(current_time))
            xdmf.write(p_, float(current_time))
            xdmf.write(phi, float(current_time))
            xdmf.write(rho, float(current_time))
            if rank == 0 and not args.quiet:
                print(f"  {dim(f'[step {completed_steps}] snapshot saved')}", flush=True)

        if rank == 0 and not args.quiet:
            step_s = cell(b(f"{completed_steps}"), W_STEP)
            time_s = cell(f"{current_time:.5f}", W_TIME)
            vrms_s = cell(cyan(f"{rms_velocity:.6e}"), W_VRMS)
            stokes_s = cell(f"{t_stokes:.2f}s", W_STOKES)
            adv_s = cell(f"{t_adv:.2f}s", W_ADV)
            uzawa_s = cell(f"{uzawa_it}", W_UZ)
            gmres_s = cell(f"{gmres_it}", W_GMRES)
            print(
                f"  {step_s} | {time_s} | {vrms_s} | {stokes_s} | "
                f"{adv_s} | {uzawa_s} | {gmres_s}"
            )

    wall_time = time.time() - t_start

    if rank == 0:
        print()
        print(dim("  " + "-" * 58))
        print(f"  {b('Steps completed')}  {completed_steps}")
        print(f"  {b('Final time')}       {current_time:.5f}")
        if v_rms_list:
            print(f"  {b('Final v_rms')}      {cyan(f'{v_rms_list[-1]:.6e}')}")
        print(f"  {dim('Stokes total')}     {total_stokes:.2f}s")
        print(f"  {dim('Advect total')}     {total_adv:.2f}s")
        print(f"  {dim('Wall total')}       {wall_time:.2f}s")
        print(dim("  " + "-" * 58))
        print()

    if xdmf is not None:
        xdmf.close()
        if rank == 0:
            print(f"  {green('✓')} Output closed ({args.output}).")


    # Explicitly release PETSc-backed objects before finalization.
    del solver_s, solver_q, solver_ad
    del u_, u_n, p_, p_n, qk, dk, hk, rho, phi, phi_new
    del gradient_op, divergence_op, scalar_mass, velocity_mass, stokes_matrix
    del bcs_u, velocity_space, scalar_space, mesh
    gc.collect()

    # PETSc 3.24 compiled with CUDA support initialises a CUDA event-object
    # pool even on CPU-only runs (triggered by the HYPRE AMG preconditioner).
    # PetscFinalize() drains that pool via std::deque reallocation while the
    # CUDA runtime is already partially torn down, corrupting the heap and
    # aborting every MPI rank.
    #
    # Workaround: flush I/O, call MPI_Finalize explicitly via mpi4py, then
    # use os._exit(0) to bypass the atexit-registered PetscFinalize.
    # The OS reclaims all remaining resources cleanly.
    sys.stdout.flush()
    sys.stderr.flush()
    comm.Barrier()
    pyMPI.Finalize()
    os._exit(0)


if __name__ == "__main__":
    main()
