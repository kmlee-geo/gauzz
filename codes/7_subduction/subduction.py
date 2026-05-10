"""
2D Subduction benchmark (Stokes + level-set advection)
with nonlinear Picard iteration (Anderson acceleration).

Velocity: P2 (quadratic Lagrange) on triangles.
Pressure / level-set: CG1.
Material properties: DG0 projection from level-set fields.
Stokes: Uzawa-CG with variable viscosity.
Level-set: stabilized Crank-Nicolson advection.
Nonlinearity: von Mises yield in slab1 with Picard + Anderson.

Usage:
    mpirun -np 4 python subduction.py
    mpirun -np 4 python subduction.py --nx 600 --ny 100 --steps 100
    mpirun -np 4 python subduction.py --quiet
    mpirun -np 4 python subduction.py --monitor --cfl 0.3
"""

# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------
DOMAIN_WIDTH  = 6
NX            = 1200        # 300*4
NY            = 200         # 50*4
MAX_ETA       = 1000.
SLAB2_ETA     = 1000.
SLAB3_ETA     = 50.
OP_ETA        = 400.
UM_ETA        = 1.
SLAB_RHO      = 5.17e5
ELSE_RHO      = 0.
COHESION      = 3.1e4
DEEP_Y_THRESH = 0.7         # slab1 depth threshold for deep creep
C_CFL         = 0.4
DT_INIT       = 4.4e-06
MAX_PICARD    = 3
MAX_STEPS     = 3000
UZAWA_MAX     = 2000
UZAWA_TOL     = 1e-3
RTOL          = 1e-7
ATOL          = 1e-14
MAX_KRYLOV_IT = 500
GMRES_RESTART = 400

import argparse
import gc
import os
import sys
import time

import numpy as np
from dolfin import *
from mpi4py import MPI as pyMPI
from petsc4py import PETSc


# ---------------------------------------------------------------------------
# Terminal colour helpers
# ---------------------------------------------------------------------------
class C:
    RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
    CYAN = "\033[36m"; GREEN = "\033[32m"; YELLOW = "\033[33m"
    RED = "\033[31m"; LIGHT_GREEN = "\033[92m"


def b(t):      return f"{C.BOLD}{t}{C.RESET}"
def dim(t):    return f"{C.DIM}{t}{C.RESET}"
def cyan(t):   return f"{C.CYAN}{t}{C.RESET}"
def green(t):  return f"{C.GREEN}{t}{C.RESET}"
def yellow(t): return f"{C.YELLOW}{t}{C.RESET}"
def red(t):    return f"{C.RED}{t}{C.RESET}"
def lgreen(t): return f"{C.LIGHT_GREEN}{t}{C.RESET}"


def strip_ansi(text):
    import re
    return re.sub(r"\033\[[0-9;]*m", "", text)


def cell(text, width, align="right"):
    visible = len(strip_ansi(text))
    pad = max(width - visible, 0)
    return text + " " * pad if align == "left" else " " * pad + text


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="2D Subduction: Stokes + level-set advection + Picard iteration"
    )
    ap.add_argument("--nx",         type=int,   default=NX)
    ap.add_argument("--ny",         type=int,   default=NY)
    ap.add_argument("--steps",      type=int,   default=MAX_STEPS)
    ap.add_argument("--picard",     type=int,   default=MAX_PICARD)
    ap.add_argument("--cfl",        type=float, default=C_CFL)
    ap.add_argument("--dt",         type=float, default=DT_INIT)
    ap.add_argument("--uzawa-iter", type=int,   default=UZAWA_MAX)
    ap.add_argument("--uzawa-tol",  type=float, default=UZAWA_TOL)
    ap.add_argument("--rtol",       type=float, default=RTOL)
    ap.add_argument("--atol",       type=float, default=ATOL)
    ap.add_argument("--max-krylov", type=int,   default=MAX_KRYLOV_IT)
    ap.add_argument("--gmres-restart", type=int, default=GMRES_RESTART)
    ap.add_argument("--output",     type=str,   default="./results/subduction.xdmf")
    ap.add_argument("--no-output",  action="store_true", default=False)
    ap.add_argument("--save-every", type=int,   default=10)
    ap.add_argument("--monitor",    action="store_true", default=False,
                    help="Print per-Uzawa-iteration convergence")
    ap.add_argument("--quiet",      action="store_true", default=False)
    ap.add_argument("--level-set-dir", type=str, default="./subduction_lv_set")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Mesh & function spaces
# ---------------------------------------------------------------------------
def build_mesh(comm, nx, ny):
    return RectangleMesh.create(
        comm,
        [Point(0.0, 0.0), Point(DOMAIN_WIDTH, 1.0)],
        [nx, ny],
        CellType.Type.triangle,
        "right",
    )


def build_spaces(mesh):
    V  = FunctionSpace(mesh, VectorElement("Lagrange", mesh.ufl_cell(), 2))
    Q  = FunctionSpace(mesh, FiniteElement("Lagrange", mesh.ufl_cell(), 1))
    Q0 = FunctionSpace(mesh, "DG", 0)
    return V, Q, Q0


def build_bcs(V):
    return [
        DirichletBC(V.sub(0), Constant(0.0),
                    f"near(x[0], 0.0) or near(x[0], {DOMAIN_WIDTH})"),
        DirichletBC(V.sub(1), Constant(0.0),
                    "near(x[1], 0.0) or near(x[1], 1.0)"),
    ]


# ---------------------------------------------------------------------------
# Operator assembly & work vectors
# ---------------------------------------------------------------------------
def build_operators(V, Q, Q0):
    u_trial = TrialFunction(V);  v_test = TestFunction(V)
    p_trial = TrialFunction(Q);  q_test = TestFunction(Q)
    p0_trial = TrialFunction(Q0); q0_test = TestFunction(Q0)

    G_  = assemble(inner(grad(p_trial), v_test) * dx)
    D_  = assemble(div(u_trial) * q_test * dx)
    M_  = assemble(p_trial * q_test * dx)
    Mv  = assemble(inner(u_trial, v_test) * dx)
    A0  = assemble(p0_trial * q0_test * dx)

    # PETSc mat handles for in-place mat-vec (no temp allocation)
    pG  = as_backend_type(G_).mat()
    pD  = as_backend_type(D_).mat()
    pM  = as_backend_type(M_).mat()
    pMv = as_backend_type(Mv).mat()

    # Pre-allocate work vectors (avoid per-iteration GPU allocation)
    uz_bs   = Function(V).vector()   # G_*dk with BCs applied
    uz_gdk  = Function(V).vector()   # G_*dk raw (cached for denom)
    uz_bq   = Function(Q).vector()   # D_*u_
    uz_bw   = Function(Q).vector()   # M_*qk
    uz_Mwk  = Function(Q).vector()   # M_*wk
    uz_tmpd = Function(Q).vector()   # dk copy for update
    uz_Mvu  = Function(V).vector()   # Mv*u_ for vel_norm
    uz_Mqk  = Function(Q).vector()   # M_*qk for div_norm
    pic_tmp = Function(V).vector()   # Picard work vector

    work = dict(
        uz_bs=uz_bs, uz_gdk=uz_gdk, uz_bq=uz_bq, uz_bw=uz_bw,
        uz_Mwk=uz_Mwk, uz_tmpd=uz_tmpd, uz_Mvu=uz_Mvu, uz_Mqk=uz_Mqk,
        pic_tmp=pic_tmp,
    )
    pmats = dict(pG=pG, pD=pD, pM=pM, pMv=pMv)

    return (G_, D_, M_, Mv, A0,
            u_trial, v_test, p_trial, q_test, q0_test,
            work, pmats)


def matvec(pmat, x, y):
    """y = pmat * x  (in-place, no allocation)."""
    pmat.mult(as_backend_type(x).vec(), as_backend_type(y).vec())


# ---------------------------------------------------------------------------
# Solver construction
# ---------------------------------------------------------------------------
def build_solvers(M_, rtol, atol, max_it, gmres_restart):
    solver_s = PETScKrylovSolver("gmres", "hypre_amg")
    ksp_s = as_backend_type(solver_s).ksp()
    ksp_s.setGMRESRestart(gmres_restart)
    ksp_s.setTolerances(rtol=rtol, atol=atol, max_it=max_it)

    solver_q = PETScKrylovSolver("cg", "hypre_amg")
    solver_q.set_operator(M_)
    solver_q.parameters["error_on_nonconvergence"] = False
    ksp_q = as_backend_type(solver_q).ksp()
    ksp_q.setTolerances(rtol=1e-10, atol=1e-12, max_it=max_it)
    ksp_q.setFromOptions()
    ksp_q.getPC().setType("hypre")
    ksp_q.getPC().setHYPREType("boomeramg")

    solver_w = PETScKrylovSolver("cg", "hypre_amg")
    solver_w.parameters["error_on_nonconvergence"] = False
    ksp_w = as_backend_type(solver_w).ksp()
    ksp_w.setTolerances(rtol=1e-10, atol=1e-12, max_it=max_it)
    ksp_w.getPC().setType("hypre")
    ksp_w.getPC().setHYPREType("boomeramg")

    solver_q0 = PETScKrylovSolver("cg", "jacobi")

    solver_ad = PETScKrylovSolver("gmres", "hypre_amg")
    ksp_ad = as_backend_type(solver_ad).ksp()
    ksp_ad.setGMRESRestart(gmres_restart)
    ksp_ad.setTolerances(rtol=1e-10, atol=1e-12, max_it=max_it)
    solver_ad.parameters["error_on_nonconvergence"] = False

    return solver_s, solver_q, solver_w, solver_q0, solver_ad


# ---------------------------------------------------------------------------
# State initialisation
# ---------------------------------------------------------------------------
def initialize_state(V, Q, Q0, level_set_dir, mesh):
    u_  = Function(V);  u_n = Function(V)
    p_  = Function(Q);  p_n = Function(Q)
    qk  = Function(Q);  wk  = Function(Q)
    dk  = Function(Q);  hk  = Function(V)
    eta = Function(Q0);  eta.vector()[:] = 1
    rho = Function(Q0)
    eps_2nd = Function(Q0)

    # Level-set fields (CG1) and their DG0 projections
    names = ["phi_op", "phi_slab1", "phi_slab2", "phi_slab3"]
    phis = {n: Function(Q) for n in names}
    phis_n = {n: Function(Q) for n in names}
    phis_0 = {n: Function(Q0) for n in names}

    for name in names:
        path = os.path.join(level_set_dir, f"{name}.h5")
        with HDF5File(mesh.mpi_comm(), path, "r") as h5:
            h5.read(phis[name], "/u")

    # Picard iteration state
    u_pic_n = Function(V)
    r_k     = Function(V)
    r_k_n   = Function(V)
    delta_r = Function(V)

    state = dict(
        u_=u_, u_n=u_n, p_=p_, p_n=p_n,
        qk=qk, wk=wk, dk=dk, hk=hk,
        eta=eta, rho=rho, eps_2nd=eps_2nd,
        phis=phis, phis_n=phis_n, phis_0=phis_0,
        u_pic_n=u_pic_n, r_k=r_k, r_k_n=r_k_n, delta_r=delta_r,
    )
    return state


# ---------------------------------------------------------------------------
# Material update (eta, rho from level-set fields)
# ---------------------------------------------------------------------------
def update_eta_rho(eta, rho, phis_0, eps_2nd):
    eta_arr = eta.vector().get_local()
    rho_arr = rho.vector().get_local()
    eps_arr = eps_2nd.vector().get_local()
    op_arr    = phis_0["phi_op"].vector().get_local()
    slab1_arr = phis_0["phi_slab1"].vector().get_local()
    slab2_arr = phis_0["phi_slab2"].vector().get_local()
    slab3_arr = phis_0["phi_slab3"].vector().get_local()

    # Base: upper mantle
    eta_arr[:] = UM_ETA
    rho_arr[:] = ELSE_RHO

    # Slab3 (weak lower slab)
    m = slab3_arr <= 0.0
    eta_arr[m] = SLAB3_ETA;  rho_arr[m] = SLAB_RHO

    # Slab2 (strong lower slab)
    m = slab2_arr <= 0.0
    eta_arr[m] = SLAB2_ETA;  rho_arr[m] = SLAB_RHO

    # Slab1 (von Mises yield)
    m = slab1_arr <= 0.0
    vm = 0.5 * COHESION / (eps_arr[m] + 1e-19)
    eta_s1 = np.maximum(1.0, np.minimum(vm, MAX_ETA))
    coords = eta.function_space().tabulate_dof_coordinates()
    eta_s1[coords[:, 1][m] < DEEP_Y_THRESH] = SLAB3_ETA
    eta_arr[m] = eta_s1;  rho_arr[m] = SLAB_RHO

    # Overriding plate
    m = op_arr <= 0.0
    eta_arr[m] = OP_ETA;  rho_arr[m] = ELSE_RHO

    eta.vector().set_local(eta_arr);  eta.vector().apply("insert")
    rho.vector().set_local(rho_arr);  rho.vector().apply("insert")


# ---------------------------------------------------------------------------
# DG0 projection of CG1 level-set
# ---------------------------------------------------------------------------
def project_level_sets(A0, solver_q0, phis, phis_0, q0_test):
    for name in phis:
        b0 = assemble(phis[name] * q0_test * dx)
        solver_q0.solve(phis_0[name].vector(), b0)


# ---------------------------------------------------------------------------
# Stokes Uzawa-CG solver
# ---------------------------------------------------------------------------
def stokes_uzawa_solve(
    state, solver_s, solver_q, solver_w,
    G_, M_, bcs, pmats, work,
    u_trial, v_test, p_trial, q_test,
    uzawa_max, uzawa_tol, monitor, rank,
):
    t0 = time.time()
    u_  = state["u_"];   u_n = state["u_n"]
    p_  = state["p_"];   p_n = state["p_n"]
    qk  = state["qk"];   wk = state["wk"]
    dk  = state["dk"];   hk  = state["hk"]
    eta = state["eta"];  rho = state["rho"]

    g = Constant((0.0, -1.0))

    def eps(u_): return sym(nabla_grad(u_))

    A = assemble(inner(2.0 * eta * eps(u_trial), eps(v_test)) * dx)
    b = assemble(inner((rho * g - grad(p_n)), v_test) * dx)
    for bc in bcs: bc.apply(A, b)
    solver_s.set_operator(A)
    solver_s.solve(u_n.vector(), b)

    matvec(pmats["pD"], u_n.vector(), work["uz_bq"])
    solver_q.solve(qk.vector(), work["uz_bq"])

    Aw = assemble((1.0 / eta) * p_trial * q_test * dx)
    bw = assemble(qk * q_test * dx)
    solver_w.set_operator(Aw)
    solver_w.solve(wk.vector(), bw)
    dk.vector().zero()
    dk.vector().axpy(-1.0, wk.vector())

    uzawa_it = 0
    for ii in range(uzawa_max):
        uzawa_it += 1

        matvec(pmats["pG"], dk.vector(), work["uz_gdk"])
        work["uz_bs"].zero()
        work["uz_bs"].axpy(1.0, work["uz_gdk"])
        for bc in bcs: bc.apply(work["uz_bs"])

        solver_s.solve(hk.vector(), work["uz_bs"])

        matvec(pmats["pM"], wk.vector(), work["uz_Mwk"])
        num   = qk.vector().inner(work["uz_Mwk"])
        denom = work["uz_gdk"].inner(hk.vector())
        if abs(denom) < 1e-30:
            break
        ak = num / denom

        p_.vector().zero()
        p_.vector().axpy(1.0, p_n.vector())
        p_.vector().axpy(ak, dk.vector())

        u_.vector().zero()
        u_.vector().axpy(1.0, u_n.vector())
        u_.vector().axpy(-ak, hk.vector())

        matvec(pmats["pD"], u_.vector(), work["uz_bq"])
        p_n.vector().zero(); p_n.vector().axpy(1.0, p_.vector())
        u_n.vector().zero(); u_n.vector().axpy(1.0, u_.vector())

        solver_q.solve(qk.vector(), work["uz_bq"])
        matvec(pmats["pM"], qk.vector(), work["uz_Mqk"])
        matvec(pmats["pMv"], u_.vector(), work["uz_Mvu"])
        div_norm = np.sqrt(max(qk.vector().inner(work["uz_Mqk"]), 0.0))
        vel_norm = np.sqrt(max(u_.vector().inner(work["uz_Mvu"]), 1e-300))

        if monitor and rank == 0:
            print(f"    {dim(f'uzawa {ii+1:4d}')}  "
                  f"div/vel = {cyan(f'{div_norm/vel_norm:.6e}')}", flush=True)

        if div_norm / vel_norm < uzawa_tol:
            break

        matvec(pmats["pM"], qk.vector(), work["uz_bw"])
        solver_w.solve(wk.vector(), work["uz_bw"])

        matvec(pmats["pM"], wk.vector(), work["uz_Mwk"])
        new_num = qk.vector().inner(work["uz_Mwk"])
        bk = new_num / num

        work["uz_tmpd"].zero()
        work["uz_tmpd"].axpy(1.0, dk.vector())
        dk.vector().zero()
        dk.vector().axpy(-1.0, wk.vector())
        dk.vector().axpy(bk, work["uz_tmpd"])

    return uzawa_it, time.time() - t0


# ---------------------------------------------------------------------------
# Level-set advection (Crank-Nicolson + SUPG)
# ---------------------------------------------------------------------------
def level_set_advection_step(h, solver_ad, velocity, phi_old, phi_new,
                              dt_val, p_trial, q_test):
    u_mag = sqrt(dot(velocity, velocity)) + Constant(1e-12)
    tau   = h / (2.0 * u_mag)
    dt_c  = Constant(dt_val)

    a  = (p_trial / dt_c) * q_test * dx
    a += Constant(0.5) * dot(velocity, grad(p_trial)) * q_test * dx
    a += tau * dot(velocity, grad(q_test)) * (
         p_trial / dt_c + Constant(0.5) * dot(velocity, grad(p_trial))) * dx

    L  = (phi_old / dt_c) * q_test * dx
    L -= Constant(0.5) * dot(velocity, grad(phi_old)) * q_test * dx
    L += tau * dot(velocity, grad(q_test)) * (
         phi_old / dt_c - Constant(0.5) * dot(velocity, grad(phi_old))) * dx

    A_a = assemble(a)
    b_a = assemble(L)
    solver_ad.solve(A_a, phi_new.vector(), b_a)


# ---------------------------------------------------------------------------
# Strain rate 2nd invariant (DG0)
# ---------------------------------------------------------------------------
def compute_strain_rate(u_, eps_2nd, solver_q0, q0_test):
    def eps(u_): return sym(nabla_grad(u_))
    b_sr = assemble(sqrt(Constant(0.5) * inner(eps(u_), eps(u_))) * q0_test * dx)
    solver_q0.solve(eps_2nd.vector(), b_sr)


# ---------------------------------------------------------------------------
# Table column widths
# ---------------------------------------------------------------------------
W_STEP  = 5;  W_TIME = 12;  W_DT   = 11;  W_VRMS  = 14
W_STOKES = 9; W_ADV  = 9;   W_UZ   = 6;   W_ALPHA = 8


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    set_log_active(False)
    parameters["ghost_mode"] = "shared_facet"

    comm = pyMPI.COMM_WORLD
    rank = comm.Get_rank()

    # --- Banner -----------------------------------------------------------
    if rank == 0:
        kw = 18
        title = "  2-D Subduction Benchmark  "
        rows = [
            ("Mesh",        f"{args.nx} x {args.ny}"),
            ("Domain",      f"{DOMAIN_WIDTH} x 1.0"),
            ("dt (init)",   f"{args.dt:.2e}  (CFL={args.cfl})"),
            ("Max steps",   str(args.steps)),
            ("Picard",      f"max {args.picard} iters"),
            ("Uzawa",       f"max {args.uzawa_iter} iters  tol={args.uzawa_tol:.0e}"),
            ("Krylov tol",  f"rtol={args.rtol:.0e}  atol={args.atol:.0e}"),
            ("Level-set",   args.level_set_dir),
            ("Output",      "disabled" if args.no_output else
                            f"every {args.save_every} steps → {args.output}"),
        ]
        row_vis = lambda k, v: len(f"  {k}{' '*(kw-len(k))}{v}") + 2
        width = max(len(title), max(row_vis(k, v) for k, v in rows)) + 2

        def pr(key, value):
            line = f"  {yellow(key)}{' '*(kw-len(key))}{C.RESET}{value}"
            pad = width - len(strip_ansi(line))
            print(b(lgreen("║")) + line + " " * max(pad, 0) + b(lgreen("║")))

        print()
        print(b(lgreen("╔" + "═" * width + "╗")))
        print(b(lgreen("║")) + b(f"{title:^{width}}") + b(lgreen("║")))
        print(b(lgreen("╠" + "═" * width + "╣")))
        for k, v in rows:
            pr(k, v)
        print(b(lgreen("╚" + "═" * width + "╝")))
        print()

    t_wall = time.time()

    # --- Build ------------------------------------------------------------
    if rank == 0:
        print(f"  {dim('Building mesh and spaces...')}", flush=True)

    mesh = build_mesh(comm, args.nx, args.ny)
    V, Q, Q0 = build_spaces(mesh)
    bcs = build_bcs(V)

    (G_, D_, M_, Mv, A0,
     u_trial, v_test, p_trial, q_test, q0_test,
     work, pmats) = build_operators(V, Q, Q0)

    solver_s, solver_q, solver_w, solver_q0, solver_ad = build_solvers(
        M_, args.rtol, args.atol, args.max_krylov, args.gmres_restart
    )
    solver_q0.set_operator(A0)

    state = initialize_state(V, Q, Q0, args.level_set_dir, mesh)
    h_cell = CellDiameter(mesh)
    h_min  = MPI.min(mesh.mpi_comm(), mesh.hmin())
    volume = DOMAIN_WIDTH * 1.0  # rectangular domain

    # Initial DG0 projection + material update
    project_level_sets(A0, solver_q0, state["phis"], state["phis_0"], q0_test)
    update_eta_rho(state["eta"], state["rho"], state["phis_0"], state["eps_2nd"])

    # XDMF output
    xdmf = None
    if not args.no_output:
        out_dir = os.path.dirname(args.output)
        if out_dir and rank == 0:
            os.makedirs(out_dir, exist_ok=True)
        comm.Barrier()
        state["u_"].rename("velocity", "velocity")
        state["p_"].rename("pressure", "pressure")
        for name, phi in state["phis"].items():
            phi.rename(name, name)
        state["eta"].rename("eta", "viscosity")
        state["rho"].rename("rho", "density")
        xdmf = XDMFFile(mesh.mpi_comm(), args.output)
        xdmf.parameters["flush_output"] = True

    if rank == 0:
        dofs_V = V.dim();  dofs_Q = Q.dim();  dofs_Q0 = Q0.dim()
        print(f"  {dim(f'DOFs: V={dofs_V}  Q={dofs_Q}  Q0={dofs_Q0}')}")
        print(f"  {dim(f'h_min={h_min:.6e}')}")
        print()

    # --- Table header -----------------------------------------------------
    if rank == 0 and not args.quiet:
        header = (
            "  "
            + b(cyan(cell("step",  W_STEP)))   + " | "
            + b(cyan(cell("t",     W_TIME)))    + " | "
            + b(cyan(cell("dt",    W_DT)))      + " | "
            + b(cyan(cell("v_rms", W_VRMS)))    + " | "
            + b(cyan(cell("t_stokes", W_STOKES))) + " | "
            + b(cyan(cell("t_adv",  W_ADV)))    + " | "
            + b(cyan(cell("uzawa", W_UZ)))      + " | "
            + b(cyan(cell("alpha",  W_ALPHA)))
        )
        sep = (
            "  "
            + "-" * W_STEP   + "-+-"
            + "-" * W_TIME   + "-+-"
            + "-" * W_DT     + "-+-"
            + "-" * W_VRMS   + "-+-"
            + "-" * W_STOKES + "-+-"
            + "-" * W_ADV    + "-+-"
            + "-" * W_UZ     + "-+-"
            + "-" * W_ALPHA
        )
        print(header)
        print(dim(sep))

    # --- Main time loop ---------------------------------------------------
    current_time   = 0.0
    dt_val         = args.dt
    total_stokes   = 0.0
    total_adv      = 0.0
    v_rms_list     = []

    for step in range(1, args.steps + 1):

        # -- Picard iteration (nonlinear Stokes) ---------------------------
        t_stokes_0 = time.time()
        alpha = 1.0
        last_alpha = 1.0
        update_eta_rho(state["eta"], state["rho"],
                       state["phis_0"], state["eps_2nd"])

        for pic in range(args.picard):
            uzawa_it, _ = stokes_uzawa_solve(
                state, solver_s, solver_q, solver_w,
                G_, M_, bcs, pmats, work,
                u_trial, v_test, p_trial, q_test,
                args.uzawa_iter, args.uzawa_tol, args.monitor, rank,
            )
            u_pic = state["u_n"]
            r_k   = state["r_k"]

            # r_k = u_pic - u_pic_n
            r_k.vector().zero()
            r_k.vector().axpy(1.0, u_pic.vector())
            r_k.vector().axpy(-1.0, state["u_pic_n"].vector())

            # Anderson acceleration
            if pic > 0:
                dr = state["delta_r"]
                dr.vector().zero()
                dr.vector().axpy(1.0, r_k.vector())
                dr.vector().axpy(-1.0, state["r_k_n"].vector())
                matvec(pmats["pMv"], dr.vector(), work["pic_tmp"])
                num   = state["r_k_n"].vector().inner(work["pic_tmp"])
                denom = dr.vector().inner(work["pic_tmp"])
                if denom > 1e-30:
                    alpha = -alpha * (num / denom)
                    alpha = max(0.05, min(1.0, alpha))
                else:
                    alpha = 1.0

            # Relax: u_pic = u_pic_n + alpha * r_k
            u_pic.vector().zero()
            u_pic.vector().axpy(1.0, state["u_pic_n"].vector())
            u_pic.vector().axpy(alpha, r_k.vector())

            # Residual (GPU vector ops only)
            work["pic_tmp"].zero()
            work["pic_tmp"].axpy(1.0, state["u_pic_n"].vector())
            work["pic_tmp"].axpy(-1.0, u_pic.vector())
            residual = work["pic_tmp"].norm("l2")
            norm_val = u_pic.vector().norm("l2")
            if args.monitor and rank == 0:
                print(f"    {dim(f'picard {pic+1}')}  "
                      f"res = {cyan(f'{residual/max(norm_val,1e-300):.6e}')}"
                      f"  alpha = {alpha:.4f}", flush=True)

            compute_strain_rate(u_pic, state["eps_2nd"], solver_q0, q0_test)
            update_eta_rho(state["eta"], state["rho"],
                           state["phis_0"], state["eps_2nd"])

            state["r_k_n"].vector().zero()
            state["r_k_n"].vector().axpy(1.0, r_k.vector())
            state["u_pic_n"].vector().zero()
            state["u_pic_n"].vector().axpy(1.0, u_pic.vector())
            last_alpha = alpha

        # Reset Picard state
        alpha = 1.0
        state["r_k_n"].vector().zero()
        state["r_k"].vector().zero()
        state["delta_r"].vector().zero()

        t_stokes = time.time() - t_stokes_0
        total_stokes += t_stokes

        # -- Diagnostics ---------------------------------------------------
        uu = state["u_pic_n"]
        matvec(pmats["pMv"], uu.vector(), work["pic_tmp"])
        v_rms = np.sqrt(uu.vector().inner(work["pic_tmp"]) / volume)
        v_rms_list.append(v_rms)

        # -- Adaptive dt ---------------------------------------------------
        max_u = uu.vector().norm("linf")
        dt_val = args.cfl * h_min / max_u if max_u > 1e-30 else args.dt

        # -- Level-set advection -------------------------------------------
        t_adv_0 = time.time()
        for name in state["phis"]:
            level_set_advection_step(
                h_cell, solver_ad, uu,
                state["phis"][name], state["phis_n"][name],
                dt_val, p_trial, q_test,
            )
            state["phis"][name].vector().zero()
            state["phis"][name].vector().axpy(1.0, state["phis_n"][name].vector())

        project_level_sets(A0, solver_q0, state["phis"], state["phis_0"], q0_test)
        t_adv = time.time() - t_adv_0
        total_adv += t_adv

        current_time += dt_val

        # -- Output --------------------------------------------------------
        if xdmf is not None and step % args.save_every == 0:
            xdmf.write(state["u_"],  float(current_time))
            xdmf.write(state["p_"],  float(current_time))
            for phi in state["phis"].values():
                xdmf.write(phi, float(current_time))
            xdmf.write(state["eta"], float(current_time))
            xdmf.write(state["rho"], float(current_time))

        # -- Print row -----------------------------------------------------
        if rank == 0 and not args.quiet:
            line = (
                "  "
                + cell(b(str(step)),            W_STEP)   + " | "
                + cell(f"{current_time:.6e}",   W_TIME)   + " | "
                + cell(f"{dt_val:.4e}",         W_DT)     + " | "
                + cell(cyan(f"{v_rms:.6e}"),    W_VRMS)   + " | "
                + cell(f"{t_stokes:.2f}s",      W_STOKES) + " | "
                + cell(f"{t_adv:.2f}s",         W_ADV)    + " | "
                + cell(str(uzawa_it),           W_UZ)     + " | "
                + cell(f"{last_alpha:.4f}",     W_ALPHA)
            )
            print(line)

    # --- Summary ----------------------------------------------------------
    wall = time.time() - t_wall
    if rank == 0:
        print()
        print(dim("  " + "-" * 60))
        print(f"  {b('Steps completed')}  {args.steps}")
        print(f"  {b('Final time')}       {current_time:.6e}")
        if v_rms_list:
            print(f"  {b('Final v_rms')}      {cyan(f'{v_rms_list[-1]:.6e}')}")
        print(f"  {dim('Stokes total')}     {total_stokes:.2f}s")
        print(f"  {dim('Advect total')}     {total_adv:.2f}s")
        print(f"  {dim('Wall total')}       {wall:.2f}s")
        print(dim("  " + "-" * 60))
        print()

    if xdmf is not None:
        xdmf.close()
        if rank == 0:
            print(f"  {green('done')} Output closed ({args.output}).")

    del solver_s, solver_q, solver_w, solver_q0, solver_ad
    gc.collect()
    sys.stdout.flush()
    sys.stderr.flush()
    comm.Barrier()
    pyMPI.Finalize()
    os._exit(0)


if __name__ == "__main__":
    main()
