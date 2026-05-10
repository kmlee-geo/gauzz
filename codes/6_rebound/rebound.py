"""
2D two-phase Stokes benchmark with DG1 level-set advection and Uzawa-CG solver.

Velocity: P1B (Mini element: P1 + Bubble) on triangles.
Pressure: CG1.
Level-set / material: DG1.
Stokes: Uzawa-CG with variable viscosity (smoothed Heaviside from level-sets).
Level-set: Crank-Nicolson DG upwind advection.

Usage:
    mpirun -np 4 python rt_stokes.py
    mpirun -np 4 python rt_stokes.py --nx 1400 --ny 800 --steps 200
    mpirun -np 4 python rt_stokes.py --quiet
    mpirun -np 4 python rt_stokes.py --monitor --cfl 0.3
"""

# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------
DOMAIN_WIDTH  = 3.5
DOMAIN_HEIGHT = 1.0
NX            = 1400
NY            = 800

ETA_ABOVE     = 1.0        # viscosity above phi1 = 0 (top layer)
ETA_BAND      = 1e5        # viscosity in band  (phi1 > 0, phi2 > 0)
ETA_ELSE      = 1e4        # viscosity below phi2 = 0 (bottom layer)
RHO_INSIDE    = 0.0        # density above phi1 = 0
RHO_OUTSIDE   = 1.69e10   # density below phi1 = 0
LS_ALPHA      = 1.0        # interface half-width = LS_ALPHA * h_min

# Initial interface positions (as fractions of domain height)
PHI1_Y0       = 700.0 / 800.0   # mean height of upper interface (~0.875)
PHI1_AMP      = 7.0   / 800.0   # sinusoidal perturbation amplitude (~0.00875)
PHI2_Y0       = 600.0 / 800.0   # height of lower interface (~0.75)

C_CFL         = 0.4
DT_INIT       = 1.23e-8
MAX_STEPS     = 100
UZAWA_MAX     = 2000
UZAWA_TOL     = 1e-7
RTOL          = 1e-10
ATOL          = 1e-12
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
        description="2D Stokes RT: P1B velocity, DG1 level-sets, Uzawa-CG"
    )
    ap.add_argument("--nx",            type=int,   default=NX)
    ap.add_argument("--ny",            type=int,   default=NY)
    ap.add_argument("--steps",         type=int,   default=MAX_STEPS)
    ap.add_argument("--cfl",           type=float, default=C_CFL)
    ap.add_argument("--dt",            type=float, default=DT_INIT)
    ap.add_argument("--uzawa-iter",    type=int,   default=UZAWA_MAX)
    ap.add_argument("--uzawa-tol",     type=float, default=UZAWA_TOL)
    ap.add_argument("--rtol",          type=float, default=RTOL)
    ap.add_argument("--atol",          type=float, default=ATOL)
    ap.add_argument("--max-krylov",    type=int,   default=MAX_KRYLOV_IT)
    ap.add_argument("--gmres-restart", type=int,   default=GMRES_RESTART)
    ap.add_argument("--ls-alpha",      type=float, default=LS_ALPHA)
    ap.add_argument("--output",        type=str,   default="./results/rt_stokes.xdmf")
    ap.add_argument("--no-output",     action="store_true", default=False)
    ap.add_argument("--save-every",    type=int,   default=10)
    ap.add_argument("--monitor",       action="store_true", default=False,
                    help="Print per-Uzawa-iteration convergence")
    ap.add_argument("--quiet",         action="store_true", default=False)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Mesh & function spaces
# ---------------------------------------------------------------------------
def build_mesh(comm, nx, ny):
    return RectangleMesh.create(
        comm,
        [Point(0.0, 0.0), Point(DOMAIN_WIDTH, DOMAIN_HEIGHT)],
        [nx, ny],
        CellType.Type.triangle,
        "left/right",
    )


def build_spaces(mesh):
    P1   = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    Bub  = FiniteElement("Bubble",   mesh.ufl_cell(), mesh.topology().dim() + 1)
    P1B  = VectorElement(NodalEnrichedElement(P1, Bub))
    V    = FunctionSpace(mesh, P1B)
    Q    = FunctionSpace(mesh, "CG", 1)
    Q_dg = FunctionSpace(mesh, "DG", 1)
    return V, Q, Q_dg


def build_bcs(V):
    return [
        DirichletBC(V.sub(0), Constant(0.0),
                    f"near(x[0], 0.0) or near(x[0], {DOMAIN_WIDTH})"),
        DirichletBC(V.sub(1), Constant(0.0),
                    f"near(x[1], {DOMAIN_HEIGHT})"),
        DirichletBC(V, Constant((0.0, 0.0)),
                    "near(x[1], 0.0)"),
    ]


# ---------------------------------------------------------------------------
# Operator assembly & work vectors
# ---------------------------------------------------------------------------
def build_operators(V, Q, Q_dg):
    u_trial  = TrialFunction(V);    v_test  = TestFunction(V)
    p_trial  = TrialFunction(Q);    q_test  = TestFunction(Q)
    pd_trial = TrialFunction(Q_dg); qd_test = TestFunction(Q_dg)

    G_   = assemble(inner(grad(p_trial), v_test) * dx)
    D_   = assemble(div(u_trial) * q_test * dx)
    M_   = assemble(p_trial * q_test * dx)
    Mv   = assemble(inner(u_trial, v_test) * dx)
    M_dg = assemble(pd_trial * qd_test * dx)

    pG   = as_backend_type(G_).mat()
    pD   = as_backend_type(D_).mat()
    pM   = as_backend_type(M_).mat()
    pMv  = as_backend_type(Mv).mat()

    # Pre-allocated work vectors (avoid per-iteration allocation)
    uz_gdk  = Function(V).vector()   # G_ * dk  (raw, before BCs)
    uz_bs   = Function(V).vector()   # G_ * dk  (after BCs applied)
    uz_bq   = Function(Q).vector()   # D_ * u_
    uz_Mwk  = Function(Q).vector()   # M_ * wk
    uz_tmpd = Function(Q).vector()   # dk copy for CG update
    uz_Mvu  = Function(V).vector()   # Mv * u_  (for vel_norm)
    uz_Mqk  = Function(Q).vector()   # M_ * qk  (for div_norm)

    work = dict(
        uz_gdk=uz_gdk, uz_bs=uz_bs, uz_bq=uz_bq,
        uz_Mwk=uz_Mwk, uz_tmpd=uz_tmpd, uz_Mvu=uz_Mvu, uz_Mqk=uz_Mqk,
    )
    pmats = dict(pG=pG, pD=pD, pM=pM, pMv=pMv)

    return (G_, D_, M_, Mv, M_dg,
            u_trial, v_test, p_trial, q_test, pd_trial, qd_test,
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
    ksp_s.getPC().setType("hypre")
    ksp_s.getPC().setHYPREType("boomeramg")

    solver_q = PETScKrylovSolver("cg", "hypre_amg")
    solver_q.set_operator(M_)
    solver_q.parameters["error_on_nonconvergence"] = False
    ksp_q = as_backend_type(solver_q).ksp()
    ksp_q.setTolerances(rtol=1e-10, atol=1e-12, max_it=max_it)
    ksp_q.getPC().setType("hypre")
    ksp_q.getPC().setHYPREType("boomeramg")

    solver_w = PETScKrylovSolver("cg", "hypre_amg")
    solver_w.parameters["error_on_nonconvergence"] = False
    ksp_w = as_backend_type(solver_w).ksp()
    ksp_w.setTolerances(rtol=1e-10, atol=1e-12, max_it=max_it)
    ksp_w.getPC().setType("hypre")
    ksp_w.getPC().setHYPREType("boomeramg")

    solver_ad = PETScKrylovSolver("gmres", "hypre_amg")
    ksp_ad = as_backend_type(solver_ad).ksp()
    ksp_ad.setGMRESRestart(gmres_restart)
    ksp_ad.setTolerances(rtol=1e-10, atol=1e-12, max_it=max_it)
    solver_ad.parameters["error_on_nonconvergence"] = False

    return solver_s, solver_q, solver_w, solver_ad


# ---------------------------------------------------------------------------
# State initialisation
# ---------------------------------------------------------------------------
def initialize_state(V, Q, Q_dg):
    u_  = Function(V);  u_n = Function(V)
    p_  = Function(Q);  p_n = Function(Q)
    qk  = Function(Q);  wk  = Function(Q)
    dk  = Function(Q);  hk  = Function(V)

    eta = Function(Q_dg)
    rho = Function(Q_dg)
    C   = Function(Q_dg)

    phi1   = Function(Q_dg)
    phi2   = Function(Q_dg)
    phi1_n = Function(Q_dg)
    phi2_n = Function(Q_dg)

    phi1.interpolate(Expression(
        "y0 + amp*cos(2.0*pi*x[0]/W) - x[1]",
        degree=2, pi=np.pi,
        y0=PHI1_Y0, amp=PHI1_AMP, W=DOMAIN_WIDTH,
    ))
    phi2.interpolate(Expression(
        "x[1] - y0",
        degree=1, y0=PHI2_Y0,
    ))

    return dict(
        u_=u_, u_n=u_n, p_=p_, p_n=p_n,
        qk=qk, wk=wk, dk=dk, hk=hk,
        eta=eta, rho=rho, C=C,
        phi1=phi1, phi2=phi2,
        phi1_n=phi1_n, phi2_n=phi2_n,
    )


# ---------------------------------------------------------------------------
# Material update (eta, rho from smoothed Heaviside of level-set fields)
# ---------------------------------------------------------------------------
def update_material(state, h_min, ls_alpha):
    """
    Assigns eta, rho, and C from phi1, phi2 via a smoothed Heaviside filter.
    Three material regions:
      top   : phi1 < 0  →  eta=ETA_ABOVE,  rho=RHO_INSIDE
      band  : phi1 > 0, phi2 > 0  →  eta=ETA_BAND,  rho=RHO_OUTSIDE
      bottom: phi2 < 0  →  eta=ETA_ELSE,   rho=RHO_OUTSIDE
    """
    eps = ls_alpha * h_min

    pA = state["phi1"].vector().get_local()
    pB = state["phi2"].vector().get_local()
    e  = state["eta"].vector().get_local()
    r  = state["rho"].vector().get_local()
    c  = state["C"].vector().get_local()

    # Smoothed Heaviside H_A (phi1), H_B (phi2)
    def smooth_H(phi, eps):
        H = np.empty_like(phi)
        lo = phi <= -eps;  hi = phi >= eps;  mid = ~(lo | hi)
        H[lo] = 0.0;  H[hi] = 1.0
        if mid.any():
            H[mid] = 0.5 + phi[mid] / (2.0 * eps)
        return H

    H_A = smooth_H(pA, eps)
    H_B = smooth_H(pB, eps)

    I_top  = np.clip(1.0 - H_A,        0.0, 1.0)   # above phi1=0
    I_band = np.clip(H_A * H_B,        0.0, 1.0)   # between phi1=0 and phi2=0
    I_bot  = np.clip(1.0 - H_B,        0.0, 1.0)   # below phi2=0

    r[:] = (RHO_OUTSIDE - RHO_INSIDE) * (I_band + I_bot) + RHO_INSIDE

    log_e = (
        I_top  * np.log10(ETA_ABOVE) +
        I_band * np.log10(ETA_BAND)  +
        I_bot  * np.log10(ETA_ELSE)
    )
    e[:] = 10.0 ** log_e

    c[:] = 3                                        # bottom
    c[(pA > 0.0) & (pB > 0.0)] = 2                 # band
    c[pA < 0.0] = 1                                 # top

    state["eta"].vector().set_local(e);  state["eta"].vector().apply("insert")
    state["rho"].vector().set_local(r);  state["rho"].vector().apply("insert")
    state["C"].vector().set_local(c);    state["C"].vector().apply("insert")


# ---------------------------------------------------------------------------
# Stokes Uzawa-CG solver
# ---------------------------------------------------------------------------
def stokes_uzawa_solve(
    state, solver_s, solver_q, solver_w,
    G_, M_, bcs, pmats, work,
    u_trial, v_test, p_trial, q_test,
    uzawa_max, uzawa_tol, monitor, rank,
):
    t0  = time.time()
    u_  = state["u_"];   u_n = state["u_n"]
    p_  = state["p_"];   p_n = state["p_n"]
    qk  = state["qk"];   wk  = state["wk"]
    dk  = state["dk"];   hk  = state["hk"]
    eta = state["eta"];  rho = state["rho"]

    g = Constant((0.0, -1.0))

    def eps(u): return sym(nabla_grad(u))

    # --- Initial velocity solve -------------------------------------------
    p_n.vector().zero()
    p_.vector().zero()

    A = assemble(inner(2.0 * eta * eps(u_trial), eps(v_test)) * dx)
    b = assemble(inner(rho * g - grad(p_n), v_test) * dx)
    for bc in bcs: bc.apply(A, b)
    solver_s.set_operator(A)
    solver_s.solve(u_n.vector(), b)

    # --- Initial pressure correction (Schur complement step) --------------
    matvec(pmats["pD"], u_n.vector(), work["uz_bq"])
    solver_q.solve(qk.vector(), work["uz_bq"])

    Aw = assemble((1.0 / eta) * p_trial * q_test * dx)
    bw = assemble(qk * q_test * dx)
    solver_w.set_operator(Aw)
    solver_w.solve(wk.vector(), bw)

    dk.vector().zero()
    dk.vector().axpy(-1.0, wk.vector())

    # --- Uzawa-CG iteration -----------------------------------------------
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

        p_n.vector().zero(); p_n.vector().axpy(1.0, p_.vector())
        u_n.vector().zero(); u_n.vector().axpy(1.0, u_.vector())

        matvec(pmats["pD"], u_.vector(), work["uz_bq"])
        solver_q.solve(qk.vector(), work["uz_bq"])

        matvec(pmats["pM"],  qk.vector(), work["uz_Mqk"])
        matvec(pmats["pMv"], u_.vector(), work["uz_Mvu"])
        div_norm = np.sqrt(max(qk.vector().inner(work["uz_Mqk"]), 0.0))
        vel_norm = np.sqrt(max(u_.vector().inner(work["uz_Mvu"]), 1e-300))

        if monitor and rank == 0:
            print(f"    {dim(f'uzawa {ii+1:4d}')}  "
                  f"div/vel = {cyan(f'{div_norm/vel_norm:.6e}')}", flush=True)

        if div_norm / vel_norm < uzawa_tol:
            break

        matvec(pmats["pM"], qk.vector(), work["uz_Mwk"])
        solver_w.solve(wk.vector(), work["uz_Mwk"])

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
# Level-set advection (Crank-Nicolson DG upwind)
# ---------------------------------------------------------------------------
def level_set_advection_step(solver_ad, velocity, n_facet,
                              phi_old, phi_new, dt_val,
                              pd_trial, qd_test):
    inv_dt = Constant(1.0 / dt_val)
    un     = dot(velocity, n_facet)
    un_pos = 0.5 * (un + abs(un))

    m     = qd_test * pd_trial * dx
    a_int = -dot(velocity, grad(qd_test)) * pd_trial * dx
    a_if  = dot(jump(qd_test),
                un_pos("+") * pd_trial("+") - un_pos("-") * pd_trial("-")) * dS
    a_bf  = qd_test * un_pos * pd_trial * ds
    a_d   = a_int + a_if + a_bf

    K_mat = assemble(inv_dt * m + Constant(0.5) * a_d)
    B_mat = assemble(inv_dt * m - Constant(0.5) * a_d)

    rhs = Vector()
    B_mat.init_vector(rhs, 0)
    B_mat.mult(phi_old.vector(), rhs)

    solver_ad.set_operator(K_mat)
    solver_ad.solve(phi_new.vector(), rhs)


# ---------------------------------------------------------------------------
# Table column widths
# ---------------------------------------------------------------------------
W_STEP   = 5;  W_TIME = 12;  W_DT  = 11;  W_VRMS = 14
W_STOKES = 9;  W_ADV  = 9;   W_UZ  = 6


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
        title = "  2-D Stokes / RT Benchmark  "
        rows = [
            ("Mesh",        f"{args.nx} x {args.ny}"),
            ("Domain",      f"{DOMAIN_WIDTH} x {DOMAIN_HEIGHT}"),
            ("Velocity",    "P1 + Bubble (Mini element)"),
            ("Level-set",   "DG1  (Crank-Nicolson upwind DG)"),
            ("dt (init)",   f"{args.dt:.2e}  (CFL={args.cfl})"),
            ("Max steps",   str(args.steps)),
            ("Uzawa",       f"max {args.uzawa_iter} iters  tol={args.uzawa_tol:.0e}"),
            ("Krylov tol",  f"rtol={args.rtol:.0e}  atol={args.atol:.0e}"),
            ("LS alpha",    str(args.ls_alpha)),
            ("Output",      "disabled" if args.no_output else
                            f"every {args.save_every} steps → {args.output}"),
        ]
        row_vis = lambda k, v: len(f"  {k}{' '*(kw-len(k))}{v}") + 2
        width   = max(len(title), max(row_vis(k, v) for k, v in rows)) + 2

        def pr(key, value):
            line = f"  {yellow(key)}{' '*(kw-len(key))}{C.RESET}{value}"
            pad  = width - len(strip_ansi(line))
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

    mesh        = build_mesh(comm, args.nx, args.ny)
    V, Q, Q_dg  = build_spaces(mesh)
    bcs         = build_bcs(V)
    n_facet     = FacetNormal(mesh)

    (G_, D_, M_, Mv, M_dg,
     u_trial, v_test, p_trial, q_test, pd_trial, qd_test,
     work, pmats) = build_operators(V, Q, Q_dg)

    solver_s, solver_q, solver_w, solver_ad = build_solvers(
        M_, args.rtol, args.atol, args.max_krylov, args.gmres_restart
    )

    state  = initialize_state(V, Q, Q_dg)
    h_min  = MPI.min(mesh.mpi_comm(), mesh.hmin())
    volume = DOMAIN_WIDTH * DOMAIN_HEIGHT

    update_material(state, h_min, args.ls_alpha)

    # XDMF output
    xdmf = None
    if not args.no_output:
        out_dir = os.path.dirname(args.output)
        if out_dir and rank == 0:
            os.makedirs(out_dir, exist_ok=True)
        comm.Barrier()
        state["u_"].rename("velocity",  "velocity")
        state["p_"].rename("pressure",  "pressure")
        state["phi1"].rename("phi1",    "phi1")
        state["phi2"].rename("phi2",    "phi2")
        state["eta"].rename("eta",      "viscosity")
        state["rho"].rename("rho",      "density")
        xdmf = XDMFFile(mesh.mpi_comm(), args.output)
        xdmf.parameters["flush_output"] = True

    if rank == 0:
        dofs_V = V.dim();  dofs_Q = Q.dim();  dofs_Qd = Q_dg.dim()
        print(f"  {dim(f'DOFs: V={dofs_V}  Q={dofs_Q}  Q_dg={dofs_Qd}')}")
        print(f"  {dim(f'h_min={h_min:.6e}')}")
        print()

    # --- Table header -----------------------------------------------------
    if rank == 0 and not args.quiet:
        header = (
            "  "
            + b(cyan(cell("step",     W_STEP)))    + " | "
            + b(cyan(cell("t",        W_TIME)))    + " | "
            + b(cyan(cell("dt",       W_DT)))      + " | "
            + b(cyan(cell("v_rms",    W_VRMS)))    + " | "
            + b(cyan(cell("t_stokes", W_STOKES)))  + " | "
            + b(cyan(cell("t_adv",    W_ADV)))     + " | "
            + b(cyan(cell("uzawa",    W_UZ)))
        )
        sep = (
            "  "
            + "-" * W_STEP   + "-+-"
            + "-" * W_TIME   + "-+-"
            + "-" * W_DT     + "-+-"
            + "-" * W_VRMS   + "-+-"
            + "-" * W_STOKES + "-+-"
            + "-" * W_ADV    + "-+-"
            + "-" * W_UZ
        )
        print(header)
        print(dim(sep))

    # --- Main time loop ---------------------------------------------------
    current_time = 0.0
    dt_val       = args.dt
    total_stokes = 0.0
    total_adv    = 0.0
    v_rms_list   = []

    for step in range(1, args.steps + 1):

        # -- Material update -----------------------------------------------
        update_material(state, h_min, args.ls_alpha)

        # -- Stokes solve --------------------------------------------------
        uzawa_it, t_stokes = stokes_uzawa_solve(
            state, solver_s, solver_q, solver_w,
            G_, M_, bcs, pmats, work,
            u_trial, v_test, p_trial, q_test,
            args.uzawa_iter, args.uzawa_tol, args.monitor, rank,
        )
        total_stokes += t_stokes

        # -- Diagnostics ---------------------------------------------------
        uu = state["u_n"]
        matvec(pmats["pMv"], uu.vector(), work["uz_Mvu"])
        v_rms = np.sqrt(uu.vector().inner(work["uz_Mvu"]) / volume)
        v_rms_list.append(v_rms)

        # -- Adaptive dt ---------------------------------------------------
        max_u  = uu.vector().norm("linf")
        dt_val = args.cfl * h_min / max_u if max_u > 1e-30 else args.dt

        # -- Level-set advection -------------------------------------------
        t_adv_0 = time.time()
        for phi_old, phi_new in [
            (state["phi1"], state["phi1_n"]),
            (state["phi2"], state["phi2_n"]),
        ]:
            level_set_advection_step(
                solver_ad, uu, n_facet,
                phi_old, phi_new, dt_val,
                pd_trial, qd_test,
            )
            phi_old.vector().zero()
            phi_old.vector().axpy(1.0, phi_new.vector())

        t_adv = time.time() - t_adv_0
        total_adv += t_adv

        current_time += dt_val

        # -- Output --------------------------------------------------------
        if xdmf is not None and step % args.save_every == 0:
            xdmf.write(state["u_"],   float(current_time))
            xdmf.write(state["p_"],   float(current_time))
            xdmf.write(state["phi1"], float(current_time))
            xdmf.write(state["phi2"], float(current_time))
            xdmf.write(state["eta"],  float(current_time))
            xdmf.write(state["rho"],  float(current_time))

        # -- Print row -----------------------------------------------------
        if rank == 0 and not args.quiet:
            line = (
                "  "
                + cell(b(str(step)),           W_STEP)    + " | "
                + cell(f"{current_time:.6e}",  W_TIME)    + " | "
                + cell(f"{dt_val:.4e}",        W_DT)      + " | "
                + cell(cyan(f"{v_rms:.6e}"),   W_VRMS)    + " | "
                + cell(f"{t_stokes:.2f}s",     W_STOKES)  + " | "
                + cell(f"{t_adv:.2f}s",        W_ADV)     + " | "
                + cell(str(uzawa_it),          W_UZ)
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

    del solver_s, solver_q, solver_w, solver_ad
    gc.collect()
    sys.stdout.flush()
    sys.stderr.flush()
    comm.Barrier()
    pyMPI.Finalize()
    os._exit(0)


if __name__ == "__main__":
    main()