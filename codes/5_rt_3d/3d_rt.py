"""
3D Rayleigh-Taylor instability benchmark (Stokes + level-set advection)
with ENO2 reinitialization (Min 2010).

Velocity: P1 + Bubble (mini-element) on tetrahedra.
Pressure / level-set: CG1.
Stokes: Uzawa-CG with variable viscosity (eta) and density (rho).
Level-set: stabilized Crank-Nicolson advection + ENO2 reinit.

Usage:
    python 3d_rt_reinit.py [options]

Examples:
    python 3d_rt_reinit.py --monitor
    python 3d_rt_reinit.py --nx 50 --ny 50 --nz 50 --dt 0.25 --final-time 200
    PETSC_OPTIONS="-use_gpu_aware_mpi 0" mpirun -n 4 python 3d_rt_reinit.py --nx 64 --ny 64 --nz 64 --quiet
    mpirun -n 4 python 3d_rt_reinit.py --reinit-interval 5 --reinit-mode mpi
"""

WIDTH         = 0.9142
DEPTH         = 0.8142
HEIGHT        = 1.0
NX            = 60
NY            = 60
NZ            = 60
DT            = 5.0
FINAL_TIME    = 500.0
MAX_STEPS     = 10000000
UZAWA_MAX     = 2000
UZAWA_TOL     = 1e-3
RTOL          = 1e-7
ATOL          = 1e-12
MAX_KRYLOV_IT = 500
GMRES_RESTART = 400
INTERFACE_AMP = 0.02
INTERFACE_Z0  = 0.2
REINIT_INTERVAL  = 0      # max-cap: reinit at least every N steps (0 = no cap)
REINIT_ITER      = 100
REINIT_TOL       = 1e-8
REINIT_THRESHOLD = 0.4    # trigger reinit when RMS(|∇φ|-1) exceeds this
ETA1          = 1e-2   # viscosity of lower fluid
ETA2          = 1.0    # viscosity of upper fluid
RHO1          = 0.0    # density of lower fluid
RHO2          = 1.0    # density of upper fluid
ALPHA_FILTER  = 1.0    # diffuse-filter bandwidth for eta = alpha * h_min
ALPHA_RHO     = 0.3    # diffuse-filter bandwidth for rho (sharper than eta)

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

# ---------------------------------------------------------------------------
# Reinitialization imports
# ---------------------------------------------------------------------------
try:
    from reinit_3d import min2010_reinitialize_gpu_3d
    _REINIT_AVAILABLE = True
except ImportError:
    _REINIT_AVAILABLE = False

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Terminal colour helpers
# ---------------------------------------------------------------------------
class C:
    RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
    CYAN = "\033[36m"; GREEN = "\033[32m"; YELLOW = "\033[33m"
    RED = "\033[31m"; LIGHT_GREEN = "\033[92m"


def b(t):    return f"{C.BOLD}{t}{C.RESET}"
def dim(t):  return f"{C.DIM}{t}{C.RESET}"
def cyan(t): return f"{C.CYAN}{t}{C.RESET}"
def green(t):   return f"{C.GREEN}{t}{C.RESET}"
def yellow(t):  return f"{C.YELLOW}{t}{C.RESET}"
def red(t):     return f"{C.RED}{t}{C.RESET}"
def lgreen(t):  return f"{C.LIGHT_GREEN}{t}{C.RESET}"


def strip_ansi(text):
    import re
    return re.sub(r"\033\[[0-9;]*m", "", text)


def cell(text, width, align="right"):
    visible = len(strip_ansi(text))
    pad = max(width - visible, 0)
    return text + " " * pad if align == "left" else " " * pad + text


def fmt_norm(value, tol=None):
    text = f"{value:.6e}"
    if tol is None:     return cyan(text)
    if value < tol:     return green(text)
    if value < tol*1e2: return yellow(text)
    return red(text)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="3D Rayleigh-Taylor: Stokes + level-set advection + ENO2 reinit"
    )
    parser.add_argument("--width",  type=float, default=WIDTH,  help=f"Domain x-extent (default: {WIDTH})")
    parser.add_argument("--depth",  type=float, default=DEPTH,  help=f"Domain y-extent (default: {DEPTH})")
    parser.add_argument("--height", type=float, default=HEIGHT, help=f"Domain z-extent (default: {HEIGHT})")
    parser.add_argument("--nx", type=int, default=NX, help=f"Mesh cells in x (default: {NX})")
    parser.add_argument("--ny", type=int, default=NY, help=f"Mesh cells in y (default: {NY})")
    parser.add_argument("--nz", type=int, default=NZ, help=f"Mesh cells in z (default: {NZ})")
    parser.add_argument("--dt", type=float, default=DT, help=f"Time step (default: {DT})")
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
                        help=f"Interface cosine amplitude (default: {INTERFACE_AMP})")
    parser.add_argument("--interface-offset", type=float, default=INTERFACE_Z0,
                        help=f"Interface z offset from bottom (default: {INTERFACE_Z0})")
    parser.add_argument("--eta1", type=float, default=ETA1, help=f"Lower fluid viscosity (default: {ETA1})")
    parser.add_argument("--eta2", type=float, default=ETA2, help=f"Upper fluid viscosity (default: {ETA2})")
    parser.add_argument("--rho1", type=float, default=RHO1, help=f"Lower fluid density (default: {RHO1})")
    parser.add_argument("--rho2", type=float, default=RHO2, help=f"Upper fluid density (default: {RHO2})")
    parser.add_argument("--alpha-filter", type=float, default=ALPHA_FILTER,
                        help=f"Diffuse filter bandwidth for eta = alpha*h (default: {ALPHA_FILTER})")
    parser.add_argument("--output", type=str,
                        default="./results/3d_rt_reinit_output.xdmf",
                        help="Output XDMF file path")
    parser.add_argument("--no-output", action="store_true", default=False,
                        help="Skip writing XDMF output")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save snapshot every N steps (default: 10); "
                             "ignored if --output-interval is set")
    parser.add_argument("--cfl", type=float, default=None,
                        help="CFL number for adaptive dt based on max velocity "
                             "(--dt becomes the upper cap; omit to use fixed --dt)")
    parser.add_argument("--output-interval", type=float, default=None,
                        help="Write output every this physical time interval (exact); "
                             "dt is trimmed to land precisely on each output time. "
                             "Replaces --save-every when set.")
    parser.add_argument("--monitor", action="store_true", default=False,
                        help="Print per-Uzawa-iteration convergence")
    parser.add_argument("--quiet", action="store_true", default=False,
                        help="Suppress per-step table; print only summary")
    parser.add_argument("--reinit-threshold", type=float, default=REINIT_THRESHOLD,
                        help=f"Trigger reinit when RMS(|∇φ|-1) > this value "
                             f"(default: {REINIT_THRESHOLD}, 0=disable threshold)")
    parser.add_argument("--reinit-interval", type=int, default=REINIT_INTERVAL,
                        help=f"Safety cap: also reinit every N steps regardless of threshold "
                             f"(default: {REINIT_INTERVAL}, 0=no cap)")
    parser.add_argument("--reinit-iter", type=int, default=REINIT_ITER,
                        help=f"Max ENO2 iterations (default: {REINIT_ITER})")
    parser.add_argument("--reinit-tol", type=float, default=REINIT_TOL,
                        help=f"ENO2 convergence tolerance (default: {REINIT_TOL:.0e})")
    parser.add_argument("--reinit-mode", choices=["rank0", "mpi"], default="rank0",
                        help="rank0: GPU reinit on rank 0 + scatter (default); "
                             "mpi: distributed ENO2 across all ranks")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Mesh & function spaces
# ---------------------------------------------------------------------------

def build_mesh(comm, nx, ny, nz, width, depth, height):
    return BoxMesh.create(
        comm,
        [Point(0.0, 0.0, 0.0), Point(width, depth, height)],
        [nx, ny, nz],
        CellType.Type.tetrahedron,
    )


def build_spaces(mesh):
    P1  = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    Bub = FiniteElement("Bubble",   mesh.ufl_cell(), mesh.topology().dim() + 1)
    P1B = VectorElement(NodalEnrichedElement(P1, Bub))
    velocity_space = FunctionSpace(mesh, P1B)
    scalar_space   = FunctionSpace(mesh, "CG", 1)
    return velocity_space, scalar_space


def build_bcs(velocity_space, width, depth, height):
    """
    Lateral walls (x and y): free-slip (normal velocity = 0).
    Top and bottom (z): no-slip (full velocity = 0).
    """
    return [
        DirichletBC(velocity_space.sub(0), Constant(0.0),
                    f"near(x[0], 0.0) or near(x[0], {width})"),
        DirichletBC(velocity_space.sub(1), Constant(0.0),
                    f"near(x[1], 0.0) or near(x[1], {depth})"),
        DirichletBC(velocity_space, Constant((0.0, 0.0, 0.0)),
                    f"near(x[2], 0.0) or near(x[2], {height})"),
    ]


# ---------------------------------------------------------------------------
# Operator assembly
# ---------------------------------------------------------------------------

def build_operators(velocity_space, scalar_space, bcs_u):
    """
    Pre-assemble the operators that do NOT depend on eta/rho:
      G_  : pressure-gradient coupling   (V* ← Q)
      D_  : divergence                   (Q* ← V)
      M_  : pressure mass matrix         (Q* ← Q)
      M_2 : velocity mass matrix         (V* ← V)
      B_  : buoyancy coupling            (V* ← Q), B_[rho] = ∫ rho*g·v dx

    Returns also the quadrature measure dxq.
    """
    u_trial = TrialFunction(velocity_space)
    v_test  = TestFunction(velocity_space)
    p_trial = TrialFunction(scalar_space)
    q_test  = TestFunction(scalar_space)
    g_vec   = Constant((0.0, 0.0, -1.0))

    dxq = dx(metadata={"quadrature_degree": 6})

    G_ = PETScMatrix(); D_ = PETScMatrix()
    M_ = PETScMatrix(); B_ = PETScMatrix()

    assemble(inner(grad(p_trial), v_test) * dxq,       tensor=G_)
    assemble(div(u_trial) * q_test * dxq,              tensor=D_)
    assemble(p_trial * q_test * dxq,                   tensor=M_)
    assemble(inner(p_trial * g_vec, v_test) * dxq,     tensor=B_)
    M_2 = assemble(inner(u_trial, v_test) * dx)

    # Working vectors
    bs   = Vector(); G_.init_vector(bs,   0)
    bq   = Vector(); D_.init_vector(bq,   0)
    tmpQ = Vector(); M_.init_vector(tmpQ, 0)
    bV   = Vector(); B_.init_vector(bV,   0)

    return G_, D_, M_, M_2, B_, bs, bq, tmpQ, bV, dxq


def build_stokes_solvers(rtol, atol, max_it, gmres_restart):
    """Three KSP solvers for Uzawa-CG: velocity, pressure, viscosity-weighted pressure."""
    solver_s = PETScKrylovSolver("gmres", "hypre_amg")
    _ksp_s = as_backend_type(solver_s).ksp()
    _ksp_s.setGMRESRestart(gmres_restart)
    _ksp_s.setTolerances(rtol=rtol, atol=atol, max_it=max_it)
    _ksp_s.getPC().setType("hypre"); _ksp_s.getPC().setHYPREType("boomeramg")

    solver_q = PETScKrylovSolver("cg", "hypre_amg")
    solver_q.parameters["error_on_nonconvergence"] = False
    _ksp_q = as_backend_type(solver_q).ksp()
    _ksp_q.setTolerances(rtol=1e-10, atol=atol, max_it=max_it)
    _ksp_q.getPC().setType("hypre"); _ksp_q.getPC().setHYPREType("boomeramg")

    solver_w = PETScKrylovSolver("cg", "hypre_amg")
    solver_w.parameters["error_on_nonconvergence"] = False
    _ksp_w = as_backend_type(solver_w).ksp()
    _ksp_w.setTolerances(rtol=1e-10, atol=atol, max_it=max_it)
    _ksp_w.getPC().setType("hypre"); _ksp_w.getPC().setHYPREType("boomeramg")

    return solver_s, solver_q, solver_w


def build_advection_solver(rtol, atol, max_it, gmres_restart):
    solver = PETScKrylovSolver("gmres", "hypre_amg")
    _ksp = as_backend_type(solver).ksp()
    _ksp.setGMRESRestart(gmres_restart)
    _ksp.setTolerances(rtol=rtol, atol=atol, max_it=max_it)
    _ksp.getPC().setType("hypre"); _ksp.getPC().setHYPREType("boomeramg")
    solver.parameters["error_on_nonconvergence"] = False
    return solver


# ---------------------------------------------------------------------------
# State initialisation
# ---------------------------------------------------------------------------

def initialize_state(velocity_space, scalar_space, width, depth, amplitude, z0):
    u_  = Function(velocity_space)
    u_n = Function(velocity_space)
    p_  = Function(scalar_space)
    p_n = Function(scalar_space)
    qk  = Function(scalar_space)
    dk  = Function(scalar_space)
    hk  = Function(velocity_space)
    wk  = Function(scalar_space)
    rho = Function(scalar_space)
    eta = Function(scalar_space)
    phi = Function(scalar_space)
    phi_new = Function(scalar_space)

    phi_expr = Expression(
        "x[2] - amp*cos(pi*x[0]/W)*cos(pi*x[1]/D) - z0",
        degree=2, pi=np.pi, amp=amplitude, W=width, D=depth, z0=z0
    )
    phi.interpolate(phi_expr)
    return u_, u_n, p_, p_n, qk, dk, hk, wk, rho, eta, phi, phi_new


# ---------------------------------------------------------------------------
# Level-set filter (diffuse Heaviside → rho, eta)
# ---------------------------------------------------------------------------

def diffuse_level_set_filter(phi, rho, eta, h_min,
                              C1, C2, eta1, eta2, alpha_rho, alpha_eta):
    p_loc = phi.vector().get_local()
    r_loc = rho.vector().get_local()
    e_loc = eta.vector().get_local()

    # --- rho: linear Heaviside with bandwidth eps_rho ---
    eps_rho = alpha_rho * h_min
    lo_r  = p_loc <= -eps_rho
    hi_r  = p_loc >=  eps_rho
    mid_r = ~(lo_r | hi_r)
    r_loc[lo_r] = C1
    r_loc[hi_r] = C2
    if mid_r.any():
        r_loc[mid_r] = (C2 - C1) * (p_loc[mid_r] / (2.0 * eps_rho)) + 0.5 * (C1 + C2)

    # --- eta: log-linear Heaviside with bandwidth eps_eta ---
    eps_eta = alpha_eta * h_min
    lo_e  = p_loc <= -eps_eta
    hi_e  = p_loc >=  eps_eta
    mid_e = ~(lo_e | hi_e)
    H = np.empty_like(p_loc)
    H[lo_e] = 0.0;  H[hi_e] = 1.0
    if mid_e.any():
        H[mid_e] = 0.5 + p_loc[mid_e] / (2.0 * eps_eta)
    loge1 = np.log10(eta1);  loge2 = np.log10(eta2)
    e_loc[:] = 10.0 ** ((1.0 - H) * loge1 + H * loge2)

    rho.vector().set_local(r_loc); rho.vector().apply("insert")
    eta.vector().set_local(e_loc); eta.vector().apply("insert")
    rho.vector().update_ghost_values()
    eta.vector().update_ghost_values()


# ---------------------------------------------------------------------------
# Rank-0 GPU reinitialization (gather → GPU ENO2 → scatter)
# ---------------------------------------------------------------------------

def reinitialize_phi_mpi_3d(phi, scalar_space, comm, rank,
                             nx, ny, nz, width, depth, height,
                             max_iter=15, tol=1e-5):
    """
    Gather all owned DOF values to rank 0, run GPU ENO2 reinit, scatter back.
    Grid convention: grid[iz, iy, ix] with shape (nz+1, ny+1, nx+1).
    """
    dx = width / nx;  dy = depth / ny;  dz = height / nz

    local_range = phi.vector().local_range()
    n_owned = int(local_range[1] - local_range[0])
    local_vals = phi.vector().get_local()[:n_owned].copy()

    coords = scalar_space.tabulate_dof_coordinates()[:n_owned]
    ix = np.clip(np.round(coords[:, 0] / dx).astype(np.int32), 0, nx)
    iy = np.clip(np.round(coords[:, 1] / dy).astype(np.int32), 0, ny)
    iz = np.clip(np.round(coords[:, 2] / dz).astype(np.int32), 0, nz)

    all_ix   = comm.gather(ix,         root=0)
    all_iy   = comm.gather(iy,         root=0)
    all_iz   = comm.gather(iz,         root=0)
    all_vals = comm.gather(local_vals, root=0)

    if rank == 0:
        grid = np.zeros((nz + 1, ny + 1, nx + 1), dtype=np.float64)
        for p_ix, p_iy, p_iz, p_v in zip(all_ix, all_iy, all_iz, all_vals):
            grid[p_iz, p_iy, p_ix] = p_v
        phi_gpu = cp.asarray(grid)
        phi_reinit_gpu, _ = min2010_reinitialize_gpu_3d(
            phi_gpu, dx, dy, dz, max_iter=max_iter, tol=tol
        )
        grid_new = cp.asnumpy(phi_reinit_gpu)
        response = [grid_new[p_iz, p_iy, p_ix]
                    for p_ix, p_iy, p_iz in zip(all_ix, all_iy, all_iz)]
    else:
        response = None

    new_local_vals = comm.scatter(response, root=0)
    full = phi.vector().get_local()
    full[:n_owned] = new_local_vals
    phi.vector().set_local(full)
    phi.vector().apply("insert")
    phi.vector().update_ghost_values()


# ---------------------------------------------------------------------------
# Stokes Uzawa-CG solver
# ---------------------------------------------------------------------------

def stokes_uzawa_solve(
    solver_s, solver_q, solver_w,
    A, Aw, G_, D_, M_, M_2, B_, bs, bq, tmpQ, bV,
    u_, u_n, p_, p_n, qk, dk, hk, wk,
    rho, eta,
    bcs_u, u_trial, v_test, p_trial, q_test, dxq,
    uzawa_max, uzawa_tol, monitor, rank,
):
    t0 = time.time()
    _ksp_s = as_backend_type(solver_s).ksp()

    p_.vector().zero();  p_.vector().apply("insert")
    p_n.vector().zero(); p_n.vector().apply("insert")

    # Reassemble eta-dependent matrices
    A.zero()
    assemble(inner(2.0 * eta * sym(nabla_grad(u_trial)), sym(nabla_grad(v_test))) * dxq,
             tensor=A)
    Aw.zero()
    assemble((1.0 / eta) * p_trial * q_test * dxq, tensor=Aw)

    # Body force: B_ * rho
    B_.mult(rho.vector(), bV)
    for bc in bcs_u:
        bc.apply(A, bV)

    solver_s.set_operator(A)
    solver_s.solve(u_n.vector(), bV)
    gmres_total = _ksp_s.getIterationNumber()

    D_.mult(u_n.vector(), bq)
    solver_q.set_operator(M_)
    solver_q.solve(qk.vector(), bq)

    solver_w.set_operator(Aw)
    M_.mult(qk.vector(), tmpQ)
    solver_w.solve(wk.vector(), tmpQ)      # wk = Aw^{-1} M qk

    dk.vector()[:] = -wk.vector()[:]
    num = qk.vector().inner(M_ * wk.vector())

    uzawa_it = 0
    for ii in range(uzawa_max):
        uzawa_it += 1

        G_.mult(dk.vector(), bs)
        for bc in bcs_u:
            bc.apply(bs)

        solver_s.solve(hk.vector(), bs)
        gmres_total += _ksp_s.getIterationNumber()

        denom = bs.inner(hk.vector())
        if abs(denom) < 1e-30:
            break
        ak = num / denom

        p_.vector().zero()
        p_.vector().axpy(1.0, p_n.vector())
        p_.vector().axpy(ak, dk.vector())

        u_.vector().zero()
        u_.vector().axpy(1.0, u_n.vector())
        u_.vector().axpy(-ak, hk.vector())

        p_n.assign(p_);  u_n.assign(u_)
        D_.mult(u_.vector(), bq)
        solver_q.solve(qk.vector(), bq)

        div_norm = (np.sqrt(max(qk.vector().inner(M_ * qk.vector()), 0.0)) /
                    max(np.sqrt(max(u_n.vector().inner(M_2 * u_n.vector()), 0.0)), 1e-300))

        if monitor and rank == 0:
            print(f"    {dim(f'uzawa {ii+1:4d}')}  "
                  f"div_norm = {fmt_norm(div_norm, uzawa_tol)}", flush=True)

        if (ii + 1) >= 3 and div_norm < uzawa_tol:
            break

        M_.mult(qk.vector(), tmpQ)
        solver_w.solve(wk.vector(), tmpQ)  # wk_new = Aw^{-1} M qk_new

        new_num = qk.vector().inner(M_ * wk.vector())
        bk = new_num / num
        num = new_num
        dk.vector()[:] = -wk.vector()[:] + bk * dk.vector()[:]

    u_.vector().update_ghost_values()
    p_.vector().update_ghost_values()
    return uzawa_it, gmres_total, time.time() - t0


# ---------------------------------------------------------------------------
# Level-set advection
# ---------------------------------------------------------------------------

def level_set_advection_step(h, solver_ad, velocity, phi_old, phi_new,
                              dt, phi_trial, psi_test, dxq_adv):
    t0 = time.time()
    velocity.vector().update_ghost_values()
    phi_old.vector().update_ghost_values()

    speed = sqrt(dot(velocity, velocity)) + Constant(1e-12)
    tau   = h / (2.0 * speed)

    a = (phi_trial / Constant(dt)) * psi_test * dxq_adv
    a += Constant(0.5) * dot(velocity, grad(phi_trial)) * psi_test * dxq_adv
    a += tau * dot(velocity, grad(psi_test)) * (
        phi_trial / Constant(dt) + Constant(0.5) * dot(velocity, grad(phi_trial))
    ) * dxq_adv

    L = (phi_old / Constant(dt)) * psi_test * dxq_adv
    L += -Constant(0.5) * dot(velocity, grad(phi_old)) * psi_test * dxq_adv
    L += tau * dot(velocity, grad(psi_test)) * (
        phi_old / Constant(dt) - Constant(0.5) * dot(velocity, grad(phi_old))
    ) * dxq_adv

    A_ad = assemble(a)
    b_ad = assemble(L)
    solver_ad.set_operator(A_ad)
    solver_ad.solve(phi_new.vector(), b_ad)
    phi_new.vector().update_ghost_values()
    return time.time() - t0


# ---------------------------------------------------------------------------
# Table column widths
# ---------------------------------------------------------------------------
W_STEP   = 5
W_TIME   = 10
W_DT     = 11
W_VRMS   = 14
W_GDEV   = 11
W_STOKES = 9
W_ADV    = 9
W_UZ     = 6
W_GMRES  = 7
W_REINIT = 9


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.dt <= 0.0:
        raise ValueError("--dt must be positive.")
    if args.final_time < 0.0:
        raise ValueError("--final-time must be non-negative.")
    if args.cfl is not None and args.cfl <= 0.0:
        raise ValueError("--cfl must be positive.")
    if args.output_interval is not None and args.output_interval <= 0.0:
        raise ValueError("--output-interval must be positive.")

    alpha_rho = ALPHA_RHO
    alpha_eta = args.alpha_filter

    do_reinit = (args.reinit_interval > 0 or args.reinit_threshold > 0)
    if do_reinit and args.reinit_mode == "rank0":
        if not (_REINIT_AVAILABLE and _CUPY_AVAILABLE):
            raise RuntimeError(
                "rank0 reinit requested but "
                + ("reinit_3d.py not found. " if not _REINIT_AVAILABLE else "")
                + ("CuPy not available. "      if not _CUPY_AVAILABLE    else "")
                + "Use --reinit-mode mpi or --reinit-interval 0."
            )

    set_log_active(False)
    parameters["ghost_mode"] = "shared_facet"

    comm = pyMPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        key_width = 20
        title = "  3-D Rayleigh-Taylor Instability Benchmark  "
        rows = [
            ("Mesh",          f"{args.nx} x {args.ny} x {args.nz}"),
            ("Domain",        f"{args.width} x {args.depth} x {args.height}"),
            ("dt / t_end",    f"{args.dt}  /  {args.final_time}"
                              + (f"  (CFL={args.cfl})" if args.cfl is not None else "")),
            ("Step cap",      str(args.steps)),
            ("Output",        (f"every {args.output_interval} time units"
                               if args.output_interval is not None
                               else f"every {args.save_every} steps")),
            ("Uzawa",         f"max {args.uzawa_iter} iters  tol={args.uzawa_tol:.0e}"),
            ("Krylov tol",    f"rtol={args.rtol:.0e}  atol={args.atol:.0e}"),
            ("GMRES restart", str(args.gmres_restart)),
            ("rho1/rho2",     f"{args.rho1} / {args.rho2}  (alpha_rho={ALPHA_RHO})"),
            ("eta1/eta2",     f"{args.eta1} / {args.eta2}  (alpha_eta={alpha_eta})"),
            ("Reinit",        (
                              (f"threshold RMS(|∇φ|-1)>{args.reinit_threshold}" if args.reinit_threshold > 0 else "")
                              + (" + " if args.reinit_threshold > 0 and args.reinit_interval > 0 else "")
                              + (f"cap every {args.reinit_interval} steps" if args.reinit_interval > 0 else "")
                              + f"  iter={args.reinit_iter}  tol={args.reinit_tol:.0e}  mode={args.reinit_mode}"
                              ) if do_reinit else "disabled"),
            ("XDMF file",     "disabled" if args.no_output else args.output),
        ]
        row_vis = lambda k, v: len(f"  {k}{' '*(key_width-len(k))}{v}") + 2
        width = max(len(title), max(row_vis(k, v) for k, v in rows)) + 2

        def print_row(key, value):
            line = f"  {yellow(key)}{' '*(key_width-len(key))}{C.RESET}{value}"
            pad  = width - len(strip_ansi(line))
            print(b(lgreen("║")) + line + " " * max(pad, 0) + b(lgreen("║")))

        print()
        print(b(lgreen("╔" + "═" * width + "╗")))
        print(b(lgreen("║")) + b(f"{title:^{width}}") + b(lgreen("║")))
        print(b(lgreen("╠" + "═" * width + "╣")))
        for k, v in rows:
            print_row(k, v)
        print(b(lgreen("╚" + "═" * width + "╝")))
        print()

    t_start = time.time()

    if rank == 0:
        print(f"  {dim('Building mesh and spaces...')}", flush=True)

    mesh = build_mesh(comm, args.nx, args.ny, args.nz,
                      args.width, args.depth, args.height)
    velocity_space, scalar_space = build_spaces(mesh)
    bcs_u = build_bcs(velocity_space, args.width, args.depth, args.height)

    (G_, D_, M_, M_2, B_, bs, bq, tmpQ, bV, dxq) = build_operators(
        velocity_space, scalar_space, bcs_u
    )
    A  = PETScMatrix()
    Aw = PETScMatrix()

    volume = assemble(Constant(1.0) * dx(domain=mesh))
    h_cell = CellDiameter(mesh)
    h_min  = mesh.hmin()

    (u_, u_n, p_, p_n, qk, dk, hk, wk,
     rho, eta, phi, phi_new) = initialize_state(
        velocity_space, scalar_space,
        args.width, args.depth,
        args.interface_amplitude,
        args.interface_offset,
    )

    solver_s, solver_q, solver_w = build_stokes_solvers(
        args.rtol, args.atol, args.max_krylov, args.gmres_restart
    )
    solver_ad = build_advection_solver(
        args.rtol, args.atol, args.max_krylov, args.gmres_restart
    )

    # Initial filter
    diffuse_level_set_filter(phi, rho, eta, h_min,
                              args.rho1, args.rho2,
                              args.eta1, args.eta2,
                              alpha_rho, alpha_eta)

    # Create trial/test functions once (reused every step)
    u_trial   = TrialFunction(velocity_space)
    v_test    = TestFunction(velocity_space)
    p_trial   = TrialFunction(scalar_space)
    q_test    = TestFunction(scalar_space)
    dxq_adv   = dx(metadata={"quadrature_degree": 4})

    # Initialize A and Aw sparsity pattern (needed so A.zero() works on first step)
    assemble(inner(2.0 * eta * sym(nabla_grad(u_trial)), sym(nabla_grad(v_test))) * dxq,
             tensor=A)
    assemble((1.0 / eta) * p_trial * q_test * dxq, tensor=Aw)

    # Build reinit callable
    if do_reinit and args.reinit_mode == "mpi":
        from reinit_mpi_3d import DistributedReinit3D
        _dist_reinit = DistributedReinit3D(
            scalar_space, comm,
            args.nx, args.ny, args.nz,
            args.width, args.depth, args.height,
        )
        def _do_reinit(phi_fn):
            _dist_reinit.reinitialize(phi_fn,
                                      max_iter=args.reinit_iter,
                                      tol=args.reinit_tol)
    elif do_reinit:  # rank0 GPU mode
        def _do_reinit(phi_fn):
            reinitialize_phi_mpi_3d(
                phi_fn, scalar_space, comm, rank,
                args.nx, args.ny, args.nz,
                args.width, args.depth, args.height,
                max_iter=args.reinit_iter, tol=args.reinit_tol,
            )
    else:
        _do_reinit = None

    if rank == 0 and not args.quiet:
        header = (
            "  "
            + b(cyan(cell("step",     W_STEP)))   + " | "
            + b(cyan(cell("t",        W_TIME)))    + " | "
            + b(cyan(cell("dt",       W_DT)))      + " | "
            + b(cyan(cell("v_rms",    W_VRMS)))    + " | "
            + b(cyan(cell("|∇φ|-1",   W_GDEV)))    + " | "
            + b(cyan(cell("t_stokes", W_STOKES)))  + " | "
            + b(cyan(cell("t_adv",    W_ADV)))     + " | "
            + b(cyan(cell("uzawa",    W_UZ)))      + " | "
            + b(cyan(cell("GMRES",    W_GMRES)))
        )
        if do_reinit:
            header += " | " + b(cyan(cell("t_reinit", W_REINIT)))
        sep = (
            "  "
            + "-" * W_STEP   + "-+-"
            + "-" * W_TIME   + "-+-"
            + "-" * W_DT     + "-+-"
            + "-" * W_VRMS   + "-+-"
            + "-" * W_GDEV   + "-+-"
            + "-" * W_STOKES + "-+-"
            + "-" * W_ADV    + "-+-"
            + "-" * W_UZ     + "-+-"
            + "-" * W_GMRES
        )
        if do_reinit:
            sep += "-+-" + "-" * W_REINIT
        print(header)
        print(dim(sep))

    current_time    = 0.0
    completed_steps = 0
    v_rms_list      = []
    total_stokes = 0.0;  total_adv = 0.0;  total_reinit = 0.0

    # Precompile |∇φ|-1 deviation form (JIT once, re-assembled each step)
    # RMS(|∇φ|-1) = sqrt( ∫(|∇φ|-1)² dΩ / Ω )
    _grad_dev_form = (sqrt(inner(grad(phi), grad(phi)) + DOLFIN_EPS) - Constant(1.0))**2 * dx(domain=mesh)

    # Output-interval tracking: next physical time at which to write output
    next_output_time = args.output_interval if args.output_interval is not None else None

    if not args.no_output:
        out_dir = os.path.dirname(args.output)
        if out_dir and rank == 0:
            os.makedirs(out_dir, exist_ok=True)
        comm.Barrier()
        u_.rename("velocity", "velocity")
        p_.rename("pressure", "pressure")
        phi.rename("phi", "level_set")
        rho.rename("rho", "density")
        eta.rename("eta", "viscosity")
        xdmf = XDMFFile(mesh.mpi_comm(), args.output)
        xdmf.parameters["flush_output"] = True
        xdmf.write(mesh)
        xdmf.write(u_,  0.0)
        xdmf.write(p_,  0.0)
        xdmf.write(phi, 0.0)
        xdmf.write(rho, 0.0)
        xdmf.write(eta, 0.0)
    else:
        xdmf = None

    while completed_steps < args.steps and current_time < args.final_time - 1e-15:

        # ---- Stokes solve (velocity field used to determine dt) ------------
        uzawa_it, gmres_it, t_stokes = stokes_uzawa_solve(
            solver_s, solver_q, solver_w,
            A, Aw, G_, D_, M_, M_2, B_, bs, bq, tmpQ, bV,
            u_, u_n, p_, p_n, qk, dk, hk, wk,
            rho, eta,
            bcs_u, u_trial, v_test, p_trial, q_test, dxq,
            args.uzawa_iter, args.uzawa_tol, args.monitor, rank,
        )
        total_stokes += t_stokes

        rms_velocity = float(np.sqrt(max(
            abs(u_.vector().inner(M_2 * u_.vector()) / volume), 0.0
        )))
        v_rms_list.append(rms_velocity)

        # ---- Compute dt (CFL-based or fixed) --------------------------------
        if args.cfl is not None:
            u_arr = u_.vector().get_local()
            u_max_local = float(np.max(np.abs(u_arr))) if len(u_arr) > 0 else 0.0
            u_max = comm.allreduce(u_max_local, op=pyMPI.MAX)
            dt_base = args.cfl * h_min / u_max if u_max > 1e-30 else args.dt
            dt_base = min(dt_base, args.dt)   # --dt acts as upper cap
        else:
            dt_base = args.dt

        # Trim to final time
        dt_step = min(dt_base, args.final_time - current_time)

        # Trim to next output time so we land exactly on it
        if next_output_time is not None:
            dt_step = min(dt_step, next_output_time - current_time)

        # ---- Level-set advection -------------------------------------------
        t_adv = level_set_advection_step(
            h_cell, solver_ad, u_, phi, phi_new, dt_step, p_trial, q_test, dxq_adv
        )
        total_adv += t_adv

        phi.vector().zero()
        phi.vector().axpy(1.0, phi_new.vector())
        phi.vector().update_ghost_values()

        # ---- |∇φ|-1 deviation check & ENO2 reinitialization ---------------
        t_reinit = 0.0
        rms_dev  = 0.0
        if _do_reinit is not None:
            dev_sq  = assemble(_grad_dev_form)
            rms_dev = float(np.sqrt(max(dev_sq / volume, 0.0)))

            trigger_threshold = (args.reinit_threshold > 0 and rms_dev > args.reinit_threshold)
            trigger_interval  = (args.reinit_interval  > 0 and
                                 (completed_steps + 1) % args.reinit_interval == 0)

            if trigger_threshold or trigger_interval:
                t0_ri = time.time()
                _do_reinit(phi)
                t_reinit = time.time() - t0_ri
                total_reinit += t_reinit

        # ---- Update rho, eta from phi --------------------------------------
        diffuse_level_set_filter(phi, rho, eta, h_min,
                                  args.rho1, args.rho2,
                                  args.eta1, args.eta2,
                                  alpha_rho, alpha_eta)

        current_time    += dt_step
        completed_steps += 1

        # ---- XDMF output ---------------------------------------------------
        do_save = False
        if xdmf is not None:
            if next_output_time is not None:
                # Exact output-interval mode: check if we just hit the target
                if current_time >= next_output_time - 1e-12 * args.output_interval:
                    do_save = True
                    next_output_time += args.output_interval
            else:
                do_save = (completed_steps % args.save_every == 0)

        if do_save:
            xdmf.write(u_,  float(current_time))
            xdmf.write(p_,  float(current_time))
            xdmf.write(phi, float(current_time))
            xdmf.write(rho, float(current_time))
            xdmf.write(eta, float(current_time))
            if rank == 0 and not args.quiet:
                print(f"  {dim(f'[step {completed_steps}  t={current_time:.5f}] snapshot saved')}", flush=True)

        if rank == 0 and not args.quiet:
            step_s   = cell(b(f"{completed_steps}"),      W_STEP)
            time_s   = cell(f"{current_time:.5f}",        W_TIME)
            dt_s     = cell(f"{dt_step:.4e}",             W_DT)
            vrms_s   = cell(cyan(f"{rms_velocity:.6e}"),  W_VRMS)
            _dev_thr = args.reinit_threshold if do_reinit else None
            gdev_s   = cell(fmt_norm(rms_dev, _dev_thr),  W_GDEV)
            stokes_s = cell(f"{t_stokes:.2f}s",           W_STOKES)
            adv_s    = cell(f"{t_adv:.2f}s",              W_ADV)
            uzawa_s  = cell(f"{uzawa_it}",                W_UZ)
            gmres_s  = cell(f"{gmres_it}",                W_GMRES)
            line = (f"  {step_s} | {time_s} | {dt_s} | {vrms_s} | {gdev_s} | {stokes_s} | "
                    f"{adv_s} | {uzawa_s} | {gmres_s}")
            if do_reinit:
                ri_s = cell(f"{t_reinit:.2f}s" if t_reinit > 0 else "---", W_REINIT)
                line += f" | {ri_s}"
            print(line)

    wall_time = time.time() - t_start

    if rank == 0:
        print()
        print(dim("  " + "-" * 60))
        print(f"  {b('Steps completed')}  {completed_steps}")
        print(f"  {b('Final time')}       {current_time:.5f}")
        if v_rms_list:
            print(f"  {b('Final v_rms')}      {cyan(f'{v_rms_list[-1]:.6e}')}")
        print(f"  {dim('Stokes total')}     {total_stokes:.2f}s")
        print(f"  {dim('Advect total')}     {total_adv:.2f}s")
        if do_reinit:
            print(f"  {dim('Reinit total')}     {total_reinit:.2f}s")
        print(f"  {dim('Wall total')}       {wall_time:.2f}s")
        print(dim("  " + "-" * 60))
        print()

    if xdmf is not None:
        xdmf.close()
        if rank == 0:
            print(f"  {green('✓')} Output closed ({args.output}).")

    del solver_s, solver_q, solver_w, solver_ad
    del u_, u_n, p_, p_n, qk, dk, hk, wk, rho, eta, phi, phi_new
    del G_, D_, M_, M_2, B_, A, Aw
    del u_trial, v_test, p_trial, q_test
    del bcs_u, velocity_space, scalar_space, mesh
    gc.collect()

    sys.stdout.flush()
    sys.stderr.flush()
    comm.Barrier()
    pyMPI.Finalize()
    os._exit(0)


if __name__ == "__main__":
    main()
