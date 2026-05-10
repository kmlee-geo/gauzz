"""
reinit_3d.py — GPU (CuPy) ENO2 reinitialization for 3D level-set fields (rank-0 mode).

Implements the Min (2010) ENO2 + Godunov scheme for the reinitialization PDE:
    ∂φ/∂τ + sgn(φ₀)(|∇φ| - 1) = 0

The grid convention is (nz_nodes, ny_nodes, nx_nodes), i.e., z is axis 0.

Usage
-----
    from reinit_3d import min2010_reinitialize_gpu_3d

    phi_gpu = cp.asarray(grid)   # shape (nz+1, ny+1, nx+1)
    phi_new, iters = min2010_reinitialize_gpu_3d(phi_gpu, dx, dy, dz)
"""

import numpy as np

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False


def _eno2_deriv_3d(g, dx, dy, dz):
    """
    ENO2 one-sided derivatives on a 3D grid g of shape (nz, ny, nx).

    All directions use edge-padding (no ghost exchange; intended for
    single-process rank-0 use on the full grid).

    Returns Dxm, Dxp, Dym, Dyp, Dzm, Dzp — same shape as g.
    """
    xp = cp.get_array_module(g) if _CUPY_AVAILABLE else np

    # x-derivatives (axis=2)
    gp  = xp.pad(g, ((0, 0), (0, 0), (2, 2)), mode='edge')
    gim2 = gp[:, :, :-4]; gim1 = gp[:, :, 1:-3]
    gc   = gp[:, :, 2:-2]; gip1 = gp[:, :, 3:-1]; gip2 = gp[:, :, 4:]
    v0 = (gim1-gim2)/dx;  v1 = (gc-gim1)/dx
    v2 = (gip1-gc  )/dx;  v3 = (gip2-gip1)/dx
    s0 = v1-v0;  s1 = v2-v1;  s2 = v3-v2
    Dxm = xp.where(xp.abs(s0) <= xp.abs(s1), v1 - 0.5*s0, v1 + 0.5*s1)
    Dxp = xp.where(xp.abs(s1) <= xp.abs(s2), v2 - 0.5*s1, v2 + 0.5*s2)

    # y-derivatives (axis=1)
    gp2  = xp.pad(g, ((0, 0), (2, 2), (0, 0)), mode='edge')
    gjm2 = gp2[:, :-4, :]; gjm1 = gp2[:, 1:-3, :]
    gjc  = gp2[:, 2:-2, :]; gjp1 = gp2[:, 3:-1, :]; gjp2 = gp2[:, 4:, :]
    v0y = (gjm1-gjm2)/dy;  v1y = (gjc-gjm1)/dy
    v2y = (gjp1-gjc )/dy;  v3y = (gjp2-gjp1)/dy
    s0y = v1y-v0y;  s1y = v2y-v1y;  s2y = v3y-v2y
    Dym = xp.where(xp.abs(s0y) <= xp.abs(s1y), v1y - 0.5*s0y, v1y + 0.5*s1y)
    Dyp = xp.where(xp.abs(s1y) <= xp.abs(s2y), v2y - 0.5*s1y, v2y + 0.5*s2y)

    # z-derivatives (axis=0)
    gp3  = xp.pad(g, ((2, 2), (0, 0), (0, 0)), mode='edge')
    gkm2 = gp3[:-4, :, :]; gkm1 = gp3[1:-3, :, :]
    gkc  = gp3[2:-2, :, :]; gkp1 = gp3[3:-1, :, :]; gkp2 = gp3[4:, :, :]
    v0z = (gkm1-gkm2)/dz;  v1z = (gkc-gkm1)/dz
    v2z = (gkp1-gkc )/dz;  v3z = (gkp2-gkp1)/dz
    s0z = v1z-v0z;  s1z = v2z-v1z;  s2z = v3z-v2z
    Dzm = xp.where(xp.abs(s0z) <= xp.abs(s1z), v1z - 0.5*s0z, v1z + 0.5*s1z)
    Dzp = xp.where(xp.abs(s1z) <= xp.abs(s2z), v2z - 0.5*s1z, v2z + 0.5*s2z)

    return Dxm, Dxp, Dym, Dyp, Dzm, Dzp


def _godunov_3d(Dxm, Dxp, Dym, Dyp, Dzm, Dzp, sgn):
    xp = cp.get_array_module(Dxm) if _CUPY_AVAILABLE else np
    ax_p = xp.maximum(Dxm, 0)**2;  bx_p = xp.minimum(Dxp, 0)**2
    ay_p = xp.maximum(Dym, 0)**2;  by_p = xp.minimum(Dyp, 0)**2
    az_p = xp.maximum(Dzm, 0)**2;  bz_p = xp.minimum(Dzp, 0)**2
    ax_n = xp.minimum(Dxm, 0)**2;  bx_n = xp.maximum(Dxp, 0)**2
    ay_n = xp.minimum(Dym, 0)**2;  by_n = xp.maximum(Dyp, 0)**2
    az_n = xp.minimum(Dzm, 0)**2;  bz_n = xp.maximum(Dzp, 0)**2
    Hp = xp.sqrt(xp.maximum(ax_p, bx_p) + xp.maximum(ay_p, by_p) + xp.maximum(az_p, bz_p)) - 1.0
    Hn = xp.sqrt(xp.maximum(ax_n, bx_n) + xp.maximum(ay_n, by_n) + xp.maximum(az_n, bz_n)) - 1.0
    return xp.where(sgn > 0, Hp, Hn)


def _smooth_heaviside_sum_3d(phi, eps, xp):
    """Sum of smooth Heaviside H(phi) over all grid nodes."""
    H = xp.empty_like(phi)
    mid = (phi > -eps) & (phi < eps)
    H[:] = xp.where(phi <= -eps, 0.0, 1.0)
    if bool(mid.any()):
        H[mid] = 0.5 + phi[mid] / (2.0 * eps)
    return float(H.sum())


def _volume_correct_3d(phi, V0, eps, xp, max_iter=60):
    """
    Shift phi by a constant c so that sum(H(phi+c)) == V0.
    Uses bisection; converges in ~50 iters to < 1e-10 tolerance.
    """
    h = eps / 1.5
    c_lo, c_hi = -15.0 * h, 15.0 * h
    for _ in range(max_iter):
        c_mid = 0.5 * (c_lo + c_hi)
        if _smooth_heaviside_sum_3d(phi + c_mid, eps, xp) < V0:
            c_lo = c_mid
        else:
            c_hi = c_mid
        if c_hi - c_lo < 1e-12:
            break
    return phi + 0.5 * (c_lo + c_hi)


def min2010_reinitialize_gpu_3d(phi_gpu, dx, dy, dz, max_iter=15, tol=1e-5,
                                 volume_correct=True):
    """
    ENO2 reinitialization on GPU (CuPy) or CPU (numpy), 3D.

    Parameters
    ----------
    phi_gpu  : CuPy (or numpy) array of shape (nz_nodes, ny_nodes, nx_nodes).
    dx, dy, dz : grid spacings.
    max_iter : maximum pseudo-time iterations.
    tol      : convergence tolerance on mean |∇φ| - 1.

    Returns
    -------
    phi_new : same type and shape as phi_gpu, reinitialized.
    iters   : number of iterations performed.
    """
    xp = cp.get_array_module(phi_gpu) if _CUPY_AVAILABLE else np

    eps = 1.5 * min(dx, dy, dz)
    phi = phi_gpu.copy()

    # Volume (H-sum) before reinit — used for correction afterwards
    V0 = _smooth_heaviside_sum_3d(phi, eps, xp)

    sgn = phi / xp.sqrt(phi**2 + eps**2)
    dt  = 0.5 * min(dx, dy, dz)

    # Subcell fix: find closest zero crossing for interface nodes
    phi_fixed = xp.zeros_like(phi)
    is_fixed  = xp.zeros(phi.shape, dtype=bool)
    c = phi[1:-1, 1:-1, 1:-1]
    best = xp.full(c.shape, xp.inf)
    bval = xp.zeros_like(c)
    for neigh, h in [
        (phi[1:-1, 1:-1, :-2], dx), (phi[1:-1, 1:-1,  2:], dx),
        (phi[1:-1,  :-2, 1:-1], dy), (phi[1:-1,   2:, 1:-1], dy),
        (phi[ :-2, 1:-1, 1:-1], dz), (phi[  2:, 1:-1, 1:-1], dz),
    ]:
        cross = (neigh * c) <= 0
        theta = xp.abs(c) / (xp.abs(c) + xp.abs(neigh) + 1e-300)
        d_abs = theta * h
        d_val = xp.sign(c) * d_abs
        imp   = cross & (d_abs < best)
        best  = xp.where(imp, d_abs, best)
        bval  = xp.where(imp, d_val, bval)
    has = xp.isfinite(best)
    phi_fixed[1:-1, 1:-1, 1:-1] = xp.where(has, bval, 0.0)
    is_fixed [1:-1, 1:-1, 1:-1] = has

    iters = 0
    for it in range(max_iter):
        Dxm, Dxp, Dym, Dyp, Dzm, Dzp = _eno2_deriv_3d(phi, dx, dy, dz)
        H1 = _godunov_3d(Dxm, Dxp, Dym, Dyp, Dzm, Dzp, sgn)

        phi_star = phi - dt * sgn * H1
        phi_star = xp.where(is_fixed, phi_fixed, phi_star)

        Dxm2, Dxp2, Dym2, Dyp2, Dzm2, Dzp2 = _eno2_deriv_3d(phi_star, dx, dy, dz)
        H2 = _godunov_3d(Dxm2, Dxp2, Dym2, Dyp2, Dzm2, Dzp2, sgn)

        phi_new = phi - 0.5 * dt * sgn * (H1 + H2)
        phi_new = xp.where(is_fixed,              phi_fixed, phi_new)
        phi_new = xp.where(~is_fixed & (sgn > 0), xp.maximum(phi_new, 0), phi_new)
        phi_new = xp.where(~is_fixed & (sgn < 0), xp.minimum(phi_new, 0), phi_new)

        phi = phi_new
        iters = it + 1

        non_fixed = ~is_fixed
        if non_fixed.any():
            gz, gy, gx = xp.gradient(phi, dz, dy, dx, edge_order=2)
            err = xp.abs(xp.sqrt(gx**2 + gy**2 + gz**2)[non_fixed] - 1.0)
            mean_err = float(err.mean())
        else:
            mean_err = 0.0

        if mean_err < tol:
            break

    # Volume correction: shift phi so enclosed volume matches pre-reinit value
    if volume_correct:
        phi = _volume_correct_3d(phi, V0, eps, xp)

    return phi, iters
