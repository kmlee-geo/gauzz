"""
reinit_mpi_3d.py — Distributed MPI + GPU ENO2 reinitialization for 3D level-set fields.

Architecture
------------
The (nz+1)×(ny+1)×(nx+1) grid is partitioned into z-slabs (one per rank).
Each rank runs ENO2 on its local slab using GPU (CuPy) if available.
Ghost plane exchange (2 planes in z) goes through CPU memory because GPU-aware
MPI is disabled in this environment (PETSC_OPTIONS="-use_gpu_aware_mpi 0").

Per reinitialize() call:
  1. Alltoallv fwd  — FEniCS DOF layout → z-slab    (CPU, O(N/P))
  2. Ghost exchange (CPU) + subcell fix               (CPU, once)
  3. Transfer slab to GPU                             (CPU → GPU, once)
  4. ENO2 iterations on GPU, ghost exchange via CPU   (GPU compute, O(ny*nx) comm)
  5. Transfer result back to CPU                      (GPU → CPU, once)
  6. Alltoallv bwd  — z-slab → FEniCS DOF layout     (CPU, O(N/P))

Without GPU (use_gpu=False or CuPy unavailable):
  Steps 3-5 are skipped; ENO2 runs on CPU with numpy (same code path).

Usage
-----
    from reinit_mpi_3d import DistributedReinit3D

    reinit = DistributedReinit3D(scalar_space, comm, nx, ny, nz, width, depth, height)
    reinit(phi, max_iter=15, tol=1e-5)        # GPU used if available
    reinit(phi, max_iter=15, use_gpu=False)   # force CPU
"""

import numpy as np
from mpi4py import MPI

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


class DistributedReinit3D:
    """
    Distributed ENO2 reinitialization for FEniCS P1 level-set fields on a BoxMesh.

    Parameters
    ----------
    scalar_space : dolfin.FunctionSpace   P1 on a BoxMesh
    comm         : mpi4py communicator
    nx, ny, nz   : number of mesh cells in x, y, z
    width        : domain extent in x
    depth        : domain extent in y
    height       : domain extent in z
    use_gpu      : use CuPy for ENO2 compute if True and CuPy available (default True)
    """

    # ------------------------------------------------------------------
    # Construction & setup
    # ------------------------------------------------------------------

    def __init__(self, scalar_space, comm, nx, ny, nz, width, depth, height, use_gpu=True):
        self.comm      = comm
        self.rank      = comm.Get_rank()
        self.size      = comm.Get_size()
        self.nx        = nx
        self.ny        = ny
        self.nz        = nz
        self.dx        = width  / nx
        self.dy        = depth  / ny
        self.dz        = height / nz
        self.nx_nodes  = nx + 1
        self.ny_nodes  = ny + 1
        self.nz_nodes  = nz + 1
        self.use_gpu   = use_gpu and _CUPY_AVAILABLE

        # z-slab partition of the (nz+1) global z-planes
        nz_n            = self.nz_nodes
        spp             = (nz_n + self.size - 1) // self.size
        self._spp       = spp
        self.slice_start = min(self.rank * spp, nz_n)
        self.slice_end   = min((self.rank + 1) * spp, nz_n)
        self.my_slices   = self.slice_end - self.slice_start  # always >= 0
        import math
        self._n_active   = min(self.size, math.ceil(nz_n / spp))  # ranks with actual data

        self._setup_maps(scalar_space)

    def _slice_to_rank(self, iz_arr):
        return np.minimum(iz_arr // self._spp, self.size - 1).astype(np.int32)

    def _setup_maps(self, scalar_space):
        """Pre-compute Alltoallv patterns (called once at construction)."""
        comm = self.comm
        size = self.size

        lr      = scalar_space.dofmap().ownership_range()
        n_owned = int(lr[1] - lr[0])
        coords  = scalar_space.tabulate_dof_coordinates()[:n_owned]

        ix = np.round(coords[:, 0] / self.dx).astype(np.int32)
        iy = np.round(coords[:, 1] / self.dy).astype(np.int32)
        iz = np.round(coords[:, 2] / self.dz).astype(np.int32)
        np.clip(ix, 0, self.nx, out=ix)
        np.clip(iy, 0, self.ny, out=iy)
        np.clip(iz, 0, self.nz, out=iz)
        self._n_owned = n_owned

        dest  = self._slice_to_rank(iz)
        order = np.argsort(dest, kind='stable')
        self._fwd_order = order

        send_iz = np.ascontiguousarray(iz[order], dtype=np.int32)
        send_iy = np.ascontiguousarray(iy[order], dtype=np.int32)
        send_ix = np.ascontiguousarray(ix[order], dtype=np.int32)

        send_counts = np.bincount(dest, minlength=size).astype(np.int32)
        send_displs = np.zeros(size, dtype=np.int32)
        if size > 1:
            send_displs[1:] = np.cumsum(send_counts[:-1])
        self._fwd_send_counts = send_counts
        self._fwd_send_displs = send_displs

        recv_counts = np.zeros(size, dtype=np.int32)
        comm.Alltoall(send_counts, recv_counts)
        recv_displs = np.zeros(size, dtype=np.int32)
        if size > 1:
            recv_displs[1:] = np.cumsum(recv_counts[:-1])
        self._fwd_recv_counts = recv_counts
        self._fwd_recv_displs = recv_displs
        self._fwd_total_recv  = int(recv_counts.sum())

        recv_iz = np.zeros(self._fwd_total_recv, dtype=np.int32)
        recv_iy = np.zeros(self._fwd_total_recv, dtype=np.int32)
        recv_ix = np.zeros(self._fwd_total_recv, dtype=np.int32)
        comm.Alltoallv([send_iz, send_counts, send_displs, MPI.INT],
                       [recv_iz, recv_counts, recv_displs, MPI.INT])
        comm.Alltoallv([send_iy, send_counts, send_displs, MPI.INT],
                       [recv_iy, recv_counts, recv_displs, MPI.INT])
        comm.Alltoallv([send_ix, send_counts, send_displs, MPI.INT],
                       [recv_ix, recv_counts, recv_displs, MPI.INT])

        self._recv_iz           = recv_iz
        self._recv_iy           = recv_iy
        self._recv_ix           = recv_ix
        self._local_recv_slices = recv_iz - self.slice_start + 2   # +2 for bottom ghost planes

        self._bwd_send_counts = recv_counts.copy()
        self._bwd_send_displs = recv_displs.copy()
        self._bwd_recv_counts = send_counts.copy()
        self._bwd_recv_displs = send_displs.copy()

    # ------------------------------------------------------------------
    # Ghost plane exchange — CPU version
    # ------------------------------------------------------------------

    def _exchange_ghosts_cpu(self, g):
        """
        g: CPU numpy array (2 + my_slices + 2, ny_nodes, nx_nodes).
        Exchanges 2 ghost planes with z-neighbours; edge-pads at domain boundaries.
        """
        rank = self.rank;  ms = self.my_slices
        n_active = self._n_active
        ny_n = self.ny_nodes;  nx_n = self.nx_nodes

        if ms == 0:
            return  # empty rank, nothing to exchange

        has_top_nbr = (rank > 0)
        has_bot_nbr = (rank < n_active - 1)

        reqs  = []
        r_top = np.empty((2, ny_n, nx_n), dtype=np.float64)
        r_bot = np.empty((2, ny_n, nx_n), dtype=np.float64)

        if has_top_nbr:
            reqs.append(self.comm.Isend(np.ascontiguousarray(g[2:4, :, :]),
                                        dest=rank - 1, tag=1))
            reqs.append(self.comm.Irecv(r_top, source=rank - 1, tag=0))
        if has_bot_nbr:
            reqs.append(self.comm.Isend(np.ascontiguousarray(g[2+ms-2:2+ms, :, :]),
                                        dest=rank + 1, tag=0))
            reqs.append(self.comm.Irecv(r_bot, source=rank + 1, tag=1))

        if reqs:
            MPI.Request.Waitall(reqs)

        if has_top_nbr:
            g[:2, :, :] = r_top
        else:
            g[0, :, :] = g[2, :, :];  g[1, :, :] = g[2, :, :]

        if has_bot_nbr:
            g[2+ms:2+ms+2, :, :] = r_bot
        else:
            g[2+ms, :, :]   = g[2+ms-1, :, :]
            g[2+ms+1, :, :] = g[2+ms-1, :, :]

    # ------------------------------------------------------------------
    # Ghost plane exchange — GPU version
    # Ghost planes are extracted to CPU, communicated via MPI, put back to GPU.
    # ------------------------------------------------------------------

    def _exchange_ghosts_gpu(self, g_gpu):
        """
        g_gpu: CuPy array (2 + my_slices + 2, ny_nodes, nx_nodes).
        Send/recv buffers pass through CPU memory (GPU-aware MPI disabled).
        """
        rank = self.rank;  ms = self.my_slices
        n_active = self._n_active
        ny_n = self.ny_nodes;  nx_n = self.nx_nodes

        if ms == 0:
            return  # empty rank, nothing to exchange

        has_top_nbr = (rank > 0)
        has_bot_nbr = (rank < n_active - 1)

        reqs  = []
        r_top = np.empty((2, ny_n, nx_n), dtype=np.float64)
        r_bot = np.empty((2, ny_n, nx_n), dtype=np.float64)

        if has_top_nbr:
            s_top = cp.asnumpy(g_gpu[2:4, :, :])
            reqs.append(self.comm.Isend(s_top, dest=rank - 1, tag=1))
            reqs.append(self.comm.Irecv(r_top, source=rank - 1, tag=0))
        if has_bot_nbr:
            s_bot = cp.asnumpy(g_gpu[2+ms-2:2+ms, :, :])
            reqs.append(self.comm.Isend(s_bot, dest=rank + 1, tag=0))
            reqs.append(self.comm.Irecv(r_bot, source=rank + 1, tag=1))

        if reqs:
            MPI.Request.Waitall(reqs)

        if has_top_nbr:
            g_gpu[:2, :, :] = cp.asarray(r_top)
        else:
            g_gpu[0, :, :] = g_gpu[2, :, :];  g_gpu[1, :, :] = g_gpu[2, :, :]

        if has_bot_nbr:
            g_gpu[2+ms:2+ms+2, :, :] = cp.asarray(r_bot)
        else:
            g_gpu[2+ms, :, :]   = g_gpu[2+ms-1, :, :]
            g_gpu[2+ms+1, :, :] = g_gpu[2+ms-1, :, :]

    # ------------------------------------------------------------------
    # ENO2 kernels — array-library agnostic (numpy or cupy)
    # ------------------------------------------------------------------

    @staticmethod
    def _eno2_deriv(g, dx, dy, dz):
        """
        ENO2 derivatives on local slab g (CPU numpy or GPU cupy).
        g shape: (2 + my_slices + 2, ny_nodes, nx_nodes). Ghost planes must be valid.

        x, y directions: edge-padded (no MPI, full xy-planes are owned).
        z direction: uses ghost planes (MPI ghost exchange already done).

        Returns Dxm, Dxp, Dym, Dyp, Dzm, Dzp of shape (my_slices, ny_nodes, nx_nodes).
        """
        xp = cp.get_array_module(g) if _CUPY_AVAILABLE else np

        # x-derivatives: pad in x (axis=2)
        gp  = xp.pad(g, ((0, 0), (0, 0), (2, 2)), mode='edge')
        gim2 = gp[:, :, :-4]; gim1 = gp[:, :, 1:-3]
        gc   = gp[:, :, 2:-2]; gip1 = gp[:, :, 3:-1]; gip2 = gp[:, :, 4:]
        v0 = (gim1-gim2)/dx;  v1 = (gc-gim1)/dx
        v2 = (gip1-gc  )/dx;  v3 = (gip2-gip1)/dx
        s0 = v1-v0;  s1 = v2-v1;  s2 = v3-v2
        Dxm = xp.where(xp.abs(s0) <= xp.abs(s1), v1 - 0.5*s0, v1 + 0.5*s1)[2:-2, :, :]
        Dxp = xp.where(xp.abs(s1) <= xp.abs(s2), v2 - 0.5*s1, v2 + 0.5*s2)[2:-2, :, :]

        # y-derivatives: pad in y (axis=1)
        gp2  = xp.pad(g, ((0, 0), (2, 2), (0, 0)), mode='edge')
        gjm2 = gp2[:, :-4, :]; gjm1 = gp2[:, 1:-3, :]
        gjc  = gp2[:, 2:-2, :]; gjp1 = gp2[:, 3:-1, :]; gjp2 = gp2[:, 4:, :]
        v0y = (gjm1-gjm2)/dy;  v1y = (gjc-gjm1)/dy
        v2y = (gjp1-gjc )/dy;  v3y = (gjp2-gjp1)/dy
        s0y = v1y-v0y;  s1y = v2y-v1y;  s2y = v3y-v2y
        Dym = xp.where(xp.abs(s0y) <= xp.abs(s1y), v1y - 0.5*s0y, v1y + 0.5*s1y)[2:-2, :, :]
        Dyp = xp.where(xp.abs(s1y) <= xp.abs(s2y), v2y - 0.5*s1y, v2y + 0.5*s2y)[2:-2, :, :]

        # z-derivatives: use ghost planes (axis=0)
        gkm2 = g[:-4, :, :]; gkm1 = g[1:-3, :, :]
        gkc  = g[2:-2, :, :]; gkp1 = g[3:-1, :, :]; gkp2 = g[4:, :, :]
        v0z = (gkm1-gkm2)/dz;  v1z = (gkc-gkm1)/dz
        v2z = (gkp1-gkc )/dz;  v3z = (gkp2-gkp1)/dz
        s0z = v1z-v0z;  s1z = v2z-v1z;  s2z = v3z-v2z
        Dzm = xp.where(xp.abs(s0z) <= xp.abs(s1z), v1z - 0.5*s0z, v1z + 0.5*s1z)
        Dzp = xp.where(xp.abs(s1z) <= xp.abs(s2z), v2z - 0.5*s1z, v2z + 0.5*s2z)

        return Dxm, Dxp, Dym, Dyp, Dzm, Dzp

    @staticmethod
    def _godunov(Dxm, Dxp, Dym, Dyp, Dzm, Dzp, sgn):
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

    # ------------------------------------------------------------------
    # Subcell fix — always on CPU (runs once per reinit call)
    # ------------------------------------------------------------------

    def _subcell_fix(self, g_cpu):
        """
        g_cpu: CPU numpy array (2 + my_slices + 2, ny_nodes, nx_nodes).
        Ghost planes must be valid (call _exchange_ghosts_cpu first).
        Returns (phi_fixed_cpu, is_fixed_cpu) — numpy, same shape as g_cpu.

        Checks all 6 face-neighbours (±x, ±y, ±z). For each interior node
        where a sign change occurs across a neighbour, records the subcell
        distance to the zero crossing.
        """
        ms = self.my_slices;  dx = self.dx;  dy = self.dy;  dz = self.dz

        phi_fixed = np.zeros_like(g_cpu)
        is_fixed  = np.zeros_like(g_cpu, dtype=bool)

        # Interior nodes: exclude x-boundary (1:-1), y-boundary (1:-1),
        # z-ghost planes handled by z-neighbor access patterns below.
        c    = g_cpu[2:2+ms, 1:-1, 1:-1]
        best = np.full(c.shape, np.inf)
        bval = np.zeros(c.shape)

        for neigh, h in (
            (g_cpu[2:2+ms, 1:-1, :-2],  dx),   # -x neighbour
            (g_cpu[2:2+ms, 1:-1,  2:],  dx),   # +x neighbour
            (g_cpu[2:2+ms,  :-2, 1:-1], dy),   # -y neighbour
            (g_cpu[2:2+ms,   2:, 1:-1], dy),   # +y neighbour
            (g_cpu[1:1+ms, 1:-1, 1:-1], dz),   # -z neighbour (may be ghost)
            (g_cpu[3:3+ms, 1:-1, 1:-1], dz),   # +z neighbour (may be ghost)
        ):
            cross = (neigh * c) <= 0
            theta = np.abs(c) / (np.abs(c) + np.abs(neigh) + 1e-300)
            d_abs = theta * h
            d_val = np.sign(c) * d_abs
            imp   = cross & (d_abs < best)
            best  = np.where(imp, d_abs, best)
            bval  = np.where(imp, d_val, bval)

        has = np.isfinite(best)
        phi_fixed[2:2+ms, 1:-1, 1:-1] = np.where(has, bval, 0.0)
        is_fixed [2:2+ms, 1:-1, 1:-1] = has
        return phi_fixed, is_fixed

    # ------------------------------------------------------------------
    # Main reinitialization
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Volume correction helpers
    # ------------------------------------------------------------------

    def _local_H_sum(self, slab, c):
        """Sum of H(slab + c) over the owned z-slab (CPU numpy, no ghosts)."""
        eps = 1.5 * min(self.dx, self.dy, self.dz)
        p = slab + c
        H = np.where(p <= -eps, 0.0, np.where(p >= eps, 1.0,
                     0.5 + p / (2.0 * eps)))
        return float(H.sum())

    def _global_H_sum(self, slab, c):
        """Allreduce H sum across all ranks."""
        return self.comm.allreduce(self._local_H_sum(slab, c), op=MPI.SUM)

    def _volume_correct(self, slab, V0, max_iter=60):
        """
        Find c via distributed bisection so that global sum(H(slab+c)) == V0,
        then return slab + c.  Each bisection step costs one Allreduce.
        """
        h = min(self.dx, self.dy, self.dz)
        c_lo, c_hi = -15.0 * h, 15.0 * h
        for _ in range(max_iter):
            c_mid = 0.5 * (c_lo + c_hi)
            if self._global_H_sum(slab, c_mid) < V0:
                c_lo = c_mid
            else:
                c_hi = c_mid
            if c_hi - c_lo < 1e-12:
                break
        return slab + 0.5 * (c_lo + c_hi)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, phi, max_iter=15, tol=1e-5, verbose=False, use_gpu=None):
        gpu = self.use_gpu if use_gpu is None else (use_gpu and _CUPY_AVAILABLE)
        return self._reinitialize(phi, max_iter=max_iter, tol=tol,
                                  verbose=verbose, use_gpu=gpu)

    def reinitialize(self, phi, max_iter=15, tol=1e-5, verbose=False, use_gpu=None):
        gpu = self.use_gpu if use_gpu is None else (use_gpu and _CUPY_AVAILABLE)
        return self._reinitialize(phi, max_iter=max_iter, tol=tol,
                                  verbose=verbose, use_gpu=gpu)

    def _reinitialize(self, phi, max_iter, tol, verbose, use_gpu):
        comm = self.comm
        ms   = self.my_slices
        dx   = self.dx;  dy = self.dy;  dz = self.dz

        # ---- 1. Forward Alltoallv: FEniCS DOFs → z-slab (CPU) -----------
        lr   = phi.vector().local_range()
        n    = int(lr[1] - lr[0])
        vals = phi.vector().get_local()[:n]

        send_v = np.ascontiguousarray(vals[self._fwd_order], dtype=np.float64)
        recv_v = np.empty(self._fwd_total_recv, dtype=np.float64)
        comm.Alltoallv(
            [send_v, self._fwd_send_counts, self._fwd_send_displs, MPI.DOUBLE],
            [recv_v, self._fwd_recv_counts, self._fwd_recv_displs, MPI.DOUBLE])

        g_cpu = np.zeros((2 + ms + 2, self.ny_nodes, self.nx_nodes), dtype=np.float64)
        g_cpu[self._local_recv_slices, self._recv_iy, self._recv_ix] = recv_v

        # Volume target: global H-sum of owned slab before reinit
        owned_pre = g_cpu[2:2+ms, :, :] if ms > 0 else np.zeros((0,), dtype=np.float64)
        V0 = self._global_H_sum(owned_pre, 0.0)

        # ---- 2. Ghost exchange + subcell fix (always CPU) ----------------
        self._exchange_ghosts_cpu(g_cpu)
        phi_fixed_cpu, is_fixed_cpu = self._subcell_fix(g_cpu)

        # ---- 3. Transfer to GPU (if enabled) ----------------------------
        if use_gpu:
            g            = cp.asarray(g_cpu)
            phi_fixed    = cp.asarray(phi_fixed_cpu)
            is_fixed     = cp.asarray(is_fixed_cpu)
            xp           = cp
            exchange_fn  = self._exchange_ghosts_gpu
        else:
            g            = g_cpu
            phi_fixed    = phi_fixed_cpu
            is_fixed     = is_fixed_cpu
            xp           = np
            exchange_fn  = self._exchange_ghosts_cpu

        owned           = g[2:2+ms, :, :]
        is_fixed_owned  = is_fixed [2:2+ms, :, :]
        phi_fixed_owned = phi_fixed[2:2+ms, :, :]

        if ms > 0:
            eps = 1.5 * min(dx, dy, dz)
            sgn = owned / xp.sqrt(owned**2 + eps**2)
        dt  = 0.5 * min(dx, dy, dz)

        # ---- 4. ENO2 iterations (GPU or CPU) ----------------------------
        for it in range(max_iter):
            if ms > 0:
                exchange_fn(g)
                Dxm, Dxp, Dym, Dyp, Dzm, Dzp = self._eno2_deriv(g, dx, dy, dz)
                H1 = self._godunov(Dxm, Dxp, Dym, Dyp, Dzm, Dzp, sgn)

                g_star = owned - dt * sgn * H1
                g_star = xp.where(is_fixed_owned, phi_fixed_owned, g_star)
                g[2:2+ms, :, :] = g_star

                exchange_fn(g)
                Dxm2, Dxp2, Dym2, Dyp2, Dzm2, Dzp2 = self._eno2_deriv(g, dx, dy, dz)
                H2 = self._godunov(Dxm2, Dxp2, Dym2, Dyp2, Dzm2, Dzp2, sgn)

                g_new = owned - 0.5 * dt * sgn * (H1 + H2)
                g_new = xp.where(is_fixed_owned,              phi_fixed_owned, g_new)
                g_new = xp.where(~is_fixed_owned & (sgn > 0), xp.maximum(g_new, 0), g_new)
                g_new = xp.where(~is_fixed_owned & (sgn < 0), xp.minimum(g_new, 0), g_new)
                g[2:2+ms, :, :] = g_new
                owned = g_new

                # Convergence check: distributed mean of |∇φ| - 1
                non_fixed = ~is_fixed_owned
                edge_ord = 1 if owned.shape[0] < 3 else 2
                if non_fixed.any() and owned.shape[0] >= edge_ord + 1:
                    gz_o, gy_o, gx_o = xp.gradient(owned, dz, dy, dx, edge_order=edge_ord)
                    err = xp.abs(xp.sqrt(gx_o**2 + gy_o**2 + gz_o**2)[non_fixed] - 1.0)
                    s_local = float(err.sum())
                    n_local = int(err.size)
                else:
                    s_local = 0.0;  n_local = 0
            else:
                s_local = 0.0;  n_local = 0

            s_global = comm.allreduce(s_local, op=MPI.SUM)
            n_global = comm.allreduce(n_local, op=MPI.SUM)
            mean_err = s_global / max(n_global, 1)

            if verbose and self.rank == 0:
                print(f"  reinit iter {it+1:3d}: mean |∇φ|-1 = {mean_err:.6f}", flush=True)

            if mean_err < tol:
                break

        # ---- 5. Transfer back to CPU (if on GPU) ------------------------
        if use_gpu:
            g_cpu = cp.asnumpy(g)

        # ---- 5b. Volume correction: shift owned slab to preserve pre-reinit volume ----
        if ms > 0:
            g_cpu[2:2+ms, :, :] = self._volume_correct(g_cpu[2:2+ms, :, :], V0)

        # ---- 6. Backward Alltoallv: z-slab → FEniCS DOFs (CPU) ----------
        bwd_v    = g_cpu[self._local_recv_slices, self._recv_iy, self._recv_ix]
        recv_bwd = np.empty(n, dtype=np.float64)
        comm.Alltoallv(
            [np.ascontiguousarray(bwd_v),
             self._bwd_send_counts, self._bwd_send_displs, MPI.DOUBLE],
            [recv_bwd,
             self._bwd_recv_counts, self._bwd_recv_displs, MPI.DOUBLE])

        new_vals = np.empty(n, dtype=np.float64)
        new_vals[self._fwd_order] = recv_bwd

        full = phi.vector().get_local()
        full[:n] = new_vals
        phi.vector().set_local(full)
        phi.vector().apply("insert")
        phi.vector().update_ghost_values()
