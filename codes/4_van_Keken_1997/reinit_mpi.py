"""
reinit_mpi.py — Distributed MPI + GPU ENO2 reinitialization.

Architecture
------------
The (ny+1)×(nx+1) grid is partitioned into horizontal row strips (one per rank).
Each rank runs ENO2 on its local strip using GPU (CuPy) if available.
Ghost row exchange (2 rows) goes through CPU memory because GPU-aware MPI is
disabled in this environment (PETSC_OPTIONS="-use_gpu_aware_mpi 0").

Per reinitialize() call:
  1. Alltoallv fwd  — FEniCS DOF layout → row-strip   (CPU, O(N/P))
  2. Ghost exchange (CPU) + subcell fix                (CPU, once)
  3. Transfer row-strip to GPU                         (CPU → GPU, once)
  4. ENO2 iterations on GPU, ghost exchange via CPU    (GPU compute, O(nx) comm)
  5. Transfer result back to CPU                       (GPU → CPU, once)
  6. Alltoallv bwd  — row-strip → FEniCS DOF layout   (CPU, O(N/P))

Without GPU (use_gpu=False or CuPy unavailable):
  Steps 3-5 are skipped; ENO2 runs on CPU with numpy (same code path).

Usage
-----
    from reinit_mpi import DistributedReinit

    reinit = DistributedReinit(scalar_space, comm, nx, ny, width, height)
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


class DistributedReinit:
    """
    Distributed ENO2 reinitialization for FEniCS P1 level-set fields.

    Parameters
    ----------
    scalar_space : dolfin.FunctionSpace   P1 on a RectangleMesh
    comm         : mpi4py communicator
    nx, ny       : number of mesh cells in x and y
    width, height: domain dimensions
    use_gpu      : use CuPy for ENO2 compute if True and CuPy available (default True)
    """

    # ------------------------------------------------------------------
    # Construction & setup
    # ------------------------------------------------------------------

    def __init__(self, scalar_space, comm, nx, ny, width, height, use_gpu=True):
        self.comm     = comm
        self.rank     = comm.Get_rank()
        self.size     = comm.Get_size()
        self.nx       = nx
        self.ny       = ny
        self.dx       = width  / nx
        self.dy       = height / ny
        self.nx_nodes = nx + 1
        self.ny_nodes = ny + 1
        self.use_gpu  = use_gpu and _CUPY_AVAILABLE

        # Row-striped partition of the (ny+1) global rows
        ny_n       = self.ny_nodes
        rpp        = (ny_n + self.size - 1) // self.size
        self._rpp  = rpp
        self.row_start = self.rank * rpp
        self.row_end   = min((self.rank + 1) * rpp, ny_n)
        self.my_rows   = self.row_end - self.row_start

        self._setup_maps(scalar_space)

    def _row_to_rank(self, iy_arr):
        return np.minimum(iy_arr // self._rpp, self.size - 1).astype(np.int32)

    def _setup_maps(self, scalar_space):
        """Pre-compute Alltoallv patterns (called once at construction)."""
        comm = self.comm
        size = self.size

        lr      = scalar_space.dofmap().ownership_range()
        n_owned = int(lr[1] - lr[0])
        coords  = scalar_space.tabulate_dof_coordinates()[:n_owned]

        ix = np.round(coords[:, 0] / self.dx).astype(np.int32)
        iy = np.round(coords[:, 1] / self.dy).astype(np.int32)
        np.clip(ix, 0, self.nx, out=ix)
        np.clip(iy, 0, self.ny, out=iy)
        self._n_owned = n_owned

        dest  = self._row_to_rank(iy)
        order = np.argsort(dest, kind='stable')
        self._fwd_order = order

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

        recv_iy = np.zeros(self._fwd_total_recv, dtype=np.int32)
        recv_ix = np.zeros(self._fwd_total_recv, dtype=np.int32)
        comm.Alltoallv([send_iy, send_counts, send_displs, MPI.INT],
                       [recv_iy, recv_counts, recv_displs, MPI.INT])
        comm.Alltoallv([send_ix, send_counts, send_displs, MPI.INT],
                       [recv_ix, recv_counts, recv_displs, MPI.INT])

        self._recv_iy          = recv_iy
        self._recv_ix          = recv_ix
        self._local_recv_rows  = recv_iy - self.row_start + 2   # +2 for top ghost

        self._bwd_send_counts = recv_counts.copy()
        self._bwd_send_displs = recv_displs.copy()
        self._bwd_recv_counts = send_counts.copy()
        self._bwd_recv_displs = send_displs.copy()

    # ------------------------------------------------------------------
    # Ghost row exchange — CPU version (used before GPU transfer)
    # ------------------------------------------------------------------

    def _exchange_ghosts_cpu(self, g):
        """
        g is a CPU numpy array of shape (2 + my_rows + 2, nx_nodes).
        Exchanges 2 ghost rows with neighbours; edge-pads at domain boundaries.
        """
        rank = self.rank;  size = self.size;  mr = self.my_rows

        reqs  = []
        r_top = np.empty((2, self.nx_nodes), dtype=np.float64)
        r_bot = np.empty((2, self.nx_nodes), dtype=np.float64)

        if rank > 0:
            reqs.append(self.comm.Isend(np.ascontiguousarray(g[2:4, :]),
                                        dest=rank - 1, tag=1))
            reqs.append(self.comm.Irecv(r_top, source=rank - 1, tag=0))
        if rank < size - 1:
            reqs.append(self.comm.Isend(np.ascontiguousarray(g[2+mr-2:2+mr, :]),
                                        dest=rank + 1, tag=0))
            reqs.append(self.comm.Irecv(r_bot, source=rank + 1, tag=1))

        if reqs:
            MPI.Request.Waitall(reqs)

        if rank > 0:
            g[:2, :] = r_top
        else:
            g[0, :] = g[2, :];  g[1, :] = g[2, :]

        if rank < size - 1:
            g[2+mr:2+mr+2, :] = r_bot
        else:
            g[2+mr, :] = g[2+mr-1, :];  g[2+mr+1, :] = g[2+mr-1, :]

    # ------------------------------------------------------------------
    # Ghost row exchange — GPU version
    # Ghost rows are extracted to CPU, communicated via MPI, put back to GPU.
    # ------------------------------------------------------------------

    def _exchange_ghosts_gpu(self, g_gpu):
        """
        g_gpu is a CuPy array of shape (2 + my_rows + 2, nx_nodes).
        Send/recv buffers pass through CPU memory (GPU-aware MPI is disabled).
        """
        rank = self.rank;  size = self.size;  mr = self.my_rows

        reqs  = []
        r_top = np.empty((2, self.nx_nodes), dtype=np.float64)
        r_bot = np.empty((2, self.nx_nodes), dtype=np.float64)

        if rank > 0:
            s_top = cp.asnumpy(g_gpu[2:4, :])          # GPU → CPU
            reqs.append(self.comm.Isend(s_top, dest=rank - 1, tag=1))
            reqs.append(self.comm.Irecv(r_top, source=rank - 1, tag=0))
        if rank < size - 1:
            s_bot = cp.asnumpy(g_gpu[2+mr-2:2+mr, :])  # GPU → CPU
            reqs.append(self.comm.Isend(s_bot, dest=rank + 1, tag=0))
            reqs.append(self.comm.Irecv(r_bot, source=rank + 1, tag=1))

        if reqs:
            MPI.Request.Waitall(reqs)

        if rank > 0:
            g_gpu[:2, :] = cp.asarray(r_top)            # CPU → GPU
        else:
            g_gpu[0, :] = g_gpu[2, :];  g_gpu[1, :] = g_gpu[2, :]

        if rank < size - 1:
            g_gpu[2+mr:2+mr+2, :] = cp.asarray(r_bot)  # CPU → GPU
        else:
            g_gpu[2+mr, :] = g_gpu[2+mr-1, :]
            g_gpu[2+mr+1, :] = g_gpu[2+mr-1, :]

    # ------------------------------------------------------------------
    # ENO2 kernels — array-library agnostic (numpy or cupy)
    # ------------------------------------------------------------------

    @staticmethod
    def _eno2_deriv(g, dx, dy):
        """
        ENO2 derivatives on local grid g (CPU numpy or GPU cupy).
        g shape: (2 + my_rows + 2, nx_nodes).  Ghost rows must be valid.
        Returns Dx_minus, Dx_plus, Dy_minus, Dy_plus of shape (my_rows, nx_nodes).
        """
        xp = cp.get_array_module(g) if _CUPY_AVAILABLE else np

        # x-derivatives: column-pad (no MPI)
        gp   = xp.pad(g, ((0, 0), (2, 2)), mode='edge')
        gim2 = gp[:, :-4]; gim1 = gp[:, 1:-3]
        gc   = gp[:, 2:-2]; gip1 = gp[:, 3:-1]; gip2 = gp[:, 4:]

        v0 = (gim1-gim2)/dx;  v1 = (gc-gim1)/dx
        v2 = (gip1-gc  )/dx;  v3 = (gip2-gip1)/dx
        s0 = v1-v0;  s1 = v2-v1;  s2 = v3-v2
        Dxm = xp.where(xp.abs(s0) <= xp.abs(s1), v1 - 0.5*s0, v1 + 0.5*s1)[2:-2, :]
        Dxp = xp.where(xp.abs(s1) <= xp.abs(s2), v2 - 0.5*s1, v2 + 0.5*s2)[2:-2, :]

        # y-derivatives: use ghost rows
        gjm2 = g[:-4, :]; gjm1 = g[1:-3, :]
        gjc  = g[2:-2, :]; gjp1 = g[3:-1, :]; gjp2 = g[4:, :]

        v0y = (gjm1-gjm2)/dy;  v1y = (gjc-gjm1)/dy
        v2y = (gjp1-gjc )/dy;  v3y = (gjp2-gjp1)/dy
        s0y = v1y-v0y;  s1y = v2y-v1y;  s2y = v3y-v2y
        Dym = xp.where(xp.abs(s0y) <= xp.abs(s1y), v1y - 0.5*s0y, v1y + 0.5*s1y)
        Dyp = xp.where(xp.abs(s1y) <= xp.abs(s2y), v2y - 0.5*s1y, v2y + 0.5*s2y)

        return Dxm, Dxp, Dym, Dyp

    @staticmethod
    def _godunov(Dxm, Dxp, Dym, Dyp, sgn):
        xp = cp.get_array_module(Dxm) if _CUPY_AVAILABLE else np
        ax_p = xp.maximum(Dxm, 0)**2;  bx_p = xp.minimum(Dxp, 0)**2
        ay_p = xp.maximum(Dym, 0)**2;  by_p = xp.minimum(Dyp, 0)**2
        ax_n = xp.minimum(Dxm, 0)**2;  bx_n = xp.maximum(Dxp, 0)**2
        ay_n = xp.minimum(Dym, 0)**2;  by_n = xp.maximum(Dyp, 0)**2
        Hp = xp.sqrt(xp.maximum(ax_p, bx_p) + xp.maximum(ay_p, by_p)) - 1.0
        Hn = xp.sqrt(xp.maximum(ax_n, bx_n) + xp.maximum(ay_n, by_n)) - 1.0
        return xp.where(sgn > 0, Hp, Hn)

    # ------------------------------------------------------------------
    # Subcell fix — always on CPU (runs once per reinit call)
    # ------------------------------------------------------------------

    def _subcell_fix(self, g_cpu):
        """
        g_cpu: CPU numpy array (2 + my_rows + 2, nx_nodes). Ghost rows must be valid.
        Returns (phi_fixed_cpu, is_fixed_cpu) — always numpy, same shape as g_cpu.
        """
        mr = self.my_rows;  dx = self.dx;  dy = self.dy

        phi_fixed = np.zeros_like(g_cpu)
        is_fixed  = np.zeros_like(g_cpu, dtype=bool)

        c    = g_cpu[2:2+mr, 1:-1]
        best = np.full(c.shape, np.inf)
        bval = np.zeros(c.shape)

        for neigh, h in (
            (g_cpu[2:2+mr, :-2], dx), (g_cpu[2:2+mr, 2:  ], dx),
            (g_cpu[1:1+mr, 1:-1], dy), (g_cpu[3:3+mr, 1:-1], dy),
        ):
            cross = (neigh * c) <= 0
            theta = np.abs(c) / (np.abs(c) + np.abs(neigh) + 1e-300)
            d_abs = theta * h
            d_val = np.sign(c) * d_abs
            imp   = cross & (d_abs < best)
            best  = np.where(imp, d_abs, best)
            bval  = np.where(imp, d_val, bval)

        has = np.isfinite(best)
        phi_fixed[2:2+mr, 1:-1] = np.where(has, bval, 0.0)
        is_fixed [2:2+mr, 1:-1] = has
        return phi_fixed, is_fixed

    # ------------------------------------------------------------------
    # Main reinitialization
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
        mr   = self.my_rows
        dx   = self.dx;  dy = self.dy

        # ---- 1. Forward Alltoallv: FEniCS DOFs → row-strip (CPU) --------
        lr   = phi.vector().local_range()
        n    = int(lr[1] - lr[0])
        vals = phi.vector().get_local()[:n]

        send_v = np.ascontiguousarray(vals[self._fwd_order], dtype=np.float64)
        recv_v = np.empty(self._fwd_total_recv, dtype=np.float64)
        comm.Alltoallv(
            [send_v, self._fwd_send_counts, self._fwd_send_displs, MPI.DOUBLE],
            [recv_v, self._fwd_recv_counts, self._fwd_recv_displs, MPI.DOUBLE])

        g_cpu = np.zeros((2 + mr + 2, self.nx_nodes), dtype=np.float64)
        g_cpu[self._local_recv_rows, self._recv_ix] = recv_v

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

        owned           = g[2:2+mr, :]
        is_fixed_owned  = is_fixed [2:2+mr, :]
        phi_fixed_owned = phi_fixed[2:2+mr, :]

        eps = 1.5 * min(dx, dy)
        sgn = owned / xp.sqrt(owned**2 + eps**2)
        dt  = 0.5 * min(dx, dy)

        # ---- 4. ENO2 iterations (GPU or CPU) ----------------------------
        for it in range(max_iter):
            exchange_fn(g)
            Dxm, Dxp, Dym, Dyp = self._eno2_deriv(g, dx, dy)
            H1 = self._godunov(Dxm, Dxp, Dym, Dyp, sgn)

            g_star = owned - dt * sgn * H1
            g_star = xp.where(is_fixed_owned, phi_fixed_owned, g_star)
            g[2:2+mr, :] = g_star

            exchange_fn(g)
            Dxm2, Dxp2, Dym2, Dyp2 = self._eno2_deriv(g, dx, dy)
            H2 = self._godunov(Dxm2, Dxp2, Dym2, Dyp2, sgn)

            g_new = owned - 0.5 * dt * sgn * (H1 + H2)
            g_new = xp.where(is_fixed_owned,              phi_fixed_owned, g_new)
            g_new = xp.where(~is_fixed_owned & (sgn > 0), xp.maximum(g_new, 0), g_new)
            g_new = xp.where(~is_fixed_owned & (sgn < 0), xp.minimum(g_new, 0), g_new)
            g[2:2+mr, :] = g_new
            owned = g_new

            # Convergence check: distributed mean of |∇φ| - 1
            non_fixed = ~is_fixed_owned
            if non_fixed.any():
                gy_o, gx_o = xp.gradient(owned, dy, dx, edge_order=2)
                err = xp.abs(xp.sqrt(gx_o**2 + gy_o**2)[non_fixed] - 1.0)
                # Reduce to CPU scalar for allreduce
                s_local = float(err.sum())
                n_local = int(err.size)
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

        # ---- 6. Backward Alltoallv: row-strip → FEniCS DOFs (CPU) -------
        bwd_v    = g_cpu[self._local_recv_rows, self._recv_ix]
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
