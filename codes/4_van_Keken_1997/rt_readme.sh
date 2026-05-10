#!/bin/bash
# ──────────────────────────────────────────────────────
#  Van Keken Benchmark (2D Compositional Buoyancy)
# ──────────────────────────────────────────────────────
#
#  Stokes + level-set advection (SUPG stabilized Crank-Nicolson).
#  Uzawa-CG with η-weighted pressure mass preconditioner.
#  P2/P1 Taylor-Hood, 2D domain (0.9142 x 1.0).
#
#  Files:
#    vankeken.py            SUPG advection only
#    vankeken_reinit.py     SUPG advection + ENO2 GPU reinitialization
#    reinit.py              ENO2 GPU reinit kernel (Min 2010) — required
#    reinit_mpi.py          MPI-distributed reinit wrapper — required
#    vankeken_weno.py       WENO advection variant
#    vankeken_weno_rho.py   WENO advection with density field
#    weno_advect_2d.py      WENO advection module
#    weno_advect_2d_mpi.py  WENO advection MPI module
#
#  Dependencies:
#    FEniCS 2019.1, PETSc, petsc4py, Hypre, mpi4py, NumPy
#    CuPy (for GPU reinitialization in vankeken_reinit.py)
#
# ──────────────────────────────────────────────────────

# --- SUPG only (no reinitialization) ---
python vankeken.py --monitor
python vankeken.py --nx 256 --ny 256 --dt 0.5 --final-time 500

# --- SUPG + ENO2 GPU reinitialization ---
python vankeken_reinit.py --monitor
python vankeken_reinit.py --nx 256 --ny 256 --dt 0.5 --final-time 500

# MPI parallel (reinit)
mpirun -n 4 python vankeken_reinit.py --nx 512 --ny 512 --quiet
mpirun -n 4 python vankeken_reinit.py --nx 512 --ny 512 --reinit-interval 5

# GPU-aware MPI workaround (if needed)
PETSC_OPTIONS="-use_gpu_aware_mpi 0" mpirun -n 4 python vankeken_reinit.py --nx 512 --ny 512 --quiet

# No output (benchmark only)
python vankeken.py --no-output --quiet
