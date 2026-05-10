# GAUZZ: GPU And UZawa for Geodynamic simulations
GPU-accelerated FEniCS environment built on CUDA 12.5, PETSc 3.24, DOLFIN 2019.1.0, and CuPy 13.3, packaged as a Docker image.

The image was developed and tested on an RTX 6000 Ada (CC 8.9). For other GPUs, change `--with-cuda-arch=89` in the `Dockerfile` (Ampere=80, Hopper=90, Turing=75) before building — otherwise PETSc kernels will fail at runtime with `cudaErrorNoKernelImageForDevice`.

## Requirements

- Linux host (tested on Ubuntu 22.04)
- NVIDIA driver 550.x or newer
- Docker + NVIDIA Container Toolkit
- ~20 GB free disk

Sanity check before building:

```bash
docker run --rm --gpus all nvidia/cuda:12.5.1-base-ubuntu22.04 nvidia-smi
```

If the GPU shows up here, the host side is fine.

## Layout

```
gpu_fenics/
├── Dockerfile
├── gpu_ghost_full.patch     # DOLFIN GPU MPI Ghost Vec patch — applied during build
├── README.md
└── codes/                   # 7 example problems (subproblem, solcx, blankenbach, ...)
```

## Build

```bash
cd ~/gpu_fenics
docker build -t gpu-fenics:latest .
```

PETSc and DOLFIN are compiled from source, so the build takes roughly an hour. Several warnings show up along the way (pip-as-root, detached HEAD on the DOLFIN tag, PETSc "out-of-date" notice for the pinned 3.24.0, debconf complaints) — none of them affect the result.

## Run

```bash
docker run -it --gpus all --ipc=host \
  --name gpu-fenics \
  -v /home:/home/work \
  gpu-fenics:latest
```

`--ipc=host` is needed so MPI and CUDA IPC can use the host's `/dev/shm`. The mount path is just my convention; replace `-v /home:/home/work` with whatever fits your layout (`-v $(pwd):/home/work`, etc.).

Inside the container, make the example scripts executable once and you're set:

```bash
cd /home/work/gpu_fenics
find . -name "*.sh" -exec chmod +x {} +
```

To re-enter later:

```bash
docker start -ai gpu-fenics       # if stopped
docker exec -it gpu-fenics bash   # if already running
```

## Running the examples

Everything in `codes/` is a standalone script. Each folder ships with a `*_readme.sh` that prints the available CLI flags, defaults, and a few representative invocations — run it once whenever you forget what a script accepts:

```bash
cd codes/3_blankenbach_1989
./blanken_readme.sh              # prints the usage panel
```

The seven problems, in increasing order of complexity:

| Folder | What it solves |
|---|---|
| `1_subproblem` | Minimal Stokes subproblem — start here to confirm the build works |
| `2_solcx` | SolCx benchmark; analytical reference fields are in `analytical_solutions/` |
| `3_blankenbach_1989` | 3D mantle convection (Stokes + temperature advection-diffusion) |
| `4_van_Keken_1997` | 2D compositional buoyancy with SUPG / WENO advection and ENO2 GPU reinit |
| `5_rt_3d` | 3D Rayleigh–Taylor with the same reinit machinery |
| `6_rebound` | Post-glacial rebound |
| `7_subduction` | 2D subduction (Stokes + level-set advection + Picard, Anderson-accelerated). Phase fields live in `subduction_lv_set/` |

Typical invocations:

```bash
# single-GPU smoke test
cd codes/2_solcx
python3 solcx.py --res 6 --monitor

# multi-rank Stokes + level-set
cd codes/7_subduction
mpirun -np 4 python3 subduction.py --monitor

# benchmark mode (no XDMF writes)
mpirun -np 4 python3 subduction.py --nx 600 --ny 100 --steps 200 --no-output --quiet
```

Two flags worth knowing across most scripts:

- `--monitor` prints per-step convergence (Uzawa, Picard, KSP residuals depending on the problem).
- `--no-output` skips XDMF writes — use it when timing.

If you hit `cudaErrorInvalidResource` or hangs on multi-rank runs that touch CuPy, fall back to host-staged MPI:

```bash
PETSC_OPTIONS="-use_gpu_aware_mpi 0" mpirun -np 4 python3 vankeken_reinit.py --nx 512 --ny 512 --quiet
```

## Troubleshooting

A few things I actually ran into:

- **`cudaErrorNoKernelImageForDevice`** — wrong CUDA arch. Rebuild with the right `--with-cuda-arch=` for your card.
- **`cudaErrorMpsConnectionFailed` / `Failed to initialize NVML`** — the MPS daemon got into a weird state. `docker restart gpu-fenics` clears it.
- **`mpirun` refuses to run as root** — should already be handled by `OMPI_ALLOW_RUN_AS_ROOT=1` in the Dockerfile. If you still hit it, recreate the container rather than the image.
- **`nvidia-smi` not visible inside the container** — `--gpus all` was missing, or NVIDIA Container Toolkit isn't installed on the host.

## Versions

Base image `nvidia/cuda:12.5.1-devel-ubuntu22.04`. Inside:

- PETSc 3.24.0 (CUDA + hypre + metis + parmetis)
- DOLFIN 2019.1.0 (commit `74d7efe1e`, with the GPU MPI Ghost Vec patch in this repo)
- FFC 2019.1.0.post0 (`a799b743`), UFL / FIAT / dijitso 2019.1.0
- CuPy 13.3.0 (`cupy-cuda12x`), NumPy 1.26.4
- OpenMPI 4.1.2, Python 3.10.12

## Cleanup

```bash
docker rm -f gpu-fenics
docker rmi gpu-fenics:latest
```
