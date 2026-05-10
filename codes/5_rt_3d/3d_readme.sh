#!/usr/bin/env bash

export LANG=C.UTF-8

GREEN="\033[92m"
CYAN="\033[36m"
MAGENTA="\033[35m"
YELLOW="\033[33m"
BOLD="\033[1m"
RESET="\033[0m"

WIDTH=88

repeat_char() {
    local char="$1"
    local n="$2"
    local i
    for ((i=0; i<n; i++)); do
        printf "%s" "$char"
    done
}

line_plain() {
    local text="$1"
    printf "${GREEN}║${RESET} %-*s ${GREEN}║${RESET}\n" "$WIDTH" "$text"
}

line_color() {
    local color="$1"
    local text="$2"
    local pad=$((WIDTH - ${#text}))
    ((pad < 0)) && pad=0

    printf "${GREEN}║${RESET} "
    printf "%b" "${color}${text}${RESET}"
    repeat_char " " "$pad"
    printf " ${GREEN}║${RESET}\n"
}

border_top() {
    printf "${GREEN}╔"
    repeat_char "═" $((WIDTH + 2))
    printf "╗${RESET}\n"
}

border_mid() {
    printf "${GREEN}╠"
    repeat_char "═" $((WIDTH + 2))
    printf "╣${RESET}\n"
}

border_bot() {
    printf "${GREEN}╚"
    repeat_char "═" $((WIDTH + 2))
    printf "╝${RESET}\n"
}

printf "\n"

border_top
line_plain "How to run 3d_rt.py / 3d_rt_reinit.py  (3D Rayleigh-Taylor Instability Benchmark)"
border_mid

line_plain ""
line_color "${CYAN}${BOLD}" "1) Basic run  (SUPG advection only, no reinitialization)"
line_plain "  python 3d_rt.py"

line_plain ""
line_color "${CYAN}${BOLD}" "2) Basic run  (SUPG + ENO2 GPU reinitialization)"
line_plain "  python 3d_rt_reinit.py"

line_plain ""
line_color "${CYAN}${BOLD}" "3) Show per-step convergence table"
line_plain "  python 3d_rt_reinit.py --monitor"

line_plain ""
line_color "${CYAN}${BOLD}" "4) Change mesh resolution and time stepping"
line_plain "  python 3d_rt_reinit.py --nx 50 --ny 50 --nz 50 --dt 0.25 --final-time 200"

line_plain ""
line_color "${CYAN}${BOLD}" "5) Custom viscosity and density"
line_plain "  python 3d_rt_reinit.py --eta1 1.0 --eta2 100.0 --rho1 0.0 --rho2 1.0 --monitor"

line_plain ""
line_color "${CYAN}${BOLD}" "6) Tune reinitialization"
line_plain "  python 3d_rt_reinit.py --reinit-interval 5 --reinit-mode mpi"
line_plain "  reinit-mode: gpu (default, CuPy)  |  mpi (MPI-distributed, reinit_mpi_3d.py)"

line_plain ""
line_color "${CYAN}${BOLD}" "7) Run with MPI"
line_plain "  mpirun -n 4 python 3d_rt_reinit.py --nx 64 --ny 64 --nz 64 --quiet"
line_plain "  mpirun -n 4 python 3d_rt_reinit.py --reinit-interval 5 --reinit-mode mpi"

line_plain ""
line_color "${CYAN}${BOLD}" "8) GPU-aware MPI workaround"
line_plain "  PETSC_OPTIONS=\"-use_gpu_aware_mpi 0\" mpirun -n 4 python 3d_rt_reinit.py --nx 64 --ny 64 --nz 64"

line_plain ""
line_color "${CYAN}${BOLD}" "9) Skip output file  (benchmark only)"
line_plain "  python 3d_rt_reinit.py --no-output --quiet"

line_plain ""
line_color "${MAGENTA}${BOLD}" "Typical examples"
line_plain "  python 3d_rt_reinit.py --monitor"
line_plain "  python 3d_rt_reinit.py --nx 50 --ny 50 --nz 50 --dt 0.25 --final-time 200"
line_plain "  mpirun -n 4 python 3d_rt_reinit.py --nx 64 --ny 64 --nz 64 --reinit-mode mpi"

line_plain ""
line_color "${YELLOW}${BOLD}" "Notes"
line_plain "  3d_rt.py        : SUPG advection only (no reinit, no CuPy required)"
line_plain "  3d_rt_reinit.py : SUPG + ENO2 reinitialization (requires CuPy for gpu mode)"
line_plain "  reinit_3d.py / reinit_mpi_3d.py must be present in the same directory"
line_plain "  --reinit-mode mpi uses reinit_mpi_3d.py; gpu mode uses CuPy kernel"
line_plain "  if GPU-aware MPI causes issues, prepend PETSC_OPTIONS=\"-use_gpu_aware_mpi 0\""
line_plain "  domain is fixed: 0.9142 x 0.8142 x 1.0"
line_plain ""

border_bot

printf "\n"
printf "%b\n" "${BOLD}Quick start:${RESET} python 3d_rt_reinit.py --monitor"
printf "%b\n" "${BOLD}MPI example:${RESET} mpirun -n 4 python 3d_rt_reinit.py --nx 64 --ny 64 --nz 64 --reinit-mode mpi"
printf "\n"