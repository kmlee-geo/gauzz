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
line_plain "How to run subduction.py  (2D Subduction: Stokes + Level-Set Advection + Picard)"
border_mid

line_plain ""
line_color "${CYAN}${BOLD}" "1) Basic run"
line_plain "  mpirun -np 4 python subduction.py"

line_plain ""
line_color "${CYAN}${BOLD}" "2) Show per-step convergence table"
line_plain "  mpirun -np 4 python subduction.py --monitor"

line_plain ""
line_color "${CYAN}${BOLD}" "3) Change mesh resolution"
line_plain "  mpirun -np 4 python subduction.py --nx 600 --ny 100 --monitor"
line_plain "  default: nx=1200  ny=200"

line_plain ""
line_color "${CYAN}${BOLD}" "4) Change number of time steps"
line_plain "  mpirun -np 4 python subduction.py --steps 500 --monitor"
line_plain "  default: steps=3000"

line_plain ""
line_color "${CYAN}${BOLD}" "5) Tune CFL and initial dt"
line_plain "  mpirun -np 4 python subduction.py --cfl 0.3 --dt 2.0e-6 --monitor"
line_plain "  default: cfl=0.4  dt=4.4e-6"

line_plain ""
line_color "${CYAN}${BOLD}" "6) Tune Uzawa-CG solver"
line_plain "  mpirun -np 4 python subduction.py --uzawa-iter 3000 --uzawa-tol 1e-4 --monitor"
line_plain "  default: uzawa-iter=2000  uzawa-tol=1e-3"

line_plain ""
line_color "${CYAN}${BOLD}" "7) Tune Picard iterations"
line_plain "  mpirun -np 4 python subduction.py --picard 5 --monitor"
line_plain "  default: picard=3  (Anderson acceleration applied from iteration 2)"

line_plain ""
line_color "${CYAN}${BOLD}" "8) Change level-set directory"
line_plain "  mpirun -np 4 python subduction.py --level-set-dir ./my_lv_set"
line_plain "  default: ./subduction_lv_set  (each .npy file defines one material phase)"

line_plain ""
line_color "${CYAN}${BOLD}" "9) Skip output file"
line_plain "  mpirun -np 4 python subduction.py --no-output --quiet"

line_plain ""
line_color "${CYAN}${BOLD}" "10) Custom output path and save frequency"
line_plain "  mpirun -np 4 python subduction.py --output ./out/sub.xdmf --save-every 20"
line_plain "  default: ./results/subduction.xdmf  save-every=10"

line_plain ""
line_color "${CYAN}${BOLD}" "11) Serial run (debugging)"
line_plain "  python subduction.py --nx 120 --ny 20 --steps 10 --monitor"

line_plain ""
line_color "${MAGENTA}${BOLD}" "Typical examples"
line_plain "  mpirun -np 4 python subduction.py --monitor"
line_plain "  mpirun -np 4 python subduction.py --nx 600 --ny 100 --steps 200 --cfl 0.3 --monitor"
line_plain "  mpirun -np 4 python subduction.py --uzawa-tol 1e-4 --picard 5 --monitor"

line_plain ""
line_color "${YELLOW}${BOLD}" "Notes"
line_plain "  requires MPI; default resolution (1200x200) is memory-intensive"
line_plain "  --monitor prints per-Uzawa and per-Picard convergence"
line_plain "  --picard controls nonlinear iterations; Anderson acceleration auto-applied"
line_plain "  --level-set-dir must contain .npy phase field files before running"
line_plain "  --no-output skips XDMF writes; useful for benchmarking"
line_plain ""

border_bot

printf "\n"
printf "%b\n" "${BOLD}Quick start:${RESET} mpirun -np 4 python subduction.py --monitor"
printf "%b\n" "${BOLD}Low-res test:${RESET} mpirun -np 4 python subduction.py --nx 300 --ny 50 --steps 100 --monitor"
printf "\n"