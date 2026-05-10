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
line_plain "How to run rebound.py  (2D Rayleigh-Taylor: Stokes + DG1 Level-Set Advection)"
border_mid

line_plain ""
line_color "${CYAN}${BOLD}" "1) Basic run"
line_plain "  mpirun -np 4 python rebound.py"

line_plain ""
line_color "${CYAN}${BOLD}" "2) Show per-step convergence table"
line_plain "  mpirun -np 4 python rebound.py --monitor"

line_plain ""
line_color "${CYAN}${BOLD}" "3) Change mesh resolution"
line_plain "  mpirun -np 4 python rebound.py --nx 700 --ny 400 --monitor"
line_plain "  default: nx=1400  ny=800  (domain: 3.5 x 1.0)"

line_plain ""
line_color "${CYAN}${BOLD}" "4) Change number of time steps"
line_plain "  mpirun -np 4 python rebound.py --steps 200 --monitor"
line_plain "  default: steps=100"

line_plain ""
line_color "${CYAN}${BOLD}" "5) Tune CFL and initial dt"
line_plain "  mpirun -np 4 python rebound.py --cfl 0.3 --dt 1.0e-8 --monitor"
line_plain "  default: cfl=0.4  dt=1.23e-8  (adaptive dt from CFL each step)"

line_plain ""
line_color "${CYAN}${BOLD}" "6) Tune Uzawa-CG solver"
line_plain "  mpirun -np 4 python rebound.py --uzawa-iter 3000 --uzawa-tol 1e-8 --monitor"
line_plain "  default: uzawa-iter=2000  uzawa-tol=1e-7"

line_plain ""
line_color "${CYAN}${BOLD}" "7) Tune level-set interface width"
line_plain "  mpirun -np 4 python rebound.py --ls-alpha 1.5 --monitor"
line_plain "  default: ls-alpha=1.0  (interface half-width = ls-alpha * h_min)"

line_plain ""
line_color "${CYAN}${BOLD}" "8) Custom output path and save frequency"
line_plain "  mpirun -np 4 python rebound.py --output ./out/rebound.xdmf --save-every 5"
line_plain "  default: ./results/rt_stokes.xdmf  save-every=10"

line_plain ""
line_color "${CYAN}${BOLD}" "9) Skip output file  (benchmark only)"
line_plain "  mpirun -np 4 python rebound.py --no-output --quiet"

line_plain ""
line_color "${CYAN}${BOLD}" "10) Serial run  (debugging)"
line_plain "  python rebound.py --nx 140 --ny 80 --steps 10 --monitor"

line_plain ""
line_color "${MAGENTA}${BOLD}" "Typical examples"
line_plain "  mpirun -np 4 python rebound.py --monitor"
line_plain "  mpirun -np 4 python rebound.py --nx 1400 --ny 800 --steps 200"
line_plain "  mpirun -np 4 python rebound.py --cfl 0.3 --uzawa-tol 1e-8 --monitor"

line_plain ""
line_color "${YELLOW}${BOLD}" "Notes"
line_plain "  velocity: P1+Bubble (Mini element)  |  level-set: DG1 Crank-Nicolson upwind"
line_plain "  two level-set fields: phi1 (upper interface), phi2 (lower interface)"
line_plain "  three material regions: top (eta=1), band (eta=1e5), bottom (eta=1e4)"
line_plain "  --ls-alpha controls smoothed Heaviside width for eta/rho transition"
line_plain "  dt is adaptive each step via CFL; --dt sets only the initial value"
line_plain "  --no-output skips XDMF writes; useful for benchmarking"
line_plain ""

border_bot

printf "\n"
printf "%b\n" "${BOLD}Quick start:${RESET} mpirun -np 4 python rebound.py --monitor"
printf "%b\n" "${BOLD}Low-res test:${RESET} mpirun -np 4 python rebound.py --nx 350 --ny 200 --steps 50 --monitor"
printf "\n"