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
line_plain "How to run blanken.py  (3D Mantle Convection: Stokes + Temperature Advection-Diffusion)"
border_mid

line_plain ""
line_color "${CYAN}${BOLD}" "1) Basic run"
line_plain "  python blanken.py"

line_plain ""
line_color "${CYAN}${BOLD}" "2) Show per-step convergence table"
line_plain "  python blanken.py --monitor"

line_plain ""
line_color "${CYAN}${BOLD}" "3) Change mesh resolution"
line_plain "  python blanken.py --nx 50 --ny 20 --nz 50 --monitor"
line_plain "  default: nx=100  ny=40  nz=100"

line_plain ""
line_color "${CYAN}${BOLD}" "4) Change Rayleigh number"
line_plain "  python blanken.py --Ra 1e5 --monitor"
line_plain "  default: Ra=1e4"

line_plain ""
line_color "${CYAN}${BOLD}" "5) Change time step and number of steps"
line_plain "  python blanken.py --dt 0.0005 --steps 500 --monitor"
line_plain "  default: dt=0.001  steps=251"

line_plain ""
line_color "${CYAN}${BOLD}" "6) Tune Uzawa-CG solver"
line_plain "  python blanken.py --uzawa-iter 3000 --uzawa-tol 1e-10 --monitor"
line_plain "  default: uzawa-iter=2000  uzawa-tol=1e-9"

line_plain ""
line_color "${CYAN}${BOLD}" "7) Run with MPI"
line_plain "  mpirun -n 4 python blanken.py --nx 100 --ny 40 --nz 100 --Ra 1e4 --steps 200"

line_plain ""
line_color "${CYAN}${BOLD}" "8) Skip output file"
line_plain "  python blanken.py --monitor --no-output"

line_plain ""
line_color "${CYAN}${BOLD}" "9) Custom output path"
line_plain "  python blanken.py --output ./output/my_blanken.xdmf"

line_plain ""
line_color "${CYAN}${BOLD}" "10) Suppress per-step table (summary only)"
line_plain "  python blanken.py --quiet"

line_plain ""
line_color "${MAGENTA}${BOLD}" "Typical examples"
line_plain "  python blanken.py --monitor"
line_plain "  python blanken.py --nx 50 --ny 20 --nz 50 --Ra 1e5 --steps 100 --monitor"
line_plain "  mpirun -n 4 python blanken.py --nx 100 --ny 40 --nz 100 --Ra 1e4 --steps 200"

line_plain ""
line_color "${YELLOW}${BOLD}" "Notes"
line_plain "  --Ra controls the Rayleigh number (convection intensity)"
line_plain "  --monitor prints per-Uzawa-iteration and per-step convergence"
line_plain "  --quiet suppresses per-step table; prints summary only"
line_plain "  --no-output skips writing XDMF; useful for benchmarking"
line_plain "  output default: ./output/blanken_output.xdmf"
line_plain ""

border_bot

printf "\n"
printf "%b\n" "${BOLD}Quick start:${RESET} python blanken.py --monitor"
printf "%b\n" "${BOLD}MPI example:${RESET} mpirun -n 4 python blanken.py --nx 100 --ny 40 --nz 100 --Ra 1e4 --steps 200"
printf "\n"