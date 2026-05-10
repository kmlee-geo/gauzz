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
line_plain "How to run solcx_single.py on GPU"
border_mid

line_plain ""
line_color "${CYAN}${BOLD}" "1) Basic run"
line_plain "  python3 solcx_single.py"

line_plain ""
line_color "${CYAN}${BOLD}" "2) Change mesh resolution"
line_plain "  python3 solcx_single.py --res 8"

line_plain ""
line_color "${CYAN}${BOLD}" "3) Print solver summary"
line_plain "  python3 solcx_single.py --monitor"

line_plain ""
line_color "${CYAN}${BOLD}" "4) Set Krylov tolerances"
line_plain "  python3 solcx_single.py --rtol 1e-10 --atol 1e-30"

line_plain ""
line_color "${CYAN}${BOLD}" "5) Set GMRES controls"
line_plain "  python3 solcx_single.py --max-it 50 --restart 400"

line_plain ""
line_color "${CYAN}${BOLD}" "6) Enable PETSc residual monitor"
line_plain "  python3 solcx_single.py --ksp-monitor"

line_plain ""
line_color "${CYAN}${BOLD}" "7) Run with MPI"
line_plain "  mpirun -n 4 python3 solcx_single.py --res 8"

line_plain ""
line_color "${CYAN}${BOLD}" "8) Skip output file"
line_plain "  python3 solcx_single.py --no-output"

line_plain ""
line_color "${CYAN}${BOLD}" "9) Custom output path"
line_plain "  python3 solcx_single.py --output ./output/my_solcx_single.xdmf"

line_plain ""
line_color "${MAGENTA}${BOLD}" "Typical examples"
line_plain "  python3 solcx_single.py"
line_plain "  python3 solcx_single.py --res 8 --monitor"
line_plain "  python3 solcx_single.py --res 8 --max-it 50 --restart 400"
line_plain "  mpirun -n 4 python3 solcx_single.py --res 8"

line_plain ""
line_color "${YELLOW}${BOLD}" "Notes"
line_plain "  --res controls mesh by nx = ny = 2^res"
line_plain "  --monitor prints final solver summary"
line_plain "  --ksp-monitor prints PETSc residual history"
line_plain "  --max-it sets maximum GMRES iterations"
line_plain "  --restart sets GMRES restart length"
line_plain "  output is written unless --no-output"
line_plain ""

border_bot

printf "\n"
printf "%b\n" "${BOLD}Quick start:${RESET} python3 solcx_single.py --monitor"
printf "%b\n" "${BOLD}MPI example:${RESET} mpirun -n 4 python3 solcx_single.py --res 8"
printf "\n"