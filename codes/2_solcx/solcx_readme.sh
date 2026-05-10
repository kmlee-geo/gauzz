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
line_plain "How to run solcx.py on GPU"
border_mid

line_plain ""
line_color "${CYAN}${BOLD}" "1) Basic run"
line_plain "  python3 solcx.py"

line_plain ""
line_color "${CYAN}${BOLD}" "2) Show iteration table"
line_plain "  python3 solcx.py --monitor"

line_plain ""
line_color "${CYAN}${BOLD}" "3) Change mesh resolution"
line_plain "  python3 solcx.py --res 8 --monitor"
line_plain "  meaning: nx = ny = 2^8 = 256"

line_plain ""
line_color "${CYAN}${BOLD}" "4) Set pressure-correction iterations"
line_plain "  python3 solcx.py --iter 20 --monitor"

line_plain ""
line_color "${CYAN}${BOLD}" "5) Disable PCG tolerance stopping"
line_plain "  python3 solcx.py --iter 20 --monitor --no-pcg-tol"

line_plain ""
line_color "${CYAN}${BOLD}" "6) Run with MPI"
line_plain "  mpirun -n 4 python3 solcx.py --res 8 --monitor"

line_plain ""
line_color "${CYAN}${BOLD}" "7) Skip output file"
line_plain "  python3 solcx.py --monitor --no-output"

line_plain ""
line_color "${CYAN}${BOLD}" "8) Custom output path"
line_plain "  python3 solcx.py --output ./output/my_solcx.xdmf"

line_plain ""
line_color "${MAGENTA}${BOLD}" "Typical examples"
line_plain "  python3 solcx.py --monitor"
line_plain "  python3 solcx.py --res 8 --iter 20 --monitor"
line_plain "  mpirun -n 4 python3 solcx.py --res 8 --monitor"

line_plain ""
line_color "${YELLOW}${BOLD}" "Notes"
line_plain "  --res controls mesh by nx = ny = 2^res"
line_plain "  --monitor prints convergence table"
line_plain "  --no-pcg-tol forces full iterations"
line_plain "  output is written unless --no-output"
line_plain ""

border_bot

printf "\n"
printf "%b\n" "${BOLD}Quick start:${RESET} python3 solcx.py --monitor"
printf "%b\n" "${BOLD}MPI example:${RESET} mpirun -n 4 python3 solcx.py --res 8 --monitor"
printf "\n"