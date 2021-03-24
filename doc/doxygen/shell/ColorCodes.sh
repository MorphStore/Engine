#!/bin/bash
### Colorcodes ### =============================================================
  RESET=$'\e[0m'
  BOLD=$'\e[1m'
  DIM=$'\e[2m'
  UNDERLINED=$'\e[4m'
  BLINK=$'\e[5m'
  REVERSE=$'\e[7m'
  HIDDEN=$'\e[8m'

  NOC=$'\e[39m\e[0m'
  BLACK=$'\e[30m'
  RED=$'\e[31m'
  GREEN=$'\e[32m'
  YELLOW=$'\e[33m'
  BLUE=$'\e[34m'
  MAGENTA=$'\e[35m'
  CYAN=$'\e[36m'
  LGRAY=$'\e[37m'
  DGRAY=$'\e[90m'
  LRED=$'\e[91m'
  LGREEN=$'\e[92m'
  LYELLOW=$'\e[93m'
  LBLUE=$'\e[94m'
  LMAGENTA=$'\e[95m'
  LCYAN=$'\e[96m'
  WHITE=$'\e[97m'

function reset_color_codes(){
  unset RESET
  unset BOLD
  unset DIM
  unset UNDERLINED
  unset BLINK
  unset REVERSE
  unset HIDDEN
  unset NOC
  unset BLACK
  unset RED
  unset GREEN
  unset YELLOW
  unset BLUE
  unset MAGENTA
  unset CYAN
  unset LGRAY
  unset DGRAY
  unset LRED
  unset LGREEN
  unset LYELLOW
  unset LBLUE
  unset LMAGENTA
  unset LCYAN
  unset WHITE
}
