#!/bin/bash

function info() {
  echo -e "${GREEN}[    INFO]${NOC}" "$@"
}

function warn() {
  echo -e "${YELLOW}[ WARNING]" "$@" "${NOC}"
}

function error() {
  echo -e "${RED}[   ERROR]" "$@" "${NOC}"
}
