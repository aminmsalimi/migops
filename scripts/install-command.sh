#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.." &&
    pwd
)"

MIGOPS_BIN="${REPO_ROOT}/.venv/bin/migops"
LINK="/usr/local/bin/migops"

if [[ ! -x "${MIGOPS_BIN}" ]]; then
    echo "MIGOps executable not found:"
    echo "  ${MIGOPS_BIN}"
    echo
    echo "Create the virtual environment and install MIGOps first:"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  python -m pip install -e ."
    exit 1
fi

if [[ "${EUID}" -eq 0 ]]; then
    ln -sfn "${MIGOPS_BIN}" "${LINK}"
else
    sudo ln -sfn "${MIGOPS_BIN}" "${LINK}"
fi

echo "Installed command:"
echo "  ${LINK} -> ${MIGOPS_BIN}"
echo
echo "You can now use:"
echo "  migops status"
echo "  sudo migops enable gpu 0"
