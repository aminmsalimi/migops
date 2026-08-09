#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/aminmsalimi/migops.git"
BRANCH="main"
INSTALL_DIR="${MIGOPS_INSTALL_DIR:-${HOME}/.local/share/migops}"
USER_BIN_DIR="${HOME}/.local/bin"
USER_LINK="${USER_BIN_DIR}/migops"
SYSTEM_LINK="/usr/local/bin/migops"

say() { printf '%s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || fail "'$1' is required but was not found."; }

say "==> Installing MIGOps"

need_cmd git
need_cmd python3

mkdir -p "$(dirname "$INSTALL_DIR")"
mkdir -p "$USER_BIN_DIR"

if [[ -d "${INSTALL_DIR}/.git" ]]; then
    say "==> Updating existing MIGOps installation"
    git -C "$INSTALL_DIR" fetch --depth 1 origin "$BRANCH"
    git -C "$INSTALL_DIR" reset --hard FETCH_HEAD
else
    if [[ -e "$INSTALL_DIR" ]]; then
        fail "${INSTALL_DIR} already exists but is not a MIGOps git checkout. Remove it or set MIGOPS_INSTALL_DIR."
    fi
    say "==> Downloading MIGOps"
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
fi

say "==> Creating Python virtual environment"
if ! python3 -m venv "${INSTALL_DIR}/.venv"; then
    cat >&2 <<'EOF'

Could not create a Python virtual environment.

On Ubuntu/Debian, install the venv package and run the installer again:

  sudo apt update
  sudo apt install -y python3-venv
EOF
    exit 1
fi

say "==> Installing MIGOps"
"${INSTALL_DIR}/.venv/bin/python" -m pip install --disable-pip-version-check -e "$INSTALL_DIR"

ln -sfn "${INSTALL_DIR}/.venv/bin/migops" "$USER_LINK"

system_link_ok=false

if [[ "${EUID}" -eq 0 ]]; then
    ln -sfn "${INSTALL_DIR}/.venv/bin/migops" "$SYSTEM_LINK"
    system_link_ok=true
elif command -v sudo >/dev/null 2>&1; then
    say "==> Creating ${SYSTEM_LINK} so 'sudo migops ...' works"
    if sudo ln -sfn "${INSTALL_DIR}/.venv/bin/migops" "$SYSTEM_LINK"; then
        system_link_ok=true
    else
        say "WARNING: Could not create ${SYSTEM_LINK}."
    fi
fi

say
say "MIGOps installation complete."
say
say "Try:"
say "  migops status"

if [[ "$system_link_ok" == true ]]; then
    say "  sudo migops enable gpu 0 --yes"
else
    say
    say "Add ${USER_BIN_DIR} to PATH if 'migops' is not found."
fi
