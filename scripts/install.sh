#!/usr/bin/env bash
# Install Wattle as a user-level `wattle` command.
#
# Usage from a local checkout:
#   scripts/install.sh
#
# Usage from a hosted install script:
#   curl -fsSL https://raw.githubusercontent.com/<owner>/wattle/main/scripts/install.sh \
#     | WATTLE_REPO_URL=https://github.com/<owner>/wattle.git bash

set -euo pipefail

INSTALL_ROOT="${WATTLE_INSTALL_ROOT:-${HOME}/.wattle}"
REPO_DIR="${WATTLE_REPO_DIR:-${INSTALL_ROOT}/wattle}"
REPO_URL="${WATTLE_REPO_URL:-}"
BIN_DIR="${HOME}/.local/bin"

is_wattle_repo() {
    [[ -f "pyproject.toml" && -f "src/wattle/cli.py" ]] \
        && grep -q '^name = "wattle"$' pyproject.toml
}

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        return
    fi

    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${BIN_DIR}:${PATH}"
}

checkout_repo() {
    if is_wattle_repo; then
        REPO_DIR="$(pwd)"
        return
    fi

    if [[ -z "${REPO_URL}" ]]; then
        echo "WATTLE_REPO_URL is required when not running from a Wattle checkout." >&2
        echo "Example: curl -fsSL https://raw.githubusercontent.com/<owner>/wattle/main/scripts/install.sh | WATTLE_REPO_URL=https://github.com/<owner>/wattle.git bash" >&2
        exit 1
    fi

    mkdir -p "${INSTALL_ROOT}"
    if [[ -d "${REPO_DIR}/.git" ]]; then
        echo "Updating Wattle in ${REPO_DIR}"
        git -C "${REPO_DIR}" pull --ff-only
    else
        echo "Cloning Wattle into ${REPO_DIR}"
        git clone "${REPO_URL}" "${REPO_DIR}"
    fi
}

ensure_uv
checkout_repo

echo "Installing wattle from ${REPO_DIR}"
uv tool install --force -e "${REPO_DIR}"

echo "Installed: $(command -v wattle || true)"
if ! command -v wattle >/dev/null 2>&1; then
    echo "Note: ${BIN_DIR} may not be on PATH. Add this to your shell rc:"
    echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
fi
