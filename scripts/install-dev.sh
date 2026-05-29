#!/usr/bin/env bash
# Install the current Wattle checkout as an editable developer tool.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${HOME}/.local/bin"

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${BIN_DIR}:${PATH}"
fi

echo "Installing editable wattle from ${REPO_DIR}"
uv tool install --force -e "${REPO_DIR}"

echo "Installed: $(command -v wattle || true)"
if ! command -v wattle >/dev/null 2>&1; then
    echo "Note: ${BIN_DIR} may not be on PATH. Add this to your shell rc:"
    echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
else
    wattle --version || true
fi
