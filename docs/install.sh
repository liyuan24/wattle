#!/usr/bin/env bash
# Install Wattle as a user-level `wattle` command.
#
# Usage from the hosted installer:
#   curl -fsSL https://wattleagent.com/install.sh | bash
#
# Pin a release:
#   curl -fsSL https://wattleagent.com/install.sh | WATTLE_VERSION=0.2.0 bash

set -euo pipefail

INSTALL_ROOT="${WATTLE_INSTALL_ROOT:-${HOME}/.wattle}"
REPO_DIR="${WATTLE_REPO_DIR:-${INSTALL_ROOT}/wattle}"
REPO_URL="${WATTLE_REPO_URL:-https://github.com/liyuan24/wattle.git}"
LATEST_VERSION_URL="${WATTLE_LATEST_VERSION_URL:-https://wattleagent.com/api/latest-version}"
WATTLE_CHANNEL="${WATTLE_CHANNEL:-stable}"
WATTLE_VERSION="${WATTLE_VERSION:-}"
BIN_DIR="${HOME}/.local/bin"

need_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 1
    fi
}

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        return
    fi

    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${BIN_DIR}:${PATH}"
}

json_field() {
    sed -n "s/.*\"$1\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p" | head -1
}

normalize_tag() {
    local raw="$1"
    if [[ -z "${raw}" ]]; then
        echo ""
    elif [[ "${raw}" == v* ]]; then
        echo "${raw}"
    else
        echo "v${raw}"
    fi
}

resolve_target_ref() {
    if [[ "${WATTLE_CHANNEL}" == "dev" ]]; then
        echo "master"
        return
    fi

    if [[ -n "${WATTLE_VERSION}" ]]; then
        normalize_tag "${WATTLE_VERSION}"
        return
    fi

    echo "Resolving latest Wattle release..." >&2
    local metadata tag version
    metadata="$(curl -fsSL "${LATEST_VERSION_URL}")"
    tag="$(printf "%s" "${metadata}" | json_field tag)"
    version="$(printf "%s" "${metadata}" | json_field version)"
    if [[ -z "${tag}" ]]; then
        tag="$(normalize_tag "${version}")"
    fi
    if [[ -z "${tag}" ]]; then
        echo "Could not resolve latest Wattle version from ${LATEST_VERSION_URL}" >&2
        exit 1
    fi
    echo "${tag}"
}

checkout_repo() {
    mkdir -p "${INSTALL_ROOT}"
    if [[ -d "${REPO_DIR}/.git" ]]; then
        echo "Updating Wattle checkout in ${REPO_DIR}"
        git -C "${REPO_DIR}" remote set-url origin "${REPO_URL}"
        git -C "${REPO_DIR}" fetch --tags --force origin
    else
        echo "Cloning Wattle into ${REPO_DIR}"
        git clone "${REPO_URL}" "${REPO_DIR}"
    fi
}

checkout_ref() {
    local target_ref="$1"
    echo "Checking out Wattle ${target_ref}"
    git -C "${REPO_DIR}" checkout --force "${target_ref}"
    git -C "${REPO_DIR}" reset --hard "${target_ref}" >/dev/null
}

install_wattle() {
    uv tool install --force "${REPO_DIR}"
}

need_command curl
need_command git
ensure_uv
target_ref="$(resolve_target_ref)"
checkout_repo
checkout_ref "${target_ref}"

echo "Installing wattle from ${REPO_DIR}"
install_wattle

echo "Installed: $(command -v wattle || true)"
if ! command -v wattle >/dev/null 2>&1; then
    echo "Note: ${BIN_DIR} may not be on PATH. Add this to your shell rc:"
    echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
else
    wattle --version || true
fi
