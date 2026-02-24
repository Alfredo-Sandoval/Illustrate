#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SETUP_SCRIPT="${SCRIPT_DIR}/setup.sh"

detect_target_os() {
  local uname_s
  uname_s="$(uname -s)"
  case "${uname_s}" in
    Darwin)
      echo "macos"
      ;;
    Linux)
      echo "linux"
      ;;
    *)
      echo ""
      ;;
  esac
}

if command -v micromamba >/dev/null 2>&1; then
  MAMBA_BIN="micromamba"
elif command -v mamba >/dev/null 2>&1; then
  MAMBA_BIN="mamba"
else
  echo "Missing dependency: install micromamba or mamba." >&2
  echo "Tip (macOS): brew install --cask miniforge" >&2
  exit 1
fi

TARGET_OS="${ARC_SETUP_OS:-}"
if [[ -z "${TARGET_OS}" ]]; then
  TARGET_OS="$(detect_target_os)"
fi
if [[ "${TARGET_OS}" != "macos" && "${TARGET_OS}" != "linux" ]]; then
  echo "Unsupported host OS for run_gui.sh. Use environment/windows/setup.ps1 on Windows." >&2
  exit 1
fi

ENV_FILE="${SCRIPT_DIR}/${TARGET_OS}/environment.yml"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing environment spec: ${ENV_FILE}" >&2
  exit 1
fi

ENV_NAME="$(awk -F':' '/^name:/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "${ENV_FILE}")"
if [[ -z "${ENV_NAME}" ]]; then
  echo "Could not parse environment name from ${ENV_FILE}." >&2
  exit 1
fi

if ! "${MAMBA_BIN}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "[run] Environment '${ENV_NAME}' not found. Running setup first..."
  bash "${SETUP_SCRIPT}"
fi

cd "${REPO_ROOT}"
exec "${MAMBA_BIN}" run -n "${ENV_NAME}" illustrate-gui
