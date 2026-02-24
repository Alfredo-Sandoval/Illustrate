#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

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
    *MINGW* | *CYGWIN* | *MSYS*)
      echo "windows"
      ;;
    *)
      echo ""
      ;;
  esac
}

TARGET_OS="${ARC_SETUP_OS:-}"
if [[ -z "${TARGET_OS}" ]]; then
  TARGET_OS="$(detect_target_os)"
fi

case "${TARGET_OS}" in
  macos | linux | windows)
    ;;
  "")
    echo "[setup] Unsupported host OS: $(uname -s). Use environment/<os>/setup.sh directly." >&2
    exit 1
    ;;
  *)
    echo "[setup] Invalid ARC_SETUP_OS='${TARGET_OS}'. Expected 'macos', 'linux', or 'windows'." >&2
    exit 1
    ;;
esac

if [[ "${TARGET_OS}" == "windows" ]]; then
  SETUP_SCRIPT="${SCRIPT_DIR}/windows/setup.ps1"
  if [[ ! -f "${SETUP_SCRIPT}" ]]; then
    echo "[setup] Missing setup script: ${SETUP_SCRIPT}" >&2
    exit 1
  fi
  if command -v powershell >/dev/null 2>&1; then
    echo "[setup] Using ${SETUP_SCRIPT}"
    exec powershell -NoProfile -ExecutionPolicy Bypass -File "${SETUP_SCRIPT}" "$@"
  elif command -v pwsh >/dev/null 2>&1; then
    echo "[setup] Using ${SETUP_SCRIPT}"
    exec pwsh -NoProfile -ExecutionPolicy Bypass -File "${SETUP_SCRIPT}" "$@"
  else
    echo "[setup] Missing dependency: install PowerShell." >&2
    exit 1
  fi
fi

SETUP_SCRIPT="${SCRIPT_DIR}/${TARGET_OS}/setup.sh"
if [[ ! -f "${SETUP_SCRIPT}" ]]; then
  echo "[setup] Missing setup script: ${SETUP_SCRIPT}" >&2
  exit 1
fi

echo "[setup] Using ${SETUP_SCRIPT}"
exec bash "${SETUP_SCRIPT}" "$@"
