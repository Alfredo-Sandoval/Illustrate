#!/usr/bin/env bash
set -euo pipefail

# Setup entrypoint for Illustrate on macOS.
#
# Creates or updates the conda environment from environment/macos/, installs
# the Python dependencies with uv pip, and registers the package.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

HOST_CHECK=0
LAUNCH_AFTER_INSTALL=0
FORCE_REINSTALL=0

usage() {
  cat <<'USAGE'
Usage: bash environment/setup.sh [options]

Options:
  --host-check        Print host inspection and exit (no changes).
  --launch            Launch illustrate-gui after setup succeeds.
  --force-reinstall   Reinstall the package even if already healthy.
  --help              Show this help text.
USAGE
}

log() { echo "[setup] $*"; }
fail() {
  echo "[setup] ERROR: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host-check)
      HOST_CHECK=1
      shift
      ;;
    --launch)
      LAUNCH_AFTER_INSTALL=1
      shift
      ;;
    --force-reinstall)
      FORCE_REINSTALL=1
      shift
      ;;
    --help | -h)
      usage
      exit 0
      ;;
    *)
      fail "Unknown argument: $1. Use --help for usage."
      ;;
  esac
done

[[ "$(uname -s)" == "Darwin" ]] || fail "Illustrate setup supports macOS only; this host is $(uname -s)."

detect_target_arch() {
  case "$(uname -m)" in
    arm64 | aarch64) echo "arm64" ;;
    x86_64 | amd64) echo "x86_64" ;;
    *) echo "" ;;
  esac
}

if [[ "${HOST_CHECK}" == "1" ]]; then
  echo "[host-check] os:        $(uname -s)"
  echo "[host-check] arch:      $(uname -m)"
  if command -v mamba >/dev/null 2>&1; then
    echo "[host-check] mamba:     $(command -v mamba)"
  elif command -v micromamba >/dev/null 2>&1; then
    echo "[host-check] mamba:     $(command -v micromamba) (micromamba)"
  else
    echo "[host-check] mamba:     not found (will bootstrap micromamba)"
  fi
  exit 0
fi

ENV_FILE="${SCRIPT_DIR}/macos/environment.yml"
REQ_FILE="${SCRIPT_DIR}/macos/requirements.txt"
[[ -f "${ENV_FILE}" ]] || fail "Missing environment spec: ${ENV_FILE}"
[[ -f "${REQ_FILE}" ]] || fail "Missing requirements file: ${REQ_FILE}"

micromamba_platform() {
  case "$(detect_target_arch)" in
    arm64) echo "osx-arm64" ;;
    x86_64) echo "osx-64" ;;
    *) echo "" ;;
  esac
}

bootstrap_micromamba() {
  local install_dir="${ILLUSTRATE_MICROMAMBA_BIN_DIR:-${HOME}/.local/bin}"
  local install_target="${install_dir}/micromamba"
  local platform
  platform="$(micromamba_platform)"
  [[ -n "${platform}" ]] || fail "Automatic micromamba install is unsupported for $(uname -m)."
  command -v curl >/dev/null 2>&1 || fail "Missing dependency: curl (required to bootstrap micromamba)."
  command -v tar >/dev/null 2>&1 || fail "Missing dependency: tar (required to bootstrap micromamba)."

  mkdir -p "${install_dir}"
  local tmp_dir
  tmp_dir="$(mktemp -d)"
  local archive_path="${tmp_dir}/micromamba.tar.bz2"
  local download_url="https://micro.mamba.pm/api/micromamba/${platform}/latest"

  log "Installing micromamba (${platform}) into ${install_dir}."
  curl -fsSL -o "${archive_path}" "${download_url}" || {
    rm -rf "${tmp_dir}"
    fail "Could not download micromamba from ${download_url}."
  }
  tar -xjf "${archive_path}" -C "${tmp_dir}" || {
    rm -rf "${tmp_dir}"
    fail "Could not extract micromamba archive."
  }
  local candidate
  candidate="$(find "${tmp_dir}" -type f -name micromamba | head -n 1 || true)"
  [[ -n "${candidate}" ]] || {
    rm -rf "${tmp_dir}"
    fail "Could not locate micromamba binary in downloaded archive."
  }
  cp "${candidate}" "${install_target}" || {
    rm -rf "${tmp_dir}"
    fail "Could not copy micromamba to ${install_target}."
  }
  chmod +x "${install_target}"
  PATH="${install_dir}:${PATH}"
  export PATH
  rm -rf "${tmp_dir}"
  log "Micromamba installed at ${install_target}."
}

resolve_mamba_bin() {
  if command -v mamba >/dev/null 2>&1; then
    command -v mamba
  elif command -v micromamba >/dev/null 2>&1; then
    command -v micromamba
  elif [[ -x "${HOME}/.local/bin/micromamba" ]]; then
    echo "${HOME}/.local/bin/micromamba"
  fi
}

MAMBA_BIN="$(resolve_mamba_bin)"
if [[ -z "${MAMBA_BIN}" ]]; then
  if [[ "${ILLUSTRATE_AUTO_INSTALL_MICROMAMBA:-1}" != "1" ]]; then
    fail "Missing dependency: install mamba (or micromamba), or set ILLUSTRATE_AUTO_INSTALL_MICROMAMBA=1."
  fi
  bootstrap_micromamba
  MAMBA_BIN="$(resolve_mamba_bin)"
fi
[[ -n "${MAMBA_BIN}" ]] || fail "Could not locate mamba after bootstrap."

ENV_NAME="$(awk -F':' '/^name:/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "${ENV_FILE}")"
[[ -n "${ENV_NAME}" ]] || fail "Could not parse environment name from ${ENV_FILE}. Expected 'name: <env_name>'."

if "${MAMBA_BIN}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  log "Updating environment '${ENV_NAME}'."
  "${MAMBA_BIN}" env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune -y
else
  log "Creating environment '${ENV_NAME}' from ${ENV_FILE}."
  "${MAMBA_BIN}" env create -n "${ENV_NAME}" -f "${ENV_FILE}" -y
fi

if ! "${MAMBA_BIN}" run -n "${ENV_NAME}" uv --version >/dev/null 2>&1; then
  log "Installing uv into '${ENV_NAME}'."
  "${MAMBA_BIN}" install -n "${ENV_NAME}" -c conda-forge uv -y
fi

log "Installing Python dependencies from ${REQ_FILE}."
"${MAMBA_BIN}" run -n "${ENV_NAME}" uv pip install -r "${REQ_FILE}"

runtime_is_healthy() {
  # -I keeps the current directory off sys.path. Setup runs from the repo
  # root, so without it the source tree imports even when the package (and its
  # illustrate-gui entry point) was never installed into the environment.
  ILLUSTRATE_REPO_ROOT="${REPO_ROOT}" "${MAMBA_BIN}" run -n "${ENV_NAME}" python -I - <<'PY'
import os
import pathlib

repo_root = pathlib.Path(os.environ["ILLUSTRATE_REPO_ROOT"]).resolve()
try:
    import PySide6  # noqa: F401
    import illustrate
    import illustrate.gui  # noqa: F401
except Exception:
    raise SystemExit(1)

if repo_root not in pathlib.Path(illustrate.__file__).resolve().parents:
    raise SystemExit(1)
PY
}

if [[ "${FORCE_REINSTALL}" == "1" ]] || ! runtime_is_healthy; then
  log "Registering the illustrate package (editable, gui extra)."
  "${MAMBA_BIN}" run -n "${ENV_NAME}" uv pip install --upgrade -e "${REPO_ROOT}[gui]"
fi

runtime_is_healthy || fail "Runtime verification failed after setup."
log "Environment '${ENV_NAME}' is ready."

if [[ "${LAUNCH_AFTER_INSTALL}" == "1" ]]; then
  log "Launching Illustrate GUI."
  cd "${REPO_ROOT}"
  exec "${MAMBA_BIN}" run -n "${ENV_NAME}" illustrate-gui
fi
