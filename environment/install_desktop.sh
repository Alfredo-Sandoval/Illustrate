#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

LAUNCH_AFTER_INSTALL=0
FORCE_REINSTALL=0

usage() {
  cat <<'USAGE'
Usage: bash environment/install_desktop.sh [--launch] [--force-reinstall]

Options:
  --launch           Launch illustrate-gui after installation succeeds.
  --force-reinstall  Reinstall runtime dependencies even if already healthy.
  --help             Show this help text.
USAGE
}

log() {
  echo "[desktop-install] $*"
}

fail() {
  echo "[desktop-install] ERROR: $*" >&2
  exit 1
}

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

detect_target_arch() {
  local uname_m
  uname_m="$(uname -m)"
  case "${uname_m}" in
    arm64 | aarch64)
      echo "arm64"
      ;;
    x86_64 | amd64)
      echo "x86_64"
      ;;
    *)
      echo ""
      ;;
  esac
}

micromamba_platform() {
  local target_os="$1"
  local target_arch="$2"
  case "${target_os}/${target_arch}" in
    macos/arm64)
      echo "osx-arm64"
      ;;
    macos/x86_64)
      echo "osx-64"
      ;;
    linux/arm64)
      echo "linux-aarch64"
      ;;
    linux/x86_64)
      echo "linux-64"
      ;;
    *)
      echo ""
      ;;
  esac
}

find_mamba_bin() {
  if command -v micromamba >/dev/null 2>&1; then
    command -v micromamba
    return 0
  fi
  if command -v mamba >/dev/null 2>&1; then
    command -v mamba
    return 0
  fi

  local default_micromamba="${HOME}/.local/bin/micromamba"
  if [[ -x "${default_micromamba}" ]]; then
    echo "${default_micromamba}"
    return 0
  fi

  return 1
}

bootstrap_micromamba() {
  local target_os="$1"
  local target_arch="$2"
  local install_dir="${ILLUSTRATE_MICROMAMBA_BIN_DIR:-${HOME}/.local/bin}"
  local install_target="${install_dir}/micromamba"
  local platform
  platform="$(micromamba_platform "${target_os}" "${target_arch}")"
  if [[ -z "${platform}" ]]; then
    fail "Automatic micromamba install is not supported for ${target_os}/${target_arch}."
  fi

  if ! command -v curl >/dev/null 2>&1; then
    fail "Missing dependency: curl (required to bootstrap micromamba)."
  fi
  if ! command -v tar >/dev/null 2>&1; then
    fail "Missing dependency: tar (required to bootstrap micromamba)."
  fi

  mkdir -p "${install_dir}"
  local tmp_dir
  tmp_dir="$(mktemp -d)"

  local archive_path="${tmp_dir}/micromamba.tar.bz2"
  local download_url="https://micro.mamba.pm/api/micromamba/${platform}/latest"

  log "Installing micromamba (${platform}) into ${install_dir}."
  if ! curl -fsSL -o "${archive_path}" "${download_url}"; then
    rm -rf "${tmp_dir}"
    fail "Could not download micromamba from ${download_url}."
  fi
  if ! tar -xjf "${archive_path}" -C "${tmp_dir}"; then
    rm -rf "${tmp_dir}"
    fail "Could not extract micromamba archive."
  fi

  local candidate
  candidate="$(find "${tmp_dir}" -type f -name micromamba | head -n 1 || true)"
  if [[ -z "${candidate}" ]]; then
    rm -rf "${tmp_dir}"
    fail "Could not locate micromamba binary in downloaded archive."
  fi

  if ! cp "${candidate}" "${install_target}"; then
    rm -rf "${tmp_dir}"
    fail "Could not copy micromamba to ${install_target}."
  fi
  chmod +x "${install_target}"
  PATH="${install_dir}:${PATH}"
  rm -rf "${tmp_dir}"
  log "Micromamba installed at ${install_target}."
}

parse_env_name() {
  local env_file="$1"
  local parsed
  parsed="$(awk -F':' '/^name:/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "${env_file}")"
  if [[ -z "${parsed}" ]]; then
    fail "Could not parse environment name from ${env_file}. Expected 'name: <env_name>'."
  fi
  echo "${parsed}"
}

env_exists() {
  local mamba_bin="$1"
  local env_name="$2"
  "${mamba_bin}" env list | awk '{print $1}' | grep -Fxq "${env_name}"
}

ensure_uv_in_env() {
  local mamba_bin="$1"
  local env_name="$2"
  if "${mamba_bin}" run -n "${env_name}" uv --version >/dev/null 2>&1; then
    return 0
  fi
  log "Installing missing uv tool into '${env_name}'."
  "${mamba_bin}" install -n "${env_name}" -c conda-forge uv -y
}

runtime_is_healthy() {
  local mamba_bin="$1"
  local env_name="$2"

  ILLUSTRATE_REPO_ROOT="${REPO_ROOT}" \
    "${mamba_bin}" run -n "${env_name}" python - <<'PY'
from __future__ import annotations

import os
import pathlib
import sys

repo_root = pathlib.Path(os.environ["ILLUSTRATE_REPO_ROOT"]).resolve()

try:
    import PySide6  # noqa: F401
    import illustrate
    import illustrate_gui  # noqa: F401
except Exception:
    raise SystemExit(1)

module_path = pathlib.Path(illustrate.__file__).resolve()
if repo_root not in module_path.parents:
    raise SystemExit(1)

sys.exit(0)
PY
}

install_runtime() {
  local mamba_bin="$1"
  local env_name="$2"
  log "Installing Illustrate GUI runtime into '${env_name}'."
  "${mamba_bin}" run -n "${env_name}" uv pip install --upgrade -e "${REPO_ROOT}[gui]"
}

for arg in "$@"; do
  case "${arg}" in
    --launch)
      LAUNCH_AFTER_INSTALL=1
      ;;
    --force-reinstall)
      FORCE_REINSTALL=1
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown argument: ${arg}. Use --help for usage."
      ;;
  esac
done

TARGET_OS="$(detect_target_os)"
if [[ -z "${TARGET_OS}" ]]; then
  fail "Unsupported host OS: $(uname -s). Desktop installer currently supports macOS and Linux."
fi

TARGET_ARCH="$(detect_target_arch)"
if [[ -z "${TARGET_ARCH}" ]]; then
  fail "Unsupported CPU architecture: $(uname -m)."
fi

ENV_FILE="${SCRIPT_DIR}/${TARGET_OS}/environment.yml"
if [[ ! -f "${ENV_FILE}" ]]; then
  fail "Missing environment spec: ${ENV_FILE}"
fi

MAMBA_BIN="$(find_mamba_bin || true)"
if [[ -z "${MAMBA_BIN}" ]]; then
  if [[ "${ILLUSTRATE_AUTO_INSTALL_MICROMAMBA:-1}" != "1" ]]; then
    fail "Missing dependency: install micromamba or mamba, or set ILLUSTRATE_AUTO_INSTALL_MICROMAMBA=1."
  fi
  bootstrap_micromamba "${TARGET_OS}" "${TARGET_ARCH}"
  MAMBA_BIN="$(find_mamba_bin || true)"
fi

if [[ -z "${MAMBA_BIN}" ]]; then
  fail "Could not locate micromamba/mamba after bootstrap."
fi

ENV_NAME="$(parse_env_name "${ENV_FILE}")"

if env_exists "${MAMBA_BIN}" "${ENV_NAME}"; then
  log "Using existing environment '${ENV_NAME}'."
else
  log "Creating environment '${ENV_NAME}' from ${ENV_FILE}."
  "${MAMBA_BIN}" env create -n "${ENV_NAME}" -f "${ENV_FILE}" -y
fi

ensure_uv_in_env "${MAMBA_BIN}" "${ENV_NAME}"

if [[ "${FORCE_REINSTALL}" == "1" ]] || ! runtime_is_healthy "${MAMBA_BIN}" "${ENV_NAME}"; then
  install_runtime "${MAMBA_BIN}" "${ENV_NAME}"
fi

if ! runtime_is_healthy "${MAMBA_BIN}" "${ENV_NAME}"; then
  fail "Runtime verification failed after installation."
fi

log "Illustrate desktop runtime is ready."

if [[ "${LAUNCH_AFTER_INSTALL}" == "1" ]]; then
  cd "${REPO_ROOT}"
  log "Launching Illustrate GUI."
  exec "${MAMBA_BIN}" run -n "${ENV_NAME}" python -m illustrate_gui
fi
