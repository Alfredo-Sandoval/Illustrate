#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/environment.yml"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

install_js_dependencies_if_present() {
  local js_roots=("${REPO_ROOT}" "${REPO_ROOT}/illustrate_web/frontend")
  local pkg_json
  local manager
  local -a install_cmd
  local found_js
  local root

  found_js=0
  for root in "${js_roots[@]}"; do
    pkg_json="${root}/package.json"
    if [[ ! -f "${pkg_json}" ]]; then
      continue
    fi

    if [[ -f "${root}/pnpm-lock.yaml" ]]; then
      manager="pnpm"
      install_cmd=(pnpm install --frozen-lockfile)
    elif [[ -f "${root}/package-lock.json" ]]; then
      manager="npm"
      install_cmd=(npm ci)
    elif [[ -f "${root}/yarn.lock" ]]; then
      manager="yarn"
      install_cmd=(yarn install --frozen-lockfile)
    else
      manager="npm"
      install_cmd=(npm install)
    fi

    if ! "${MAMBA_BIN}" run -n "${ENV_NAME}" "${manager}" --version >/dev/null 2>&1; then
      echo "Missing dependency in environment '${ENV_NAME}': ${manager} (required by ${pkg_json})." >&2
      exit 1
    fi

    echo "[setup] Installing JS dependencies for ${pkg_json} using ${manager}."
    (
      cd "${root}"
      "${MAMBA_BIN}" run -n "${ENV_NAME}" "${install_cmd[@]}"
    )
    found_js=1
  done

  if [[ "${found_js}" == "0" ]]; then
    echo "[setup] No package.json found for JS install." >&2
  fi
}

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing environment spec: ${ENV_FILE}" >&2
  exit 1
fi

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "Missing requirements file: ${REQ_FILE}" >&2
  exit 1
fi

if command -v micromamba >/dev/null 2>&1; then
  MAMBA_BIN="micromamba"
elif command -v mamba >/dev/null 2>&1; then
  MAMBA_BIN="mamba"
else
  echo "Missing dependency: install micromamba or mamba." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Missing dependency: install uv." >&2
  exit 1
fi

ENV_NAME="$(awk -F':' '/^name:/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "${ENV_FILE}")"
if [[ -z "${ENV_NAME}" ]]; then
  echo "Could not parse environment name from ${ENV_FILE}. Expected 'name: <env_name>'." >&2
  exit 1
fi

if "${MAMBA_BIN}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  "${MAMBA_BIN}" env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune -y
else
  "${MAMBA_BIN}" env create -n "${ENV_NAME}" -f "${ENV_FILE}" -y
fi

"${MAMBA_BIN}" run -n "${ENV_NAME}" uv pip install -r "${REQ_FILE}"
install_js_dependencies_if_present

echo "Environment '${ENV_NAME}' is ready."
