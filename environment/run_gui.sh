#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_SCRIPT="${SCRIPT_DIR}/install_desktop.sh"

if [[ ! -f "${INSTALL_SCRIPT}" ]]; then
  echo "Missing installer script: ${INSTALL_SCRIPT}" >&2
  exit 1
fi

exec bash "${INSTALL_SCRIPT}" --launch "$@"
