#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "packaging/macos/build_app.sh must run on macOS." >&2
  exit 1
fi

cd "${REPO_ROOT}"
python packaging/pyinstaller/build_desktop.py "$@"
