#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

APP_NAME="${APP_NAME:-Illustrate}"
VERSION="${VERSION:-}"
if [[ -z "${VERSION}" ]]; then
  VERSION="$(python - <<'PY'
import tomllib
from pathlib import Path
payload = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
print(payload.get("project", {}).get("version", "0.0.0"))
PY
)"
fi

BUILD_ROOT="${REPO_ROOT}/dist/package/macos"
WORK_PATH="${BUILD_ROOT}/build"
DIST_PATH="${BUILD_ROOT}/dist"
SPEC_PATH="${BUILD_ROOT}/spec"
STAGE_PATH="${BUILD_ROOT}/dmg-root"
APP_PATH="${DIST_PATH}/${APP_NAME}.app"
DMG_PATH="${BUILD_ROOT}/${APP_NAME}-${VERSION}-macOS.dmg"
ZIP_PATH="${BUILD_ROOT}/${APP_NAME}-${VERSION}-macOS-app.zip"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "packaging/macos/build_app.sh must run on macOS." >&2
  exit 1
fi

cd "${REPO_ROOT}"

if ! command -v hdiutil >/dev/null 2>&1; then
  echo "Missing dependency: hdiutil (macOS tool)." >&2
  exit 1
fi

if ! python -c "import PyInstaller" >/dev/null 2>&1; then
  echo "[package] Installing pyinstaller into active Python..."
  python -m pip install pyinstaller
fi

rm -rf "${WORK_PATH}" "${DIST_PATH}" "${SPEC_PATH}" "${STAGE_PATH}" "${DMG_PATH}" "${ZIP_PATH}"
mkdir -p "${WORK_PATH}" "${DIST_PATH}" "${SPEC_PATH}" "${STAGE_PATH}"

echo "[package] Building ${APP_NAME}.app (${VERSION})..."
python -m PyInstaller \
  --noconfirm \
  --clean \
  --windowed \
  --name "${APP_NAME}" \
  --icon "${REPO_ROOT}/data/icon.icns" \
  --add-data "${REPO_ROOT}/data:data" \
  --collect-submodules "illustrate_gui" \
  --workpath "${WORK_PATH}" \
  --distpath "${DIST_PATH}" \
  --specpath "${SPEC_PATH}" \
  "${REPO_ROOT}/illustrate_gui/main.py"

if [[ ! -d "${APP_PATH}" ]]; then
  echo "Expected app bundle not found: ${APP_PATH}" >&2
  exit 1
fi

if [[ -n "${CODESIGN_IDENTITY:-}" ]]; then
  echo "[package] Signing app with identity: ${CODESIGN_IDENTITY}"
  codesign --force --deep --timestamp --options runtime --sign "${CODESIGN_IDENTITY}" "${APP_PATH}"
fi

cp -R "${APP_PATH}" "${STAGE_PATH}/"
ln -s /Applications "${STAGE_PATH}/Applications"

echo "[package] Creating DMG..."
hdiutil create \
  -volname "${APP_NAME}" \
  -srcfolder "${STAGE_PATH}" \
  -ov \
  -format UDZO \
  "${DMG_PATH}"

echo "[package] Creating ZIP for app bundle..."
ditto -c -k --sequesterRsrc --keepParent "${APP_PATH}" "${ZIP_PATH}"

echo "[package] Done."
echo "  App: ${APP_PATH}"
echo "  DMG: ${DMG_PATH}"
echo "  ZIP: ${ZIP_PATH}"
