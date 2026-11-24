#!/usr/bin/env python3
"""Build the macOS desktop app with PyInstaller."""

from __future__ import annotations

import argparse
from importlib import import_module
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
ICONS = REPO_ROOT / "illustrate" / "gui" / "icons"
DEFAULT_APP_NAME = "Illustrate"


def _run(command: list[str], *, cwd: Path = REPO_ROOT) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _project_version() -> str:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    in_project = False
    for line in pyproject.splitlines():
        stripped = line.strip()
        if stripped == "[project]":
            in_project = True
            continue
        if in_project and stripped.startswith("["):
            break
        if in_project:
            match = re.match(r'^version\s*=\s*["\']([^"\']+)["\']', stripped)
            if match:
                return match.group(1)
    return "0.0.0"


def _ensure_pyinstaller() -> None:
    try:
        import_module("PyInstaller")
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: PyInstaller. Run `make env` or install pyinstaller in the active environment."
        ) from exc


def _ensure_package_metadata(version: str) -> None:
    """Require installed ``illustrate-py`` metadata that matches the release version.

    The bundle copies this metadata so the frozen app can report its own version;
    without it the update check treats every published release as newer.
    """
    from importlib.metadata import PackageNotFoundError, version as package_version

    try:
        installed = package_version("illustrate-py")
    except PackageNotFoundError as exc:
        raise SystemExit(
            "Missing package metadata: illustrate-py is not installed in this environment. "
            "Run `make env` or install the repository (`uv pip install .`) before packaging."
        ) from exc
    if installed != version:
        raise SystemExit(
            f"Release version {version} does not match installed illustrate-py {installed}. "
            "Update [project].version in pyproject.toml and reinstall the package."
        )


def _pyinstaller_command(
    *,
    app_name: str,
    build_root: Path,
    work_path: Path,
    dist_path: Path,
    spec_path: Path,
) -> list[str]:
    entrypoint = REPO_ROOT / "illustrate" / "gui" / "main.py"

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name",
        app_name,
        "--collect-submodules",
        "illustrate.gui",
        # The compiled CPU backend and Numba are imported by name at runtime.
        "--hidden-import",
        "illustrate.rendering.numba_kernels",
        "--hidden-import",
        "numba",
        # The GUI loads its QSS, color tables, SVG icons, stylesheets and window icons
        # from package data at runtime.
        "--collect-data",
        "illustrate.gui",
        "--copy-metadata",
        "illustrate-py",
        "--workpath",
        str(work_path),
        "--distpath",
        str(dist_path),
        "--specpath",
        str(spec_path),
    ]

    command.append("--windowed")
    icon = ICONS / "icon.icns"
    if icon.exists():
        command.extend(["--icon", str(icon)])

    command.append(str(entrypoint))
    return command


def _package_macos(bundle: Path, build_root: Path, app_name: str, version: str) -> list[Path]:
    if not shutil.which("hdiutil"):
        raise SystemExit("Missing dependency: hdiutil (macOS tool).")
    if not shutil.which("ditto"):
        raise SystemExit("Missing dependency: ditto (macOS tool).")

    dmg_root = build_root / "dmg-root"
    dmg_path = build_root / f"{app_name}-{version}-macOS.dmg"
    zip_path = build_root / f"{app_name}-{version}-macOS-app.zip"
    shutil.rmtree(dmg_root, ignore_errors=True)
    if dmg_path.exists():
        dmg_path.unlink()
    if zip_path.exists():
        zip_path.unlink()

    # Sign the .app before it is copied or archived, so the dmg and the zip
    # both carry the signed bundle.
    codesign_identity = os.environ.get("CODESIGN_IDENTITY", "").strip()
    if codesign_identity:
        codesign_command = [
            "codesign",
            "--force",
            "--deep",
            "--timestamp",
            "--options",
            "runtime",
        ]
        entitlements = REPO_ROOT / "packaging" / "macos" / "entitlements.plist"
        if entitlements.exists():
            codesign_command.extend(["--entitlements", str(entitlements)])
        codesign_command.extend(["--sign", codesign_identity, str(bundle)])
        _run(codesign_command)
        _run(["codesign", "--verify", "--strict", "--verbose=2", str(bundle)])

    dmg_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(bundle, dmg_root / bundle.name, symlinks=True)
    applications_link = dmg_root / "Applications"
    if not applications_link.exists():
        applications_link.symlink_to("/Applications")

    _run(["hdiutil", "create", "-volname", app_name, "-srcfolder", str(dmg_root), "-ov", "-format", "UDZO", str(dmg_path)])

    # Codesign the disk image itself (not just the app inside it) so Gatekeeper
    # reports a Developer ID signature on the downloaded dmg. Must happen before
    # notarization/stapling, since re-signing invalidates a stapled ticket.
    if codesign_identity:
        _run(["codesign", "--force", "--timestamp", "--sign", codesign_identity, str(dmg_path)])

    # Notarize + staple when a notarytool keychain profile is provided. The
    # submission notarizes the app inside the dmg; we staple both the dmg and
    # the bundle so the standalone zip also carries the ticket.
    notary_profile = os.environ.get("NOTARY_PROFILE", "").strip()
    if notary_profile:
        if not codesign_identity:
            raise SystemExit("NOTARY_PROFILE is set but CODESIGN_IDENTITY is empty; notarization requires a signed app.")
        _run(["xcrun", "notarytool", "submit", str(dmg_path), "--keychain-profile", notary_profile, "--wait"])
        _run(["xcrun", "stapler", "staple", str(dmg_path)])
        _run(["xcrun", "stapler", "validate", str(dmg_path)])
        _run(["xcrun", "stapler", "staple", str(bundle)])

    _run(["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(bundle), str(zip_path)])
    return [dmg_path, zip_path]


def build(app_name: str, version: str, *, skip_package: bool) -> list[Path]:
    _ensure_pyinstaller()
    _ensure_package_metadata(version)

    if sys.platform != "darwin":
        raise SystemExit("The desktop app is built on macOS only.")

    build_root = REPO_ROOT / "dist" / "package" / "macos"
    work_path = build_root / "build"
    dist_path = build_root / "dist"
    spec_path = build_root / "spec"

    shutil.rmtree(work_path, ignore_errors=True)
    shutil.rmtree(dist_path, ignore_errors=True)
    shutil.rmtree(spec_path, ignore_errors=True)
    build_root.mkdir(parents=True, exist_ok=True)
    work_path.mkdir(parents=True, exist_ok=True)
    dist_path.mkdir(parents=True, exist_ok=True)
    spec_path.mkdir(parents=True, exist_ok=True)

    _run(
        _pyinstaller_command(
            app_name=app_name,
            build_root=build_root,
            work_path=work_path,
            dist_path=dist_path,
            spec_path=spec_path,
        )
    )

    bundle = dist_path / f"{app_name}.app"
    if not bundle.exists():
        raise SystemExit(f"Expected PyInstaller bundle not found: {bundle}")

    artifacts = [bundle]
    if not skip_package:
        artifacts.extend(_package_macos(bundle, build_root, app_name, version))

    print("[package] Built artifacts:", flush=True)
    for artifact in artifacts:
        print(f"  {artifact}", flush=True)
    return artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-name", default=DEFAULT_APP_NAME)
    parser.add_argument("--version", default="")
    parser.add_argument("--skip-package", action="store_true", help="Build PyInstaller output without archive/DMG packaging.")
    args = parser.parse_args()

    version = args.version.strip() or os.environ.get("VERSION", "").strip() or _project_version()
    build(args.app_name, version, skip_package=bool(args.skip_package))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
