"""GitHub release lookup for the desktop Help → Check Updates action."""

from __future__ import annotations

from dataclasses import dataclass
import json
from importlib.metadata import PackageNotFoundError, version as package_version
import re
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_MANIFEST_URL = "https://github.com/Alfredo-Sandoval/Illustrate/releases/latest/download/latest.json"
RELEASES_PAGE_URL = "https://github.com/Alfredo-Sandoval/Illustrate/releases"


@dataclass(frozen=True)
class UpdateCheckResult:
    status: str
    current_version: str
    latest_version: str | None = None
    download_url: str | None = None
    message: str = ""


def _installed_version() -> str:
    try:
        return str(package_version("illustrate-py"))
    except PackageNotFoundError:
        return "0.0.0"


def _version_key(value: str) -> tuple[int, ...]:
    numeric = [int(part) for part in re.findall(r"\d+", str(value))]
    return tuple(numeric) if numeric else (0,)


def _is_newer(latest: str, current: str) -> bool:
    a = _version_key(latest)
    b = _version_key(current)
    width = max(len(a), len(b))
    a_padded = a + (0,) * (width - len(a))
    b_padded = b + (0,) * (width - len(b))
    return a_padded > b_padded


def _extract_download_url(payload: dict[str, object]) -> str | None:
    channels = payload.get("channels")
    if isinstance(channels, dict):
        stable = cast(dict[str, object], channels).get("stable")
        if isinstance(stable, dict):
            platform_entry = cast(dict[str, object], stable).get("macos")
            if isinstance(platform_entry, dict):
                candidate = cast(dict[str, object], platform_entry).get("url")
                if isinstance(candidate, str) and candidate:
                    return candidate
    candidate = payload.get("url")
    if isinstance(candidate, str) and candidate:
        return candidate
    return None


def check_for_updates(timeout_s: float = 4.0) -> UpdateCheckResult:
    """Compare the installed version with the latest GitHub release manifest.

    This makes a blocking network request. The desktop app calls it in a
    worker. An available update includes the stable macOS download URL when
    the manifest supplies one.
    """
    current = _installed_version()
    request = Request(
        _MANIFEST_URL,
        headers={
            "User-Agent": "Illustrate-Desktop-UpdateCheck/1.0",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            body = response.read()
    except HTTPError as exc:
        return UpdateCheckResult(
            status="error",
            current_version=current,
            message=f"Update server returned HTTP {exc.code}.",
        )
    except URLError as exc:
        return UpdateCheckResult(
            status="error",
            current_version=current,
            message=f"Could not reach update server: {exc.reason}",
        )
    except TimeoutError as exc:
        return UpdateCheckResult(
            status="error",
            current_version=current,
            message=f"Update check failed: {exc}",
        )

    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return UpdateCheckResult(
            status="error",
            current_version=current,
            message=f"Invalid update payload: {exc}",
        )
    if not isinstance(payload, dict):
        return UpdateCheckResult(
            status="error",
            current_version=current,
            message="Invalid update payload: expected JSON object.",
        )

    manifest = cast(dict[str, object], payload)
    latest = manifest.get("version")
    if not isinstance(latest, str) or not latest.strip():
        return UpdateCheckResult(
            status="error",
            current_version=current,
            message="Invalid update payload: missing version.",
        )
    latest = latest.strip()
    download_url = _extract_download_url(manifest)

    if _is_newer(latest, current):
        return UpdateCheckResult(
            status="update_available",
            current_version=current,
            latest_version=latest,
            download_url=download_url,
            message=f"New version available: {latest}",
        )
    return UpdateCheckResult(
        status="up_to_date",
        current_version=current,
        latest_version=latest,
        download_url=download_url,
        message=f"You are up to date ({current}).",
    )
