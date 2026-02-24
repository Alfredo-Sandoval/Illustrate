"""Simple release manifest update checker for desktop GUI."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from importlib.metadata import PackageNotFoundError, version as package_version
except Exception:  # pragma: no cover - stdlib import safety
    PackageNotFoundError = Exception  # type: ignore[assignment]
    package_version = None  # type: ignore[assignment]


MANIFEST_URL = "https://github.com/Alfredo-Sandoval/Illustrate/releases/latest/download/latest.json"
RELEASES_PAGE_URL = "https://github.com/Alfredo-Sandoval/Illustrate/releases"


@dataclass(frozen=True)
class UpdateCheckResult:
    status: str
    current_version: str
    latest_version: str | None = None
    download_url: str | None = None
    message: str = ""


def _installed_version() -> str:
    if package_version is None:
        return "0.0.0"
    try:
        return str(package_version("illustrate"))
    except PackageNotFoundError:
        return "0.0.0"
    except Exception:
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


def _platform_name() -> str:
    if sys.platform.startswith("darwin"):
        return "macos"
    if sys.platform.startswith("win"):
        return "windows"
    return "linux"


def _extract_download_url(payload: dict[str, Any]) -> str | None:
    channels = payload.get("channels")
    if isinstance(channels, dict):
        stable = channels.get("stable")
        if isinstance(stable, dict):
            platform_entry = stable.get(_platform_name())
            if isinstance(platform_entry, dict):
                candidate = platform_entry.get("url")
                if isinstance(candidate, str) and candidate:
                    return candidate
    candidate = payload.get("url")
    if isinstance(candidate, str) and candidate:
        return candidate
    return None


def check_for_updates(timeout_s: float = 4.0) -> UpdateCheckResult:
    current = _installed_version()
    request = Request(
        MANIFEST_URL,
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
    except Exception as exc:
        return UpdateCheckResult(
            status="error",
            current_version=current,
            message=f"Update check failed: {exc}",
        )

    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception as exc:
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

    latest = payload.get("version")
    if not isinstance(latest, str) or not latest.strip():
        return UpdateCheckResult(
            status="error",
            current_version=current,
            message="Invalid update payload: missing version.",
        )
    latest = latest.strip()
    download_url = _extract_download_url(payload)

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
