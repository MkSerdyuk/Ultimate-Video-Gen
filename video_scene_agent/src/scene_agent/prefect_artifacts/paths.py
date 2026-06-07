
from __future__ import annotations

from pathlib import Path
from typing import Any

from scene_agent.prefect_artifacts.keys import _relative_to

def _uri_to_local_path(storage: Any, uri: str | None) -> Path | None:
    if not uri:
        return None
    try:
        return Path(storage.uri_to_local_path(uri)).resolve()
    except Exception:
        try:
            return Path(str(uri)).resolve()
        except Exception:
            return None


def _path_to_public_url(artifacts_dir: Path, storage: Any, path: Path | None) -> str:
    if path is None:
        return ""
    public_url_base = getattr(storage, "public_url_base", None)
    if not public_url_base:
        return ""
    relative_key = _relative_to(artifacts_dir, path)
    url = storage.get_public_uri(relative_key)
    return "" if str(url).startswith("file://") else str(url)


def _glob_relative(artifacts_dir: Path, pattern: str) -> list[str]:
    return [_relative_to(artifacts_dir, path) for path in artifacts_dir.glob(pattern)]


def _path_reference(artifacts_dir: Path, path: Path, storage: Any) -> str:
    public_url = _path_to_public_url(artifacts_dir, storage, path)
    local_path = f"`{path}`"
    if public_url:
        return f"[public link]({public_url}) ({local_path})"
    return f"{local_path} (local-only)"
