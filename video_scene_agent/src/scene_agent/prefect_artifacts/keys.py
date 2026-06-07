
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

def artifact_key(run_id: str, suffix: str) -> str:
    """Build a Prefect-safe artifact key for a run-scoped view."""
    raw = f"scene-agent-{run_id}-{suffix}".lower()
    key = re.sub(r"[^a-z0-9-]+", "-", raw).strip("-")
    return key[:255]


def _basename(value: Any) -> str:
    if not value:
        return ""
    text = str(value)
    if "://" in text:
        text = text.rstrip("/").split("/")[-1]
    return Path(text).name


def _relative_to(base: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return path.as_posix()
