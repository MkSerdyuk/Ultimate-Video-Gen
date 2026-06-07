
from __future__ import annotations

from pathlib import Path
from typing import Any

from scene_agent.models import SceneState
from scene_agent.prefect_artifacts.paths import _glob_relative, _path_reference, _path_to_public_url, _uri_to_local_path

def build_artifact_links_markdown(artifacts_dir: Path, storage: Any, final: bool = False) -> str:
    """Render a compact artifact index with public links when available."""
    sections = [
        ("Core outputs", ["manifest.json", "world.json", "storyboard.json", "reports/run-report.md"]),
        ("Rendered videos", sorted(_glob_relative(artifacts_dir, "*.mp4"))),
        ("Media previews", sorted(_glob_relative(artifacts_dir, "previews/*"))),
        ("Storyboard markdown", sorted(_glob_relative(artifacts_dir, "storyboards/*.md"))),
        ("Review payloads", sorted(_glob_relative(artifacts_dir, "reviews/*.json"))),
    ]

    lines = ["# Artifact Links", ""]
    if final:
        lines.append("Run is in a terminal state. Links below point to the latest persisted outputs.")
        lines.append("")

    for title, items in sections:
        available = [item for item in items if (artifacts_dir / item).exists()]
        if not available:
            continue
        lines.append(f"## {title}")
        lines.append("")
        for item in available:
            ref = _path_reference(artifacts_dir, artifacts_dir / item, storage)
            lines.append(f"- **{Path(item).name}**: {ref}")
        lines.append("")

    return "\n".join(lines).strip()


def final_video_public_url(state: SceneState, artifacts_dir: Path, storage: Any) -> str:
    """Return the public URL for the final video when it is available."""
    local_path = _uri_to_local_path(storage, state.final_video_uri)
    return _path_to_public_url(artifacts_dir, storage, local_path) if local_path else ""
