
from __future__ import annotations

from pathlib import Path
from typing import Any

from scene_agent.models import RunEvent, SceneState
from scene_agent.prefect_artifacts.paths import _path_to_public_url, _uri_to_local_path

def build_media_catalog_rows(state: SceneState, artifacts_dir: Path, storage: Any) -> list[dict[str, Any]]:
    """Build user-facing rows for generated media and the final video."""
    rows: list[dict[str, Any]] = []

    for idx, uri in enumerate(state.frame_uris):
        rows.append(
            _media_row(
                artifacts_dir=artifacts_dir,
                storage=storage,
                kind="frame",
                label=f"Keyframe {idx + 1}",
                uri=uri,
            )
        )

    storyboard = state.storyboard
    for idx, uri in enumerate(state.segment_uris):
        if storyboard and idx < len(storyboard.segments):
            segment = storyboard.segments[idx]
            detail = f"frame {segment.start_frame_idx + 1} -> frame {segment.end_frame_idx + 1}"
        else:
            detail = ""

        rows.append(
            _media_row(
                artifacts_dir=artifacts_dir,
                storage=storage,
                kind="segment",
                label=f"Segment {idx + 1}: {detail}" if detail else f"Segment {idx + 1}",
                uri=uri,
            )
        )

    if state.final_video_uri:
        rows.append(
            _media_row(
                artifacts_dir=artifacts_dir,
                storage=storage,
                kind="final_video",
                label="Final video",
                uri=state.final_video_uri,
            )
        )

    return rows


def build_review_summary_rows(events: list[RunEvent]) -> list[dict[str, Any]]:
    """Build rows for storyboard/video review and fix history."""
    rows: list[dict[str, Any]] = []
    for event in events:
        if event.asset_kind not in {"review", "fix"}:
            continue

        rows.append(
            {
                "stage": event.stage,
                "label": event.label,
                "action": event.action,
                "iteration": event.counts.get("iteration", ""),
                "issues": event.counts.get("issues", ""),
                "mode": event.mode or "",
                "regen_frames": _format_indices(event.details.get("regen_frames", [])),
                "regen_segments": _format_indices(event.details.get("regen_segments", [])),
                "edit_segments": _format_indices(event.details.get("edit_segments", [])),
                "retry": event.retry,
                "error": event.error or "",
            }
        )

    return rows

def _media_row(artifacts_dir: Path, storage: Any, kind: str, label: str, uri: str) -> dict[str, Any]:
    local_path = _uri_to_local_path(storage, uri)
    public_url = _path_to_public_url(artifacts_dir, storage, local_path) if local_path else ""
    return {
        "type": kind,
        "label": label,
        "source_uri": uri,
        "local_path": str(local_path) if local_path else "",
        "public_url": public_url,
        "access": "public" if public_url else "local-only",
    }

def _format_indices(indices: list[int]) -> str:
    if not indices:
        return ""
    return ", ".join(str(idx) for idx in indices)
