
from __future__ import annotations

from typing import Any

from prefect import runtime as prefect_runtime

from scene_agent.models import SceneState
from scene_agent.prefect_logging import get_prefect_logger
from scene_agent.runtime import RuntimeSettings, SceneAgentRuntime

def _runtime(settings: RuntimeSettings) -> SceneAgentRuntime:
    """Build a run-scoped runtime for a Prefect task."""
    return SceneAgentRuntime(settings)


def _record_review(state: SceneState, bucket: str, payload: dict[str, Any]) -> None:
    """Append review payloads to the state review history."""
    state.reviews.setdefault(bucket, [])
    state.reviews[bucket].append(payload)


def _task_logger(settings: RuntimeSettings, step: str):
    """Get a Prefect run logger for a task step."""
    return get_prefect_logger(component="task", step=step, scene_run_id=settings.run_id)


def _task_retry_count() -> int:
    """Return the zero-based retry count for the current Prefect task attempt."""
    try:
        run_count = int(prefect_runtime.task_run.run_count or 1)
    except Exception:
        run_count = 1
    return max(run_count - 1, 0)


def _sync_state(runtime: SceneAgentRuntime, state: SceneState, *, final: bool = False) -> None:
    """Persist local artifacts and refresh Prefect UI artifacts."""
    runtime.persist_state(state)
    runtime.publish_prefect_artifacts(state, final=final)


def _record_frame_asset_events(
    runtime: SceneAgentRuntime,
    before: SceneState,
    after: SceneState,
    *,
    stage: str,
    retry: int,
) -> int:
    """Record per-frame URI replacements for reporting."""
    changed = 0
    max_len = max(len(before.frame_uris), len(after.frame_uris))
    for idx in range(max_len):
        old_uri = before.frame_uris[idx] if idx < len(before.frame_uris) else None
        new_uri = after.frame_uris[idx] if idx < len(after.frame_uris) else None
        if not new_uri or old_uri == new_uri:
            continue
        changed += 1
        runtime.record_event(
            after,
            stage=stage,
            action="replaced" if old_uri else "generated",
            asset_kind="frame",
            label=f"Keyframe {idx + 1}",
            from_value=old_uri,
            to_value=new_uri,
            indices=[idx],
            retry=retry,
        )
    return changed


def _record_segment_asset_events(
    runtime: SceneAgentRuntime,
    before: SceneState,
    after: SceneState,
    *,
    stage: str,
    retry: int,
) -> int:
    """Record per-segment URI replacements for reporting."""
    changed = 0
    max_len = max(len(before.segment_uris), len(after.segment_uris))
    for idx in range(max_len):
        old_uri = before.segment_uris[idx] if idx < len(before.segment_uris) else None
        new_uri = after.segment_uris[idx] if idx < len(after.segment_uris) else None
        if not new_uri or old_uri == new_uri:
            continue
        changed += 1
        if after.storyboard and idx < len(after.storyboard.segments):
            segment = after.storyboard.segments[idx]
            label = f"Segment {idx + 1}: frame {segment.start_frame_idx + 1} -> frame {segment.end_frame_idx + 1}"
        else:
            label = f"Segment {idx + 1}"
        runtime.record_event(
            after,
            stage=stage,
            action="replaced" if old_uri else "generated",
            asset_kind="segment",
            label=label,
            from_value=old_uri,
            to_value=new_uri,
            indices=[idx],
            retry=retry,
        )
    return changed


def _record_prompt_change_events(
    runtime: SceneAgentRuntime,
    before: SceneState,
    after: SceneState,
    *,
    stage: str,
    retry: int,
) -> int:
    """Record keyframe prompt updates."""
    changed = 0
    for idx, prompt_change in sorted(after.prompt_changes.items()):
        previous = before.prompt_changes.get(idx)
        if previous == prompt_change:
            continue
        changed += 1
        old_value = previous.get("updated") if previous else prompt_change.get("original")
        runtime.record_event(
            after,
            stage=stage,
            action="updated",
            asset_kind="prompt",
            label=f"Prompt for keyframe {idx + 1}",
            from_value=old_value,
            to_value=prompt_change.get("updated"),
            indices=[idx],
            retry=retry,
            details={"original": prompt_change.get("original", "")},
        )
    return changed
