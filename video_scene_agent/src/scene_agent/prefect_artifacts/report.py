
from __future__ import annotations

from collections import Counter, defaultdict

from scene_agent.models import RunEvent, SceneState
from scene_agent.prefect_artifacts.keys import _basename
from scene_agent.prefect_artifacts.tables import _format_indices

def render_run_report(state: SceneState) -> str:
    """Render a deterministic operational report from structured events and state."""
    requested_duration = state.constraints.target_duration_sec or state.constraints.duration_sec
    storyboard_duration = state.storyboard.total_duration() if state.storyboard else 0.0
    frame_count = len(state.frame_uris)
    segment_count = len(state.segment_uris)

    lines = [
        "# Run Report",
        "",
        "## Request",
        "",
        f"- **Status:** `{state.status}`",
        f"- **Run ID:** `{state.run_id or ''}`",
        f"- **Requested duration:** `{requested_duration:.1f}s`",
        f"- **Storyboard duration:** `{storyboard_duration:.1f}s`",
        f"- **Requested keyframes:** `{state.constraints.num_keyframes or 'auto'}`",
        f"- **Produced keyframes:** `{frame_count}`",
        f"- **Produced segments:** `{segment_count}`",
        f"- **Storyboard review iterations:** `{state.sb_iteration}`",
        f"- **Video review iterations:** `{state.vid_iteration}`",
        "",
        "## Brief",
        "",
        state.user_brief.strip(),
        "",
    ]

    if state.final_video_uri:
        lines.extend(["## Final Output", "", f"- Final video URI: `{state.final_video_uri}`", ""])

    prompt_events = [event for event in state.events if event.asset_kind == "prompt"]
    if prompt_events:
        lines.extend(["## Prompt Changes", ""])
        for event in prompt_events:
            lines.append(
                f"- {event.label}: `{str(event.from_value)[:120]}` -> `{str(event.to_value)[:120]}`"
            )
        lines.append("")

    frame_events = [event for event in state.events if event.asset_kind == "frame"]
    if frame_events:
        lines.extend(["## Frame Replacements", ""])
        for line in _replacement_lines(frame_events):
            lines.append(f"- {line}")
        lines.append("")

    segment_events = [event for event in state.events if event.asset_kind == "segment"]
    if segment_events:
        lines.extend(["## Segment Replacements", ""])
        for line in _replacement_lines(segment_events):
            lines.append(f"- {line}")
        lines.append("")

    review_events = [event for event in state.events if event.asset_kind in {"review", "fix"}]
    if review_events:
        lines.extend(["## Review Loop Summary", ""])
        for event in review_events:
            bits = [
                event.label,
                f"action={event.action}",
            ]
            if "iteration" in event.counts:
                bits.append(f"iteration={event.counts['iteration']}")
            if "issues" in event.counts:
                bits.append(f"issues={event.counts['issues']}")
            if event.mode:
                bits.append(f"mode={event.mode}")
            regen_frames = _format_indices(event.details.get("regen_frames", []))
            if regen_frames:
                bits.append(f"regen_frames={regen_frames}")
            regen_segments = _format_indices(event.details.get("regen_segments", []))
            if regen_segments:
                bits.append(f"regen_segments={regen_segments}")
            edit_segments = _format_indices(event.details.get("edit_segments", []))
            if edit_segments:
                bits.append(f"edit_segments={edit_segments}")
            if event.error:
                bits.append(f"error={event.error}")
            lines.append(f"- {', '.join(bits)}")
        lines.append("")

    retry_lines = _retry_lines(state.events)
    if retry_lines:
        lines.extend(["## Retries", ""])
        for line in retry_lines:
            lines.append(f"- {line}")
        lines.append("")

    degraded = [
        event for event in state.events
        if event.asset_kind == "review" and event.mode and event.mode != "multimodal"
    ]
    if degraded:
        lines.extend(["## Degraded Review Modes", ""])
        for event in degraded:
            lines.append(
                f"- {event.label}: mode=`{event.mode}`"
                + (f", error=`{event.error}`" if event.error else "")
            )
        lines.append("")

    if state.error:
        lines.extend(["## Error", "", f"- `{state.error_code or 'Error'}`: {state.error}", ""])

    return "\n".join(lines).strip() + "\n"

def _replacement_lines(events: list[RunEvent]) -> list[str]:
    counts: Counter[int] = Counter()
    latest: dict[int, RunEvent] = {}
    for event in events:
        if not event.indices:
            continue
        idx = event.indices[0]
        counts[idx] += 1
        latest[idx] = event

    lines: list[str] = []
    for idx in sorted(latest):
        event = latest[idx]
        old_name = _basename(event.from_value) or "none"
        new_name = _basename(event.to_value) or "none"
        lines.append(f"{event.label}: `{old_name}` -> `{new_name}` ({counts[idx]} updates)")
    return lines


def _retry_lines(events: list[RunEvent]) -> list[str]:
    retries: defaultdict[str, int] = defaultdict(int)
    seen: set[tuple[str, str, int, str]] = set()

    for event in events:
        if event.retry <= 0:
            continue
        signature = (
            event.stage,
            str(event.counts.get("iteration", "")),
            event.retry,
            event.action,
        )
        if signature in seen:
            continue
        seen.add(signature)
        retries[event.stage] = max(retries[event.stage], event.retry)

    return [f"{stage}: {count} retries" for stage, count in sorted(retries.items())]
