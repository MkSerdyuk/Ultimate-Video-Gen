from __future__ import annotations

from typing import Any

from scene_agent.models import SceneState


def route_after_sb_review(state: SceneState) -> str:
    return "sb_editor_fix" if state.sb_issues else "segments_generate"


def route_after_sb_fix(state: SceneState) -> str:
    return "keyframes_generate" if state.regen_frames else "sb_editor_review"


def route_after_vid_review(state: SceneState) -> str:
    return "vid_editor_fix" if state.vid_issues else "__end__"


def route_after_vid_fix(state: SceneState) -> str:
    if state.regen_frames:
        return "keyframes_generate"
    if state.edit_segments:
        return "segments_edit"
    if state.regen_segments:
        return "segments_generate"
    return "__end__"


def segments_touching_frames(storyboard: Any, frame_indices: list[int]) -> list[int]:
    changed_frames = {int(idx) for idx in frame_indices}
    if not changed_frames or not storyboard:
        return []

    segments = storyboard.get("segments", []) if isinstance(storyboard, dict) else getattr(storyboard, "segments", [])
    affected: set[int] = set()
    for default_idx, segment in enumerate(segments or []):
        if isinstance(segment, dict):
            seg_idx = int(segment.get("idx", segment.get("index", default_idx)))
            start_idx = int(segment.get("start_frame_idx", -1))
            end_idx = int(segment.get("end_frame_idx", -1))
        else:
            seg_idx = int(getattr(segment, "idx", default_idx))
            start_idx = int(getattr(segment, "start_frame_idx", -1))
            end_idx = int(getattr(segment, "end_frame_idx", -1))
        if start_idx in changed_frames or end_idx in changed_frames:
            affected.add(seg_idx)
    return sorted(affected)
