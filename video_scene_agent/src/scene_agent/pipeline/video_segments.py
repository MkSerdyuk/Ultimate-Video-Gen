
from __future__ import annotations

import logging
from copy import deepcopy

from scene_agent.models import SceneState
from scene_agent.pipeline.operator_tools import OperatorTools
from scene_agent.pipeline.video_budget import (
    _budget_metadata,
    _create_placeholder_segment,
    _inner_video_tool,
    _record_paid_call,
    _record_skipped_call,
    _supports_kling_budget,
    _would_exceed_budget,
)
from scene_agent.tools.kling import duration_to_num_frames
from scene_agent.utils.aspect_ratio import normalize_aspect_ratio

log = logging.getLogger(__name__)

def _segment_prompt(segment: dict) -> str:
    return segment.get("video_prompt") or segment.get("kling_prompt") or segment.get("transition_text", "")


def _combined_negative(style_guide: dict, segment: dict) -> str:
    global_negative = style_guide.get("global_negative", [])
    segment_negative = segment.get("negative", [])
    if isinstance(global_negative, str):
        global_negative = [global_negative]
    if isinstance(segment_negative, str):
        segment_negative = [segment_negative]
    return ", ".join(dict.fromkeys([*global_negative, *segment_negative]))


def _segment_issues(state: SceneState, seg_idx: int) -> list[str]:
    metadata = state.provider_metadata.get("vid_fix", {}) if state.provider_metadata else {}
    issue_map = metadata.get("edit_problem_map", {})
    problems = list(issue_map.get(str(seg_idx), [])) + list(issue_map.get(seg_idx, []))
    if problems:
        return [str(problem) for problem in problems if str(problem).strip()]

    issues = []
    for issue in state.vid_issues or []:
        if issue.get("target") in {f"segment:{seg_idx}", "global"} and issue.get("problem"):
            issues.append(str(issue["problem"]))
    return issues


def _build_edit_prompt(state: SceneState, segment: dict, seg_idx: int) -> str:
    base_prompt = _segment_prompt(segment)
    issues = _segment_issues(state, seg_idx)
    issue_lines = "\n".join(f"- {issue}" for issue in issues) or "- Follow the updated storyboard segment exactly."
    return (
        f"{base_prompt}\n\n"
        "Repair strategy:\n"
        "- Generate a replacement segment using the current video as motion and camera reference, not as a direct base-video edit.\n"
        "- Use the provided start keyframe as the first-frame visual target when available.\n"
        "- The end keyframe is a continuity target described in text; do not invent a hard end-frame input for video-reference mode.\n\n"
        "Preserve what already works:\n"
        "- Keep the existing segment timing, continuity, character identity, wardrobe, props, lighting, and camera grammar unless listed below.\n"
        "- Preserve any successful motion beats and the segment's relationship to neighboring clips.\n\n"
        "Correct only these problems:\n"
        f"{issue_lines}\n\n"
        "Boundary continuity:\n"
        "- Keep the first and final moments aligned with the original segment and neighboring keyframes.\n"
        f"- End-state continuity target: {segment.get('end_match_notes') or 'land cleanly on the next keyframe.'}"
    )

def segments_generate(state: SceneState, tools: OperatorTools) -> dict:
    """
    Generate video segments using the configured image-to-video provider.

    Args:
        state: Current graph state
        tools: Operator tools (video generation, stitch)

    Returns:
        Partial state update with segment URIs
    """
    log.info("segments_generate: Generating video segments")

    storyboard_data = deepcopy(state.storyboard_raw or {})
    if not storyboard_data:
        raise ValueError("Storyboard not available")

    frames = storyboard_data.get("frames", [])
    segments = storyboard_data.get("segments", [])
    frame_uris = list(state.frame_uris or [])
    existing_segment_uris = list(state.segment_uris or [])

    if not frame_uris:
        raise ValueError("No frame URIs available")

    style_guide = storyboard_data.get("style_guide", {})
    requested_indices = set(state.regen_segments or range(len(segments)))
    segment_specs: list[tuple[int, dict]] = []
    fps = state.constraints.fps
    aspect_ratio = normalize_aspect_ratio(state.constraints.aspect_ratio)

    for seg_idx, segment in enumerate(segments):
        if seg_idx not in requested_indices:
            continue

        start_idx = segment.get("start_frame_idx", 0)
        end_idx = segment.get("end_frame_idx", 1)

        if start_idx >= len(frame_uris) or end_idx >= len(frame_uris):
            log.warning(f"Segment indices out of range, skipping")
            continue

        duration_sec = segment.get("duration", 5.0)

        spec = {
            "start_image_uri": frame_uris[start_idx],
            "end_image_uri": frame_uris[end_idx],
            "prompt": _segment_prompt(segment),
            "negative_prompt": _combined_negative(style_guide, segment),
            "duration_sec": duration_sec,
            "num_frames": duration_to_num_frames(duration_sec, fps),
            "fps": fps,
            "aspect_ratio": aspect_ratio,
        }
        segment_specs.append((seg_idx, spec))

    if len(existing_segment_uris) < len(segments):
        existing_segment_uris.extend([""] * (len(segments) - len(existing_segment_uris)))

    log.info(f"Generating {len(segment_specs)} video segments")
    video_tool = _inner_video_tool(tools.video_tool)
    generated_uris: list[str] = []
    task_ids: list[str] = []
    paid_indices: list[int] = []
    budget_skipped_indices: list[int] = []
    placeholder_indices: list[int] = []
    reused_existing_indices: list[int] = []
    media_input_urls: list[str] = []
    provider_metadata: dict[str, Any] = {}

    if _supports_kling_budget(tools.video_tool):
        budget = _budget_metadata(state, tools.video_tool)
        for seg_idx, spec in segment_specs:
            estimated_tokens = float(video_tool.estimate_generation_tokens(spec))
            if _would_exceed_budget(budget, estimated_tokens):
                if existing_segment_uris[seg_idx]:
                    uri = existing_segment_uris[seg_idx]
                    reused_existing_indices.append(seg_idx)
                    substitute = "existing_segment"
                else:
                    uri = _create_placeholder_segment(
                        state,
                        tools,
                        segment_index=seg_idx,
                        start_image_uri=spec["start_image_uri"],
                        duration_sec=float(spec.get("duration_sec", 5.0)),
                        fps=fps,
                    )
                    existing_segment_uris[seg_idx] = uri
                    placeholder_indices.append(seg_idx)
                    substitute = "still_placeholder"
                segments[seg_idx]["result_uri"] = uri
                budget_skipped_indices.append(seg_idx)
                _record_skipped_call(
                    budget,
                    kind="generate",
                    segment_index=seg_idx,
                    estimated_tokens=estimated_tokens,
                    substitute=substitute,
                )
                continue

            segment_uris = tools.video_tool.generate_multiple_segments([spec])
            if not segment_uris:
                raise ValueError(f"Video provider returned no URI for segment {seg_idx}")
            uri = segment_uris[0]
            generated_uris.append(uri)
            paid_indices.append(seg_idx)
            segments[seg_idx]["result_uri"] = uri
            existing_segment_uris[seg_idx] = uri
            _record_paid_call(
                budget,
                kind="generate",
                segment_index=seg_idx,
                estimated_tokens=estimated_tokens,
            )
            task_ids.extend(list(getattr(video_tool, "last_generation_task_ids", [])))
            media_input_urls.extend(list(getattr(video_tool, "last_media_input_urls", [])))
        provider_metadata["kling_budget"] = budget
    else:
        generated_uris = tools.video_tool.generate_multiple_segments([spec for _, spec in segment_specs])
        task_ids = list(getattr(video_tool, "last_generation_task_ids", []))
        media_input_urls = list(getattr(video_tool, "last_media_input_urls", []))
        paid_indices = [seg_idx for seg_idx, _ in segment_specs]
        for (seg_idx, _), uri in zip(segment_specs, generated_uris):
            segments[seg_idx]["result_uri"] = uri
            existing_segment_uris[seg_idx] = uri

    storyboard_data["segments"] = segments

    log.info(f"Generated {len(generated_uris)} video segments")
    provider_metadata["segments_generate"] = {
        "provider": "kling",
        "model": getattr(video_tool, "model_name", ""),
        "mode": getattr(video_tool, "mode", ""),
        "sound": getattr(video_tool, "sound", ""),
        "task_ids": task_ids,
        "regenerated_indices": sorted(requested_indices),
        "paid_indices": paid_indices,
        "budget_skipped_indices": budget_skipped_indices,
        "placeholder_indices": placeholder_indices,
        "reused_existing_indices": reused_existing_indices,
        "media_input_urls": list(dict.fromkeys(media_input_urls)),
    }

    return {
        "segment_uris": existing_segment_uris,
        "storyboard_raw": storyboard_data,
        "regen_segments": [],
        "provider_metadata": provider_metadata,
    }


def segments_edit(state: SceneState, tools: OperatorTools) -> dict:
    """
    Repair existing video segments using the configured video-edit provider.

    Segments without an existing source video are automatically returned as
    `regen_segments` so the flow can rerender them from keyframe anchors.
    """
    log.info("segments_edit: Repairing video segments")

    storyboard_data = deepcopy(state.storyboard_raw or {})
    if not storyboard_data:
        raise ValueError("Storyboard not available")

    frames = storyboard_data.get("frames", [])
    segments = storyboard_data.get("segments", [])
    frame_uris = list(state.frame_uris or [])
    existing_segment_uris = list(state.segment_uris or [])
    style_guide = storyboard_data.get("style_guide", {})
    fps = state.constraints.fps
    aspect_ratio = normalize_aspect_ratio(state.constraints.aspect_ratio)

    requested_indices = set(state.edit_segments or [])
    if not requested_indices:
        return {"edit_segments": []}

    edit_specs: list[tuple[int, dict]] = []
    missing_source_regen: set[int] = set()

    for seg_idx in sorted(requested_indices):
        if seg_idx >= len(segments) or seg_idx < 0:
            log.warning("Edit segment index out of range, skipping: %s", seg_idx)
            continue
        if seg_idx >= len(existing_segment_uris) or not existing_segment_uris[seg_idx]:
            missing_source_regen.add(seg_idx)
            continue

        segment = segments[seg_idx]
        start_idx = segment.get("start_frame_idx", 0)
        end_idx = segment.get("end_frame_idx", 1)
        duration_sec = segment.get("duration", 5.0)

        spec = {
            "segment_uri": existing_segment_uris[seg_idx],
            "prompt": _build_edit_prompt(state, segment, seg_idx),
            "negative_prompt": _combined_negative(style_guide, segment),
            "duration_sec": duration_sec,
            "num_frames": duration_to_num_frames(duration_sec, fps),
            "fps": fps,
            "aspect_ratio": aspect_ratio,
            "match_video_length": True,
            "match_input_fps": True,
            "video_size": "auto",
            "generate_audio": False,
            "use_multiscale": True,
        }
        if start_idx < len(frame_uris):
            spec["start_image_uri"] = frame_uris[start_idx]
        if end_idx < len(frame_uris):
            spec["end_image_uri"] = frame_uris[end_idx]
        edit_specs.append((seg_idx, spec))

    if len(existing_segment_uris) < len(segments):
        existing_segment_uris.extend([""] * (len(segments) - len(existing_segment_uris)))

    video_tool = _inner_video_tool(tools.video_tool)
    edited_uris: list[str] = []
    task_ids: list[str] = []
    edited_indices: list[int] = []
    budget_skipped_indices: list[int] = []
    media_input_urls: list[str] = []
    provider_metadata: dict[str, Any] = {}

    if _supports_kling_budget(tools.video_tool):
        budget = _budget_metadata(state, tools.video_tool)
        for seg_idx, spec in edit_specs:
            estimated_tokens = float(video_tool.estimate_edit_tokens(spec))
            if _would_exceed_budget(budget, estimated_tokens):
                budget_skipped_indices.append(seg_idx)
                _record_skipped_call(
                    budget,
                    kind="edit",
                    segment_index=seg_idx,
                    estimated_tokens=estimated_tokens,
                    substitute="existing_segment",
                )
                continue

            segment_uris = tools.video_tool.edit_multiple_segments([spec])
            if not segment_uris:
                raise ValueError(f"Video edit provider returned no URI for segment {seg_idx}")
            uri = segment_uris[0]
            edited_uris.append(uri)
            edited_indices.append(seg_idx)
            segments[seg_idx]["result_uri"] = uri
            existing_segment_uris[seg_idx] = uri
            _record_paid_call(
                budget,
                kind="edit",
                segment_index=seg_idx,
                estimated_tokens=estimated_tokens,
            )
            task_ids.extend(list(getattr(video_tool, "last_edit_task_ids", [])))
            media_input_urls.extend(list(getattr(video_tool, "last_media_input_urls", [])))
        provider_metadata["kling_budget"] = budget
    else:
        edited_uris = tools.video_tool.edit_multiple_segments([spec for _, spec in edit_specs]) if edit_specs else []
        task_ids = list(getattr(video_tool, "last_edit_task_ids", []))
        media_input_urls = list(getattr(video_tool, "last_media_input_urls", []))
        edited_indices = [idx for idx, _ in edit_specs]
        for (seg_idx, _), uri in zip(edit_specs, edited_uris):
            segments[seg_idx]["result_uri"] = uri
            existing_segment_uris[seg_idx] = uri

    storyboard_data["segments"] = segments
    regen_segments = sorted(set(state.regen_segments or []) | missing_source_regen)

    log.info(
        "Edited %d video segments; missing-source regen segments=%s",
        len(edited_uris),
        regen_segments,
    )
    provider_metadata["segments_edit"] = {
        "provider": "kling",
        "model": getattr(video_tool, "model_name", ""),
        "mode": getattr(video_tool, "mode", ""),
        "sound": getattr(video_tool, "sound", ""),
        "task_ids": task_ids,
        "edited_indices": edited_indices,
        "budget_skipped_indices": budget_skipped_indices,
        "missing_source_regen_indices": sorted(missing_source_regen),
        "media_input_urls": list(dict.fromkeys(media_input_urls)),
    }

    return {
        "segment_uris": existing_segment_uris,
        "storyboard_raw": storyboard_data,
        "edit_segments": [],
        "regen_segments": regen_segments,
        "provider_metadata": provider_metadata,
    }


def stitch_video(state: SceneState, tools: OperatorTools) -> dict:
    """
    Stitch video segments into final video.

    Args:
        state: Current graph state
        tools: Operator tools

    Returns:
        Partial state update with final video URI
    """
    log.info("stitch_video: Stitching segments into final video")

    fps = state.constraints.fps
    segment_uris = list(state.segment_uris or [])
    storyboard_data = deepcopy(state.storyboard_raw or {})
    segments = storyboard_data.get("segments", []) if storyboard_data else []
    frame_uris = list(state.frame_uris or [])
    stitch_placeholder_indices: list[int] = []

    if not segment_uris and not (segments and frame_uris):
        raise ValueError("No segment URIs available for stitching")

    if segments and len(segment_uris) < len(segments):
        segment_uris.extend([""] * (len(segments) - len(segment_uris)))

    for seg_idx, segment in enumerate(segments):
        if seg_idx < len(segment_uris) and segment_uris[seg_idx]:
            continue
        start_idx = segment.get("start_frame_idx", 0)
        if start_idx >= len(frame_uris):
            log.warning("Cannot create missing segment placeholder; frame index out of range: %s", seg_idx)
            continue
        uri = _create_placeholder_segment(
            state,
            tools,
            segment_index=seg_idx,
            start_image_uri=frame_uris[start_idx],
            duration_sec=float(segment.get("duration", 5.0)),
            fps=fps,
        )
        segment_uris[seg_idx] = uri
        segment["result_uri"] = uri
        stitch_placeholder_indices.append(seg_idx)

    stitchable_segment_uris = [uri for uri in segment_uris if uri]
    if not stitchable_segment_uris:
        raise ValueError("No usable segment URIs available for stitching")

    output_key = f"final_video_{state.run_id}.mp4" if state.run_id else f"final_video_{id(state)}.mp4"

    final_uri = tools.stitch_tool.stitch(
        segment_uris=stitchable_segment_uris,
        fps=fps,
        output_key=output_key,
    )

    log.info(f"Stitched final video: {final_uri}")
    metadata: dict[str, Any] = {
        "stitch_video": {
            "provider": "ffmpeg",
            "fps": fps,
            "segment_count": len(stitchable_segment_uris),
            "placeholder_indices": stitch_placeholder_indices,
        }
    }

    update: dict[str, Any] = {
        "final_video_uri": final_uri,
        "segment_uris": segment_uris,
        "provider_metadata": metadata,
    }
    if stitch_placeholder_indices:
        storyboard_data["segments"] = segments
        update["storyboard_raw"] = storyboard_data
    return update
