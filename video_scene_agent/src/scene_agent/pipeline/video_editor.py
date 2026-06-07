
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse

from scene_agent.models import Issue, SceneState, StoryboardData, VideoFixResult
from scene_agent.prompts import (
    VID_FIX_SYSTEM,
    VID_REVIEW_SYSTEM,
    format_vid_fix_user,
    format_vid_review_user,
    hydrate_storyboard_payload_prompts,
    to_json,
)
from scene_agent.utils.json_llm import clean_json_response, parse_partial_json

log = logging.getLogger(__name__)

class VideoEditorTools:
    """Dependencies needed for video editor nodes."""

    def __init__(self, llm, video_review_tool):
        self.llm = llm
        self.video_review_tool = video_review_tool


def _normalize_issues(raw_issues: list) -> list[Issue]:
    """Validate and normalize raw issue payloads."""
    issues: list[Issue] = []
    for raw in raw_issues or []:
        try:
            issues.append(Issue.model_validate(raw))
        except Exception as exc:
            log.warning(f"Skipping invalid issue payload {raw!r}: {exc}")
    return issues


def _blocking_issues(issues: list[Issue]) -> list[Issue]:
    """Return only issues that should stop the flow for repair."""
    return [issue for issue in issues if issue.severity == "error"]


def _issue_payloads(issues: list[Issue]) -> list[dict]:
    """Serialize issues with stable aliases."""
    return [issue.model_dump(by_alias=True) for issue in issues]


def _parse_fps(rate: str | None) -> float | None:
    if not rate:
        return None
    if "/" in rate:
        numerator, denominator = rate.split("/", 1)
        try:
            denominator_value = float(denominator)
            if denominator_value == 0:
                return None
            return float(numerator) / denominator_value
        except ValueError:
            return None
    try:
        return float(rate)
    except ValueError:
        return None


def _local_video_metadata(video_uri: str) -> dict:
    """Read trusted local video metadata when the rendered MP4 is available on disk."""
    parsed = urlparse(video_uri)
    if parsed.scheme and parsed.scheme != "file":
        return {}

    path = Path(unquote(parsed.path if parsed.scheme == "file" else video_uri))
    if not path.exists():
        return {}

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,codec_name",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return {}
        payload = json.loads(result.stdout)
    except Exception as exc:
        log.warning("Failed to read local video metadata: %s", exc)
        return {}

    stream = (payload.get("streams") or [{}])[0]
    width = stream.get("width")
    height = stream.get("height")
    metadata = {
        "width": width,
        "height": height,
        "fps": _parse_fps(stream.get("avg_frame_rate") or stream.get("r_frame_rate")),
        "duration": float(payload.get("format", {}).get("duration", 0) or 0),
        "codec": stream.get("codec_name"),
    }
    if width and height:
        metadata["aspect_ratio"] = f"{width}:{height}"
    return {key: value for key, value in metadata.items() if value not in (None, "", 0)}


def _aspect_ratio_matches(metadata: dict, expected: str) -> bool:
    if ":" not in expected:
        return False
    width = metadata.get("width")
    height = metadata.get("height")
    if not width or not height:
        return False
    try:
        expected_w, expected_h = expected.split(":", 1)
        expected_ratio = float(expected_w) / float(expected_h)
        actual_ratio = float(width) / float(height)
    except (TypeError, ValueError, ZeroDivisionError):
        return False
    return abs(actual_ratio - expected_ratio) <= 0.03


def _calibrate_video_issue(issue: Issue, metadata: dict, state: SceneState) -> Issue:
    """Downgrade non-blocking polish issues so repair loops focus on story failures."""
    if issue.severity != "error":
        return issue

    problem = issue.problem.lower()
    if "aspect ratio" in problem and _aspect_ratio_matches(metadata, state.constraints.aspect_ratio):
        return issue.model_copy(update={"severity": "warning"})

    story_breaking_markers = (
        "does not move",
        "fails to land",
        "failure to land",
        "wrong subject",
        "missing",
        "absent",
        "missing subject",
        "not visible",
        "disappears",
        "identity drift",
        "premature reveal",
        "hidden-object",
        "black frame",
        "blank",
        "text appears",
    )
    if any(marker in problem for marker in story_breaking_markers):
        return issue

    polish_markers = (
        "slight",
        "slightly",
        "perfectly",
        "not perfectly",
        "centered",
        "off-center",
        "lighting",
        "shadow",
        "eye-level",
        "camera angle",
        "background",
        "seamless",
        "horizon",
        "rectangular prism",
        "shape",
        "polish",
    )
    if any(marker in problem for marker in polish_markers):
        return issue.model_copy(update={"severity": "warning"})

    return issue


def _segment_issue_map(issues: list[Issue], target_segments: list[int]) -> dict[str, list[str]]:
    """Build segment-indexed issue text for video-to-video repair prompts."""
    issue_map: dict[str, list[str]] = {str(idx): [] for idx in target_segments}
    global_problems = [issue.problem for issue in issues if issue.target == "global"]
    for issue in issues:
        if not issue.target.startswith("segment:"):
            continue
        try:
            seg_idx = int(issue.target.split(":", 1)[1])
        except ValueError:
            continue
        issue_map.setdefault(str(seg_idx), []).append(issue.problem)

    for idx in target_segments:
        issue_map.setdefault(str(idx), [])
        issue_map[str(idx)].extend(global_problems)

    return {idx: list(dict.fromkeys(problems)) for idx, problems in issue_map.items() if problems}


def vid_editor_review(state: SceneState, tools: VideoEditorTools) -> dict:
    """
    Review final video and identify issues.

    Args:
        state: Current graph state
        tools: Video editor tools

    Returns:
        Partial state update with issues
    """
    log.info("vid_editor_review: Reviewing final video")

    if state.final_video_uri is None:
        raise ValueError("No video available for review")

    storyboard_data = state.storyboard_raw or {}
    if not storyboard_data:
        raise ValueError("Storyboard not available for review context")

    # Check iteration limit
    if state.vid_iteration >= state.constraints.K_vid:
        raise RuntimeError(f"Video review exceeded max iterations ({state.constraints.K_vid})")

    storyboard_json = to_json(storyboard_data)
    video_metadata = _local_video_metadata(state.final_video_uri)
    technical_context = json.dumps(video_metadata, sort_keys=True) if video_metadata else ""

    # Format prompt
    user_prompt = format_vid_review_user(state.final_video_uri, storyboard_json, technical_context)
    review_mode = "multimodal"

    # Review video using multimodal LLM. If the provider cannot inspect the
    # actual rendered MP4, the run must fail.
    try:
        review_text = tools.video_review_tool.review(
            system=VID_REVIEW_SYSTEM,
            user_text=user_prompt,
            video_uri=state.final_video_uri,
        )

        # Clean and parse JSON response
        cleaned = clean_json_response(review_text)
        issues = parse_partial_json(cleaned)

        if issues is None:
            log.warning(f"Failed to parse video review. Raw: {review_text[:200]}")
            raise ValueError("Video review did not return valid JSON")
        elif not isinstance(issues, list):
            # If the response isn't a list, try to extract from a wrapper object
            if isinstance(issues, dict):
                issues = issues.get("issues", [])

    except Exception as e:
        log.error("Video review failed; the rendered MP4 was not accepted by the provider: %s", e)
        raise

    all_issues = [
        _calibrate_video_issue(issue, video_metadata, state)
        for issue in _normalize_issues(issues)
    ]
    all_issue_payloads = _issue_payloads(all_issues)
    blocking_issue_payloads = _issue_payloads(_blocking_issues(all_issues))

    log.info("Found %d video issues (%d blocking)", len(all_issue_payloads), len(blocking_issue_payloads))

    return {
        "vid_issues": blocking_issue_payloads,
        "vid_iteration": state.vid_iteration + 1,
        "vid_review_mode": review_mode,
        "vid_review_error": None,
        "provider_metadata": {
            "vid_review": {
                "mode": review_mode,
                "model": getattr(getattr(tools.llm, "tool", tools.llm), "default_model", ""),
                "video_model": getattr(getattr(tools.video_review_tool, "tool", tools.video_review_tool), "model", ""),
                "all_issues": all_issue_payloads,
                "blocking_issues": blocking_issue_payloads,
                "total_issue_count": len(all_issue_payloads),
                "blocking_issue_count": len(blocking_issue_payloads),
                "video_metadata": video_metadata,
            }
        },
    }


def vid_editor_fix(state: SceneState, tools: VideoEditorTools) -> dict:
    """
    Fix video based on identified issues.

    Args:
        state: Current graph state
        tools: Video editor tools

    Returns:
        Partial state update with fix instructions
    """
    log.info("vid_editor_fix: Determining fix strategy")

    storyboard_data = state.storyboard_raw or {}
    if not storyboard_data:
        raise ValueError("Storyboard not available for fixing")

    if not state.vid_issues:
        log.warning("No issues to fix")
        return {}

    storyboard_json = to_json(storyboard_data)
    normalized_issues = _normalize_issues(state.vid_issues)
    issues_json = to_json([issue.model_dump(by_alias=True) for issue in normalized_issues])

    # Format prompt
    user_prompt = format_vid_fix_user(storyboard_json, issues_json)

    # Call LLM
    response_text = tools.llm.chat(
        user=user_prompt,
        system=VID_FIX_SYSTEM,
        json_mode=True,
    )

    # Clean and parse JSON response
    cleaned = clean_json_response(response_text)
    fix_result = parse_partial_json(cleaned)

    if fix_result is None:
        log.error(f"Failed to parse LLM response. Raw: {response_text[:500]}")
        raise ValueError(f"LLM did not return valid JSON. Response: {response_text[:200]}...")

    validated_result = VideoFixResult.model_validate(fix_result)
    updated_storyboard = validated_result.storyboard or StoryboardData.model_validate(storyboard_data)
    updated_storyboard = hydrate_storyboard_payload_prompts(updated_storyboard)
    edit_segments = sorted(set(validated_result.edit_segments))
    regen_segments = list(validated_result.regen_segments)
    regen_frames = list(validated_result.regen_frames)

    if validated_result.regen_all:
        frame_count = len(updated_storyboard.frames)
        segment_count = len(updated_storyboard.segments)
        regen_frames = list(range(frame_count))
        regen_segments = list(range(segment_count))
        edit_segments = []
    else:
        regen_segments = sorted(set(regen_segments))
        edit_segments = sorted(set(edit_segments) - set(regen_segments))
        missing_source_segments = [
            seg_idx
            for seg_idx in edit_segments
            if seg_idx >= len(state.segment_uris) or not state.segment_uris[seg_idx]
        ]
        if missing_source_segments:
            log.info(
                "Video edit requested for segments without source videos; rerendering instead: %s",
                missing_source_segments,
            )
            regen_segments = sorted(set(regen_segments) | set(missing_source_segments))
            edit_segments = [idx for idx in edit_segments if idx not in missing_source_segments]

    edit_problem_map = _segment_issue_map(normalized_issues, edit_segments)

    log.info(
        "Fix strategy: edit_segments=%s, regen_segments=%s, regen_frames=%s",
        edit_segments,
        regen_segments,
        regen_frames,
    )

    return {
        "storyboard_raw": updated_storyboard.model_dump(mode="json", by_alias=True),
        "storyboard": updated_storyboard,
        "vid_issues": [],
        "edit_segments": edit_segments,
        "regen_segments": regen_segments,
        "regen_frames": regen_frames,
        "provider_metadata": {
            "vid_fix": {
                "provider": "openrouter",
                "model": getattr(getattr(tools.llm, "tool", tools.llm), "default_model", ""),
                "edit_problem_map": edit_problem_map,
            }
        },
    }
