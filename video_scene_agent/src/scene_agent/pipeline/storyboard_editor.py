
from __future__ import annotations

import logging
from copy import deepcopy

from scene_agent.models import Issue, SceneState, StoryboardData, StoryboardFixResult
from scene_agent.pipeline.storyboard_checks import _calibrate_storyboard_issue, _deterministic_storyboard_issues, _sample_indices
from scene_agent.prompts import (
    SB_FIX_SYSTEM,
    SB_REVIEW_SYSTEM,
    format_sb_fix_user,
    format_sb_review_user,
    hydrate_storyboard_payload_prompts,
    to_json,
)
from scene_agent.tools.vision_rewriter import build_neighbors_vision_messages_base64, normalize_negative_items
from scene_agent.utils.json_llm import clean_json_response, parse_partial_json

log = logging.getLogger(__name__)

class EditorTools:
    """Dependencies needed for editor nodes."""

    def __init__(self, llm, vision_rewriter=None, storage=None):
        self.llm = llm
        self.vision_rewriter = vision_rewriter
        self.storage = storage


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


def _normalized_direction(value: str | None) -> str:
    """Normalize non-direction placeholders before continuity checks."""
    text = (value or "").strip().lower().rstrip(".")
    if not text:
        return ""
    if text in {"n/a", "na", "none", "unknown", "unspecified", "not applicable"}:
        return ""
    if (
        "n/a" in text
        or "not applicable" in text
        or "unspecified" in text
        or "static" in text
        or "stationary" in text
        or "stopped" in text
        or "paused" in text
        or "backward" in text
    ):
        return ""
    return text


def _allows_direction_change(segment) -> bool:
    """Return true when the storyboard explicitly handles a direction/axis change."""
    fields = [
        segment.screen_direction_rule or "",
        segment.camera_axis_rule or "",
        " ".join(segment.entry_exit_actions or []),
        segment.offscreen_justification or "",
    ]
    text = " ".join(fields).lower()
    markers = (
        "intentional reorientation",
        "reorientation",
        "reorient",
        "re-establish",
        "reestablish",
        "new axis",
        "new screen direction",
        "establishes the monster as the new focal point",
        "establishing the monster as the new focal point",
        "pivot",
        "turn",
        "reverse",
    )
    return any(marker in text for marker in markers)


def _sample_indices(total: int, limit: int) -> list[int]:
    """Sample up to limit indices across a sequence."""
    if total <= 0 or limit <= 0:
        return []
    if total <= limit:
        return list(range(total))
    if limit == 1:
        return [0]
    return sorted({round(i * (total - 1) / (limit - 1)) for i in range(limit)})

def sb_editor_review(state: SceneState, tools: EditorTools) -> dict:
    """
    Review storyboard and identify issues.

    Args:
        state: Current graph state
        tools: Editor tools (LLM)

    Returns:
        Partial state update with issues
    """
    log.info("sb_editor_review: Reviewing storyboard")

    storyboard_data = state.storyboard_raw or {}
    if not storyboard_data:
        raise ValueError("Storyboard not available for review")

    # Check iteration limit
    if state.sb_iteration >= state.constraints.K_sb:
        raise RuntimeError(f"Storyboard review exceeded max iterations ({state.constraints.K_sb})")

    storyboard_json = to_json(storyboard_data)

    user_prompt = format_sb_review_user(storyboard_json)
    review_mode = "multimodal"
    review_error = None

    if not tools.storage or not state.frame_uris:
        raise ValueError("Storyboard multimodal review requires storage and generated frame URIs")

    try:
        from scene_agent.utils.image_data_url import encode_uri_to_data_url

        content: list[dict] = [
            {"type": "text", "text": user_prompt},
        ]
        sample_indices = _sample_indices(
            len(state.frame_uris),
            state.constraints.vision_max_images_per_request,
        )
        for idx in sample_indices:
            uri = state.frame_uris[idx]
            if not uri:
                continue
            data_url = encode_uri_to_data_url(
                storage=tools.storage,
                uri=uri,
                max_side_px=state.constraints.vision_image_max_side_px,
                jpeg_quality=state.constraints.vision_image_jpeg_quality,
                target_mime=state.constraints.vision_image_mime,
            )
            content.append({"type": "image_url", "image_url": {"url": data_url}})

        response_text = tools.llm.chat_with_history(
            messages=[
                {"role": "system", "content": SB_REVIEW_SYSTEM},
                {"role": "user", "content": content},
            ],
            json_mode=True,
        )
    except Exception as exc:
        log.error("Storyboard multimodal review failed; generated frames were not accepted: %s", exc)
        raise

    # Clean and parse JSON response
    cleaned = clean_json_response(response_text)
    issues = parse_partial_json(cleaned)

    if issues is None:
        log.error(f"Failed to parse LLM response. Raw: {response_text[:500]}")
        raise ValueError("Storyboard review did not return valid JSON")
    elif isinstance(issues, dict):
        issues = issues.get("issues", [])

    normalized_issues = _normalize_issues(issues)
    try:
        deterministic = _deterministic_storyboard_issues(StoryboardData.model_validate(storyboard_data))
    except Exception as exc:
        log.warning(f"Skipping deterministic storyboard checks due to invalid storyboard payload: {exc}")
        deterministic = []

    merged: list[Issue] = []
    seen = set()
    for raw_issue in normalized_issues + deterministic:
        issue = _calibrate_storyboard_issue(raw_issue)
        key = (issue.target, issue.severity, issue.problem)
        if key in seen:
            continue
        seen.add(key)
        merged.append(issue)

    all_issue_payloads = _issue_payloads(merged)
    blocking_issue_payloads = _issue_payloads(_blocking_issues(merged))

    log.info("Found %d storyboard issues (%d blocking)", len(all_issue_payloads), len(blocking_issue_payloads))

    return {
        "sb_issues": blocking_issue_payloads,
        "sb_iteration": state.sb_iteration + 1,
        "sb_review_mode": review_mode,
        "sb_review_error": review_error,
        "provider_metadata": {
            "sb_review": {
                "mode": review_mode,
                "error": review_error,
                "model": getattr(getattr(tools.llm, "tool", tools.llm), "default_model", ""),
                "all_issues": all_issue_payloads,
                "blocking_issues": blocking_issue_payloads,
                "total_issue_count": len(all_issue_payloads),
                "blocking_issue_count": len(blocking_issue_payloads),
            }
        },
    }


def sb_editor_fix(state: SceneState, tools: EditorTools) -> dict:
    """
    Fix storyboard based on identified issues.

    For frame issues, uses vision model with neighbor context to generate new prompts.

    Args:
        state: Current graph state
        tools: Editor tools (LLM, vision_rewriter, storage)

    Returns:
        Partial state update with fixed storyboard and regen_frames
    """
    log.info("sb_editor_fix: Fixing storyboard with vision context")

    storyboard_data = deepcopy(state.storyboard_raw or {})
    if not storyboard_data:
        raise ValueError("Storyboard not available for fixing")

    if not state.sb_issues:
        log.warning("No issues to fix")
        return {}

    issues = _normalize_issues(state.sb_issues)
    if not issues:
        log.warning("No valid issues to fix after normalization")
        return {"sb_issues": []}

    frames = storyboard_data.get("frames", [])
    frame_uris = state.frame_uris or []

    # Separate frame issues from other issues
    frame_issues: list[tuple[int, str]] = []
    non_frame_issues: list[Issue] = []

    for issue in issues:
        target = issue.target
        if target.startswith("frame:"):
            try:
                idx = int(target.split(":")[1])
                frame_issues.append((idx, issue.problem))
            except (ValueError, IndexError):
                log.warning(f"Invalid frame target: {target}")
        else:
            non_frame_issues.append(issue)

    regen_frames: set[int] = set()

    # Track prompt changes for markdown documentation
    prompt_changes = dict(state.prompt_changes or {})  # Copy existing changes

    if frame_issues and not (tools.vision_rewriter and tools.storage):
        raise ValueError("Frame-level storyboard fixes require vision_rewriter and storage")

    # Use the storyboard-fix LLM pass for non-frame issues.
    should_call_fix_llm = bool(non_frame_issues)
    if should_call_fix_llm:
        issues_json = to_json([issue.model_dump(by_alias=True) for issue in issues])
        user_prompt = format_sb_fix_user(to_json(storyboard_data), issues_json)

        response_text = tools.llm.chat(
            user=user_prompt,
            system=SB_FIX_SYSTEM,
            json_mode=True,
        )

        cleaned = clean_json_response(response_text)
        parsed = parse_partial_json(cleaned)
        if parsed is None:
            raise ValueError(f"Storyboard fix did not return valid JSON: {response_text[:200]}...")

        fix_result = StoryboardFixResult.model_validate(parsed)
        hydrated_storyboard = hydrate_storyboard_payload_prompts(fix_result.storyboard)
        storyboard_data = hydrated_storyboard.model_dump(mode="json", by_alias=True)
        frames = storyboard_data.get("frames", [])
        regen_frames.update(fix_result.regen_frames)

    # Process frame issues with vision model
    if frame_issues and tools.vision_rewriter and tools.storage:
        from scene_agent.utils.image_data_url import encode_uri_to_data_url

        for frame_idx, problem_desc in frame_issues:
            if frame_idx >= len(frames):
                log.warning(f"Frame index {frame_idx} out of range, skipping")
                continue

            frame = frames[frame_idx]
            frame["fix_description"] = problem_desc
            original_prompt = frame.get("image_prompt", "")
            original_negative = normalize_negative_items(frame.get("negative", []))

            log.info(f"Processing frame {frame_idx} with vision: {problem_desc[:50]}...")

            # Gather neighbor frames as base64
            neighbor_data_urls = {}

            # Previous frame
            if frame_idx > 0 and frame_idx - 1 < len(frame_uris):
                try:
                    prev_uri = frame_uris[frame_idx - 1]
                    data_url = encode_uri_to_data_url(
                        storage=tools.storage,
                        uri=prev_uri,
                        max_side_px=state.constraints.vision_image_max_side_px,
                        jpeg_quality=state.constraints.vision_image_jpeg_quality,
                        target_mime=state.constraints.vision_image_mime,
                    )
                    neighbor_data_urls["prev"] = data_url
                except Exception as e:
                    raise RuntimeError(f"Failed to encode prev frame {frame_idx-1}: {e}") from e

            # Current frame (if exists)
            if frame_idx < len(frame_uris) and frame_uris[frame_idx]:
                try:
                    data_url = encode_uri_to_data_url(
                        storage=tools.storage,
                        uri=frame_uris[frame_idx],
                        max_side_px=state.constraints.vision_image_max_side_px,
                        jpeg_quality=state.constraints.vision_image_jpeg_quality,
                        target_mime=state.constraints.vision_image_mime,
                    )
                    neighbor_data_urls["current"] = data_url
                except Exception as e:
                    raise RuntimeError(f"Failed to encode current frame {frame_idx}: {e}") from e

            # Next frame
            if frame_idx + 1 < len(frames) and frame_idx + 1 < len(frame_uris):
                try:
                    next_uri = frame_uris[frame_idx + 1]
                    data_url = encode_uri_to_data_url(
                        storage=tools.storage,
                        uri=next_uri,
                        max_side_px=state.constraints.vision_image_max_side_px,
                        jpeg_quality=state.constraints.vision_image_jpeg_quality,
                        target_mime=state.constraints.vision_image_mime,
                    )
                    neighbor_data_urls["next"] = data_url
                except Exception as e:
                    raise RuntimeError(f"Failed to encode next frame {frame_idx+1}: {e}") from e

            # Build vision prompt with neighbor context
            messages = build_neighbors_vision_messages_base64(
                current_prompt=original_prompt,
                current_negative=original_negative,
                neighbor_data_urls=neighbor_data_urls,
                frame_idx=frame_idx,
                total_frames=len(frames),
                fix_description=problem_desc,
            )

            try:
                result = tools.vision_rewriter.complete_json_messages(messages)
                if result:
                    new_prompt = result.get("image_prompt", original_prompt)
                    neg = normalize_negative_items(result.get("negative", original_negative))
                    final_negative = list(dict.fromkeys([*neg, *original_negative]))

                    # Track prompt change if vision rewriter modified it
                    if new_prompt != original_prompt:
                        prompt_changes[frame_idx] = {
                            "original": original_prompt,
                            "updated": new_prompt,
                        }

                    # Update frame with new prompt
                    frame["image_prompt"] = new_prompt
                    frame["negative"] = final_negative

                    log.info(f"Frame {frame_idx}: prompt regenerated via vision")
                    regen_frames.add(frame_idx)
                else:
                    raise ValueError(f"Frame {frame_idx}: no parsed JSON in vision response")

            except Exception as e:
                log.error("Frame %s vision API failed: %s", frame_idx, e)
                raise

    elif frame_issues:
        raise ValueError("Frame-level storyboard fixes require vision_rewriter and storage")

    # Clear issues after processing
    regen_frame_list = sorted(regen_frames)
    validated_storyboard = hydrate_storyboard_payload_prompts(StoryboardData.model_validate(storyboard_data))

    log.info(f"Applied fixes, regen_frames: {regen_frame_list}")

    return {
        "storyboard_raw": validated_storyboard.model_dump(mode="json", by_alias=True),
        "storyboard": validated_storyboard,
        "sb_issues": [],
        "regen_frames": regen_frame_list,
        "prompt_changes": prompt_changes,
        "provider_metadata": {
            "sb_fix": {
                "provider": "openrouter",
                "model": getattr(getattr(tools.llm, "tool", tools.llm), "default_model", ""),
                "vision_model": getattr(tools.vision_rewriter, "vision_model", ""),
            }
        },
    }
