
from __future__ import annotations

from typing import Any

from scene_agent.models import SceneState
from scene_agent.pipeline.operator_tools import OperatorTools

def _inner_video_tool(video_tool):
    return getattr(video_tool, "tool", video_tool)


def _supports_kling_budget(video_tool) -> bool:
    tool = _inner_video_tool(video_tool)
    return all(
        hasattr(tool, attr)
        for attr in ("estimate_generation_tokens", "estimate_edit_tokens", "config")
    )


def _budget_metadata(state: SceneState, video_tool) -> dict[str, Any]:
    tool = _inner_video_tool(video_tool)
    config = tool.config
    existing = state.provider_metadata.get("kling_budget", {}) if state.provider_metadata else {}
    return {
        "provider": "kling",
        "limit_tokens": float(getattr(config, "kling_run_token_limit", 60.0)),
        "spent_tokens": float(existing.get("spent_tokens", 0.0) or 0.0),
        "generation_tokens_per_second": float(
            getattr(config, "kling_generation_tokens_per_second", 0.6)
        ),
        "edit_tokens_per_second": float(getattr(config, "kling_edit_tokens_per_second", 0.9)),
        "paid_calls": list(existing.get("paid_calls", [])),
        "skipped_calls": list(existing.get("skipped_calls", [])),
    }


def _would_exceed_budget(budget: dict[str, Any], estimated_tokens: float) -> bool:
    return budget["spent_tokens"] + estimated_tokens > budget["limit_tokens"] + 1e-9


def _record_paid_call(
    budget: dict[str, Any],
    *,
    kind: str,
    segment_index: int,
    estimated_tokens: float,
) -> None:
    budget["spent_tokens"] = round(budget["spent_tokens"] + estimated_tokens, 4)
    budget["paid_calls"].append(
        {
            "kind": kind,
            "segment_index": segment_index,
            "estimated_tokens": estimated_tokens,
            "spent_after_tokens": budget["spent_tokens"],
        }
    )


def _record_skipped_call(
    budget: dict[str, Any],
    *,
    kind: str,
    segment_index: int,
    estimated_tokens: float,
    substitute: str,
) -> None:
    budget["skipped_calls"].append(
        {
            "kind": kind,
            "segment_index": segment_index,
            "estimated_tokens": estimated_tokens,
            "spent_before_tokens": budget["spent_tokens"],
            "limit_tokens": budget["limit_tokens"],
            "substitute": substitute,
            "reason": "kling_run_token_limit_reached",
        }
    )


def _placeholder_output_key(state: SceneState, seg_idx: int) -> str:
    run_part = state.run_id or f"state-{id(state)}"
    return f"segments/kling-budget-placeholder-{run_part}-{seg_idx}.mp4"


def _create_placeholder_segment(
    state: SceneState,
    tools: OperatorTools,
    *,
    segment_index: int,
    start_image_uri: str,
    duration_sec: float,
    fps: int,
) -> str:
    if not tools.stitch_tool or not hasattr(tools.stitch_tool, "create_still_clip"):
        raise RuntimeError("A StitchTool with create_still_clip is required for Kling budget placeholders")
    return tools.stitch_tool.create_still_clip(
        image_uri=start_image_uri,
        duration_sec=duration_sec,
        fps=fps,
        output_key=_placeholder_output_key(state, segment_index),
    )
