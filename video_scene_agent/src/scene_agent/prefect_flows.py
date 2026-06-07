from __future__ import annotations
"""Prefect flow for video scene generation."""

from typing import Any
from uuid import uuid4

from prefect import flow, runtime as prefect_runtime
from prefect.states import Failed

from scene_agent.config import Config
from scene_agent.flows.routing import segments_touching_frames
from scene_agent.flows.tasks import (
    director_storyboard_task,
    director_world_task,
    keyframes_task,
    segments_edit_task,
    segments_task,
    stitch_task,
    storyboard_fix_task,
    storyboard_review_task,
    video_fix_task,
    video_review_task,
)
from scene_agent.models import Constraints, SceneRunResult
from scene_agent.prefect_logging import configure_prefect_logging, get_prefect_logger
from scene_agent.runtime import RuntimeSettings, SceneAgentRuntime


def _coerce_constraints(constraints: Constraints | dict[str, Any] | None) -> Constraints:
    """Normalize constraint input into a Constraints model."""
    if constraints is None:
        return Constraints()
    if isinstance(constraints, Constraints):
        return constraints
    return Constraints(**constraints)


def _resolve_run_id(run_options: dict[str, Any] | None) -> str:
    """Resolve a stable run identifier for artifact scoping."""
    if run_options and run_options.get("run_id"):
        return str(run_options["run_id"])
    try:
        flow_run_id = prefect_runtime.flow_run.id
    except Exception:
        flow_run_id = None
    return str(flow_run_id or uuid4())


def _coerce_forced_edit_segments(run_options: dict[str, Any] | None) -> list[int]:
    """Normalize the smoke-only forced edit segment option."""
    if not run_options or "force_edit_segments" not in run_options:
        return []

    raw = run_options.get("force_edit_segments")
    if raw is None or raw is False:
        return []
    if raw is True:
        raw_values = [0]
    elif isinstance(raw, int):
        raw_values = [raw]
    elif isinstance(raw, str):
        raw_values = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        raw_values = list(raw)

    indices: set[int] = set()
    for value in raw_values:
        idx = int(value)
        if idx < 0:
            raise ValueError("force_edit_segments must contain non-negative segment indices")
        indices.add(idx)
    return sorted(indices)


@flow(name="generate_scene_flow", log_prints=True)
def generate_scene_flow(
    user_brief: str,
    constraints: Constraints | dict[str, Any] | None = None,
    config: Config | None = None,
    run_options: dict[str, Any] | None = None,
) -> SceneRunResult:
    """Generate a scene end-to-end using Prefect tasks."""
    configure_prefect_logging()
    config = config or Config.from_env()
    constraints_obj = _coerce_constraints(constraints)
    run_id = _resolve_run_id(run_options)
    forced_edit_segments = _coerce_forced_edit_segments(run_options)
    logger = get_prefect_logger(component="flow", scene_run_id=run_id)
    settings = RuntimeSettings(
        config=config,
        run_id=run_id,
        user_brief=user_brief,
        constraints=constraints_obj,
    )
    runtime = SceneAgentRuntime(settings)
    state = runtime.initial_state()
    state.status = "running"
    runtime.record_event(
        state,
        stage="flow",
        action="started",
        asset_kind="run",
        label="Scene generation run",
    )
    runtime.persist_state(state)
    runtime.publish_prefect_artifacts(state)
    logger.info("Starting scene generation flow")

    try:
        state = director_world_task(settings, state)
        state = director_storyboard_task(settings, state)
        state = keyframes_task(settings, state)

        while True:
            state = storyboard_review_task(settings, state)
            if not state.sb_issues:
                break
            if state.sb_iteration >= state.constraints.K_sb:
                raise RuntimeError(
                    f"Storyboard review found {len(state.sb_issues)} blocking issues "
                    f"at max iterations ({state.constraints.K_sb})"
                )
            state = storyboard_fix_task(settings, state)
            if state.regen_frames:
                state = keyframes_task(settings, state)

        state = segments_task(settings, state)
        state = stitch_task(settings, state)

        if forced_edit_segments:
            logger.info("Forcing segment repair smoke path for segments: %s", forced_edit_segments)
            state.edit_segments = list(forced_edit_segments)
            state = segments_edit_task(settings, state)
            if state.regen_segments:
                state = segments_task(settings, state)
            state = stitch_task(settings, state)

        while True:
            state = video_review_task(settings, state)
            if not state.vid_issues:
                break
            if state.vid_iteration >= state.constraints.K_vid:
                raise RuntimeError(
                    f"Video review found {len(state.vid_issues)} blocking issues "
                    f"at max iterations ({state.constraints.K_vid})"
                )

            state = video_fix_task(settings, state)

            if state.regen_frames:
                requested_regen_frames = list(state.regen_frames)
                explicit_regen_segments = set(state.regen_segments or [])
                state = keyframes_task(settings, state)
                affected_segments = segments_touching_frames(
                    state.storyboard_raw or state.storyboard,
                    requested_regen_frames,
                )
                state.regen_segments = sorted(explicit_regen_segments | set(affected_segments))
                state.edit_segments = []
                state = segments_task(settings, state)
                state = stitch_task(settings, state)
                continue

            if state.edit_segments:
                state = segments_edit_task(settings, state)
                if state.regen_segments:
                    state = segments_task(settings, state)
                state = stitch_task(settings, state)
                continue

            if state.regen_segments:
                state = segments_task(settings, state)
                state = stitch_task(settings, state)
                continue

            break

        state.status = "completed"
        runtime.record_event(
            state,
            stage="flow",
            action="completed",
            asset_kind="run",
            label="Scene generation run",
            counts={
                "frames": len(state.frame_uris),
                "segments": len(state.segment_uris),
                "storyboard_reviews": state.sb_iteration,
                "video_reviews": state.vid_iteration,
            },
        )
        runtime.persist_state(state)
        runtime.publish_prefect_artifacts(state, final=True)
        logger.info("Scene generation flow completed successfully")
        return runtime.build_result(state)

    except Exception as exc:
        state.status = "failed"
        state.error = str(exc)
        state.error_code = type(exc).__name__
        runtime.record_event(
            state,
            stage="flow",
            action="failed",
            asset_kind="run",
            label="Scene generation run",
            error=str(exc),
            counts={
                "frames": len(state.frame_uris),
                "segments": len(state.segment_uris),
                "storyboard_reviews": state.sb_iteration,
                "video_reviews": state.vid_iteration,
            },
        )
        runtime.persist_state(state)
        runtime.publish_prefect_artifacts(state, final=True)
        logger.exception("Scene generation flow failed")
        return Failed(message=f"Scene generation flow failed: {exc}")
