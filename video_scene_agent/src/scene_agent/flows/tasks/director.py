
from __future__ import annotations

from prefect import task

from scene_agent.flows.tasks.common import (
    _record_frame_asset_events,
    _record_prompt_change_events,
    _runtime,
    _sync_state,
    _task_logger,
    _task_retry_count,
)
from scene_agent.models import SceneState
from scene_agent.pipeline.director import director_storyboard, director_world
from scene_agent.pipeline.keyframes import keyframes_generate
from scene_agent.runtime import RuntimeSettings, apply_state_update

@task(name="director_world", retries=2, retry_delay_seconds=5, timeout_seconds=180, log_prints=True)
def director_world_task(settings: RuntimeSettings, state: SceneState) -> SceneState:
    logger = _task_logger(settings, "director_world")
    runtime = _runtime(settings)
    retry = _task_retry_count()
    if state.world_raw:
        logger.info("Skipping director_world because world_raw is already present")
        runtime.record_event(
            state,
            stage="director_world",
            action="skipped",
            asset_kind="world",
            label="World package",
            retry=retry,
        )
        _sync_state(runtime, state)
        return state

    logger.info("Generating world package")
    before = state.model_copy(deep=True)
    update = director_world(state, runtime.director_tools)
    apply_state_update(state, update)
    runtime.save_json_artifact("world.json", state.world_raw or {})
    runtime.record_event(
        state,
        stage="director_world",
        action="generated",
        asset_kind="world",
        label="World package",
        from_value=before.world_raw,
        to_value=state.world_raw,
        retry=retry,
        counts={
            "objects": len(state.world.objects) if state.world else 0,
        },
    )
    _sync_state(runtime, state)
    logger.info("World package generated")
    return state


@task(name="director_storyboard", retries=2, retry_delay_seconds=5, timeout_seconds=180, log_prints=True)
def director_storyboard_task(settings: RuntimeSettings, state: SceneState) -> SceneState:
    logger = _task_logger(settings, "director_storyboard")
    runtime = _runtime(settings)
    retry = _task_retry_count()
    if state.storyboard_raw:
        logger.info("Skipping director_storyboard because storyboard_raw is already present")
        runtime.record_event(
            state,
            stage="director_storyboard",
            action="skipped",
            asset_kind="storyboard",
            label="Storyboard",
            retry=retry,
        )
        _sync_state(runtime, state)
        return state

    logger.info("Generating storyboard")
    before = state.model_copy(deep=True)
    update = director_storyboard(state, runtime.director_tools)
    apply_state_update(state, update)
    runtime.save_json_artifact("storyboard.json", state.storyboard_raw or {})
    runtime.record_event(
        state,
        stage="director_storyboard",
        action="generated",
        asset_kind="storyboard",
        label="Storyboard",
        from_value=before.storyboard_raw,
        to_value=state.storyboard_raw,
        retry=retry,
        counts={
            "frames": len(state.storyboard.frames) if state.storyboard else 0,
            "segments": len(state.storyboard.segments) if state.storyboard else 0,
            "duration_sec": state.storyboard.total_duration() if state.storyboard else 0.0,
        },
    )
    _sync_state(runtime, state)
    logger.info("Storyboard generated")
    return state


@task(name="keyframes_generate", retries=2, retry_delay_seconds=10, timeout_seconds=600, log_prints=True)
def keyframes_task(settings: RuntimeSettings, state: SceneState) -> SceneState:
    logger = _task_logger(settings, "keyframes_generate")
    runtime = _runtime(settings)
    retry = _task_retry_count()
    if state.frame_uris and not state.regen_frames:
        logger.info("Skipping keyframe generation because frames already exist and no regeneration was requested")
        runtime.record_event(
            state,
            stage="keyframes_generate",
            action="skipped",
            asset_kind="stage",
            label="Keyframe generation",
            retry=retry,
        )
        _sync_state(runtime, state)
        return state

    logger.info("Generating keyframes")
    requested_regen = list(state.regen_frames)
    before = state.model_copy(deep=True)
    update = keyframes_generate(state, runtime.director_tools)
    apply_state_update(state, update)
    changed_frames = _record_frame_asset_events(
        runtime,
        before,
        state,
        stage="keyframes_generate",
        retry=retry,
    )
    changed_prompts = _record_prompt_change_events(
        runtime,
        before,
        state,
        stage="keyframes_generate",
        retry=retry,
    )
    state.regen_frames = []
    state.final_video_uri = None
    runtime.save_json_artifact("storyboard.json", state.storyboard_raw or {})
    runtime.record_event(
        state,
        stage="keyframes_generate",
        action="completed",
        asset_kind="stage",
        label="Keyframe generation",
        retry=retry,
        counts={
            "frames_changed": changed_frames,
            "prompt_changes": changed_prompts,
            "frame_count": len(state.frame_uris),
        },
        details={"requested_regen_frames": requested_regen},
    )
    _sync_state(runtime, state)
    logger.info("Keyframe generation finished with %d frames", len(state.frame_uris))
    return state
