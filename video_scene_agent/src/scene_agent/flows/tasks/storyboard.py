
from __future__ import annotations

from prefect import task

from scene_agent.flows.tasks.common import (
    _record_prompt_change_events,
    _record_review,
    _runtime,
    _sync_state,
    _task_logger,
    _task_retry_count,
)
from scene_agent.models import SceneState
from scene_agent.pipeline.storyboard_editor import sb_editor_fix, sb_editor_review
from scene_agent.runtime import RuntimeSettings, apply_state_update

@task(name="storyboard_review", retries=1, retry_delay_seconds=2, timeout_seconds=180, log_prints=True)
def storyboard_review_task(settings: RuntimeSettings, state: SceneState) -> SceneState:
    logger = _task_logger(settings, "storyboard_review")
    runtime = _runtime(settings)
    retry = _task_retry_count()
    logger.info("Running storyboard review")
    update = sb_editor_review(state, runtime.editor_tools)
    apply_state_update(state, update)
    review_meta = state.provider_metadata.get("sb_review", {}) if state.provider_metadata else {}
    all_issues = list(review_meta.get("all_issues") or state.sb_issues)
    review_payload = {
        "iteration": state.sb_iteration,
        "issues": list(state.sb_issues),
        "all_issues": all_issues,
        "blocking_issues": list(state.sb_issues),
        "mode": state.sb_review_mode,
        "error": state.sb_review_error,
    }
    _record_review(state, "storyboard", review_payload)
    runtime.save_json_artifact(
        f"reviews/storyboard_review_{state.sb_iteration:02d}.json",
        review_payload,
    )
    runtime.record_event(
        state,
        stage="storyboard_review",
        action="completed",
        asset_kind="review",
        label=f"Storyboard review #{state.sb_iteration}",
        retry=retry,
        counts={
            "iteration": state.sb_iteration,
            "issues": len(all_issues),
            "blocking_issues": len(state.sb_issues),
        },
        mode=state.sb_review_mode,
        error=state.sb_review_error,
    )
    _sync_state(runtime, state)
    logger.info(
        "Storyboard review finished: mode=%s issues=%d blocking=%d",
        state.sb_review_mode,
        len(all_issues),
        len(state.sb_issues),
    )
    return state


@task(name="storyboard_fix", retries=1, retry_delay_seconds=2, timeout_seconds=300, log_prints=True)
def storyboard_fix_task(settings: RuntimeSettings, state: SceneState) -> SceneState:
    logger = _task_logger(settings, "storyboard_fix")
    runtime = _runtime(settings)
    retry = _task_retry_count()
    if not state.sb_issues:
        logger.info("Skipping storyboard fix because there are no issues")
        runtime.record_event(
            state,
            stage="storyboard_fix",
            action="skipped",
            asset_kind="fix",
            label="Storyboard fix",
            retry=retry,
        )
        _sync_state(runtime, state)
        return state

    logger.info("Applying storyboard fixes for %d issues", len(state.sb_issues))
    before = state.model_copy(deep=True)
    update = sb_editor_fix(state, runtime.editor_tools)
    apply_state_update(state, update)
    state.segment_uris = []
    state.final_video_uri = None
    runtime.save_json_artifact("storyboard.json", state.storyboard_raw or {})
    changed_prompts = _record_prompt_change_events(
        runtime,
        before,
        state,
        stage="storyboard_fix",
        retry=retry,
    )
    runtime.record_event(
        state,
        stage="storyboard_fix",
        action="completed",
        asset_kind="fix",
        label=f"Storyboard fix #{state.sb_iteration}",
        retry=retry,
        counts={
            "iteration": state.sb_iteration,
            "issues": len(before.sb_issues),
            "prompt_changes": changed_prompts,
        },
        details={
            "regen_frames": list(state.regen_frames),
            "regen_segments": list(state.regen_segments),
            "edit_segments": list(state.edit_segments),
        },
    )
    _sync_state(runtime, state)
    logger.info("Storyboard fix finished: regen_frames=%s", state.regen_frames)
    return state
