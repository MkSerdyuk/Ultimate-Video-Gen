
from __future__ import annotations

from prefect import task

from scene_agent.flows.tasks.common import (
    _record_review,
    _record_segment_asset_events,
    _runtime,
    _sync_state,
    _task_logger,
    _task_retry_count,
)
from scene_agent.models import SceneState
from scene_agent.pipeline.video_editor import vid_editor_fix, vid_editor_review
from scene_agent.pipeline.video_segments import segments_edit, segments_generate, stitch_video
from scene_agent.runtime import RuntimeSettings, apply_state_update

@task(name="segments_generate", retries=2, retry_delay_seconds=60, timeout_seconds=2400, log_prints=True)
def segments_task(settings: RuntimeSettings, state: SceneState) -> SceneState:
    logger = _task_logger(settings, "segments_generate")
    runtime = _runtime(settings)
    retry = _task_retry_count()
    if state.segment_uris and not state.regen_segments:
        logger.info("Skipping segment generation because segments already exist and no regeneration was requested")
        runtime.record_event(
            state,
            stage="segments_generate",
            action="skipped",
            asset_kind="stage",
            label="Segment generation",
            retry=retry,
        )
        _sync_state(runtime, state)
        return state

    logger.info("Generating video segments")
    requested_regen = list(state.regen_segments)
    before = state.model_copy(deep=True)
    update = segments_generate(state, runtime.operator_tools)
    apply_state_update(state, update)
    changed_segments = _record_segment_asset_events(
        runtime,
        before,
        state,
        stage="segments_generate",
        retry=retry,
    )
    state.final_video_uri = None
    runtime.save_json_artifact("storyboard.json", state.storyboard_raw or {})
    runtime.record_event(
        state,
        stage="segments_generate",
        action="completed",
        asset_kind="stage",
        label="Segment generation",
        retry=retry,
        counts={
            "segments_changed": changed_segments,
            "segment_count": len(state.segment_uris),
        },
        details={"requested_regen_segments": requested_regen},
    )
    _sync_state(runtime, state)
    logger.info("Segment generation finished with %d segments", len(state.segment_uris))
    return state


@task(name="segments_edit", retries=2, retry_delay_seconds=60, timeout_seconds=2400, log_prints=True)
def segments_edit_task(settings: RuntimeSettings, state: SceneState) -> SceneState:
    logger = _task_logger(settings, "segments_edit")
    runtime = _runtime(settings)
    retry = _task_retry_count()
    if not state.edit_segments:
        logger.info("Skipping segment repair because no edit_segments were requested")
        runtime.record_event(
            state,
            stage="segments_edit",
            action="skipped",
            asset_kind="stage",
            label="Segment repair",
            retry=retry,
        )
        _sync_state(runtime, state)
        return state

    logger.info("Repairing video segments: %s", state.edit_segments)
    requested_edit = list(state.edit_segments)
    before = state.model_copy(deep=True)
    update = segments_edit(state, runtime.operator_tools)
    apply_state_update(state, update)
    changed_segments = _record_segment_asset_events(
        runtime,
        before,
        state,
        stage="segments_edit",
        retry=retry,
    )
    state.final_video_uri = None
    runtime.save_json_artifact("storyboard.json", state.storyboard_raw or {})
    runtime.record_event(
        state,
        stage="segments_edit",
        action="completed",
        asset_kind="stage",
        label="Segment repair",
        retry=retry,
        counts={
            "segments_changed": changed_segments,
            "segment_count": len(state.segment_uris),
        },
        details={
            "edit_segments": requested_edit,
            "requested_edit_segments": requested_edit,
            "missing_source_regen_segments": list(state.regen_segments),
        },
    )
    _sync_state(runtime, state)
    logger.info("Segment repair finished with %d changed segments", changed_segments)
    return state


@task(name="stitch_video", retries=1, retry_delay_seconds=5, timeout_seconds=300, log_prints=True)
def stitch_task(settings: RuntimeSettings, state: SceneState) -> SceneState:
    logger = _task_logger(settings, "stitch_video")
    runtime = _runtime(settings)
    retry = _task_retry_count()
    if state.final_video_uri and not state.regen_frames and not state.regen_segments and not state.edit_segments:
        logger.info("Skipping stitch because final video already exists and no regeneration was requested")
        runtime.record_event(
            state,
            stage="stitch_video",
            action="skipped",
            asset_kind="video",
            label="Final video",
            retry=retry,
        )
        _sync_state(runtime, state)
        return state

    logger.info("Stitching final video from %d segments", len(state.segment_uris))
    before = state.model_copy(deep=True)
    update = stitch_video(state, runtime.operator_tools)
    apply_state_update(state, update)
    runtime.record_event(
        state,
        stage="stitch_video",
        action="replaced" if before.final_video_uri else "generated",
        asset_kind="video",
        label="Final video",
        from_value=before.final_video_uri,
        to_value=state.final_video_uri,
        retry=retry,
        counts={"segments": len(state.segment_uris)},
    )
    _sync_state(runtime, state)
    logger.info("Stitch finished: final_video_uri=%s", state.final_video_uri)
    return state


@task(name="video_review", retries=1, retry_delay_seconds=2, timeout_seconds=300, log_prints=True)
def video_review_task(settings: RuntimeSettings, state: SceneState) -> SceneState:
    logger = _task_logger(settings, "video_review")
    runtime = _runtime(settings)
    retry = _task_retry_count()
    logger.info("Running final video review")
    update = vid_editor_review(state, runtime.video_editor_tools)
    apply_state_update(state, update)
    review_meta = state.provider_metadata.get("vid_review", {}) if state.provider_metadata else {}
    all_issues = list(review_meta.get("all_issues") or state.vid_issues)
    review_payload = {
        "iteration": state.vid_iteration,
        "issues": list(state.vid_issues),
        "all_issues": all_issues,
        "blocking_issues": list(state.vid_issues),
        "mode": state.vid_review_mode,
        "error": state.vid_review_error,
    }
    _record_review(state, "video", review_payload)
    runtime.save_json_artifact(
        f"reviews/video_review_{state.vid_iteration:02d}.json",
        review_payload,
    )
    runtime.record_event(
        state,
        stage="video_review",
        action="completed",
        asset_kind="review",
        label=f"Video review #{state.vid_iteration}",
        retry=retry,
        counts={
            "iteration": state.vid_iteration,
            "issues": len(all_issues),
            "blocking_issues": len(state.vid_issues),
        },
        mode=state.vid_review_mode,
        error=state.vid_review_error,
    )
    _sync_state(runtime, state)
    logger.info(
        "Video review finished: mode=%s issues=%d blocking=%d",
        state.vid_review_mode,
        len(all_issues),
        len(state.vid_issues),
    )
    return state


@task(name="video_fix", retries=1, retry_delay_seconds=2, timeout_seconds=300, log_prints=True)
def video_fix_task(settings: RuntimeSettings, state: SceneState) -> SceneState:
    logger = _task_logger(settings, "video_fix")
    runtime = _runtime(settings)
    retry = _task_retry_count()
    if not state.vid_issues:
        logger.info("Skipping video fix because there are no issues")
        runtime.record_event(
            state,
            stage="video_fix",
            action="skipped",
            asset_kind="fix",
            label="Video fix",
            retry=retry,
        )
        _sync_state(runtime, state)
        return state

    logger.info("Applying video fixes for %d issues", len(state.vid_issues))
    before = state.model_copy(deep=True)
    update = vid_editor_fix(state, runtime.video_editor_tools)
    apply_state_update(state, update)
    if state.regen_frames or state.regen_segments or state.edit_segments:
        state.final_video_uri = None
    runtime.save_json_artifact("storyboard.json", state.storyboard_raw or {})
    runtime.record_event(
        state,
        stage="video_fix",
        action="completed",
        asset_kind="fix",
        label=f"Video fix #{state.vid_iteration}",
        retry=retry,
        counts={
            "iteration": state.vid_iteration,
            "issues": len(before.vid_issues),
        },
        details={
            "regen_frames": list(state.regen_frames),
            "regen_segments": list(state.regen_segments),
            "edit_segments": list(state.edit_segments),
        },
    )
    _sync_state(runtime, state)
    logger.info(
        "Video fix finished: regen_frames=%s edit_segments=%s regen_segments=%s",
        state.regen_frames,
        state.edit_segments,
        state.regen_segments,
    )
    return state
