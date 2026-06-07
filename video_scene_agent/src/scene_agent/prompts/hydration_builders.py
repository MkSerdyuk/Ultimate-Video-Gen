
from __future__ import annotations

from typing import Any

from scene_agent.models import StoryboardData, StoryboardFrameData, StoryboardSegmentData, WorldDescription
from scene_agent.prompts.hydration_beats import _derive_story_beats
from scene_agent.prompts.hydration_core import (
    _clean_text,
    _hidden_exclusions,
    _hint_clauses,
    _join_csv,
    _join_lines,
    _non_empty,
    _object_identity_phrase,
    _objects_by_ids,
    normalize_world_description,
)
from scene_agent.prompts.hydration_inference import (
    _relevant_objects_for_frame,
    _strip_hidden_object_mentions,
    _style_phrase,
    summarize_frame_camera,
    summarize_segment_transition,
)

def derive_frame_image_prompt(storyboard: StoryboardData, frame: StoryboardFrameData) -> str:
    """Build a deterministic image payload prompt from structured shot metadata."""
    style_guide = storyboard.style_guide
    visible_objects = _objects_by_ids(storyboard, frame.visible_object_ids) or _relevant_objects_for_frame(storyboard, frame)
    hidden_or_hinted_ids = list(dict.fromkeys(frame.hidden_object_ids + frame.hint_object_ids))
    hidden_objects = _objects_by_ids(storyboard, hidden_or_hinted_ids)
    identity = (
        "; ".join(_object_identity_phrase(obj) for obj in visible_objects)
        or "Follow only the visible on-screen subject cues from the action, blocking, and must-have list."
    )
    continuity_items = [
        item for item in (style_guide.public_continuity_locks or style_guide.continuity_bible)
        if _strip_hidden_object_mentions(item, hidden_objects)
    ] + [
        item for item in frame.continuity_anchors
        if _strip_hidden_object_mentions(item, hidden_objects)
    ]
    continuity = "; ".join(dict.fromkeys(continuity_items)) or "Maintain exact scene continuity."
    camera_summary = summarize_frame_camera(frame) or frame.camera
    lighting_and_color = "; ".join(
        _non_empty(
            frame.lighting,
            _strip_hidden_object_mentions(style_guide.lighting_logic, hidden_objects),
            _strip_hidden_object_mentions(style_guide.palette, hidden_objects),
        )
    ) or "Keep lighting and palette stable."

    lines = [
        "Create a single cinematic keyframe.",
        f"Beat objective: {frame.beat_id or 'unassigned beat'} / {frame.frame_class or 'unclassified frame'}",
        f"Subject identity: {identity}",
        f"Wardrobe and props: {'; '.join(frame.must_have) if frame.must_have else 'Only the canonical scene elements that are already established.'}",
        f"Setting: {storyboard.scene_background}",
        f"Shot design: {camera_summary or 'Use the established scene camera language.'}",
        f"Composition and blocking: {'; '.join(_non_empty(frame.composition, frame.blocking)) or 'Preserve the established composition and spatial relationships.'}",
        f"Screen geography: position={frame.screen_position or 'screen_unspecified'}; body_facing={frame.body_facing or 'body_orientation_unspecified'}; travel={frame.travel_direction or 'movement_direction_unspecified'}; gaze={frame.gaze_direction or 'gaze_direction_unspecified'}; gaze_target={frame.gaze_target or 'gaze_target_unspecified'}; axis={frame.camera_axis_side or 'preserve_current_axis'}",
        f"Primary subject presence: {frame.hero_presence or 'No primary-subject presence metadata provided.'}",
        f"Primary subject scale: {frame.hero_scale or 'No primary-subject scale metadata provided.'}",
        f"Action and expression: {'; '.join(_non_empty(frame.action_in_frame, frame.emotional_beat)) or 'Hold the planned screen moment with clear readable body language.'}",
        f"Lighting and color: {lighting_and_color}",
        f"Style and camera language: {_style_phrase(style_guide, hidden_objects) or 'Use the defined cinematic style consistently.'}",
        "Surface finish: keep materials, textures, and detail density coherent with the scene style.",
        f"Continuity locks: {continuity}",
        f"Allowed hints only: {'; '.join(_hint_clauses(storyboard, frame.hint_object_ids)) or 'No hidden-object hinting in this frame.'}",
        f"Forbidden early reveal: {'; '.join(_hidden_exclusions(storyboard, hidden_or_hinted_ids)) or 'No hidden-object leak.'}",
        "Do not reveal off-screen or future-reveal story elements that are not part of this frame.",
    ]
    return _join_lines(lines)
def derive_segment_video_prompt(storyboard: StoryboardData, segment: StoryboardSegmentData) -> str:
    """Build a deterministic video payload prompt from structured segment metadata."""
    style_guide = storyboard.style_guide
    start_frame = storyboard.frames[segment.start_frame_idx]
    end_frame = storyboard.frames[segment.end_frame_idx]
    hidden_or_hinted_ids = list(dict.fromkeys(segment.hidden_object_ids + segment.hint_object_ids))
    hidden_objects = _objects_by_ids(storyboard, hidden_or_hinted_ids)
    continuity = "; ".join(
        dict.fromkeys(
            [
                item
                for item in (style_guide.global_continuity_locks or style_guide.continuity_bible) + segment.continuity_anchors
                if _strip_hidden_object_mentions(item, hidden_objects)
            ]
        )
    ) or "Preserve exact continuity."
    motion_beats = segment.motion_beats or [_clean_text(segment.transition_text)] if segment.transition_text else []
    motion_beats_text = " | ".join(_non_empty(*motion_beats)) or "Keep the motion simple, readable, and continuous."
    end_match = segment.end_match_notes or (
        f"Land on the end keyframe exactly: {_join_csv([end_frame.shot_size, end_frame.camera_angle, end_frame.lens])}; "
        f"{_clean_text(end_frame.action_in_frame)}; {_clean_text(end_frame.composition)}; {_clean_text(end_frame.blocking)}"
    )

    lines = [
        "Generate one continuous image-to-video bridge between the provided start and end keyframes.",
        f"Beat objective: {segment.beat_id or 'unassigned beat'} / {segment.segment_class or 'unclassified segment'}",
        f"Clip objective: {_clean_text(segment.transition_text) or 'Create a clean cinematic bridge between the two anchor frames.'}",
        f"Camera move: {_clean_text(segment.camera_move) or 'One restrained camera move only.'}",
        f"Subject motion: {_clean_text(segment.subject_motion) or 'Keep subject motion readable and physically plausible.'}",
        f"Environment motion: {_clean_text(segment.environment_motion) or 'Only subtle environmental support motion.'}",
        f"Screen direction continuity: {_clean_text(segment.screen_direction_rule) or 'Preserve established screen direction.'}",
        f"Gaze continuity: {_clean_text(segment.gaze_continuity_rule) or 'Preserve readable eyeline continuity.'}",
        f"Camera axis continuity: {_clean_text(segment.camera_axis_rule) or 'Preserve current screen axis.'}",
        f"Motion beats: {motion_beats_text}",
        f"Start frame anchor: {_clean_text(start_frame.action_in_frame)}; {_join_csv([start_frame.shot_size, start_frame.camera_angle, start_frame.lens])}",
        f"End-state must match: {end_match}",
        f"Style and camera language: {_style_phrase(style_guide, hidden_objects) or 'Preserve the scene style and camera language.'}",
        f"Continuity locks: {continuity}",
        f"Visibility transition: persistent={', '.join(segment.visibility_transition.get('persistent', [])) or 'none'}; revealed={', '.join(segment.visibility_transition.get('revealed', [])) or 'none'}; hinted={', '.join(segment.visibility_transition.get('hinted', [])) or 'none'}",
        f"Primary subject continuity: {segment.hero_presence_transition or 'No primary-subject transition metadata provided.'}",
        f"Entry / exit actions: {', '.join(segment.entry_exit_actions) or 'No subject exits or entries are planned.'}",
        f"Off-screen justification: {_clean_text(segment.offscreen_justification) or 'Primary subjects should stay visually accounted for.'}",
        f"Allowed hints only: {'; '.join(_hint_clauses(storyboard, segment.hint_object_ids)) or 'No hidden-object hinting in this segment.'}",
        f"Forbidden early reveal: {'; '.join(_hidden_exclusions(storyboard, hidden_or_hinted_ids)) or 'No hidden-object leak.'}",
        "Physics and tempo: natural timing, stable anatomy, coherent scale, no jumpy interpolation.",
        "Forbidden failures: identity drift, wardrobe drift, unstable background, broken hands, extra limbs, text artifacts, pseudo-cuts, lighting drift, style drift, missed end-frame landing.",
    ]
    return _join_lines(lines)
def hydrate_storyboard_payload_prompts(storyboard: StoryboardData) -> StoryboardData:
    """Populate runtime payload prompts from structured cinematic metadata."""
    hydrated = normalize_world_description(
        WorldDescription(
            scene_background=storyboard.scene_background,
            objects=list(storyboard.objects),
            primary_subject_ids=list(storyboard.primary_subject_ids),
            style_guide=storyboard.style_guide,
        )
    )
    hydrated_storyboard = storyboard.model_copy(
        update={
            "objects": hydrated.objects,
            "primary_subject_ids": hydrated.primary_subject_ids,
            "style_guide": hydrated.style_guide,
        },
        deep=True,
    )
    hydrated = _derive_story_beats(hydrated_storyboard)

    for frame in hydrated.frames:
        if not frame.camera:
            frame.camera = summarize_frame_camera(frame)
        if not frame.lighting and hydrated.style_guide.lighting_logic:
            frame.lighting = hydrated.style_guide.lighting_logic
        frame.image_prompt = derive_frame_image_prompt(hydrated, frame)

    for segment in hydrated.segments:
        if not segment.transition_text:
            segment.transition_text = summarize_segment_transition(segment)
        segment.video_prompt = derive_segment_video_prompt(hydrated, segment)

    return hydrated
def hydrate_storyboard_payload_prompts_dict(storyboard_data: dict[str, Any]) -> dict[str, Any]:
    """Validate a storyboard dict and fill deterministic runtime prompts."""
    hydrated = hydrate_storyboard_payload_prompts(StoryboardData.model_validate(storyboard_data))
    return hydrated.model_dump(mode="json", by_alias=True)
