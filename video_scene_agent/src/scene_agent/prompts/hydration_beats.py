
from __future__ import annotations

from scene_agent.models import StoryBeatData, StoryboardData
from scene_agent.prompts.hydration_core import _beat_lookup, _slugify
from scene_agent.prompts.hydration_inference import (
    _infer_body_facing,
    _infer_camera_axis_side,
    _infer_frame_class,
    _infer_frame_visibility,
    _infer_gaze_direction,
    _infer_gaze_target,
    _infer_hero_presence,
    _infer_hero_scale,
    _infer_screen_position,
    _infer_segment_axis_rule,
    _infer_segment_class,
    _infer_segment_gaze_rule,
    _infer_segment_screen_direction_rule,
    _infer_travel_direction,
    _primary_subject_ids,
)

def _derive_story_beats(storyboard: StoryboardData) -> StoryboardData:
    normalized = storyboard.model_copy(deep=True)
    total_frames = len(normalized.frames)
    if not normalized.primary_subject_ids:
        normalized.primary_subject_ids = _primary_subject_ids(normalized)

    for idx, frame in enumerate(normalized.frames):
        if not frame.frame_class:
            frame.frame_class = _infer_frame_class(frame, idx, total_frames)

    revealed_so_far: set[str] = set()
    beat_signatures: list[tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = []
    beat_ids: list[str] = []

    for frame in normalized.frames:
        visible, hinted, hidden, newly_revealed = _infer_frame_visibility(normalized, frame, revealed_so_far=revealed_so_far)
        frame.visible_object_ids = visible
        frame.hint_object_ids = hinted
        frame.hidden_object_ids = hidden
        if not frame.screen_position:
            frame.screen_position = _infer_screen_position(frame)
        if not frame.body_facing:
            frame.body_facing = _infer_body_facing(frame)
        if not frame.travel_direction:
            frame.travel_direction = _infer_travel_direction(frame)
        if not frame.gaze_direction:
            frame.gaze_direction = _infer_gaze_direction(frame)
        if not frame.gaze_target:
            frame.gaze_target = _infer_gaze_target(normalized, frame)
        if not frame.camera_axis_side:
            frame.camera_axis_side = _infer_camera_axis_side(frame)
        frame.hero_presence = _infer_hero_presence(normalized, frame)
        frame.hero_scale = _infer_hero_scale(frame, normalized)
        revealed_so_far.update(newly_revealed)

        signature = (
            frame.frame_class,
            tuple(visible),
            tuple(hinted),
            tuple(hidden),
        )
        if signature in beat_signatures:
            beat_idx = beat_signatures.index(signature)
        else:
            beat_signatures.append(signature)
            beat_idx = len(beat_signatures) - 1
        frame.beat_id = frame.beat_id or f"beat-{beat_idx:02d}-{_slugify(frame.frame_class)}"
        beat_ids.append(frame.beat_id)

    beats: list[StoryBeatData] = []
    seen_beats: set[str] = set()
    existing_beats = _beat_lookup(normalized)
    for frame in normalized.frames:
        if frame.beat_id in seen_beats:
            continue
        seen_beats.add(frame.beat_id)
        existing = existing_beats.get(frame.beat_id)
        beat = StoryBeatData(
            idx=len(beats),
            id=frame.beat_id,
            label=existing.label if existing and existing.label else frame.frame_class.replace("_", " ").title(),
            narrative_function=existing.narrative_function if existing and existing.narrative_function else frame.frame_class,
            goal=existing.goal if existing and existing.goal else frame.action_in_frame,
            emotion=existing.emotion if existing and existing.emotion else frame.emotional_beat,
            visible_objects=list(existing.visible_objects) if existing and existing.visible_objects else list(frame.visible_object_ids),
            latent_objects=list(existing.latent_objects) if existing and existing.latent_objects else list(frame.hidden_object_ids),
            allowed_hints=list(existing.allowed_hints) if existing and existing.allowed_hints else [
                hint
                for obj_id in frame.hint_object_ids
                for obj in normalized.objects
                if obj.id == obj_id
                for hint in obj.pre_reveal_hints
            ],
            motion_intensity=existing.motion_intensity if existing and existing.motion_intensity else (
                "high" if frame.frame_class in {"reveal", "escalation"} else "medium" if frame.frame_class in {"reaction", "transition"} else "low"
            ),
        )
        beats.append(beat)
    normalized.story_beats = beats

    for idx, segment in enumerate(normalized.segments):
        start_frame = normalized.frames[segment.start_frame_idx]
        end_frame = normalized.frames[segment.end_frame_idx]
        segment.beat_id = segment.beat_id or end_frame.beat_id
        if not segment.segment_class:
            segment.segment_class = _infer_segment_class(segment, start_frame, end_frame)

        persistent = sorted(set(start_frame.visible_object_ids) & set(end_frame.visible_object_ids))
        revealed = sorted(set(end_frame.visible_object_ids) - set(start_frame.visible_object_ids))
        hidden = sorted(set(end_frame.hidden_object_ids))
        hinted = sorted(set(end_frame.hint_object_ids) | set(start_frame.hint_object_ids))

        segment.visible_object_ids = list(dict.fromkeys(segment.visible_object_ids or (persistent + revealed)))
        segment.hint_object_ids = list(dict.fromkeys(segment.hint_object_ids or hinted))
        segment.hidden_object_ids = list(dict.fromkeys(segment.hidden_object_ids or [obj.id for obj in normalized.objects if obj.id not in segment.visible_object_ids and obj.id not in segment.hint_object_ids]))
        segment.visibility_transition = segment.visibility_transition or {
            "persistent": persistent,
            "revealed": revealed,
            "hidden": hidden,
            "hinted": hinted,
        }
        if not segment.screen_direction_rule:
            segment.screen_direction_rule = _infer_segment_screen_direction_rule(start_frame, end_frame)
        if not segment.gaze_continuity_rule:
            segment.gaze_continuity_rule = _infer_segment_gaze_rule(start_frame, end_frame)
        if not segment.camera_axis_rule:
            segment.camera_axis_rule = _infer_segment_axis_rule(start_frame, end_frame)
        if not segment.hero_presence_transition:
            segment.hero_presence_transition = {
                hero_id: f"{start_frame.hero_presence.get(hero_id, '')} -> {end_frame.hero_presence.get(hero_id, '')}"
                for hero_id in normalized.primary_subject_ids
                if start_frame.hero_presence.get(hero_id, '') or end_frame.hero_presence.get(hero_id, '')
            }
        if not segment.entry_exit_actions:
            actions: list[str] = []
            for hero_id in normalized.primary_subject_ids:
                end_presence = end_frame.hero_presence.get(hero_id, "")
                start_presence = start_frame.hero_presence.get(hero_id, "")
                if end_presence in {"off_screen_left", "off_screen_right"} and start_presence.startswith("on_screen"):
                    actions.append(f"{hero_id} exits into {end_presence}")
            segment.entry_exit_actions = actions
        if not segment.offscreen_justification and any(
            not end_frame.hero_presence.get(hero_id, "")
            and start_frame.hero_presence.get(hero_id, "").startswith("on_screen")
            for hero_id in normalized.primary_subject_ids
        ):
            segment.offscreen_justification = "Primary subject absence must be justified or corrected before final approval."

    return normalized
