
from __future__ import annotations

import re
from typing import Any

from scene_agent.models import StoryStyleGuide, StoryboardData, StoryboardFrameData, StoryboardSegmentData, WorldObject
from scene_agent.prompts.hydration_core import (
    _beat_lookup,
    _clean_text,
    _join_csv,
    _non_empty,
    _normalize_words,
    _object_aliases,
    _objects_by_ids,
)

def _primary_subject_ids(storyboard: StoryboardData) -> list[str]:
    if storyboard.primary_subject_ids:
        return list(dict.fromkeys(storyboard.primary_subject_ids))
    return [obj.id for obj in storyboard.objects if obj.story_role == "character"] or [obj.id for obj in storyboard.objects[:1]]
def _text_blob(*values: Any) -> str:
    return " ".join(_non_empty(*values)).lower()
def _infer_screen_position(frame: StoryboardFrameData) -> str:
    text = _text_blob(frame.composition, frame.blocking, frame.action_in_frame)
    if "left third" in text or "screen left" in text or "left side" in text:
        return "screen_left"
    if "right third" in text or "screen right" in text or "right side" in text:
        return "screen_right"
    if "center" in text or "central" in text or "middle" in text:
        return "screen_center"
    if "background" in text:
        return "screen_background"
    return "screen_unspecified"
def _infer_body_facing(frame: StoryboardFrameData) -> str:
    text = _text_blob(frame.blocking, frame.action_in_frame, frame.composition)
    if "facing left" in text:
        return "face_screen_left"
    if "facing right" in text:
        return "face_screen_right"
    if "back to camera" in text or "away from camera" in text:
        return "face_away_from_camera"
    if "toward camera" in text or "facing camera" in text:
        return "face_toward_camera"
    return "body_orientation_unspecified"
def _infer_travel_direction(frame: StoryboardFrameData) -> str:
    text = _text_blob(frame.blocking, frame.action_in_frame, frame.composition)
    if "left to right" in text:
        return "screen_left_to_right"
    if "right to left" in text:
        return "screen_right_to_left"
    if "toward camera" in text:
        return "toward_camera"
    if "away from camera" in text:
        return "away_from_camera"
    if any(token in text for token in ("stands", "static", "holds position", "stops", "pause")):
        return "static"
    return "movement_direction_unspecified"
def _infer_gaze_direction(frame: StoryboardFrameData) -> str:
    text = _text_blob(frame.blocking, frame.action_in_frame, frame.composition)
    if "looks left" in text or "looking left" in text or "gaze left" in text:
        return "look_screen_left"
    if "looks right" in text or "looking right" in text or "gaze right" in text:
        return "look_screen_right"
    if "looks toward camera" in text or "looking at camera" in text:
        return "look_toward_camera"
    if "looks away from camera" in text:
        return "look_away_from_camera"
    if "looks down" in text or "looking down" in text:
        return "look_down"
    if "looks up" in text or "looking up" in text:
        return "look_up"
    return "gaze_direction_unspecified"
def _infer_gaze_target(storyboard: StoryboardData, frame: StoryboardFrameData) -> str:
    text = _text_blob(frame.action_in_frame, frame.blocking, frame.composition)
    if "ocean" in text or "water" in text or "surface" in text or "wave" in text:
        for obj_id in frame.visible_object_ids + frame.hint_object_ids:
            if "ocean" in obj_id or "water" in obj_id or "surface" in obj_id:
                return f"look_at_object:{obj_id}"
        return "look_at_environment:ocean_surface"
    for obj in _objects_by_ids(storyboard, frame.visible_object_ids + frame.hint_object_ids):
        if any(alias in text for alias in _object_aliases(obj)):
            return f"look_at_object:{obj.id}"
    return "gaze_target_unspecified"
def _infer_camera_axis_side(frame: StoryboardFrameData) -> str:
    text = _text_blob(frame.camera, frame.composition, frame.blocking)
    if "camera left" in text or "left side of axis" in text:
        return "camera_left_of_axis"
    if "camera right" in text or "right side of axis" in text:
        return "camera_right_of_axis"
    return "preserve_current_axis"
def _infer_hero_presence(storyboard: StoryboardData, frame: StoryboardFrameData) -> dict[str, str]:
    presence: dict[str, str] = dict(frame.hero_presence or {})
    text = _text_blob(frame.action_in_frame, frame.blocking, frame.composition)
    for hero_id in _primary_subject_ids(storyboard):
        if hero_id in presence:
            continue
        if hero_id in frame.visible_object_ids:
            if "tiny" in text and hero_id in text:
                presence[hero_id] = "tiny_in_frame"
            elif "edge" in text and hero_id in text:
                presence[hero_id] = "partial_frame_edge"
            else:
                presence[hero_id] = "on_screen_primary"
        elif hero_id in frame.hint_object_ids:
            presence[hero_id] = "occluded_but_present"
        else:
            presence[hero_id] = ""
    return presence
def _infer_hero_scale(frame: StoryboardFrameData, storyboard: StoryboardData) -> dict[str, str]:
    scale: dict[str, str] = dict(frame.hero_scale or {})
    shot = _text_blob(frame.shot_size)
    default_scale = "scale_unspecified"
    if "extreme wide" in shot or "very wide" in shot or "wide shot" in shot:
        default_scale = "small_in_frame"
    elif "medium" in shot:
        default_scale = "mid_scale"
    elif "close" in shot:
        default_scale = "large_in_frame"
    for hero_id in _primary_subject_ids(storyboard):
        if hero_id not in scale and frame.hero_presence.get(hero_id, ""):
            scale[hero_id] = default_scale
    return scale
def _infer_segment_screen_direction_rule(start_frame: StoryboardFrameData, end_frame: StoryboardFrameData) -> str:
    start_dir = start_frame.travel_direction
    end_dir = end_frame.travel_direction
    if start_dir and end_dir and start_dir == end_dir:
        return f"preserve {start_dir}"
    if start_dir and end_dir and start_dir != end_dir and "unspecified" not in start_dir + end_dir:
        return f"intentional reorientation from {start_dir} to {end_dir}"
    return "preserve established screen direction"
def _infer_segment_gaze_rule(start_frame: StoryboardFrameData, end_frame: StoryboardFrameData) -> str:
    if start_frame.gaze_target and end_frame.gaze_target and start_frame.gaze_target == end_frame.gaze_target:
        return f"maintain eyeline on {start_frame.gaze_target}"
    if start_frame.gaze_direction and end_frame.gaze_direction and start_frame.gaze_direction == end_frame.gaze_direction:
        return f"preserve {start_frame.gaze_direction}"
    return "maintain readable eyeline continuity"
def _infer_segment_axis_rule(start_frame: StoryboardFrameData, end_frame: StoryboardFrameData) -> str:
    if start_frame.camera_axis_side and end_frame.camera_axis_side and start_frame.camera_axis_side == end_frame.camera_axis_side:
        return f"preserve {start_frame.camera_axis_side}"
    return "preserve current screen axis"
def _strip_hidden_object_mentions(text: str, hidden_objects: list[WorldObject]) -> str:
    """Remove clauses that explicitly mention hidden/off-screen objects."""
    cleaned = _clean_text(text)
    if not cleaned or not hidden_objects:
        return cleaned

    aliases = [alias for obj in hidden_objects for alias in _object_aliases(obj) if alias]
    clauses = [clause.strip() for clause in re.split(r"[;\n]+", cleaned) if clause.strip()]
    visible_clauses = [
        clause for clause in clauses
        if not any(alias in clause.lower() for alias in aliases)
    ]
    if not visible_clauses:
        return ""
    return "; ".join(visible_clauses)
def _style_phrase(style_guide: StoryStyleGuide, hidden_objects: list[WorldObject] | None = None) -> str:
    hidden_objects = hidden_objects or []
    parts = _non_empty(
        style_guide.style,
        _strip_hidden_object_mentions(style_guide.palette, hidden_objects),
        _strip_hidden_object_mentions(style_guide.cinematic_intent, hidden_objects),
        _strip_hidden_object_mentions(style_guide.camera_language, hidden_objects),
        _strip_hidden_object_mentions(style_guide.lighting_logic, hidden_objects),
    )
    return "; ".join(parts)
def _relevant_objects_for_frame(storyboard: StoryboardData, frame: StoryboardFrameData) -> list[WorldObject]:
    """Select objects that appear to be on-screen for a specific frame."""
    if frame.visible_object_ids:
        visible = [obj for obj in storyboard.objects if obj.id in frame.visible_object_ids]
        if visible:
            return visible

    context = " ".join(
        _non_empty(
            frame.action_in_frame,
            frame.frame_class,
            frame.camera,
            frame.composition,
            frame.blocking,
            frame.emotional_beat,
            " ".join(frame.must_have),
            " ".join(frame.continuity_anchors),
        )
    ).lower()
    context_words = set(_normalize_words(context))

    relevant: list[WorldObject] = []
    for obj in storyboard.objects:
        phrases = [
            obj.name.lower(),
            obj.id.lower(),
            obj.id.replace("_", " ").lower(),
        ]
        phrase_match = any(phrase and phrase in context for phrase in phrases)
        word_match = bool(set(_normalize_words(obj.name)) & context_words)
        if phrase_match or word_match:
            relevant.append(obj)
        elif obj.default_visibility == "always_visible" and obj.story_role in {"character", "environment"}:
            if frame.frame_class in {"establishing", "discovery", "reaction", "transition", "aftermath"}:
                relevant.append(obj)

    if relevant:
        deduped: list[WorldObject] = []
        seen: set[str] = set()
        for obj in relevant:
            if obj.id not in seen:
                deduped.append(obj)
                seen.add(obj.id)
        return deduped

    return []
def summarize_frame_camera(frame: StoryboardFrameData) -> str:
    """Build a concise human-readable frame camera summary from structured fields."""
    return _join_csv([
        frame.shot_size,
        frame.camera_angle,
        frame.lens,
        frame.camera_support,
    ])
def summarize_segment_transition(segment: StoryboardSegmentData) -> str:
    """Build a concise human-readable segment transition summary."""
    return "; ".join(
        _non_empty(
            segment.camera_move,
            segment.subject_motion,
            segment.environment_motion,
        )
    )
def _infer_frame_class(frame: StoryboardFrameData, idx: int, total_frames: int) -> str:
    text = " ".join(
        _non_empty(
            frame.action_in_frame,
            frame.emotional_beat,
            frame.blocking,
            " ".join(frame.must_have),
        )
    ).lower()
    if any(token in text for token in ("erupts", "emerges", "rises", "bursts", "reveal", "surfaces", "appears")):
        return "reveal"
    if any(token in text for token in ("runs", "sprints", "flees", "escapes", "chases")):
        return "escalation"
    if any(token in text for token in ("fear", "terror", "panic", "reacts", "turns away", "startles")):
        return "reaction"
    if any(token in text for token in ("notices", "spots", "studies", "discovers", "something beneath", "disturbance")):
        return "discovery"
    if idx == 0:
        return "establishing"
    if idx == total_frames - 1:
        return "aftermath"
    return "transition"
def _infer_segment_class(segment: StoryboardSegmentData, start_frame: StoryboardFrameData, end_frame: StoryboardFrameData) -> str:
    if end_frame.frame_class == "reveal":
        return "reveal_bridge"
    if end_frame.frame_class == "escalation":
        return "escalation_bridge"
    if end_frame.frame_class == "reaction":
        return "reaction_bridge"
    if end_frame.frame_class == "discovery":
        return "discovery_bridge"
    if end_frame.frame_class == "aftermath":
        return "aftermath_bridge"
    if start_frame.frame_class == end_frame.frame_class == "establishing":
        return "continuation"
    return "continuation"
def _context_mentions_alias(frame: StoryboardFrameData, obj: WorldObject) -> bool:
    text = " ".join(
        _non_empty(
            frame.action_in_frame,
            frame.blocking,
            frame.composition,
            frame.emotional_beat,
            " ".join(frame.must_have),
            " ".join(frame.continuity_anchors),
        )
    ).lower()
    return any(alias and alias in text for alias in _object_aliases(obj))
def _infer_frame_visibility(
    storyboard: StoryboardData,
    frame: StoryboardFrameData,
    *,
    revealed_so_far: set[str],
) -> tuple[list[str], list[str], list[str], set[str]]:
    beat = _beat_lookup(storyboard).get(frame.beat_id) if frame.beat_id else None
    explicit_visible = list(dict.fromkeys(frame.visible_object_ids))
    explicit_hidden = list(dict.fromkeys(frame.hidden_object_ids))
    explicit_hints = list(dict.fromkeys(frame.hint_object_ids))

    visible: list[str] = list(explicit_visible or (beat.visible_objects if beat else []))
    hinted: list[str] = list(explicit_hints)
    if beat:
        hinted.extend(
            [
                obj.id
                for obj in storyboard.objects
                if any(hint in beat.allowed_hints for hint in obj.pre_reveal_hints)
            ]
        )
    newly_revealed: set[str] = set()

    for obj in storyboard.objects:
        if obj.id in visible:
            if obj.default_visibility in {"latent", "reveal_only"}:
                newly_revealed.add(obj.id)
            continue
        alias_match = _context_mentions_alias(frame, obj)
        if obj.id in explicit_hidden:
            continue
        if obj.id in explicit_hints:
            hinted.append(obj.id)
            continue
        if beat and obj.id in beat.latent_objects:
            continue

        if obj.default_visibility == "always_visible":
            if alias_match or obj.story_role in {"character", "environment"}:
                visible.append(obj.id)
            continue

        if obj.id in revealed_so_far:
            if frame.frame_class in {"reveal", "aftermath", "escalation", "reaction", "transition"} or alias_match:
                visible.append(obj.id)
            continue

        if frame.frame_class == "reveal" and (alias_match or obj.story_role in {"threat", "creature", "vehicle"}):
            visible.append(obj.id)
            newly_revealed.add(obj.id)
            continue

        hint_match = any(hint and hint.lower() in " ".join(_non_empty(frame.action_in_frame, frame.blocking, frame.composition)).lower() for hint in obj.pre_reveal_hints)
        if alias_match or hint_match:
            hinted.append(obj.id)

    visible = list(dict.fromkeys(visible))
    hinted = [obj_id for obj_id in dict.fromkeys(hinted) if obj_id not in visible]
    hidden = [obj.id for obj in storyboard.objects if obj.id not in visible and obj.id not in hinted]
    return visible, hinted, hidden, newly_revealed
