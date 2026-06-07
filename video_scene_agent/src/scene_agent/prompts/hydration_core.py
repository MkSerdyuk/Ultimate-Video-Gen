
from __future__ import annotations

from typing import Any

from scene_agent.models import StoryBeatData, StoryboardData, WorldDescription, WorldObject

def _clean_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""
def _non_empty(*values: Any) -> list[str]:
    return [text for value in values if (text := _clean_text(value))]
def _join_csv(values: list[str]) -> str:
    return ", ".join(_non_empty(*values))
def _join_lines(lines: list[str]) -> str:
    return "\n".join(line for line in lines if line)
def _normalize_words(text: str) -> list[str]:
    return [word for word in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if word]
def _slugify(text: str) -> str:
    words = _normalize_words(text)
    return "-".join(words[:6]) or "beat"
def _object_identity_phrase(obj: WorldObject) -> str:
    bits = [f"{obj.name}: {obj.appearance}"]
    if obj.constraints:
        bits.append("must preserve " + ", ".join(obj.constraints))
    return "; ".join(bits)
def _object_aliases(obj: WorldObject) -> list[str]:
    return [obj.name.lower(), obj.id.lower(), obj.id.replace("_", " ").lower()]
def _infer_story_role(obj: WorldObject) -> str:
    text = " ".join([obj.id, obj.name, obj.appearance, " ".join(obj.constraints)]).lower()
    if any(token in text for token in ("woman", "man", "girl", "boy", "hero", "heroine", "protagonist", "person", "character")):
        return "character"
    if any(token in text for token in ("shoreline", "beach", "ocean", "sky", "sunset", "water", "sand", "forest", "city", "room")):
        return "environment"
    if any(token in text for token in ("tank", "vehicle", "truck", "car", "ship", "submarine")):
        return "vehicle"
    if any(token in text for token in ("orca", "creature", "monster", "beast", "alien", "dragon", "shark", "predator")):
        return "creature"
    if any(token in text for token in ("weapon", "gun", "sword", "bag", "document", "cup", "phone", "prop")):
        return "prop"
    return "object"
def _infer_default_visibility(obj: WorldObject) -> str:
    if obj.story_role in {"character", "environment"}:
        return "always_visible"
    if obj.story_role in {"threat", "creature", "vehicle"}:
        return "latent"
    return "conditional"
def normalize_world_description(world: WorldDescription) -> WorldDescription:
    """Backfill temporal visibility metadata on the world package."""
    normalized = world.model_copy(deep=True)

    for obj in normalized.objects:
        if not obj.story_role or obj.story_role == "object":
            obj.story_role = _infer_story_role(obj)
        if obj.default_visibility == "always_visible" and obj.story_role in {"creature", "vehicle"}:
            obj.default_visibility = "latent"
        elif not obj.default_visibility:
            obj.default_visibility = _infer_default_visibility(obj)

        if not obj.reveal_rules and obj.default_visibility in {"latent", "reveal_only"}:
            obj.reveal_rules = [
                "Only become fully visible during a reveal beat or when explicitly promoted to visible_object_ids.",
            ]
        if not obj.pre_reveal_hints and obj.default_visibility in {"latent", "reveal_only"}:
            obj.pre_reveal_hints = [
                "subtle off-screen presence",
                "indirect disturbance in the environment",
            ]
        if not obj.hard_exclusions_before_reveal and obj.default_visibility in {"latent", "reveal_only"}:
            obj.hard_exclusions_before_reveal = [
                f"do not show the full {obj.id} before its reveal beat",
            ]

    style = normalized.style_guide
    if not style.global_continuity_locks:
        style.global_continuity_locks = list(style.continuity_bible)
    if not style.public_continuity_locks:
        style.public_continuity_locks = list(style.continuity_bible)

    hidden_objects = [obj for obj in normalized.objects if obj.default_visibility in {"latent", "reveal_only"}]
    hidden_aliases = {alias for obj in hidden_objects for alias in _object_aliases(obj)}
    if not style.reveal_locked_traits:
        reveal_clauses: list[str] = []
        for clause in style.continuity_bible:
            if any(alias in clause.lower() for alias in hidden_aliases):
                reveal_clauses.append(clause)
        style.reveal_locked_traits = reveal_clauses
    if not style.public_continuity_locks:
        style.public_continuity_locks = list(style.global_continuity_locks)
    else:
        style.public_continuity_locks = [
            clause for clause in style.public_continuity_locks
            if not any(alias in clause.lower() for alias in hidden_aliases)
        ] or list(style.public_continuity_locks)

    if not normalized.primary_subject_ids:
        normalized.primary_subject_ids = [
            obj.id
            for obj in normalized.objects
            if obj.story_role == "character"
        ] or [obj.id for obj in normalized.objects[:1]]

    return normalized
def _objects_by_ids(storyboard: StoryboardData, object_ids: list[str]) -> list[WorldObject]:
    object_set = set(object_ids)
    return [obj for obj in storyboard.objects if obj.id in object_set]
def _beat_lookup(storyboard: StoryboardData) -> dict[str, StoryBeatData]:
    return {beat.id: beat for beat in storyboard.story_beats}
def _hint_clauses(storyboard: StoryboardData, hint_object_ids: list[str]) -> list[str]:
    clauses: list[str] = []
    for obj in _objects_by_ids(storyboard, hint_object_ids):
        clauses.extend(obj.pre_reveal_hints)
    return list(dict.fromkeys(_non_empty(*clauses)))
def _hidden_exclusions(storyboard: StoryboardData, hidden_object_ids: list[str]) -> list[str]:
    clauses: list[str] = []
    for obj in _objects_by_ids(storyboard, hidden_object_ids):
        clauses.extend(obj.hard_exclusions_before_reveal)
    return list(dict.fromkeys(_non_empty(*clauses)))
