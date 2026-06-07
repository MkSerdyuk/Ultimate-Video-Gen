
from __future__ import annotations

import logging

from scene_agent.models import SceneState, StoryboardData, WorldDescription
from scene_agent.pipeline.director_tools import DirectorTools
from scene_agent.prompts import (
    DIRECTOR_STORYBOARD_SYSTEM,
    DIRECTOR_WORLD_SYSTEM,
    format_director_storyboard_user,
    format_director_world_user,
    hydrate_storyboard_payload_prompts,
    normalize_world_description,
    to_json,
)
from scene_agent.utils.json_llm import clean_json_response, parse_partial_json

log = logging.getLogger(__name__)

def _normalize_segment_durations_for_runtime(storyboard_data: dict, constraints) -> dict:
    """
    Normalize storyboard segment duration fields for the video runtime.
    """
    segments = storyboard_data.get("segments", [])
    if not segments:
        return storyboard_data

    target_total = constraints.target_duration_sec or constraints.duration_sec
    if not target_total:
        return storyboard_data

    default_duration = round(float(target_total) / len(segments), 3)
    for segment in segments:
        duration = float(segment.get("duration", segment.get("duration_sec", default_duration)))
        segment["duration"] = duration
        segment["duration_sec"] = duration

    storyboard_data["segments"] = segments
    return storyboard_data

def director_world(state: SceneState, tools: DirectorTools) -> dict:
    """
    Generate world description from user brief.

    Args:
        state: Current graph state
        tools: Director tools (LLM, image tool)

    Returns:
        Partial state update with world
    """
    log.info("director_world: Generating world description")

    user_brief = state.user_brief
    constraints_dict = {
        "aspect_ratio": state.constraints.aspect_ratio,
        "duration_sec": state.constraints.duration_sec,
        "style_tags": state.constraints.style_tags,
    }
    constraints_json = to_json(constraints_dict)

    # Format prompt
    user_prompt = format_director_world_user(user_brief, constraints_json)

    # Call LLM with retry for incomplete JSON
    response_text = tools.llm.chat_with_retry(
        user=user_prompt,
        system=DIRECTOR_WORLD_SYSTEM,
        json_mode=True,
        retries=2,
    )

    # Log raw response for debugging
    log.debug(f"LLM response (first 200 chars): {response_text[:200]}...")

    # Clean and parse JSON response
    cleaned = clean_json_response(response_text)
    world_data = parse_partial_json(cleaned)

    if world_data is None:
        log.error(f"Failed to parse LLM response. Raw: {response_text[:500]}")
        raise ValueError(f"LLM did not return valid JSON. Response: {response_text[:200]}...")

    log.info(f"Generated world: {world_data.get('scene_background', '')[:50]}...")

    validated_world = normalize_world_description(WorldDescription.model_validate(world_data))
    return {
        "world_raw": validated_world.model_dump(mode="json"),
        "world": validated_world,
        "provider_metadata": {
            "director_world": {
                "provider": "openrouter",
                "model": getattr(getattr(tools.llm, "tool", tools.llm), "default_model", ""),
            }
        },
    }

def director_storyboard(state: SceneState, tools: DirectorTools) -> dict:
    """
    Generate storyboard from world description.

    Args:
        state: Current graph state
        tools: Director tools

    Returns:
        Partial state update with storyboard
    """
    log.info("director_storyboard: Generating storyboard")

    world_data = state.world_raw or {}
    if not world_data:
        raise ValueError("World not generated yet")

    user_brief = state.user_brief
    constraints_dict = {
        "aspect_ratio": state.constraints.aspect_ratio,
        "duration_sec": state.constraints.duration_sec,
        "target_duration_sec": state.constraints.target_duration_sec or state.constraints.duration_sec,
        "fps": state.constraints.fps,
        "num_keyframes": state.constraints.num_keyframes,
    }
    constraints_json = to_json(constraints_dict)
    world_json = to_json(world_data)

    # Format prompt
    user_prompt = format_director_storyboard_user(user_brief, constraints_json, world_json)

    # Call LLM with retry for incomplete JSON
    response_text = tools.llm.chat_with_retry(
        user=user_prompt,
        system=DIRECTOR_STORYBOARD_SYSTEM,
        json_mode=True,
        retries=2,
    )

    # Log raw response for debugging
    log.debug(f"LLM response (first 200 chars): {response_text[:200]}...")

    # Clean and parse JSON response
    cleaned = clean_json_response(response_text)
    storyboard_data = parse_partial_json(cleaned)

    if storyboard_data is None:
        log.error(f"Failed to parse LLM response. Raw: {response_text[:500]}")
        raise ValueError(f"LLM did not return valid JSON. Response: {response_text[:200]}...")

    storyboard_data = _normalize_segment_durations_for_runtime(storyboard_data, state.constraints)

    log.info(
        f"Generated storyboard: {len(storyboard_data.get('frames', []))} frames, "
        f"{len(storyboard_data.get('segments', []))} segments"
    )

    validated_storyboard = hydrate_storyboard_payload_prompts(StoryboardData.model_validate(storyboard_data))
    return {
        "storyboard_raw": validated_storyboard.model_dump(mode="json", by_alias=True),
        "storyboard": validated_storyboard,
        "provider_metadata": {
            "director_storyboard": {
                "provider": "openrouter",
                "model": getattr(getattr(tools.llm, "tool", tools.llm), "default_model", ""),
            }
        },
    }
