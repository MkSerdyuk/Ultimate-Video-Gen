
from __future__ import annotations

import logging

from scene_agent.models import SceneState, StoryboardData
from scene_agent.pipeline.director_tools import DirectorTools
from scene_agent.pipeline.storyboard_markdown import write_storyboard_markdown
from scene_agent.tools.vision_rewriter import normalize_negative_items

log = logging.getLogger(__name__)

def keyframes_generate(state: SceneState, tools: DirectorTools) -> dict:
    """
    Generate key frame images from storyboard with vision context for consistency.

    Now uses sequential generation where each frame (except the first) is
    generated with context from previous N frames via base64 images.

    Args:
        state: Current graph state
        tools: Director tools

    Returns:
        Partial state update with frame URIs
    """
    log.info("keyframes_generate: Generating key frame images with vision context")

    storyboard_data = state.storyboard_raw or {}
    if not storyboard_data:
        raise ValueError("Storyboard not generated yet")

    frames = storyboard_data.get("frames", [])
    aspect_ratio = state.constraints.aspect_ratio
    prev_n = state.constraints.frame_context_prev_n

    # If regen_frames is specified, only regenerate those frames
    regen_indices = state.regen_frames if state.regen_frames else []

    frame_uris = state.frame_uris or []
    all_frames = frames

    # Track prompt changes for markdown documentation (merge with existing)
    prompt_changes: dict[int, dict[str, str]] = dict(state.prompt_changes or {})

    # Generate frames sequentially
    for i, frame in enumerate(all_frames):
        # Skip if not in regen_indices (when regenerating specific frames)
        if regen_indices and i not in regen_indices:
            # Keep existing URI
            existing_uri = frame.get("keyframe_uri")
            if existing_uri and i < len(frame_uris):
                continue  # Skip, use existing

        # Get current prompt info
        original_prompt = frame.get("image_prompt", "")
        original_negative = normalize_negative_items(frame.get("negative", []))

        action = frame.get("action_in_frame", "")
        camera = frame.get("camera", "")
        lighting = frame.get("lighting", "")

        log.info(f"Generating key frame {i + 1}/{len(all_frames)} (idx={frame.get('idx', i)})")

        # Apply vision rewriter for context consistency (if enabled and not first frame)
        final_prompt = original_prompt
        final_negative = original_negative

        if i > 0 and prev_n > 0 and tools.vision_rewriter and tools.storage:
            # Gather previous N frames as base64 context
            prev_indices = list(range(max(0, i - prev_n), i))

            if prev_indices:
                log.info(f"Frame {i}: using context from frames {prev_indices}")

                # Convert previous frame URIs to base64 data URLs
                context_data_urls = []
                for prev_idx in prev_indices:
                    if prev_idx < len(frame_uris):
                        prev_uri = frame_uris[prev_idx]
                        try:
                            from scene_agent.utils.image_data_url import encode_uri_to_data_url

                            data_url = encode_uri_to_data_url(
                                storage=tools.storage,
                                uri=prev_uri,
                                max_side_px=state.constraints.vision_image_max_side_px,
                                jpeg_quality=state.constraints.vision_image_jpeg_quality,
                                target_mime=state.constraints.vision_image_mime,
                            )
                            context_data_urls.append(data_url)
                        except Exception as e:
                            log.warning(f"Failed to encode frame {prev_idx}: {e}")

                if context_data_urls:
                    # Call vision rewriter
                    frame_info = {
                        "idx": i,
                        "action": action,
                        "camera": camera,
                        "lighting": lighting,
                    }

                    result = tools.vision_rewriter.rewrite_with_context(
                        current_prompt=original_prompt,
                        current_negative=original_negative,
                        context_images=context_data_urls,
                        frame_info=frame_info,
                        constraints=state.constraints,
                    )

                    final_prompt = result.get("image_prompt", original_prompt)
                    result_negative = normalize_negative_items(result.get("negative", original_negative))
                    final_negative = list(dict.fromkeys([*result_negative, *original_negative]))

                    # Track prompt change if vision rewriter modified it
                    if final_prompt != original_prompt:
                        prompt_changes[i] = {
                            "original": original_prompt,
                            "updated": final_prompt,
                        }

                    log.info(f"Frame {i}: prompt refined by vision rewriter")

        # Generate image with (possibly refined) prompt
        uri = tools.image_tool.generate(
            prompt=final_prompt,
            negative_prompt=", ".join(final_negative) if final_negative else "",
            aspect_ratio=aspect_ratio,
        )

        # Store URI
        if i < len(frame_uris):
            frame_uris[i] = uri
        else:
            frame_uris.append(uri)

        # Update storyboard with URI
        frame["keyframe_uri"] = uri
        frame["image_prompt"] = final_prompt  # Store refined prompt
        if final_negative:
            frame["negative"] = final_negative

    log.info(f"Generated {len(frame_uris)} key frame images")

    # Write storyboard markdown to artifacts folder
    md_path = write_storyboard_markdown(state, tools.storage, storyboard_data, frame_uris, prompt_changes)
    if md_path:
        log.info(f"Director's cut saved to: {md_path}")

    return {
        "frame_uris": frame_uris,
        "storyboard_raw": storyboard_data,
        "storyboard": StoryboardData.model_validate(storyboard_data),
        "prompt_changes": prompt_changes,
        "provider_metadata": {
            "keyframes_generate": {
                "provider": "openrouter",
                "image_model": getattr(getattr(tools.image_tool, "tool", tools.image_tool), "model", ""),
                "vision_model": getattr(tools.vision_rewriter, "vision_model", ""),
            }
        },
    }


def keyframes_regenerate(state: SceneState, tools: DirectorTools) -> dict:
    """
    Regenerate specific keyframes with neighbor context for fixes.

    When regen_frames is set, this function regenerates only those frames
    using their neighbors (prev/next) as visual context via base64 images.

    Args:
        state: Current graph state
        tools: Director tools

    Returns:
        Partial state update with updated frame URIs
    """
    log.info(f"keyframes_regenerate: Regenerating frames {state.regen_frames}")

    storyboard_data = state.storyboard_raw or {}
    if not storyboard_data:
        raise ValueError("Storyboard not generated yet")

    frames = storyboard_data.get("frames", [])
    frame_uris = state.frame_uris or []
    regen_indices = state.regen_frames

    if not regen_indices:
        return {"frame_uris": frame_uris, "storyboard_raw": storyboard_data}

    aspect_ratio = state.constraints.aspect_ratio

    # Track prompt changes for markdown documentation (merge with existing)
    prompt_changes: dict[int, dict[str, str]] = dict(state.prompt_changes or {})

    for i in regen_indices:
        if i >= len(frames):
            log.warning(f"Frame index {i} out of range, skipping")
            continue

        frame = frames[i]
        original_prompt = frame.get("image_prompt", "")
        original_negative = normalize_negative_items(frame.get("negative", []))

        log.info(f"Regenerating key frame {i} with neighbor context")

        # Gather neighbor frames as base64 context
        neighbor_data_urls = {}
        frame_info = {
            "idx": i,
            "action": frame.get("action_in_frame", ""),
            "camera": frame.get("camera", ""),
        }

        # Previous frame
        if i > 0 and i - 1 < len(frame_uris):
            try:
                from scene_agent.utils.image_data_url import encode_uri_to_data_url

                prev_uri = frame_uris[i - 1]
                data_url = encode_uri_to_data_url(
                    storage=tools.storage,
                    uri=prev_uri,
                    max_side_px=state.constraints.vision_image_max_side_px,
                    jpeg_quality=state.constraints.vision_image_jpeg_quality,
                    target_mime=state.constraints.vision_image_mime,
                )
                neighbor_data_urls["prev"] = data_url
            except Exception as e:
                raise RuntimeError(f"Failed to encode prev frame {i-1}: {e}") from e

        # Current frame (if exists)
        if i < len(frame_uris) and frame_uris[i]:
            try:
                from scene_agent.utils.image_data_url import encode_uri_to_data_url

                data_url = encode_uri_to_data_url(
                    storage=tools.storage,
                    uri=frame_uris[i],
                    max_side_px=state.constraints.vision_image_max_side_px,
                    jpeg_quality=state.constraints.vision_image_jpeg_quality,
                    target_mime=state.constraints.vision_image_mime,
                )
                neighbor_data_urls["current"] = data_url
            except Exception as e:
                raise RuntimeError(f"Failed to encode current frame {i}: {e}") from e

        # Next frame
        if i + 1 < len(frames) and i + 1 < len(frame_uris):
            try:
                from scene_agent.utils.image_data_url import encode_uri_to_data_url

                next_uri = frame_uris[i + 1]
                data_url = encode_uri_to_data_url(
                    storage=tools.storage,
                    uri=next_uri,
                    max_side_px=state.constraints.vision_image_max_side_px,
                    jpeg_quality=state.constraints.vision_image_jpeg_quality,
                    target_mime=state.constraints.vision_image_mime,
                )
                neighbor_data_urls["next"] = data_url
            except Exception as e:
                raise RuntimeError(f"Failed to encode next frame {i+1}: {e}") from e

        # Use vision rewriter if we have context
        final_prompt = original_prompt
        final_negative = original_negative

        if neighbor_data_urls and tools.vision_rewriter:
            from scene_agent.tools.vision_rewriter import build_neighbors_vision_messages_base64

            # Build prompt with neighbor context
            fix_desc = frame.get("fix_description", "Regenerate for consistency")

            # Use the helper to build messages
            messages = build_neighbors_vision_messages_base64(
                current_prompt=original_prompt,
                current_negative=original_negative,
                neighbor_data_urls=neighbor_data_urls,
                frame_idx=i,
                total_frames=len(frames),
                fix_description=fix_desc,
            )

            try:
                result = tools.vision_rewriter.complete_json_messages(messages)
                if result:
                    final_prompt = result.get("image_prompt", original_prompt)
                    neg = normalize_negative_items(result.get("negative", original_negative))
                    final_negative = list(dict.fromkeys([*neg, *original_negative]))

                    # Track prompt change if vision rewriter modified it
                    if final_prompt != original_prompt:
                        prompt_changes[i] = {
                            "original": original_prompt,
                            "updated": final_prompt,
                        }

                    log.info(f"Frame {i}: prompt refined by vision rewriter")
                else:
                    raise ValueError(f"Vision rewriter returned no JSON for frame {i}")

            except Exception as e:
                log.error("Vision rewriter failed for frame %s: %s", i, e)
                raise

        # Generate image
        uri = tools.image_tool.generate(
            prompt=final_prompt,
            negative_prompt=", ".join(final_negative) if final_negative else "",
            aspect_ratio=aspect_ratio,
        )

        # Update storage
        if i < len(frame_uris):
            frame_uris[i] = uri
        else:
            frame_uris.append(uri)

        # Update storyboard
        frame["keyframe_uri"] = uri
        frame["image_prompt"] = final_prompt
        if final_negative:
            frame["negative"] = final_negative

    log.info(f"Regenerated {len(regen_indices)} key frames")

    # Write updated storyboard markdown
    md_path = write_storyboard_markdown(state, tools.storage, storyboard_data, frame_uris, prompt_changes)
    if md_path:
        log.info(f"Updated director's cut saved to: {md_path}")

    return {
        "frame_uris": frame_uris,
        "storyboard_raw": storyboard_data,
        "storyboard": StoryboardData.model_validate(storyboard_data),
        "regen_frames": [],
        "prompt_changes": prompt_changes,
        "provider_metadata": {
            "keyframes_regenerate": {
                "provider": "openrouter",
                "image_model": getattr(getattr(tools.image_tool, "tool", tools.image_tool), "model", ""),
                "vision_model": getattr(tools.vision_rewriter, "vision_model", ""),
            }
        },
    }
