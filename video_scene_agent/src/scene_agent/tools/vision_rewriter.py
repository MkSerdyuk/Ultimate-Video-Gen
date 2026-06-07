"""Vision rewriter for context-aware keyframe generation."""

from __future__ import annotations

import logging
from typing import Any, Optional

import requests

from scene_agent.config import Config
from scene_agent.models import Constraints

log = logging.getLogger(__name__)


def normalize_negative_items(value: Any) -> list[str]:
    """Normalize optional negative prompt payloads returned by vision models."""
    if value is None:
        return []
    if isinstance(value, str):
        item = value.strip()
        return [item] if item else []
    if isinstance(value, (list, tuple, set)):
        items: list[str] = []
        for raw in value:
            if raw is None:
                continue
            item = str(raw).strip()
            if item:
                items.append(item)
        return items
    item = str(value).strip()
    return [item] if item else []


# System prompt for vision-based prompt rewriting
VISION_REWRITER_SYSTEM = """You are a visual consistency expert for AI image generation.

Your task is to analyze the provided keyframe images and rewrite the image prompt
to ensure PERFECT visual consistency with the reference frames.

CRITICAL CONSISTENCY REQUIREMENTS:
1. **LIGHTING**: Match exact lighting direction, intensity, shadows, highlights, time of day
2. **COLOR PALETTE**: Replicate the exact color grading, saturation levels, tone, mood
3. **STYLE & AESTHETICS**: Same art style, brush strokes, texture quality, rendering technique
4. **CHARACTER APPEARANCE**: IDENTICAL facial features, hairstyle, clothing, accessories, body proportions
5. **OBJECT PLACEMENT**: Objects must be in same relative positions, same scale, same angles
6. **ENVIRONMENT**: Same background elements, atmospheric conditions, depth of field
7. **SHOT DESIGN**: Preserve shot size, camera angle, lens language, camera support, blocking, and composition

IMPERATIVE RULES:
- COPY the visual style from context images EXACTLY
- DO NOT invent new visual elements not present in context
- DO NOT change the action, shot size, camera angle, lens, camera support, composition, or blocking - only refine visual surface description
- Keep negative prompts consistent (no text, watermarks, blur, distortion)

Return ONLY valid JSON:
{
  "image_prompt": "...",
  "negative": ["...", "..."]  // optional, merge with existing
}
"""


class VisionRewriterTool:
    """
    Tool for rewriting image prompts based on visual context from previous frames.

    Uses OpenRouter vision models with base64-encoded images for context.
    """

    def __init__(self, config: Config):
        """
        Initialize vision rewriter tool.

        Args:
            config: Configuration object
        """
        self.config = config
        self.api_key = config.openrouter_api_key
        self.base_url = config.openrouter_base_url
        self.model = config.openrouter_text_model  # Use vision-capable model

        # Vision model for image understanding (preferably different from text model)
        self.vision_model = "google/gemini-2.5-flash"  # Has vision capabilities

        log.info(f"VisionRewriterTool initialized with vision model: {self.vision_model}")

    def complete_json_messages(self, messages: list[dict], *, max_tokens: int = 500) -> dict | None:
        """Send multimodal messages to the vision model and parse JSON output."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/video-scene-agent",
            "X-OpenRouter-Title": "Video Scene Agent",
        }
        request_body = {
            "model": self.vision_model,
            "messages": [
                {"role": "system", "content": VISION_REWRITER_SYSTEM},
                *messages,
            ],
            "temperature": self.config.openrouter_temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "plugins": [{"id": "response-healing"}],
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=request_body,
            timeout=min(self.config.request_timeout, 60),
        )
        response.raise_for_status()

        data = response.json()
        if "choices" not in data or not data["choices"]:
            return None

        message = data["choices"][0].get("message", {})
        content = message.get("content", "")

        from scene_agent.utils.json_llm import clean_json_response, parse_partial_json

        cleaned = clean_json_response(content)
        return parse_partial_json(cleaned)

    def rewrite_with_context(
        self,
        current_prompt: str,
        current_negative: list[str],
        context_images: list[str],  # Base64 data URLs
        frame_info: dict,
        constraints: Constraints,
    ) -> dict:
        """
        Rewrite image prompt based on visual context from previous frames.

        Args:
            current_prompt: Current image prompt to be refined
            current_negative: Current negative prompts
            context_images: List of base64 data URLs of previous frames
            frame_info: Information about current frame (idx, timestamp, etc.)
            constraints: Generation constraints

        Returns:
            Dict with updated prompt and negative
        """
        if not context_images:
            # No context, return as-is
            return {
                "image_prompt": current_prompt,
                "negative": current_negative,
            }

        # Build the message content with text first, then images
        content = [
            {
                "type": "text",
                "text": self._build_context_prompt(
                    current_prompt=current_prompt,
                    current_negative=current_negative,
                    frame_info=frame_info,
                    num_context=len(context_images),
                )
            }
        ]

        # Add base64 images
        for i, img_data_url in enumerate(context_images):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": img_data_url
                }
            })

        try:
            log.info(f"Calling vision rewriter for frame {frame_info.get('idx', '?')}, "
                    f"context images: {len(context_images)}")
            result = self.complete_json_messages([{"role": "user", "content": content}])
            if result is None:
                raise ValueError("Failed to parse vision rewriter response")

            negative = normalize_negative_items(result.get("negative", current_negative))
            current_negative_items = normalize_negative_items(current_negative)
            if current_negative_items:
                negative = list(dict.fromkeys([*negative, *current_negative_items]))

            return {
                "image_prompt": result.get("image_prompt", current_prompt),
                "negative": negative,
            }

        except requests.RequestException as e:
            log.error(f"Vision rewriter API error: {e}")
            raise

    def _build_context_prompt(
        self,
        current_prompt: str,
        current_negative: list[str],
        frame_info: dict,
        num_context: int,
    ) -> str:
        """Build the text prompt for vision rewriter."""
        idx = frame_info.get("idx", "?")
        action = frame_info.get("action", frame_info.get("action_in_frame", ""))
        camera = frame_info.get("camera", frame_info.get("camera_movement", ""))

        negative_str = ", ".join(current_negative) if current_negative else "none"

        prompt = f"""You are refining the image prompt for frame {idx} of a video storyboard.

CURRENT PROMPT:
{current_prompt}

NEGATIVE CONSTRAINTS:
{negative_str}

FRAME INFO:
- Frame index: {idx}
- Action: {action}
- Camera: {camera}

CONTEXT:
You will see {num_context} previous keyframe(s) below. Study them EXTREMELY carefully.

MANDATORY VISUAL MATCHING - you MUST replicate:
1. **LIGHTING**: Direction of light source, shadow placement, brightness, warm/cool tones, golden hour etc.
2. **COLOR GRADING**: Exact hues, saturation levels, contrast, color temperature
3. **ART STYLE**: Brushwork, texture detail, rendering style (photorealistic/painted/cel-shaded/etc.)
4. **CHARACTER**: Every detail of face, hair, clothes, accessories - MUST be identical
5. **COMPOSITION**: Object positions, camera angle, framing, perspective
6. **ATMOSPHERE**: Fog, glow, particles, reflections, environmental effects
7. **SHOT GRAMMAR**: shot size, angle, lens, support, composition, blocking must remain intact

DO NOT invent visual elements. COPY from context images.

REWRITE the image_prompt to match the visual style from context images PERFECTLY.
Keep the action and full shot design unchanged - only enhance the visual description.

Return JSON:
{{
  "image_prompt": "refined visual description matching context frames exactly...",
  "negative": ["optional", "merged", "negatives"]
}}"""
        return prompt


def create_vision_rewriter(config: Config) -> VisionRewriterTool:
    """Factory function to create vision rewriter tool."""
    return VisionRewriterTool(config)


# Helper functions for building vision requests

def build_prev_n_vision_messages_base64(
    current_prompt: str,
    current_negative: list[str],
    context_data_urls: list[str],  # Base64 data URLs
    frame_idx: int,
    total_frames: int,
) -> list[dict]:
    """
    Build messages for vision rewriter with previous N frames as context.

    Args:
        current_prompt: Current frame's image prompt
        current_negative: Current negative prompts
        context_data_urls: Base64 data URLs of previous frames
        frame_idx: Current frame index
        total_frames: Total number of frames

    Returns:
        Messages list for API request
    """
    content = [
        {
            "type": "text",
            "text": f"""Rewrite the image prompt for frame {frame_idx}/{total_frames} to ensure PERFECT consistency with the {len(context_data_urls)} previous frame(s) shown below.

Current prompt: {current_prompt}
Negative: {', '.join(current_negative) if current_negative else 'none'}

STUDY the context images EXTREMELY carefully. You MUST replicate:

**LIGHTING**: Light direction, shadow positions, intensity, color temperature, time of day
**COLORS**: Exact color palette, saturation, contrast, grading
**STYLE**: Art technique, texture quality, rendering method, aesthetic
**CHARACTER**: IDENTICAL appearance - face, hair, clothes, accessories, proportions
**COMPOSITION**: Same object placement, camera angle, framing, perspective
**ENVIRONMENT**: Matching background, atmosphere, effects
**SHOT GRAMMAR**: Preserve shot size, lens choice, camera support, blocking, and composition exactly

DO NOT invent new visual elements. COPY what you see in context frames.
Keep the same action and shot grammar - only refine the visual surface description.

Return JSON:
{{"image_prompt": "...", "negative": ["..."]}}"""
        }
    ]

    # Add context images
    for data_url in context_data_urls:
        content.append({
            "type": "image_url",
            "image_url": {"url": data_url}
        })

    return [{"role": "user", "content": content}]


def build_neighbors_vision_messages_base64(
    current_prompt: str,
    current_negative: list[str],
    neighbor_data_urls: dict[str, str],  # {"prev": "...", "current": "...", "next": "..."}
    frame_idx: int,
    total_frames: int,
    fix_description: str = "",
) -> list[dict]:
    """
    Build messages for vision rewriter with neighbor frames during regeneration.

    Args:
        current_prompt: Current frame's image prompt
        current_negative: Current negative prompts
        neighbor_data_urls: Dict with 'prev', 'current', 'next' base64 data URLs
        frame_idx: Current frame index
        total_frames: Total number of frames
        fix_description: Optional description of what to fix

    Returns:
        Messages list for API request
    """
    context_desc = []
    if "prev" in neighbor_data_urls:
        context_desc.append("previous frame")
    if "current" in neighbor_data_urls:
        context_desc.append("current frame (what we're fixing)")
    if "next" in neighbor_data_urls:
        context_desc.append("next frame")

    context_str = ", ".join(context_desc)

    content = [
        {
            "type": "text",
            "text": f"""Regenerate frame {frame_idx}/{total_frames} with PERFECT consistency context.

Current prompt: {current_prompt}
Negative: {', '.join(current_negative) if current_negative else 'none'}

Context images show: {context_str}

STUDY these reference frames EXTREMELY carefully. You MUST replicate:

**LIGHTING**: Light source direction, shadow placement, brightness, color temperature
**COLOR PALETTE**: Exact hues, saturation, contrast, grading from reference frames
**ART STYLE**: Same technique, texture detail, rendering style, aesthetic quality
**CHARACTER**: IDENTICAL face, hair, clothes, accessories - every detail must match
**COMPOSITION**: Same object positions, angles, scale, perspective, framing
**ENVIRONMENT**: Matching background elements, atmosphere, effects
**SHOT GRAMMAR**: Keep the planned shot size, lens, support, blocking, and composition unchanged
"""
        }
    ]

    if fix_description:
        content[0]["text"] += f"\nFix goal: {fix_description}\n"

    content[0]["text"] += """DO NOT invent visual elements. COPY from reference frames.

Return JSON:
{{"image_prompt": "...", "negative": ["..."]}}"""

    # Add images in order: prev -> current -> next
    for key in ["prev", "current", "next"]:
        if key in neighbor_data_urls:
            content.append({
                "type": "image_url",
                "image_url": {"url": neighbor_data_urls[key]}
            })

    return [{"role": "user", "content": content}]
