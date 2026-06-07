from __future__ import annotations
"""Tools for external API integrations."""

from scene_agent.tools.storage import (
    StorageBackend,
    LocalStorageBackend,
    decode_base64_image,
    encode_base64,
)
from scene_agent.tools.openrouter_llm import OpenRouterLLM, create_llm
from scene_agent.tools.openrouter_image import OpenRouterImageTool, create_image_tool
from scene_agent.tools.openrouter_video_review import (
    OpenRouterVideoReviewTool,
    create_video_review_tool,
)
from scene_agent.tools.kling import KlingTool, create_kling_tool
from scene_agent.tools.stitch import StitchTool, create_stitch_tool
from scene_agent.tools.tmpfiles import TmpfilesMediaPublisher

__all__ = [
    "StorageBackend",
    "LocalStorageBackend",
    "decode_base64_image",
    "encode_base64",
    "OpenRouterLLM",
    "create_llm",
    "OpenRouterImageTool",
    "create_image_tool",
    "OpenRouterVideoReviewTool",
    "create_video_review_tool",
    "KlingTool",
    "create_kling_tool",
    "StitchTool",
    "create_stitch_tool",
    "TmpfilesMediaPublisher",
]
