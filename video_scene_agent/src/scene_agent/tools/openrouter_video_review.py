from __future__ import annotations
"""Video Review Tool using Gemini 2.5 Flash via OpenRouter."""

import base64
import logging
import mimetypes
from pathlib import Path

from openai import OpenAI, APIError
from scene_agent.config import Config
from scene_agent.tools.storage import StorageBackend

log = logging.getLogger(__name__)


class OpenRouterVideoReviewTool:
    """
    Video review using Gemini 2.5 Flash multimodal capabilities.

    Can analyze video files and provide textual feedback.
    """

    def __init__(self, config: Config, storage: StorageBackend):
        """
        Initialize video review tool.

        Args:
            config: Configuration object
            storage: Storage backend for reading videos
        """
        self.config = config
        self.storage = storage
        self.client = OpenAI(
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
            timeout=config.request_timeout,
        )
        self.model = config.openrouter_video_model

        log.info(f"OpenRouterVideoReviewTool initialized with model: {self.model}")

    @staticmethod
    def _video_part(url: str) -> dict:
        """Build an OpenAI-SDK-compatible OpenRouter video input content part."""
        return {
            "type": "video_url",
            "video_url": {
                "url": url,
            },
        }

    def review(
        self,
        system: str,
        user_text: str,
        video_uri: str,
    ) -> str:
        """
        Review a video file.

        Args:
            system: System prompt for review
            user_text: User's question/context
            video_uri: URI of the video to review

        Returns:
            Review text response
        """
        # Prepare video content
        video_content = self._prepare_video_content(video_uri)

        # Build messages
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    *video_content,
                ]
            }
        ]

        try:
            log.debug(f"Reviewing video: {video_uri}")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.config.openrouter_temperature,
            )

            content = response.choices[0].message.content or ""
            log.debug(f"Review response length: {len(content)}")

            return content

        except APIError as e:
            log.error(f"API error in video review: {e}")
            raise

    def review_multiple(
        self,
        system: str,
        user_text: str,
        video_uris: list[str],
    ) -> str:
        """
        Review multiple videos.

        Args:
            system: System prompt for review
            user_text: User's question/context
            video_uris: List of video URIs to review

        Returns:
            Combined review text
        """
        all_video_content = []
        for uri in video_uris:
            content = self._prepare_video_content(uri)
            all_video_content.extend(content)

        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    *all_video_content,
                ]
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.config.openrouter_temperature,
            )

            return response.choices[0].message.content or ""

        except APIError as e:
            log.error(f"API error in multiple video review: {e}")
            raise

    def _prepare_video_content(self, video_uri: str) -> list[dict]:
        """
        Prepare video content for API request.

        Args:
            video_uri: URI of the video

        Returns:
            List of content items for the API
        """
        # Check if it's an HTTP URL (can pass directly)
        if video_uri.startswith("http://") or video_uri.startswith("https://"):
            return [self._video_part(video_uri)]

        # For local files, read and encode as base64
        local_path = self.storage.uri_to_local_path(video_uri)
        path = Path(local_path)

        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {local_path}")

        # Read file
        with open(path, "rb") as f:
            video_data = f.read()

        # Determine content type
        content_type, _ = mimetypes.guess_type(str(path))
        if not content_type:
            content_type = "video/mp4"

        # Encode as base64
        b64_data = base64.b64encode(video_data).decode("ascii")
        data_url = f"data:{content_type};base64,{b64_data}"

        return [self._video_part(data_url)]

    def _get_public_url(self, uri: str) -> str:
        """Get a public URL for the video URI if possible."""
        if uri.startswith("http://") or uri.startswith("https://"):
            return uri

        # Try to use storage's public URL
        if hasattr(self.storage, "get_public_uri"):
            # Extract key from URI
            local_path = self.storage.uri_to_local_path(uri)
            if hasattr(self.storage, "base_path"):
                key = str(Path(local_path).relative_to(self.storage.base_path))
                return self.storage.get_public_uri(key)

        return uri


def create_video_review_tool(config: Config, storage: StorageBackend) -> OpenRouterVideoReviewTool:
    """Factory function to create video review tool."""
    return OpenRouterVideoReviewTool(config, storage)
