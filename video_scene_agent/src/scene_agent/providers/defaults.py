from __future__ import annotations
"""Default provider adapters wrapping the current tool implementations."""

import logging

from openai import APIConnectionError, APIError, RateLimitError
from requests import HTTPError, RequestException

from scene_agent.providers.base import (
    PermanentProviderError,
    TransientProviderError,
)

log = logging.getLogger(__name__)


class OpenRouterTextAdapter:
    """Retry-aware wrapper around the OpenRouter LLM tool."""

    def __init__(self, tool) -> None:
        self.tool = tool

    def chat(self, user: str, system: str | None = None, **kwargs) -> str:
        try:
            return self.tool.chat(user=user, system=system, **kwargs)
        except (RateLimitError, APIConnectionError) as exc:
            raise TransientProviderError("openrouter", "chat", str(exc)) from exc
        except APIError as exc:
            raise PermanentProviderError("openrouter", "chat", str(exc)) from exc
        except Exception as exc:
            raise PermanentProviderError("openrouter", "chat", str(exc)) from exc

    def chat_with_retry(self, user: str, system: str | None = None, **kwargs) -> str:
        try:
            return self.tool.chat_with_retry(user=user, system=system, **kwargs)
        except (RateLimitError, APIConnectionError) as exc:
            raise TransientProviderError("openrouter", "chat_with_retry", str(exc)) from exc
        except APIError as exc:
            raise PermanentProviderError("openrouter", "chat_with_retry", str(exc)) from exc
        except Exception as exc:
            raise PermanentProviderError("openrouter", "chat_with_retry", str(exc)) from exc

    def chat_with_history(self, messages: list[dict], **kwargs) -> str:
        try:
            return self.tool.chat_with_history(messages=messages, **kwargs)
        except (RateLimitError, APIConnectionError) as exc:
            raise TransientProviderError("openrouter", "chat_with_history", str(exc)) from exc
        except APIError as exc:
            raise PermanentProviderError("openrouter", "chat_with_history", str(exc)) from exc
        except Exception as exc:
            raise PermanentProviderError("openrouter", "chat_with_history", str(exc)) from exc


class OpenRouterImageAdapter:
    """Retry-aware wrapper around the OpenRouter image generation tool."""

    def __init__(self, tool) -> None:
        self.tool = tool

    def generate(self, prompt: str, negative_prompt: str = "", aspect_ratio: str = "16:9") -> str:
        try:
            return self.tool.generate(prompt=prompt, negative_prompt=negative_prompt, aspect_ratio=aspect_ratio)
        except (RequestException, HTTPError) as exc:
            raise TransientProviderError("openrouter", "image_generate", str(exc)) from exc
        except Exception as exc:
            raise PermanentProviderError("openrouter", "image_generate", str(exc)) from exc


class OpenRouterVideoReviewAdapter:
    """Retry-aware wrapper around the OpenRouter video review tool."""

    def __init__(self, tool) -> None:
        self.tool = tool

    def review(self, system: str, user_text: str, video_uri: str) -> str:
        try:
            return self.tool.review(system=system, user_text=user_text, video_uri=video_uri)
        except (RequestException, HTTPError, APIConnectionError, RateLimitError) as exc:
            raise TransientProviderError("openrouter", "video_review", str(exc)) from exc
        except APIError as exc:
            raise PermanentProviderError("openrouter", "video_review", str(exc)) from exc
        except Exception as exc:
            raise PermanentProviderError("openrouter", "video_review", str(exc)) from exc


def _is_retryable_video_error(exc: Exception) -> bool:
    """Classify common video-provider/client/network failures."""
    if isinstance(exc, (RequestException, HTTPError, TimeoutError)):
        return True

    status_code = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True

    text = str(exc).lower()
    retryable_markers = (
        "timeout",
        "timed out",
        "rate limit",
        "temporarily",
        "temporary",
        "connection",
        "server error",
        "service unavailable",
        "too many requests",
    )
    return any(marker in text for marker in retryable_markers)


class KlingVideoAdapter:
    """Retry-aware wrapper around the direct Kling video generation/editing tool."""

    def __init__(self, tool) -> None:
        self.tool = tool

    def generate_multiple_segments(self, specs: list[dict]) -> list[str]:
        try:
            return self.tool.generate_multiple_segments(specs)
        except Exception as exc:
            if _is_retryable_video_error(exc):
                raise TransientProviderError("kling", "generate_multiple_segments", str(exc)) from exc
            raise PermanentProviderError("kling", "generate_multiple_segments", str(exc)) from exc

    def edit_multiple_segments(self, specs: list[dict]) -> list[str]:
        try:
            return self.tool.edit_multiple_segments(specs)
        except Exception as exc:
            if _is_retryable_video_error(exc):
                raise TransientProviderError("kling", "edit_multiple_segments", str(exc)) from exc
            raise PermanentProviderError("kling", "edit_multiple_segments", str(exc)) from exc
