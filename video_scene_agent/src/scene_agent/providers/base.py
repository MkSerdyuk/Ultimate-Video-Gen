from __future__ import annotations
"""Provider protocols and normalized provider errors."""

from typing import Protocol, runtime_checkable


class ProviderError(RuntimeError):
    """Base error type for provider adapters."""

    def __init__(self, provider: str, operation: str, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.provider = provider
        self.operation = operation
        self.retryable = retryable


class TransientProviderError(ProviderError):
    """Provider error that may succeed on retry."""

    def __init__(self, provider: str, operation: str, message: str) -> None:
        super().__init__(provider, operation, message, retryable=True)


class PermanentProviderError(ProviderError):
    """Provider error that should fail fast."""

    def __init__(self, provider: str, operation: str, message: str) -> None:
        super().__init__(provider, operation, message, retryable=False)


@runtime_checkable
class TextLLMProvider(Protocol):
    """Text generation provider used by director/editor tasks."""

    def chat(self, user: str, system: str | None = None, **kwargs) -> str:
        ...

    def chat_with_retry(self, user: str, system: str | None = None, **kwargs) -> str:
        ...

    def chat_with_history(self, messages: list[dict], **kwargs) -> str:
        ...


@runtime_checkable
class ImageProvider(Protocol):
    """Image generation provider used by keyframe tasks."""

    def generate(self, prompt: str, negative_prompt: str = "", aspect_ratio: str = "16:9") -> str:
        ...


@runtime_checkable
class VideoReviewProvider(Protocol):
    """Video review provider used by final QC tasks."""

    def review(self, system: str, user_text: str, video_uri: str) -> str:
        ...


@runtime_checkable
class VideoGenerationProvider(Protocol):
    """Video generation provider used by segment tasks."""

    def generate_multiple_segments(self, specs: list[dict]) -> list[str]:
        ...

    def edit_multiple_segments(self, specs: list[dict]) -> list[str]:
        ...
