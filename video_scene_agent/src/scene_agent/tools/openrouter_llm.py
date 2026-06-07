from __future__ import annotations
"""OpenRouter LLM client for text generation."""

import logging
from typing import Any, Optional

from openai import OpenAI, APIError, APIConnectionError, RateLimitError
from scene_agent.config import Config

log = logging.getLogger(__name__)


class OpenRouterLLM:
    """
    Client for LLM via OpenRouter API.

    Uses OpenAI SDK with OpenRouter base URL.
    """

    def __init__(self, config: Config):
        """
        Initialize OpenRouter LLM client.

        Args:
            config: Configuration object
        """
        self.config = config
        self.client = OpenAI(
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
            timeout=config.request_timeout,
            max_retries=config.max_retries,
        )
        self.default_model = config.openrouter_text_model

        log.info(f"OpenRouterLLM initialized with model: {self.default_model}")

    def _create_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ):
        """Create a chat completion with optional JSON mode controls."""
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
            kwargs["extra_body"] = {
                "plugins": [{"id": "response-healing"}],
            }
        return self.client.chat.completions.create(**kwargs)

    def _resolve_temperature(self, temperature: float | None) -> float:
        """Keep text/review planning deterministic even if callers pass legacy values."""
        return self.config.openrouter_temperature

    def chat(
        self,
        user: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        """
        Generate text response.

        Args:
            user: User message content
            system: Optional system message
            model: Model to use (defaults to config default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            json_mode: If True, ensure higher max_tokens for JSON responses

        Returns:
            Generated text response
        """
        model = model or self.default_model

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        # Don't limit max_tokens - let model generate full response
        # Only use explicitly provided max_tokens, otherwise None
        # max_tokens = max_tokens  # No default limit

        try:
            log.info(f"Sending request to {model}, user prompt length: {len(user)}")

            response = self._create_completion(
                model=model,
                messages=messages,
                temperature=self._resolve_temperature(temperature),
                max_tokens=max_tokens,
                json_mode=json_mode,
            )

            content = response.choices[0].message.content

            if not content:
                log.error(f"LLM returned empty response! Model: {model}")
                log.debug(f"Full response: {response}")
                raise ValueError(f"LLM returned empty response from model {model}. "
                               f"The model may not support this format or be unavailable.")

            # Check for potential truncation
            finish_reason = response.choices[0].finish_reason
            if finish_reason == "length":
                log.warning(f"Response truncated due to max_tokens ({max_tokens}). Consider increasing limit.")

            log.info(f"Received response, length: {len(content)}, finish_reason: {finish_reason}")

            return content

        except RateLimitError as e:
            log.error(f"Rate limit error: {e}")
            raise
        except APIConnectionError as e:
            log.error(f"Connection error: {e}")
            raise
        except APIError as e:
            log.error(f"API error: {e}")
            raise

    def chat_with_retry(
        self,
        user: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        retries: int = 2,
    ) -> str:
        """
        Generate text response with retry for incomplete JSON.

        If JSON response is incomplete, asks the model to continue.

        Args:
            user: User message content
            system: Optional system message
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            json_mode: If True, use higher max_tokens
            retries: Number of retries for incomplete JSON

        Returns:
            Generated text response
        """
        model = model or self.default_model
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        # Don't limit max_tokens - let model generate full response
        # Only use explicitly provided max_tokens, otherwise None
        if max_tokens is None:
            max_tokens = None  # No default limit

        accumulated_content = ""

        for attempt in range(retries + 1):
            try:
                log.info(f"Request to {model}, attempt {attempt + 1}/{retries + 1}")

                response = self._create_completion(
                    model=model,
                    messages=messages,
                    temperature=self._resolve_temperature(temperature),
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                )

                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason

                if not content:
                    raise ValueError(f"LLM returned empty response")

                candidate_content = f"{accumulated_content}{content}"

                # For JSON mode, check if response is complete
                if json_mode and attempt < retries:
                    from scene_agent.utils.json_llm import clean_json_response, parse_partial_json

                    cleaned = clean_json_response(candidate_content)
                    parsed = parse_partial_json(cleaned)

                    content_str = str(cleaned)
                    incomplete_indicators = [
                        finish_reason == "length",
                        content_str.count('"') % 2 == 1,
                        content_str.rstrip().endswith('": "'),
                        content_str.rstrip().endswith('{\n'),
                        content_str.rstrip().endswith(","),
                    ]

                    if '"frames": []' in content_str and "segments" not in content_str:
                        incomplete_indicators.append(True)

                    needs_continuation = parsed is None or any(incomplete_indicators)
                    if needs_continuation:
                        log.warning("Response appears incomplete, asking for continuation...")
                        accumulated_content = candidate_content
                        messages.append({"role": "assistant", "content": content})
                        messages.append({
                            "role": "user",
                            "content": (
                                "Continue the JSON from where you left off. "
                                "Do not repeat previous content. Return only the remaining JSON fragment."
                            ),
                        })
                        continue

                log.info(
                    f"Response length: {len(candidate_content)}, finish_reason: {finish_reason}"
                )
                return candidate_content

            except Exception as e:
                if attempt < retries:
                    log.warning(f"Attempt {attempt + 1} failed: {e}, retrying...")
                    continue
                raise

        raise ValueError(f"Failed after {retries + 1} attempts")

    def chat_with_history(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        """
        Generate text response with conversation history.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        model = model or self.default_model

        try:
            response = self._create_completion(
                model=model,
                messages=list(messages),
                temperature=self._resolve_temperature(temperature),
                max_tokens=max_tokens,
                json_mode=json_mode,
            )

            return response.choices[0].message.content or ""

        except (RateLimitError, APIConnectionError, APIError) as e:
            log.error(f"API error in chat_with_history: {e}")
            raise

    def stream_chat(
        self,
        user: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ):
        """
        Stream chat response.

        Args:
            user: User message content
            system: Optional system message
            model: Model to use
            temperature: Sampling temperature

        Yields:
            Chunks of generated text
        """
        model = model or self.default_model

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=self._resolve_temperature(temperature),
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except (RateLimitError, APIConnectionError, APIError) as e:
            log.error(f"API error in stream_chat: {e}")
            raise


def create_llm(config: Config) -> OpenRouterLLM:
    """Factory function to create OpenRouter LLM client."""
    return OpenRouterLLM(config)
