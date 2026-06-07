from __future__ import annotations
"""OpenRouter image generation tool using multimodal chat completions."""

import base64
import logging
from typing import Optional

import requests
from openai import OpenAI, APIError
from scene_agent.config import Config
from scene_agent.tools.storage import StorageBackend, decode_base64_image
from scene_agent.utils.aspect_ratio import normalize_aspect_ratio

log = logging.getLogger(__name__)


class OpenRouterImageTool:
    """
    Image generation via OpenRouter using Gemini 2.5 Flash Image Preview.

    Uses the multimodal API with image generation capabilities.
    """

    def __init__(self, config: Config, storage: StorageBackend):
        """
        Initialize image generation tool.

        Args:
            config: Configuration object
            storage: Storage backend for generated images
        """
        self.config = config
        self.storage = storage
        self.client = OpenAI(
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
            timeout=config.request_timeout,
        )
        self.model = config.openrouter_image_model
        self.api_key = config.openrouter_api_key
        self.base_url = config.openrouter_base_url

        log.info(f"OpenRouterImageTool initialized with model: {self.model}")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        aspect_ratio: str = "16:9",
    ) -> str:
        """
        Generate an image from text prompt.

        Args:
            prompt: Image generation prompt
            negative_prompt: Negative prompt (what to avoid)
            aspect_ratio: Aspect ratio (e.g., "16:9", "9:16", "1:1")

        Returns:
            URI of the generated image in storage
        """
        gemini_aspect = normalize_aspect_ratio(aspect_ratio)

        image_model = self.model

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/video-scene-agent",
            "X-OpenRouter-Title": "Video Scene Agent",
        }

        content = prompt
        if negative_prompt.strip():
            content = (
                f"{prompt}\n\n"
                "Avoid the following in the generated image:\n"
                f"{negative_prompt.strip()}"
            )

        # Build request body per OpenRouter image generation docs
        request_body = {
            "model": image_model,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            # BOTH image and text modalities are required
            "modalities": ["image", "text"],
            "temperature": self.config.openrouter_temperature,
            # Gemini-specific image config
            "image_config": {
                "aspect_ratio": gemini_aspect,
            }
        }

        try:
            log.info(f"Generating image with model: {image_model}, aspect: {gemini_aspect}")
            log.debug(f"Request prompt: {prompt[:200]}...")

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=request_body,
                timeout=self.config.request_timeout,
            )
            response.raise_for_status()

            data = response.json()

            # Log response structure for debugging
            log.debug(f"Response keys: {list(data.keys())}")

            if "error" in data:
                log.error(f"API error: {data['error']}")
                raise ValueError(f"API error: {data['error']}")

            # Extract image from response per OpenRouter docs format
            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                message = choice.get("message", {})

                log.debug(f"Message keys: {list(message.keys())}")

                # Check for images array (OpenRouter format)
                images = message.get("images", [])
                if images:
                    for image in images:
                        image_url_obj = image.get("image_url") or image.get("imageUrl") or {}
                        url = image_url_obj.get("url", "")
                        if url:
                            log.info(f"Got image from API, data URL length: {len(url)}")
                            return self._handle_image_response(url)

                # Check alternate content shape - sometimes images come in content field
                content = message.get("content", "")
                if content:
                    log.debug(f"Content field present, type: {type(content)}, length: {len(str(content))}")
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "image_url":
                                image_url_obj = item.get("image_url") or item.get("imageUrl") or {}
                                url = image_url_obj.get("url", "")
                                if url:
                                    log.info(f"Found image in content list, URL length: {len(url)}")
                                    return self._handle_image_response(url)
                    # Also check for nested images in content items
                    for item in content:
                        if isinstance(item, dict):
                            images_nested = item.get("images", [])
                            for img_nested in images_nested:
                                image_url_obj = (
                                    img_nested.get("image_url") or img_nested.get("imageUrl") or {}
                                )
                                url = image_url_obj.get("url", "")
                                if url:
                                    log.info(f"Found nested image in content, URL length: {len(url)}")
                                    return self._handle_image_response(url)

            # If we get here, no image was found
            log.error(f"No image data in response. Full response: {data}")
            # Log the message content for debugging
            if "choices" in data and len(data["choices"]) > 0:
                message = data["choices"][0].get("message", {})
                content = message.get("content", "")
                log.error(f"Message content type: {type(content)}, length: {len(str(content))}")
                if isinstance(content, str):
                    log.error(f"Message content (first 500 chars): {content[:500]}")
                elif isinstance(content, list) and content:
                    log.error(f"Message content items: {len(content)} items")
                    for i, item in enumerate(content[:3]):
                        log.error(f"  Item {i}: {str(item)[:200]}")
            raise ValueError(f"No image data in response from model {image_model}")

        except requests.RequestException as e:
            log.error(f"API error generating image: {e}")
            if hasattr(e, 'response') and e.response is not None:
                log.error(f"Response status: {e.response.status_code}")
                log.error(f"Response body: {e.response.text[:1000]}")
            raise
        except Exception as e:
            log.error(f"Error generating image: {e}")
            raise

    def _convert_aspect_ratio(self, ratio: str) -> str:
        """Convert aspect ratio to OpenRouter format."""
        # OpenRouter uses different format
        mapping = {
            "16:9": "16:9",
            "9:16": "9:16",
            "1:1": "1:1",
            "4:3": "4:3",
            "3:4": "3:4",
            "21:9": "21:9",
        }
        return mapping.get(ratio, "16:9")

    def _extract_image_from_content(self, content: str) -> str:
        """Extract image URL or base64 from content."""
        import re

        # Check for markdown image syntax
        url_match = re.search(r'!\[.*?\]\((https?://[^\)]+)\)', content)
        if url_match:
            return self._download_and_store(url_match.group(1))

        # Check for base64 data URL
        if content.startswith("data:image"):
            return self._handle_base64_response(content)

        # Look for raw URL
        url_match = re.search(r'(https?://[^\s]+\.(?:png|jpg|jpeg|webp))', content)
        if url_match:
            return self._download_and_store(url_match.group(1))

        raise ValueError(f"Could not extract image from content: {content[:200]}...")

    def _get_image_size(self, aspect_ratio: str) -> str:
        """Get appropriate image size for aspect ratio."""
        sizes = {
            "16:9": "1344x768",
            "9:16": "768x1344",
            "1:1": "1024x1024",
            "4:3": "1024x768",
            "3:4": "768x1024",
            "21:9": "1536x640",
        }
        return sizes.get(aspect_ratio, "1024x1024")

    def _handle_image_response(self, url_or_b64: str) -> str:
        """
        Handle image response (URL or base64).

        Args:
            url_or_b64: Either a URL or base64 data URL

        Returns:
            Storage URI
        """
        if url_or_b64.startswith("data:"):
            return self._handle_base64_response(url_or_b64)
        elif url_or_b64.startswith("http"):
            return self._download_and_store(url_or_b64)
        else:
            # Assume it's base64 without data URL prefix
            return self._handle_raw_base64(url_or_b64)

    def _handle_base64_response(self, data_url: str) -> str:
        """Decode base64 data URL and store."""
        try:
            image_bytes, content_type = decode_base64_image(data_url)
            uri = self.storage.put_bytes(image_bytes, content_type)
            log.debug(f"Stored generated image: {uri}")
            return uri
        except Exception as e:
            log.error(f"Error handling base64 response: {e}")
            raise

    def _handle_raw_base64(self, b64_data: str) -> str:
        """Handle raw base64 string without data URL prefix."""
        try:
            # Add padding if needed
            padding = 4 - len(b64_data) % 4
            if padding != 4:
                b64_data += "=" * padding

            image_bytes = base64.b64decode(b64_data)
            uri = self.storage.put_bytes(image_bytes, "image/png")
            log.debug(f"Stored generated image from raw base64")
            return uri
        except Exception as e:
            log.error(f"Error handling raw base64: {e}")
            raise

    def _download_and_store(self, url: str) -> str:
        """Download image from URL and store."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "image/png")
            uri = self.storage.put_bytes(response.content, content_type)
            log.debug(f"Downloaded and stored image from URL")
            return uri
        except Exception as e:
            log.error(f"Error downloading image: {e}")
            raise

    def generate_multiple(
        self,
        prompts: list[str],
        negative_prompt: str = "",
        aspect_ratio: str = "16:9",
    ) -> list[str]:
        """
        Generate multiple images.

        Args:
            prompts: List of image generation prompts
            negative_prompt: Common negative prompt
            aspect_ratio: Aspect ratio for all images

        Returns:
            List of URIs
        """
        uris = []
        for i, prompt in enumerate(prompts):
            log.info(f"Generating image {i + 1}/{len(prompts)}")
            uri = self.generate(prompt, negative_prompt, aspect_ratio)
            uris.append(uri)
        return uris


def create_image_tool(config: Config, storage: StorageBackend) -> OpenRouterImageTool:
    """Factory function to create image generation tool."""
    return OpenRouterImageTool(config, storage)
