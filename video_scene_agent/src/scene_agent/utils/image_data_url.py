"""Utilities for encoding images to base64 data URLs for vision models.

This module provides functions to convert image URIs to base64-encoded data URLs
that can be embedded in OpenRouter API requests for vision models.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import mimetypes
from pathlib import Path
from typing import Optional, Tuple
from functools import lru_cache

log = logging.getLogger(__name__)


# Cache for encoded data URLs to avoid re-encoding the same image
_data_url_cache: dict[str, str] = {}


def to_data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    """
    Convert image bytes to base64 data URL.

    Args:
        image_bytes: Raw image bytes
        mime: MIME type (e.g., "image/jpeg", "image/png")

    Returns:
        Data URL string like "data:image/jpeg;base64,..."
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def guess_mime_from_uri(uri: str, default: str = "image/jpeg") -> str:
    """
    Guess MIME type from file URI.

    Args:
        uri: Image URI
        default: Default MIME type if guess fails

    Returns:
        MIME type string
    """
    mime, _ = mimetypes.guess_type(uri)
    return mime or default


def clear_cache():
    """Clear the data URL cache."""
    global _data_url_cache
    _data_url_cache.clear()
    log.debug("Data URL cache cleared")


def get_cache_size() -> int:
    """Get the current cache size."""
    return len(_data_url_cache)


def encode_uri_to_data_url(
    storage,
    uri: str,
    max_side_px: int = 768,
    jpeg_quality: int = 80,
    target_mime: str = "image/jpeg",
    use_cache: bool = True,
) -> str:
    """
    Convert image URI to base64 data URL with optional resizing/compression.

    This function:
    1. Downloads the image from storage
    2. Resizes if needed (preserving aspect ratio)
    3. Compresses to JPEG if needed
    4. Encodes to base64 data URL
    5. Caches the result

    Args:
        storage: Storage backend instance
        uri: Image URI to convert
        max_side_px: Maximum dimension for resizing
        jpeg_quality: JPEG quality (1-100) for compression
        target_mime: Target MIME type for output
        use_cache: Whether to use cached result

    Returns:
        Base64 data URL string
    """
    if use_cache and uri in _data_url_cache:
        log.debug(f"Cache hit for URI: {uri}")
        return _data_url_cache[uri]

    # Download image bytes from storage
    try:
        image_bytes = storage.get_bytes(uri)
    except Exception as e:
        log.error(f"Failed to get bytes from storage for {uri}: {e}")
        raise

    # Process the image
    result = _process_image_to_data_url(
        image_bytes=image_bytes,
        max_side_px=max_side_px,
        jpeg_quality=jpeg_quality,
        target_mime=target_mime,
        cache_key=uri if use_cache else None,
    )

    return result


def _process_image_to_data_url(
    image_bytes: bytes,
    max_side_px: int = 768,
    jpeg_quality: int = 80,
    target_mime: str = "image/jpeg",
    cache_key: Optional[str] = None,
) -> str:
    """
    Process image bytes to data URL with resizing and compression.

    Args:
        image_bytes: Raw image bytes
        max_side_px: Maximum dimension for resizing
        jpeg_quality: JPEG quality (1-100)
        target_mime: Target MIME type
        cache_key: Optional key for caching

    Returns:
        Base64 data URL string
    """
    try:
        # Try to use PIL for processing
        from PIL import Image as PILImage
        from io import BytesIO

        # Open image
        img = PILImage.open(BytesIO(image_bytes))

        # Get original format
        original_format = img.format or "JPEG"

        # Resize if needed
        width, height = img.size
        if max(width, height) > max_side_px:
            # Calculate new dimensions preserving aspect ratio
            if width > height:
                new_width = max_side_px
                new_height = int(height * max_side_px / width)
            else:
                new_height = max_side_px
                new_width = int(width * max_side_px / height)

            img = img.resize((new_width, new_height), PILImage.LANCZOS)
            log.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")

        # Convert to target format with compression
        output = BytesIO()

        if target_mime == "image/jpeg":
            img.convert("RGB").save(output, format="JPEG", quality=jpeg_quality, optimize=True)
        elif target_mime == "image/png":
            img.save(output, format="PNG", optimize=True)
        elif target_mime == "image/webp":
            img.save(output, format="WEBP", quality=jpeg_quality)
        else:
            raise ValueError(f"Unsupported target image MIME type: {target_mime}")

        processed_bytes = output.getvalue()

    except ImportError:
        raise RuntimeError("PIL is required for image data URL preparation")
    except Exception as e:
        raise RuntimeError(f"Image processing failed: {e}") from e

    # Encode to base64 data URL
    data_url = to_data_url(processed_bytes, target_mime)

    # Cache if key provided
    if cache_key:
        _data_url_cache[cache_key] = data_url

    log.debug(f"Encoded image to data URL, size: {len(data_url)} chars (truncated)")

    return data_url


def encode_multiple_uris_to_data_urls(
    storage,
    uris: list[str],
    max_side_px: int = 768,
    jpeg_quality: int = 80,
    target_mime: str = "image/jpeg",
    limit: int = 6,
) -> list[str]:
    """
    Convert multiple image URIs to base64 data URLs.

    Args:
        storage: Storage backend instance
        uris: List of image URIs to convert
        max_side_px: Maximum dimension for resizing
        jpeg_quality: JPEG quality
        target_mime: Target MIME type
        limit: Maximum number of images to process

    Returns:
        List of base64 data URL strings (up to `limit` items)
    """
    if limit:
        uris = uris[:limit]

    result = []
    for uri in uris:
        try:
            data_url = encode_uri_to_data_url(
                storage=storage,
                uri=uri,
                max_side_px=max_side_px,
                jpeg_quality=jpeg_quality,
                target_mime=target_mime,
            )
            result.append(data_url)
        except Exception as e:
            log.warning(f"Failed to encode URI {uri}: {e}")
            # Continue with other images
            continue

    return result


def get_image_dimensions(storage, uri: str) -> Tuple[int, int]:
    """
    Get image dimensions from URI.

    Args:
        storage: Storage backend instance
        uri: Image URI

    Returns:
        Tuple of (width, height)
    """
    try:
        from PIL import Image as PILImage
        from io import BytesIO

        image_bytes = storage.get_bytes(uri)
        img = PILImage.open(BytesIO(image_bytes))
        return img.size
    except ImportError:
        log.warning("PIL not available, cannot get image dimensions")
        return (0, 0)
    except Exception as e:
        log.warning(f"Failed to get image dimensions: {e}")
        return (0, 0)
