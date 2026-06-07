from __future__ import annotations
"""Aspect-ratio helpers shared by image and video providers."""

SUPPORTED_ASPECT_RATIOS = {"16:9", "9:16", "1:1", "4:3", "3:4", "21:9"}
DEFAULT_ASPECT_RATIO = "16:9"


def normalize_aspect_ratio(value: str | None, *, default: str = DEFAULT_ASPECT_RATIO) -> str:
    """Return a provider-supported aspect ratio, falling back consistently."""
    ratio = str(value or "").strip()
    return ratio if ratio in SUPPORTED_ASPECT_RATIOS else default
