from __future__ import annotations
"""Logging utilities for scene agent."""

import logging
import sys
from typing import Optional


def setup_logging(
    level: int | str = logging.INFO,
    format_string: Optional[str] = None,
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string
    """
    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"
        )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in root_logger.handlers:
        if getattr(handler, "_scene_agent_handler", False):
            handler.setLevel(level)
            handler.setFormatter(logging.Formatter(format_string))
            break
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler._scene_agent_handler = True
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(format_string))
        root_logger.addHandler(handler)

    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)
