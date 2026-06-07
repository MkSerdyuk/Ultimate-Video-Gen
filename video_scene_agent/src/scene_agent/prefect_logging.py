from __future__ import annotations
"""Prefect-native logging helpers for flow and task execution."""

from typing import Any

from prefect.logging import get_run_logger
from prefect.logging.configuration import setup_logging as ensure_logging_setup


def configure_prefect_logging() -> None:
    """Ensure Prefect logging handlers are configured in the current process."""
    ensure_logging_setup()


def get_prefect_logger(**context: Any):
    """Return a Prefect run logger enriched with stable context."""
    configure_prefect_logging()
    return get_run_logger(**context)
