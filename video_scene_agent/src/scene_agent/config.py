from __future__ import annotations
"""Configuration management for scene agent."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    """Configuration for video scene generation."""

    # OpenRouter
    openrouter_api_key: str
    openrouter_text_model: str = "google/gemini-2.5-flash"
    openrouter_image_model: str = "google/gemini-2.5-flash-image"
    openrouter_video_model: str = "google/gemini-2.5-flash"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_temperature: float = 0.0

    # Kling 3.0 video generation/editing
    kling_access_key: str = ""
    kling_secret_key: str = ""
    kling_api_base: str = "https://api-singapore.klingai.com"
    kling_video_model: str = "kling-v3-omni"
    kling_mode: str = "std"
    kling_sound: str = "off"
    kling_poll_timeout_sec: int = 900
    kling_poll_request_timeout_sec: int = 30
    kling_poll_interval_sec: float = 2.0
    kling_media_public_url_base: Optional[str] = None
    kling_run_token_limit: float = 60.0
    kling_generation_tokens_per_second: float = 0.6
    kling_edit_tokens_per_second: float = 0.9
    kling_use_tmpfiles: bool = True
    kling_tmpfiles_upload_url: str = "https://tmpfiles.org/api/v1/upload"
    kling_tmpfiles_ttl_sec: int = 172800
    kling_tmpfiles_max_bytes: int = 100_000_000
    kling_tmpfiles_timeout_sec: int = 120

    # Storage
    storage_path: str = "./artifacts"
    storage_public_url_base: Optional[str] = None

    # Prefect runtime
    prefect_api_url: str = ""
    prefect_work_pool: str = "video-scene-agent"
    prefect_docker_image: str = "video-scene-agent:latest"

    # Defaults
    default_aspect_ratio: str = "16:9"
    default_fps: int = 24
    default_K_sb: int = 3
    default_K_vid: int = 2

    # Advanced
    image_size: str = "1024x1024"  # For image generation
    request_timeout: int = 120  # seconds
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> "Config":
        """Create Config from environment variables."""
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is required. "
                "Create a .env file or set it in your environment."
            )

        artifacts_root = (
            os.getenv("SCENE_AGENT_ARTIFACTS_ROOT")
            or os.getenv("STORAGE_PATH")
            or "./artifacts"
        )

        return cls(
            openrouter_api_key=api_key,
            kling_access_key=os.getenv("KLING_ACCESS_KEY", ""),
            kling_secret_key=os.getenv("KLING_SECRET_KEY", ""),
            kling_api_base=os.getenv("KLING_API_BASE", "https://api-singapore.klingai.com"),
            kling_video_model=os.getenv("KLING_VIDEO_MODEL", "kling-v3-omni"),
            kling_mode=os.getenv("KLING_MODE", "std"),
            kling_sound=os.getenv("KLING_SOUND", "off"),
            kling_poll_timeout_sec=int(os.getenv("KLING_POLL_TIMEOUT_SEC", "900")),
            kling_poll_request_timeout_sec=int(os.getenv("KLING_POLL_REQUEST_TIMEOUT_SEC", "30")),
            kling_poll_interval_sec=float(os.getenv("KLING_POLL_INTERVAL_SEC", "2.0")),
            kling_media_public_url_base=os.getenv("KLING_MEDIA_PUBLIC_URL_BASE"),
            kling_run_token_limit=float(os.getenv("KLING_RUN_TOKEN_LIMIT", "60")),
            kling_generation_tokens_per_second=float(
                os.getenv("KLING_GENERATION_TOKENS_PER_SECOND", "0.6")
            ),
            kling_edit_tokens_per_second=float(os.getenv("KLING_EDIT_TOKENS_PER_SECOND", "0.9")),
            kling_use_tmpfiles=_env_bool("KLING_USE_TMPFILES", True),
            kling_tmpfiles_upload_url=os.getenv(
                "KLING_TMPFILES_UPLOAD_URL",
                "https://tmpfiles.org/api/v1/upload",
            ),
            kling_tmpfiles_ttl_sec=int(os.getenv("KLING_TMPFILES_TTL_SEC", "172800")),
            kling_tmpfiles_max_bytes=int(os.getenv("KLING_TMPFILES_MAX_BYTES", "100000000")),
            kling_tmpfiles_timeout_sec=int(os.getenv("KLING_TMPFILES_TIMEOUT_SEC", "120")),
            storage_path=artifacts_root,
            storage_public_url_base=os.getenv("STORAGE_PUBLIC_URL_BASE"),
            prefect_api_url=os.getenv("PREFECT_API_URL", ""),
            prefect_work_pool=os.getenv("PREFECT_WORK_POOL", "video-scene-agent"),
            prefect_docker_image=os.getenv("PREFECT_DOCKER_IMAGE", "video-scene-agent:latest"),
            openrouter_text_model=os.getenv(
                "OPENROUTER_TEXT_MODEL", "google/gemini-2.5-flash"
            ),
            openrouter_image_model=os.getenv(
                "OPENROUTER_IMAGE_MODEL", "google/gemini-2.5-flash-image"
            ),
            openrouter_video_model=os.getenv(
                "OPENROUTER_VIDEO_MODEL", "google/gemini-2.5-flash"
            ),
            openrouter_temperature=0.0,
            default_aspect_ratio=os.getenv("DEFAULT_ASPECT_RATIO", "16:9"),
            default_fps=int(os.getenv("DEFAULT_FPS", "24")),
            default_K_sb=int(os.getenv("DEFAULT_K_SB", "3")),
            default_K_vid=int(os.getenv("DEFAULT_K_VID", "2")),
            image_size=os.getenv("IMAGE_SIZE", "1024x1024"),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "120")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
        )

    def ensure_storage_dir(self) -> Path:
        """Ensure storage directory exists and return its path."""
        path = Path(self.storage_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run_artifacts_dir(self, run_id: str) -> Path:
        """Return the run-scoped artifact directory, creating parents if needed."""
        root = self.ensure_storage_dir()
        path = root / "runs" / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run_public_url_base(self, run_id: str) -> str | None:
        """Return the public URL base for a run-scoped artifact directory."""
        if not self.storage_public_url_base:
            return None
        return f"{self.storage_public_url_base.rstrip('/')}/runs/{str(run_id).strip('/')}"

    def run_kling_media_public_url_base(self, run_id: str) -> str | None:
        """Return the run-scoped public URL base used for Kling media inputs."""
        base = self.kling_media_public_url_base or self.storage_public_url_base
        if not base:
            return None
        return f"{base.rstrip('/')}/runs/{str(run_id).strip('/')}"
