from __future__ import annotations
"""Register Prefect deployments for the app and healthcheck flows."""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scene_agent.prefect_flows import generate_scene_flow
from scene_agent.prefect_ops import prefect_healthcheck_flow


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if value is None:
        return None
    return str(value)


def ensure_work_pool(pool_name: str) -> None:
    """Create or update the docker work pool."""
    subprocess.run(
        [
            "python3",
            "-m",
            "prefect",
            "work-pool",
            "create",
            pool_name,
            "--type",
            "docker",
            "--overwrite",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def base_job_variables(image: str) -> dict:
    """Build shared docker job variables for flow-run containers."""
    env = {
        "PREFECT_API_URL": _env("PREFECT_INTERNAL_API_URL", "http://prefect-server:4200/api"),
        "PREFECT_API_AUTH_STRING": _env("PREFECT_API_AUTH_STRING", ""),
        "PREFECT_LOGGING_LEVEL": _env("PREFECT_LOGGING_LEVEL", "INFO"),
        "PREFECT_LOGGING_TO_API_ENABLED": _env("PREFECT_LOGGING_TO_API_ENABLED", "true"),
        "PREFECT_LOGGING_LOG_PRINTS": _env("PREFECT_LOGGING_LOG_PRINTS", "true"),
        "PREFECT_LOGGING_EXTRA_LOGGERS": _env(
            "PREFECT_LOGGING_EXTRA_LOGGERS",
            "scene_agent,prefect.workers,prefect.runner,prefect_docker",
        ),
        "PREFECT_LOGGING_TO_API_WHEN_MISSING_FLOW": _env(
            "PREFECT_LOGGING_TO_API_WHEN_MISSING_FLOW",
            "ignore",
        ),
        "STORAGE_PATH": "/app/artifacts",
    }
    forwarded_env_names = (
        "OPENROUTER_API_KEY",
        "KLING_ACCESS_KEY",
        "KLING_SECRET_KEY",
        "KLING_API_BASE",
        "KLING_VIDEO_MODEL",
        "KLING_MODE",
        "KLING_SOUND",
        "KLING_POLL_TIMEOUT_SEC",
        "KLING_POLL_REQUEST_TIMEOUT_SEC",
        "KLING_POLL_INTERVAL_SEC",
        "KLING_RUN_TOKEN_LIMIT",
        "KLING_GENERATION_TOKENS_PER_SECOND",
        "KLING_EDIT_TOKENS_PER_SECOND",
        "KLING_USE_TMPFILES",
        "KLING_TMPFILES_UPLOAD_URL",
        "KLING_TMPFILES_TTL_SEC",
        "KLING_TMPFILES_MAX_BYTES",
        "KLING_TMPFILES_TIMEOUT_SEC",
        "KLING_MEDIA_PUBLIC_URL_BASE",
        "STORAGE_PUBLIC_URL_BASE",
        "IMAGE_SIZE",
        "REQUEST_TIMEOUT",
        "MAX_RETRIES",
    )
    for name in forwarded_env_names:
        if os.getenv(name):
            env[name] = os.environ[name]

    return {
        "image": image,
        "image_pull_policy": "Never",
        "networks": ["prefect"],
        "stream_output": True,
        "auto_remove": True,
        "volumes": [f"{ARTIFACTS_DIR}:/app/artifacts"],
        "env": env,
    }


def main() -> None:
    pool_name = _env("PREFECT_WORK_POOL", "video-scene-agent-docker")
    image = _env("VIDEO_SCENE_AGENT_IMAGE", "video-scene-agent-prefect:local")
    if not pool_name:
        raise ValueError("PREFECT_WORK_POOL is required")
    if not image:
        raise ValueError("VIDEO_SCENE_AGENT_IMAGE is required")

    ensure_work_pool(pool_name)
    job_variables = base_job_variables(image)

    generate_scene_flow.deploy(
        name="generate-scene",
        work_pool_name=pool_name,
        image=image,
        build=False,
        push=False,
        job_variables=job_variables,
        parameters={
            "user_brief": "A calm ocean wave at sunset",
            "constraints": {},
            "run_options": {},
        },
        description="Primary deployment for video scene generation.",
        tags=["scene-agent", "docker"],
        print_next_steps=False,
    )

    prefect_healthcheck_flow.deploy(
        name="healthcheck",
        work_pool_name=pool_name,
        image=image,
        build=False,
        push=False,
        job_variables=job_variables,
        description="Operational smoke test deployment for worker/UI logging.",
        tags=["prefect", "healthcheck"],
        print_next_steps=False,
    )


if __name__ == "__main__":
    main()
