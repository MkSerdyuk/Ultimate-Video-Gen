from __future__ import annotations
"""Operational Prefect flows for infrastructure smoke checks."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from prefect import flow, runtime, task

from scene_agent.prefect_logging import get_prefect_logger


@task(name="prefect_healthcheck_task", log_prints=True)
def prefect_healthcheck_task() -> dict:
    """Emit deterministic logs and persist a small health artifact."""
    run_id = str(runtime.flow_run.id or "local-healthcheck")
    logger = get_prefect_logger(component="task", step="prefect_healthcheck", scene_run_id=run_id)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "cwd": os.getcwd(),
        "hostname": os.uname().nodename,
    }
    artifacts_dir = Path(os.getenv("STORAGE_PATH", "/app/artifacts")) / "healthchecks"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifacts_dir / f"{run_id}.json"
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Healthcheck artifact written to %s", artifact_path)
    print(f"Prefect healthcheck completed for run {run_id}")
    return payload


@flow(name="prefect_healthcheck_flow", log_prints=True)
def prefect_healthcheck_flow() -> dict:
    """Run a lightweight smoke flow to validate worker/UI logging."""
    run_id = str(runtime.flow_run.id or "local-healthcheck")
    logger = get_prefect_logger(component="flow", scene_run_id=run_id)
    logger.info("Starting Prefect healthcheck flow")
    result = prefect_healthcheck_task()
    logger.info("Prefect healthcheck flow completed")
    return result
