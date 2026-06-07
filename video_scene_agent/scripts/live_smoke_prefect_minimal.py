from __future__ import annotations
"""Minimal paid Prefect smoke for the full scene flow and forced edit path.

Run only when you intentionally want live API calls:

    RUN_LIVE_SMOKE=1 python3 scripts/live_smoke_prefect_minimal.py

Use --dry-run to print the exact low-cost flow parameters without network calls.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scene_agent.config import Config
from scene_agent.models import Constraints
from scene_agent.prefect_flows import generate_scene_flow


def _require_live_env() -> None:
    missing = [
        name
        for name in ("OPENROUTER_API_KEY", "KLING_ACCESS_KEY", "KLING_SECRET_KEY")
        if not os.getenv(name)
    ]
    if missing:
        raise SystemExit(f"Missing required env vars for live Prefect smoke: {', '.join(missing)}")
    if os.getenv("RUN_LIVE_SMOKE") != "1":
        raise SystemExit("Set RUN_LIVE_SMOKE=1 to allow paid live smoke calls.")


def _smoke_payload() -> tuple[str, Constraints, dict[str, object]]:
    run_id = datetime.now(timezone.utc).strftime("prefect-smoke-%Y%m%d-%H%M%S")
    constraints = Constraints(
        aspect_ratio="1:1",
        duration_sec=3.0,
        target_duration_sec=3.0,
        fps=12,
        num_keyframes=2,
        K_sb=3,
        K_vid=2,
        style_tags=["minimal studio test", "plain background", "stationary product shot"],
    )
    run_options = {
        "run_id": run_id,
        "force_edit_segments": [0],
    }
    brief = (
        "Minimal smoke test: a single red cube remains still on a plain gray studio floor. "
        "Use a locked-off camera, stable lighting, simple centered composition, and no text."
    )
    return brief, constraints, run_options


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal live Prefect smoke for the scene flow.")
    parser.add_argument("--dry-run", action="store_true", help="Print flow parameters without API calls.")
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    os.environ.setdefault("KLING_MODE", "std")
    os.environ.setdefault("KLING_SOUND", "off")
    os.environ.setdefault("KLING_RUN_TOKEN_LIMIT", "60")
    os.environ.setdefault("KLING_USE_TMPFILES", "1")
    os.environ.setdefault("IMAGE_SIZE", "512x512")

    brief, constraints, run_options = _smoke_payload()
    payload = {
        "user_brief": brief,
        "constraints": constraints.model_dump(mode="json"),
        "run_options": run_options,
        "effective_env": {
            "KLING_VIDEO_MODEL": os.getenv("KLING_VIDEO_MODEL", "kling-v3-omni"),
            "KLING_MODE": os.getenv("KLING_MODE"),
            "KLING_SOUND": os.getenv("KLING_SOUND"),
            "KLING_RUN_TOKEN_LIMIT": os.getenv("KLING_RUN_TOKEN_LIMIT"),
            "KLING_USE_TMPFILES": os.getenv("KLING_USE_TMPFILES"),
            "KLING_TMPFILES_TTL_SEC": os.getenv("KLING_TMPFILES_TTL_SEC", "172800"),
            "IMAGE_SIZE": os.getenv("IMAGE_SIZE"),
            "STORAGE_PUBLIC_URL_BASE": os.getenv("STORAGE_PUBLIC_URL_BASE", ""),
            "KLING_MEDIA_PUBLIC_URL_BASE": os.getenv("KLING_MEDIA_PUBLIC_URL_BASE", ""),
        },
    }

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    _require_live_env()
    result = generate_scene_flow(
        brief,
        constraints=constraints,
        config=Config.from_env(),
        run_options=run_options,
    )
    print(result.model_dump_json(indent=2, exclude_none=True))


if __name__ == "__main__":
    main()
