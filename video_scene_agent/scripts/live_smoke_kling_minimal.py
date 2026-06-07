from __future__ import annotations
"""Minimal paid smoke test for direct Kling 3.0 Standard video paths.

    RUN_LIVE_SMOKE=1 python3 scripts/live_smoke_kling_minimal.py --duration 4

Use --dry-run to print payloads without API calls.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scene_agent.config import Config
from scene_agent.tools.kling import KlingTool
from scene_agent.tools.storage import LocalStorageBackend

DEFAULT_START_IMAGE_URL = "https://placehold.co/1280x720/50648c/e6e6e6.png?text=START"
DEFAULT_END_IMAGE_URL = "https://placehold.co/1280x720/78966e/e6e6e6.png?text=END"


class DryRunMediaPublisher:
    def publish(self, uri: str, *, expected_kind: str | None = None) -> str:
        suffix = ".mp4" if expected_kind == "video" else ".png"
        name = Path(uri.split("?", 1)[0]).name or f"media{suffix}"
        if "." not in name:
            name = f"{name}{suffix}"
        return f"https://tmpfiles.org/dl/dry-run/{name}"


def _require_live_env() -> None:
    missing = [
        name
        for name in ("KLING_ACCESS_KEY", "KLING_SECRET_KEY")
        if not os.getenv(name)
    ]
    if missing:
        raise SystemExit(f"Missing required env vars for live Kling smoke: {', '.join(missing)}")
    if os.getenv("RUN_LIVE_SMOKE") != "1":
        raise SystemExit("Set RUN_LIVE_SMOKE=1 to allow paid live smoke calls.")


def _write_anchor(path: Path, label: str, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1280, 720), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((520, 260, 760, 500), fill=(230, 230, 230), outline=(20, 20, 20), width=6)
    draw.text((548, 358), label, fill=(20, 20, 20))
    image.save(path, "PNG")


def _specs(
    config: Config,
    storage: LocalStorageBackend,
    run_id: str,
    duration_sec: float,
    aspect_ratio: str,
    start_image_url: str | None,
    end_image_url: str | None,
    *,
    dry_run: bool = False,
) -> tuple[KlingTool, dict, dict]:
    tool = KlingTool(
        config,
        storage,
        media_public_url_base=config.run_kling_media_public_url_base(run_id),
        media_publisher=DryRunMediaPublisher() if dry_run else None,
    )
    if start_image_url and end_image_url:
        start_uri = start_image_url
        end_uri = end_image_url
    else:
        start = storage.base_path / "frames" / "start.png"
        end = storage.base_path / "frames" / "end.png"
        _write_anchor(start, "START", (80, 100, 140))
        _write_anchor(end, "END", (120, 150, 110))
        start_uri = f"file://{start}"
        end_uri = f"file://{end}"

    i2v_spec = {
        "start_image_uri": start_uri,
        "end_image_uri": end_uri,
        "prompt": "A simple centered cube slowly rotates in place on a plain studio floor.",
        "negative_prompt": "text, watermark, cuts, camera shake, extra objects",
        "duration_sec": duration_sec,
        "fps": 24,
        "aspect_ratio": aspect_ratio,
    }
    v2v_spec = {
        "segment_uri": "",
        "start_image_uri": start_uri,
        "end_image_uri": end_uri,
        "prompt": "Repair the clip only by stabilizing the cube and preserving the same simple studio composition.",
        "negative_prompt": "text, watermark, cuts, camera shake, extra objects",
        "duration_sec": duration_sec,
        "fps": 24,
        "aspect_ratio": aspect_ratio,
    }
    return tool, i2v_spec, v2v_spec


def _config_from_env_for_smoke() -> Config:
    return Config(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", "kling-smoke"),
        kling_access_key=os.getenv("KLING_ACCESS_KEY", ""),
        kling_secret_key=os.getenv("KLING_SECRET_KEY", ""),
        kling_api_base=os.getenv("KLING_API_BASE", "https://api-singapore.klingai.com"),
        kling_video_model=os.getenv("KLING_VIDEO_MODEL", "kling-v3-omni"),
        kling_mode=os.getenv("KLING_MODE", "std"),
        kling_sound=os.getenv("KLING_SOUND", "off"),
        kling_run_token_limit=float(os.getenv("KLING_RUN_TOKEN_LIMIT", "60")),
        kling_generation_tokens_per_second=float(os.getenv("KLING_GENERATION_TOKENS_PER_SECOND", "0.6")),
        kling_edit_tokens_per_second=float(os.getenv("KLING_EDIT_TOKENS_PER_SECOND", "0.9")),
        kling_use_tmpfiles=os.getenv("KLING_USE_TMPFILES", "1").lower() not in {"0", "false", "no", "off"},
        kling_tmpfiles_upload_url=os.getenv("KLING_TMPFILES_UPLOAD_URL", "https://tmpfiles.org/api/v1/upload"),
        kling_tmpfiles_ttl_sec=int(os.getenv("KLING_TMPFILES_TTL_SEC", "172800")),
        kling_tmpfiles_max_bytes=int(os.getenv("KLING_TMPFILES_MAX_BYTES", "100000000")),
        kling_tmpfiles_timeout_sec=int(os.getenv("KLING_TMPFILES_TIMEOUT_SEC", "120")),
        kling_media_public_url_base=os.getenv("KLING_MEDIA_PUBLIC_URL_BASE"),
        storage_public_url_base=os.getenv("STORAGE_PUBLIC_URL_BASE"),
        storage_path=os.getenv("STORAGE_PATH", "./artifacts"),
        request_timeout=int(os.getenv("REQUEST_TIMEOUT", "120")),
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
    )


def _run_live_smoke(tool: KlingTool, i2v_spec: dict, v2v_spec: dict) -> dict:
    generation_task_id = tool._submit_task(tool.build_generation_payload(i2v_spec))
    tool.last_generation_task_ids = [generation_task_id]
    generated_url = tool._poll_task(generation_task_id)
    segment_uri = tool._download_and_store(generated_url, key=f"segments/kling-{generation_task_id}.mp4")

    edit_spec = dict(v2v_spec)
    edit_spec["segment_uri"] = segment_uri
    edit_task_id = tool._submit_task(tool.build_edit_payload(edit_spec))
    tool.last_edit_task_ids = [edit_task_id]
    edited_url = tool._poll_task(edit_task_id)
    edited_uri = tool._download_and_store(edited_url, key=f"segments/kling-edit-{edit_task_id}.mp4")

    return {
        "segment_uri": segment_uri,
        "edited_uri": edited_uri,
        "generation_task_id": generation_task_id,
        "edit_task_id": edit_task_id,
        "estimated_generation_tokens": tool.estimate_generation_tokens(i2v_spec),
        "estimated_edit_tokens": tool.estimate_edit_tokens(v2v_spec),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal live smoke for direct Kling video paths.")
    parser.add_argument("--dry-run", action="store_true", help="Build minimal Kling payloads without API calls.")
    parser.add_argument("--duration", type=float, default=4.0, help="Smoke segment duration in seconds.")
    parser.add_argument("--aspect-ratio", default="16:9", help="Smoke aspect ratio.")
    parser.add_argument("--remote-anchors", action="store_true", help="Use public HTTPS placeholder anchors.")
    parser.add_argument("--start-image-url", default=DEFAULT_START_IMAGE_URL, help="HTTPS first-frame anchor URL.")
    parser.add_argument("--end-image-url", default=DEFAULT_END_IMAGE_URL, help="HTTPS end-frame anchor URL.")
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    run_id = datetime.now(timezone.utc).strftime("kling-smoke-%Y%m%d-%H%M%S")
    config = _config_from_env_for_smoke()
    storage = LocalStorageBackend(
        base_path=config.run_artifacts_dir(run_id),
        public_url_base=config.run_public_url_base(run_id),
    )
    start_image_url = args.start_image_url if args.remote_anchors else None
    end_image_url = args.end_image_url if args.remote_anchors else None
    tool, i2v_spec, v2v_spec = _specs(
        config,
        storage,
        run_id,
        args.duration,
        args.aspect_ratio,
        start_image_url,
        end_image_url,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "image_to_video": tool.build_generation_payload(i2v_spec),
                    "estimated_generation_tokens": tool.estimate_generation_tokens(i2v_spec),
                    "estimated_edit_tokens": tool.estimate_edit_tokens(v2v_spec),
                    "run_token_limit": config.kling_run_token_limit,
                },
                indent=2,
            )
        )
        return

    _require_live_env()
    print(json.dumps(_run_live_smoke(tool, i2v_spec, v2v_spec), indent=2))


if __name__ == "__main__":
    main()
