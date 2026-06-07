from __future__ import annotations
"""Main entry point for video scene generation."""

import sys
import logging
from typing import Any, Optional
from uuid import uuid4

from scene_agent.config import Config
from scene_agent.models import Constraints
from scene_agent.runtime import load_run_result_from_disk
from scene_agent.utils.log import setup_logging

log = logging.getLogger(__name__)


def run(
    user_brief: str,
    constraints: Optional[dict[str, Any]] = None,
    config: Optional[Config] = None,
    run_options: Optional[dict[str, Any]] = None,
) -> dict:
    """
    Run the video scene generation workflow.

    Args:
        user_brief: User's text description of the desired scene
        constraints: Optional generation parameters
        config: Optional configuration object (uses env defaults if not provided)
        run_options: Optional runtime controls such as an explicit run_id for resume

    Returns:
        Dict with:
            - run_id: Stable execution/run identifier
            - final_video_uri: URI of generated video
            - storyboard: Storyboard object
            - world: World description
            - frame_uris: List of key frame URIs
            - segment_uris: List of segment URIs
    """
    if config is None:
        config = Config.from_env()

    # Setup logging
    setup_logging()
    log.info(f"Starting video generation for: {user_brief[:100]}...")
    effective_run_options = dict(run_options or {})
    effective_run_options.setdefault("run_id", str(uuid4()))
    run_id = effective_run_options["run_id"]

    try:
        from scene_agent.prefect_flows import generate_scene_flow

        constraints_obj = Constraints(**(constraints or {}))

        log.info("Invoking Prefect flow...")
        result = generate_scene_flow(
            user_brief=user_brief,
            constraints=constraints_obj,
            config=config,
            run_options=effective_run_options,
        )
        output = result.model_dump(exclude_none=True, mode="json")

        log.info(f"Workflow finished with status={output.get('status')}")
        return output

    except Exception as e:
        log.error(f"Workflow failed: {e}", exc_info=True)
        persisted = load_run_result_from_disk(config, run_id)
        if persisted is not None:
            return persisted.model_dump(exclude_none=True, mode="json")
        return {
            "run_id": run_id,
            "error": str(e),
            "error_code": type(e).__name__,
            "status": "failed",
        }


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate video scenes from text descriptions"
    )
    parser.add_argument(
        "brief",
        help="Text description of the desired scene"
    )
    parser.add_argument(
        "--aspect-ratio",
        default="16:9",
        help="Aspect ratio (default: 16:9)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Target duration in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second (default: 24)"
    )
    parser.add_argument(
        "--style",
        action="append",
        dest="style_tags",
        help="Style tags (can be used multiple times)"
    )
    parser.add_argument(
        "--ksb",
        type=int,
        default=3,
        help="Max storyboard review iterations (default: 3)"
    )
    parser.add_argument(
        "--kvid",
        type=int,
        default=2,
        help="Max video review iterations (default: 2)"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level)

    # Build constraints
    constraints = {
        "aspect_ratio": args.aspect_ratio,
        "duration_sec": args.duration,
        "fps": args.fps,
        "style_tags": args.style_tags or [],
        "K_sb": args.ksb,
        "K_vid": args.kvid,
    }

    # Run workflow
    result = run(args.brief, constraints)

    # Print result
    if result.get("status") == "completed":
        print("\n" + "=" * 50)
        print("Video generation completed!")
        print("=" * 50)
        if result.get("run_id"):
            print(f"Run ID: {result['run_id']}")
        if result.get("artifacts_dir"):
            print(f"Artifacts: {result['artifacts_dir']}")
        if result.get("final_video_uri"):
            print(f"Video: {result['final_video_uri']}")
        if result.get("frame_uris"):
            print(f"Key frames: {len(result['frame_uris'])}")
        if result.get("segment_uris"):
            print(f"Segments: {len(result['segment_uris'])}")
    else:
        print(f"\nError: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
