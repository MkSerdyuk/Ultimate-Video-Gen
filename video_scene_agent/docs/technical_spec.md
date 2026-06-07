# Technical Specification: Video Scene Agent

## Overview

Video Scene Agent is a Prefect-based video generation system that creates short video scenes from text descriptions using:

- **Text / review**: OpenRouter
- **Image generation**: `google/gemini-2.5-flash-image` via OpenRouter
- **Video generation / repair**: Kling 3.0 Omni Standard via direct Kling API; segment repair uses feature-guided regeneration (`video_list.refer_type=feature` plus a first-frame keyframe), not base-video edit mode
- **Video review**: OpenRouter multimodal video input
- **Storage**: local filesystem backend with run-scoped manifests and artifacts

## Flow

```text
director_world
  -> director_storyboard
  -> keyframes_generate
  -> storyboard review/fix loop
  -> segments_generate
  -> stitch_video
  -> video review/fix loop
  -> END
```

## Components

### Models (`models/`)

Canonical workflow contracts live under `scene_agent.models`:

- `Constraints`
- `WorldDescription`
- `StoryboardData`
- `StoryboardFrameData`
- `StoryboardSegmentData`
- `Issue`
- `StoryboardFixResult`
- `VideoFixResult`
- `ArtifactManifest`
- `SceneRunResult`
- `SceneState`

Legacy Pydantic models (`Storyboard`, `FrameSpec`, `SegmentSpec`, `WorldPackage`) are retained for compatibility with older graph code and tests.

### Runtime (`runtime.py`)

Run-scoped runtime that:

- builds `LocalStorageBackend` under `artifacts/runs/<run_id>/`
- maps `STORAGE_PUBLIC_URL_BASE` to `.../runs/<run_id>` for public artifact links
- publishes every Kling media input through tmpfiles by default, returning direct `https://tmpfiles.org/dl/...` URLs
- keeps `KLING_MEDIA_PUBLIC_URL_BASE` only as an optional custom media-hosting path when tmpfiles is disabled
- constructs provider adapters and legacy node tool bindings
- persists and restores `manifest.json`
- converts state to `SceneRunResult`

Aspect ratio is normalized once per runtime path and passed consistently into
OpenRouter keyframe generation, Kling image-to-video generation, and Kling
feature-guided repair generation. Unsupported values fall back to `16:9`.

### Provider Layer (`providers/`)

Adapters wrap the current tool implementations and normalize retryable vs permanent provider failures:

- `OpenRouterTextAdapter`
- `OpenRouterImageAdapter`
- `OpenRouterVideoReviewAdapter`
- `KlingVideoAdapter`

### Prefect Orchestration

- `flows/tasks/`: atomic task wrappers around director/editor/operator/video-editor behaviors
- `prefect_flows.py`: end-to-end flow with explicit bounded loops
- `main.py`: public entrypoint that invokes the Prefect flow directly
- final runs publish a Prefect link artifact for the MP4 and an image artifact for a poster frame when public artifact URLs are configured

### Storage (`tools/storage.py`)

Filesystem-backed artifact storage with:

- `file://` URIs
- optional HTTP server
- run-scoped artifact directories

### External Tools

- `tools/openrouter_llm.py`
- `tools/openrouter_image.py`
- `tools/openrouter_video_review.py`
- `tools/kling.py`
- `tools/stitch.py`
- `tools/vision_rewriter.py`

## Public API

### Python

```python
from scene_agent.main import run

result = run(
    user_brief="Закат над океаном, камера медленно наезжает",
    constraints={"aspect_ratio": "16:9", "duration_sec": 5.0},
)
```

### Result Envelope

`run()` returns a dict derived from `SceneRunResult`:

- `run_id`
- `status`
- `artifacts_dir`
- `final_video_uri`
- `storyboard`
- `world`
- `frame_uris`
- `segment_uris`
- `reviews`
- `error`
- `error_code`

## Environment Variables

### Required for live generation

- `OPENROUTER_API_KEY`
- `KLING_ACCESS_KEY`
- `KLING_SECRET_KEY`
- `STORAGE_PATH`

### Optional runtime settings

- `SCENE_AGENT_ARTIFACTS_ROOT`
- `STORAGE_PUBLIC_URL_BASE`
- `KLING_API_BASE`
- `KLING_VIDEO_MODEL`
- `KLING_MODE`
- `KLING_SOUND`
- `KLING_POLL_TIMEOUT_SEC`
- `KLING_POLL_REQUEST_TIMEOUT_SEC`
- `KLING_POLL_INTERVAL_SEC`
- `KLING_RUN_TOKEN_LIMIT`
- `KLING_GENERATION_TOKENS_PER_SECOND`
- `KLING_EDIT_TOKENS_PER_SECOND`
- `KLING_USE_TMPFILES`
- `KLING_TMPFILES_UPLOAD_URL`
- `KLING_TMPFILES_TTL_SEC`
- `KLING_TMPFILES_MAX_BYTES`
- `KLING_TMPFILES_TIMEOUT_SEC`
- `KLING_MEDIA_PUBLIC_URL_BASE`
- `OPENROUTER_TEXT_MODEL`
- `OPENROUTER_IMAGE_MODEL`
- `OPENROUTER_VIDEO_MODEL`
- `DEFAULT_ASPECT_RATIO`
- `DEFAULT_FPS`
- `DEFAULT_K_SB`
- `DEFAULT_K_VID`
- `REQUEST_TIMEOUT`
- `MAX_RETRIES`

### Optional Prefect deployment settings

- `PREFECT_API_URL`
- `PREFECT_WORK_POOL`
- `PREFECT_DOCKER_IMAGE`
- `PREFECT_SERVER_API_AUTH_STRING`
- `PREFECT_API_AUTH_STRING`

## Self-Hosted Prefect Notes

Stable self-hosted target is:

- `postgres`
- `redis`
- `prefect-server`
- `prefect-services`
- `prefect-worker`

For local experimentation, Prefect can still run with SQLite.

Public deployment should place Nginx or another reverse proxy in front of both
`/prefect/` and `/artifacts/`. The same auth layer must protect both paths so
Prefect UI media links work for authenticated users and return `401` or SSO
login for unauthenticated direct access.
