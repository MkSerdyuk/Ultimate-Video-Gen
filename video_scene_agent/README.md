# Video Scene Agent

Prefect-based video scene generation using OpenRouter, direct Kling 3.0, and FFmpeg.

## Overview

The current runtime executes the scene pipeline as a Prefect flow with run-scoped artifacts under `artifacts/runs/<run_id>/`.
The logical subsystems are unchanged:

- `Director`: world + storyboard generation
- `Storyboard Editor`: storyboard review/fix loop
- `Operator`: segment generation + stitching
- `Video Editor`: final video review/fix loop

`scene_agent.main.run()` and `python -m scene_agent` invoke the Prefect flow directly.

## Features

- **Text / review generation**: OpenRouter
- **Image generation**: `google/gemini-2.5-flash-image` via OpenRouter
- **Video generation / repair**: Kling 3.0 Omni Standard via direct Kling API
- **Video stitching**: FFmpeg
- **Artifacts**: local filesystem storage with run manifest persistence
- **Resumability**: Prefect flow state + `manifest.json` under each run directory

## Installation

```bash
cd video_scene_agent
python3 -m pip install -e ".[dev]"
```

### System Requirements

- Python 3.10+
- FFmpeg

```bash
brew install ffmpeg
# or
sudo apt install ffmpeg
```

## Configuration

Runtime configuration belongs in `video_scene_agent/.env`. When working from the
package directory, create it from the checked-in template:

```bash
cp .env.example .env
```

Required for live generation:

```bash
OPENROUTER_API_KEY=sk-or-...
KLING_ACCESS_KEY=...
KLING_SECRET_KEY=...
STORAGE_PATH=./artifacts
```

Optional Kling settings:

```bash
KLING_API_BASE=https://api-singapore.klingai.com
KLING_VIDEO_MODEL=kling-v3-omni
KLING_MODE=std
KLING_SOUND=off
KLING_POLL_TIMEOUT_SEC=900
KLING_POLL_REQUEST_TIMEOUT_SEC=30
KLING_POLL_INTERVAL_SEC=2.0
KLING_RUN_TOKEN_LIMIT=60
KLING_GENERATION_TOKENS_PER_SECOND=0.6
KLING_EDIT_TOKENS_PER_SECOND=0.9
KLING_USE_TMPFILES=1
KLING_TMPFILES_UPLOAD_URL=https://tmpfiles.org/api/v1/upload
KLING_TMPFILES_TTL_SEC=172800
KLING_TMPFILES_MAX_BYTES=100000000
KLING_TMPFILES_TIMEOUT_SEC=120
```

Optional Prefect settings for self-hosted deployments:

```bash
PREFECT_API_URL=http://127.0.0.1:4200/api
PREFECT_WORK_POOL=video-scene-agent
PREFECT_DOCKER_IMAGE=video-scene-agent:latest
STORAGE_PUBLIC_URL_BASE=https://example.com/artifacts
PREFECT_SERVER_API_AUTH_STRING=admin:replace-with-long-random-password
PREFECT_API_AUTH_STRING=admin:replace-with-long-random-password
```

Kling media inputs are published through tmpfiles by default. Local keyframes
and source segment videos stay in `artifacts/runs/...`, and only temporary
copies are uploaded as direct `https://tmpfiles.org/dl/...` inputs for Kling.
`STORAGE_PUBLIC_URL_BASE` and `KLING_MEDIA_PUBLIC_URL_BASE` are optional now;
use them for public artifact links or if `KLING_USE_TMPFILES=0` and you provide
your own provider-fetchable media hosting.

`KLING_RUN_TOKEN_LIMIT` is a hard run-level cap in Kling resource units. When
the cap is reached, the runtime stops paid Kling calls, keeps existing segments
for skipped edits, and uses local FFmpeg still clips for missing generated
segments so the final video can still be stitched and returned.

Keyframe images, image-to-video segments, and feature-guided segment repairs all
receive the same normalized `constraints.aspect_ratio` value. Unknown aspect
ratios fall back to `16:9` consistently across image and video providers.

## Usage

### CLI

```bash
python3 -m scene_agent "Ночная сцена: герой идет по дождливой улице..."
```

### Python API

```python
from scene_agent.main import run

result = run(
    user_brief="Красивый закат над океаном, камера медленно наезжает",
    constraints={
        "aspect_ratio": "16:9",
        "duration_sec": 5.0,
    },
)
```

The returned dict includes:

- `run_id`
- `status`
- `artifacts_dir`
- `final_video_uri`
- `storyboard`
- `world`
- `frame_uris`
- `segment_uris`
- `reviews`

## Runtime Layout

Generated outputs are stored under:

```text
artifacts/
  runs/
    <run_id>/
      manifest.json
      world.json
      storyboard.json
      reviews/
      storyboards/
      *.png / *.mp4
```

## Self-Hosted Prefect Target

The intended stable deployment topology is:

- `postgres`
- `redis`
- `prefect-server`
- `prefect-services`
- `prefect-worker`

For lightweight local experimentation, Prefect can still run with SQLite, but production/self-hosted stable usage should use PostgreSQL and Redis.

## Tests

```bash
PYTHONPATH=src python3 -m pytest -q
python3 -m compileall src
```

## VBench2 API-Proxy Benchmark

Benchmark code is isolated in root `benchmarking/`. The checked-in artifact
bundle contains the final fixed 30-sample comparison set. Scores are API-proxy
LLM-as-judge scores through `qwen/qwen3.6-flash`, not official GPU VBench2
leaderboard scores.

```bash
cd ..
python3 benchmarking/run_sample_bench.py report
python3 benchmarking/run_sample_bench.py judge --model our --sample-id complex_plot_000 --allow-paid
python3 benchmarking/generate_vbench2_samples.py --sample-id complex_plot_000 --allow-paid
```

## Minimal Live Smoke

The guarded smoke script checks direct Kling 3.0 Standard start/end-frame generation and segment edit with low-cost settings:

```bash
python3 scripts/live_smoke_kling_minimal.py --dry-run --duration 4
RUN_LIVE_SMOKE=1 python3 scripts/live_smoke_kling_minimal.py --duration 4
```

For an end-to-end Prefect smoke, use a tiny run with `duration_sec=3.0`,
`fps=12`, `num_keyframes=2`, `K_sb=3`, `K_vid=2`, and
`run_options={"force_edit_segments": [0]}`. A successful public deployment
shows a final-video link artifact and a final-video-poster image artifact in
the Prefect UI; the same URLs should require auth outside an authenticated
browser session.

```bash
python3 scripts/live_smoke_prefect_minimal.py --dry-run
RUN_LIVE_SMOKE=1 python3 scripts/live_smoke_prefect_minimal.py
```
