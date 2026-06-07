# Implementation Status: Prefect Migration

## Summary

The repository uses a Prefect-only primary runtime.

Primary runtime path:

- `scene_agent.main.run()`
- `scene_agent.prefect_flows.generate_scene_flow()`
- `scene_agent.flows.tasks.*`

## Delivered Changes

### 1. Canonical workflow contracts

Implemented under `src/scene_agent/models/`:

- `WorldDescription`
- `StoryboardData`
- `StoryboardFrameData`
- `StoryboardSegmentData`
- `Issue`
- `StoryboardFixResult`
- `VideoFixResult`
- `ArtifactManifest`
- `SceneRunResult`

Cross-task contracts remain in `scene_agent.models`.

### 2. Prefect orchestration

Implemented:

- explicit Prefect flow with bounded storyboard and video review loops
- run-scoped artifacts at `artifacts/runs/<run_id>/`
- manifest persistence and restore
- public `run()` cutover to Prefect flow

### 3. Provider and tool stabilization

Implemented:

- provider adapter layer under `src/scene_agent/providers/`
- normalized provider errors
- fixed OpenRouter video review payload to use `video_url`
- fixed `LocalStorageBackend.start_http_server()`
- fixed LLM JSON continuation accumulation
- fixed idempotent logging setup
- fixed selective `regen_segments`
- fixed storyboard fix handling for `segment/global` issues
- added `regen_all` support in video fix handling

### 4. Config and env normalization

Implemented:

- direct Kling 3.0 Standard video provider via `KLING_ACCESS_KEY` / `KLING_SECRET_KEY`
- segment repair path via Kling 3.0 Omni video edit
- Prefect deployment env placeholders
- updated OpenRouter defaults to current models used by the new flow

### 5. Test coverage

Implemented focused tests for:

- canonical models
- Prefect routing and repair behavior
- storage HTTP server
- idempotent logging
- OpenRouter video review payload shape
- LLM continuation JSON assembly
- selective segment regeneration
- Prefect flow completion and resume behavior

## Current Verification

Validated locally:

```bash
python3 -m compileall src
python3 -m pytest -q
```

At the time of the last run:

- `93 passed`
- `11 skipped`

Skipped tests are live integration tests gated by missing external API credentials.

## Remaining Operational Notes

- Stable self-hosted Prefect deployment should use PostgreSQL + Redis.
- Docker worker runtime still depends on host Docker daemon access and available RAM.
- Benchmark scripts live in root `benchmarking/` and production runtime code remains under `src/scene_agent/`.
