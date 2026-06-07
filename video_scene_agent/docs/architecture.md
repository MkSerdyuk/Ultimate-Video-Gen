# Architecture: Video Scene Agent

## System Architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│                           Prefect Flow                              │
│                                                                     │
│  director_world                                                     │
│      -> director_storyboard                                         │
│      -> keyframes_generate                                          │
│      -> storyboard review/fix loop                                  │
│      -> segments_generate                                           │
│      -> stitch_video                                                │
│      -> video review/fix loop                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          External APIs                              │
│                                                                     │
│  OpenRouter           Kling 3.0            Storage / FFmpeg         │
│  - LLM text           - image-to-video     - Local FS artifacts     │
│  - Image gen          - feature-guided     - tmpfiles media bridge  │
│  - Video review         video repair       - FFmpeg stitch          │
│                       - video download                              │
└─────────────────────────────────────────────────────────────────────┘
```

## Runtime Model

The production code path is a Prefect flow executed by `scene_agent.main.run()`.
Each run gets a stable `run_id`, and all outputs are written under:

```text
artifacts/runs/<run_id>/
```

The run directory contains:

- `manifest.json`
- `world.json`
- `storyboard.json`
- `reviews/*.json`
- generated frames and video segments
- stitched final video
- storyboard markdown / director's cut

The runtime is Prefect-only. Business logic lives under `src/scene_agent/pipeline/`, and Prefect task wrappers live under `src/scene_agent/flows/tasks/`.

## Component Details

### 1. Director

Purpose: generate the creative foundation.

Outputs:

- `WorldDescription`
- `StoryboardData`
- keyframe image URIs

### 2. Storyboard Editor

Purpose: review the storyboard and produce bounded fix loops.

Issue format:

- `target`: `global`, `frame:N`, `segment:N`
- `severity`: `info`, `warning`, `error`
- `problem`
- optional `suggestion`

### 3. Operator

Purpose: generate video segments and stitch them.

Behavior:

- supports selective `regen_segments`
- supports selective `edit_segments` with segment-scoped video-to-video repair
- preserves segment indices and prior artifacts
- stitches with FFmpeg into a run-scoped final output

### 4. Video Editor

Purpose: review the final video and decide regeneration strategy.

Supported outcomes:

- `regen_frames`
- `edit_segments`
- `regen_segments`
- `regen_all`

## Data Flow

```text
User Brief
  -> WorldDescription
  -> StoryboardData
  -> Keyframes
  -> Storyboard QC loop
  -> Video segments
  -> Final stitched video
  -> Video QC loop
  -> Done
```

## State and Persistence

The flow keeps working state in `SceneState` and persists durable outputs in `ArtifactManifest`.

Tracked state includes:

- input: `user_brief`, `constraints`, `run_id`
- creative: `world`, `storyboard`
- assets: `frame_uris`, `segment_uris`, `final_video_uri`
- review: `sb_iteration`, `vid_iteration`, `sb_issues`, `vid_issues`
- regeneration/repair: `regen_frames`, `edit_segments`, `regen_segments`
- output status: `status`, `error`, `error_code`

## Self-Hosted Prefect Topology

Stable self-hosted target:

- `postgres`
- `redis`
- `prefect-server`
- `prefect-services`
- `prefect-worker`

SQLite remains acceptable only for lightweight local runs.
