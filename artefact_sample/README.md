# Artefact Sample

This folder contains complete generation runs copied from the agent runtime artifacts.

Runs:

- `prefect-smoke-20260607-1`: simple one-segment smoke sample with two keyframes.
- `artifact-sample-fisher-cat-simple-ru-10s-20260607`: 10-second two-segment sample from a simple Russian prompt, where a fisherman catches a fish and gives it to a cat that runs up to him in one scene without a camera-angle change.

Contents:

- `world.json`: generated world package for the scene.
- `storyboard.json`: structured storyboard with frames, segments, prompts, and continuity constraints.
- `storyboards/`: human-readable storyboard markdown snapshot.
- `*.png`: generated keyframe anchors.
- `segments/`: generated video segments between keyframes.
- `final_video_*.mp4`: stitched final video.
- `reviews/`: storyboard and final video review outputs.
- `reports/run-report.md`: compact run report.
- `manifest.json`: persisted run manifest tying all artifacts together.
