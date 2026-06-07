# Artefact Sample

This folder contains one complete generation run copied from the agent runtime artifacts.

Source run:

```text
prefect-smoke-20260607-1
```

Contents:

- `world.json`: generated world package for the scene.
- `storyboard.json`: structured storyboard with frames, segments, prompts, and continuity constraints.
- `storyboards/`: human-readable storyboard markdown snapshot.
- `*.png`: generated keyframe anchors.
- `segments/`: generated video segment between the keyframes.
- `final_video_prefect-smoke-20260607-1.mp4`: stitched final video.
- `reviews/`: storyboard and final video review outputs.
- `reports/run-report.md`: compact run report.
- `manifest.json`: persisted run manifest tying all artifacts together.
