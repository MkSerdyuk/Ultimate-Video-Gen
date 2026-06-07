
from __future__ import annotations

import base64
import logging
from datetime import datetime

from scene_agent.models import SceneState

log = logging.getLogger(__name__)

def write_storyboard_markdown(state: SceneState, storage, storyboard_data: dict, frame_uris: list[str], prompt_changes: dict = None) -> str:
    """
    Write director's cut storyboard to markdown file in artifacts folder.

    Args:
        state: Current scene state
        storage: Storage backend for path conversion
        storyboard_data: Storyboard dict with frames
        frame_uris: List of generated image URIs
        prompt_changes: Dict of {frame_idx: {"original": str, "updated": str}}

    Returns:
        Path to the created markdown file
    """
    import base64

    if not storage or not hasattr(storage, "base_path"):
        log.warning("Storage backend not available for markdown logging")
        return ""

    try:
        artifacts_dir = storage.base_path
        md_dir = artifacts_dir / "storyboards"
        md_dir.mkdir(parents=True, exist_ok=True)

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_brief = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in state.user_brief[:30])
        filename = f"storyboard_{timestamp}_{safe_brief}.md"
        md_path = md_dir / filename

        # Build markdown content
        lines = []
        lines.append(f"# Режиссёрский сценарий / Director's Cut")
        lines.append(f"\n**Создано:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Aspect Ratio:** {state.constraints.aspect_ratio}")
        lines.append(f"**Duration:** {state.constraints.duration_sec}s")
        lines.append(f"**FPS:** {state.constraints.fps}")
        lines.append("\n---\n")

        # User brief
        lines.append("## Сценарий / User Brief")
        lines.append(f"\n{state.user_brief}")
        lines.append("\n---\n")

        # World info
        if state.world_raw:
            lines.append("## Мир / World")
            world = state.world_raw
            if scene_bg := world.get("scene_background"):
                lines.append(f"\n**Локация:** {scene_bg}")
            if style := world.get("style_guide"):
                lines.append(f"**Стиль:** {style.get('style', 'N/A')}")
                if palette := style.get('palette'):
                    lines.append(f"**Палитра:** {palette}")
            lines.append("\n---\n")

        # Prompt changes section (if any)
        if prompt_changes:
            lines.append("## Изменения промптов / Prompt Changes\n")
            for idx in sorted(prompt_changes.keys()):
                change = prompt_changes[idx]
                lines.append(f"### Кадр {idx + 1}")
                lines.append(f"**До:** {change['original'][:100]}...")
                lines.append(f"**После:** {change['updated'][:100]}...")
                lines.append("")
            lines.append("---\n")

        # Storyboard frames
        if beats := storyboard_data.get("story_beats", []):
            lines.append("## Beats / Story Beats\n")
            for beat in beats:
                lines.append(f"### {beat.get('id', 'beat')} / {beat.get('label', '')}")
                lines.append(f"- **Function:** {beat.get('narrative_function', '')}")
                lines.append(f"- **Goal:** {beat.get('goal', '')}")
                lines.append(f"- **Emotion:** {beat.get('emotion', '')}")
                if visible := beat.get("visible_objects"):
                    lines.append(f"- **Visible Objects:** {', '.join(visible)}")
                if latent := beat.get("latent_objects"):
                    lines.append(f"- **Latent Objects:** {', '.join(latent)}")
                if hints := beat.get("allowed_hints"):
                    lines.append(f"- **Allowed Hints:** {', '.join(hints)}")
                if intensity := beat.get("motion_intensity"):
                    lines.append(f"- **Motion Intensity:** {intensity}")
                lines.append("")

        lines.append("## Кадры / Frames\n")
        frames = storyboard_data.get("frames", [])

        for i, frame in enumerate(frames):
            lines.append(f"### Кадр {i + 1} / Frame {i + 1}")
            lines.append("")

            # Embed image as base64
            if i < len(frame_uris) and frame_uris[i]:
                uri = frame_uris[i]
                try:
                    # Read image and convert to base64
                    local_path = storage.uri_to_local_path(uri)
                    with open(local_path, "rb") as img_file:
                        img_data = img_file.read()
                        img_ext = "png" if local_path.endswith(".png") else "jpg"
                        b64_data = base64.b64encode(img_data).decode("ascii")
                        # Embed as markdown image with full base64 data
                        lines.append(f"![Кадр {i + 1}](data:image/{img_ext};base64,{b64_data})")
                        lines.append("")
                except Exception as e:
                    log.warning(f"Failed to embed image for frame {i}: {e}")

            # Frame details
            if action := frame.get("action_in_frame"):
                lines.append(f"**Действие:** {action}")
            if camera := frame.get("camera"):
                lines.append(f"**Камера:** {camera}")
            if lighting := frame.get("lighting"):
                lines.append(f"**Свет:** {lighting}")
            if shot_size := frame.get("shot_size"):
                lines.append(f"**Shot Size:** {shot_size}")
            if camera_angle := frame.get("camera_angle"):
                lines.append(f"**Angle:** {camera_angle}")
            if lens := frame.get("lens"):
                lines.append(f"**Lens:** {lens}")
            if support := frame.get("camera_support"):
                lines.append(f"**Support:** {support}")
            if composition := frame.get("composition"):
                lines.append(f"**Composition:** {composition}")
            if blocking := frame.get("blocking"):
                lines.append(f"**Blocking:** {blocking}")
            if beat := frame.get("emotional_beat"):
                lines.append(f"**Emotional Beat:** {beat}")
            if beat_id := frame.get("beat_id"):
                lines.append(f"**Beat ID:** {beat_id}")
            if frame_class := frame.get("frame_class"):
                lines.append(f"**Frame Class:** {frame_class}")
            if anchors := frame.get("continuity_anchors"):
                lines.append(f"**Continuity Anchors:** {', '.join(anchors)}")
            if visible_ids := frame.get("visible_object_ids"):
                lines.append(f"**Visible Objects:** {', '.join(visible_ids)}")
            if hint_ids := frame.get("hint_object_ids"):
                lines.append(f"**Hint Objects:** {', '.join(hint_ids)}")
            if hidden_ids := frame.get("hidden_object_ids"):
                lines.append(f"**Hidden Objects:** {', '.join(hidden_ids)}")
            if duration := frame.get("duration_sec"):
                lines.append(f"**Длительность:** {duration}s")

            # Image prompt
            if prompt := frame.get("image_prompt"):
                lines.append(f"\n**Prompt:**\n```\n{prompt}\n```")

            # Negative prompt
            if neg := frame.get("negative"):
                if isinstance(neg, list):
                    neg = ", ".join(neg)
                lines.append(f"\n**Negative:**\n```\n{neg}\n```")

            lines.append("\n---\n")

        # Segments (if any)
        if "segments" in storyboard_data:
            lines.append("## Сегменты видео / Video Segments\n")
            for j, seg in enumerate(storyboard_data["segments"]):
                lines.append(f"### Сегмент {j + 1}")
                lines.append(f"- **Кадры:** {seg.get('start_frame_idx')} → {seg.get('end_frame_idx')}")
                lines.append(f"- **Длительность:** {seg.get('duration', seg.get('duration_sec'))}s")
                if camera_move := seg.get("camera_move"):
                    lines.append(f"- **Camera Move:** {camera_move}")
                if subject_motion := seg.get("subject_motion"):
                    lines.append(f"- **Subject Motion:** {subject_motion}")
                if env_motion := seg.get("environment_motion"):
                    lines.append(f"- **Environment Motion:** {env_motion}")
                if beats := seg.get("motion_beats"):
                    lines.append(f"- **Motion Beats:** {' | '.join(beats)}")
                if beat_id := seg.get("beat_id"):
                    lines.append(f"- **Beat ID:** {beat_id}")
                if seg_class := seg.get("segment_class"):
                    lines.append(f"- **Segment Class:** {seg_class}")
                if anchors := seg.get("continuity_anchors"):
                    lines.append(f"- **Continuity Anchors:** {', '.join(anchors)}")
                if visible_ids := seg.get("visible_object_ids"):
                    lines.append(f"- **Visible Objects:** {', '.join(visible_ids)}")
                if hint_ids := seg.get("hint_object_ids"):
                    lines.append(f"- **Hint Objects:** {', '.join(hint_ids)}")
                if hidden_ids := seg.get("hidden_object_ids"):
                    lines.append(f"- **Hidden Objects:** {', '.join(hidden_ids)}")
                if transition := seg.get("visibility_transition"):
                    lines.append(f"- **Visibility Transition:** {transition}")
                if end_match := seg.get("end_match_notes"):
                    lines.append(f"- **End Match:** {end_match}")
                if prompt := (seg.get("video_prompt") or seg.get("kling_prompt")):
                    lines.append(f"- **Prompt:** {prompt}")
                lines.append("")

        # Write to file
        md_content = "\n".join(lines)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        log.info(f"Storyboard saved to: {md_path}")
        return str(md_path)

    except Exception as e:
        log.warning(f"Failed to write storyboard markdown: {e}")
        return ""
