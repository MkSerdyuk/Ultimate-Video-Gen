from __future__ import annotations
"""Video stitching tool using FFmpeg."""

import os
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Optional

from scene_agent.tools.storage import StorageBackend

log = logging.getLogger(__name__)


class StitchTool:
    """
    Video stitching tool using FFmpeg.

    Concatenates multiple video segments into a single video.
    """

    def __init__(self, storage: StorageBackend, fps: int = 24):
        """
        Initialize stitch tool.

        Args:
            storage: Storage backend for reading/writing videos
            fps: Target FPS for output video
        """
        self.storage = storage
        self.fps = fps

        # Check FFmpeg availability
        self._ffmpeg_available = self._check_ffmpeg()

        if not self._ffmpeg_available:
            log.warning("FFmpeg not available. Video stitching will fail.")

    def stitch(
        self,
        segment_uris: list[str],
        fps: int | None = None,
        output_key: str | None = None,
    ) -> str:
        """
        Stitch multiple video segments together.

        Args:
            segment_uris: List of video segment URIs
            fps: Override default FPS
            output_key: Optional storage key for output

        Returns:
            URI of the stitched video

        Raises:
            RuntimeError: If FFmpeg is not available
            ValueError: If no segments provided
        """
        if not self._ffmpeg_available:
            raise RuntimeError("FFmpeg is not available. Install FFmpeg to use StitchTool.")

        if not segment_uris:
            raise ValueError("No segments provided for stitching")

        fps = fps or self.fps
        log.info(f"Stitching {len(segment_uris)} segments at {fps} FPS")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Download all segments to temp directory
            segment_files = []
            for i, uri in enumerate(segment_uris):
                dest_path = temp_path / f"segment_{i:03d}.mp4"
                self.storage.download_to_file(uri, str(dest_path))
                segment_files.append(dest_path)

            # Create concat file
            concat_file = temp_path / "concat.txt"
            with open(concat_file, "w") as f:
                for seg_file in segment_files:
                    # Escape single quotes in filename
                    safe_path = str(seg_file).replace("'", "'\\''")
                    f.write(f"file '{safe_path}'\n")

            # Output file
            output_path = temp_path / "output.mp4"

            # Run FFmpeg
            try:
                self._run_ffmpeg(str(concat_file), str(output_path), fps)
            except subprocess.CalledProcessError:
                log.warning("Stream-copy concat failed; retrying stitch with re-encode")
                self._run_ffmpeg_reencode(segment_files, output_path, fps)

            # Store result
            with open(output_path, "rb") as f:
                video_data = f.read()

            uri = self.storage.put_bytes(video_data, "video/mp4", output_key)

            log.info(f"Stitched video saved: {uri}")
            return uri

    def _run_ffmpeg(self, concat_file: str, output_path: str, fps: int) -> None:
        """
        Run FFmpeg to concatenate videos.

        Args:
            concat_file: Path to concat.txt file
            output_path: Path for output video
            fps: Target FPS

        Raises:
            subprocess.CalledProcessError: If FFmpeg fails
        """
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",  # Copy streams without re-encoding
            "-y",  # Overwrite output file
            output_path,
        ]

        log.debug(f"Running FFmpeg: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                log.error(f"FFmpeg stderr: {result.stderr}")
                raise subprocess.CalledProcessError(
                    result.returncode, cmd, result.stdout, result.stderr
                )

            log.debug(f"FFmpeg output: {result.stdout}")

        except subprocess.TimeoutExpired:
            raise RuntimeError("FFmpeg timed out")
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg.")

    def _probe_video_size(self, path: Path) -> tuple[int, int]:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            log.warning("FFprobe failed while probing %s: %s", path, result.stderr)
            return (1280, 720)

        import json

        data = json.loads(result.stdout or "{}")
        stream = (data.get("streams") or [{}])[0]
        width = int(stream.get("width") or 1280)
        height = int(stream.get("height") or 720)
        return (max(2, width // 2 * 2), max(2, height // 2 * 2))

    def _run_ffmpeg_reencode(self, segment_files: list[Path], output_path: Path, fps: int) -> None:
        """Concatenate segments through a normalizing re-encode path."""
        if not segment_files:
            raise ValueError("No segment files provided for FFmpeg re-encode")

        width, height = self._probe_video_size(segment_files[0])
        inputs: list[str] = []
        filters: list[str] = []
        labels: list[str] = []
        for idx, seg_file in enumerate(segment_files):
            inputs.extend(["-i", str(seg_file)])
            label = f"v{idx}"
            labels.append(f"[{label}]")
            filters.append(
                f"[{idx}:v]fps={fps},"
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
                f"setsar=1,format=yuv420p[{label}]"
            )

        filters.append(f"{''.join(labels)}concat=n={len(segment_files)}:v=1:a=0[v]")
        cmd = [
            "ffmpeg",
            *inputs,
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            "-y",
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            log.error("FFmpeg re-encode stderr: %s", result.stderr)
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

    def create_still_clip(
        self,
        image_uri: str,
        duration_sec: float,
        fps: int | None = None,
        output_key: str | None = None,
    ) -> str:
        """Create a silent mp4 clip from one still image."""
        if not self._ffmpeg_available:
            raise RuntimeError("FFmpeg is not available. Install FFmpeg to create placeholder clips.")

        fps = fps or self.fps
        duration = max(0.1, float(duration_sec))

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_suffix = Path(self.storage.uri_to_local_path(image_uri)).suffix or ".png"
            source_path = temp_path / f"source{source_suffix}"
            output_path = temp_path / "still.mp4"
            self.storage.download_to_file(image_uri, str(source_path))

            cmd = [
                "ffmpeg",
                "-loop",
                "1",
                "-framerate",
                str(fps),
                "-i",
                str(source_path),
                "-t",
                f"{duration:.3f}",
                "-vf",
                f"fps={fps},scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                "-y",
                str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=max(30, int(duration) + 30))
            if result.returncode != 0:
                log.error(f"FFmpeg still-clip stderr: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)

            with open(output_path, "rb") as f:
                return self.storage.put_bytes(f.read(), "video/mp4", output_key)

    def stitch_with_transitions(
        self,
        segment_uris: list[str],
        transition_duration: float = 0.5,
        transition_type: str = "fade",
        fps: int | None = None,
    ) -> str:
        """
        Stitch videos with transitions between segments.

        This requires re-encoding and is slower than simple stitching.

        Args:
            segment_uris: List of video segment URIs
            transition_duration: Duration of transition in seconds
            transition_type: Type of transition ("fade", "crossfade")
            fps: Target FPS

        Returns:
            URI of the stitched video
        """
        if not self._ffmpeg_available:
            raise RuntimeError("FFmpeg is not available")

        if len(segment_uris) < 2:
            # No transitions needed for single segment
            return segment_uris[0] if segment_uris else ""

        fps = fps or self.fps
        log.info(f"Stitching with {transition_type} transitions ({transition_duration}s)")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Download segments
            segment_files = []
            for i, uri in enumerate(segment_uris):
                dest_path = temp_path / f"segment_{i:03d}.mp4"
                self.storage.download_to_file(uri, str(dest_path))
                segment_files.append(dest_path)

            # Build complex filter for crossfade
            output_path = temp_path / "output.mp4"
            self._run_ffmpeg_with_transitions(
                segment_files, output_path, transition_duration, fps
            )

            # Store result
            with open(output_path, "rb") as f:
                video_data = f.read()

            return self.storage.put_bytes(video_data, "video/mp4")

    def _run_ffmpeg_with_transitions(
        self,
        segment_files: list[Path],
        output_path: Path,
        transition_duration: float,
        fps: int,
    ) -> None:
        """Run FFmpeg with xfade filter for transitions."""
        # Build xfade filter chain
        # This is a simplified version - real implementation would be more complex
        filter_parts = []
        inputs = []

        for i, seg_file in enumerate(segment_files):
            inputs.extend(["-i", str(seg_file)])

        # For N segments, we need N-1 transitions
        # xfade filter format: [0:v][1:v]xfade=transition=fade:duration=0.5:offset=4.5[v1]
        current_offset = 0.0

        for i in range(len(segment_files) - 1):
            # Calculate offset (duration of first segment minus transition)
            # We'd need to get actual segment durations for this to work properly
            offset = current_offset + 4.5  # Simplified - assumes ~5s segments

            if i == 0:
                filter_parts.append(
                    f"[0:v][1:v]xfade=transition=fade:duration={transition_duration}:offset={offset}[v{i}]"
                )
            else:
                filter_parts.append(
                    f"[v{i-1}][{i+1}:v]xfade=transition=fade:duration={transition_duration}:offset={offset}[v{i}]"
                )

            current_offset = offset - transition_duration

        filter_complex = ";".join(filter_parts)

        cmd = [
            "ffmpeg",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", f"[v{len(segment_files)-2}]",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-y",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            raise RuntimeError("FFmpeg with transitions timed out")

    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def get_video_info(self, uri: str) -> dict:
        """
        Get information about a video file.

        Args:
            uri: Video URI

        Returns:
            Dict with duration, fps, width, height, codec
        """
        if not self._ffmpeg_available:
            return {}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "video.mp4"
            self.storage.download_to_file(uri, str(temp_path))

            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,codec_name",
                "-show_entries", "format=duration",
                "-of", "json",
                str(temp_path),
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    import json
                    return json.loads(result.stdout)
            except Exception as e:
                log.warning(f"Failed to get video info: {e}")

        return {}


def create_stitch_tool(storage: StorageBackend, fps: int = 24) -> StitchTool:
    """Factory function to create stitch tool."""
    return StitchTool(storage, fps)
