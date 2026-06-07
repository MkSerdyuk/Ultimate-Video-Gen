
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

class ObjectSpec(BaseModel):
    """Legacy object model kept for compatibility with the original API."""

    name: str = Field(description="Object name")
    position: str = Field(description="Position description")
    appearance: str = Field(description="Visual appearance details")
    motion: str = Field(default="", description="Motion description")
class StyleGuide(BaseModel):
    """Legacy style guide model kept for compatibility with the original API."""

    visual_style: str = Field(description="Overall visual style")
    color_palette: list[str] = Field(default_factory=list, description="Key colors")
    lighting: str = Field(description="Lighting description")
    camera_style: str = Field(description="Camera movement style")
class FrameSpec(BaseModel):
    """Legacy frame model kept for compatibility with the original API."""

    index: int = Field(ge=0, description="Frame index")
    timestamp: float = Field(ge=0.0, description="Timestamp in seconds")
    prompt: str = Field(description="Image generation prompt")
    negative_prompt: str = Field(default="", description="Negative prompt")
    camera_movement: str = Field(default="", description="Camera movement to this frame")
class SegmentSpec(BaseModel):
    """Legacy segment model kept for compatibility with the original API."""

    index: int = Field(ge=0, description="Segment index")
    start_frame_idx: int = Field(ge=0, description="Index of start frame")
    end_frame_idx: int = Field(ge=0, description="Index of end frame")
    prompt: str = Field(description="Video generation prompt")
    negative_prompt: str = Field(default="", description="Negative prompt")
    duration_sec: float = Field(ge=3.0, le=15.0, description="Target duration")
class Storyboard(BaseModel):
    """Legacy storyboard model kept for compatibility with the original API."""

    frames: list[FrameSpec] = Field(default_factory=list)
    segments: list[SegmentSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_indices(self) -> "Storyboard":
        """Validate that frames and segments line up."""
        n_frames = len(self.frames)
        n_segments = len(self.segments)

        if n_frames == 0:
            raise ValueError("Storyboard must have at least one frame")

        expected_segments = max(0, n_frames - 1)
        if n_segments != expected_segments:
            raise ValueError(
                f"Storyboard must have N-1 segments: {n_frames} frames require "
                f"{expected_segments} segments, got {n_segments}"
            )

        for i, seg in enumerate(self.segments):
            if seg.index != i:
                raise ValueError(f"Segment {i} index must be {i}, got {seg.index}")
            if seg.start_frame_idx != i:
                raise ValueError(f"Segment {i} start_frame_idx must be {i}, got {seg.start_frame_idx}")
            if seg.end_frame_idx != i + 1:
                raise ValueError(f"Segment {i} end_frame_idx must be {i + 1}, got {seg.end_frame_idx}")

        for i, frame in enumerate(self.frames):
            if frame.index != i:
                raise ValueError(f"Frame {i} index must be {i}, got {frame.index}")

        return self

    def total_duration(self) -> float:
        """Calculate total segment duration."""
        return sum(seg.duration_sec for seg in self.segments)
class IssuesOut(BaseModel):
    """Output from review nodes."""

    issues: list[Issue] = Field(default_factory=list)
    approved: bool = Field(default=False, description="True if no issues were found")
class StoryboardFix(BaseModel):
    """Legacy fix model kept for compatibility with older graph responses."""

    type: Literal["update_frame", "update_segment", "add_frame", "remove_frame"] = Field(
        description="Type of fix"
    )
    target_index: int = Field(ge=0, description="Target frame or segment index")
    updates: dict[str, Any] = Field(default_factory=dict, description="Field updates to apply")
class VideoFix(BaseModel):
    """Legacy video fix model kept for compatibility with older graph responses."""

    type: Literal["regen_frames", "regen_segments", "edit_segments", "regen_all"] = Field(
        description="Type of repair"
    )
    target_indices: list[int] = Field(default_factory=list, description="Target frame or segment indices")
class WorldPackage(BaseModel):
    """Legacy world model kept for compatibility with the original API."""

    background: str = Field(description="Background/environment description")
    objects: list[ObjectSpec] = Field(default_factory=list, description="Objects in scene")
    style: StyleGuide = Field(description="Style guide")
class SBFixOut(BaseModel):
    """Legacy storyboard fix output kept for compatibility with older graph responses."""

    fixes: list[StoryboardFix] = Field(default_factory=list)
    reasoning: str = Field(default="", description="Explanation of fixes")
class VidFixOut(BaseModel):
    """Legacy video fix output kept for compatibility with older graph responses."""

    fixes: list[VideoFix] = Field(default_factory=list)
    reasoning: str = Field(default="", description="Explanation of fixes")
