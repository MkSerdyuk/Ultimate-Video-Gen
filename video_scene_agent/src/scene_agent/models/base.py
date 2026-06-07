
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

class Constraints(BaseModel):
    """Generation parameters from user input or defaults."""

    aspect_ratio: str = Field(default="16:9", description="Aspect ratio, e.g. 16:9 or 9:16")
    duration_sec: float = Field(default=5.0, ge=1.0, le=120.0, description="Target scene duration")
    fps: int = Field(default=24, ge=12, le=60, description="Frames per second")
    style_tags: list[str] = Field(default_factory=list, description="Optional style hints")
    num_keyframes: Optional[int] = Field(
        default=None,
        ge=1,
        le=12,
        description="Optional explicit number of keyframes for storyboard generation",
    )
    target_duration_sec: Optional[float] = Field(
        default=None,
        ge=1.0,
        le=120.0,
        description="Optional explicit target duration exposed to prompts",
    )
    K_sb: int = Field(default=3, ge=1, le=10, description="Max storyboard review iterations")
    K_vid: int = Field(default=2, ge=1, le=5, description="Max video review iterations")
    frame_context_prev_n: int = Field(
        default=4,
        ge=0,
        le=5,
        description="How many previous keyframes to send into the vision rewriter",
    )
    vision_image_mime: str = Field(default="image/jpeg", description="MIME type for vision thumbnails")
    vision_image_max_side_px: int = Field(
        default=768,
        ge=256,
        le=2048,
        description="Max dimension for vision thumbnails",
    )
    vision_image_jpeg_quality: int = Field(
        default=80,
        ge=50,
        le=100,
        description="JPEG quality for vision thumbnails",
    )
    vision_max_images_per_request: int = Field(
        default=6,
        ge=1,
        le=10,
        description="Upper bound for images sent to one vision request",
    )
class Issue(BaseModel):
    """Canonical review issue for storyboard and video review loops."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    severity: Literal["info", "warning", "error"] = Field(
        default="warning",
        description="Issue severity",
    )
    target: str = Field(description="Target: global, beat:N, frame:N, or segment:N")
    problem: str = Field(
        validation_alias=AliasChoices("problem", "description"),
        serialization_alias="problem",
        description="Human-readable issue description",
    )
    suggestion: str = Field(default="", description="Optional suggested fix")

    @property
    def description(self) -> str:
        """Compatibility alias for the old field name."""
        return self.problem

    @field_validator("target")
    @classmethod
    def validate_target(cls, v: str) -> str:
        """Validate the target format."""
        if v == "global":
            return v

        parts = v.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid target format: {v}. Expected global, beat:N, frame:N, or segment:N")

        prefix, num_str = parts
        if prefix not in ("beat", "frame", "segment"):
            raise ValueError(f"Invalid target prefix: {prefix}. Must be beat, frame, or segment")

        try:
            index = int(num_str)
        except ValueError as exc:
            raise ValueError(f"Invalid index in target: {v}") from exc

        if index < 0:
            raise ValueError(f"Index must be non-negative, got {index}")

        return v
class RunEvent(BaseModel):
    """Structured operational event used for reports and Prefect artifact views."""

    model_config = ConfigDict(extra="ignore")

    ts: str = Field(description="UTC timestamp in ISO-8601 format")
    stage: str = Field(description="Workflow stage name")
    action: str = Field(description="Operation performed at this stage")
    asset_kind: str = Field(description="Asset category, e.g. frame, segment, review, or run")
    label: str = Field(description="Human-readable label for the event")
    from_value: Any = Field(default=None, description="Previous value, when relevant")
    to_value: Any = Field(default=None, description="New value, when relevant")
    indices: list[int] = Field(default_factory=list, description="Affected frame/segment indices")
    counts: dict[str, Any] = Field(default_factory=dict, description="Structured counters for reporting")
    mode: Optional[str] = Field(default=None, description="Review mode")
    retry: int = Field(default=0, ge=0, description="Retry count for the current task attempt")
    error: Optional[str] = Field(default=None, description="Associated error, if any")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional structured metadata")
