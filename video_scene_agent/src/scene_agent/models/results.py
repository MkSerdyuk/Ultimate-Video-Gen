
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from scene_agent.models.base import Constraints, RunEvent
from scene_agent.models.storyboard import StoryboardData, WorldDescription

class StoryboardFixResult(BaseModel):
    """Canonical output from storyboard fix."""

    model_config = ConfigDict(extra="ignore")

    storyboard: StoryboardData
    regen_frames: list[int] = Field(default_factory=list)
    reasoning: str = Field(default="", description="Optional reasoning emitted by the model")
class VideoFixResult(BaseModel):
    """Canonical output from video fix."""

    model_config = ConfigDict(extra="ignore")

    storyboard: Optional[StoryboardData] = None
    edit_segments: list[int] = Field(default_factory=list)
    regen_segments: list[int] = Field(default_factory=list)
    regen_frames: list[int] = Field(default_factory=list)
    regen_all: bool = Field(default=False)
    reasoning: str = Field(default="", description="Optional reasoning emitted by the model")
class ArtifactManifest(BaseModel):
    """Persistent manifest stored alongside generated run artifacts."""

    model_config = ConfigDict(extra="ignore")

    run_id: str
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    world: Optional[WorldDescription] = None
    storyboard: Optional[StoryboardData] = None
    frame_uris: list[str] = Field(default_factory=list)
    segment_uris: list[str] = Field(default_factory=list)
    final_video_uri: Optional[str] = None
    reviews: dict[str, Any] = Field(default_factory=dict)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)
    events: list[RunEvent] = Field(default_factory=list)
    sb_review_mode: str = "not_run"
    vid_review_mode: str = "not_run"
    sb_review_error: Optional[str] = None
    vid_review_error: Optional[str] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
class SceneRunResult(BaseModel):
    """Public result envelope returned by the new Prefect-based workflow."""

    model_config = ConfigDict(extra="ignore")

    run_id: str
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    final_video_uri: Optional[str] = None
    storyboard: Optional[StoryboardData] = None
    world: Optional[WorldDescription] = None
    frame_uris: list[str] = Field(default_factory=list)
    segment_uris: list[str] = Field(default_factory=list)
    reviews: dict[str, Any] = Field(default_factory=dict)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)
    events: list[RunEvent] = Field(default_factory=list)
    artifacts_dir: Optional[str] = None
    sb_review_mode: str = "not_run"
    vid_review_mode: str = "not_run"
    error: Optional[str] = None
    error_code: Optional[str] = None
class ProviderErrorEnvelope(BaseModel):
    """Normalized provider error emitted by adapters and orchestration tasks."""

    provider: str
    operation: str
    retryable: bool = Field(default=False)
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
class SceneState(BaseModel):
    """State container shared by Prefect tasks and pipeline helpers."""

    model_config = ConfigDict(extra="ignore")

    user_brief: str = Field(description="User's text description")
    constraints: Constraints = Field(default_factory=Constraints)
    run_id: Optional[str] = None
    artifacts_dir: Optional[str] = None

    world: Optional[WorldDescription] = None
    storyboard: Optional[StoryboardData] = None

    world_raw: Optional[dict[str, Any]] = None
    storyboard_raw: Optional[dict[str, Any]] = None

    frame_uris: list[str] = Field(default_factory=list, description="URIs of generated keyframes")
    segment_uris: list[str] = Field(default_factory=list, description="URIs of generated video segments")
    final_video_uri: Optional[str] = None

    sb_iteration: int = Field(default=0, description="Storyboard review iteration count")
    vid_iteration: int = Field(default=0, description="Video review iteration count")
    sb_issues: list[dict[str, Any]] = Field(default_factory=list, description="Storyboard review issues")
    vid_issues: list[dict[str, Any]] = Field(default_factory=list, description="Video review issues")
    regen_frames: list[int] = Field(default_factory=list, description="Frame indices to regenerate")
    regen_segments: list[int] = Field(default_factory=list, description="Segment indices to regenerate")
    edit_segments: list[int] = Field(default_factory=list, description="Segment indices to repair with video-to-video")
    prompt_changes: dict[int, dict[str, str]] = Field(default_factory=dict)
    reviews: dict[str, Any] = Field(default_factory=dict)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)
    events: list[RunEvent] = Field(default_factory=list)
    sb_review_mode: str = "not_run"
    vid_review_mode: str = "not_run"
    sb_review_error: Optional[str] = None
    vid_review_error: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    status: Literal["pending", "running", "completed", "failed"] = "pending"
