from __future__ import annotations
"""Runtime helpers for Prefect orchestration and run-scoped artifacts."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any

from scene_agent.config import Config
from scene_agent.models import (
    ArtifactManifest,
    Constraints,
    RunEvent,
    SceneRunResult,
    SceneState,
    StoryboardData,
    WorldDescription,
)
from scene_agent.prefect_artifacts import PrefectArtifactPublisher
from scene_agent.pipeline.director_tools import DirectorTools
from scene_agent.pipeline.operator_tools import OperatorTools
from scene_agent.pipeline.storyboard_editor import EditorTools
from scene_agent.pipeline.video_editor import VideoEditorTools
from scene_agent.providers import (
    KlingVideoAdapter,
    OpenRouterImageAdapter,
    OpenRouterTextAdapter,
    OpenRouterVideoReviewAdapter,
)
from scene_agent.tools.kling import KlingTool
from scene_agent.tools.openrouter_image import OpenRouterImageTool
from scene_agent.tools.openrouter_llm import OpenRouterLLM
from scene_agent.tools.openrouter_video_review import OpenRouterVideoReviewTool
from scene_agent.tools.stitch import StitchTool
from scene_agent.tools.storage import LocalStorageBackend
from scene_agent.tools.vision_rewriter import create_vision_rewriter

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeSettings:
    """Serializable runtime settings passed into Prefect tasks."""

    config: Config
    run_id: str
    user_brief: str
    constraints: Constraints


def apply_state_update(state: SceneState, update: dict[str, Any]) -> SceneState:
    """Apply a partial node/task update onto a SceneState instance."""
    for key, value in update.items():
        current = getattr(state, key, None)
        if isinstance(current, dict) and isinstance(value, dict):
            merged = dict(current)
            merged.update(value)
            setattr(state, key, merged)
        else:
            setattr(state, key, value)
    return state


class SceneAgentRuntime:
    """Per-run runtime bundling tools, adapters, storage, and the manifest."""

    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self.config = settings.config
        self.run_id = settings.run_id
        self.artifacts_dir = self.config.run_artifacts_dir(self.run_id)
        self.storage = LocalStorageBackend(
            base_path=self.artifacts_dir,
            public_url_base=self.config.run_public_url_base(self.run_id),
        )

        llm_tool = OpenRouterLLM(self.config)
        image_tool = OpenRouterImageTool(self.config, self.storage)
        kling_tool = KlingTool(
            self.config,
            self.storage,
            media_public_url_base=self.config.run_kling_media_public_url_base(self.run_id),
        )
        stitch_tool = StitchTool(self.storage, self.config.default_fps)
        video_review_tool = OpenRouterVideoReviewTool(self.config, self.storage)
        vision_rewriter = create_vision_rewriter(self.config)

        self.llm = OpenRouterTextAdapter(llm_tool)
        self.image = OpenRouterImageAdapter(image_tool)
        self.video_gen = KlingVideoAdapter(kling_tool)
        self.video_review = OpenRouterVideoReviewAdapter(video_review_tool)
        self.vision_rewriter = vision_rewriter
        self.stitch = stitch_tool
        self.prefect_artifacts = PrefectArtifactPublisher(self.run_id, self.artifacts_dir, self.storage)

        self.director_tools = DirectorTools(self.llm, self.image, self.vision_rewriter, self.storage)
        self.editor_tools = EditorTools(self.llm, self.vision_rewriter, self.storage)
        self.operator_tools = OperatorTools(self.video_gen, self.stitch)
        self.video_editor_tools = VideoEditorTools(self.llm, self.video_review)

    @property
    def manifest_path(self) -> Path:
        """Path to the run manifest."""
        return self.artifacts_dir / "manifest.json"

    def load_manifest(self) -> ArtifactManifest | None:
        """Load the persisted artifact manifest if it exists."""
        if not self.manifest_path.exists():
            return None
        return ArtifactManifest.model_validate_json(self.manifest_path.read_text(encoding="utf-8"))

    def save_manifest(self, manifest: ArtifactManifest) -> None:
        """Persist the artifact manifest."""
        self.manifest_path.write_text(
            manifest.model_dump_json(indent=2, exclude_none=True),
            encoding="utf-8",
        )

    def initial_state(self) -> SceneState:
        """Create initial state, restoring from the manifest when possible."""
        manifest = self.load_manifest()
        if manifest is None:
            return SceneState(
                user_brief=self.settings.user_brief,
                constraints=self.settings.constraints,
                run_id=self.run_id,
                artifacts_dir=str(self.artifacts_dir),
                status="running",
            )

        return SceneState(
            user_brief=self.settings.user_brief,
            constraints=self.settings.constraints,
            run_id=self.run_id,
            artifacts_dir=str(self.artifacts_dir),
            world=manifest.world,
            storyboard=manifest.storyboard,
            world_raw=manifest.world.model_dump(mode="json") if manifest.world else None,
            storyboard_raw=manifest.storyboard.model_dump(mode="json", by_alias=True)
            if manifest.storyboard
            else None,
            frame_uris=list(manifest.frame_uris),
            segment_uris=list(manifest.segment_uris),
            final_video_uri=manifest.final_video_uri,
            reviews=dict(manifest.reviews),
            provider_metadata=dict(manifest.provider_metadata),
            events=list(manifest.events),
            sb_review_mode=manifest.sb_review_mode,
            vid_review_mode=manifest.vid_review_mode,
            sb_review_error=manifest.sb_review_error,
            vid_review_error=manifest.vid_review_error,
            error=manifest.error,
            error_code=manifest.error_code,
            status=manifest.status,
        )

    def persist_state(self, state: SceneState) -> ArtifactManifest:
        """Write state into the run manifest."""
        world = state.world
        storyboard = state.storyboard

        if world is None and state.world_raw:
            try:
                world = WorldDescription.model_validate(state.world_raw)
            except Exception:
                world = None

        if storyboard is None and state.storyboard_raw:
            try:
                storyboard = StoryboardData.model_validate(state.storyboard_raw)
            except Exception:
                storyboard = None

        manifest = ArtifactManifest(
            run_id=self.run_id,
            status=state.status,
            world=world,
            storyboard=storyboard,
            frame_uris=list(state.frame_uris),
            segment_uris=list(state.segment_uris),
            final_video_uri=state.final_video_uri,
            reviews=dict(state.reviews),
            provider_metadata=dict(state.provider_metadata),
            events=list(state.events),
            sb_review_mode=state.sb_review_mode,
            vid_review_mode=state.vid_review_mode,
            sb_review_error=state.sb_review_error,
            vid_review_error=state.vid_review_error,
            error=state.error,
            error_code=state.error_code,
        )
        self.save_manifest(manifest)
        return manifest

    def save_json_artifact(self, relative_path: str, payload: Any) -> Path:
        """Persist a JSON artifact under the run directory."""
        path = self.artifacts_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        return path

    def save_text_artifact(self, relative_path: str, payload: str) -> Path:
        """Persist a UTF-8 text artifact under the run directory."""
        path = self.artifacts_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        return path

    def record_event(
        self,
        state: SceneState,
        *,
        stage: str,
        action: str,
        asset_kind: str,
        label: str,
        from_value: Any = None,
        to_value: Any = None,
        indices: list[int] | None = None,
        counts: dict[str, Any] | None = None,
        mode: str | None = None,
        retry: int = 0,
        error: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> RunEvent:
        """Append a structured operational event to the state."""
        event = RunEvent(
            ts=datetime.now(timezone.utc).isoformat(),
            stage=stage,
            action=action,
            asset_kind=asset_kind,
            label=label,
            from_value=from_value,
            to_value=to_value,
            indices=list(indices or []),
            counts=dict(counts or {}),
            mode=mode,
            retry=max(retry, 0),
            error=error,
            details=dict(details or {}),
        )
        state.events.append(event)
        return event

    def publish_prefect_artifacts(self, state: SceneState, final: bool = False) -> None:
        """Refresh curated Prefect artifacts for the current run."""
        try:
            self.prefect_artifacts.publish(state, final=final)
        except Exception:
            log.exception("Failed to publish Prefect artifacts for run %s", self.run_id)

    def build_result(self, state: SceneState) -> SceneRunResult:
        """Convert the current state into the public result envelope."""
        return SceneRunResult(
            run_id=self.run_id,
            status=state.status,
            final_video_uri=state.final_video_uri,
            storyboard=state.storyboard
            or (StoryboardData.model_validate(state.storyboard_raw) if state.storyboard_raw else None),
            world=state.world
            or (WorldDescription.model_validate(state.world_raw) if state.world_raw else None),
            frame_uris=list(state.frame_uris),
            segment_uris=list(state.segment_uris),
            reviews=dict(state.reviews),
            provider_metadata=dict(state.provider_metadata),
            events=list(state.events),
            artifacts_dir=str(self.artifacts_dir),
            sb_review_mode=state.sb_review_mode,
            vid_review_mode=state.vid_review_mode,
            error=state.error,
            error_code=state.error_code,
        )


def load_run_result_from_disk(config: Config, run_id: str) -> SceneRunResult | None:
    """Load a persisted run result from manifest.json without creating tools."""
    manifest_path = config.run_artifacts_dir(run_id) / "manifest.json"
    if not manifest_path.exists():
        return None

    manifest = ArtifactManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    return SceneRunResult(
        run_id=manifest.run_id,
        status=manifest.status,
        final_video_uri=manifest.final_video_uri,
        storyboard=manifest.storyboard,
        world=manifest.world,
        frame_uris=list(manifest.frame_uris),
        segment_uris=list(manifest.segment_uris),
        reviews=dict(manifest.reviews),
        provider_metadata=dict(manifest.provider_metadata),
        events=list(manifest.events),
        artifacts_dir=str(config.run_artifacts_dir(run_id)),
        sb_review_mode=manifest.sb_review_mode,
        vid_review_mode=manifest.vid_review_mode,
        error=manifest.error,
        error_code=manifest.error_code,
    )
