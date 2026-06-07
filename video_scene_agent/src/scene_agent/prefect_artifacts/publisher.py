
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from prefect.artifacts import (
    Artifact,
    create_image_artifact,
    create_link_artifact,
    create_markdown_artifact,
    create_table_artifact,
)
from prefect.client.orchestration import get_client
from prefect.client.schemas.actions import ArtifactUpdate

from scene_agent.models import SceneState
from scene_agent.prefect_artifacts.keys import artifact_key
from scene_agent.prefect_artifacts.links import build_artifact_links_markdown, final_video_public_url
from scene_agent.prefect_artifacts.paths import _path_to_public_url, _uri_to_local_path
from scene_agent.prefect_artifacts.report import render_run_report
from scene_agent.prefect_artifacts.tables import build_media_catalog_rows, build_review_summary_rows

log = logging.getLogger(__name__)

class PrefectArtifactPublisher:
    """Publishes a compact set of run-scoped artifacts into the Prefect UI."""

    def __init__(self, run_id: str, artifacts_dir: Path, storage: Any) -> None:
        self.run_id = run_id
        self.artifacts_dir = Path(artifacts_dir)
        self.storage = storage
        self._client = None

    def publish(self, state: SceneState, final: bool = False) -> None:
        """Write or update the curated artifact set for this run."""
        report_markdown = render_run_report(state)
        self._write_text_file("reports/run-report.md", report_markdown)

        self._upsert_markdown(
            key=artifact_key(self.run_id, "summary"),
            markdown=report_markdown,
            description="Operational summary and final automatic report for this run.",
        )
        self._upsert_table(
            key=artifact_key(self.run_id, "media"),
            table=build_media_catalog_rows(state, self.artifacts_dir, self.storage),
            description="Generated keyframes, video segments, and the final stitched output.",
        )
        self._upsert_table(
            key=artifact_key(self.run_id, "reviews"),
            table=build_review_summary_rows(state.events),
            description="Storyboard and video review iterations, review modes, and fix actions.",
        )
        if final:
            self._publish_final_video_artifacts(state)
        self._upsert_markdown(
            key=artifact_key(self.run_id, "links"),
            markdown=build_artifact_links_markdown(self.artifacts_dir, self.storage, final=final),
            description="Structured output files for this run: manifests, JSONs, storyboards, reviews, and reports.",
        )

    def _write_text_file(self, relative_path: str, text: str) -> Path:
        path = self.artifacts_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def _sync_client(self):
        if self._client is None:
            self._client = get_client(sync_client=True)
        return self._client

    def _upsert_markdown(self, key: str, markdown: str, description: str) -> None:
        existing = Artifact.get(key=key)
        if existing is None:
            create_markdown_artifact(markdown=markdown, key=key, description=description)
            return

        client = self._sync_client()
        client.update_artifact(
            artifact_id=existing.id,
            artifact=ArtifactUpdate(data=markdown, description=description),
        )

    def _upsert_table(self, key: str, table: list[dict[str, Any]], description: str) -> None:
        existing = Artifact.get(key=key)
        if existing is None:
            create_table_artifact(table=table, key=key, description=description)
            return

        client = self._sync_client()
        client.update_artifact(
            artifact_id=existing.id,
            artifact=ArtifactUpdate(data=table, description=description),
        )

    def _upsert_link(self, key: str, link: str, link_text: str, description: str) -> None:
        existing = Artifact.get(key=key)
        if existing is None:
            create_link_artifact(link=link, link_text=link_text, key=key, description=description)
            return

        client = self._sync_client()
        client.update_artifact(
            artifact_id=existing.id,
            artifact=ArtifactUpdate(data=f"[{link_text}]({link})", description=description),
        )

    def _upsert_image(self, key: str, image_url: str, description: str) -> None:
        existing = Artifact.get(key=key)
        if existing is None:
            create_image_artifact(image_url=image_url, key=key, description=description)
            return

        client = self._sync_client()
        client.update_artifact(
            artifact_id=existing.id,
            artifact=ArtifactUpdate(data=image_url, description=description),
        )

    def _publish_final_video_artifacts(self, state: SceneState) -> None:
        final_url = final_video_public_url(state, self.artifacts_dir, self.storage)
        if not final_url:
            log.info("Skipping final video link artifact for run %s because no public URL is available", self.run_id)
            return

        self._upsert_link(
            key=artifact_key(self.run_id, "final-video"),
            link=final_url,
            link_text="Open final video",
            description="Authenticated link to the final stitched MP4 for this run.",
        )

        poster_url = self._ensure_final_video_poster(state)
        if poster_url:
            self._upsert_image(
                key=artifact_key(self.run_id, "final-video-poster"),
                image_url=poster_url,
                description="Poster frame extracted from the final stitched video.",
            )

    def _ensure_final_video_poster(self, state: SceneState) -> str:
        local_path = _uri_to_local_path(self.storage, state.final_video_uri)
        if local_path is None or not local_path.exists():
            log.warning("Cannot create final video poster for run %s because the MP4 is not local", self.run_id)
            return ""

        poster_path = self.artifacts_dir / "previews" / "final-video-poster.jpg"
        poster_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(local_path),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "3",
                    str(poster_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except FileNotFoundError:
            log.warning("Cannot create final video poster for run %s because ffmpeg is not installed", self.run_id)
            return ""
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            log.warning("Cannot create final video poster for run %s: %s", self.run_id, stderr[-500:])
            return ""

        if not poster_path.exists():
            log.warning("Cannot create final video poster for run %s because ffmpeg produced no output", self.run_id)
            return ""
        return _path_to_public_url(self.artifacts_dir, self.storage, poster_path)
