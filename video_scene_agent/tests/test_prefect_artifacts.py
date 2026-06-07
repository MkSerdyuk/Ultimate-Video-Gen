import subprocess
from pathlib import Path
from types import SimpleNamespace

from scene_agent.models import Constraints, RunEvent, SceneState
from scene_agent.prefect_artifacts import (
    PrefectArtifactPublisher,
    artifact_key,
    build_artifact_links_markdown,
    build_media_catalog_rows,
    build_review_summary_rows,
    final_video_public_url,
    render_run_report,
)
from scene_agent.tools.storage import LocalStorageBackend


def _event(**kwargs) -> RunEvent:
    base = {
        "ts": "2026-05-24T10:00:00+00:00",
        "stage": "keyframes_generate",
        "action": "completed",
        "asset_kind": "stage",
        "label": "Keyframe generation",
    }
    base.update(kwargs)
    return RunEvent(**base)


def test_artifact_key_is_prefect_safe():
    key = artifact_key("3D64A753-098B-4B78-BC7C-F8FFFD3E209A", "Run Summary")
    assert key == "scene-agent-3d64a753-098b-4b78-bc7c-f8fffd3e209a-run-summary"


def test_build_media_catalog_rows_uses_public_urls(tmp_path):
    storage = LocalStorageBackend(base_path=tmp_path, public_url_base="http://host/assets")
    frame_uri = storage.put_bytes(b"frame", "image/png", "frames/frame_0.png")
    segment_uri = storage.put_bytes(b"segment", "video/mp4", "segments/segment_0.mp4")
    final_uri = storage.put_bytes(b"final", "video/mp4", "final/final.mp4")
    state = SceneState(
        user_brief="brief",
        constraints=Constraints(),
        frame_uris=[frame_uri],
        segment_uris=[segment_uri],
        final_video_uri=final_uri,
    )

    rows = build_media_catalog_rows(state, Path(tmp_path), storage)

    assert [row["label"] for row in rows] == ["Keyframe 1", "Segment 1", "Final video"]
    assert rows[0]["public_url"] == "http://host/assets/frames/frame_0.png"
    assert rows[2]["access"] == "public"


def test_build_review_rows_and_report_include_regens_and_retries():
    state = SceneState(
        user_brief="A girl runs from the water",
        constraints=Constraints(duration_sec=30.0, num_keyframes=4),
        run_id="run-1",
        frame_uris=["file:///frame-0.png"],
        segment_uris=["file:///segment-0.mp4"],
        final_video_uri="file:///final.mp4",
        sb_iteration=2,
        vid_iteration=1,
        status="completed",
        error=None,
    )
    state.events = [
        _event(
            stage="storyboard_review",
            action="completed",
            asset_kind="review",
            label="Storyboard review #1",
            counts={"iteration": 1, "issues": 2},
            mode="multimodal",
        ),
        _event(
            stage="storyboard_fix",
            action="completed",
            asset_kind="fix",
            label="Storyboard fix #1",
            counts={"iteration": 1, "issues": 2},
            details={"regen_frames": [0, 1], "regen_segments": []},
        ),
        _event(
            stage="keyframes_generate",
            action="replaced",
            asset_kind="frame",
            label="Keyframe 1",
            from_value="file:///old.png",
            to_value="file:///new.png",
            indices=[0],
            retry=1,
        ),
        _event(
            stage="segments_generate",
            action="replaced",
            asset_kind="segment",
            label="Segment 1: frame 1 -> frame 2",
            from_value="file:///segment-old.mp4",
            to_value="file:///segment-new.mp4",
            indices=[0],
        ),
        _event(
            stage="video_review",
            action="completed",
            asset_kind="review",
            label="Video review #1",
            counts={"iteration": 1, "issues": 0},
            mode="multimodal",
        ),
        _event(
            stage="keyframes_generate",
            action="completed",
            asset_kind="stage",
            label="Keyframe generation",
            counts={"frames_changed": 1},
            retry=1,
        ),
    ]

    review_rows = build_review_summary_rows(state.events)
    report = render_run_report(state)

    assert review_rows[0]["label"] == "Storyboard review #1"
    assert review_rows[1]["regen_frames"] == "0, 1"
    assert "Keyframe 1: `old.png` -> `new.png` (1 updates)" in report
    assert "Segment 1: frame 1 -> frame 2: `segment-old.mp4` -> `segment-new.mp4` (1 updates)" in report
    assert "keyframes_generate: 1 retries" in report
    assert "mode=multimodal" in report


def test_build_artifact_links_markdown_marks_local_only(tmp_path):
    storage = LocalStorageBackend(base_path=tmp_path)
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "world.json").write_text("{}", encoding="utf-8")
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "run-report.md").write_text("# Report", encoding="utf-8")

    markdown = build_artifact_links_markdown(Path(tmp_path), storage, final=True)

    assert "Run is in a terminal state" in markdown
    assert "manifest.json" in markdown
    assert "local-only" in markdown


def test_final_video_public_url_uses_run_scoped_storage_base(tmp_path):
    run_dir = tmp_path / "runs" / "run-1"
    storage = LocalStorageBackend(base_path=run_dir, public_url_base="https://host/artifacts/runs/run-1")
    final_uri = storage.put_bytes(b"final", "video/mp4", "final_video_run-1.mp4")
    state = SceneState(user_brief="brief", constraints=Constraints(), final_video_uri=final_uri)

    assert final_video_public_url(state, run_dir, storage) == (
        "https://host/artifacts/runs/run-1/final_video_run-1.mp4"
    )


def test_publish_final_video_link_and_poster_artifacts(monkeypatch, tmp_path):
    storage = LocalStorageBackend(base_path=tmp_path, public_url_base="https://host/artifacts/runs/run-1")
    final_uri = storage.put_bytes(b"fake mp4", "video/mp4", "final_video_run-1.mp4")
    state = SceneState(
        user_brief="brief",
        constraints=Constraints(),
        final_video_uri=final_uri,
        status="completed",
    )
    created: dict[str, list[dict]] = {
        "markdown": [],
        "table": [],
        "link": [],
        "image": [],
    }

    monkeypatch.setattr("scene_agent.prefect_artifacts.publisher.Artifact.get", lambda key: None)
    monkeypatch.setattr(
        "scene_agent.prefect_artifacts.publisher.create_markdown_artifact",
        lambda **kwargs: created["markdown"].append(kwargs),
    )
    monkeypatch.setattr(
        "scene_agent.prefect_artifacts.publisher.create_table_artifact",
        lambda **kwargs: created["table"].append(kwargs),
    )
    monkeypatch.setattr(
        "scene_agent.prefect_artifacts.publisher.create_link_artifact",
        lambda **kwargs: created["link"].append(kwargs),
    )
    monkeypatch.setattr(
        "scene_agent.prefect_artifacts.publisher.create_image_artifact",
        lambda **kwargs: created["image"].append(kwargs),
    )

    def fake_ffmpeg(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"poster")
        return SimpleNamespace(stderr=b"")

    monkeypatch.setattr("scene_agent.prefect_artifacts.publisher.subprocess.run", fake_ffmpeg)

    PrefectArtifactPublisher("run-1", tmp_path, storage).publish(state, final=True)

    assert created["link"][0]["key"] == artifact_key("run-1", "final-video")
    assert created["link"][0]["link"] == "https://host/artifacts/runs/run-1/final_video_run-1.mp4"
    assert created["link"][0]["link_text"] == "Open final video"
    assert created["image"][0]["key"] == artifact_key("run-1", "final-video-poster")
    assert created["image"][0]["image_url"] == "https://host/artifacts/runs/run-1/previews/final-video-poster.jpg"
    assert (tmp_path / "previews" / "final-video-poster.jpg").exists()


def test_publish_final_video_link_survives_poster_failure(monkeypatch, tmp_path):
    storage = LocalStorageBackend(base_path=tmp_path, public_url_base="https://host/artifacts/runs/run-1")
    final_uri = storage.put_bytes(b"fake mp4", "video/mp4", "final_video_run-1.mp4")
    state = SceneState(
        user_brief="brief",
        constraints=Constraints(),
        final_video_uri=final_uri,
        status="completed",
    )
    created: dict[str, list[dict]] = {"markdown": [], "table": [], "link": [], "image": []}

    monkeypatch.setattr("scene_agent.prefect_artifacts.publisher.Artifact.get", lambda key: None)
    monkeypatch.setattr(
        "scene_agent.prefect_artifacts.publisher.create_markdown_artifact",
        lambda **kwargs: created["markdown"].append(kwargs),
    )
    monkeypatch.setattr(
        "scene_agent.prefect_artifacts.publisher.create_table_artifact",
        lambda **kwargs: created["table"].append(kwargs),
    )
    monkeypatch.setattr(
        "scene_agent.prefect_artifacts.publisher.create_link_artifact",
        lambda **kwargs: created["link"].append(kwargs),
    )
    monkeypatch.setattr(
        "scene_agent.prefect_artifacts.publisher.create_image_artifact",
        lambda **kwargs: created["image"].append(kwargs),
    )

    def failed_ffmpeg(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr=b"invalid video")

    monkeypatch.setattr("scene_agent.prefect_artifacts.publisher.subprocess.run", failed_ffmpeg)

    PrefectArtifactPublisher("run-1", tmp_path, storage).publish(state, final=True)

    assert created["link"][0]["link"] == "https://host/artifacts/runs/run-1/final_video_run-1.mp4"
    assert created["image"] == []
