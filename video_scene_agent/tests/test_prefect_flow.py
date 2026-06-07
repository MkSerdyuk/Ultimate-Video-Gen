"""Offline tests for the Prefect-based orchestration path."""

from pathlib import Path

import pytest

from scene_agent.config import Config
from scene_agent.models import Constraints, StoryStyleGuide, StoryboardData, StoryboardFrameData, StoryboardSegmentData, WorldDescription
from scene_agent.prefect_flows import generate_scene_flow


def _world() -> WorldDescription:
    return WorldDescription(
        scene_background="Night city",
        objects=[],
        style_guide=StoryStyleGuide(style="neo noir", palette="blue", global_negative=["text"]),
    )


def _storyboard() -> StoryboardData:
    return StoryboardData(
        scene_background="Night city",
        objects=[],
        style_guide=StoryStyleGuide(style="neo noir", palette="blue", global_negative=["text"]),
        frames=[
            StoryboardFrameData(idx=0, image_prompt="frame 0"),
            StoryboardFrameData(idx=1, image_prompt="frame 1"),
        ],
        segments=[
            StoryboardSegmentData(
                idx=0,
                start_frame_idx=0,
                end_frame_idx=1,
                duration=3.0,
                video_prompt="bridge",
            )
        ],
    )


class TestPrefectFlow:
    def test_generate_scene_flow_completes_and_persists_artifacts(self, monkeypatch, tmp_path):
        world = _world()
        storyboard = _storyboard()

        monkeypatch.setattr(
            "scene_agent.flows.tasks.director.director_world",
            lambda state, tools: {"world": world, "world_raw": world.model_dump(mode="json")},
        )
        monkeypatch.setattr(
            "scene_agent.flows.tasks.director.director_storyboard",
            lambda state, tools: {
                "storyboard": storyboard,
                "storyboard_raw": storyboard.model_dump(mode="json", by_alias=True),
            },
        )
        monkeypatch.setattr(
            "scene_agent.flows.tasks.director.keyframes_generate",
            lambda state, tools: {
                "frame_uris": ["file:///frame-0.png", "file:///frame-1.png"],
                "storyboard": storyboard,
                "storyboard_raw": storyboard.model_dump(mode="json", by_alias=True),
            },
        )
        monkeypatch.setattr(
            "scene_agent.flows.tasks.storyboard.sb_editor_review",
            lambda state, tools: {"sb_issues": [], "sb_iteration": state.sb_iteration + 1},
        )
        monkeypatch.setattr(
            "scene_agent.flows.tasks.video.segments_generate",
            lambda state, tools: {
                "segment_uris": ["file:///segment-0.mp4"],
                "storyboard": storyboard,
                "storyboard_raw": {
                    **storyboard.model_dump(mode="json", by_alias=True),
                    "segments": [{**storyboard.segments[0].model_dump(mode="json", by_alias=True), "result_uri": "file:///segment-0.mp4"}],
                },
                "regen_segments": [],
            },
        )
        monkeypatch.setattr(
            "scene_agent.flows.tasks.video.stitch_video",
            lambda state, tools: {"final_video_uri": "file:///final.mp4"},
        )
        monkeypatch.setattr(
            "scene_agent.flows.tasks.video.vid_editor_review",
            lambda state, tools: {"vid_issues": [], "vid_iteration": state.vid_iteration + 1},
        )

        result = generate_scene_flow(
            "brief",
            constraints=Constraints(),
            config=Config(openrouter_api_key="test-key", storage_path=str(tmp_path)),
            run_options={"run_id": "flow-test"},
        )

        assert result.status == "completed"
        assert result.final_video_uri == "file:///final.mp4"
        assert Path(result.artifacts_dir).exists()
        assert (Path(result.artifacts_dir) / "manifest.json").exists()
        assert (Path(result.artifacts_dir) / "world.json").exists()
        assert (Path(result.artifacts_dir) / "storyboard.json").exists()
        assert (Path(result.artifacts_dir) / "reports" / "run-report.md").exists()

    def test_generate_scene_flow_resumes_from_manifest(self, monkeypatch, tmp_path):
        world = _world()
        storyboard = _storyboard()
        calls = {"director_world": 0, "director_storyboard": 0}

        def fake_world(state, tools):
            calls["director_world"] += 1
            return {"world": world, "world_raw": world.model_dump(mode="json")}

        def fake_storyboard(state, tools):
            calls["director_storyboard"] += 1
            return {
                "storyboard": storyboard,
                "storyboard_raw": storyboard.model_dump(mode="json", by_alias=True),
            }

        monkeypatch.setattr("scene_agent.flows.tasks.director.director_world", fake_world)
        monkeypatch.setattr("scene_agent.flows.tasks.director.director_storyboard", fake_storyboard)
        monkeypatch.setattr(
            "scene_agent.flows.tasks.director.keyframes_generate",
            lambda state, tools: {
                "frame_uris": ["file:///frame-0.png", "file:///frame-1.png"],
                "storyboard": storyboard,
                "storyboard_raw": storyboard.model_dump(mode="json", by_alias=True),
            },
        )
        monkeypatch.setattr(
            "scene_agent.flows.tasks.storyboard.sb_editor_review",
            lambda state, tools: {"sb_issues": [], "sb_iteration": state.sb_iteration + 1},
        )
        monkeypatch.setattr(
            "scene_agent.flows.tasks.video.segments_generate",
            lambda state, tools: {
                "segment_uris": ["file:///segment-0.mp4"],
                "storyboard": storyboard,
                "storyboard_raw": storyboard.model_dump(mode="json", by_alias=True),
                "regen_segments": [],
            },
        )
        monkeypatch.setattr(
            "scene_agent.flows.tasks.video.stitch_video",
            lambda state, tools: {"final_video_uri": "file:///final.mp4"},
        )
        monkeypatch.setattr(
            "scene_agent.flows.tasks.video.vid_editor_review",
            lambda state, tools: {"vid_issues": [], "vid_iteration": state.vid_iteration + 1},
        )

        config = Config(openrouter_api_key="test-key", storage_path=str(tmp_path))

        result1 = generate_scene_flow("brief", config=config, run_options={"run_id": "resume-test"})
        result2 = generate_scene_flow("brief", config=config, run_options={"run_id": "resume-test"})

        assert result1.status == "completed"
        assert result2.status == "completed"
        assert calls["director_world"] == 1
        assert calls["director_storyboard"] == 1

    def test_generate_scene_flow_routes_video_fix_edit_segments(self, monkeypatch, tmp_path):
        world = _world()
        storyboard = _storyboard()
        calls = {"segments_edit": 0, "stitch": 0}

        def fake_world(settings, state):
            state.world = world
            state.world_raw = world.model_dump(mode="json")
            return state

        def fake_storyboard(settings, state):
            state.storyboard = storyboard
            state.storyboard_raw = storyboard.model_dump(mode="json", by_alias=True)
            return state

        def fake_keyframes(settings, state):
            state.frame_uris = ["file:///frame-0.png", "file:///frame-1.png"]
            return state

        def fake_segments(settings, state):
            state.segment_uris = ["file:///segment-0.mp4"]
            state.regen_segments = []
            return state

        def fake_stitch(settings, state):
            calls["stitch"] += 1
            state.final_video_uri = f"file:///final-{calls['stitch']}.mp4"
            return state

        def fake_video_review(settings, state):
            if state.vid_iteration == 0:
                state.vid_issues = [{"target": "segment:0", "severity": "warning", "problem": "motion drift"}]
            else:
                state.vid_issues = []
            state.vid_iteration += 1
            return state

        def fake_video_fix(settings, state):
            state.vid_issues = []
            state.edit_segments = [0]
            return state

        def fake_segments_edit(settings, state):
            calls["segments_edit"] += 1
            state.segment_uris = ["file:///segment-0-edited.mp4"]
            state.edit_segments = []
            return state

        monkeypatch.setattr("scene_agent.prefect_flows.director_world_task", fake_world)
        monkeypatch.setattr("scene_agent.prefect_flows.director_storyboard_task", fake_storyboard)
        monkeypatch.setattr("scene_agent.prefect_flows.keyframes_task", fake_keyframes)
        monkeypatch.setattr("scene_agent.prefect_flows.storyboard_review_task", lambda settings, state: state)
        monkeypatch.setattr("scene_agent.prefect_flows.segments_task", fake_segments)
        monkeypatch.setattr("scene_agent.prefect_flows.stitch_task", fake_stitch)
        monkeypatch.setattr("scene_agent.prefect_flows.video_review_task", fake_video_review)
        monkeypatch.setattr("scene_agent.prefect_flows.video_fix_task", fake_video_fix)
        monkeypatch.setattr("scene_agent.prefect_flows.segments_edit_task", fake_segments_edit)

        result = generate_scene_flow(
            "brief",
            constraints=Constraints(),
            config=Config(openrouter_api_key="test-key", storage_path=str(tmp_path)),
            run_options={"run_id": "edit-flow-test"},
        )

        assert result.status == "completed"
        assert result.segment_uris == ["file:///segment-0-edited.mp4"]
        assert result.final_video_uri == "file:///final-2.mp4"
        assert calls["segments_edit"] == 1

    def test_generate_scene_flow_regen_frames_only_rerenders_touched_segments(self, monkeypatch, tmp_path):
        world = _world()
        storyboard = StoryboardData(
            scene_background="Night city",
            objects=[],
            style_guide=StoryStyleGuide(style="neo noir", palette="blue", global_negative=["text"]),
            frames=[
                StoryboardFrameData(idx=0, image_prompt="frame 0"),
                StoryboardFrameData(idx=1, image_prompt="frame 1"),
                StoryboardFrameData(idx=2, image_prompt="frame 2"),
                StoryboardFrameData(idx=3, image_prompt="frame 3"),
            ],
            segments=[
                StoryboardSegmentData(idx=0, start_frame_idx=0, end_frame_idx=1, duration=3.0, video_prompt="s0"),
                StoryboardSegmentData(idx=1, start_frame_idx=1, end_frame_idx=2, duration=3.0, video_prompt="s1"),
                StoryboardSegmentData(idx=2, start_frame_idx=2, end_frame_idx=3, duration=3.0, video_prompt="s2"),
            ],
        )
        calls = {"segments": []}

        def fake_world(settings, state):
            state.world = world
            state.world_raw = world.model_dump(mode="json")
            return state

        def fake_storyboard(settings, state):
            state.storyboard = storyboard
            state.storyboard_raw = storyboard.model_dump(mode="json", by_alias=True)
            return state

        def fake_keyframes(settings, state):
            state.frame_uris = [
                "file:///frame-0.png",
                "file:///frame-1.png",
                "file:///frame-2.png",
                "file:///frame-3.png",
            ]
            state.storyboard = storyboard
            state.storyboard_raw = storyboard.model_dump(mode="json", by_alias=True)
            state.regen_frames = []
            return state

        def fake_segments(settings, state):
            calls["segments"].append(list(state.regen_segments))
            state.segment_uris = [
                "file:///segment-0.mp4",
                "file:///segment-1.mp4",
                "file:///segment-2.mp4",
            ]
            state.regen_segments = []
            return state

        def fake_video_review(settings, state):
            if state.vid_iteration == 0:
                state.vid_issues = [{"target": "segment:0", "severity": "blocker", "problem": "bad anchor"}]
            else:
                state.vid_issues = []
            state.vid_iteration += 1
            return state

        def fake_video_fix(settings, state):
            state.vid_issues = []
            state.regen_frames = [1]
            return state

        monkeypatch.setattr("scene_agent.prefect_flows.director_world_task", fake_world)
        monkeypatch.setattr("scene_agent.prefect_flows.director_storyboard_task", fake_storyboard)
        monkeypatch.setattr("scene_agent.prefect_flows.keyframes_task", fake_keyframes)
        monkeypatch.setattr("scene_agent.prefect_flows.storyboard_review_task", lambda settings, state: state)
        monkeypatch.setattr("scene_agent.prefect_flows.segments_task", fake_segments)
        monkeypatch.setattr(
            "scene_agent.prefect_flows.stitch_task",
            lambda settings, state: (setattr(state, "final_video_uri", "file:///final.mp4") or state),
        )
        monkeypatch.setattr("scene_agent.prefect_flows.video_review_task", fake_video_review)
        monkeypatch.setattr("scene_agent.prefect_flows.video_fix_task", fake_video_fix)

        result = generate_scene_flow(
            "brief",
            constraints=Constraints(),
            config=Config(openrouter_api_key="test-key", storage_path=str(tmp_path)),
            run_options={"run_id": "regen-frame-flow-test"},
        )

        assert result.status == "completed"
        assert calls["segments"] == [[], [0, 1]]

    def test_generate_scene_flow_force_edit_segments_smoke_path(self, monkeypatch, tmp_path):
        world = _world()
        storyboard = _storyboard()
        calls = {"segments_edit": 0, "stitch": 0}

        def fake_world(settings, state):
            state.world = world
            state.world_raw = world.model_dump(mode="json")
            return state

        def fake_storyboard(settings, state):
            state.storyboard = storyboard
            state.storyboard_raw = storyboard.model_dump(mode="json", by_alias=True)
            return state

        def fake_keyframes(settings, state):
            state.frame_uris = ["file:///frame-0.png", "file:///frame-1.png"]
            return state

        def fake_segments(settings, state):
            state.segment_uris = ["file:///segment-0.mp4"]
            state.regen_segments = []
            return state

        def fake_stitch(settings, state):
            calls["stitch"] += 1
            state.final_video_uri = f"file:///final-{calls['stitch']}.mp4"
            return state

        def fake_segments_edit(settings, state):
            calls["segments_edit"] += 1
            assert state.edit_segments == [0]
            state.segment_uris = ["file:///segment-0-edited.mp4"]
            state.edit_segments = []
            return state

        monkeypatch.setattr("scene_agent.prefect_flows.director_world_task", fake_world)
        monkeypatch.setattr("scene_agent.prefect_flows.director_storyboard_task", fake_storyboard)
        monkeypatch.setattr("scene_agent.prefect_flows.keyframes_task", fake_keyframes)
        monkeypatch.setattr("scene_agent.prefect_flows.storyboard_review_task", lambda settings, state: state)
        monkeypatch.setattr("scene_agent.prefect_flows.segments_task", fake_segments)
        monkeypatch.setattr("scene_agent.prefect_flows.stitch_task", fake_stitch)
        monkeypatch.setattr("scene_agent.prefect_flows.segments_edit_task", fake_segments_edit)
        monkeypatch.setattr(
            "scene_agent.prefect_flows.video_review_task",
            lambda settings, state: (setattr(state, "vid_issues", []) or state),
        )

        result = generate_scene_flow(
            "brief",
            constraints=Constraints(),
            config=Config(openrouter_api_key="test-key", storage_path=str(tmp_path)),
            run_options={"run_id": "forced-edit-flow-test", "force_edit_segments": [0]},
        )

        assert result.status == "completed"
        assert result.segment_uris == ["file:///segment-0-edited.mp4"]
        assert result.final_video_uri == "file:///final-2.mp4"
        assert calls["segments_edit"] == 1
        assert calls["stitch"] == 2

    def test_generate_scene_flow_marks_prefect_run_failed_on_exception(self, monkeypatch, tmp_path):
        def boom(settings, state):
            raise RuntimeError("kling timeout")

        monkeypatch.setattr("scene_agent.prefect_flows.director_world_task", boom)

        with pytest.raises(Exception):
            generate_scene_flow(
                "brief",
                config=Config(openrouter_api_key="test-key", storage_path=str(tmp_path)),
                run_options={"run_id": "failed-flow"},
            )

        manifest = Path(tmp_path) / "runs" / "failed-flow" / "manifest.json"
        text = manifest.read_text(encoding="utf-8")
        assert '"status": "failed"' in text
        assert 'kling timeout' in text
