"""Unit tests for Prefect routing and stabilized pipeline behavior."""

import pytest

from scene_agent.flows.routing import (
    route_after_sb_fix,
    route_after_sb_review,
    route_after_vid_fix,
    route_after_vid_review,
)
from scene_agent.models import Constraints, SceneState


def _write_tiny_png(path):
    from PIL import Image

    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(path, format="PNG")


class TestRouteAfterSbReview:
    def test_no_issues_routes_to_segments(self):
        state = SceneState(user_brief="Test", sb_issues=[])
        assert route_after_sb_review(state) == "segments_generate"

    def test_with_issues_routes_to_fix(self):
        state = SceneState(user_brief="Test", sb_issues=[{"target": "frame:0", "problem": "Test"}])
        assert route_after_sb_review(state) == "sb_editor_fix"


class TestRouteAfterVidReview:
    def test_no_issues_routes_to_end(self):
        state = SceneState(user_brief="Test", vid_issues=[], final_video_uri="test.mp4")
        assert route_after_vid_review(state) == "__end__"

    def test_with_issues_routes_to_fix(self):
        state = SceneState(
            user_brief="Test",
            vid_issues=[{"target": "segment:0", "problem": "Test"}],
            final_video_uri="test.mp4",
        )
        assert route_after_vid_review(state) == "vid_editor_fix"


class TestRouteAfterSbFix:
    def test_regen_frames_routes_to_keyframes(self):
        state = SceneState(user_brief="Test", regen_frames=[0, 2])
        assert route_after_sb_fix(state) == "keyframes_generate"

    def test_no_regen_frames_routes_to_review(self):
        state = SceneState(user_brief="Test", regen_frames=[])
        assert route_after_sb_fix(state) == "sb_editor_review"


class TestRouteAfterVidFix:
    def test_regen_frames_routes_to_keyframes(self):
        state = SceneState(user_brief="Test", regen_frames=[1], regen_segments=[])
        assert route_after_vid_fix(state) == "keyframes_generate"

    def test_regen_segments_routes_to_segments(self):
        state = SceneState(user_brief="Test", regen_frames=[], regen_segments=[0, 1])
        assert route_after_vid_fix(state) == "segments_generate"

    def test_edit_segments_routes_to_segments_edit(self):
        state = SceneState(user_brief="Test", regen_frames=[], edit_segments=[0], regen_segments=[])
        assert route_after_vid_fix(state) == "segments_edit"

    def test_no_regen_routes_to_end(self):
        state = SceneState(user_brief="Test", regen_frames=[], regen_segments=[])
        assert route_after_vid_fix(state) == "__end__"

    def test_both_regen_prioritizes_frames(self):
        state = SceneState(user_brief="Test", regen_frames=[0], edit_segments=[1], regen_segments=[2])
        assert route_after_vid_fix(state) == "keyframes_generate"


class TestIterationLimitBehavior:
    def test_sb_iteration_limit_raises(self):
        from scene_agent.pipeline.storyboard_editor import EditorTools, sb_editor_review

        class MockLLM:
            def chat(self, **kwargs):
                raise AssertionError("LLM should not be called at iteration limit")

        state = SceneState(
            user_brief="Test",
            constraints=Constraints(K_sb=2),
            storyboard_raw={"frames": [{"idx": 0, "image_prompt": "x"}], "segments": []},
            sb_iteration=2,
        )
        with pytest.raises(RuntimeError, match="Storyboard review exceeded max iterations"):
            sb_editor_review(state, EditorTools(MockLLM()))

    def test_vid_iteration_limit_raises(self):
        from scene_agent.pipeline.video_editor import VideoEditorTools, vid_editor_review

        class MockLLM:
            def chat(self, **kwargs):
                raise AssertionError("LLM should not be called at iteration limit")

        class MockVideoReview:
            def review(self, **kwargs):
                raise AssertionError("Video review should not be called at iteration limit")

        state = SceneState(
            user_brief="Test",
            constraints=Constraints(K_vid=1),
            storyboard_raw={"frames": [{"idx": 0, "image_prompt": "x"}], "segments": []},
            final_video_uri="test.mp4",
            vid_iteration=1,
        )
        with pytest.raises(RuntimeError, match="Video review exceeded max iterations"):
            vid_editor_review(state, VideoEditorTools(MockLLM(), MockVideoReview()))

