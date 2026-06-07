"""Unit tests for stabilized review pipeline behavior."""

import pytest

from scene_agent.models import Constraints, SceneState


def _write_tiny_png(path):
    from PIL import Image

    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(path, format="PNG")


class TestStabilizedFixBehavior:
    def test_sb_editor_review_uses_multimodal_history_when_frames_available(self, tmp_path):
        from scene_agent.pipeline.storyboard_editor import EditorTools, sb_editor_review
        from scene_agent.tools.storage import LocalStorageBackend

        called = {}

        class MockLLM:
            default_model = "test-model"

            def chat_with_history(self, messages, **kwargs):
                called["messages"] = messages
                called["kwargs"] = kwargs
                return '[{"target":"frame:0","severity":"warning","problem":"lighting drift"}]'

            def chat(self, **kwargs):
                raise AssertionError("text-only review should not be used")

        storage = LocalStorageBackend(base_path=tmp_path)
        frame_path = tmp_path / "frame.png"
        _write_tiny_png(frame_path)
        frame_uri = f"file://{frame_path}"

        state = SceneState(
            user_brief="Test",
            storyboard_raw={
                "scene_background": "bg",
                "objects": [],
                "style_guide": {"style": "cinematic", "palette": "blue", "global_negative": []},
                "frames": [{"idx": 0, "image_prompt": "a"}],
                "segments": [],
            },
            frame_uris=[frame_uri],
        )

        result = sb_editor_review(state, EditorTools(MockLLM(), storage=storage))
        assert result["sb_review_mode"] == "multimodal"
        assert result["sb_issues"] == []
        assert result["provider_metadata"]["sb_review"]["all_issues"][0]["problem"] == "lighting drift"
        assert result["provider_metadata"]["sb_review"]["blocking_issue_count"] == 0
        assert called["messages"][1]["content"][0]["type"] == "text"
        assert called["messages"][1]["content"][1]["type"] == "image_url"
        assert called["kwargs"]["json_mode"] is True

    def test_sb_editor_review_invalid_json_raises(self, tmp_path):
        from scene_agent.pipeline.storyboard_editor import EditorTools, sb_editor_review
        from scene_agent.tools.storage import LocalStorageBackend

        class MockLLM:
            def chat_with_history(self, messages, **kwargs):
                return "not json"

        storage = LocalStorageBackend(base_path=tmp_path)
        frame_path = tmp_path / "frame.png"
        _write_tiny_png(frame_path)

        state = SceneState(
            user_brief="Test",
            storyboard_raw={
                "scene_background": "bg",
                "objects": [],
                "style_guide": {"style": "cinematic", "palette": "blue", "global_negative": []},
                "frames": [{"idx": 0, "image_prompt": "a"}],
                "segments": [],
            },
            frame_uris=[f"file://{frame_path}"],
        )

        with pytest.raises(ValueError):
            sb_editor_review(state, EditorTools(MockLLM(), storage=storage))

    def test_sb_editor_review_adds_deterministic_continuity_issues(self, tmp_path):
        from scene_agent.pipeline.storyboard_editor import EditorTools, sb_editor_review
        from scene_agent.tools.storage import LocalStorageBackend

        class MockLLM:
            def chat_with_history(self, messages, **kwargs):
                return "[]"

        storage = LocalStorageBackend(base_path=tmp_path)
        frame_0 = tmp_path / "frame-0.png"
        frame_1 = tmp_path / "frame-1.png"
        _write_tiny_png(frame_0)
        _write_tiny_png(frame_1)

        state = SceneState(
            user_brief="Test",
            storyboard_raw={
                "scene_background": "bg",
                "primary_subject_ids": ["hero"],
                "objects": [{"id": "hero", "name": "Hero", "appearance": "same face", "story_role": "character"}],
                "style_guide": {"style": "cinematic", "palette": "blue", "global_negative": []},
                "frames": [
                    {
                        "idx": 0,
                        "image_prompt": "a",
                        "travel_direction": "screen_left_to_right",
                        "gaze_direction": "look_screen_right",
                        "hero_presence": {"hero": "on_screen_primary"},
                        "visible_object_ids": ["hero"],
                    },
                    {
                        "idx": 1,
                        "image_prompt": "b",
                        "travel_direction": "screen_right_to_left",
                        "gaze_direction": "look_screen_left",
                        "hero_presence": {"hero": ""},
                        "visible_object_ids": [],
                    },
                ],
                "segments": [{
                    "idx": 0,
                    "start_frame_idx": 0,
                    "end_frame_idx": 1,
                    "duration": 3.0,
                    "video_prompt": "prompt",
                    "screen_direction_rule": "preserve established screen direction",
                    "gaze_continuity_rule": "maintain readable eyeline continuity",
                    "offscreen_justification": "",
                    "entry_exit_actions": [],
                }],
            },
            frame_uris=[f"file://{frame_0}", f"file://{frame_1}"],
        )

        result = sb_editor_review(state, EditorTools(MockLLM(), storage=storage))
        problems = [issue["problem"] for issue in result["sb_issues"]]
        assert any("Screen direction flips" in problem for problem in problems)
        assert any("unaccounted for" in problem for problem in problems)

    def test_sb_editor_review_does_not_treat_stationary_as_direction_flip(self, tmp_path):
        from scene_agent.pipeline.storyboard_editor import EditorTools, sb_editor_review
        from scene_agent.tools.storage import LocalStorageBackend

        class MockLLM:
            def chat_with_history(self, messages, **kwargs):
                return "[]"

        storage = LocalStorageBackend(base_path=tmp_path)
        frame_0 = tmp_path / "frame-0.png"
        frame_1 = tmp_path / "frame-1.png"
        _write_tiny_png(frame_0)
        _write_tiny_png(frame_1)

        state = SceneState(
            user_brief="Test",
            storyboard_raw={
                "scene_background": "bg",
                "primary_subject_ids": ["hero"],
                "objects": [{"id": "hero", "name": "Hero", "appearance": "same", "story_role": "character"}],
                "style_guide": {"style": "cinematic", "palette": "blue", "global_negative": []},
                "frames": [
                    {
                        "idx": 0,
                        "image_prompt": "a",
                        "travel_direction": "Stationary",
                        "gaze_direction": "look_screen_right",
                        "hero_presence": {"hero": "on_screen_primary"},
                        "visible_object_ids": ["hero"],
                    },
                    {
                        "idx": 1,
                        "image_prompt": "b",
                        "travel_direction": "Running left, away from the ocean",
                        "gaze_direction": "look_screen_right",
                        "hero_presence": {"hero": "on_screen_primary"},
                        "visible_object_ids": ["hero"],
                    },
                ],
                "segments": [{
                    "idx": 0,
                    "start_frame_idx": 0,
                    "end_frame_idx": 1,
                    "duration": 3.0,
                    "video_prompt": "prompt",
                    "screen_direction_rule": "preserve established screen direction",
                    "gaze_continuity_rule": "maintain readable eyeline continuity",
                }],
            },
            frame_uris=[f"file://{frame_0}", f"file://{frame_1}"],
        )

        result = sb_editor_review(state, EditorTools(MockLLM(), storage=storage))
        assert not any("Screen direction flips" in issue["problem"] for issue in result["sb_issues"])

    def test_sb_editor_review_accepts_explicit_reorientation_wording(self, tmp_path):
        from scene_agent.pipeline.storyboard_editor import EditorTools, sb_editor_review
        from scene_agent.tools.storage import LocalStorageBackend

        class MockLLM:
            def chat_with_history(self, messages, **kwargs):
                return "[]"

        storage = LocalStorageBackend(base_path=tmp_path)
        frame_0 = tmp_path / "frame-0.png"
        frame_1 = tmp_path / "frame-1.png"
        _write_tiny_png(frame_0)
        _write_tiny_png(frame_1)

        state = SceneState(
            user_brief="Test",
            storyboard_raw={
                "scene_background": "bg",
                "primary_subject_ids": ["hero"],
                "objects": [{"id": "hero", "name": "Hero", "appearance": "same", "story_role": "character"}],
                "style_guide": {"style": "cinematic", "palette": "blue", "global_negative": []},
                "frames": [
                    {
                        "idx": 0,
                        "image_prompt": "a",
                        "travel_direction": "Right",
                        "gaze_direction": "look_screen_right",
                        "hero_presence": {"hero": "on_screen_primary"},
                        "visible_object_ids": ["hero"],
                    },
                    {
                        "idx": 1,
                        "image_prompt": "b",
                        "travel_direction": "Left",
                        "gaze_direction": "look_screen_left",
                        "hero_presence": {"hero": "on_screen_primary"},
                        "visible_object_ids": ["hero"],
                    },
                ],
                "segments": [{
                    "idx": 0,
                    "start_frame_idx": 0,
                    "end_frame_idx": 1,
                    "duration": 3.0,
                    "video_prompt": "prompt",
                    "screen_direction_rule": "The camera makes a clear reorientation before the subject moves left.",
                    "gaze_continuity_rule": "motivated head turn",
                }],
            },
            frame_uris=[f"file://{frame_0}", f"file://{frame_1}"],
        )

        result = sb_editor_review(state, EditorTools(MockLLM(), storage=storage))
        assert not any("Screen direction flips" in issue["problem"] for issue in result["sb_issues"])

    def test_vid_editor_review_provider_failure_raises(self):
        from scene_agent.pipeline.video_editor import VideoEditorTools, vid_editor_review

        class MockVideoReview:
            def review(self, **kwargs):
                raise RuntimeError("provider down")

        class MockLLM:
            default_model = "test-model"

            def chat(self, **kwargs):
                return '[{"target":"segment:0","severity":"warning","problem":"drift"}]'

        state = SceneState(
            user_brief="Test",
            storyboard_raw={
                "scene_background": "bg",
                "objects": [],
                "style_guide": {"style": "cinematic", "palette": "blue", "global_negative": []},
                "frames": [{"idx": 0, "image_prompt": "a"}, {"idx": 1, "image_prompt": "b"}],
                "segments": [{
                    "idx": 0,
                    "start_frame_idx": 0,
                    "end_frame_idx": 1,
                    "duration": 3.0,
                    "video_prompt": "prompt",
                }],
            },
            final_video_uri="file:///tmp/video.mp4",
        )

        with pytest.raises(RuntimeError, match="provider down"):
            vid_editor_review(state, VideoEditorTools(MockLLM(), MockVideoReview()))

    def test_vid_editor_review_keeps_minor_errors_nonblocking(self, monkeypatch):
        from scene_agent.pipeline import video_editor
        from scene_agent.pipeline.video_editor import VideoEditorTools, vid_editor_review

        class MockVideoReview:
            model = "video-model"

            def review(self, **kwargs):
                return """
                [
                  {"target":"global","severity":"error","problem":"The aspect ratio is not 1:1; it appears 9:16."},
                  {"target":"segment:0","severity":"error","problem":"The cube is slightly off-center."}
                ]
                """

        class MockLLM:
            default_model = "test-model"

        monkeypatch.setattr(
            video_editor,
            "_local_video_metadata",
            lambda uri: {"width": 512, "height": 512, "fps": 12.0, "duration": 3.0, "aspect_ratio": "512:512"},
        )

        state = SceneState(
            user_brief="Test",
            constraints=Constraints(aspect_ratio="1:1"),
            storyboard_raw={
                "scene_background": "bg",
                "objects": [],
                "style_guide": {"style": "cinematic", "palette": "blue", "global_negative": []},
                "frames": [{"idx": 0, "image_prompt": "a"}, {"idx": 1, "image_prompt": "b"}],
                "segments": [{
                    "idx": 0,
                    "start_frame_idx": 0,
                    "end_frame_idx": 1,
                    "duration": 3.0,
                    "video_prompt": "prompt",
                }],
            },
            final_video_uri="file:///tmp/video.mp4",
        )

        result = vid_editor_review(state, VideoEditorTools(MockLLM(), MockVideoReview()))
        assert result["vid_review_mode"] == "multimodal"
        assert result["vid_issues"] == []
        assert result["provider_metadata"]["vid_review"]["blocking_issue_count"] == 0
        assert {issue["severity"] for issue in result["provider_metadata"]["vid_review"]["all_issues"]} == {"warning"}

    def test_sb_editor_fix_applies_segment_issue(self):
        from scene_agent.pipeline.storyboard_editor import EditorTools, sb_editor_fix

        class MockLLM:
            def chat(self, **kwargs):
                return """
                {
                  "storyboard": {
                    "scene_background": "bg",
                    "objects": [],
                    "style_guide": {"style": "cinematic", "palette": "blue", "global_negative": []},
                    "frames": [{"idx": 0, "image_prompt": "a"}, {"idx": 1, "image_prompt": "b"}],
                    "segments": [{
                      "idx": 0,
                      "start_frame_idx": 0,
                      "end_frame_idx": 1,
                      "duration": 3.0,
                      "transition_text": "The camera eases left while the subject settles into the next pose.",
                      "camera_move": "slow pan left only",
                      "subject_motion": "the subject subtly turns into the next pose",
                      "environment_motion": "background remains stable",
                      "motion_beats": ["hold", "ease left", "settle into the next pose"],
                      "continuity_anchors": ["same silhouette", "same background geometry"],
                      "end_match_notes": "Land on the exact pose and framing of the end keyframe.",
                      "video_prompt": "",
                      "negative": ["blur"]
                    }]
                  },
                  "regen_frames": []
                }
                """

        state = SceneState(
            user_brief="Test",
            storyboard_raw={
                "scene_background": "bg",
                "objects": [],
                "style_guide": {"style": "cinematic", "palette": "blue", "global_negative": []},
                "frames": [{"idx": 0, "image_prompt": "a"}, {"idx": 1, "image_prompt": "b"}],
                "segments": [{
                    "idx": 0,
                    "start_frame_idx": 0,
                    "end_frame_idx": 1,
                    "duration": 3.0,
                    "transition_text": "old",
                    "video_prompt": "old",
                    "negative": []
                }],
            },
            sb_issues=[{"target": "segment:0", "problem": "Too vague"}],
        )

        result = sb_editor_fix(state, EditorTools(MockLLM()))
        assert result["storyboard_raw"]["segments"][0]["camera_move"] == "slow pan left only"
        assert "Camera move: slow pan left only" in result["storyboard_raw"]["segments"][0]["video_prompt"]
        assert result["regen_frames"] == []
