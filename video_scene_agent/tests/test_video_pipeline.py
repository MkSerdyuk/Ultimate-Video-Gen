from scene_agent.models import Constraints, SceneState


class TestVideoPipeline:
    def test_segments_generate_respects_regen_segments(self):
        from scene_agent.pipeline.video_segments import OperatorTools, segments_generate

        class MockVideo:
            def generate_multiple_segments(self, specs):
                assert len(specs) == 1
                assert specs[0]["num_frames"] == 37
                assert specs[0]["fps"] == 12
                assert specs[0]["aspect_ratio"] == "16:9"
                assert "text" in specs[0]["negative_prompt"]
                return ["new-segment-1"]

        state = SceneState(
            user_brief="Test",
            frame_uris=["frame-0", "frame-1", "frame-2"],
            segment_uris=["segment-0", "segment-1"],
            regen_segments=[1],
            constraints=Constraints(fps=12),
            storyboard_raw={
                "style_guide": {"global_negative": ["text"]},
                "frames": [
                    {"idx": 0, "keyframe_uri": "frame-0"},
                    {"idx": 1, "keyframe_uri": "frame-1"},
                    {"idx": 2, "keyframe_uri": "frame-2"},
                ],
                "segments": [
                    {
                        "idx": 0,
                        "start_frame_idx": 0,
                        "end_frame_idx": 1,
                        "duration": 3.0,
                        "video_prompt": "keep old",
                        "negative": [],
                        "result_uri": "segment-0",
                    },
                    {
                        "idx": 1,
                        "start_frame_idx": 1,
                        "end_frame_idx": 2,
                        "duration": 3.0,
                        "video_prompt": "replace me",
                        "negative": [],
                        "result_uri": "segment-1",
                    },
                ],
            },
        )

        result = segments_generate(state, OperatorTools(MockVideo(), None))
        assert result["segment_uris"] == ["segment-0", "new-segment-1"]
        assert result["storyboard_raw"]["segments"][0]["result_uri"] == "segment-0"
        assert result["storyboard_raw"]["segments"][1]["result_uri"] == "new-segment-1"

    def test_segments_generate_stops_at_kling_budget_and_fills_placeholder(self):
        from types import SimpleNamespace

        from scene_agent.pipeline.video_segments import OperatorTools, segments_generate

        class MockVideo:
            model_name = "kling-v3-omni"
            mode = "std"
            sound = "off"
            config = SimpleNamespace(
                kling_run_token_limit=2.5,
                kling_generation_tokens_per_second=0.6,
                kling_edit_tokens_per_second=0.9,
            )

            def __init__(self):
                self.calls = 0
                self.last_generation_task_ids = []

            def estimate_generation_tokens(self, spec):
                return spec["duration_sec"] * self.config.kling_generation_tokens_per_second

            def estimate_edit_tokens(self, spec):
                return spec["duration_sec"] * self.config.kling_edit_tokens_per_second

            def generate_multiple_segments(self, specs):
                assert len(specs) == 1
                self.calls += 1
                self.last_generation_task_ids = [f"task-{self.calls}"]
                return [f"paid-segment-{self.calls}"]

        class MockStitch:
            def create_still_clip(self, *, image_uri, duration_sec, fps, output_key):
                assert image_uri == "frame-1"
                assert duration_sec == 4.0
                assert fps == 24
                return f"placeholder:{output_key}"

        video = MockVideo()
        state = SceneState(
            user_brief="Test",
            run_id="run-1",
            frame_uris=["frame-0", "frame-1", "frame-2"],
            segment_uris=[],
            constraints=Constraints(fps=24),
            storyboard_raw={
                "style_guide": {"global_negative": []},
                "frames": [{"idx": 0}, {"idx": 1}, {"idx": 2}],
                "segments": [
                    {
                        "idx": 0,
                        "start_frame_idx": 0,
                        "end_frame_idx": 1,
                        "duration": 4.0,
                        "video_prompt": "first",
                    },
                    {
                        "idx": 1,
                        "start_frame_idx": 1,
                        "end_frame_idx": 2,
                        "duration": 4.0,
                        "video_prompt": "second",
                    },
                ],
            },
        )

        result = segments_generate(state, OperatorTools(video, MockStitch()))

        assert result["segment_uris"] == [
            "paid-segment-1",
            "placeholder:segments/kling-budget-placeholder-run-1-1.mp4",
        ]
        assert video.calls == 1
        assert result["provider_metadata"]["segments_generate"]["paid_indices"] == [0]
        assert result["provider_metadata"]["segments_generate"]["budget_skipped_indices"] == [1]
        assert result["provider_metadata"]["segments_generate"]["placeholder_indices"] == [1]
        assert result["provider_metadata"]["kling_budget"]["spent_tokens"] == 2.4
        assert result["provider_metadata"]["kling_budget"]["skipped_calls"][0]["substitute"] == "still_placeholder"

    def test_segments_edit_repairs_existing_segment(self):
        from scene_agent.pipeline.video_segments import OperatorTools, segments_edit

        captured = {}

        class MockVideo:
            model_name = "kling-v3-omni"
            mode = "std"
            sound = "off"

            def edit_multiple_segments(self, specs):
                captured["specs"] = specs
                return ["edited-segment-0"]

        state = SceneState(
            user_brief="Test",
            frame_uris=["frame-0", "frame-1"],
            segment_uris=["segment-0"],
            edit_segments=[0],
            constraints=Constraints(fps=12),
            provider_metadata={"vid_fix": {"edit_problem_map": {"0": ["motion drifts"]}}},
            storyboard_raw={
                "style_guide": {"global_negative": ["text"]},
                "frames": [{"idx": 0}, {"idx": 1}],
                "segments": [
                    {
                        "idx": 0,
                        "start_frame_idx": 0,
                        "end_frame_idx": 1,
                        "duration": 3.0,
                        "video_prompt": "repair me",
                        "negative": ["jitter"],
                        "result_uri": "segment-0",
                        "end_match_notes": "land on frame 1",
                    }
                ],
            },
        )

        result = segments_edit(state, OperatorTools(MockVideo(), None))
        assert result["segment_uris"] == ["edited-segment-0"]
        assert result["edit_segments"] == []
        assert result["regen_segments"] == []
        assert captured["specs"][0]["segment_uri"] == "segment-0"
        assert captured["specs"][0]["num_frames"] == 37
        assert captured["specs"][0]["aspect_ratio"] == "16:9"
        assert captured["specs"][0]["match_video_length"] is True
        assert captured["specs"][0]["match_input_fps"] is True
        assert "motion drifts" in captured["specs"][0]["prompt"]
        assert "current video as motion and camera reference" in captured["specs"][0]["prompt"]
        assert captured["specs"][0]["negative_prompt"] == "text, jitter"

    def test_segments_edit_routes_to_regen_when_source_missing(self):
        from scene_agent.pipeline.video_segments import OperatorTools, segments_edit

        class MockVideo:
            def edit_multiple_segments(self, specs):
                raise AssertionError("edit path should not be called without a source segment")

        state = SceneState(
            user_brief="Test",
            frame_uris=["frame-0", "frame-1"],
            segment_uris=[],
            edit_segments=[0],
            storyboard_raw={
                "style_guide": {"global_negative": []},
                "frames": [{"idx": 0}, {"idx": 1}],
                "segments": [
                    {
                        "idx": 0,
                        "start_frame_idx": 0,
                        "end_frame_idx": 1,
                        "duration": 3.0,
                        "video_prompt": "repair me",
                    }
                ],
            },
        )

        result = segments_edit(state, OperatorTools(MockVideo(), None))
        assert result["edit_segments"] == []
        assert result["regen_segments"] == [0]

    def test_segments_edit_stops_at_kling_budget_and_keeps_existing_segment(self):
        from types import SimpleNamespace

        from scene_agent.pipeline.video_segments import OperatorTools, segments_edit

        class MockVideo:
            model_name = "kling-v3-omni"
            mode = "std"
            sound = "off"
            config = SimpleNamespace(
                kling_run_token_limit=3.0,
                kling_generation_tokens_per_second=0.6,
                kling_edit_tokens_per_second=0.9,
            )

            def estimate_generation_tokens(self, spec):
                return spec["duration_sec"] * self.config.kling_generation_tokens_per_second

            def estimate_edit_tokens(self, spec):
                return spec["duration_sec"] * self.config.kling_edit_tokens_per_second

            def edit_multiple_segments(self, specs):
                raise AssertionError("edit path should not be called after budget is exhausted")

        state = SceneState(
            user_brief="Test",
            frame_uris=["frame-0", "frame-1"],
            segment_uris=["segment-0"],
            edit_segments=[0],
            provider_metadata={"kling_budget": {"spent_tokens": 2.4}},
            storyboard_raw={
                "style_guide": {"global_negative": []},
                "frames": [{"idx": 0}, {"idx": 1}],
                "segments": [
                    {
                        "idx": 0,
                        "start_frame_idx": 0,
                        "end_frame_idx": 1,
                        "duration": 4.0,
                        "video_prompt": "repair me",
                    }
                ],
            },
        )

        result = segments_edit(state, OperatorTools(MockVideo(), None))

        assert result["segment_uris"] == ["segment-0"]
        assert result["edit_segments"] == []
        assert result["provider_metadata"]["segments_edit"]["edited_indices"] == []
        assert result["provider_metadata"]["segments_edit"]["budget_skipped_indices"] == [0]
        assert result["provider_metadata"]["kling_budget"]["spent_tokens"] == 2.4
        assert result["provider_metadata"]["kling_budget"]["skipped_calls"][0]["substitute"] == "existing_segment"

    def test_vid_editor_fix_supports_regen_all(self):
        from scene_agent.pipeline.video_editor import VideoEditorTools, vid_editor_fix

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
                      "transition_text": "pan",
                      "video_prompt": "prompt",
                      "negative": []
                    }]
                  },
                  "regen_all": true
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
                    "transition_text": "pan",
                    "video_prompt": "prompt",
                    "negative": []
                }],
            },
            vid_issues=[{"target": "global", "problem": "Start over"}],
        )

        result = vid_editor_fix(state, VideoEditorTools(MockLLM(), None))
        assert result["regen_frames"] == [0, 1]
        assert result["regen_segments"] == [0]
        assert result["edit_segments"] == []

    def test_vid_editor_fix_supports_edit_segments(self):
        from scene_agent.pipeline.video_editor import VideoEditorTools, vid_editor_fix

        class MockLLM:
            default_model = "test-model"

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
                      "transition_text": "pan",
                      "video_prompt": "prompt",
                      "negative": []
                    }]
                  },
                  "edit_segments": [0],
                  "regen_segments": []
                }
                """

        state = SceneState(
            user_brief="Test",
            segment_uris=["segment-0"],
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
                    "transition_text": "pan",
                    "video_prompt": "prompt",
                    "negative": []
                }],
            },
            vid_issues=[{"target": "segment:0", "problem": "motion drift"}],
        )

        result = vid_editor_fix(state, VideoEditorTools(MockLLM(), None))
        assert result["edit_segments"] == [0]
        assert result["regen_segments"] == []
        assert result["provider_metadata"]["vid_fix"]["edit_problem_map"] == {"0": ["motion drift"]}
