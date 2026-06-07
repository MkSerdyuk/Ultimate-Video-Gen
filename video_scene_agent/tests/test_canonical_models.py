from scene_agent.models import (
    ArtifactManifest,
    RunEvent,
    SceneRunResult,
    SceneState,
    StoryBeatData,
    StoryStyleGuide,
    StoryboardData,
    StoryboardFixResult,
    StoryboardFrameData,
    StoryboardSegmentData,
    VideoFixResult,
    WorldDescription,
    WorldObject,
)


class TestCanonicalWorkflowModels:
    def test_world_description(self):
        world = WorldDescription(
            scene_background="Night city",
            primary_subject_ids=["hero"],
            objects=[
                WorldObject(
                    id="hero",
                    name="Hero",
                    appearance="Long coat",
                    constraints=["same face"],
                )
            ],
            style_guide=StoryStyleGuide(
                style="neo noir",
                palette="teal and orange",
                cinematic_intent="slow-burn tension",
                camera_language="restrained handheld coverage",
                lighting_logic="motivated street practicals with cold fill",
                continuity_bible=["same long coat silhouette"],
                global_continuity_locks=["same long coat silhouette"],
                public_continuity_locks=["same long coat silhouette"],
                global_negative=["text"],
            ),
        )
        assert world.scene_background == "Night city"
        assert world.objects[0].id == "hero"
        assert world.style_guide.cinematic_intent == "slow-burn tension"
        assert world.primary_subject_ids == ["hero"]

    def test_storyboard_data_accepts_prompt_schema(self):
        storyboard = StoryboardData(
            scene_background="Night city",
            objects=[],
            style_guide=StoryStyleGuide(
                style="neo noir",
                palette="teal and orange",
                public_continuity_locks=["same neon haze"],
                global_negative=["text"],
            ),
            story_beats=[
                StoryBeatData(
                    idx=0,
                    id="beat-00-establishing",
                    label="Establishing",
                    narrative_function="establishing",
                )
            ],
            frames=[
                StoryboardFrameData(
                    idx=0,
                    image_prompt="frame 0",
                    shot_size="medium shot",
                    beat_id="beat-00-establishing",
                    screen_position="screen_left",
                    body_facing="face_screen_right",
                ),
                StoryboardFrameData(
                    idx=1,
                    image_prompt="frame 1",
                    shot_size="wide shot",
                    beat_id="beat-00-establishing",
                    screen_position="screen_right",
                    body_facing="face_screen_left",
                ),
            ],
            segments=[
                StoryboardSegmentData(
                    idx=0,
                    start_frame_idx=0,
                    end_frame_idx=1,
                    duration=3.0,
                    video_prompt="bridge",
                    camera_move="slow dolly out",
                    beat_id="beat-00-establishing",
                    screen_direction_rule="preserve screen_left_to_right",
                )
            ],
        )
        assert storyboard.total_duration() == 3.0
        assert storyboard.frames[0].idx == 0
        assert storyboard.segments[0].duration == 3.0
        assert storyboard.segments[0].video_prompt == "bridge"
        assert storyboard.frames[0].shot_size == "medium shot"
        assert storyboard.segments[0].camera_move == "slow dolly out"
        assert storyboard.story_beats[0].id == "beat-00-establishing"
        assert storyboard.frames[0].screen_position == "screen_left"
        assert storyboard.segments[0].screen_direction_rule == "preserve screen_left_to_right"

    def test_storyboard_fix_result(self):
        fix = StoryboardFixResult.model_validate(
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
                        "video_prompt": "bridge",
                    }],
                },
                "regen_frames": [1],
            }
        )
        assert fix.regen_frames == [1]
        assert fix.storyboard.frames[1].idx == 1

    def test_storyboard_segment_accepts_legacy_video_prompt_alias(self):
        segment = StoryboardSegmentData.model_validate(
            {
                "idx": 0,
                "start_frame_idx": 0,
                "end_frame_idx": 1,
                "duration": 3.0,
                "kling_prompt": "legacy bridge",
            }
        )
        assert segment.video_prompt == "legacy bridge"
        assert segment.kling_prompt == "legacy bridge"

    def test_video_fix_result_regen_all(self):
        fix = VideoFixResult.model_validate({"regen_all": True})
        assert fix.regen_all is True

    def test_video_fix_result_edit_segments(self):
        fix = VideoFixResult.model_validate({"edit_segments": [0], "regen_segments": [1]})
        assert fix.edit_segments == [0]
        assert fix.regen_segments == [1]

    def test_scene_run_result(self):
        result = SceneRunResult(
            run_id="run-1",
            status="completed",
            final_video_uri="file:///tmp/final.mp4",
            artifacts_dir="/tmp/run-1",
        )
        assert result.run_id == "run-1"
        assert result.status == "completed"

    def test_artifact_manifest(self):
        manifest = ArtifactManifest(run_id="run-1", status="running")
        dumped = manifest.model_dump_json()
        loaded = ArtifactManifest.model_validate_json(dumped)
        assert loaded.run_id == "run-1"

    def test_run_event_round_trip(self):
        event = RunEvent(
            ts="2026-05-24T12:00:00+00:00",
            stage="keyframes_generate",
            action="replaced",
            asset_kind="frame",
            label="Keyframe 1",
            from_value="file:///old.png",
            to_value="file:///new.png",
            indices=[0],
            counts={"frame_count": 4},
            retry=1,
        )
        state = SceneState(user_brief="Test", events=[event])
        dumped = state.model_dump_json()
        loaded = SceneState.model_validate_json(dumped)
        assert loaded.events[0].label == "Keyframe 1"
        assert loaded.events[0].retry == 1
