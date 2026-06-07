"""Unit tests for prompts module."""

import json

from scene_agent.prompts import (
    JSON_RULE,
    DIRECTOR_WORLD_SYSTEM,
    DIRECTOR_STORYBOARD_SYSTEM,
    SB_REVIEW_SYSTEM,
    SB_FIX_SYSTEM,
    VID_REVIEW_SYSTEM,
    VID_FIX_SYSTEM,
    DIRECTOR_KEYFRAMES_BATCH_SYSTEM,
    format_director_world_user,
    format_director_storyboard_user,
    format_sb_review_user,
    format_sb_fix_user,
    format_vid_review_user,
    format_vid_fix_user,
    derive_frame_image_prompt,
    derive_segment_video_prompt,
    hydrate_storyboard_payload_prompts,
    normalize_world_description,
    to_json,
)
from scene_agent.models import StoryBeatData, StoryStyleGuide, StoryboardData, StoryboardFrameData, StoryboardSegmentData, WorldDescription, WorldObject


class TestJsonRule:
    """Tests for JSON_RULE constant."""

    def test_json_rule_exists(self):
        """Test that JSON_RULE is defined."""
        assert JSON_RULE
        assert "JSON" in JSON_RULE
        assert "valid JSON" in JSON_RULE


class TestDirectorWorldPrompts:
    """Tests for director world prompts."""

    def test_director_world_system_exists(self):
        """Test that director world system prompt is defined."""
        assert DIRECTOR_WORLD_SYSTEM
        assert "film director" in DIRECTOR_WORLD_SYSTEM.lower()
        assert "continuity_bible" in DIRECTOR_WORLD_SYSTEM
        assert "default_visibility" in DIRECTOR_WORLD_SYSTEM

    def test_format_director_world_user(self):
        """Test formatting director world user prompt."""
        result = format_director_world_user(
            user_brief="A sunset over the ocean",
            constraints_json='{"aspect_ratio": "16:9", "duration": 5.0}',
        )

        assert "USER_BRIEF:" in result
        assert "A sunset over the ocean" in result
        assert "16:9" in result
        assert "5.0" in result
        assert "Return all content in English." in result


class TestDirectorStoryboardPrompts:
    """Tests for director storyboard prompts."""

    def test_director_storyboard_system_exists(self):
        """Test that director storyboard system prompt is defined."""
        assert DIRECTOR_STORYBOARD_SYSTEM
        assert "film director" in DIRECTOR_STORYBOARD_SYSTEM.lower()
        assert "shot_size" in DIRECTOR_STORYBOARD_SYSTEM
        assert "camera_move" in DIRECTOR_STORYBOARD_SYSTEM
        assert "story_beats" in DIRECTOR_STORYBOARD_SYSTEM
        assert "visible_object_ids" in DIRECTOR_STORYBOARD_SYSTEM
        assert "gaze_direction" in DIRECTOR_STORYBOARD_SYSTEM
        assert "hero_presence" in DIRECTOR_STORYBOARD_SYSTEM

    def test_format_director_storyboard_user(self):
        """Test formatting director storyboard user prompt."""
        result = format_director_storyboard_user(
            user_brief="Night city scene",
            constraints_json='{"duration": 10}',
            world_json='{"scene_background": "Dark city"}',
        )

        assert "USER_BRIEF:" in result
        assert "Night city scene" in result
        assert "Dark city" in result
        assert "Return all content in English." in result


class TestStoryboardEditorPrompts:
    """Tests for storyboard editor prompts."""

    def test_sb_review_system_exists(self):
        """Test that sb review system prompt is defined."""
        assert SB_REVIEW_SYSTEM
        assert "storyboard editor" in SB_REVIEW_SYSTEM.lower()
        assert "cinematic grammar" in SB_REVIEW_SYSTEM.lower()
        assert "beat:<idx>" in SB_REVIEW_SYSTEM

    def test_sb_fix_system_exists(self):
        """Test that sb fix system prompt is defined."""
        assert SB_FIX_SYSTEM

    def test_format_sb_review_user(self):
        """Test formatting sb review user prompt."""
        storyboard_json = '{"frames": [{"idx": 0}], "segments": []}'
        result = format_sb_review_user(storyboard_json)

        assert "STORYBOARD_JSON" in result
        assert storyboard_json in result

    def test_format_sb_fix_user(self):
        """Test formatting sb fix user prompt."""
        storyboard_json = '{"frames": []}'
        issues_json = '[{"target": "frame:0", "problem": "test"}]'
        result = format_sb_fix_user(storyboard_json, issues_json)

        assert "CURRENT_STORYBOARD_JSON" in result
        assert storyboard_json in result
        assert issues_json in result


class TestVideoEditorPrompts:
    """Tests for video editor prompts."""

    def test_vid_review_system_exists(self):
        """Test that vid review system prompt is defined."""
        assert VID_REVIEW_SYSTEM
        assert "video editor" in VID_REVIEW_SYSTEM.lower()
        assert "end frame" in VID_REVIEW_SYSTEM.lower()
        assert "beat:<idx>" in VID_REVIEW_SYSTEM

    def test_vid_fix_system_exists(self):
        """Test that vid fix system prompt is defined."""
        assert VID_FIX_SYSTEM

    def test_format_vid_review_user(self):
        """Test formatting vid review user prompt."""
        video_uri = "https://example.com/video.mp4"
        storyboard_json = '{"frames": [], "segments": []}'
        result = format_vid_review_user(video_uri, storyboard_json)

        assert "VIDEO_URI:" in result
        assert video_uri in result
        assert storyboard_json in result

    def test_format_vid_fix_user(self):
        """Test formatting vid fix user prompt."""
        storyboard_json = '{"frames": []}'
        issues_json = '[{"target": "segment:0", "problem": "test"}]'
        result = format_vid_fix_user(storyboard_json, issues_json)

        assert "CURRENT_STORYBOARD_JSON" in result
        assert storyboard_json in result
        assert issues_json in result


class TestDirectorKeyframesBatchPrompts:
    """Tests for director keyframes batch prompts."""

    def test_director_keyframes_batch_system_exists(self):
        """Test that director keyframes batch system prompt is defined."""
        from scene_agent.prompts import DIRECTOR_KEYFRAMES_BATCH_SYSTEM
        assert DIRECTOR_KEYFRAMES_BATCH_SYSTEM
        assert "batch" in DIRECTOR_KEYFRAMES_BATCH_SYSTEM.lower()

    def test_format_director_keyframes_batch_user(self):
        """Test formatting director keyframes batch user prompt."""
        storyboard_json = '{"frames": [{"idx": 0}], "segments": []}'
        from scene_agent.prompts import format_director_keyframes_batch_user
        result = format_director_keyframes_batch_user(storyboard_json)

        assert "STORYBOARD_JSON" in result
        assert storyboard_json in result


class TestOperatorPrompts:
    """Tests for operator prompts."""

    def test_operator_generate_segments_jobs_system_exists(self):
        """Test that operator generate segments jobs system prompt is defined."""
        from scene_agent.prompts import OPERATOR_GENERATE_SEGMENTS_JOBS_SYSTEM
        assert OPERATOR_GENERATE_SEGMENTS_JOBS_SYSTEM
        assert "image-to-video" in OPERATOR_GENERATE_SEGMENTS_JOBS_SYSTEM

    def test_format_operator_generate_segments_jobs_user(self):
        """Test formatting operator generate segments jobs user prompt."""
        storyboard_json = '{"segments": [{"idx": 0}]}'
        from scene_agent.prompts import format_operator_generate_segments_jobs_user
        result = format_operator_generate_segments_jobs_user(storyboard_json)

        assert "STORYBOARD_JSON" in result
        assert storyboard_json in result

    def test_operator_stitch_plan_system_exists(self):
        """Test that operator stitch plan system prompt is defined."""
        from scene_agent.prompts import OPERATOR_STITCH_PLAN_SYSTEM
        assert OPERATOR_STITCH_PLAN_SYSTEM

    def test_format_operator_stitch_plan_user(self):
        """Test formatting operator stitch plan user prompt."""
        storyboard_json = '{"segments": []}'
        from scene_agent.prompts import format_operator_stitch_plan_user
        result = format_operator_stitch_plan_user(storyboard_json)

        assert "STORYBOARD_JSON" in result
        assert storyboard_json in result


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_to_json(self):
        """Test to_json helper function."""
        obj = {"key": "value", "number": 42}
        result = to_json(obj)

        assert isinstance(result, str)
        assert json.loads(result) == obj

    def test_to_json_indent(self):
        """Test to_json with indentation."""
        obj = {"nested": {"key": "value"}}
        result = to_json(obj, indent=True)

        assert "  " in result  # has indentation

    def test_derive_frame_image_prompt(self):
        storyboard = StoryboardData(
            scene_background="Rainy neon alley at night",
            objects=[
                WorldObject(id="hero", name="Young Woman", appearance="dark hair, pale dress", constraints=["same face"]),
                WorldObject(id="orca_tank", name="Orca Tank", appearance="massive biomechanical orca on a tank", constraints=["do not reveal early"]),
            ],
            style_guide=StoryStyleGuide(
                style="grounded neo-noir realism",
                palette="teal and amber",
                cinematic_intent="tense urban reveal",
                camera_language="restrained handheld coverage",
                lighting_logic="motivated sodium rim light with cyan practical spill",
                continuity_bible=["same wet leather coat", "same neon sign geometry"],
                global_negative=["text"],
            ),
            frames=[
                StoryboardFrameData(
                    idx=0,
                    action_in_frame="The woman pauses and looks over her shoulder.",
                    shot_size="medium close-up",
                    camera_angle="eye level",
                    lens="50mm spherical lens",
                    camera_support="handheld shoulder rig",
                    composition="subject on the left third, neon sign deep in background",
                    blocking="torso turned away, head snapped back toward camera",
                    screen_position="screen_left",
                    body_facing="face_screen_right",
                    travel_direction="screen_left_to_right",
                    gaze_direction="look_screen_right",
                    gaze_target="look_at_environment:ocean_surface",
                    camera_axis_side="camera_right_of_axis",
                    emotional_beat="sudden suspicion",
                    continuity_anchors=["same hairstyle silhouette"],
                    hero_presence={"hero": "on_screen_primary"},
                    hero_scale={"hero": "mid_scale"},
                    must_have=["wet leather coat", "silver pendant"],
                ),
                StoryboardFrameData(idx=1),
            ],
            segments=[
                StoryboardSegmentData(
                    idx=0,
                    start_frame_idx=0,
                    end_frame_idx=1,
                    duration=3.0,
                )
            ],
        )
        prompt = derive_frame_image_prompt(storyboard, storyboard.frames[0])
        assert "Subject identity:" in prompt
        assert "Shot design: medium close-up, eye level, 50mm spherical lens, handheld shoulder rig" in prompt
        assert "Continuity locks: same wet leather coat; same neon sign geometry; same hairstyle silhouette" in prompt
        assert "Young Woman:" in prompt
        assert "Orca Tank:" not in prompt
        assert "Screen geography: position=screen_left; body_facing=face_screen_right; travel=screen_left_to_right; gaze=look_screen_right; gaze_target=look_at_environment:ocean_surface; axis=camera_right_of_axis" in prompt
        assert "Primary subject presence:" in prompt

    def test_derive_segment_video_prompt(self):
        storyboard = StoryboardData(
            scene_background="Rainy neon alley at night",
            objects=[],
            style_guide=StoryStyleGuide(
                style="grounded neo-noir realism",
                palette="teal and amber",
                cinematic_intent="tense urban reveal",
                camera_language="restrained handheld coverage",
                lighting_logic="motivated sodium rim light with cyan practical spill",
                continuity_bible=["same wet leather coat", "same neon sign geometry"],
                global_negative=["text"],
            ),
            frames=[
                StoryboardFrameData(idx=0, action_in_frame="She freezes mid-step.", shot_size="medium shot", camera_angle="eye level", lens="50mm"),
                StoryboardFrameData(idx=1, action_in_frame="She starts to run.", shot_size="medium wide shot", camera_angle="eye level", lens="35mm", composition="open alley escape lane"),
            ],
            segments=[
                StoryboardSegmentData(
                    idx=0,
                    start_frame_idx=0,
                    end_frame_idx=1,
                    duration=3.0,
                    transition_text="She realizes danger and breaks into motion.",
                    camera_move="short handheld pullback only",
                    subject_motion="she pivots and starts running away from the water",
                    environment_motion="rain flickers in the practical lights",
                    motion_beats=["freeze", "pivot", "accelerate into run"],
                    continuity_anchors=["same wet leather coat", "same neon sign geometry"],
                    screen_direction_rule="preserve screen_left_to_right",
                    gaze_continuity_rule="maintain eyeline on look_at_object:ocean_surface",
                    camera_axis_rule="preserve camera_right_of_axis",
                    hero_presence_transition={"hero": "on_screen_primary -> on_screen_primary"},
                    entry_exit_actions=[],
                    offscreen_justification="",
                    end_match_notes="End on the exact running pose and wider framing of the end keyframe.",
                )
            ],
        )
        prompt = derive_segment_video_prompt(storyboard, storyboard.segments[0])
        assert "Generate one continuous image-to-video bridge" in prompt
        assert "Camera move: short handheld pullback only" in prompt
        assert "End-state must match: End on the exact running pose" in prompt
        assert "Forbidden failures:" in prompt
        assert "Screen direction continuity: preserve screen_left_to_right" in prompt
        assert "Gaze continuity: maintain eyeline on look_at_object:ocean_surface" in prompt
        assert "Camera axis continuity: preserve camera_right_of_axis" in prompt

    def test_hydrate_storyboard_payload_prompts_populates_runtime_fields(self):
        storyboard = StoryboardData(
            scene_background="Rainy neon alley at night",
            objects=[
                WorldObject(
                    id="hero",
                    name="Young Woman",
                    appearance="dark hair, pale dress",
                    constraints=["same face"],
                ),
                WorldObject(
                    id="orca_tank",
                    name="Orca Tank",
                    appearance="massive biomechanical orca on a tank",
                    constraints=["do not reveal early"],
                    story_role="vehicle",
                    default_visibility="latent",
                    pre_reveal_hints=["subtle disturbance under water"],
                    hard_exclusions_before_reveal=["do not show the full orca_tank before its reveal beat"],
                ),
            ],
            style_guide=StoryStyleGuide(
                style="grounded neo-noir realism",
                palette="teal and amber",
                lighting_logic="motivated sodium rim light with cyan practical spill",
                continuity_bible=["same shoreline silhouette", "same orca_tank glowing seams after reveal"],
                public_continuity_locks=["same shoreline silhouette"],
                global_negative=["text"],
            ),
            frames=[
                StoryboardFrameData(
                    idx=0,
                    action_in_frame="The woman studies the water and notices a subtle disturbance.",
                    shot_size="close-up",
                    camera_angle="low angle",
                    lens="85mm",
                    hint_object_ids=["orca_tank"],
                ),
                StoryboardFrameData(
                    idx=1,
                    action_in_frame="The orca tank erupts from the water.",
                    shot_size="medium shot",
                    camera_angle="eye level",
                    lens="50mm",
                    frame_class="reveal",
                    visible_object_ids=["orca_tank", "hero"],
                ),
            ],
            segments=[
                StoryboardSegmentData(
                    idx=0,
                    start_frame_idx=0,
                    end_frame_idx=1,
                    duration=3.0,
                    camera_move="slow push-in",
                    subject_motion="the hero lifts their chin",
                )
            ],
        )
        hydrated = hydrate_storyboard_payload_prompts(storyboard)
        assert hydrated.frames[0].image_prompt
        assert hydrated.frames[0].camera == "close-up, low angle, 85mm"
        assert hydrated.segments[0].video_prompt
        assert hydrated.segments[0].transition_text == "slow push-in; the hero lifts their chin"
        assert hydrated.story_beats
        assert hydrated.frames[0].hint_object_ids == ["orca_tank"]
        assert "Forbidden early reveal:" in hydrated.frames[0].image_prompt
        assert "do not show the full orca_tank before its reveal beat" in hydrated.frames[0].image_prompt
        assert hydrated.frames[1].visible_object_ids == ["orca_tank", "hero"]
        assert hydrated.segments[0].visibility_transition["revealed"] == ["orca_tank"]

    def test_normalize_world_description_backfills_visibility_metadata(self):
        world = WorldDescription(
            scene_background="Shoreline at sunset",
            objects=[
                WorldObject(
                    id="orca_tank",
                    name="Orca Tank",
                    appearance="a colossal orca fused to a tank chassis",
                    constraints=["same silhouette"],
                )
            ],
            style_guide=StoryStyleGuide(
                style="grounded realism",
                palette="amber and blue",
                continuity_bible=["same shoreline", "same orca_tank glowing seams after reveal"],
                global_negative=["text"],
            ),
        )
        normalized = normalize_world_description(world)
        assert normalized.objects[0].default_visibility == "latent"
        assert normalized.objects[0].pre_reveal_hints
        assert normalized.style_guide.public_continuity_locks == ["same shoreline"]


class TestSystemPromptsContainJsonRule:
    """Tests that all system prompts contain JSON rule."""

    def test_director_world_has_json_rule(self):
        """Test that director world system prompt mentions JSON."""
        assert "JSON" in DIRECTOR_WORLD_SYSTEM

    def test_director_storyboard_has_json_rule(self):
        """Test that director storyboard system prompt mentions JSON."""
        assert "JSON" in DIRECTOR_STORYBOARD_SYSTEM

    def test_sb_review_has_json_rule(self):
        """Test that sb review system prompt mentions JSON."""
        assert "JSON" in SB_REVIEW_SYSTEM

    def test_sb_fix_has_json_rule(self):
        """Test that sb fix system prompt mentions JSON."""
        assert "JSON" in SB_FIX_SYSTEM

    def test_vid_review_has_json_rule(self):
        """Test that vid review system prompt mentions JSON."""
        assert "JSON" in VID_REVIEW_SYSTEM

    def test_vid_fix_has_json_rule(self):
        """Test that vid fix system prompt mentions JSON."""
        assert "JSON" in VID_FIX_SYSTEM
