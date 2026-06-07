"""Unit tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from scene_agent.models import (
    Constraints,
    FrameSpec,
    Issue,
    ObjectSpec,
    SegmentSpec,
    SceneState,
    Storyboard,
    StyleGuide,
)


class TestConstraints:
    """Tests for Constraints model."""

    def test_default_constraints(self):
        """Test creating constraints with default values."""
        c = Constraints()
        assert c.aspect_ratio == "16:9"
        assert c.duration_sec == 5.0
        assert c.fps == 24
        assert c.style_tags == []
        assert c.K_sb == 3
        assert c.K_vid == 2

    def test_custom_constraints(self):
        """Test creating constraints with custom values."""
        c = Constraints(
            aspect_ratio="9:16",
            duration_sec=10.0,
            fps=30,
            style_tags=["cyberpunk", "noir"],
            K_sb=5,
            K_vid=3,
        )
        assert c.aspect_ratio == "9:16"
        assert c.duration_sec == 10.0
        assert c.fps == 30
        assert c.style_tags == ["cyberpunk", "noir"]
        assert c.K_sb == 5
        assert c.K_vid == 3

    def test_duration_validation(self):
        """Test duration validation."""
        # Valid
        Constraints(duration_sec=1.0)
        Constraints(duration_sec=10.0)
        Constraints(duration_sec=30.0)

        # Invalid
        with pytest.raises(ValidationError):
            Constraints(duration_sec=0.5)
        with pytest.raises(ValidationError):
            Constraints(duration_sec=121.0)

    def test_fps_validation(self):
        """Test FPS validation."""
        # Valid
        Constraints(fps=12)
        Constraints(fps=60)

        # Invalid
        with pytest.raises(ValidationError):
            Constraints(fps=10)
        with pytest.raises(ValidationError):
            Constraints(fps=70)


class TestObjectSpec:
    """Tests for ObjectSpec model."""

    def test_object_spec(self):
        """Test creating object spec."""
        obj = ObjectSpec(
            name="street lamp",
            position="foreground left",
            appearance="Old iron lamp with warm yellow light",
        )
        assert obj.name == "street lamp"
        assert obj.position == "foreground left"
        assert obj.appearance == "Old iron lamp with warm yellow light"
        assert obj.motion == ""  # default


class TestStyleGuide:
    """Tests for StyleGuide model."""

    def test_style_guide(self):
        """Test creating style guide."""
        style = StyleGuide(
            visual_style="Cyberpunk neon",
            color_palette=["#000000", "#ff6b35"],
            lighting="Dramatic side lighting",
            camera_style="Slow dolly in",
        )
        assert style.visual_style == "Cyberpunk neon"
        assert style.color_palette == ["#000000", "#ff6b35"]
        assert style.lighting == "Dramatic side lighting"
        assert style.camera_style == "Slow dolly in"


class TestFrameSpec:
    """Tests for FrameSpec model."""

    def test_frame_spec(self):
        """Test creating frame spec."""
        frame = FrameSpec(
            index=0,
            timestamp=0.0,
            prompt="A sunset over the ocean",
            negative_prompt="text, people",
        )
        assert frame.index == 0
        assert frame.timestamp == 0.0
        assert frame.prompt == "A sunset over the ocean"
        assert frame.negative_prompt == "text, people"
        assert frame.camera_movement == ""  # default

    def test_index_validation(self):
        """Test index validation."""
        FrameSpec(index=0, timestamp=0.0, prompt="test")
        FrameSpec(index=10, timestamp=0.0, prompt="test")

        with pytest.raises(ValidationError):
            FrameSpec(index=-1, timestamp=0.0, prompt="test")


class TestSegmentSpec:
    """Tests for SegmentSpec model."""

    def test_segment_spec(self):
        """Test creating segment spec."""
        segment = SegmentSpec(
            index=0,
            start_frame_idx=0,
            end_frame_idx=1,
            prompt="Camera moves forward",
            duration_sec=5.0,
        )
        assert segment.index == 0
        assert segment.start_frame_idx == 0
        assert segment.end_frame_idx == 1
        assert segment.prompt == "Camera moves forward"
        assert segment.duration_sec == 5.0

    def test_duration_validation(self):
        """Test segment duration validation."""
        # Valid
        SegmentSpec(index=0, start_frame_idx=0, end_frame_idx=1, prompt="test", duration_sec=3.0)
        SegmentSpec(index=0, start_frame_idx=0, end_frame_idx=1, prompt="test", duration_sec=15.0)

        # Invalid
        with pytest.raises(ValidationError):
            SegmentSpec(index=0, start_frame_idx=0, end_frame_idx=1, prompt="test", duration_sec=2.0)
        with pytest.raises(ValidationError):
            SegmentSpec(index=0, start_frame_idx=0, end_frame_idx=1, prompt="test", duration_sec=16.0)


class TestStoryboard:
    """Tests for Storyboard model."""

    def test_empty_storyboard_raises_error(self):
        """Test that empty storyboard raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Storyboard(frames=[], segments=[])

    def test_single_frame_storyboard(self):
        """Test storyboard with single frame (no segments)."""
        sb = Storyboard(
            frames=[FrameSpec(
                index=0,
                timestamp=0.0,
                prompt="Frame 0",
            )],
            segments=[],
        )
        assert len(sb.frames) == 1
        assert len(sb.segments) == 0

    def test_two_frame_storyboard(self):
        """Test storyboard with two frames and one segment."""
        sb = Storyboard(
            frames=[
                FrameSpec(index=0, timestamp=0.0, prompt="Frame 0"),
                FrameSpec(index=1, timestamp=3.0, prompt="Frame 1"),
            ],
            segments=[
                SegmentSpec(
                    index=0,
                    start_frame_idx=0,
                    end_frame_idx=1,
                    prompt="Transition",
                    duration_sec=3.0,
                ),
            ],
        )
        assert len(sb.frames) == 2
        assert len(sb.segments) == 1

    def test_invalid_frame_indices(self):
        """Test that mismatched frame indices raise error."""
        with pytest.raises(ValidationError) as exc_info:
            Storyboard(
                frames=[
                    FrameSpec(index=0, timestamp=0.0, prompt="Frame 0"),
                    FrameSpec(index=2, timestamp=3.0, prompt="Frame 1"),  # wrong index
                ],
                segments=[
                    SegmentSpec(
                        index=0,
                        start_frame_idx=0,
                        end_frame_idx=1,
                        prompt="Transition",
                        duration_sec=3.0,
                    ),
                ],
            )

    def test_invalid_segment_count(self):
        """Test that wrong segment count raises error."""
        with pytest.raises(ValidationError) as exc_info:
            Storyboard(
                frames=[
                    FrameSpec(index=0, timestamp=0.0, prompt="Frame 0"),
                    FrameSpec(index=1, timestamp=3.0, prompt="Frame 1"),
                ],
                segments=[  # should have 1 segment, not 2
                    SegmentSpec(index=0, start_frame_idx=0, end_frame_idx=1, prompt="A", duration_sec=3.0),
                    SegmentSpec(index=1, start_frame_idx=1, end_frame_idx=2, prompt="B", duration_sec=3.0),
                ],
            )

    def test_total_duration(self):
        """Test total duration calculation."""
        sb = Storyboard(
            frames=[
                FrameSpec(index=0, timestamp=0.0, prompt="F0"),
                FrameSpec(index=1, timestamp=3.0, prompt="F1"),
                FrameSpec(index=2, timestamp=6.0, prompt="F2"),
            ],
            segments=[
                SegmentSpec(index=0, start_frame_idx=0, end_frame_idx=1, prompt="T1", duration_sec=3.0),
                SegmentSpec(index=1, start_frame_idx=1, end_frame_idx=2, prompt="T2", duration_sec=3.0),
            ],
        )
        assert sb.total_duration() == 6.0


class TestIssue:
    """Tests for Issue model."""

    def test_valid_global_target(self):
        """Test valid global target."""
        i = Issue(target="global", severity="info", description="Test")
        assert i.target == "global"
        assert i.problem == "Test"
        assert i.description == "Test"

    def test_valid_frame_target(self):
        """Test valid frame targets."""
        Issue(target="beat:0", severity="info", description="Test")
        Issue(target="frame:0", severity="warning", description="Test")
        Issue(target="frame:10", severity="error", description="Test")
        Issue(target="segment:5", severity="info", description="Test")

    def test_invalid_target_format(self):
        """Test invalid target format."""
        with pytest.raises(ValidationError):
            Issue(target="invalid", severity="info", description="Test")

        with pytest.raises(ValidationError):
            Issue(target="frame:-1", severity="info", description="Test")

        with pytest.raises(ValidationError):
            Issue(target="frame:abc", severity="info", description="Test")

        with pytest.raises(ValidationError):
            Issue(target="movie:0", severity="info", description="Test")


class TestSceneState:
    """Tests for SceneState model."""

    def test_default_state(self):
        """Test creating state with defaults."""
        state = SceneState(user_brief="Test brief")
        assert state.user_brief == "Test brief"
        assert isinstance(state.constraints, Constraints)
        assert state.world_raw is None
        assert state.storyboard_raw is None
        assert state.frame_uris == []
        assert state.segment_uris == []
        assert state.sb_iteration == 0
        assert state.vid_iteration == 0
        assert state.sb_issues == []
        assert state.vid_issues == []
        assert state.regen_frames == []
        assert state.regen_segments == []
        assert state.edit_segments == []
        assert state.status == "pending"

    def test_full_state(self):
        """Test state with all fields."""
        state = SceneState(
            user_brief="Test",
            constraints=Constraints(aspect_ratio="9:16"),
            world_raw={"scene_background": "Test"},
            storyboard_raw={"frames": [], "segments": []},
            frame_uris=["uri1", "uri2"],
            segment_uris=["seg1"],
            sb_iteration=2,
            vid_iteration=1,
            sb_issues=[{"target": "frame:0", "problem": "Test"}],
            vid_issues=[],
            regen_frames=[0],
            regen_segments=[],
            status="running",
        )
        assert state.status == "running"
        assert len(state.frame_uris) == 2
        assert state.regen_frames == [0]
        assert state.events == []
