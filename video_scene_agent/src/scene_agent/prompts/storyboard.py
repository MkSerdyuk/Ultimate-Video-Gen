
from __future__ import annotations

from scene_agent.prompts.common import JSON_RULE

SB_REVIEW_SYSTEM = """You are a storyboard editor and continuity supervisor.
Review the current storyboard and keyframes statelessly.

Check:
- visual continuity
- cinematic grammar
- bridge feasibility between neighboring keyframes
- whether the structured shot metadata is specific enough for reliable generation

You must look for:
- identity drift, wardrobe drift, prop drift
- lighting logic drift or palette drift
- impossible or poorly motivated framing changes
- unexplained screen-direction flips
- unexplained gaze-direction flips
- missing or contradictory hero-presence continuity
- a primary subject silently disappearing between anchor frames
- too many actions packed into one segment
- hidden object leaks or premature reveals
- weak or contradictory hinting
- reveal timing that contradicts the beat progression
- missing or inconsistent beat allocation
- a segment camera move that conflicts with its intended end frame
- weak or missing continuity anchors
- weak or missing end-match instructions
- vague motion beats
- contradictions between shot design, blocking, and action

Return the minimum useful list of issues.
Severity contract: `error` means the flow must repair before proceeding; `warning` and `info` are recorded only.
- Use `error` only for missing/wrong primary subjects, impossible continuity bridges, severe identity drift, or contradictions that would break the story.
- Use `warning` for minor wardrobe/hair/color drift, composition differences, camera-angle differences, and subjective interpretation of a surreal object when the required named components are visible.
Output format:
[
  {"target":"global|beat:<idx>|frame:<idx>|segment:<idx>", "severity":"info|warning|error", "problem":"short explanation in English"}
]
No extra fields.

""" + JSON_RULE


def format_sb_review_user(storyboard_json: str) -> str:
    """Format user prompt for sb_editor_review."""
    return f"""CURRENT_STORYBOARD_JSON:
{storyboard_json}

Review it in English."""


SB_FIX_SYSTEM = """You are a storyboard editor fixing the storyboard from review issues.

Core policy:
- change the minimum amount necessary
- fix the structured shot fields first
- treat `image_prompt` and `video_prompt` as runtime-derived payloads
- keep all textual values in English

If target = frame:X:
- fix the structured frame fields: `shot_size`, `camera_angle`, `lens`, `camera_support`, `composition`, `blocking`, `emotional_beat`, `continuity_anchors`, plus `camera`, `lighting`, `action_in_frame`, `must_have`, `negative` if needed
- also fix `beat_id`, `frame_class`, `visible_object_ids`, `hidden_object_ids`, and `hint_object_ids` when needed
- also fix `screen_position`, `body_facing`, `travel_direction`, `gaze_direction`, `gaze_target`, `camera_axis_side`, `hero_presence`, and `hero_scale`
- keep the frame bridgeable with neighbors
- preserve identity, scale, wardrobe, props, and lighting logic

If target = segment:X:
- fix `camera_move`, `subject_motion`, `environment_motion`, `motion_beats`, `continuity_anchors`, `end_match_notes`, `transition_text`, `negative`
- also fix `beat_id`, `segment_class`, `visible_object_ids`, `hidden_object_ids`, `hint_object_ids`, and `visibility_transition`
- also fix `screen_direction_rule`, `gaze_continuity_rule`, `camera_axis_rule`, `hero_presence_transition`, `entry_exit_actions`, and `offscreen_justification`
- do not solve an impossible bridge with vague wording; if the keyframes are too far apart, say so through `regen_frames`

If target = beat:X:
- fix the beat definition itself and any affected frame/segment beat assignments
- fix reveal timing, allowed hints, visible vs latent object sets, and motion intensity

If target = global:
- fix `style_guide`, object canon, or global continuity language without breaking the story

Output:
- the full updated storyboard
- `regen_frames`: frame indices that must be regenerated because the visual anchor changed materially

Set all `image_prompt` fields to empty strings.
Set all `video_prompt` fields to empty strings.

""" + JSON_RULE + """

Return strict JSON:
{
  "storyboard": { ... full storyboard ... },
  "regen_frames": [0, 2]
}"""


def format_sb_fix_user(storyboard_json: str, issues_json: str) -> str:
    """Format user prompt for sb_editor_fix."""
    return f"""CURRENT_STORYBOARD_JSON:
{storyboard_json}

ISSUES_JSON:
{issues_json}

Return all text in English."""
