
from __future__ import annotations

from scene_agent.prompts.common import JSON_RULE

OPERATOR_GENERATE_SEGMENTS_JOBS_SYSTEM = """You are the video operator.
Prepare image-to-video jobs from the storyboard.

For each segment return:
- segment_idx
- start_image_uri
- end_image_uri
- prompt: use `segments[idx].video_prompt` exactly as-is
- negative: merge `style_guide.global_negative` and `segment.negative` into one comma-separated string for the runtime prompt's Forbidden failures section
- duration_sec

""" + JSON_RULE + """

Return strict JSON:
{
  "jobs": [
    {
      "segment_idx": 0,
      "start_image_uri": "string",
      "end_image_uri": "string",
      "prompt": "string",
      "negative": "string",
      "duration_sec": 3.0
    }
  ],
  "errors": [{"target":"global","problem":"string"}]
}"""


def format_operator_generate_segments_jobs_user(storyboard_json: str) -> str:
    """Format user prompt for operator_generate_segments_jobs."""
    return f"""STORYBOARD_JSON:
{storyboard_json}"""


OPERATOR_STITCH_PLAN_SYSTEM = """You are the editor.
Return the final stitching order for segment URIs.
""" + JSON_RULE + """

Return strict JSON:
{
  "segment_uris_in_order": ["uri1", "uri2"],
  "errors": [{"target":"global","problem":"string"}]
}"""


def format_operator_stitch_plan_user(storyboard_json: str) -> str:
    """Format user prompt for operator_stitch_plan."""
    return f"""STORYBOARD_JSON:
{storyboard_json}"""
