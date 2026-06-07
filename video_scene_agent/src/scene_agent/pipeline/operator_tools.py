
from __future__ import annotations

from scene_agent.tools.stitch import StitchTool


class OperatorTools:
    def __init__(self, video_tool, stitch_tool: StitchTool):
        self.video_tool = video_tool
        self.stitch_tool = stitch_tool
