
from __future__ import annotations


class DirectorTools:
    def __init__(self, llm, image_tool, vision_rewriter=None, storage=None):
        self.llm = llm
        self.image_tool = image_tool
        self.vision_rewriter = vision_rewriter
        self.storage = storage
