
from __future__ import annotations

import json
from typing import Any

JSON_RULE = """Global JSON rule:
- Return valid JSON only.
- No markdown, no commentary, no code fences.
- If uncertain, still return the best JSON you can that matches the schema exactly."""


def to_json(obj: Any, *, indent: bool | None = None) -> str:
    if indent is None:
        return json.dumps(obj, ensure_ascii=False)
    return json.dumps(obj, ensure_ascii=False, indent=indent)


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)
