from __future__ import annotations
"""JSON parsing utilities for LLM responses with retries."""

import json
import re
import logging
from typing import TypeVar, Type, Callable, Any
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)
log = logging.getLogger(__name__)


def call_llm_json_with_retries(
    llm_call_fn: Callable[[], str],
    out_model: Type[T],
    retries: int = 2,
    strip_markdown: bool = True,
) -> T:
    """
    Call an LLM function and parse JSON response with retries.

    Args:
        llm_call_fn: Function that returns raw LLM response (JSON string)
        out_model: Pydantic model class to parse into
        retries: Number of retries on parse failure
        strip_markdown: Whether to strip markdown code blocks

    Returns:
        Parsed instance of out_model

    Raises:
        ValueError: If parsing fails after all retries
        ValidationError: If data doesn't match model
    """
    last_error = None
    last_raw = None

    for attempt in range(retries + 1):
        try:
            raw = llm_call_fn()
            last_raw = raw

            # Extract JSON from markdown code blocks if present
            json_str = raw
            if strip_markdown:
                json_str = _extract_json_from_markdown(raw)

            # Parse JSON
            data = json.loads(json_str)
            return out_model.model_validate(data)

        except json.JSONDecodeError as e:
            last_error = e
            log.warning(f"JSON decode error (attempt {attempt + 1}/{retries + 1}): {e}")
            if attempt < retries:
                continue

        except ValidationError as e:
            last_error = e
            log.warning(f"Validation error (attempt {attempt + 1}/{retries + 1}): {e}")
            if attempt < retries:
                continue

        except Exception as e:
            last_error = e
            log.warning(f"Unexpected error (attempt {attempt + 1}/{retries + 1}): {e}")
            if attempt < retries:
                continue

    # All retries exhausted
    raise ValueError(
        f"Failed to parse LLM response as {out_model.__name__} after {retries + 1} attempts.\n"
        f"Last error: {last_error}\n"
        f"Last raw output: {last_raw[:1000] if last_raw else 'None'}..."
    ) from last_error


def _extract_json_from_markdown(text: str) -> str:
    """Extract JSON from markdown code blocks."""
    text = text.strip()

    if text.startswith("{") or text.startswith("["):
        return text

    # Try to find ```json...``` blocks
    json_pattern = r"```(?:json)?\s*\n?([\s\S]*?)```"
    matches = re.findall(json_pattern, text, re.IGNORECASE)
    if matches:
        return matches[0].strip()

    # Try to find [...] pattern before object pattern
    array_pattern = r"\[[\s\S]*\]"
    array_matches = re.findall(array_pattern, text)
    if array_matches:
        return max(array_matches, key=len)

    # Try to find {...} pattern
    brace_pattern = r"\{[\s\S]*\}"
    matches = re.findall(brace_pattern, text)
    if matches:
        # Find the largest match (most likely to be the full object)
        return max(matches, key=len)

    # Return as-is if no patterns found
    return text


def clean_json_response(text: str) -> str:
    """
    Clean a JSON response from an LLM.

    Handles:
    - Markdown code blocks
    - Leading/trailing whitespace
    - Common LLM quirks

    Args:
        text: Raw LLM response

    Returns:
        Cleaned JSON string
    """
    return _extract_json_from_markdown(text)


def parse_partial_json(text: str) -> dict | None:
    """
    Attempt to parse JSON, handling common partial/incomplete responses.

    Args:
        text: Text that might contain JSON

    Returns:
        Parsed dict or None if parsing fails
    """
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown
    cleaned = _extract_json_from_markdown(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try fixing common issues with truncated JSON
    try:
        return _fix_and_parse_json(cleaned)
    except json.JSONDecodeError:
        pass

    # Try fixing on original text too
    try:
        return _fix_and_parse_json(text)
    except json.JSONDecodeError:
        return None


def _fix_and_parse_json(text: str) -> dict:
    """
    Attempt to fix and parse truncated/incomplete JSON.

    Strategy:
    1. Find where JSON becomes invalid
    2. Remove incomplete objects/arrays
    3. Close all remaining braces/brackets properly

    Args:
        text: Potentially incomplete JSON string

    Returns:
        Parsed dict

    Raises:
        json.JSONDecodeError: If parsing fails
    """
    # Remove trailing whitespace
    text = text.rstrip()

    # Track structure to find last valid complete key-value pair
    stack = []  # Track opening braces/brackets
    in_string = False
    escape_next = False
    last_complete_value_end = 0
    current_key = None
    in_key = False
    in_value = False
    colon_pos = -1

    i = 0
    while i < len(text):
        char = text[i]

        if escape_next:
            escape_next = False
            i += 1
            continue

        if char == "\\":
            escape_next = True
            i += 1
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            if not in_string:
                # String closed - this could be end of key or value
                if in_key:
                    in_key = False
                elif in_value:
                    in_value = False
                    last_complete_value_end = i + 1
            i += 1
            continue

        if not in_string:
            if char == ":" and colon_pos == -1:
                colon_pos = i
            elif char == "{" or char == "[":
                stack.append(char)
            elif char == "}" or char == "]":
                if stack:
                    opening = stack[-1]
                    if (opening == "{" and char == "}") or (opening == "[" and char == "]"):
                        stack.pop()
                        last_complete_value_end = i + 1

        i += 1

    # If we're not at the end, truncate to last complete position
    if last_complete_value_end > 0 and last_complete_value_end < len(text) - 10:
        # Truncate but keep the structure up to this point
        text = text[:last_complete_value_end]

        # If we truncated mid-value, add a placeholder or remove the key
        # Simplest: truncate to last complete key-value pair we know is valid
        # Then close the structures

    # Count what we have
    open_braces = text.count("{")
    close_braces = text.count("}")
    open_brackets = text.count("[")
    close_brackets = text.count("]")

    # Check if we're inside a string
    if text.count('"') % 2 == 1:
        text += '"'

    # Close brackets first
    for _ in range(max(0, open_brackets - close_brackets)):
        text += "]"

    # Close braces
    for _ in range(max(0, open_braces - close_braces)):
        text += "}"

    # Try parsing - if still fails, try removing trailing incomplete content more aggressively
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Last resort: progressively remove content from end until it parses
        for cut_len in [10, 50, 100, 200, 500, 1000]:
            if len(text) <= cut_len:
                break
            trimmed = text[:-cut_len]
            # Fix counts after trimming
            open_braces = trimmed.count("{")
            close_braces = trimmed.count("}")
            open_brackets = trimmed.count("[")
            close_brackets = trimmed.count("]")

            if trimmed.count('"') % 2 == 1:
                trimmed += '"'

            for _ in range(max(0, open_brackets - close_braces)):
                trimmed += "]"
            for _ in range(max(0, open_braces - close_braces)):
                trimmed += "}"

            try:
                return json.loads(trimmed)
            except:
                continue

        raise e
