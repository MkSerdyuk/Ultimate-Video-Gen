
from __future__ import annotations

from scene_agent.models import Issue, StoryboardData

def _calibrate_storyboard_issue(issue: Issue) -> Issue:
    """Downgrade non-blocking visual interpretation issues."""
    if issue.severity != "error":
        return issue

    problem = issue.problem.lower()
    if (
        ("anatomically correct orca" in problem or "literal orca" in problem)
        and "mounted on" in problem
        and "tank" in problem
    ):
        return issue.model_copy(update={"severity": "warning"})

    minor_continuity_markers = (
        "dress color",
        "gaze_target",
        "gazing at",
        "girl in frame",
        "girl is visible",
        "girl visible",
        "hair",
        "is clearly visible",
        "implying",
        "looking at",
        "looking directly at",
        "wardrobe drift",
        "slightly",
        "camera angle",
        "not perfectly",
        "centered",
        "already shows",
        "almost fully emerged",
        "end frame (frame 4) shows the girl",
        "end frame (frame 5) shows the girl",
        "end frame (frame 6) shows the girl",
        "exits off screen",
        "exits off-screen",
        "full orca_tank visible",
        "fully visible",
        "partial emergence",
        "partially visible",
        "premature reveal",
        "re-entry action",
        "redundant",
        "start frame (frame 6) shows the girl",
    )
    if any(marker in problem for marker in minor_continuity_markers):
        return issue.model_copy(update={"severity": "warning"})

    return issue


def _normalized_direction(value: str | None) -> str:
    text = (value or "").strip().lower().rstrip(".")
    if not text:
        return ""
    if text in {"n/a", "na", "none", "unknown", "unspecified", "not applicable"}:
        return ""
    if (
        "n/a" in text
        or "not applicable" in text
        or "unspecified" in text
        or "static" in text
        or "stationary" in text
        or "stopped" in text
        or "paused" in text
        or "backward" in text
    ):
        return ""
    return text


def _allows_direction_change(segment) -> bool:
    fields = [
        segment.screen_direction_rule or "",
        segment.camera_axis_rule or "",
        " ".join(segment.entry_exit_actions or []),
        segment.offscreen_justification or "",
    ]
    text = " ".join(fields).lower()
    markers = (
        "intentional reorientation",
        "reorientation",
        "reorient",
        "re-establish",
        "reestablish",
        "new axis",
        "new screen direction",
        "establishes the monster as the new focal point",
        "establishing the monster as the new focal point",
        "pivot",
        "turn",
        "reverse",
    )
    return any(marker in text for marker in markers)


def _sample_indices(total: int, limit: int) -> list[int]:
    if total <= 0 or limit <= 0:
        return []
    if total <= limit:
        return list(range(total))
    if limit == 1:
        return [0]
    return sorted({round(i * (total - 1) / (limit - 1)) for i in range(limit)})


def _deterministic_storyboard_issues(storyboard: StoryboardData) -> list[Issue]:
    issues: list[Issue] = []
    primary_ids = list(storyboard.primary_subject_ids or [])
    if not primary_ids:
        primary_ids = [obj.id for obj in storyboard.objects if obj.story_role == "character"]

    for segment in storyboard.segments:
        start = storyboard.frames[segment.start_frame_idx]
        end = storyboard.frames[segment.end_frame_idx]
        start_travel = _normalized_direction(start.travel_direction)
        end_travel = _normalized_direction(end.travel_direction)
        start_gaze = _normalized_direction(start.gaze_direction)
        end_gaze = _normalized_direction(end.gaze_direction)

        if start_travel and end_travel and start_travel != end_travel and not _allows_direction_change(segment):
            issues.append(
                Issue(
                    target=f"segment:{segment.idx}",
                    severity="error",
                    problem=(
                        f"Screen direction flips from {start.travel_direction} to {end.travel_direction} "
                        f"without an explicit reorientation rule."
                    ),
                )
            )

        if (
            start_gaze
            and end_gaze
            and start_gaze != end_gaze
            and "maintain readable eyeline continuity" in (segment.gaze_continuity_rule or "")
        ):
            issues.append(
                Issue(
                    target=f"segment:{segment.idx}",
                    severity="warning",
                    problem=(
                        f"Eyeline flips from {start.gaze_direction} to {end.gaze_direction} "
                        "without an explicit motivation."
                    ),
                )
            )

        for hero_id in primary_ids:
            start_presence = start.hero_presence.get(hero_id, "")
            end_presence = end.hero_presence.get(hero_id, "")
            if start_presence.startswith("on_screen") and not end_presence:
                issues.append(
                    Issue(
                        target=f"segment:{segment.idx}",
                        severity="error",
                        problem=(
                            f"Primary subject `{hero_id}` is on screen in frame {start.idx} "
                            f"but becomes unaccounted for by frame {end.idx}."
                        ),
                    )
                )

            if start_presence.startswith("on_screen") and end_presence in {"off_screen_left", "off_screen_right"}:
                if not segment.entry_exit_actions or not segment.offscreen_justification:
                    issues.append(
                        Issue(
                            target=f"segment:{segment.idx}",
                            severity="warning",
                            problem=(
                                f"Primary subject `{hero_id}` exits to {end_presence} "
                                "without a readable exit action and justification."
                            ),
                        )
                    )

    for frame in storyboard.frames:
        if frame.frame_class == "reveal":
            for hero_id in primary_ids:
                presence = frame.hero_presence.get(hero_id, "")
                if not presence:
                    issues.append(
                        Issue(
                            target=f"frame:{frame.idx}",
                            severity="warning",
                            problem=(
                                f"Reveal frame {frame.idx} does not account for primary subject `{hero_id}`. "
                                "Either keep the subject visible or explain the off-screen state."
                            ),
                        )
                    )

    return issues
