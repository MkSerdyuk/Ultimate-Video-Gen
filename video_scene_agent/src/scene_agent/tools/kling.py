from __future__ import annotations
"""Direct Kling 3.0 Omni video generation and segment repair."""

import logging
import mimetypes
import time
from pathlib import Path
from typing import Any

import jwt
import requests

from scene_agent.config import Config
from scene_agent.tools.storage import StorageBackend
from scene_agent.tools.tmpfiles import TmpfilesMediaPublisher
from scene_agent.utils.aspect_ratio import normalize_aspect_ratio

log = logging.getLogger(__name__)


OMNI_VIDEO_ENDPOINT = "/v1/videos/omni-video"


def duration_to_num_frames(duration_sec: float, fps: int | float) -> int:
    """Map storyboard duration to an inclusive frame count for metadata/tests."""
    return max(2, round(float(duration_sec) * float(fps)) + 1)


def normalize_kling_duration(duration_sec: float | int | str) -> str:
    """Normalize runtime duration to Kling 3.0 Omni Standard's supported 3-15 second range."""
    seconds = int(round(float(duration_sec)))
    return str(min(15, max(3, seconds)))


def _estimated_duration_seconds(duration_sec: float | int | str) -> float:
    return float(normalize_kling_duration(duration_sec))


def _normalize_negative_items(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def fold_negative_into_prompt(prompt: str, negative: Any) -> str:
    """Fold negative constraints into the Omni prompt as explicit forbidden failures."""
    items = list(dict.fromkeys(_normalize_negative_items(negative)))
    base = prompt.strip()
    if not items:
        return base

    forbidden = "\n".join(f"- {item}" for item in items)
    return f"{base}\n\nForbidden failures / must not:\n{forbidden}".strip()


class KlingTool:
    """Synchronous direct connector for Kling 3.0 Omni video generation and editing."""

    def __init__(
        self,
        config: Config,
        storage: StorageBackend,
        *,
        media_public_url_base: str | None = None,
        media_publisher: Any | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config
        self.storage = storage
        self.media_public_url_base = media_public_url_base
        self.session = session or requests.Session()
        self.media_publisher = media_publisher or (
            TmpfilesMediaPublisher(config, storage, session=self.session)
            if config.kling_use_tmpfiles
            else None
        )
        self.base_url = config.kling_api_base.rstrip("/")
        self.model_name = config.kling_video_model
        self.mode = config.kling_mode
        self.sound = config.kling_sound
        self.poll_timeout = config.kling_poll_timeout_sec
        self.poll_request_timeout = config.kling_poll_request_timeout_sec
        self.poll_interval = config.kling_poll_interval_sec
        self.last_generation_task_ids: list[str] = []
        self.last_edit_task_ids: list[str] = []
        self.last_media_input_urls: list[str] = []

        log.info("KlingTool initialized with model=%s mode=%s sound=%s", self.model_name, self.mode, self.sound)

    def _encode_jwt_token(self) -> str:
        """Generate Kling's HS256 JWT bearer token."""
        if not self.config.kling_access_key or not self.config.kling_secret_key:
            raise ValueError("KLING_ACCESS_KEY and KLING_SECRET_KEY are required for Kling video generation")

        now = int(time.time())
        return jwt.encode(
            {"iss": self.config.kling_access_key, "exp": now + 1800, "nbf": now - 5},
            self.config.kling_secret_key,
            algorithm="HS256",
            headers={"alg": "HS256", "typ": "JWT"},
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._encode_jwt_token()}",
            "Content-Type": "application/json",
        }

    def _public_media_url(self, uri: str) -> str:
        """Resolve a local artifact URI to a provider-reachable HTTPS URL."""
        if uri.startswith("https://"):
            return uri
        if uri.startswith("http://"):
            raise ValueError(f"Kling media inputs must use HTTPS URLs, got: {uri}")
        if uri.startswith("data:"):
            raise ValueError("Kling media inputs must be provider-reachable HTTPS URLs, not data URLs")

        public_base = self.media_public_url_base or getattr(self.storage, "public_url_base", None)
        if not public_base:
            raise ValueError(
                "Kling requires public HTTPS media inputs. Set STORAGE_PUBLIC_URL_BASE "
                "or KLING_MEDIA_PUBLIC_URL_BASE for run artifacts."
            )
        if not public_base.startswith("https://"):
            raise ValueError(f"Kling media public URL base must be HTTPS, got: {public_base}")

        local_path = Path(self.storage.uri_to_local_path(uri)).resolve()
        base_path = Path(getattr(self.storage, "base_path", "")).resolve()
        try:
            key = local_path.relative_to(base_path).as_posix()
        except ValueError as exc:
            raise ValueError(f"Kling media file is outside storage base path: {local_path}") from exc

        return f"{public_base.rstrip('/')}/{key.lstrip('/')}"

    def _kling_media_url(self, uri: str, *, expected_kind: str) -> str:
        """Resolve a media input to a provider-fetchable URL for Kling."""
        if self.media_publisher is not None:
            url = self.media_publisher.publish(uri, expected_kind=expected_kind)
        else:
            url = self._public_media_url(uri)
        self.last_media_input_urls.append(url)
        return url

    def build_generation_payload(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Build a Kling 3.0 Omni start/end-frame generation request."""
        prompt = fold_negative_into_prompt(spec.get("prompt", ""), spec.get("negative_prompt", ""))
        prompt = (
            "Use <<<image_1>>> as the exact first frame and <<<image_2>>> as the exact end frame. "
            "Generate one continuous bridge that lands cleanly on <<<image_2>>>.\n\n"
            f"{prompt}"
        ).strip()[:2500]

        payload = {
            "model_name": self.model_name,
            "prompt": prompt,
            "image_list": [
                {
                    "image_url": self._kling_media_url(spec["start_image_uri"], expected_kind="image"),
                    "type": "first_frame",
                },
                {
                    "image_url": self._kling_media_url(spec["end_image_uri"], expected_kind="image"),
                    "type": "end_frame",
                },
            ],
            "mode": self.mode,
            "duration": normalize_kling_duration(spec.get("duration_sec", 5.0)),
            "sound": self.sound,
        }
        if spec.get("aspect_ratio"):
            payload["aspect_ratio"] = normalize_aspect_ratio(spec.get("aspect_ratio"))
        return payload

    def estimate_generation_tokens(self, spec: dict[str, Any]) -> float:
        """Estimate Kling resource units for a start/end-frame generation request."""
        seconds = _estimated_duration_seconds(spec.get("duration_sec", 5.0))
        return round(seconds * self.config.kling_generation_tokens_per_second, 4)

    def estimate_edit_tokens(self, spec: dict[str, Any]) -> float:
        """Estimate Kling resource units for a video-input edit request."""
        seconds = _estimated_duration_seconds(spec.get("duration_sec", 5.0))
        return round(seconds * self.config.kling_edit_tokens_per_second, 4)

    def build_edit_payload(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Build a Kling 3.0 Omni feature-guided regeneration request for segment repair."""
        prompt = (
            "Generate a new clip, not a base video edit. Use <<<video_1>>> as the source motion, "
            "camera, timing, and composition reference."
        )
        if spec.get("start_image_uri"):
            prompt += " Use <<<image_1>>> as the exact first-frame visual target."
        prompt += (
            " Produce an edit-style improved version that corrects only the requested defects "
            "while preserving scene identity and continuity."
        )
        prompt = f"{prompt}\n\n{fold_negative_into_prompt(spec.get('prompt', ''), spec.get('negative_prompt', ''))}".strip()
        prompt = prompt[:2500]

        payload: dict[str, Any] = {
            "model_name": self.model_name,
            "prompt": prompt,
            "video_list": [
                {
                    "video_url": self._kling_media_url(spec["segment_uri"], expected_kind="video"),
                    "refer_type": "feature",
                    "keep_original_sound": "no",
                }
            ],
            "mode": self.mode,
            "duration": normalize_kling_duration(spec.get("duration_sec", 5.0)),
            "sound": self.sound,
        }
        if spec.get("aspect_ratio"):
            payload["aspect_ratio"] = normalize_aspect_ratio(spec.get("aspect_ratio"))
        if spec.get("start_image_uri"):
            payload["image_list"] = [
                {
                    "image_url": self._kling_media_url(spec["start_image_uri"], expected_kind="image"),
                    "type": "first_frame",
                }
            ]
        return payload

    def _submit_task(self, payload: dict[str, Any]) -> str:
        url = f"{self.base_url}{OMNI_VIDEO_ENDPOINT}"
        response = self.session.post(
            url,
            headers=self._headers(),
            json=payload,
            timeout=min(self.config.request_timeout, 30),
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            message = response.text[:1000] if response.text else str(exc)
            raise requests.HTTPError(
                f"Kling API HTTP {response.status_code}: {message}",
                response=response,
            ) from exc
        result = response.json()
        if result.get("code") not in {0, "0", None}:
            raise ValueError(f"Kling API error: {result.get('message') or result}")

        task_id = (result.get("data") or {}).get("task_id")
        if not task_id:
            raise ValueError(f"Kling response did not include data.task_id: {result}")
        return str(task_id)

    def _poll_task(self, task_id: str) -> str:
        url = f"{self.base_url}{OMNI_VIDEO_ENDPOINT}/{task_id}"
        deadline = time.time() + self.poll_timeout

        while time.time() < deadline:
            response = self.session.get(
                url,
                headers=self._headers(),
                timeout=self.poll_request_timeout,
            )
            response.raise_for_status()
            result = response.json()
            if result.get("code") not in {0, "0", None}:
                raise ValueError(f"Kling task poll error: {result.get('message') or result}")

            data = result.get("data") or {}
            videos = ((data.get("task_result") or {}).get("videos") or [])
            if videos and videos[0].get("url"):
                return str(videos[0]["url"])

            status = str(data.get("task_status", "")).lower()
            if status == "failed":
                raise ValueError(f"Kling task failed: {data.get('task_status_msg') or result}")
            if status in {"succeed", "success"} and not videos:
                raise ValueError(f"Kling task succeeded without a result video: {result}")

            time.sleep(self.poll_interval)

        raise TimeoutError(f"Kling video task timed out after {self.poll_timeout} seconds: {task_id}")

    def generate_segment(self, spec: dict[str, Any]) -> str:
        """Generate one segment from start/end keyframes."""
        task_id = self._submit_task(self.build_generation_payload(spec))
        self.last_generation_task_ids.append(task_id)
        return self._download_and_store(self._poll_task(task_id), key=f"segments/kling-{task_id}.mp4")

    def edit_segment(self, spec: dict[str, Any]) -> str:
        """Repair one existing segment using Kling instruction-based video editing."""
        task_id = self._submit_task(self.build_edit_payload(spec))
        self.last_edit_task_ids.append(task_id)
        return self._download_and_store(self._poll_task(task_id), key=f"segments/kling-edit-{task_id}.mp4")

    def generate_multiple_segments(self, specs: list[dict[str, Any]]) -> list[str]:
        """Generate multiple image-to-video segments sequentially."""
        self.last_generation_task_ids = []
        self.last_media_input_urls = []
        uris: list[str] = []
        for idx, spec in enumerate(specs):
            log.info("Generating Kling segment %d/%d", idx + 1, len(specs))
            uris.append(self.generate_segment(spec))
        return uris

    def edit_multiple_segments(self, specs: list[dict[str, Any]]) -> list[str]:
        """Repair multiple existing segments sequentially."""
        self.last_edit_task_ids = []
        self.last_media_input_urls = []
        uris: list[str] = []
        for idx, spec in enumerate(specs):
            log.info("Editing Kling segment %d/%d", idx + 1, len(specs))
            uris.append(self.edit_segment(spec))
        return uris

    def _download_and_store(self, url: str, *, key: str) -> str:
        response = self.session.get(url, timeout=60)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type") or mimetypes.guess_type(url)[0] or "video/mp4"
        if not Path(key).suffix:
            suffix = mimetypes.guess_extension(content_type) or ".mp4"
            key = f"{key}{suffix}"
        return self.storage.put_bytes(response.content, content_type, key=key)


def create_kling_tool(
    config: Config,
    storage: StorageBackend,
    *,
    media_public_url_base: str | None = None,
    media_publisher: Any | None = None,
) -> KlingTool:
    """Factory function to create the direct Kling video tool."""
    return KlingTool(
        config,
        storage,
        media_public_url_base=media_public_url_base,
        media_publisher=media_publisher,
    )
