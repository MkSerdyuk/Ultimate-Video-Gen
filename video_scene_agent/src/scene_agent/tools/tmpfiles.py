from __future__ import annotations
"""Temporary public media publishing through tmpfiles.org."""

import base64
import hashlib
import logging
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

import requests

from scene_agent.config import Config
from scene_agent.tools.storage import StorageBackend

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PublishedTmpfilesMedia:
    """Metadata for a tmpfiles upload used as a provider input."""

    source_uri: str
    public_url: str
    content_type: str
    size_bytes: int
    cache_key: str


class TmpfilesMediaPublisher:
    """Upload local/remote media to tmpfiles and return direct `/dl/...` URLs."""

    def __init__(
        self,
        config: Config,
        storage: StorageBackend,
        *,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config
        self.storage = storage
        self.session = session or requests.Session()
        self.upload_url = config.kling_tmpfiles_upload_url
        self.ttl_sec = config.kling_tmpfiles_ttl_sec
        self.max_bytes = config.kling_tmpfiles_max_bytes
        self.timeout = config.kling_tmpfiles_timeout_sec
        self._cache: dict[str, str] = {}
        self.uploads: list[PublishedTmpfilesMedia] = []

    def publish(self, uri: str, *, expected_kind: str | None = None) -> str:
        """Publish media to tmpfiles and return a direct public HTTPS URL."""
        if self._is_tmpfiles_direct_url(uri):
            self._verify_public_media(uri, expected_kind)
            return uri

        data, filename, content_type = self._load_media(uri)
        self._validate_media(uri, data, content_type, expected_kind)
        cache_key = self._cache_key(data, content_type)
        if cache_key in self._cache:
            return self._cache[cache_key]

        public_url = self._upload_bytes(filename, data, content_type)
        self._verify_public_media(public_url, expected_kind or _media_kind(content_type))
        self._cache[cache_key] = public_url
        self.uploads.append(
            PublishedTmpfilesMedia(
                source_uri=uri,
                public_url=public_url,
                content_type=content_type,
                size_bytes=len(data),
                cache_key=cache_key,
            )
        )
        return public_url

    def _load_media(self, uri: str) -> tuple[bytes, str, str]:
        if uri.startswith("data:"):
            return self._load_data_url(uri)
        if uri.startswith("http://") or uri.startswith("https://"):
            return self._load_remote_media(uri)
        return self._load_local_media(uri)

    def _load_local_media(self, uri: str) -> tuple[bytes, str, str]:
        local_path = Path(self.storage.uri_to_local_path(uri)).expanduser().resolve()
        if not local_path.exists():
            raise FileNotFoundError(f"Kling input media file not found: {local_path}")
        size = local_path.stat().st_size
        if size > self.max_bytes:
            raise ValueError(f"Kling input media exceeds tmpfiles limit: {size} > {self.max_bytes} bytes")
        data = local_path.read_bytes()
        content_type = mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"
        return data, local_path.name, content_type

    def _load_remote_media(self, uri: str) -> tuple[bytes, str, str]:
        response = self.session.get(uri, timeout=self.timeout, stream=True)
        response.raise_for_status()

        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > self.max_bytes:
            raise ValueError(
                f"Kling input media exceeds tmpfiles limit: {content_length} > {self.max_bytes} bytes"
            )

        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > self.max_bytes:
                raise ValueError(f"Kling input media exceeds tmpfiles limit: {total} > {self.max_bytes} bytes")
            chunks.append(chunk)
        response.close()

        data = b"".join(chunks)
        content_type = _strip_content_type(response.headers.get("Content-Type")) or _guess_content_type(uri)
        filename = _filename_from_uri(uri, content_type)
        return data, filename, content_type

    def _load_data_url(self, uri: str) -> tuple[bytes, str, str]:
        header, sep, payload = uri.partition(",")
        if not sep or ";base64" not in header:
            raise ValueError("Only base64 data URLs are supported for Kling tmpfiles inputs")
        content_type = header.removeprefix("data:").split(";", 1)[0] or "application/octet-stream"
        data = base64.b64decode(payload)
        suffix = mimetypes.guess_extension(content_type) or ".bin"
        return data, f"media{suffix}", content_type

    def _upload_bytes(self, filename: str, data: bytes, content_type: str) -> str:
        response = self.session.post(
            self.upload_url,
            data={"expire": str(self.ttl_sec)},
            files={"file": (filename, data, content_type)},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "success":
            raise ValueError(f"tmpfiles upload failed: {payload}")

        url = ((payload.get("data") or {}).get("url") or payload.get("url") or "").strip()
        if not url:
            raise ValueError(f"tmpfiles response did not include a URL: {payload}")
        return self._direct_download_url(url)

    def _verify_public_media(self, url: str, expected_kind: str | None) -> None:
        response = self.session.get(
            url,
            headers={"Range": "bytes=0-0"},
            timeout=self.timeout,
            stream=True,
        )
        response.raise_for_status()
        content_type = _strip_content_type(response.headers.get("Content-Type"))
        response.close()
        kind = _media_kind(content_type)
        if kind not in {"image", "video"}:
            raise ValueError(f"tmpfiles direct URL returned non-media content type {content_type!r}: {url}")
        if expected_kind and kind != expected_kind:
            raise ValueError(
                f"tmpfiles direct URL returned {content_type!r}, expected {expected_kind}: {url}"
            )

    def _validate_media(
        self,
        uri: str,
        data: bytes,
        content_type: str,
        expected_kind: str | None,
    ) -> None:
        if not data:
            raise ValueError(f"Kling input media is empty: {uri}")
        if len(data) > self.max_bytes:
            raise ValueError(f"Kling input media exceeds tmpfiles limit: {len(data)} > {self.max_bytes} bytes")
        kind = _media_kind(content_type)
        if kind not in {"image", "video"}:
            raise ValueError(f"Kling input media must be image/video, got {content_type!r}: {uri}")
        if expected_kind and kind != expected_kind:
            raise ValueError(f"Kling input media must be {expected_kind}, got {content_type!r}: {uri}")

    def _direct_download_url(self, url: str) -> str:
        parts = urlsplit(url)
        if parts.scheme != "https":
            raise ValueError(f"tmpfiles URL must be HTTPS: {url}")
        path = parts.path
        if not path.startswith("/dl/"):
            path = f"/dl/{path.lstrip('/')}"
        return urlunsplit((parts.scheme, parts.netloc, path, "", ""))

    def _is_tmpfiles_direct_url(self, uri: str) -> bool:
        parts = urlsplit(uri)
        return parts.scheme == "https" and parts.netloc == "tmpfiles.org" and parts.path.startswith("/dl/")

    def _cache_key(self, data: bytes, content_type: str) -> str:
        digest = hashlib.sha256(data).hexdigest()
        return f"{digest}:{content_type}"


def _strip_content_type(value: str | None) -> str:
    return (value or "").split(";", 1)[0].strip().lower()


def _media_kind(content_type: str) -> str:
    if content_type.startswith("image/"):
        return "image"
    if content_type.startswith("video/"):
        return "video"
    return ""


def _guess_content_type(uri: str) -> str:
    path = unquote(urlsplit(uri).path)
    return mimetypes.guess_type(path)[0] or "application/octet-stream"


def _filename_from_uri(uri: str, content_type: str) -> str:
    name = Path(unquote(urlsplit(uri).path)).name
    if name:
        return name
    suffix = mimetypes.guess_extension(content_type) or ".bin"
    return f"media{suffix}"
