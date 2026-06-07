from __future__ import annotations
"""Storage backend for artifacts (images, videos)."""

import os
import base64
import hashlib
import shutil
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional
import logging

try:
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    import socketserver
    import threading
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False

log = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract storage backend for artifacts."""

    @abstractmethod
    def put_bytes(self, data: bytes, content_type: str, key: str) -> str:
        """
        Store binary data and return a URI.

        Args:
            data: Binary data to store
            content_type: MIME type (e.g., "image/png", "video/mp4")
            key: Storage key (if None, will generate one)

        Returns:
            URI for accessing the stored data
        """
        pass

    @abstractmethod
    def get_bytes(self, uri: str) -> bytes:
        """Retrieve binary data from URI."""
        pass

    @abstractmethod
    def download_to_file(self, uri: str, path: str) -> None:
        """Download data from URI to a local file path."""
        pass

    @abstractmethod
    def get_public_uri(self, key: str) -> str:
        """Get a publicly accessible URI for the given key."""
        pass

    @abstractmethod
    def uri_to_local_path(self, uri: str) -> str:
        """Convert URI to local file path if applicable."""
        pass


class LocalStorageBackend(StorageBackend):
    """
    Local filesystem storage backend.

    URIs are formatted as: `file:///absolute/path` or `http://localhost:PORT/artifacts/{key}`
    when HTTP server is enabled.
    """

    def __init__(
        self,
        base_path: str | Path = "./artifacts",
        public_url_base: str | None = None,
    ):
        """
        Initialize local storage.

        Args:
            base_path: Base directory for storage
            public_url_base: Optional base URL for public access (e.g., "http://localhost:8000")
        """
        self.base_path = Path(base_path).expanduser().resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.public_url_base = public_url_base
        self._http_server: Optional[threading.Thread] = None
        self._http_port: Optional[int] = None
        self._httpd = None

        log.info(f"LocalStorageBackend initialized at {self.base_path}")

    def _generate_key(self, content_type: str, data: bytes | None = None) -> str:
        """Generate a unique storage key based on content hash."""
        ext = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/webp": ".webp",
            "video/mp4": ".mp4",
            "video/webm": ".webm",
        }.get(content_type, ".bin")

        if data:
            hash_digest = hashlib.sha256(data).hexdigest()[:16]
            return f"{hash_digest}{ext}"

        import time
        timestamp = int(time.time() * 1000)
        return f"{timestamp}{ext}"

    def put_bytes(self, data: bytes, content_type: str, key: str | None = None) -> str:
        """Store binary data and return a URI."""
        if key is None:
            key = self._generate_key(content_type, data)

        # Ensure key is safe
        key = key.lstrip("/")

        file_path = self.base_path / key
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(data)

        log.debug(f"Stored {len(data)} bytes to {file_path}")

        # Return file:// URI
        return f"file://{file_path}"

    def get_bytes(self, uri: str) -> bytes:
        """Retrieve binary data from URI."""
        path = self.uri_to_local_path(uri)
        with open(path, "rb") as f:
            return f.read()

    def download_to_file(self, uri: str, path: str) -> None:
        """Download data from URI to a local file path."""
        data = self.get_bytes(uri)
        dest_path = Path(path).expanduser().resolve()
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        with open(dest_path, "wb") as f:
            f.write(data)

        log.debug(f"Downloaded {uri} to {dest_path}")

    def get_public_uri(self, key: str) -> str:
        """Get a publicly accessible URI for the given key."""
        if self.public_url_base:
            # Remove leading slash from key if present
            key = key.lstrip("/")
            return f"{self.public_url_base.rstrip('/')}/{key}"
        # Fall back to file:// URI
        return f"file://{self.base_path / key.lstrip('/')}"

    def uri_to_local_path(self, uri: str) -> str:
        """Convert URI to local file path."""
        if uri.startswith("file://"):
            path = uri[7:]  # Remove "file://"
        elif uri.startswith("http://") or uri.startswith("https://"):
            # For HTTP URIs, assume they reference our storage
            # Extract the key part and map to local path
            parts = uri.split("/", 3)
            if len(parts) > 3:
                key = parts[3]
                path = str(self.base_path / key)
            else:
                raise ValueError(f"Cannot convert HTTP URI to local path: {uri}")
        else:
            # Assume it's already a local path
            path = uri

        return str(Path(path).expanduser().resolve())

    def start_http_server(self, port: int = 8000) -> str:
        """
        Start a simple HTTP server for serving files.

        Args:
            port: Port to listen on

        Returns:
            The base URL for accessing files
        """
        if not HTTP_AVAILABLE:
            log.warning("HTTP server not available, missing dependencies")
            return ""

        if self._http_server is not None:
            return f"http://localhost:{self._http_port}"

        base_path = self.base_path

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(base_path), **kwargs)

            def log_message(self, format, *args):
                pass  # Suppress HTTP logs

        handler = lambda *args, **kwargs: Handler(*args, **kwargs)

        try:
            try:
                server = socketserver.TCPServer(("", port), handler)
            except OSError:
                if port == 0:
                    raise
                log.warning("Port %s is busy; retrying HTTP storage server on a free port", port)
                server = socketserver.TCPServer(("", 0), handler)
            self._httpd = server
            self._http_port = int(server.server_address[1])

            def run_server():
                server.serve_forever()

            self._http_server = threading.Thread(
                target=run_server,
                daemon=True,
            )
            self._http_server.start()

            log.info(f"HTTP server started on port {self._http_port}")
            self.public_url_base = f"http://localhost:{self._http_port}"
            return self.public_url_base

        except OSError as e:
            log.warning(f"Failed to start HTTP server: {e}")
            return ""

    def stop_http_server(self) -> None:
        """Stop the HTTP server if running."""
        if self._http_server is not None:
            if self._httpd is not None:
                self._httpd.shutdown()
                self._httpd.server_close()
                self._httpd = None
            self._http_server = None
            self._http_port = None
            log.info("HTTP server reference cleared")


def decode_base64_image(data: str) -> tuple[bytes, str]:
    """
    Decode base64 image data.

    Args:
        data: Base64 data URL or raw base64 string

    Returns:
        Tuple of (bytes, content_type)
    """
    if data.startswith("data:"):
        # Parse data URL
        parts = data.split(",", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid data URL format")

        header = parts[0]
        content_type = header.split(":")[1].split(";")[0]
        b64_data = parts[1]
    else:
        content_type = "image/png"
        b64_data = data

    # Add padding if needed
    padding = 4 - len(b64_data) % 4
    if padding != 4:
        b64_data += "=" * padding

    image_bytes = base64.b64decode(b64_data)
    return image_bytes, content_type


def encode_base64(data: bytes, content_type: str = "image/png") -> str:
    """Encode binary data as base64 data URL."""
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{content_type};base64,{b64}"
