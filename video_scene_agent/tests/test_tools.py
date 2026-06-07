"""Focused regression tests for tool-layer fixes."""

from pathlib import Path

import jwt
import pytest
import requests

from scene_agent.config import Config
from scene_agent.providers import KlingVideoAdapter, PermanentProviderError, TransientProviderError
from scene_agent.tools.kling import KlingTool, duration_to_num_frames, fold_negative_into_prompt, normalize_kling_duration
from scene_agent.tools.openrouter_llm import OpenRouterLLM
from scene_agent.tools.openrouter_video_review import OpenRouterVideoReviewTool
from scene_agent.tools.stitch import StitchTool
from scene_agent.tools.storage import LocalStorageBackend
from scene_agent.tools.tmpfiles import TmpfilesMediaPublisher
from scene_agent.tools.vision_rewriter import normalize_negative_items


class FakeResponse:
    def __init__(
        self,
        *,
        json_data=None,
        headers=None,
        content: bytes = b"",
        status_code: int = 200,
    ):
        self._json_data = json_data or {}
        self.headers = headers or {}
        self.content = content
        self.status_code = status_code
        self.text = str(self._json_data)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self):
        return self._json_data

    def iter_content(self, chunk_size=8192):
        yield self.content

    def close(self):
        pass


class FakeTmpfilesSession:
    def __init__(self, *, verify_content_type: str = "image/png"):
        self.verify_content_type = verify_content_type
        self.posts = []
        self.gets = []

    def post(self, url, *, data=None, files=None, timeout=None):
        self.posts.append({"url": url, "data": data, "files": files, "timeout": timeout})
        filename = files["file"][0]
        return FakeResponse(
            json_data={"status": "success", "data": {"url": f"https://tmpfiles.org/abc/{filename}"}}
        )

    def get(self, url, *, headers=None, timeout=None, stream=False):
        self.gets.append({"url": url, "headers": headers, "timeout": timeout, "stream": stream})
        return FakeResponse(headers={"Content-Type": self.verify_content_type}, content=b"x")


class FakeMediaPublisher:
    def __init__(self):
        self.calls = []

    def publish(self, uri: str, *, expected_kind: str | None = None) -> str:
        self.calls.append((uri, expected_kind))
        ext = "mp4" if expected_kind == "video" else "png"
        return f"https://tmpfiles.org/dl/fake/media-{len(self.calls)}.{ext}"


class TestVisionRewriterHelpers:
    def test_normalize_negative_items_accepts_null_string_and_sequence(self):
        assert normalize_negative_items(None) == []
        assert normalize_negative_items(" text ") == ["text"]
        assert normalize_negative_items([" blur ", None, "", "warp"]) == ["blur", "warp"]


class TestVideoReviewPayload:
    def test_prepare_video_content_uses_video_url(self, tmp_path):
        storage = LocalStorageBackend(base_path=tmp_path)
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"0000")
        tool = OpenRouterVideoReviewTool(Config(openrouter_api_key="test-key"), storage)

        payload = tool._prepare_video_content(f"file://{video_path}")
        assert payload[0]["type"] == "video_url"
        assert payload[0]["video_url"]["url"].startswith("data:video/mp4;base64,")


class TestLlmContinuation:
    def test_chat_with_retry_accumulates_continuation(self):
        class Message:
            def __init__(self, content):
                self.content = content

        class Choice:
            def __init__(self, content, finish_reason="stop"):
                self.message = Message(content)
                self.finish_reason = finish_reason

        class Response:
            def __init__(self, content, finish_reason="stop"):
                self.choices = [Choice(content, finish_reason=finish_reason)]

        class FakeClient:
            def __init__(self):
                self.calls = 0
                self.chat = self
                self.completions = self

            def create(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return Response('{"hello":"wor', finish_reason="length")
                return Response('ld"}')

        llm = OpenRouterLLM(Config(openrouter_api_key="test-key"))
        llm.client = FakeClient()

        result = llm.chat_with_retry(
            user="u",
            system="s",
            json_mode=True,
            retries=1,
        )
        assert result == '{"hello":"world"}'


class TestLlmJsonMode:
    def test_chat_uses_response_format_for_json_mode(self):
        captured = {}

        class Message:
            def __init__(self, content):
                self.content = content

        class Choice:
            def __init__(self, content):
                self.message = Message(content)
                self.finish_reason = "stop"

        class Response:
            def __init__(self, content):
                self.choices = [Choice(content)]

        class FakeClient:
            def __init__(self):
                self.chat = self
                self.completions = self

            def create(self, **kwargs):
                captured.update(kwargs)
                return Response('{"ok":true}')

        llm = OpenRouterLLM(Config(openrouter_api_key="test-key"))
        llm.client = FakeClient()
        llm.chat(user="u", system="s", json_mode=True)

        assert captured["response_format"] == {"type": "json_object"}
        assert captured["extra_body"]["plugins"][0]["id"] == "response-healing"


class TestTmpfilesMediaPublisher:
    def test_upload_converts_tmpfiles_url_to_direct_dl_url_and_caches(self, tmp_path):
        storage = LocalStorageBackend(base_path=tmp_path)
        image = tmp_path / "frame.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\nimage")
        session = FakeTmpfilesSession()
        publisher = TmpfilesMediaPublisher(
            Config(openrouter_api_key="test-key"),
            storage,
            session=session,
        )

        first = publisher.publish(f"file://{image}", expected_kind="image")
        second = publisher.publish(f"file://{image}", expected_kind="image")

        assert first == "https://tmpfiles.org/dl/abc/frame.png"
        assert second == first
        assert len(session.posts) == 1
        assert session.posts[0]["data"]["expire"] == "172800"
        assert session.posts[0]["files"]["file"][:2] == ("frame.png", b"\x89PNG\r\n\x1a\nimage")
        assert session.gets[0]["url"] == "https://tmpfiles.org/dl/abc/frame.png"
        assert session.gets[0]["headers"] == {"Range": "bytes=0-0"}

    def test_upload_rejects_html_direct_url(self, tmp_path):
        storage = LocalStorageBackend(base_path=tmp_path)
        image = tmp_path / "frame.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\nimage")
        publisher = TmpfilesMediaPublisher(
            Config(openrouter_api_key="test-key"),
            storage,
            session=FakeTmpfilesSession(verify_content_type="text/html"),
        )

        with pytest.raises(ValueError, match="non-media"):
            publisher.publish(f"file://{image}", expected_kind="image")


class TestKlingPayloads:
    def test_duration_to_num_frames_uses_inclusive_endpoint_frame(self):
        assert duration_to_num_frames(1.0, 12) == 13
        assert duration_to_num_frames(2.0, 24) == 49
        assert duration_to_num_frames(0.01, 12) == 2

    def test_kling_duration_is_clamped_to_supported_range(self):
        assert normalize_kling_duration(1.0) == "3"
        assert normalize_kling_duration(7.2) == "7"
        assert normalize_kling_duration(20.0) == "15"

    def test_kling_token_estimates_use_budget_rates(self, tmp_path):
        config = Config(
            openrouter_api_key="test-key",
            kling_generation_tokens_per_second=0.6,
            kling_edit_tokens_per_second=0.9,
        )
        tool = KlingTool(config, LocalStorageBackend(base_path=tmp_path))

        assert tool.estimate_generation_tokens({"duration_sec": 4.0}) == 2.4
        assert tool.estimate_edit_tokens({"duration_sec": 4.0}) == 3.6
        assert tool.estimate_generation_tokens({"duration_sec": 1.0}) == 1.8

    def test_fold_negative_into_prompt(self):
        prompt = fold_negative_into_prompt("Move cleanly.", ["text", "watermark", "text"])
        assert "Move cleanly." in prompt
        assert "Forbidden failures / must not:" in prompt
        assert prompt.count("- text") == 1
        assert "- watermark" in prompt

    def test_jwt_uses_kling_access_and_secret(self, tmp_path):
        config = Config(
            openrouter_api_key="test-key",
            kling_access_key="access",
            kling_secret_key="secret-secret-secret-secret-secret-32",
        )
        tool = KlingTool(config, LocalStorageBackend(base_path=tmp_path))

        token = tool._encode_jwt_token()
        payload = jwt.decode(token, "secret-secret-secret-secret-secret-32", algorithms=["HS256"])

        assert payload["iss"] == "access"
        assert payload["exp"] > payload["nbf"]

    def test_generation_payload_is_standard_no_audio_and_anchor_aware(self, tmp_path):
        storage = LocalStorageBackend(base_path=tmp_path)
        start = tmp_path / "frames" / "start.png"
        end = tmp_path / "frames" / "end.png"
        start.parent.mkdir(parents=True)
        start.write_bytes(b"start")
        end.write_bytes(b"end")
        publisher = FakeMediaPublisher()
        tool = KlingTool(Config(openrouter_api_key="test-key"), storage, media_publisher=publisher)

        args = tool.build_generation_payload(
            {
                "start_image_uri": f"file://{start}",
                "end_image_uri": f"file://{end}",
                "prompt": "One continuous bridge.",
                "negative_prompt": "text, watermark",
                "duration_sec": 3.0,
                "fps": 12,
                "aspect_ratio": "16:9",
            }
        )

        assert args["model_name"] == "kling-v3-omni"
        assert args["mode"] == "std"
        assert args["sound"] == "off"
        assert "One continuous bridge." in args["prompt"]
        assert "Forbidden failures / must not" in args["prompt"]
        assert "negative_prompt" not in args
        assert args["image_list"] == [
            {"image_url": "https://tmpfiles.org/dl/fake/media-1.png", "type": "first_frame"},
            {"image_url": "https://tmpfiles.org/dl/fake/media-2.png", "type": "end_frame"},
        ]
        assert publisher.calls == [(f"file://{start}", "image"), (f"file://{end}", "image")]
        assert args["duration"] == "3"
        assert args["aspect_ratio"] == "16:9"

    def test_video_payloads_normalize_unknown_aspect_ratio_consistently(self, tmp_path):
        tool = KlingTool(
            Config(openrouter_api_key="test-key"),
            LocalStorageBackend(base_path=tmp_path),
            media_publisher=FakeMediaPublisher(),
        )

        generation = tool.build_generation_payload(
            {
                "start_image_uri": "https://example.com/start.png",
                "end_image_uri": "https://example.com/end.png",
                "prompt": "Move.",
                "aspect_ratio": "bad-ratio",
            }
        )
        edit = tool.build_edit_payload(
            {
                "segment_uri": "https://example.com/segment.mp4",
                "start_image_uri": "https://example.com/start.png",
                "prompt": "Repair.",
                "aspect_ratio": "bad-ratio",
            }
        )

        assert generation["aspect_ratio"] == "16:9"
        assert edit["aspect_ratio"] == "16:9"

    def test_edit_payload_uses_feature_generation_with_first_frame_reference(self, tmp_path):
        publisher = FakeMediaPublisher()
        tool = KlingTool(
            Config(openrouter_api_key="test-key"),
            LocalStorageBackend(base_path=tmp_path),
            media_publisher=publisher,
        )

        args = tool.build_edit_payload(
            {
                "segment_uri": "https://example.com/segment.mp4",
                "start_image_uri": "https://example.com/start.png",
                "end_image_uri": "https://example.com/end.png",
                "prompt": "Repair only the listed defect.",
                "negative_prompt": ["jitter"],
                "duration_sec": 1.0,
                "fps": 12,
                "aspect_ratio": "16:9",
            }
        )

        assert args["video_list"] == [
            {
                "video_url": "https://tmpfiles.org/dl/fake/media-1.mp4",
                "refer_type": "feature",
                "keep_original_sound": "no",
            }
        ]
        assert args["image_list"] == [
            {"image_url": "https://tmpfiles.org/dl/fake/media-2.png", "type": "first_frame"}
        ]
        assert publisher.calls == [
            ("https://example.com/segment.mp4", "video"),
            ("https://example.com/start.png", "image"),
        ]
        assert args["mode"] == "std"
        assert args["sound"] == "off"
        assert args["duration"] == "3"
        assert args["aspect_ratio"] == "16:9"
        assert "https://example.com/end.png" not in str(args)
        assert "not a base video edit" in args["prompt"]
        assert "jitter" in args["prompt"]

    def test_local_media_without_public_https_base_uses_tmpfiles(self, tmp_path):
        storage = LocalStorageBackend(base_path=tmp_path)
        image = tmp_path / "image.png"
        image.write_bytes(b"image")
        publisher = FakeMediaPublisher()
        tool = KlingTool(Config(openrouter_api_key="test-key"), storage, media_publisher=publisher)

        args = tool.build_generation_payload(
            {
                "start_image_uri": f"file://{image}",
                "end_image_uri": f"file://{image}",
                "prompt": "Move.",
            }
        )

        assert args["image_list"][0]["image_url"].startswith("https://tmpfiles.org/dl/")
        assert publisher.calls == [(f"file://{image}", "image"), (f"file://{image}", "image")]

    def test_local_media_without_tmpfiles_or_public_https_base_fails_before_provider_call(self, tmp_path):
        storage = LocalStorageBackend(base_path=tmp_path)
        image = tmp_path / "image.png"
        image.write_bytes(b"image")
        tool = KlingTool(Config(openrouter_api_key="test-key", kling_use_tmpfiles=False), storage)

        with pytest.raises(ValueError, match="public HTTPS"):
            tool.build_generation_payload(
                {
                    "start_image_uri": f"file://{image}",
                    "end_image_uri": f"file://{image}",
                    "prompt": "Move.",
                }
            )


class TestKlingAdapter:
    def test_retryable_generation_failure_maps_to_transient(self):
        class Tool:
            def generate_multiple_segments(self, specs):
                raise TimeoutError("timeout")

        with pytest.raises(TransientProviderError):
            KlingVideoAdapter(Tool()).generate_multiple_segments([{}])

    def test_permanent_generation_failure_maps_to_permanent(self):
        class Tool:
            def generate_multiple_segments(self, specs):
                raise ValueError("bad input")

        with pytest.raises(PermanentProviderError):
            KlingVideoAdapter(Tool()).generate_multiple_segments([{}])

    def test_retryable_edit_failure_maps_to_transient(self):
        class Tool:
            def edit_multiple_segments(self, specs):
                raise RuntimeError("rate limit")

        with pytest.raises(TransientProviderError):
            KlingVideoAdapter(Tool()).edit_multiple_segments([{}])


class TestStitchPlaceholders:
    def test_reencode_fallback_handles_mixed_still_placeholder_sizes(self, tmp_path):
        import shutil
        from io import BytesIO

        from PIL import Image

        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            pytest.skip("FFmpeg and FFprobe are required")

        storage = LocalStorageBackend(base_path=tmp_path)
        tool = StitchTool(storage, fps=12)

        uris = []
        for idx, size in enumerate([(640, 360), (320, 320)]):
            buffer = BytesIO()
            Image.new("RGB", size, (80 + idx * 30, 100, 130)).save(buffer, "PNG")
            image_uri = storage.put_bytes(buffer.getvalue(), "image/png", key=f"frame-{idx}.png")
            uris.append(
                tool.create_still_clip(
                    image_uri=image_uri,
                    duration_sec=0.5,
                    fps=12,
                    output_key=f"placeholder-{idx}.mp4",
                )
            )

        final_uri = tool.stitch(uris, fps=12, output_key="stitched-placeholder-test.mp4")
        assert Path(storage.uri_to_local_path(final_uri)).exists()
