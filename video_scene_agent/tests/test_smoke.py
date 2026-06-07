"""Smoke tests for main components.

These are basic integration tests to verify that core components
can be initialized and basic operations work.
"""

import importlib.util
import logging
import os
import tempfile
import urllib.request
from pathlib import Path

import pytest

from scene_agent.config import Config
from scene_agent.models import Constraints, SceneState
from scene_agent.prompts import (
    DIRECTOR_WORLD_SYSTEM,
    format_director_world_user,
)
from scene_agent.tools.openrouter_llm import OpenRouterLLM
from scene_agent.tools.storage import LocalStorageBackend


class TestConfig:
    """Smoke tests for Config module."""

    def test_config_from_env_missing_key(self):
        """Test that Config raises error without API key."""
        # Ensure no API key in environment
        original = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = ""

        try:
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
                Config.from_env()
        finally:
            if original:
                os.environ["OPENROUTER_API_KEY"] = original
            elif "OPENROUTER_API_KEY" in os.environ:
                del os.environ["OPENROUTER_API_KEY"]

    def test_config_from_env_uses_kling_defaults(self, monkeypatch):
        """Test that Kling keys are the canonical video provider credentials."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter")
        monkeypatch.setenv("KLING_ACCESS_KEY", "test-access")
        monkeypatch.setenv("KLING_SECRET_KEY", "test-secret")
        for name in [
            "KLING_API_BASE",
            "KLING_VIDEO_MODEL",
            "KLING_MODE",
            "KLING_SOUND",
            "KLING_POLL_TIMEOUT_SEC",
            "KLING_POLL_REQUEST_TIMEOUT_SEC",
            "KLING_POLL_INTERVAL_SEC",
            "KLING_MEDIA_PUBLIC_URL_BASE",
            "KLING_RUN_TOKEN_LIMIT",
            "KLING_GENERATION_TOKENS_PER_SECOND",
            "KLING_EDIT_TOKENS_PER_SECOND",
            "KLING_USE_TMPFILES",
            "KLING_TMPFILES_UPLOAD_URL",
            "KLING_TMPFILES_TTL_SEC",
            "KLING_TMPFILES_MAX_BYTES",
            "KLING_TMPFILES_TIMEOUT_SEC",
        ]:
            monkeypatch.delenv(name, raising=False)

        config = Config.from_env()

        assert config.kling_access_key == "test-access"
        assert config.kling_secret_key == "test-secret"
        assert config.kling_api_base == "https://api-singapore.klingai.com"
        assert config.kling_video_model == "kling-v3-omni"
        assert config.kling_mode == "std"
        assert config.kling_sound == "off"
        assert config.kling_poll_timeout_sec == 900
        assert config.kling_poll_request_timeout_sec == 30
        assert config.kling_poll_interval_sec == 2.0
        assert config.kling_media_public_url_base is None
        assert config.kling_run_token_limit == 60.0
        assert config.kling_generation_tokens_per_second == 0.6
        assert config.kling_edit_tokens_per_second == 0.9
        assert config.kling_use_tmpfiles is True
        assert config.kling_tmpfiles_upload_url == "https://tmpfiles.org/api/v1/upload"
        assert config.kling_tmpfiles_ttl_sec == 172800
        assert config.kling_tmpfiles_max_bytes == 100000000
        assert config.kling_tmpfiles_timeout_sec == 120
        assert config.openrouter_temperature == 0.0

    def test_run_public_url_base_appends_run_scope(self):
        config = Config(
            openrouter_api_key="test-openrouter",
            storage_public_url_base="https://example.com/artifacts/",
        )

        assert config.run_public_url_base("run-1") == "https://example.com/artifacts/runs/run-1"
        assert config.run_kling_media_public_url_base("run-1") == "https://example.com/artifacts/runs/run-1"

    def test_deployment_job_variables_forward_public_storage_and_smoke_env(self, monkeypatch):
        register_path = Path(__file__).resolve().parents[1] / "deploy" / "prefect" / "register_deployments.py"
        spec = importlib.util.spec_from_file_location("register_deployments", register_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter")
        monkeypatch.setenv("KLING_ACCESS_KEY", "access")
        monkeypatch.setenv("KLING_SECRET_KEY", "secret")
        monkeypatch.setenv("STORAGE_PUBLIC_URL_BASE", "https://example.com/artifacts")
        monkeypatch.setenv("KLING_VIDEO_MODEL", "kling-v3-omni")
        monkeypatch.setenv("KLING_MODE", "std")
        monkeypatch.setenv("KLING_SOUND", "off")
        monkeypatch.setenv("KLING_RUN_TOKEN_LIMIT", "60")
        monkeypatch.setenv("KLING_GENERATION_TOKENS_PER_SECOND", "0.6")
        monkeypatch.setenv("KLING_EDIT_TOKENS_PER_SECOND", "0.9")
        monkeypatch.setenv("KLING_USE_TMPFILES", "1")
        monkeypatch.setenv("KLING_TMPFILES_TTL_SEC", "172800")
        monkeypatch.setenv("IMAGE_SIZE", "512x512")

        job = module.base_job_variables("image:test")
        env = job["env"]

        assert env["STORAGE_PATH"] == "/app/artifacts"
        assert env["STORAGE_PUBLIC_URL_BASE"] == "https://example.com/artifacts"
        assert env["KLING_ACCESS_KEY"] == "access"
        assert env["KLING_SECRET_KEY"] == "secret"
        assert env["KLING_VIDEO_MODEL"] == "kling-v3-omni"
        assert env["KLING_MODE"] == "std"
        assert env["KLING_SOUND"] == "off"
        assert env["KLING_RUN_TOKEN_LIMIT"] == "60"
        assert env["KLING_GENERATION_TOKENS_PER_SECOND"] == "0.6"
        assert env["KLING_EDIT_TOKENS_PER_SECOND"] == "0.9"
        assert env["KLING_USE_TMPFILES"] == "1"
        assert env["KLING_TMPFILES_TTL_SEC"] == "172800"
        assert env["IMAGE_SIZE"] == "512x512"


class TestStorageBackend:
    """Smoke tests for StorageBackend."""

    def test_local_storage_initialization(self):
        """Test that LocalStorageBackend can be initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorageBackend(base_path=tmpdir)
            assert storage.base_path == Path(tmpdir).resolve()

    def test_local_storage_put_and_get(self):
        """Test that LocalStorageBackend can store and retrieve data."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorageBackend(base_path=tmpdir)

            # Test put_bytes
            data = b"test image data"
            uri = storage.put_bytes(data, "image/png", "test.png")
            assert uri

            # Test get_bytes
            retrieved = storage.get_bytes(uri)
            assert retrieved == data

    def test_local_storage_public_uri(self):
        """Test public URI generation."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorageBackend(base_path=tmpdir)
            uri = storage.put_bytes(b"data", "text/plain", "test.txt")

            # Get public URI
            public = storage.get_public_uri("test.txt")
            assert "test.txt" in public or "file://" in public

    def test_local_storage_http_server_serves_file(self):
        """Test that the optional HTTP server actually serves stored files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorageBackend(base_path=tmpdir)
            storage.put_bytes(b"hello", "text/plain", "hello.txt")
            base_url = storage.start_http_server(port=8126)
            response = urllib.request.urlopen(f"{base_url}/hello.txt", timeout=3)
            assert response.read() == b"hello"
            storage.stop_http_server()


class TestOpenRouterLLM:
    """Smoke tests for OpenRouter LLM client."""

    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="Requires OPENROUTER_API_KEY"
    )
    def test_llm_initialization(self):
        """Test that LLM can be initialized with config."""
        config = Config.from_env()
        llm = OpenRouterLLM(config)
        assert llm is not None
        assert llm.default_model is not None

    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="Requires OPENROUTER_API_KEY"
    )
    def test_llm_chat(self):
        """Test basic LLM chat call."""
        from openai import NotFoundError
        config = Config.from_env()
        llm = OpenRouterLLM(config)

        try:
            response = llm.chat(
                user="Say 'test' in uppercase",
                system="You are a helpful assistant.",
                temperature=0.0,
            )

            assert isinstance(response, str)
            assert len(response) > 0
        except NotFoundError as e:
            # Model not found - skip test if default model is invalid
            if "404" in str(e) or "not found" in str(e).lower():
                pytest.skip("Default model not available on OpenRouter")
            raise


class TestLogging:
    """Smoke tests for logging setup."""

    def test_setup_logging_is_idempotent(self):
        from scene_agent.utils.log import setup_logging

        root_logger = logging.getLogger()
        original_handlers = list(root_logger.handlers)
        try:
            root_logger.handlers = []
            setup_logging()
            assert len(root_logger.handlers) == 1
            setup_logging()
            assert len(root_logger.handlers) == 1
        finally:
            root_logger.handlers = original_handlers

    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="Requires OPENROUTER_API_KEY"
    )
    def test_llm_chat_with_history(self):
        """Test LLM chat with message history."""
        from openai import NotFoundError
        config = Config.from_env()
        llm = OpenRouterLLM(config)

        try:
            messages = [
                {"role": "user", "content": "Remember the number 42"},
                {"role": "assistant", "content": "I'll remember 42"},
            ]

            response = llm.chat_with_history(messages=messages)

            assert isinstance(response, str)
            assert len(response) > 0
        except NotFoundError as e:
            # Model not found - skip test if default model is invalid
            if "404" in str(e) or "not found" in str(e).lower():
                pytest.skip("Default model not available on OpenRouter")
            raise


class TestSceneState:
    """Smoke tests for SceneState model."""

    def test_state_initialization(self):
        """Test that state can be initialized."""
        state = SceneState(user_brief="Test brief")
        assert state.user_brief == "Test brief"
        assert state.status == "pending"

    def test_state_with_constraints(self):
        """Test state with custom constraints."""
        state = SceneState(
            user_brief="Test",
            constraints=Constraints(
                aspect_ratio="9:16",
                duration_sec=10.0,
                style_tags=["test"],
            ),
        )
        assert state.constraints.aspect_ratio == "9:16"
        assert state.constraints.style_tags == ["test"]

    def test_state_update_fields(self):
        """Test that state fields can be updated."""
        state = SceneState(user_brief="Test")
        state.frame_uris = ["frame1", "frame2"]
        state.sb_iteration = 1

        assert len(state.frame_uris) == 2
        assert state.sb_iteration == 1

    def test_state_status_transitions(self):
        """Test state status transitions."""
        state = SceneState(user_brief="Test")
        assert state.status == "pending"

        state.status = "running"
        assert state.status == "running"

        state.status = "completed"
        assert state.status == "completed"

        state.status = "failed"
        assert state.status == "failed"


class TestPrompts:
    """Smoke tests for prompts module."""

    def test_all_system_prompts_defined(self):
        """Test that all system prompts are defined."""
        from scene_agent.prompts import (
            DIRECTOR_WORLD_SYSTEM,
            DIRECTOR_STORYBOARD_SYSTEM,
            DIRECTOR_KEYFRAMES_BATCH_SYSTEM,
            SB_REVIEW_SYSTEM,
            SB_FIX_SYSTEM,
            VID_REVIEW_SYSTEM,
            VID_FIX_SYSTEM,
        )

        assert DIRECTOR_WORLD_SYSTEM
        assert DIRECTOR_STORYBOARD_SYSTEM
        assert DIRECTOR_KEYFRAMES_BATCH_SYSTEM
        assert SB_REVIEW_SYSTEM
        assert SB_FIX_SYSTEM
        assert VID_REVIEW_SYSTEM
        assert VID_FIX_SYSTEM

    def test_all_format_functions_defined(self):
        """Test that all format functions are defined."""
        from scene_agent.prompts import (
            format_director_world_user,
            format_director_storyboard_user,
            format_sb_review_user,
            format_sb_fix_user,
            format_vid_review_user,
            format_vid_fix_user,
            format_director_keyframes_batch_user,
            format_operator_generate_segments_jobs_user,
            format_operator_stitch_plan_user,
        )

        # Each should be callable and return a string
        assert callable(format_director_world_user)
        assert callable(format_director_storyboard_user)
        assert callable(format_sb_review_user)
        assert callable(format_sb_fix_user)
        assert callable(format_vid_review_user)
        assert callable(format_vid_fix_user)
        assert callable(format_director_keyframes_batch_user)
        assert callable(format_operator_generate_segments_jobs_user)
        assert callable(format_operator_stitch_plan_user)

    def test_format_functions_return_strings(self):
        """Test that format functions return valid strings."""
        from scene_agent.prompts import (
            format_director_world_user,
            format_director_storyboard_user,
            format_sb_review_user,
            format_sb_fix_user,
            format_vid_review_user,
            format_vid_fix_user,
        )
        assert format_director_world_user("test", "{}") != ""
        assert format_director_storyboard_user("test", "{}", "{}") != ""
        assert format_sb_review_user("{}") != ""
        assert format_sb_fix_user("{}", "[]") != ""
        assert format_vid_review_user("http://test", "{}") != ""
        assert format_vid_fix_user("{}", "[]") != ""


class TestModuleImports:
    """Smoke tests for module imports."""

    def test_imports_main_modules(self):
        """Test that main modules can be imported."""
        from scene_agent import config, models, prompts
        from scene_agent.tools import storage, openrouter_llm
        from scene_agent.pipeline import director, storyboard_editor
        assert config is not None
        assert models is not None
        assert prompts is not None
        assert storage is not None
        assert openrouter_llm is not None

    def test_imports_prefect_routing(self):
        from scene_agent.flows.routing import (
            route_after_sb_fix,
            route_after_sb_review,
            route_after_vid_fix,
            route_after_vid_review,
        )
        assert callable(route_after_sb_review)
        assert callable(route_after_vid_review)
        assert callable(route_after_sb_fix)
        assert callable(route_after_vid_fix)

    def test_imports_prefect_flow_and_tasks(self):
        from scene_agent.flows.tasks import (
            director_storyboard_task,
            director_world_task,
            keyframes_task,
            segments_edit_task,
            segments_task,
            stitch_task,
            storyboard_fix_task,
            storyboard_review_task,
            video_fix_task,
            video_review_task,
        )
        from scene_agent.prefect_flows import generate_scene_flow

        assert callable(generate_scene_flow)
        for task_fn in [
            director_world_task,
            director_storyboard_task,
            keyframes_task,
            storyboard_review_task,
            storyboard_fix_task,
            segments_task,
            segments_edit_task,
            stitch_task,
            video_review_task,
            video_fix_task,
        ]:
            assert task_fn is not None

    def test_imports_main(self):
        """Test that main module can be imported."""
        from scene_agent.main import run, main
        assert callable(run)
        assert callable(main)


