"""Integration tests with real API calls.

These tests require valid API keys in .env file.
Run with: pytest tests/test_integration.py -v -s
"""

import os
import pytest
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from scene_agent.config import Config
from scene_agent.tools.openrouter_llm import OpenRouterLLM
from scene_agent.tools.openrouter_image import OpenRouterImageTool
from scene_agent.tools.storage import LocalStorageBackend
from scene_agent.tools.kling import KlingTool


@pytest.fixture
def config() -> Config:
    """Get config from environment."""
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("Requires OPENROUTER_API_KEY")
    return Config.from_env()


@pytest.fixture
def storage(tmp_path) -> LocalStorageBackend:
    """Get storage backend for tests."""
    return LocalStorageBackend(base_path=str(tmp_path))


class TestOpenRouterText:
    """Integration tests for OpenRouter text generation."""

    def test_text_generation_simple(self, config: Config):
        """Test basic text generation."""
        llm = OpenRouterLLM(config)

        response = llm.chat(
            user="Say 'Hello World' in uppercase",
            system="You are a helpful assistant.",
            temperature=0.0,
        )

        assert isinstance(response, str)
        assert len(response) > 0
        print(f"Text response: {response}")

    def test_text_generation_json(self, config: Config):
        """Test text generation with JSON output."""
        llm = OpenRouterLLM(config)

        response = llm.chat(
            user='Return JSON: {"status": "ok", "value": 42}',
            system="You are a JSON API. Return only valid JSON.",
            temperature=0.0,
        )

        assert isinstance(response, str)
        assert len(response) > 0
        print(f"JSON response: {response}")

        # Try to parse as JSON (use clean_json_response like the actual code)
        from scene_agent.utils.json_llm import clean_json_response, parse_partial_json
        cleaned = clean_json_response(response)
        data = parse_partial_json(cleaned)
        assert data is not None
        assert isinstance(data, dict)


class TestOpenRouterImage:
    """Integration tests for OpenRouter image generation."""

    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="Requires OPENROUTER_API_KEY"
    )
    def test_image_generation_simple(self, config: Config, storage: LocalStorageBackend):
        """Test basic image generation."""
        image_tool = OpenRouterImageTool(config, storage)

        uri = image_tool.generate(
            prompt="A simple red circle on white background",
            aspect_ratio="1:1",
        )

        assert uri
        assert isinstance(uri, str)
        print(f"Image URI: {uri}")

        # Verify file exists
        from scene_agent.tools.storage import LocalStorageBackend
        if isinstance(storage, LocalStorageBackend):
            local_path = storage.uri_to_local_path(uri)
            assert Path(local_path).exists()
            file_size = Path(local_path).stat().st_size
            assert file_size > 1000  # At least 1KB
            print(f"Image size: {file_size} bytes")

    def test_list_available_models(self, config: Config):
        """List available models on OpenRouter to find working ones."""
        import requests

        headers = {
            "Authorization": f"Bearer {config.openrouter_api_key}",
            "Content-Type": "application/json",
        }

        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        print(f"\n=== Available models on OpenRouter ===")
        print(f"Total models: {len(data.get('data', []))}")

        # Find image-capable models
        image_models = []
        for model in data.get("data", []):
            model_id = model.get("id", "")
            name = model.get("name", "")
            context_length = model.get("context_length", 0)
            modalities = model.get("output_modalities", [])

            if "image" in modalities:
                image_models.append(model_id)
                print(f"  IMAGE MODEL: {model_id}")
                print(f"    Name: {name}")
                print(f"    Context: {context_length}")
                print(f"    Modalities: {modalities}")

        print(f"\nFound {len(image_models)} image-capable models")

        assert len(data.get("data", [])) > 0, "No models found"


class TestKlingAPI:
    """Integration tests for Kling video configuration."""

    @pytest.mark.skipif(
        not os.getenv("KLING_ACCESS_KEY") or not os.getenv("KLING_SECRET_KEY"),
        reason="Requires KLING_ACCESS_KEY and KLING_SECRET_KEY"
    )
    def test_kling_tool_configuration(self, config: Config, storage: LocalStorageBackend):
        """Test direct Kling tool initialization without starting a paid generation."""
        tool = KlingTool(config, storage)
        assert config.kling_access_key
        assert config.kling_secret_key
        assert tool.model_name == config.kling_video_model
        assert tool.mode == "std"
        assert tool.sound == "off"


class TestFullWorkflow:
    """Integration tests for the full workflow."""

    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="Requires OPENROUTER_API_KEY"
    )
    def test_director_world_node(self, config: Config):
        """Test the director_world node with real LLM."""
        from scene_agent.pipeline.director import DirectorTools, director_world
        from scene_agent.models import SceneState, Constraints

        llm = OpenRouterLLM(config)
        # Create a dummy image tool
        from unittest.mock import Mock
        image_tool = Mock()

        tools = DirectorTools(llm, image_tool)

        state = SceneState(
            user_brief="A peaceful mountain landscape at sunset",
            constraints=Constraints(
                aspect_ratio="16:9",
                duration_sec=5.0,
            ),
        )

        result = director_world(state, tools)

        assert "world_raw" in result
        assert result["world_raw"] is not None
        assert isinstance(result["world_raw"], dict)
        print(f"\nWorld generated: {result['world_raw']}")

        # Check expected fields
        assert "scene_background" in result["world_raw"] or "background" in result["world_raw"]

    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="Requires OPENROUTER_API_KEY"
    )
    def test_director_storyboard_node(self, config: Config):
        """Test the director_storyboard node with real LLM."""
        from scene_agent.pipeline.director import DirectorTools, director_storyboard
        from scene_agent.models import SceneState, Constraints

        llm = OpenRouterLLM(config)
        from unittest.mock import Mock
        image_tool = Mock()

        tools = DirectorTools(llm, image_tool)

        # Create state with pre-generated world
        state = SceneState(
            user_brief="A cat sitting on a windowsill",
            constraints=Constraints(
                aspect_ratio="16:9",
                duration_sec=5.0,
            ),
            world_raw={
                "scene_background": "Cozy living room with a large window",
                "objects": [
                    {"id": "cat", "name": "cat", "appearance": "orange tabby cat"}
                ],
                "style_guide": {
                    "style": "warm and cozy",
                    "palette": "warm orange, soft beige",
                }
            },
        )

        result = director_storyboard(state, tools)

        assert "storyboard_raw" in result
        assert result["storyboard_raw"] is not None
        assert isinstance(result["storyboard_raw"], dict)
        print(f"\nStoryboard generated: {result['storyboard_raw']}")

        # Check for frames
        storyboard = result["storyboard_raw"]
        assert "frames" in storyboard
        assert len(storyboard["frames"]) > 0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
