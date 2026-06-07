from scene_agent.config import Config


class TestPublicRun:
    """Smoke tests for the public run() helper."""

    def test_run_returns_completed_result(self, monkeypatch):
        from scene_agent import main as main_module
        from scene_agent.models import SceneRunResult

        def fake_flow(**kwargs):
            return SceneRunResult(
                run_id="run-1",
                status="completed",
                final_video_uri="file:///tmp/final.mp4",
                frame_uris=["frame-0"],
                segment_uris=["segment-0"],
                artifacts_dir="/tmp/run-1",
            )

        monkeypatch.setattr("scene_agent.prefect_flows.generate_scene_flow", fake_flow)

        result = main_module.run("brief", config=Config(openrouter_api_key="test-key"))
        assert result["status"] == "completed"
        assert result["run_id"] == "run-1"

    def test_run_returns_failed_result_from_manifest_when_flow_raises(self, monkeypatch, tmp_path):
        from scene_agent import main as main_module

        config = Config(openrouter_api_key="test-key", storage_path=str(tmp_path))
        run_id = "failed-run"
        run_dir = config.run_artifacts_dir(run_id)
        (run_dir / "manifest.json").write_text(
            """
            {
              "run_id": "failed-run",
              "status": "failed",
              "frame_uris": [],
              "segment_uris": [],
              "reviews": {},
              "provider_metadata": {},
              "sb_review_mode": "not_run",
              "vid_review_mode": "not_run",
              "error": "kling timeout",
              "error_code": "TransientProviderError"
            }
            """.strip(),
            encoding="utf-8",
        )

        def fake_flow(**kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr("scene_agent.prefect_flows.generate_scene_flow", fake_flow)

        result = main_module.run(
            "brief",
            config=config,
            run_options={"run_id": run_id},
        )
        assert result["status"] == "failed"
        assert result["error"] == "kling timeout"
