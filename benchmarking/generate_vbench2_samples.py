from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlparse

from dotenv import load_dotenv

BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parent
PACKAGE_ROOT = REPO_ROOT / "video_scene_agent"
SRC_ROOT = PACKAGE_ROOT / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scene_agent.config import Config
from scene_agent.main import run

DEFAULT_ARTIFACT_ROOT = BENCHMARK_DIR / "artifacts" / "fixed30_qwen36"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def selected_samples(manifest: dict, sample_ids: list[str] | None) -> list[dict]:
    samples = manifest.get("samples", [])
    if not sample_ids:
        return samples
    wanted = set(sample_ids)
    selected = [sample for sample in samples if sample["sample_id"] in wanted]
    missing = sorted(wanted - {sample["sample_id"] for sample in selected})
    if missing:
        raise ValueError(f"Unknown sample ids: {missing}")
    return selected


def local_path_from_uri(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if not parsed.scheme:
        return Path(uri)
    return None


def copy_video(uri: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = local_path_from_uri(uri)
    if source and source.exists():
        shutil.copy2(source, destination)
        return
    with urllib.request.urlopen(uri, timeout=120) as response:
        destination.write_bytes(response.read())


def upsert_video_row(manifest: dict, row: dict) -> None:
    rows = list(manifest.get("videos", []))
    key = (row["model_key"], row["sample_id"])
    for index, existing in enumerate(rows):
        if (existing.get("model_key"), existing.get("sample_id")) == key:
            rows[index] = row
            manifest["videos"] = rows
            return
    rows.append(row)
    manifest["videos"] = rows


def ensure_model_row(manifest: dict, model_key: str, model: str, model_slug: str) -> None:
    rows = list(manifest.get("models", []))
    if any(row.get("model_key") == model_key for row in rows):
        return
    rows.append({"model_key": model_key, "model": model, "model_slug": model_slug})
    manifest["models"] = rows


def generate(args: argparse.Namespace) -> None:
    if not args.allow_paid:
        raise SystemExit("Refusing live generation without --allow-paid")
    load_dotenv(PACKAGE_ROOT / ".env")
    config = Config.from_env()
    manifest_path = args.artifact_root / "manifest.json"
    manifest = read_json(manifest_path)
    ensure_model_row(manifest, args.model_key, args.model_label, args.model_slug)
    samples = selected_samples(manifest, args.sample_id)
    constraints = {
        "duration_sec": args.duration,
        "target_duration_sec": args.duration,
        "fps": args.fps,
        "aspect_ratio": args.aspect_ratio,
        "num_keyframes": args.num_keyframes,
        "K_sb": args.storyboard_iterations,
        "K_vid": args.video_iterations,
    }

    for sample in samples:
        sample_id = sample["sample_id"]
        output_rel = Path("videos") / args.model_slug / f"{sample_id}.mp4"
        output_path = args.artifact_root / output_rel
        if output_path.exists() and not args.force:
            print(f"skip {sample_id}: {output_path}")
            continue
        run_id = f"{args.run_id_prefix}-{sample_id}"
        result = run(
            sample["prompt"],
            constraints=constraints,
            config=config,
            run_options={"run_id": run_id},
        )
        if result.get("status") != "completed" or not result.get("final_video_uri"):
            raise RuntimeError(f"Generation failed for {sample_id}: {result}")
        copy_video(str(result["final_video_uri"]), output_path)
        upsert_video_row(
            manifest,
            {
                "model": args.model_label,
                "model_key": args.model_key,
                "model_slug": args.model_slug,
                "sample_id": sample_id,
                "dimension": sample["dimension"],
                "source_index": sample["source_index"],
                "video_path": output_rel.as_posix(),
                "source_video_path": result.get("final_video_uri"),
                "link_mode": "copy",
                "hf_path": None,
                "selected_variant": None,
            },
        )
        write_json(manifest_path, manifest)
        print(f"generated {sample_id}: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate VBench2 sample videos through the Prefect pipeline")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--aspect-ratio", default="16:9")
    parser.add_argument("--num-keyframes", type=int, default=3)
    parser.add_argument("--storyboard-iterations", type=int, default=3)
    parser.add_argument("--video-iterations", type=int, default=2)
    parser.add_argument("--run-id-prefix", default="vbench2")
    parser.add_argument("--model-key", default="our")
    parser.add_argument("--model-label", default="Our pipeline")
    parser.add_argument("--model-slug", default="our_pipeline")
    return parser


def main(argv: list[str] | None = None) -> None:
    generate(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
