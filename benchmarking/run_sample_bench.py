from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

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
from scene_agent.tools.openrouter_video_review import OpenRouterVideoReviewTool
from scene_agent.tools.storage import LocalStorageBackend
from scene_agent.tools.tmpfiles import TmpfilesMediaPublisher
from scene_agent.utils.json_llm import clean_json_response, parse_partial_json

JUDGE_MODEL = "qwen/qwen3.6-flash"
DEFAULT_ARTIFACT_ROOT = BENCHMARK_DIR / "artifacts" / "fixed30_qwen36"
SUPPORTED_DIMENSIONS = {"Complex_Plot", "Multi-View_Consistency"}

JUDGE_SYSTEM = """You are an exacting video benchmark judge.

Inspect the supplied video media, compare it to the prompt, and return only strict JSON.
Scores are 0 to 10, where 10 means the video fully satisfies the criterion.
This is an API-proxy VBench2-like evaluation, not the official GPU benchmark.
"""


@dataclass(frozen=True)
class EvalSample:
    sample_id: str
    dimension: str
    source_index: int
    prompt: str


@dataclass(frozen=True)
class CriterionScore:
    name: str
    score: float
    rationale: str


@dataclass(frozen=True)
class JudgeScore:
    prompt_adherence: float
    overall_score: float
    criteria: list[CriterionScore]
    blocking_failures: list[str]
    rationale: str


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_manifest(artifact_root: Path) -> dict[str, Any]:
    path = artifact_root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Benchmark manifest not found: {path}")
    manifest = read_json(path)
    dimensions = {sample.get("dimension") for sample in manifest.get("samples", [])}
    unsupported = sorted(dimensions - SUPPORTED_DIMENSIONS)
    if unsupported:
        raise ValueError(f"Unsupported dimensions in manifest: {unsupported}")
    return manifest


def load_results(artifact_root: Path) -> dict[str, Any]:
    path = artifact_root / "scores" / "results.json"
    if path.exists():
        return read_json(path)
    return {
        "judge_model": JUDGE_MODEL,
        "sample_manifest": "manifest.json",
        "per_sample_results": [],
        "summary_rows": [],
    }


def sample_rows(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): row for row in manifest.get("samples", [])}


def video_rows(manifest: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["model_key"]), str(row["sample_id"])): row for row in manifest.get("videos", [])}


def model_rows(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["model_key"]): row for row in manifest.get("models", [])}


def selected_models(manifest: dict[str, Any], requested: list[str] | None) -> list[str]:
    models = model_rows(manifest)
    if not requested or "all" in requested:
        return list(models)
    selected: list[str] = []
    for key in requested:
        if key not in models:
            raise ValueError(f"Unsupported model {key!r}; choose from all, {', '.join(models)}")
        if key not in selected:
            selected.append(key)
    return selected


def score_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["model"]), str(row["sample_id"])


def score_value(payload: dict[str, Any], key: str) -> float:
    value = float(payload.get(key, 0.0))
    if value < 0.0 or value > 10.0:
        raise ValueError(f"Judge score {key} out of range: {value}")
    return value


def build_judge_user_prompt(sample: EvalSample) -> str:
    if sample.dimension == "Complex_Plot":
        criteria = [
            "plot_coverage: the major events from the prompt appear on screen",
            "action_order: events happen in the requested temporal order",
            "causal_coherence: visual transitions make sense as a story",
            "temporal_continuity: subjects and scene state do not reset incoherently",
            "prompt_adherence: the video follows the exact prompt instead of a generic scene",
        ]
    elif sample.dimension == "Multi-View_Consistency":
        criteria = [
            "camera_orbit: the camera visibly changes viewpoint around the subject",
            "identity_preservation: the same object/place remains recognizable",
            "geometry_consistency: shape, layout, and perspective stay plausible",
            "background_continuity: surroundings do not jump or transform randomly",
            "prompt_adherence: the requested subject and orbit behavior are present",
        ]
    else:
        raise ValueError(f"Unsupported judge dimension: {sample.dimension}")

    return f"""Evaluate this generated video for VBench2 dimension `{sample.dimension}`.

Prompt:
{sample.prompt}

Criteria:
{chr(10).join(f"- {item}" for item in criteria)}

Return only JSON with this shape:
{{
  "mode": "api_proxy",
  "dimension": "{sample.dimension}",
  "sample_id": "{sample.sample_id}",
  "prompt_adherence": 0,
  "overall_score": 0,
  "criteria": [
    {{"name": "criterion_name", "score": 0, "rationale": "short visual evidence"}}
  ],
  "blocking_failures": ["short failure strings, or empty list"],
  "rationale": "short overall judgment"
}}
"""


def parse_score(raw: str, sample: EvalSample) -> JudgeScore:
    parsed = parse_partial_json(clean_json_response(raw))
    if not isinstance(parsed, dict):
        raise ValueError("Judge response did not contain a JSON object")
    criteria = []
    for item in parsed.get("criteria") or []:
        if isinstance(item, dict):
            criteria.append(
                CriterionScore(
                    name=str(item.get("name") or ""),
                    score=score_value(item, "score"),
                    rationale=str(item.get("rationale") or ""),
                )
            )
    blocking = parsed.get("blocking_failures") or []
    if not isinstance(blocking, list):
        blocking = [str(blocking)]
    return JudgeScore(
        prompt_adherence=score_value(parsed, "prompt_adherence"),
        overall_score=score_value(parsed, "overall_score"),
        criteria=criteria,
        blocking_failures=[str(item) for item in blocking],
        rationale=str(parsed.get("rationale") or ""),
    )


def judge_video(config: Config, artifact_root: Path, model_slug: str, sample: EvalSample, video_path: Path) -> tuple[JudgeScore, str]:
    storage = LocalStorageBackend(base_path=artifact_root)
    review_tool = OpenRouterVideoReviewTool(config, storage)
    publisher = TmpfilesMediaPublisher(config, storage)
    user_text = build_judge_user_prompt(sample)
    local_uri = f"file://{video_path.resolve()}"
    request_base = {
        "model": JUDGE_MODEL,
        "temperature": 0.0,
        "system": JUDGE_SYSTEM,
        "user_text": user_text,
    }
    judge_dir = artifact_root / "judge" / model_slug
    last_error: Exception | None = None
    for attempt in range(1, 4):
        for transport, uri in [("local_base64", local_uri), ("tmpfiles_url", None)]:
            stem = f"{sample.sample_id}.{attempt}.{transport}"
            try:
                video_uri = uri or publisher.publish(local_uri, expected_kind="video")
                write_json(judge_dir / f"{stem}.request.json", {**request_base, "video_uri": video_uri, "transport": transport})
                raw = review_tool.review(JUDGE_SYSTEM, user_text, video_uri)
                write_json(judge_dir / f"{stem}.response.json", {"raw_response": raw})
                return parse_score(raw, sample), transport
            except Exception as exc:
                last_error = exc
                write_json(judge_dir / f"{stem}.error.json", {"error": str(exc)})
        time.sleep(2 * attempt)
    raise RuntimeError(str(last_error))


def upsert_score(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    key = score_key(row)
    for index, existing in enumerate(rows):
        if score_key(existing) == key:
            rows[index] = row
            return
    rows.append(row)


def summary_rows(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for model in manifest.get("models", []):
        label = str(model["model"])
        model_scores = [row for row in rows if row.get("model") == label and row.get("status", "judged") == "judged"]
        if not model_scores:
            continue

        def avg(dimension: str, key: str) -> float | None:
            values = [float(row[key]) for row in model_scores if row.get("dimension") == dimension]
            return round(statistics.mean(values), 4) if values else None

        summary.append(
            {
                "model": label,
                "overall": round(statistics.mean(float(row["overall_score"]) for row in model_scores), 4),
                "prompt": round(statistics.mean(float(row["prompt_adherence"]) for row in model_scores), 4),
                "complex_overall": avg("Complex_Plot", "overall_score"),
                "complex_prompt": avg("Complex_Plot", "prompt_adherence"),
                "multiview_overall": avg("Multi-View_Consistency", "overall_score"),
                "multiview_prompt": avg("Multi-View_Consistency", "prompt_adherence"),
            }
        )
    return summary


def write_report(artifact_root: Path, results: dict[str, Any]) -> None:
    lines = [
        "# Fixed30 Qwen36 Benchmark",
        "",
        f"Judge model: `{JUDGE_MODEL}`",
        "",
        "| Model | Total overall | Total prompt | Complex overall | Complex prompt | Multi-view overall | Multi-view prompt |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results.get("summary_rows", []):
        lines.append(
            f"| {row['model']} | {row['overall']} | {row['prompt']} | "
            f"{row['complex_overall']} | {row['complex_prompt']} | "
            f"{row['multiview_overall']} | {row['multiview_prompt']} |"
        )
    (artifact_root / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_results(artifact_root: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "judge_model": JUDGE_MODEL,
        "sample_manifest": "manifest.json",
        "per_sample_results": rows,
        "summary_rows": summary_rows(manifest, rows),
    }
    write_json(artifact_root / "scores" / "results.json", payload)
    write_report(artifact_root, payload)
    return payload


def print_table(rows: list[dict[str, Any]]) -> None:
    print("| Model | Total overall | Total prompt | Complex overall | Complex prompt | Multi-view overall | Multi-view prompt |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['model']} | {row['overall']} | {row['prompt']} | "
            f"{row['complex_overall']} | {row['complex_prompt']} | "
            f"{row['multiview_overall']} | {row['multiview_prompt']} |"
        )


def cmd_judge(args: argparse.Namespace) -> None:
    if not args.allow_paid:
        raise SystemExit("Refusing paid Qwen judging without --allow-paid")
    load_dotenv(PACKAGE_ROOT / ".env")
    config = replace(Config.from_env(), openrouter_video_model=JUDGE_MODEL, openrouter_temperature=0.0)
    manifest = load_manifest(args.artifact_root)
    samples = sample_rows(manifest)
    videos = video_rows(manifest)
    models = model_rows(manifest)
    sample_ids = set(args.sample_id or samples.keys())
    selected = selected_models(manifest, args.model)
    results = load_results(args.artifact_root)
    rows = list(results.get("per_sample_results", []))
    existing = {score_key(row): row for row in rows if row.get("status", "judged") == "judged"}

    for model_key in selected:
        model = models[model_key]
        label = str(model["model"])
        slug = str(model["model_slug"])
        for sample_id in sorted(sample_ids):
            sample_row = samples[sample_id]
            if (label, sample_id) in existing and not args.force:
                continue
            video_row = videos.get((model_key, sample_id))
            if not video_row:
                raise FileNotFoundError(f"Missing video for {model_key}/{sample_id}")
            video_path = args.artifact_root / str(video_row["video_path"])
            sample = EvalSample(
                sample_id=sample_id,
                dimension=str(sample_row["dimension"]),
                source_index=int(sample_row["source_index"]),
                prompt=str(sample_row["prompt"]),
            )
            score, transport = judge_video(config, args.artifact_root, slug, sample, video_path)
            row = {
                "model": label,
                "model_key": model_key,
                "sample_id": sample_id,
                "dimension": sample.dimension,
                "source_index": sample.source_index,
                "prompt": sample.prompt,
                "status": "judged",
                "overall_score": score.overall_score,
                "prompt_adherence": score.prompt_adherence,
                "criteria": [criterion.__dict__ for criterion in score.criteria],
                "rationale": score.rationale,
                "blocking_failures": score.blocking_failures,
                "judge_transport": transport,
                "video_path": video_row["video_path"],
                "hf_path": video_row.get("hf_path"),
                "selected_variant": video_row.get("selected_variant"),
            }
            upsert_score(rows, row)
            existing[(label, sample_id)] = row
            save_results(args.artifact_root, manifest, rows)
            print(f"judged {label} {sample_id}: {score.overall_score} / {score.prompt_adherence} ({transport})")
    print_table(save_results(args.artifact_root, manifest, rows)["summary_rows"])


def cmd_report(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.artifact_root)
    results = load_results(args.artifact_root)
    results["summary_rows"] = summary_rows(manifest, results.get("per_sample_results", []))
    write_json(args.artifact_root / "scores" / "results.json", results)
    write_report(args.artifact_root, results)
    print_table(results["summary_rows"])
    print(args.artifact_root / "results.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Judge/report the fixed VBench2 sample bundle with Qwen 3.6 Flash")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    judge = sub.add_parser("judge")
    judge.add_argument("--model", action="append", help="Model key from manifest, or all")
    judge.add_argument("--sample-id", action="append")
    judge.add_argument("--force", action="store_true")
    judge.add_argument("--allow-paid", action="store_true")
    judge.set_defaults(func=cmd_judge)

    report = sub.add_parser("report")
    report.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
