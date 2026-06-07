# Ultimate Video Gen

This repository contains two related pieces:

- `video_scene_agent/`: the production video scene agent.
- `benchmarking/`: a fixed sample benchmark harness and committed benchmark artifacts.
- `artefact_sample/`: one complete sample generation run with world, storyboard,
  keyframes, segment video, final video, reviews, and manifest.

## Video Scene Agent

`video_scene_agent` is a Prefect-based Python package that turns a text brief into a short stitched video. The flow plans a scene, generates keyframes with OpenRouter image models, generates or repairs video segments with Kling, stitches segments with FFmpeg, and runs multimodal review through OpenRouter.

The main package code lives in `video_scene_agent/src/scene_agent/`. Generated runtime outputs are written under `video_scene_agent/artifacts/`, which is ignored by git.

Basic local setup:

```bash
cd video_scene_agent
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Live generation requires `video_scene_agent/.env` with:

```bash
OPENROUTER_API_KEY=...
KLING_ACCESS_KEY=...
KLING_SECRET_KEY=...
STORAGE_PATH=./artifacts
```

Create it from the package template:

```bash
cp video_scene_agent/.env.example video_scene_agent/.env
```

All live commands in this repository use that same file. The benchmark scripts
also load secrets from `video_scene_agent/.env`; there is no separate
`benchmarking/.env`.

Run the CLI:

```bash
cd video_scene_agent
PYTHONPATH=src python -m scene_agent.main "A short cinematic scene..."
```

Run tests:

```bash
cd video_scene_agent
PYTHONPATH=src pytest
```

## Benchmarking

`benchmarking/` contains a small VBench2-like API-judge benchmark over a fixed 30-video sample set: 15 `Complex_Plot` prompts and 15 `Multi-View_Consistency` prompts. The committed artifact bundle includes videos for the current pipeline and comparison models, plus Qwen 3.6 Flash judge outputs.

The benchmark is intentionally not the official GPU VBench2 implementation. It uses an LLM-as-a-judge flow through OpenRouter video review, with the judge model fixed to `qwen/qwen3.6-flash`.

See `benchmarking/README.md` for the artifact layout and commands.

## Artefact Sample

`artefact_sample/` contains one compact complete run copied from the agent's
runtime artifacts. It can be used to inspect the generated world package,
storyboard, human-readable storyboard markdown, keyframe anchors, generated
segment, stitched final video, review outputs, report, and manifest without
running live generation.

## Отчет об использовании генеративного ИИ

При подготовке данной работы в период с февраля по июнь 2026 года использовались генеративные модели и инструменты искусственного интеллекта: Claude Opus 4.5, Claude Opus 4.6, GPT-5.5 и агенты Claude Code и Codex.

ИИ применялся при разработке программной части работы, в том числе при создании мультиагентной системы и переносе написанного авторами кода с фреймворка LangGraph на инфраструктуру Prefect. Также ИИ использовался при портировании кода VBench-2.0 для работы с API OpenRouter, а также при адаптации процедуры оценки: вместо полного запуска бенчмарка была реализована оценка на подвыборке запросов.

Формат использования ИИ в программной части работы: генерация и модификация кода с последующей человеческой оценкой, проверкой, редактированием и интеграцией в итоговую систему.

Кроме того, генеративный ИИ использовался при подготовке текстовой части отчёта: для генерации черновых вариантов аннотации и заключения с последующим редактированием авторами, форматирования списка источников, подготовки предварительных идей для отдельных разделов документа, создания и уточнения визуальных материалов, включая блок-схемы и таблицы, а также для исправления орфографических, грамматических и стилистических ошибок. ИИ также применялся для подготовки проектной документации и Markdown-файлов, включая README-файлы и описания процедур запуска, настройки и оценки системы.

ИИ также использовался при подготовке настоящего раздела об использовании генеративных моделей — для корректной формулировки описания применения ИИ в соответствии с правилами НИУ ВШЭ. Итоговая редакция данного раздела была проверена и утверждена авторами.

Все сгенерированные материалы были проверены, отредактированы и при необходимости переработаны авторами. ИИ не использовался как самостоятельный источник научных данных или неподтверждённых фактических утверждений. Итоговые решения, структура работы, интерпретация результатов, выводы и ответственность за содержание работы принадлежат авторам.
