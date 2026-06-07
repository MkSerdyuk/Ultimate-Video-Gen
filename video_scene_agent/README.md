# Агент генерации видео

Генератор коротких видеосцен на Prefect с OpenRouter, прямым API Kling 3.0 и FFmpeg.

## Обзор

Текущий runtime выполняет пайплайн сцены как Prefect flow. Артефакты каждого запуска сохраняются отдельно в `artifacts/runs/<run_id>/`.

Основные подсистемы:

- `Director`: генерация мира и storyboard.
- `Storyboard Editor`: цикл проверки и исправления storyboard.
- `Operator`: генерация видеосегментов и склейка.
- `Video Editor`: цикл проверки и исправления итогового видео.

`scene_agent.main.run()` и `python -m scene_agent` напрямую запускают Prefect flow.

## Возможности

- **Текст и review**: OpenRouter.
- **Генерация изображений**: `google/gemini-2.5-flash-image` через OpenRouter.
- **Генерация и ремонт видео**: Kling 3.0 Omni Standard через прямой Kling API.
- **Склейка видео**: FFmpeg.
- **Артефакты**: локальное файловое хранилище и `manifest.json` для каждого запуска.
- **Возобновление запуска**: состояние Prefect flow и manifest-файл в папке запуска.

## Установка

```bash
cd video_scene_agent
python3 -m pip install -e ".[dev]"
```

### Системные требования

- Python 3.9+
- FFmpeg

```bash
brew install ffmpeg
# или
sudo apt install ffmpeg
```

## Конфигурация

Runtime-конфигурация хранится в `video_scene_agent/.env`. Если вы работаете из папки пакета, создайте файл из шаблона:

```bash
cp .env.example .env
```

Обязательные переменные для live-генерации:

```bash
OPENROUTER_API_KEY=...
KLING_ACCESS_KEY=...
KLING_SECRET_KEY=...
STORAGE_PATH=./artifacts
```

Опциональные настройки Kling:

```bash
KLING_API_BASE=https://api-singapore.klingai.com
KLING_VIDEO_MODEL=kling-v3-omni
KLING_MODE=std
KLING_SOUND=off
KLING_POLL_TIMEOUT_SEC=900
KLING_POLL_REQUEST_TIMEOUT_SEC=30
KLING_POLL_INTERVAL_SEC=2.0
KLING_RUN_TOKEN_LIMIT=60
KLING_GENERATION_TOKENS_PER_SECOND=0.6
KLING_EDIT_TOKENS_PER_SECOND=0.9
KLING_USE_TMPFILES=1
KLING_TMPFILES_UPLOAD_URL=https://tmpfiles.org/api/v1/upload
KLING_TMPFILES_TTL_SEC=172800
KLING_TMPFILES_MAX_BYTES=100000000
KLING_TMPFILES_TIMEOUT_SEC=120
```

Опциональные настройки Prefect для self-hosted-развёртывания:

```bash
PREFECT_API_URL=http://127.0.0.1:4200/api
PREFECT_WORK_POOL=video-scene-agent
PREFECT_DOCKER_IMAGE=video-scene-agent:latest
STORAGE_PUBLIC_URL_BASE=https://example.com/artifacts
PREFECT_SERVER_API_AUTH_STRING=admin:replace-with-long-random-password
PREFECT_API_AUTH_STRING=admin:replace-with-long-random-password
```

Входные медиа для Kling по умолчанию публикуются через tmpfiles. Локальные keyframes и исходные сегменты остаются в `artifacts/runs/...`; во внешний сервис отправляются только временные копии как прямые `https://tmpfiles.org/dl/...` URL для Kling.

`STORAGE_PUBLIC_URL_BASE` и `KLING_MEDIA_PUBLIC_URL_BASE` теперь опциональны. Они нужны для публичных ссылок на артефакты или для случая `KLING_USE_TMPFILES=0`, когда вы предоставляете собственный HTTPS-хостинг медиа, доступный внешнему провайдеру.

`KLING_RUN_TOKEN_LIMIT` — жёсткий лимит на один запуск в ресурсных единицах Kling. Когда лимит достигнут, runtime прекращает платные вызовы Kling, сохраняет уже существующие сегменты для пропущенных edit-операций и создаёт локальные FFmpeg still-клипы для недостающих сегментов, чтобы итоговое видео всё равно можно было склеить и вернуть.

Опорные кадры, image-to-video-сегменты и feature-guided repair получают одно и то же нормализованное значение `constraints.aspect_ratio`. Неизвестные aspect ratio везде одинаково приводятся к `16:9`.

## Использование

### CLI

```bash
python3 -m scene_agent "Ночная сцена: герой идет по дождливой улице..."
```

### API Python

```python
from scene_agent.main import run

result = run(
    user_brief="Красивый закат над океаном, камера медленно наезжает",
    constraints={
        "aspect_ratio": "16:9",
        "duration_sec": 5.0,
    },
)
```

Возвращаемый словарь включает:

- `run_id`
- `status`
- `artifacts_dir`
- `final_video_uri`
- `storyboard`
- `world`
- `frame_uris`
- `segment_uris`
- `reviews`

## Структура runtime-артефактов

Сгенерированные файлы сохраняются в:

```text
artifacts/
  runs/
    <run_id>/
      manifest.json
      world.json
      storyboard.json
      reviews/
      storyboards/
      *.png / *.mp4
```

## Self-hosted-развёртывание Prefect

Целевая стабильная схема развёртывания:

- `postgres`
- `redis`
- `prefect-server`
- `prefect-services`
- `prefect-worker`

Для лёгких локальных экспериментов Prefect может запускаться с SQLite. Для стабильного production/self-hosted-использования нужно использовать PostgreSQL и Redis.

## Тесты

```bash
PYTHONPATH=src python3 -m pytest -q
python3 -m compileall src
```

## VBench2 API-proxy-бенчмарк

Код бенчмарка вынесен в корневую папку `benchmarking/`. Закоммиченный набор артефактов содержит финальную фиксированную выборку из 30 примеров. Оценки — это API-proxy LLM-as-a-judge через `qwen/qwen3.6-flash`, а не официальные GPU-оценки из leaderboard VBench2.

```bash
cd ..
python3 benchmarking/run_sample_bench.py report
python3 benchmarking/run_sample_bench.py judge --model our --sample-id complex_plot_000 --allow-paid
python3 benchmarking/generate_vbench2_samples.py --sample-id complex_plot_000 --allow-paid
```

## Минимальный live smoke

Smoke-скрипт с защитой от случайного платного запуска проверяет прямую генерацию Kling 3.0 Standard по начальному и конечному кадру, а также segment edit в низкостоимостном режиме:

```bash
python3 scripts/live_smoke_kling_minimal.py --dry-run --duration 4
RUN_LIVE_SMOKE=1 python3 scripts/live_smoke_kling_minimal.py --duration 4
```

Для end-to-end Prefect smoke используйте маленький запуск с `duration_sec=3.0`, `fps=12`, `num_keyframes=2`, `K_sb=3`, `K_vid=2` и `run_options={"force_edit_segments": [0]}`. Успешное публичное развёртывание показывает в Prefect UI link-артефакт итогового видео и image-артефакт постера итогового видео. Те же URL должны требовать авторизацию вне уже аутентифицированной браузерной сессии.

```bash
python3 scripts/live_smoke_prefect_minimal.py --dry-run
RUN_LIVE_SMOKE=1 python3 scripts/live_smoke_prefect_minimal.py
```
