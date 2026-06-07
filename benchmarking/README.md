# Бенчмаркинг

Эта папка содержит фиксированный sample benchmark для сравнения текущего видеопайплайна с внешними моделями.

## Что входит в набор

Корневая папка артефактов по умолчанию:

```bash
benchmarking/artifacts/fixed30_qwen36
```

В ней лежат:

- `manifest.json`: выбранные prompts, список моделей и пути к видео.
- `videos/`: видео, оценённые для каждой модели.
- `scores/results.json`: per-sample оценки судьи Qwen 3.6 Flash.
- `results.md`: сгенерированная итоговая таблица.

Папка `videos/` — основной каталог с примерами генерации. В текущем зафиксированном наборе там 120 MP4-файлов: по 30 видео для нашего пайплайна, Veo3, Wanx и Vidu_Q1.

Фиксированная выборка содержит 30 prompts:

- 15 из `Complex_Plot`
- 15 из `Multi-View_Consistency`

Сейчас в наборе зафиксированы модели:

- `our`: видео, сгенерированные Prefect-пайплайном этого репозитория.
- `veo3`: sampled VBench2 videos из датасета Hugging Face.
- `wanx`: sampled VBench2 videos из датасета Hugging Face.
- `vidu_q1`: sampled VBench2 videos из датасета Hugging Face.

## Оценка

`run_sample_bench.py` оценивает видео через OpenRouter video review с моделью:

```text
qwen/qwen3.6-flash
```

Для каждого видео сохраняются:

- `overall_score`
- `prompt_adherence`
- оценки и объяснения по отдельным критериям
- transport metadata для способа передачи видео судье

Это VBench2-like API-proxy benchmark. Он полезен для regression tracking и сравнения моделей на фиксированной выборке, но не является официальной GPU-оценкой VBench2.

## Конфигурация окружения

Скрипты бенчмарка используют ту же runtime-конфигурацию, что и агент. Секреты нужно задавать в:

```text
video_scene_agent/.env
```

Файл создаётся из шаблона пакета:

```bash
cp video_scene_agent/.env.example video_scene_agent/.env
```

Для оценки уже существующих benchmark-видео нужны минимум:

```bash
OPENROUTER_API_KEY=...
STORAGE_PATH=./artifacts
```

Для перегенерации benchmark-видео текущим пайплайном дополнительно нужны:

```bash
KLING_ACCESS_KEY=...
KLING_SECRET_KEY=...
```

У папки `benchmarking/` нет собственного `.env`. Локальные секреты должны храниться только в `video_scene_agent/.env`; этот файл игнорируется git.

## Команды

Напечатать или обновить итоговую таблицу по уже сохранённым оценкам:

```bash
python3 benchmarking/run_sample_bench.py report
```

Оценить недостающие или принудительно выбранные samples через Qwen 3.6 Flash:

```bash
python3 benchmarking/run_sample_bench.py judge --model all --allow-paid
```

Оценить одну модель:

```bash
python3 benchmarking/run_sample_bench.py judge --model our --allow-paid
```

Оценить один sample:

```bash
python3 benchmarking/run_sample_bench.py judge --model our --sample-id complex_plot_000 --allow-paid
```

Перегенерировать видео текущего пайплайна для выбранных samples через Prefect flow:

```bash
python3 benchmarking/generate_vbench2_samples.py --sample-id complex_plot_000 --allow-paid
```

Сгенерировать все samples через Prefect flow:

```bash
python3 benchmarking/generate_vbench2_samples.py --allow-paid
```

Live-оценка и live-генерация читают секреты из `video_scene_agent/.env`. Этот файл нельзя коммитить.

## Восстановление после сбоев

Бенчмарк записывает результаты инкрементально после каждого оценённого sample. Если оценка упала в середине, повторите ту же команду без `--force`; уже завершённые строки будут пропущены.

Генерация тоже сразу копирует каждое завершённое видео в папку артефактов и обновляет `manifest.json` после каждого sample. Поэтому повторный запуск может продолжить работу с уже сохранённых видео.
