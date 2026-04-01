# Think-with-Videos SFT Builder

A simple and practical project for constructing **tool-augmented SFT data** from raw silent-video QA / temporal-grounding samples using **Qwen3-VL-235B-A22B-Instruct** through an OpenAI-compatible HTTP endpoint.

This project is designed for the setting discussed in the paper-style analysis:
- most raw data only has **QA**
- a subset also has **temporal grounding intervals**
- no audio is used
- the model should **not rely only on its own next-turn timestamp proposal**
- for medium / long videos, the builder first creates an **external temporal memory index**, then retrieves candidate intervals, verifies them, and finally writes a **tool-use SFT trajectory**

## What this project builds

For each raw sample, the pipeline produces:

1. **Global overview** of the full silent video
2. **Temporal memory index** made of coarse video segments + segment captions
3. **Retrieved candidate intervals**
4. **Verified evidence clips**
5. **Route label**
   - `direct_answer`
   - `direct_localize`
   - `boundary_refine`
   - `memory_retrieve`
   - `reflection_repair`
   - `multi_hop_compose`
6. **Tool-augmented SFT record**
   - structured JSON
   - ShareGPT-style conversation
   - flattened assistant trace

## Core idea

Instead of asking the model to directly guess the next timestamp from the full video, this builder uses a simple external memory workflow:

**overview -> coarse temporal memory -> retrieve candidates -> verify local clips -> generate faithful trajectory**

This makes the produced traces much more suitable for long videos and sparse evidence.

## Directory structure

```text
think_with_videos_sft_builder/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── scripts/
│   └── build_sft.py
└── sft_builder/
    ├── __init__.py
    ├── config.py
    ├── exporters.py
    ├── llm_client.py
    ├── pipeline.py
    ├── prompt_templates.py
    ├── retrieval.py
    ├── routing.py
    ├── schema.py
    ├── text_utils.py
    └── video_utils.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You also need:

- `ffmpeg`
- `ffprobe`

## API configuration

Set your API key through an environment variable:

```bash
export QWEN_API_KEY="YOUR_API_KEY"
```

Then edit `configs/default.yaml`:

```yaml
api:
  chat_completions_url: "https://YOUR_HOST/v1/chat/completions"
  api_key_env: "QWEN_API_KEY"
  model: "qwen__qwen3-vl-235b-a22b-instruct"
```

This same code also works with a local vLLM OpenAI-compatible endpoint such as:

```yaml
api:
  chat_completions_url: "http://localhost:8000/v1/chat/completions"
  api_key_env: "QWEN_API_KEY"
  model: "qwen__qwen3-vl-235b-a22b-instruct"
```

## Input format

Each line in the input JSONL should look like this:

```json
{"id": "000000", "dataset": "llava-video-178k", "task_type": "qa", "video_path": "path/to/video.mp4", "question": "Why does the text 'I HAVE RESPECT FOR YOU' appear on the screen?\nA. Because ...\nB. Because ...\nC. Because ...\nD. Because ...\nPlease respond with only the letter of the correct answer.", "answer": "C."}
```

For temporal-grounding samples, one or more `start_time` / `end_time` values may also be present:

```json
{"id": "000001", "dataset": "custom", "task_type": "tg", "video_path": "path/to/video.mp4", "question": "...", "answer": "B.", "start_time": [12.0, 43.5], "end_time": [18.0, 48.0]}
```

Supported interval formats:
- scalar `start_time` + scalar `end_time`
- list `start_time` + list `end_time`

## Run

```bash
python scripts/build_sft.py \
  --config configs/default.yaml \
  --input /path/to/raw_samples.jsonl \
  --output-dir /path/to/output_dir
```

Optional:

```bash
python scripts/build_sft.py \
  --config configs/default.yaml \
  --input /path/to/raw_samples.jsonl \
  --output-dir /path/to/output_dir \
  --limit 100 \
  --start-index 0
```

## Output files

After running, the output directory contains:

```text
output_dir/
├── cache/
│   ├── llm/
│   └── video_index/
├── proxies/
├── intermediate/
│   ├── video_indexes/
│   └── sample_records/
├── final/
│   ├── sft_records.jsonl
│   ├── sharegpt_records.jsonl
│   ├── flattened_traces.jsonl
│   └── stats.json
└── logs/
```

## Design choices

### 1. English-only prompts
All prompts are written in English and designed to:
- avoid audio assumptions
- reduce hallucination
- force strict JSON
- separate **memory retrieval**, **clip verification**, and **trajectory writing**

### 2. Lightweight video proxies
The pipeline never sends the raw original video directly when it does not need to.
It first creates smaller proxy clips using `ffmpeg`:
- low-frame-rate full-video overview
- low-cost coarse memory clips
- higher-detail fine clips for verification

This is critical for large videos.

### 3. Reusable video index
If multiple questions use the same video, the builder reuses:
- duration
- global overview
- coarse temporal memory entries

This saves a large amount of compute.

### 4. Simple tool space
The generated traces use only two explicit tools:
- `retrieve_memory(query, top_k)`
- `crop_video(start_time, end_time)`

This keeps the training target simple while still capturing the intended behavior.

## Notes on training format

The project exports:
- a structured record for inspection
- a ShareGPT-style multi-turn conversation
- a flattened assistant trace

Different training frameworks prefer different formats, so all three are provided.

## Recommended first run

Before processing a large dataset, run on 20-50 samples and manually inspect:

- `final/sft_records.jsonl`
- `intermediate/sample_records/*.json`

Pay special attention to:
- whether evidence intervals are visually correct
- whether `reflection_repair` is actually useful
- whether `boundary_refine` traces are faithful
- whether the final answer exactly matches the gold answer

## Practical advice

For a first stable version:
- keep `top_k_retrieve` small (6-8)
- keep `coarse_target_segments` around `10-14`
- keep `fine_max_per_coarse` around `3-4`
- inspect failures before scaling

If the endpoint is expensive, the biggest savings usually come from:
- aggressive caching
- reusing video indexes
- reducing coarse segment count
- lowering proxy resolution
