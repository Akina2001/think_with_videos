#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from tqdm import tqdm

from sft_builder.config import AppConfig
from sft_builder.pipeline import SFTBuilderPipeline
from sft_builder.text_utils import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build tool-augmented SFT data for think-with-videos.")
    parser.add_argument("--config", type=str, default="/mnt/volumes/base-vla-ali-sh-mix/sunqihao/0-workspace/think_with_videos_sft_builder/configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--input", type=str, default="/mnt/volumes/base-vla-ali-sh-mix/sunqihao/0-workspace/data/llava.jsonl", help="Path to input JSONL.")
    parser.add_argument("--output-dir", type=str, default="/mnt/volumes/base-vla-ali-sh-mix/sunqihao/0-workspace/outputs", help="Output directory.")
    parser.add_argument("--limit", type=int, default=10, help="Optional number of samples to process.")
    parser.add_argument("--start-index", type=int, default=0, help="Start index in the input JSONL.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AppConfig.from_yaml(args.config)
    pipeline = SFTBuilderPipeline(config=config, output_dir=args.output_dir)

    rows = read_jsonl(args.input)
    rows = rows[args.start_index :]
    if args.limit is not None:
        rows = rows[: args.limit]

    records = []
    failures = 0

    for row in tqdm(rows, desc="Building SFT records"):
        try:
            sample = pipeline.parse_raw_sample(row)
            record = pipeline.process_sample(sample)
            if record is not None:
                records.append(record)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            failures += 1
            sample_id = row.get("id", "unknown")
            print(f"[WARN] failed on sample {sample_id}: {exc}", file=sys.stderr)

    pipeline.export_records(records)
    print(f"Finished. records={len(records)}, failures={failures}, output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
