from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class APIConfig:
    chat_completions_url: str
    api_key: str
    model: str
    timeout: int = 600
    temperature: float = 0.2
    top_p: float = 0.7
    max_tokens: int = 2048
    default_mm_processor_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    short_video_sec: float = 90.0
    long_video_sec: float = 900.0

    overview_target_frames: int = 96
    overview_width: int = 384
    overview_crf: int = 32

    coarse_target_segments: int = 12
    coarse_min_sec: float = 12.0
    coarse_max_sec: float = 150.0
    coarse_target_frames: int = 24
    coarse_width: int = 384
    coarse_crf: int = 32

    fine_target_sec: float = 12.0
    fine_min_sec: float = 4.0
    fine_max_sec: float = 20.0
    fine_target_frames: int = 32
    fine_width: int = 448
    fine_crf: int = 30
    fine_max_per_coarse: int = 4

    top_k_retrieve: int = 8
    top_k_rerank: int = 5
    support_confidence_threshold: float = 0.65
    partial_confidence_threshold: float = 0.55
    overview_answerable_confidence: float = 0.85
    merge_gap_sec: float = 2.0
    reflection_probability: float = 0.4

    keep_invalid_records: bool = False
    save_intermediate_json: bool = True


@dataclass
class OutputConfig:
    export_sharegpt: bool = True
    export_flattened_text: bool = True


@dataclass
class AppConfig:
    api: APIConfig
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        api = APIConfig(**raw["api"])
        pipeline = PipelineConfig(**raw.get("pipeline", {}))
        output = OutputConfig(**raw.get("output", {}))
        return cls(api=api, pipeline=pipeline, output=output)
