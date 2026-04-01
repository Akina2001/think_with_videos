from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import AppConfig
from .exporters import make_sft_record
from .llm_client import OpenAICompatibleVideoClient
from .prompt_templates import (
    overview_answerability_system_prompt,
    overview_answerability_user_prompt,
    overview_system_prompt,
    overview_user_prompt,
    rerank_system_prompt,
    rerank_user_prompt,
    segment_caption_system_prompt,
    segment_caption_user_prompt,
    trajectory_check_system_prompt,
    trajectory_check_user_prompt,
    trajectory_generation_system_prompt,
    trajectory_generation_user_prompt,
    verify_clip_system_prompt,
    verify_clip_user_prompt,
)
from .retrieval import apply_model_rerank, rank_by_lexical_overlap
from .routing import decide_route_type, select_final_intervals
from .schema import (
    Interval,
    MemoryEntry,
    RawSample,
    RetrievalDecision,
    Trajectory,
    TrajectoryTurn,
    VerificationResult,
    VideoIndex,
    VideoOverview,
)
from .text_utils import (
    answer_text_for_retrieval,
    build_retrieval_query,
    clip_float,
    dedupe_preserve_order,
    ensure_dir,
    load_json,
    merge_intervals,
    normalize_answer_letter,
    parse_json_response,
    parse_mcq_options,
    stable_video_key,
    write_json,
)
from .video_utils import build_proxy_clip, expand_interval, get_video_duration, make_equal_segments, split_interval_evenly


class SFTBuilderPipeline:
    def __init__(self, config: AppConfig, output_dir: str | Path) -> None:
        self.config = config
        self.output_dir = Path(output_dir)

        self.cache_dir = ensure_dir(self.output_dir / "cache")
        self.llm_cache_dir = ensure_dir(self.cache_dir / "llm")
        self.video_index_cache_dir = ensure_dir(self.cache_dir / "video_index")
        self.proxies_dir = ensure_dir(self.output_dir / "proxies")
        self.intermediate_dir = ensure_dir(self.output_dir / "intermediate")
        self.video_indexes_dir = ensure_dir(self.intermediate_dir / "video_indexes")
        self.sample_records_dir = ensure_dir(self.intermediate_dir / "sample_records")
        self.final_dir = ensure_dir(self.output_dir / "final")

        self.client = OpenAICompatibleVideoClient(config.api, self.llm_cache_dir)

    def parse_raw_sample(self, row: Dict[str, Any]) -> RawSample:
        sample_id = str(row.get("id", row.get("sample_id", "")))
        dataset = str(row.get("dataset", "unknown"))
        task_type = str(row.get("task_type", "qa"))
        video_path = str(row["video_path"])
        question = str(row["question"])
        answer = str(row["answer"])

        start_times = row.get("start_time")
        end_times = row.get("end_time")
        gt_intervals: List[Interval] = []

        if start_times is not None and end_times is not None:
            if not isinstance(start_times, list):
                start_times = [start_times]
            if not isinstance(end_times, list):
                end_times = [end_times]
            for s, e in zip(start_times, end_times):
                gt_intervals.append(
                    Interval(
                        start_time=float(s),
                        end_time=float(e),
                        source="ground_truth",
                        score=1.0,
                        summary="Provided temporal grounding interval",
                    )
                )

        return RawSample(
            sample_id=sample_id,
            dataset=dataset,
            task_type=task_type,
            video_path=video_path,
            question=question,
            answer=answer,
            gt_intervals=gt_intervals,
            raw=row,
        )

    def _video_cache_path(self, video_path: str) -> Path:
        return self.video_index_cache_dir / f"{stable_video_key(video_path)}.json"

    def build_or_load_video_index(self, video_path: str) -> VideoIndex:
        cache_path = self._video_cache_path(video_path)
        if cache_path.exists():
            raw = load_json(cache_path)
            overview = VideoOverview(**raw["overview"])
            entries = [MemoryEntry(**entry) for entry in raw["memory_entries"]]
            return VideoIndex(
                video_path=raw["video_path"],
                video_key=raw["video_key"],
                duration=float(raw["duration"]),
                overview_proxy_path=raw["overview_proxy_path"],
                overview=overview,
                memory_entries=entries,
            )

        duration = get_video_duration(video_path)
        video_key = stable_video_key(video_path)
        video_proxy_root = ensure_dir(self.proxies_dir / video_key)
        overview_proxy = build_proxy_clip(
            video_path=video_path,
            output_path=str(video_proxy_root / "overview.mp4"),
            start_time=None,
            end_time=None,
            target_frames=self.config.pipeline.overview_target_frames,
            width=self.config.pipeline.overview_width,
            crf=self.config.pipeline.overview_crf,
        )
        overview_json = self.client.call(
            system_prompt=overview_system_prompt(),
            user_prompt=overview_user_prompt(duration),
            video_path=overview_proxy,
            mm_processor_kwargs={"fps": 1.0, "do_sample_frames": True},
            force_json=True,
        )
        overview = VideoOverview(**overview_json)

        coarse_segments = make_equal_segments(
            duration=duration,
            target_segments=self.config.pipeline.coarse_target_segments,
            min_sec=self.config.pipeline.coarse_min_sec,
            max_sec=self.config.pipeline.coarse_max_sec,
        )

        memory_entries: List[MemoryEntry] = []
        for idx, (start_time, end_time) in enumerate(coarse_segments):
            seg_id = f"seg_{idx:04d}"
            proxy_path = build_proxy_clip(
                video_path=video_path,
                output_path=str(video_proxy_root / f"{seg_id}.mp4"),
                start_time=start_time,
                end_time=end_time,
                target_frames=self.config.pipeline.coarse_target_frames,
                width=self.config.pipeline.coarse_width,
                crf=self.config.pipeline.coarse_crf,
            )
            segment_json = self.client.call(
                system_prompt=segment_caption_system_prompt(),
                user_prompt=segment_caption_user_prompt(start_time, end_time, duration),
                video_path=proxy_path,
                mm_processor_kwargs={"fps": 1.0, "do_sample_frames": True},
                force_json=True,
            )
            memory_entries.append(
                MemoryEntry(
                    segment_id=seg_id,
                    start_time=start_time,
                    end_time=end_time,
                    proxy_path=proxy_path,
                    segment_summary=segment_json["segment_summary"],
                    events=segment_json.get("events", []),
                    entities=segment_json.get("entities", []),
                    visible_text=segment_json.get("visible_text", []),
                    fine_grained_cues=segment_json.get("fine_grained_cues", []),
                    uncertainties=segment_json.get("uncertainties", []),
                )
            )

        video_index = VideoIndex(
            video_path=video_path,
            video_key=video_key,
            duration=duration,
            overview_proxy_path=overview_proxy,
            overview=overview,
            memory_entries=memory_entries,
        )
        write_json(cache_path, video_index.to_dict())
        write_json(self.video_indexes_dir / f"{video_key}.json", video_index.to_dict())
        return video_index

    def judge_overview_answerability(self, sample: RawSample, video_index: VideoIndex) -> Tuple[bool, float, str]:
        result = self.client.call(
            system_prompt=overview_answerability_system_prompt(),
            user_prompt=overview_answerability_user_prompt(
                question=sample.question,
                answer=sample.answer,
                overview=video_index.overview.to_dict(),
            ),
            video_path=None,
            force_json=True,
        )
        return (
            bool(result.get("answerable_from_overview", False)),
            float(result.get("confidence", 0.0)),
            str(result.get("reason", "")),
        )

    def rerank_candidates(self, sample: RawSample, video_index: VideoIndex) -> List[MemoryEntry]:
        query = build_retrieval_query(sample.question, sample.answer)
        lexical_top = rank_by_lexical_overlap(
            query=query,
            entries=list(video_index.memory_entries),
            top_k=self.config.pipeline.top_k_retrieve,
        )
        if not lexical_top:
            return []

        rerank_payload = []
        for entry in lexical_top:
            rerank_payload.append(
                {
                    "segment_id": entry.segment_id,
                    "start_time": entry.start_time,
                    "end_time": entry.end_time,
                    "segment_summary": entry.segment_summary,
                    "events": entry.events,
                    "visible_text": entry.visible_text,
                    "fine_grained_cues": entry.fine_grained_cues,
                    "lexical_score": entry.lexical_score,
                }
            )

        rerank_json = self.client.call(
            system_prompt=rerank_system_prompt(),
            user_prompt=rerank_user_prompt(
                question=sample.question,
                answer=sample.answer,
                overview=video_index.overview.to_dict(),
                candidates=rerank_payload,
            ),
            video_path=None,
            force_json=True,
        )
        reranked = apply_model_rerank(lexical_top, rerank_json)
        return reranked[: self.config.pipeline.top_k_rerank]

    def _verify_interval(
        self,
        sample: RawSample,
        video_index: VideoIndex,
        interval: Interval,
        tag: str,
    ) -> VerificationResult:
        proxy_root = ensure_dir(self.proxies_dir / video_index.video_key / "verify")
        clip_name = f"{tag}_{interval.start_time:.3f}_{interval.end_time:.3f}.mp4".replace("/", "_")
        proxy_path = build_proxy_clip(
            video_path=sample.video_path,
            output_path=str(proxy_root / clip_name),
            start_time=interval.start_time,
            end_time=interval.end_time,
            target_frames=self.config.pipeline.fine_target_frames,
            width=self.config.pipeline.fine_width,
            crf=self.config.pipeline.fine_crf,
        )
        verify_json = self.client.call(
            system_prompt=verify_clip_system_prompt(),
            user_prompt=verify_clip_user_prompt(
                question=sample.question,
                answer=sample.answer,
                start_time=interval.start_time,
                end_time=interval.end_time,
            ),
            video_path=proxy_path,
            mm_processor_kwargs={"fps": 2.0, "do_sample_frames": True},
            force_json=True,
        )
        return VerificationResult(
            label=str(verify_json["label"]),
            confidence=float(verify_json.get("confidence", 0.0)),
            clip_summary=str(verify_json.get("clip_summary", "")),
            evidence=verify_json.get("evidence", []),
            contradictions=verify_json.get("contradictions", []),
            suggested_interval_quality=str(verify_json.get("suggested_interval_quality", "wrong_region")),
            answer_supported=bool(verify_json.get("answer_supported", False)),
            interval=interval,
            proxy_path=proxy_path,
        )

    def _fine_refine_from_entry(self, sample: RawSample, video_index: VideoIndex, entry: MemoryEntry) -> List[VerificationResult]:
        windows = split_interval_evenly(
            start_time=entry.start_time,
            end_time=entry.end_time,
            target_window_sec=self.config.pipeline.fine_target_sec,
            min_window_sec=self.config.pipeline.fine_min_sec,
            max_window_sec=self.config.pipeline.fine_max_sec,
            max_windows=self.config.pipeline.fine_max_per_coarse,
        )
        results: List[VerificationResult] = []
        for idx, (s, e) in enumerate(windows):
            interval = Interval(
                start_time=s,
                end_time=e,
                source=f"fine_from_{entry.segment_id}",
                score=max(entry.rerank_score, entry.lexical_score),
                summary=entry.segment_summary,
            )
            results.append(self._verify_interval(sample, video_index, interval, tag=f"{entry.segment_id}_fine_{idx:02d}"))
        return results

    def decide_intervals(self, sample: RawSample, video_index: VideoIndex) -> RetrievalDecision:
        overview_answerable, overview_confidence, overview_reason = self.judge_overview_answerability(sample, video_index)

        if sample.gt_intervals:
            verification_results: List[VerificationResult] = []
            for idx, gt in enumerate(sample.gt_intervals):
                duration = video_index.duration
                expanded_start, expanded_end = expand_interval(
                    gt.start_time,
                    gt.end_time,
                    duration,
                    left_margin=max(0.5, (gt.end_time - gt.start_time) * 0.2),
                    right_margin=max(0.5, (gt.end_time - gt.start_time) * 0.2),
                )
                interval = Interval(
                    start_time=expanded_start,
                    end_time=expanded_end,
                    source="ground_truth_expanded",
                    score=1.0,
                    summary="Expanded interval around provided grounding",
                )
                verification_results.append(
                    self._verify_interval(sample, video_index, interval, tag=f"gt_{idx:02d}")
                )

            broad_interval = None
            if len(sample.gt_intervals) == 1:
                gt = sample.gt_intervals[0]
                broad_start, broad_end = expand_interval(
                    gt.start_time,
                    gt.end_time,
                    video_index.duration,
                    left_margin=max(2.0, (gt.end_time - gt.start_time)),
                    right_margin=max(2.0, (gt.end_time - gt.start_time)),
                )
                broad_interval = Interval(
                    start_time=broad_start,
                    end_time=broad_end,
                    source="broad_from_gt",
                    score=0.9,
                    summary="Broad context interval synthesized from ground truth",
                )

            decision = RetrievalDecision(
                retrieved_entries=[],
                selected_intervals=sample.gt_intervals,
                broad_interval=broad_interval,
                distractor_interval=None,
                verification_results=verification_results,
                overview_answerable=overview_answerable,
                overview_answerable_confidence=overview_confidence,
                overview_answerable_reason=overview_reason,
            )
            decision.route_type = decide_route_type(video_index.duration, sample.gt_intervals, decision, self.config.pipeline)
            return decision

        reranked_entries = self.rerank_candidates(sample, video_index)
        verification_results: List[VerificationResult] = []
        for entry in reranked_entries[: max(1, min(3, len(reranked_entries)))]:
            verification_results.extend(self._fine_refine_from_entry(sample, video_index, entry))

        selected_intervals = select_final_intervals([], verification_results, self.config.pipeline)

        broad_interval = None
        if reranked_entries and selected_intervals:
            best_entry = reranked_entries[0]
            broad_interval = Interval(
                start_time=best_entry.start_time,
                end_time=best_entry.end_time,
                source=f"broad_{best_entry.segment_id}",
                score=max(best_entry.rerank_score, best_entry.lexical_score),
                summary=best_entry.segment_summary,
            )

        distractor_interval = None
        if reranked_entries:
            for entry in reranked_entries:
                rejected_fines = [
                    vr for vr in verification_results
                    if vr.interval.source.startswith(f"fine_from_{entry.segment_id}")
                    and vr.label == "reject"
                    and vr.confidence >= self.config.pipeline.partial_confidence_threshold
                ]
                if rejected_fines:
                    first = rejected_fines[0].interval
                    distractor_interval = Interval(
                        start_time=first.start_time,
                        end_time=first.end_time,
                        source="distractor_rejected",
                        score=rejected_fines[0].confidence,
                        summary=rejected_fines[0].clip_summary,
                    )
                    break

        decision = RetrievalDecision(
            retrieved_entries=reranked_entries,
            selected_intervals=selected_intervals,
            broad_interval=broad_interval,
            distractor_interval=distractor_interval if selected_intervals else None,
            verification_results=verification_results,
            overview_answerable=overview_answerable,
            overview_answerable_confidence=overview_confidence,
            overview_answerable_reason=overview_reason,
        )
        decision.route_type = decide_route_type(video_index.duration, [], decision, self.config.pipeline)
        return decision

    def _build_memory_observations(self, decision: RetrievalDecision) -> List[Dict[str, Any]]:
        items = []
        for entry in decision.retrieved_entries:
            items.append(
                {
                    "segment_id": entry.segment_id,
                    "start_time": entry.start_time,
                    "end_time": entry.end_time,
                    "summary": entry.segment_summary,
                    "events": entry.events,
                    "visible_text": entry.visible_text,
                    "fine_grained_cues": entry.fine_grained_cues,
                    "rerank_score": entry.rerank_score,
                    "rerank_note": entry.rerank_note,
                }
            )
        return items

    def _build_verified_observations(self, decision: RetrievalDecision) -> List[Dict[str, Any]]:
        obs = []
        for vr in decision.verification_results:
            obs.append(
                {
                    "interval": [vr.interval.start_time, vr.interval.end_time],
                    "label": vr.label,
                    "confidence": vr.confidence,
                    "clip_summary": vr.clip_summary,
                    "evidence": vr.evidence,
                    "contradictions": vr.contradictions,
                    "suggested_interval_quality": vr.suggested_interval_quality,
                    "answer_supported": vr.answer_supported,
                }
            )
        return obs

    def generate_trajectory(
        self,
        sample: RawSample,
        video_index: VideoIndex,
        decision: RetrievalDecision,
    ) -> Trajectory:
        payload = {
            "video_duration": video_index.duration,
            "question": sample.question,
            "gold_answer": sample.answer,
            "route_type": decision.route_type,
            "overview": video_index.overview.to_dict(),
            "overview_answerability": {
                "answerable": decision.overview_answerable,
                "confidence": decision.overview_answerable_confidence,
                "reason": decision.overview_answerable_reason,
            },
            "memory_entries": self._build_memory_observations(decision),
            "verified_evidence": self._build_verified_observations(decision),
            "broad_interval": (
                [decision.broad_interval.start_time, decision.broad_interval.end_time]
                if decision.broad_interval
                else None
            ),
            "distractor_interval": (
                [decision.distractor_interval.start_time, decision.distractor_interval.end_time]
                if decision.distractor_interval
                else None
            ),
            "final_target_intervals": [
                [i.start_time, i.end_time] for i in decision.selected_intervals
            ],
            "route_guidance": {
                "direct_answer": "Only if overview evidence is clearly sufficient.",
                "direct_localize": "Usually one local crop is enough.",
                "boundary_refine": "Use a broader candidate first, then a refined crop.",
                "memory_retrieve": "Retrieve coarse memory first, then inspect one or two regions.",
                "reflection_repair": "Inspect a plausible but wrong/insufficient region, reject it, then inspect the correct region.",
                "multi_hop_compose": "Inspect at least two distinct evidence regions and combine them.",
            },
        }
        traj_json = self.client.call(
            system_prompt=trajectory_generation_system_prompt(),
            user_prompt=trajectory_generation_user_prompt(payload),
            video_path=None,
            force_json=True,
        )

        turns = [
            TrajectoryTurn(
                thought=str(turn["thought"]),
                action=turn.get("action"),
                observation=turn.get("observation"),
            )
            for turn in traj_json.get("turns", [])
        ]
        trajectory = Trajectory(
            route_type=str(traj_json["route_type"]),
            turns=turns,
            final_answer=str(traj_json["final_answer"]),
            final_evidence_intervals=traj_json.get("final_evidence_intervals", []),
            quality_notes=str(traj_json.get("quality_notes", "")),
        )

        check_payload = {
            "question": sample.question,
            "gold_answer": sample.answer,
            "overview": video_index.overview.to_dict(),
            "route_type": decision.route_type,
            "evidence_package": {
                "memory_entries": self._build_memory_observations(decision),
                "verified_evidence": self._build_verified_observations(decision),
                "selected_intervals": [[i.start_time, i.end_time] for i in decision.selected_intervals],
            },
            "trajectory": trajectory.to_dict(),
        }
        check_json = self.client.call(
            system_prompt=trajectory_check_system_prompt(),
            user_prompt=trajectory_check_user_prompt(check_payload),
            video_path=None,
            force_json=True,
        )

        valid = bool(check_json.get("valid", False))
        trajectory.valid = valid
        trajectory.validation_issues = [str(x) for x in check_json.get("issues", [])]
        if not valid and check_json.get("repaired_trajectory"):
            repaired = check_json["repaired_trajectory"]
            trajectory.route_type = repaired["route_type"]
            trajectory.turns = [
                TrajectoryTurn(
                    thought=str(turn["thought"]),
                    action=turn.get("action"),
                    observation=turn.get("observation"),
                )
                for turn in repaired.get("turns", [])
            ]
            trajectory.final_answer = str(repaired["final_answer"])
            trajectory.final_evidence_intervals = repaired.get("final_evidence_intervals", [])
            trajectory.quality_notes = str(repaired.get("quality_notes", ""))
            trajectory.valid = True
        return trajectory

    def process_sample(self, sample: RawSample):
        video_index = self.build_or_load_video_index(sample.video_path)
        decision = self.decide_intervals(sample, video_index)
        trajectory = self.generate_trajectory(sample, video_index, decision)

        if not trajectory.valid and not self.config.pipeline.keep_invalid_records:
            return None

        final_route_type = trajectory.route_type or decision.route_type
        record = make_sft_record(
            sample=sample,
            duration=video_index.duration,
            route_type=final_route_type,
            overview=video_index.overview.to_dict(),
            retrieval_decision=decision.to_dict(),
            trajectory=trajectory.to_dict(),
        )

        if self.config.pipeline.save_intermediate_json:
            write_json(
                self.sample_records_dir / f"{sample.sample_id}.json",
                record.to_dict(),
            )
        return record

    def export_records(self, records: List) -> None:
        from .text_utils import write_json, write_jsonl

        sft_rows = [record.to_dict() for record in records]
        write_jsonl(self.final_dir / "sft_records.jsonl", sft_rows)

        if self.config.output.export_sharegpt:
            sharegpt_rows = []
            for record in records:
                sharegpt_rows.append(
                    {
                        "id": record.sample.sample_id,
                        "video": record.sample.video_path,
                        "messages": record.messages,
                        "route_type": record.route_type,
                    }
                )
            write_jsonl(self.final_dir / "sharegpt_records.jsonl", sharegpt_rows)

        if self.config.output.export_flattened_text:
            flat_rows = []
            for record in records:
                flat_rows.append(
                    {
                        "id": record.sample.sample_id,
                        "video": record.sample.video_path,
                        "question": record.sample.question,
                        "route_type": record.route_type,
                        "assistant_trace": record.flattened_trace,
                    }
                )
            write_jsonl(self.final_dir / "flattened_traces.jsonl", flat_rows)

        route_stats: Dict[str, int] = {}
        valid_count = 0
        for record in records:
            route_stats[record.route_type] = route_stats.get(record.route_type, 0) + 1
            if record.trajectory.get("valid", False):
                valid_count += 1

        stats = {
            "num_records": len(records),
            "num_valid_records": valid_count,
            "route_distribution": route_stats,
        }
        write_json(self.final_dir / "stats.json", stats)
