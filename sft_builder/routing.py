from __future__ import annotations

from typing import List, Optional

from .config import PipelineConfig
from .schema import Interval, RetrievalDecision, VerificationResult
from .text_utils import intervals_are_far_apart, merge_intervals


def _supportive_results(results: List[VerificationResult], threshold: float) -> List[VerificationResult]:
    return [
        r
        for r in results
        if r.answer_supported and r.label in {"support", "partial"} and r.confidence >= threshold
    ]


def select_final_intervals(
    gt_intervals: List[Interval],
    verification_results: List[VerificationResult],
    cfg: PipelineConfig,
) -> List[Interval]:
    if gt_intervals:
        return gt_intervals

    supportive = _supportive_results(verification_results, cfg.partial_confidence_threshold)
    if not supportive:
        return []

    intervals = [(r.interval.start_time, r.interval.end_time) for r in supportive]
    merged = merge_intervals(intervals, gap=cfg.merge_gap_sec)

    chosen: List[Interval] = []
    for idx, (s, e) in enumerate(merged):
        chosen.append(
            Interval(
                start_time=s,
                end_time=e,
                source="verified_merge",
                score=max(
                    [
                        r.confidence
                        for r in supportive
                        if not (r.interval.end_time < s or r.interval.start_time > e)
                    ]
                    or [0.0]
                ),
                summary="Merged verified evidence",
            )
        )
    return chosen


def decide_route_type(
    duration: float,
    gt_intervals: List[Interval],
    decision: RetrievalDecision,
    cfg: PipelineConfig,
) -> str:
    if decision.overview_answerable and duration <= cfg.short_video_sec:
        return "direct_answer"

    if len(gt_intervals) > 1:
        return "multi_hop_compose"

    if decision.distractor_interval is not None and decision.selected_intervals:
        return "reflection_repair"

    if len(decision.selected_intervals) > 1:
        gap_view = [(i.start_time, i.end_time) for i in decision.selected_intervals]
        if intervals_are_far_apart(gap_view, gap=max(5.0, cfg.merge_gap_sec * 2)):
            return "multi_hop_compose"
        return "boundary_refine"

    if len(decision.selected_intervals) == 1:
        if gt_intervals:
            return "direct_localize"
        if duration >= cfg.long_video_sec:
            return "memory_retrieve"
        if decision.broad_interval is not None:
            broad = decision.broad_interval
            final = decision.selected_intervals[0]
            broad_len = broad.end_time - broad.start_time
            final_len = final.end_time - final.start_time
            if broad_len > final_len * 1.5:
                return "boundary_refine"
        return "direct_localize"

    return "memory_retrieve"
