from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Interval:
    start_time: float
    end_time: float
    source: str = "unknown"
    score: float = 0.0
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RawSample:
    sample_id: str
    dataset: str
    task_type: str
    video_path: str
    question: str
    answer: str
    gt_intervals: List[Interval] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "task_type": self.task_type,
            "video_path": self.video_path,
            "question": self.question,
            "answer": self.answer,
            "gt_intervals": [i.to_dict() for i in self.gt_intervals],
            "raw": self.raw,
        }


@dataclass
class VideoOverview:
    global_summary: str
    timeline: List[Dict[str, Any]]
    entities: List[str]
    objects: List[str]
    visible_text: List[str]
    reusable_retrieval_hints: List[str]
    uncertainties: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryEntry:
    segment_id: str
    start_time: float
    end_time: float
    proxy_path: str
    segment_summary: str
    events: List[str]
    entities: List[str]
    visible_text: List[str]
    fine_grained_cues: List[str]
    uncertainties: List[str]
    lexical_score: float = 0.0
    rerank_score: float = 0.0
    rerank_note: str = ""

    def to_text(self) -> str:
        parts = [
            f"[{self.segment_id}] [{self.start_time:.2f}, {self.end_time:.2f}]",
            self.segment_summary,
        ]
        if self.events:
            parts.append("Events: " + "; ".join(self.events))
        if self.fine_grained_cues:
            parts.append("Fine cues: " + "; ".join(self.fine_grained_cues))
        if self.visible_text:
            parts.append("Visible text: " + "; ".join(self.visible_text))
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VideoIndex:
    video_path: str
    video_key: str
    duration: float
    overview_proxy_path: str
    overview: VideoOverview
    memory_entries: List[MemoryEntry]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_path": self.video_path,
            "video_key": self.video_key,
            "duration": self.duration,
            "overview_proxy_path": self.overview_proxy_path,
            "overview": self.overview.to_dict(),
            "memory_entries": [m.to_dict() for m in self.memory_entries],
        }


@dataclass
class VerificationResult:
    label: str
    confidence: float
    clip_summary: str
    evidence: List[str]
    contradictions: List[str]
    suggested_interval_quality: str
    answer_supported: bool
    interval: Interval
    proxy_path: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["interval"] = self.interval.to_dict()
        return data


@dataclass
class RetrievalDecision:
    retrieved_entries: List[MemoryEntry]
    selected_intervals: List[Interval]
    broad_interval: Optional[Interval]
    distractor_interval: Optional[Interval]
    verification_results: List[VerificationResult]
    overview_answerable: bool = False
    overview_answerable_confidence: float = 0.0
    overview_answerable_reason: str = ""
    route_type: str = "memory_retrieve"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retrieved_entries": [m.to_dict() for m in self.retrieved_entries],
            "selected_intervals": [i.to_dict() for i in self.selected_intervals],
            "broad_interval": self.broad_interval.to_dict() if self.broad_interval else None,
            "distractor_interval": self.distractor_interval.to_dict() if self.distractor_interval else None,
            "verification_results": [v.to_dict() for v in self.verification_results],
            "overview_answerable": self.overview_answerable,
            "overview_answerable_confidence": self.overview_answerable_confidence,
            "overview_answerable_reason": self.overview_answerable_reason,
            "route_type": self.route_type,
        }


@dataclass
class TrajectoryTurn:
    thought: str
    action: Optional[Dict[str, Any]]
    observation: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Trajectory:
    route_type: str
    turns: List[TrajectoryTurn]
    final_answer: str
    final_evidence_intervals: List[List[float]]
    quality_notes: str
    valid: bool = True
    validation_issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route_type": self.route_type,
            "turns": [t.to_dict() for t in self.turns],
            "final_answer": self.final_answer,
            "final_evidence_intervals": self.final_evidence_intervals,
            "quality_notes": self.quality_notes,
            "valid": self.valid,
            "validation_issues": self.validation_issues,
        }


@dataclass
class SFTRecord:
    sample: RawSample
    duration: float
    route_type: str
    overview: Dict[str, Any]
    retrieval_decision: Dict[str, Any]
    trajectory: Dict[str, Any]
    messages: List[Dict[str, Any]]
    flattened_trace: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample": self.sample.to_dict(),
            "duration": self.duration,
            "route_type": self.route_type,
            "overview": self.overview,
            "retrieval_decision": self.retrieval_decision,
            "trajectory": self.trajectory,
            "messages": self.messages,
            "flattened_trace": self.flattened_trace,
        }
