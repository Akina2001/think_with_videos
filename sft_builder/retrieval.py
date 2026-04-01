from __future__ import annotations

from typing import Dict, List

from .schema import MemoryEntry
from .text_utils import lexical_overlap_score


def rank_by_lexical_overlap(query: str, entries: List[MemoryEntry], top_k: int) -> List[MemoryEntry]:
    scored: List[MemoryEntry] = []
    for entry in entries:
        text = "\n".join(
            [
                entry.segment_summary,
                " ".join(entry.events),
                " ".join(entry.entities),
                " ".join(entry.visible_text),
                " ".join(entry.fine_grained_cues),
            ]
        )
        entry.lexical_score = lexical_overlap_score(query, text)
        scored.append(entry)
    scored.sort(key=lambda x: x.lexical_score, reverse=True)
    return scored[:top_k]


def apply_model_rerank(entries: List[MemoryEntry], rerank_result: Dict[str, object]) -> List[MemoryEntry]:
    scores = rerank_result.get("scores", {})
    notes = rerank_result.get("notes", {})
    ranked_ids = rerank_result.get("ranked_segment_ids", [])

    id_to_entry = {e.segment_id: e for e in entries}
    for seg_id, score in scores.items():
        if seg_id in id_to_entry:
            id_to_entry[seg_id].rerank_score = float(score)
    for seg_id, note in notes.items():
        if seg_id in id_to_entry:
            id_to_entry[seg_id].rerank_note = str(note)

    reranked: List[MemoryEntry] = []
    for seg_id in ranked_ids:
        if seg_id in id_to_entry:
            reranked.append(id_to_entry[seg_id])

    remaining = [e for e in entries if e.segment_id not in {x.segment_id for x in reranked}]
    remaining.sort(key=lambda x: (x.rerank_score, x.lexical_score), reverse=True)
    reranked.extend(remaining)
    return reranked
