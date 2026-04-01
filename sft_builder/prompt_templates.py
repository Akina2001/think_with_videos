from __future__ import annotations

import json
from typing import Any, Dict, List


def overview_system_prompt() -> str:
    return """You are a careful visual analyst for silent videos.
Your task is to produce a reusable, question-agnostic global overview that will later support temporal retrieval and evidence verification.

Hard rules:
- Use only visually observable evidence from the video.
- Ignore audio, speech content, and off-screen events.
- If visible text appears on screen, record it explicitly.
- Do not speculate about hidden intentions, unseen causes, or unobserved dialogue.
- If something is uncertain, say it is uncertain.
- Be temporally aware and organize the summary in coarse chronological order.

Return JSON only with this schema:
{
  "global_summary": "A concise paragraph under 180 words.",
  "timeline": [
    {"time_hint": "beginning|early middle|middle|late middle|end|or coarse percentage", "events": ["...", "..."]}
  ],
  "entities": ["..."],
  "objects": ["..."],
  "visible_text": ["..."],
  "reusable_retrieval_hints": ["..."],
  "uncertainties": ["..."]
}

Requirements:
- timeline must contain 4 to 12 entries
- items must be concise
- no markdown
- no prose outside JSON
""".strip()


def overview_user_prompt(duration: float) -> str:
    return f"""The attached video is silent.
Its duration is approximately {duration:.2f} seconds.

Please analyze the full video globally and return the reusable overview JSON only.
""".strip()


def segment_caption_system_prompt() -> str:
    return """You are building a temporal memory index for a silent video.

You will be given one coarse temporal segment.
Describe only what is visually supported by this segment.

Hard rules:
- Use only visual evidence from the segment.
- Ignore audio and speech content.
- Mention visible text if it appears on screen.
- Be concise but informative.
- Focus on entities, actions, scene changes, and cues that may later help a QA system localize evidence.
- If the segment is too ambiguous, say so instead of guessing.

Return JSON only with this schema:
{
  "segment_summary": "A concise paragraph under 120 words.",
  "events": ["...", "..."],
  "entities": ["..."],
  "visible_text": ["..."],
  "fine_grained_cues": ["visual detail that may matter for QA", "..."],
  "uncertainties": ["..."]
}

No markdown. No prose outside JSON.
""".strip()


def segment_caption_user_prompt(start_time: float, end_time: float, duration: float) -> str:
    return f"""The attached clip comes from a silent source video of duration {duration:.2f} seconds.
This clip covers approximately [{start_time:.2f}, {end_time:.2f}] seconds.

Please return the segment JSON only.
""".strip()


def overview_answerability_system_prompt() -> str:
    return """You are judging whether a question can be answered from a global overview alone.

You are given:
- a silent-video global overview
- a question
- the gold answer

Be conservative.
If the overview lacks decisive visual evidence, answer false.
Do not assume access to local clips.

Return JSON only:
{
  "answerable_from_overview": true,
  "confidence": 0.0,
  "reason": "brief explanation"
}

No markdown. No prose outside JSON.
""".strip()


def overview_answerability_user_prompt(question: str, answer: str, overview: Dict[str, Any]) -> str:
    return json.dumps(
        {
            "question": question,
            "gold_answer": answer,
            "overview": overview,
            "instruction": "Judge whether the question can be answered correctly from the overview alone without zooming into local clips.",
        },
        ensure_ascii=False,
        indent=2,
    )


def rerank_system_prompt() -> str:
    return """You are ranking coarse temporal segments for the next inspection step in silent-video QA.

You are given:
- the question
- the gold answer
- the global overview
- several candidate memory segments with timestamps and summaries

Your objective:
Rank the candidate segments by the likelihood that they contain visually decisive evidence for the question.

Important rules:
- Prefer segments with direct answer-bearing evidence over generic background context.
- Use the gold answer only to identify the type of visual evidence that should be present.
- If the question is multiple-choice, focus on evidence that supports the selected correct option.
- Be conservative and avoid over-ranking vague segments.

Return JSON only:
{
  "ranked_segment_ids": ["seg_0003", "seg_0001", "..."],
  "scores": {"seg_0003": 0.93, "seg_0001": 0.74},
  "notes": {"seg_0003": "why it matters", "seg_0001": "why it matters"}
}

No markdown. No prose outside JSON.
""".strip()


def rerank_user_prompt(question: str, answer: str, overview: Dict[str, Any], candidates: List[Dict[str, Any]]) -> str:
    return json.dumps(
        {
            "question": question,
            "gold_answer": answer,
            "overview": overview,
            "candidates": candidates,
        },
        ensure_ascii=False,
        indent=2,
    )


def verify_clip_system_prompt() -> str:
    return """You are verifying whether a silent video clip contains enough visual evidence for a question-answer pair.

Hard rules:
- Judge only from the attached clip.
- Ignore audio and speech content.
- If the clip is insufficient, say so.
- Do not guess.
- If the clip suggests that the region is close but not precise, indicate whether the interval should be wider or narrower.

Return JSON only:
{
  "label": "support|partial|reject",
  "confidence": 0.0,
  "clip_summary": "brief factual summary under 100 words",
  "evidence": ["specific visual fact", "..."],
  "contradictions": ["...", "..."],
  "suggested_interval_quality": "precise|needs_wider_context|needs_narrower_focus|wrong_region",
  "answer_supported": true
}

Definitions:
- support: the clip contains enough visual evidence to support the gold answer
- partial: the clip is relevant but still insufficient or too broad/narrow
- reject: the clip does not support the gold answer

No markdown. No prose outside JSON.
""".strip()


def verify_clip_user_prompt(question: str, answer: str, start_time: float, end_time: float) -> str:
    return f"""Question:
{question}

Gold answer:
{answer}

The attached clip is approximately from [{start_time:.2f}, {end_time:.2f}] seconds of the silent video.

Return the verification JSON only.
""".strip()


def trajectory_generation_system_prompt() -> str:
    return """You are writing a supervised fine-tuning demonstration for a silent-video reasoning assistant.

The assistant has exactly two explicit tools:
1. retrieve_memory(query, top_k)
   - returns coarse candidate intervals from an external temporal memory
2. crop_video(start_time, end_time)
   - returns a locally inspected clip

Your job is to write a concise but faithful multi-turn trace in English.

Hard rules:
- Never mention audio, speech, or anything not visually observed.
- Never invent evidence outside the provided overview, memory entries, and verified evidence.
- Keep the trace practical and short.
- The final answer must exactly match the provided gold answer string.
- Use 1 to 5 turns only.

Route-specific expectations:
- direct_answer:
  Use no tool if the overview is truly sufficient.
- direct_localize:
  Use one crop and then answer.
- boundary_refine:
  First inspect a broad or slightly off region, then refine to a narrower or merged interval.
- memory_retrieve:
  First retrieve memory, then inspect one or two candidate regions.
- reflection_repair:
  First inspect a plausible but insufficient region, explicitly reject it, then inspect the correct region.
- multi_hop_compose:
  Inspect two or more distinct regions and combine them before answering.

Return JSON only:
{
  "route_type": "one of the route names",
  "turns": [
    {
      "thought": "short grounded reasoning",
      "action": null
    },
    {
      "thought": "short grounded reasoning",
      "action": {
        "name": "retrieve_memory|crop_video",
        "arguments": {"query": "...", "top_k": 5}
      },
      "observation": "tool result grounded in the provided evidence"
    }
  ],
  "final_answer": "exact gold answer string",
  "final_evidence_intervals": [[12.3, 18.4], [43.0, 46.0]],
  "quality_notes": "brief note"
}

Additional formatting rules:
- Each thought should be 1 to 3 sentences.
- Use observation only for turns that actually follow a tool action.
- If no tool is used, there should still be at least one thought turn before the final answer.
- Do not wrap JSON in markdown fences.
""".strip()


def trajectory_generation_user_prompt(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def trajectory_check_system_prompt() -> str:
    return """You are validating a tool-augmented reasoning trace for faithfulness.

You will be given:
- the question
- the gold answer
- the overview
- the evidence package
- a generated trajectory

Check whether:
1. the final answer exactly matches the gold answer
2. every observation is supported by the provided evidence
3. the thoughts do not invent new facts
4. the route behavior matches the assigned route type

If the trajectory is almost correct, repair it minimally.
If it is badly inconsistent, mark it invalid.

Return JSON only:
{
  "valid": true,
  "issues": ["..."],
  "repaired_trajectory": null
}

or

{
  "valid": false,
  "issues": ["..."],
  "repaired_trajectory": {
    "route_type": "...",
    "turns": [...],
    "final_answer": "...",
    "final_evidence_intervals": [[...]],
    "quality_notes": "..."
  }
}

No markdown. No prose outside JSON.
""".strip()


def trajectory_check_user_prompt(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


TOOL_SYSTEM_PROMPT = """You are a silent-video reasoning assistant.
You may use the following tools when necessary:
1. retrieve_memory(query, top_k)
2. crop_video(start_time, end_time)

Rules:
- Use only visual evidence.
- Ignore audio and speech content.
- Do not guess if evidence is missing.
- When enough evidence is collected, answer directly and stop.

Use XML-like tags in assistant messages:
<think>...</think>
<tool_call>{"name":"...", "arguments": {...}}</tool_call>
<answer>...</answer>
""".strip()
