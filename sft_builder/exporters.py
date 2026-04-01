from __future__ import annotations

import json
from typing import Any, Dict, List

from .prompt_templates import TOOL_SYSTEM_PROMPT
from .schema import SFTRecord, Trajectory


def render_flattened_trace(trajectory: Dict[str, Any]) -> str:
    parts: List[str] = []
    for turn in trajectory["turns"]:
        parts.append(f"<think>{turn['thought']}</think>")
        if turn.get("action"):
            parts.append(
                "<tool_call>"
                + json.dumps(turn["action"], ensure_ascii=False)
                + "</tool_call>"
            )
        if turn.get("observation"):
            parts.append(f"<tool_response>{turn['observation']}</tool_response>")
    parts.append(f"<answer>{trajectory['final_answer']}</answer>")
    return "\n".join(parts)


def build_sharegpt_messages(
    question: str,
    trajectory: Dict[str, Any],
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    turns = trajectory["turns"]
    for turn in turns:
        assistant_parts = [f"<think>{turn['thought']}</think>"]
        if turn.get("action"):
            assistant_parts.append(
                "<tool_call>"
                + json.dumps(turn["action"], ensure_ascii=False)
                + "</tool_call>"
            )
            messages.append({"role": "assistant", "content": "\n".join(assistant_parts)})
            if turn.get("observation"):
                tool_name = turn["action"]["name"]
                messages.append(
                    {
                        "role": "tool",
                        "name": tool_name,
                        "content": turn["observation"],
                    }
                )
        else:
            messages.append({"role": "assistant", "content": "\n".join(assistant_parts)})
    messages.append({"role": "assistant", "content": f"<answer>{trajectory['final_answer']}</answer>"})
    return messages


def make_sft_record(
    sample,
    duration: float,
    route_type: str,
    overview: Dict[str, Any],
    retrieval_decision: Dict[str, Any],
    trajectory: Dict[str, Any],
) -> SFTRecord:
    messages = build_sharegpt_messages(sample.question, trajectory)
    flattened = render_flattened_trace(trajectory)
    return SFTRecord(
        sample=sample,
        duration=duration,
        route_type=route_type,
        overview=overview,
        retrieval_decision=retrieval_decision,
        trajectory=trajectory,
        messages=messages,
        flattened_trace=flattened,
    )
