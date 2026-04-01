from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


WORD_RE = re.compile(r"[A-Za-z0-9']+")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def sha1_file(path: str | Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def stable_video_key(video_path: str | Path) -> str:
    p = Path(video_path)
    stat = p.stat()
    base = f"{p.resolve()}::{stat.st_size}::{int(stat.st_mtime)}"
    return sha1_text(base)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, obj: Any) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def extract_json_block(text: str) -> str:
    cleaned = strip_code_fences(text)

    start_candidates = [idx for idx in [cleaned.find("{"), cleaned.find("[")] if idx != -1]
    if not start_candidates:
        raise ValueError("No JSON start token found.")
    start = min(start_candidates)

    end_obj = cleaned.rfind("}")
    end_arr = cleaned.rfind("]")
    end = max(end_obj, end_arr)
    if end == -1 or end <= start:
        raise ValueError("No valid JSON end token found.")
    return cleaned[start : end + 1]


def parse_json_response(text: str) -> Any:
    block = extract_json_block(text)
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        block = re.sub(r",\s*([}\]])", r"\1", block)
        return json.loads(block)


def tokenize(text: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(text or "")]


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_mcq_options(question: str) -> Dict[str, str]:
    options: Dict[str, str] = {}
    pattern = re.compile(
        r"(?:(?:^|\n)\s*([A-H])[\.\):]\s*(.+?))(?=(?:\n\s*[A-H][\.\):]\s)|$)",
        flags=re.DOTALL,
    )
    for key, value in pattern.findall(question):
        options[key.upper()] = normalize_space(value)
    return options


def normalize_answer_letter(answer: str) -> Optional[str]:
    if not answer:
        return None
    m = re.search(r"\b([A-H])\b", answer.strip().upper())
    return m.group(1) if m else None


def answer_text_for_retrieval(question: str, answer: str) -> str:
    options = parse_mcq_options(question)
    letter = normalize_answer_letter(answer)
    if letter and letter in options:
        return options[letter]
    return normalize_space(answer)


def build_retrieval_query(question: str, answer: str) -> str:
    answer_text = answer_text_for_retrieval(question, answer)
    if answer_text:
        return normalize_space(f"{question}\nCorrect answer content: {answer_text}")
    return normalize_space(question)


def lexical_overlap_score(query: str, text: str) -> float:
    q_tokens = tokenize(query)
    t_tokens = tokenize(text)
    if not q_tokens or not t_tokens:
        return 0.0

    q_counts: Dict[str, int] = {}
    t_counts: Dict[str, int] = {}
    for token in q_tokens:
        q_counts[token] = q_counts.get(token, 0) + 1
    for token in t_tokens:
        t_counts[token] = t_counts.get(token, 0) + 1

    overlap = sum(min(q_counts[k], t_counts.get(k, 0)) for k in q_counts)
    denom = math.sqrt(len(q_tokens) * len(t_tokens))
    if denom == 0:
        return 0.0
    return overlap / denom


def clip_float(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        item = normalize_space(str(item))
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def maybe_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def merge_intervals(intervals: List[Tuple[float, float]], gap: float = 2.0) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged: List[Tuple[float, float]] = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def intervals_are_far_apart(intervals: List[Tuple[float, float]], gap: float = 5.0) -> bool:
    if len(intervals) <= 1:
        return False
    intervals = sorted(intervals, key=lambda x: x[0])
    for (s1, e1), (s2, e2) in zip(intervals, intervals[1:]):
        if s2 - e1 > gap:
            return True
    return False


def seeded_random_choice(items: List[Any], seed_text: str) -> Optional[Any]:
    if not items:
        return None
    rnd = random.Random(sha1_text(seed_text))
    return rnd.choice(items)
