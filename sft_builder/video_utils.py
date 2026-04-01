from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

from .text_utils import clip_float, ensure_dir


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe are required but were not found in PATH.")


def get_video_duration(video_path: str) -> float:
    check_ffmpeg()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def build_proxy_clip(
    video_path: str,
    output_path: str,
    start_time: float | None,
    end_time: float | None,
    target_frames: int,
    width: int,
    crf: int,
) -> str:
    check_ffmpeg()
    out = Path(output_path)
    ensure_dir(out.parent)

    duration = get_video_duration(video_path)
    if start_time is None:
        start_time = 0.0
    if end_time is None:
        end_time = duration

    start_time = clip_float(float(start_time), 0.0, duration)
    end_time = clip_float(float(end_time), start_time + 0.05, duration)
    clip_dur = max(0.05, end_time - start_time)

    fps = max(0.1, min(8.0, target_frames / clip_dur))

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_time:.3f}",
        "-to",
        f"{end_time:.3f}",
        "-i",
        video_path,
        "-an",
        "-vf",
        f"fps={fps:.6f},scale={width}:-2:flags=bicubic",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-movflags",
        "+faststart",
        str(out),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return str(out)


def make_equal_segments(
    duration: float,
    target_segments: int,
    min_sec: float,
    max_sec: float,
) -> List[Tuple[float, float]]:
    if duration <= min_sec:
        return [(0.0, duration)]

    approx_len = duration / max(1, target_segments)
    chunk_len = max(min_sec, min(max_sec, approx_len))
    num_segments = max(1, math.ceil(duration / chunk_len))
    actual_len = duration / num_segments

    segments: List[Tuple[float, float]] = []
    for idx in range(num_segments):
        s = idx * actual_len
        e = duration if idx == num_segments - 1 else min(duration, (idx + 1) * actual_len)
        segments.append((round(s, 3), round(e, 3)))
    return segments


def split_interval_evenly(
    start_time: float,
    end_time: float,
    target_window_sec: float,
    min_window_sec: float,
    max_window_sec: float,
    max_windows: int,
) -> List[Tuple[float, float]]:
    total = max(0.05, end_time - start_time)
    if total <= max_window_sec:
        return [(round(start_time, 3), round(end_time, 3))]

    window = max(min_window_sec, min(max_window_sec, target_window_sec))
    n = max(1, math.ceil(total / window))
    n = min(n, max_windows)
    actual = total / n

    windows: List[Tuple[float, float]] = []
    for i in range(n):
        s = start_time + i * actual
        e = end_time if i == n - 1 else min(end_time, start_time + (i + 1) * actual)
        windows.append((round(s, 3), round(e, 3)))
    return windows


def expand_interval(start_time: float, end_time: float, duration: float, left_margin: float, right_margin: float) -> Tuple[float, float]:
    s = clip_float(start_time - left_margin, 0.0, duration)
    e = clip_float(end_time + right_margin, s + 0.05, duration)
    return (round(s, 3), round(e, 3))
