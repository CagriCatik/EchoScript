"""Subtitle export utilities for EchoScript transcripts."""
from __future__ import annotations

import math
import os
import re
from collections.abc import Sequence
from typing import List, Tuple

__all__ = [
    "to_srt",
    "to_vtt",
    "write_srt",
    "write_vtt",
]

_WHITESPACE_RE = re.compile(r"\s+")


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return ""

    lines: List[str] = []
    for line in normalized.split("\n"):
        stripped = line.strip()
        if stripped:
            lines.append(_WHITESPACE_RE.sub(" ", stripped))
        else:
            lines.append("")

    return "\n".join(lines)


def _seconds_to_milliseconds(value: float) -> int:
    return int(math.floor(value * 1000.0 + 0.5))


def _format_timestamp(milliseconds: int, separator: str) -> str:
    seconds, millis = divmod(milliseconds, 1000)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{millis:03d}"


def _prepare_segments(segments: Sequence[dict]) -> List[Tuple[float, float, str]]:
    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
        raise ValueError("Segments must be a sequence of dictionaries")

    prepared: List[Tuple[float, float, str]] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"Segment {index}: expected dict, got {type(segment).__name__}")

        for key in ("start", "end", "text"):
            if key not in segment:
                raise ValueError(f"Segment {index}: missing key '{key}'")

        start_raw = segment["start"]
        end_raw = segment["end"]
        text_raw = segment["text"]

        if not _is_number(start_raw):
            raise ValueError(f"Segment {index}: 'start' must be a number")
        if not _is_number(end_raw):
            raise ValueError(f"Segment {index}: 'end' must be a number")

        start = float(start_raw)
        end = float(end_raw)

        if not math.isfinite(start):
            raise ValueError(f"Segment {index}: 'start' must be finite")
        if not math.isfinite(end):
            raise ValueError(f"Segment {index}: 'end' must be finite")

        if start < 0.0:
            raise ValueError(f"Segment {index}: 'start' must be >= 0")
        if end < start:
            raise ValueError(f"Segment {index}: 'end' must be >= 'start'")

        if not isinstance(text_raw, str):
            raise ValueError(f"Segment {index}: 'text' must be a string")

        text = _normalize_text(text_raw)
        if not text:
            continue
        if end <= start:
            continue

        prepared.append((start, end, text))

    prepared.sort(key=lambda item: (item[0], item[1]))
    return prepared


def to_srt(segments: Sequence[dict]) -> str:
    """Convert transcript segments into an SRT document.

    Args:
        segments: A sequence of segment dictionaries from Whisper-like output.

    Returns:
        The formatted SRT document as a Unicode string.

    Raises:
        ValueError: If any segment is invalid.
    """

    prepared = _prepare_segments(segments)
    lines: List[str] = []

    for cue_number, (start, end, text) in enumerate(prepared, start=1):
        start_ms = _seconds_to_milliseconds(start)
        end_ms = _seconds_to_milliseconds(end)
        if end_ms < start_ms:
            end_ms = start_ms

        lines.append(str(cue_number))
        lines.append(
            f"{_format_timestamp(start_ms, ',')} --> {_format_timestamp(end_ms, ',')}"
        )
        lines.extend(text.split("\n"))
        lines.append("")

    return "\n".join(lines) + ("\n" if lines else "")


def to_vtt(segments: Sequence[dict]) -> str:
    """Convert transcript segments into a WebVTT document.

    Args:
        segments: A sequence of segment dictionaries from Whisper-like output.

    Returns:
        The formatted WebVTT document as a Unicode string.

    Raises:
        ValueError: If any segment is invalid.
    """

    prepared = _prepare_segments(segments)
    lines: List[str] = ["WEBVTT", ""]

    for start, end, text in prepared:
        start_ms = _seconds_to_milliseconds(start)
        end_ms = _seconds_to_milliseconds(end)
        if end_ms < start_ms:
            end_ms = start_ms

        lines.append(
            f"{_format_timestamp(start_ms, '.')} --> {_format_timestamp(end_ms, '.')}"
        )
        lines.extend(text.split("\n"))
        lines.append("")

    result = "\n".join(lines)
    if lines:
        result += "\n"
    return result


def write_srt(segments: Sequence[dict], path: str) -> None:
    """Write transcript segments to an SRT file.

    Args:
        segments: A sequence of segment dictionaries from Whisper-like output.
        path: Destination file path.

    Raises:
        ValueError: If any segment is invalid.
        OSError: If writing to ``path`` fails.
    """

    content = to_srt(segments)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="\n") as file:
        file.write(content)


def write_vtt(segments: Sequence[dict], path: str) -> None:
    """Write transcript segments to a WebVTT file.

    Args:
        segments: A sequence of segment dictionaries from Whisper-like output.
        path: Destination file path.

    Raises:
        ValueError: If any segment is invalid.
        OSError: If writing to ``path`` fails.
    """

    content = to_vtt(segments)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="\n") as file:
        file.write(content)
