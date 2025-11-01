import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from echoscript.exporters import to_srt, to_vtt, write_srt, write_vtt  # noqa: E402


def test_to_srt_basic():
    segments = [
        {"start": 0.0, "end": 1.2, "text": " Hello world "},
        {"start": 1.3, "end": 2.6, "text": "Second line"},
    ]

    expected = (
        "1\n"
        "00:00:00,000 --> 00:00:01,200\n"
        "Hello world\n"
        "\n"
        "2\n"
        "00:00:01,300 --> 00:00:02,600\n"
        "Second line\n"
        "\n"
    )

    assert to_srt(segments) == expected


def test_to_vtt_basic():
    segments = [
        {"start": 0.0, "end": 1.2, "text": " Hello world "},
        {"start": 1.3, "end": 2.6, "text": "Second line"},
    ]

    expected = (
        "WEBVTT\n"
        "\n"
        "00:00:00.000 --> 00:00:01.200\n"
        "Hello world\n"
        "\n"
        "00:00:01.300 --> 00:00:02.600\n"
        "Second line\n"
        "\n"
    )

    assert to_vtt(segments) == expected


def test_segments_are_sorted_by_start_then_end():
    segments = [
        {"start": 2.0, "end": 3.0, "text": "Third"},
        {"start": 1.0, "end": 1.5, "text": "Second"},
        {"start": 0.0, "end": 0.5, "text": "First"},
        {"start": 1.0, "end": 1.2, "text": "Second earlier"},
    ]

    result = to_srt(segments)
    cues = [block.splitlines() for block in result.strip().split("\n\n")]
    texts = [cue[2] for cue in cues]
    assert texts == ["First", "Second earlier", "Second", "Third"]


def test_rounding_clamps_end():
    segments = [{"start": 0.0004, "end": 0.0005, "text": "Ping"}]
    lines = to_srt(segments).splitlines()
    assert lines[1] == "00:00:00,000 --> 00:00:00,001"


def test_text_normalization_rules():
    segment = {
        "start": 0.0,
        "end": 2.0,
        "text": "  Hello   world  \r\nSecond\t\tline \n\nThird  line   ",
    }

    result = to_srt([segment])
    assert "Hello world\nSecond line\n\nThird line" in result


def test_omits_empty_or_zero_length_segments():
    segments = [
        {"start": 0.0, "end": 0.0, "text": "Drop"},
        {"start": 1.0, "end": 2.0, "text": "   "},
        {"start": 2.0, "end": 3.5, "text": "Keep"},
    ]

    result = to_srt(segments)
    assert "Keep" in result
    assert "Drop" not in result
    assert result.startswith("1\n")
    assert result.count("\n\n") == 1  # Only one cue


@pytest.mark.parametrize(
    "segments, message",
    [
        ([{"end": 1.0, "text": "a"}], "missing key 'start'"),
        ([{"start": "0", "end": 1.0, "text": "a"}], "'start' must be a number"),
        ([{"start": 0.0, "end": "1", "text": "a"}], "'end' must be a number"),
        ([{"start": float('nan'), "end": 1.0, "text": "a"}], "'start' must be finite"),
        ([{"start": 0.0, "end": float('inf'), "text": "a"}], "'end' must be finite"),
        ([{"start": -0.1, "end": 1.0, "text": "a"}], "'start' must be >= 0"),
        ([{"start": 1.0, "end": 0.9, "text": "a"}], "'end' must be >= 'start'"),
        ([{"start": 0.0, "end": 1.0, "text": 123}], "'text' must be a string"),
        (["not a dict"], "expected dict"),
    ],
)
def test_validation_errors(segments, message):
    with pytest.raises(ValueError, match=message):
        to_srt(segments)


def test_write_functions_create_directories(tmp_path):
    segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]

    srt_path = tmp_path / "nested" / "output.srt"
    vtt_path = tmp_path / "other" / "output.vtt"

    write_srt(segments, srt_path.as_posix())
    write_vtt(segments, vtt_path.as_posix())

    srt_content = srt_path.read_text(encoding="utf-8")
    vtt_content = vtt_path.read_text(encoding="utf-8")

    assert srt_content.endswith("\n")
    assert vtt_content.endswith("\n")
    assert "WEBVTT" in vtt_content


def test_formatted_times_never_invert():
    segments = []
    for base in [0.0, 0.0004, 1.2345, 3599.9994]:
        for duration in [0.0002, 0.0006, 1.0, 12.345]:
            segments.append(
                {
                    "start": base,
                    "end": base + duration,
                    "text": "Sample",
                }
            )

    result = to_srt(segments)
    pattern = re.compile(
        r"(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})"
    )

    def to_millis(match: re.Match[str]) -> tuple[int, int]:
        start_h, start_m, start_s, start_ms, end_h, end_m, end_s, end_ms = match.groups()
        start_total = (
            int(start_h) * 3600000
            + int(start_m) * 60000
            + int(start_s) * 1000
            + int(start_ms)
        )
        end_total = (
            int(end_h) * 3600000
            + int(end_m) * 60000
            + int(end_s) * 1000
            + int(end_ms)
        )
        return start_total, end_total

    for match in pattern.finditer(result):
        start_ms, end_ms = to_millis(match)
        assert start_ms <= end_ms
