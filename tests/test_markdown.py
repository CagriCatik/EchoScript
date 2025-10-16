from pathlib import Path

import pytest

from markdown_io import MarkdownRenderer
from summarizer import SummaryResult


def test_atomic_save(tmp_path: Path):
    renderer = MarkdownRenderer()
    path = tmp_path / "out.md"
    markdown = "hello"
    renderer.save_atomic(markdown, path)
    assert path.read_text() == markdown


def test_markdown_includes_summary():
    renderer = MarkdownRenderer()
    summary = SummaryResult(
        summary_text="Summary text",
        model="llama",
        params={"temperature": 0.2},
        created_iso="2024-01-01T00:00:00Z",
        tokens_out=42,
        latency_sec=1.2,
    )
    md = renderer.render(
        "Transcript body",
        [{"start": 0.0, "end": 1.0, "text": "Hello"}],
        {"title": "Test", "created": "now", "model": "small", "language": "en", "translate_to_en": False},
        summary,
    )
    assert "Summary text" in md
    assert "LLM Model" in md
