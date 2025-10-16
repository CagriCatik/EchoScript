"""Markdown rendering and atomic save helpers for EchoScript."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from summarizer import SummaryResult


class MarkdownRenderer:
    def render(
        self,
        text: str,
        segments: List[Dict[str, object]],
        meta: Dict[str, str],
        summary: Optional[SummaryResult],
    ) -> str:
        lines: List[str] = []
        title = meta.get("title", "Transcription")
        lines.append(f"# {title}")
        lines.append("")
        lines.append("## Meta")
        lines.append("")
        meta_items = {
            "Created": meta.get("created", ""),
            "Whisper Model": meta.get("model", ""),
            "Language": meta.get("language", ""),
            "Translate to English": "yes" if meta.get("translate_to_en") else "no",
        }
        if summary:
            meta_items["Summary LLM"] = summary.model
        for key, value in meta_items.items():
            if value is not None and value != "":
                lines.append(f"- {key}: {value}")
        lines.append("")
        if summary:
            lines.append("## Summary")
            lines.append("")
            lines.append(summary.summary_text)
            lines.append("")
            lines.append("### Summary Meta")
            lines.append("")
            lines.append(f"- LLM Model: {summary.model}")
            param_line = ", ".join(
                f"{k}={v}" for k, v in summary.params.items() if k != "stop"
            )
            lines.append(f"- LLM Params: {param_line}")
            if summary.params.get("stop"):
                lines.append(f"- Stop sequences: {', '.join(summary.params['stop'])}")
            if summary.tokens_out is not None:
                lines.append(f"- Tokens (output): {summary.tokens_out}")
            lines.append(f"- Summary created: {summary.created_iso}")
            lines.append("")
        lines.append("## Transcript")
        lines.append("")
        lines.append(text.strip())
        if segments:
            lines.append("")
            lines.append("## Segments")
            lines.append("")
            for idx, seg in enumerate(segments, 1):
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", 0.0))
                seg_text = (seg.get("text") or "").strip()
                lines.append(f"{idx}. [{start:.2f} -> {end:.2f}] {seg_text}")
        return "\n".join(lines).strip() + "\n"

    def save_atomic(self, markdown: str, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(markdown, encoding="utf-8")
        tmp_path.replace(path)
        return path
