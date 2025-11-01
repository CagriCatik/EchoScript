from docx import Document
from PySide6.QtCore import QObject, Signal, Slot

class DocxExporter(QObject):
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, content: str, out_path: str):
        super().__init__()
        self.content = content
        self.out_path = out_path

    @Slot()
    def run(self):
        try:
            doc = Document()
            for line in self.content.splitlines():
                doc.add_paragraph(line)
            doc.save(self.out_path)
            self.finished.emit(self.out_path)
        except Exception as e:
            self.error.emit(str(e))

def render_markdown_document(text: str, result: dict, meta: dict) -> str:
    title = meta.get("title", "Transcription")
    model = meta.get("model", "")
    lang = meta.get("language", "")
    translate = "yes" if meta.get("translate_to_en") else "no"
    created = meta.get("created", "")
    segs = result.get("segments", []) if isinstance(result, dict) else []
    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Meta")
    lines.append("")
    lines.append(f"- Created: {created}")
    if model:
        lines.append(f"- Model: {model}")
    lines.append(f"- Language: {lang or 'auto'}")
    lines.append(f"- Translate to English: {translate}")
    lines.append("")
    lines.append("## Transcript")
    lines.append("")
    lines.append((text or "").strip())
    if segs:
        lines.append("")
        lines.append("## Segments")
        lines.append("")
        for i, seg in enumerate(segs, start=1):
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            seg_text = (seg.get("text") or "").strip()
            lines.append(f"{i}. [{start:.2f} -> {end:.2f}] {seg_text}")
    return "\n".join(lines) + "\n"
