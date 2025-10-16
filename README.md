# EchoScript

EchoScript is a desktop transcription console that captures microphone audio, runs CPU-only OpenAI Whisper transcription, and saves enriched Markdown reports. The refactored architecture introduces modular workers, optional local LLM summarisation via Ollama, and a performance-focused pipeline while maintaining the familiar UX.

---

## Features

- Start / Pause / Resume / Stop audio capture from a chosen input device.
- Rolling int16 ring buffer (configurable 1..60 minutes) with live input level meter and elapsed time display.
- CPU-only Whisper transcription (`tiny | base | small | medium`) with language selection or auto-detect and optional translation to English.
- Segments tab with timestamped entries, totals, and language footer.
- Optional local LLM meeting-style summary powered by Ollama with configurable decode parameters (temperature, top-k, top-p, repeat penalty, max tokens, context window, stop sequences).
- Ollama model discovery (HTTP API with CLI fallback), selectable summarisation model, and cached status reporting.
- Summary tab showing the model/parameter snapshot, output text, and returned token count.
- Autosave to Markdown after each transcription plus manual `.md` export. Markdown embeds transcript, segments, and summary metadata.
- Persistent settings via `QSettings` (`org=CATIK`, `app=EchoScript`) for audio, transcription, and LLM options.
- Diagnostics dialog exposing whisper/ollama runtime information and latency metrics.

---

## Requirements

- Python 3.9+
- FFmpeg on PATH (required by `openai-whisper`).
- Optional: [Ollama](https://ollama.com) running locally for summary generation.

### Python dependencies

```
PySide6
pyaudio
numpy
torch
openai-whisper
python-docx
requests
pytest
```

Install with:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip wheel
pip install -r requirements.txt
```

For `pyaudio`, ensure PortAudio development headers are available (`sudo apt-get install portaudio19-dev`, `brew install portaudio`, or use `pipwin` on Windows).

---

## Running

```bash
python app.py
```

On first launch:

1. Select microphone, Whisper model, language (or auto), and whether to translate to English.
2. (Optional) Enable **Generate Summary**, pick an Ollama model, and adjust decode parameters.
3. Press **Start** to begin recording, **Pause/Resume** as needed, and **Stop** to transcribe.

After transcription:

- Transcript text populates the main pane, segments appear in the Segments tab, and autosave writes to `recordings/` (overridable in settings).
- If summarisation is enabled and Ollama is reachable, the Summary tab displays the generated brief along with the model/parameter snapshot and token count.
- Use **Export Markdown** for manual export.

The status bar shows `Whisper: <model> (CPU) | LLM: <model or Disabled>` and reflects buffer overruns.

---

## Ollama integration

- The UI lists installed Ollama models and exposes a **Reload** button.
- When Ollama is unavailable, summary controls automatically disable with a banner message.
- Decode parameters are persisted and embedded in Markdown exports alongside the LLM metadata.

---

## Testing

Unit tests cover the ring buffer, Markdown atomic save, settings validation, and summariser retry behaviour.

```bash
pytest
```

---

## Architecture overview

| Module | Responsibility |
| --- | --- |
| `audio_io.py` | Ring buffer, PyAudio recorder worker, device enumeration |
| `transcribe.py` | Whisper model manager and asynchronous transcription worker |
| `summarizer.py` | Ollama HTTP/CLI integration and summary generation |
| `markdown_io.py` | Markdown rendering and atomic save helper |
| `settings.py` | Typed accessors for persisted settings |
| `ui_main.py` | PySide6 MainWindow wiring, UI logic, threads |
| `app.py` | Entry point |

`tests/` holds targeted unit coverage for core utilities.

---

## Optional DOCX export

The legacy DOCX export tooling remains available via `python-docx` should you wish to extend the UI. The current release focuses on Markdown workflows.
