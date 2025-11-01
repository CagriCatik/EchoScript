<p align="center" style="font-size:48px; font-weight:500; line-height:1.8; margin:10;">
  EchoScript
</p>

---

Desktop tool for recording microphone audio, transcribing with OpenAI Whisper on CPU, and generating clean Markdown documentation. Modular design, with a system prompt overlay for safe LLM post-processing via Ollama.

<p align="center">
  <img src="./assets/image.png" alt="alt text" style="max-width:100%; width:600px; height:auto;">
</p>

---

## What is new

* Modular architecture with clear separation of concerns.
* Documentation generation and summarization via local Ollama models.
* System prompt overlay (trusted, system-role) to harden against instruction leakage from transcripts.
* DOCX export.
* Autosave and manual save for Markdown.
* Persistent settings via QSettings.

---

## Features

Audio and transcription

* Start/Pause/Resume/Stop audio capture from a selected input device.
* CPU-only Whisper transcription: models `tiny|base|small|medium`.
* Optional translation to English (`task=translate`) or native-language transcription.
* Language selection or auto-detection.
* Live input-level meter and elapsed time display.
* Segment list with timestamps.
* Rolling capture window capped at ~10 minutes.

Authoring and export

* Autosave to Markdown after each transcription and manual `.md` export.
* Executive summary generation of the transcript.
* Full documentation generation in Markdown using Ollama.
* DOCX export of transcript or generated docs.
* Persistent settings via QSettings (model, language, translate flag, mic device, save dir, overlay policy, LLM model).

Trust and safety

* System prompt overlay applied only in the system role (trusted).
* Transcript and segments passed only as user content (untrusted).
* Overlay enforces a fixed Markdown section layout and ignores instructions inside transcript data.


---

## Requirements

* Python 3.9+ recommended.
* FFmpeg on PATH (required by `openai-whisper`).
* OS: Linux, macOS, Windows.

Optional for LLM post-processing

* Ollama running locally (default `http://localhost:11434`) with at least one chat model pulled (e.g., `llama3`, `mistral`, `qwen2.5`).

### Python dependencies

Install:

```sh
python -m venv .venv
. .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip wheel
pip install -r requirements.txt
# or:
# pip install PySide6 numpy openai-whisper pyaudio requests python-docx
```

FFmpeg

* macOS: brew install ffmpeg
* Debian/Ubuntu: sudo apt-get install ffmpeg
* Windows: download FFmpeg and add bin to PATH

PyAudio

* Linux: sudo apt-get install portaudio19-dev before pip install pyaudio
* macOS: brew install portaudio then pip install pyaudio
* Windows: if wheels fail, use: pip install pipwin && pipwin install pyaudio

---

## Run

Quick start (root launcher):

```sh
python app.py
```

Module mode (src layout):

```sh
# Unix/macOS
PYTHONPATH=src python -m echoscript.app
# Windows PowerShell
$env:PYTHONPATH="src"; python -m echoscript.app
```

First run, select:

* Whisper model
* Language (empty = auto)
* Translate to English (checkbox)
* Mic device
* Ollama model (for docs/summary), optional

Main actions:

* Start: begin recording
* Pause/Resume: toggle capture
* Stop & Transcribe: stop and run Whisper on CPU
* Generate Docs: call Ollama to produce structured Markdown
* Summarize Transcript: call Ollama for an executive summary
* Save .md: manually export Markdown
* Export .docx: export current document or transcript

Autosave path default:

```
./recordings/transcription_YYYYMMDD_HHMMSS.md
```

Generated docs default:

```
./recordings/documentation_YYYYMMDD_HHMMSS.md
```

---

## How it works

Audio capture

* Backend: PyAudio
* Format: 16-bit PCM, mono, 16 kHz
* Buffer: 1024 frames
* Rolling window: ~10 minutes; oldest audio is discarded once full

Transcription

* Library: openai-whisper
* Device: CPU (`device="cpu"`, `fp16=False`)
* Task: transcribe (default) or translate (if enabled)
* Language: explicit choice or auto (`language=None`)
* Output: transcript text pane + segment timestamps pane

Documentation and summary (Ollama)

* System prompt overlay is applied in the system role only (trusted policy).
* Transcript and segments are passed only as untrusted user content.
* The overlay enforces a fixed section order and instructs the model to ignore any instructions from the transcript.

DOCX export

* Converts current documentation (or transcript) into a `.docx` file.

---

## Architecture

Modules (under `src/echoscript`)

* `audio.py`: audio capture worker (PyAudio, QThread)
* `transcription.py`: Whisper CPU transcriber worker (QThread)
* `llm.py`: Ollama client (HTTP and CLI model discovery, chat API)
* `prompts.py`: prompt builders (system overlay, user payload, summary task)
* `exporters.py`: DOCX exporter and Markdown rendering for transcripts
* `settings.py`: QSettings keys and accessors
* `utils.py`: utilities (I/O, time labels, audio normalization)
* `ui_main.py`: Qt UI and orchestration
* `app.py`: application entry point

Separation of concerns

* UI orchestrates workers via signals/slots; no model logic in widgets.
* Audio capture and transcription are isolated QObjects in their own threads.
* LLM client is a small service; prompts are pure builders.
* Exporters handle output formats; no UI dependencies.
* Settings are centralized; modules remain stateless.

---

## System prompt overlay

UI section: "System prompt overlay"

* Enable/disable checkbox.
* Editable overlay text stored in QSettings.
* Applied only in the system role alongside a fixed baseline policy.
* Transcript and segments are always passed as untrusted user content, wrapped in literal blocks and tags.

---

## Limitations

* CPU-only Whisper; no GPU acceleration.
* Offline processing; no live streaming transcription.
* Fixed sample rate (16 kHz), mono.
* Rolling window trims to last ~10 minutes.
* Simple RMS level meter; not calibrated.
* LLM post-processing quality depends on the local model and chat template.

---

## Roadmap

* Optional GPU support (CUDA/Metal) for Whisper.
* Configurable rolling window duration.
* Live partial transcription during recording.
* VAD-based auto-pause.
* Export SRT/VTT.

---

## Security and privacy

* Audio resides in memory until transcription; the app writes only Markdown and optional DOCX files.
* No network calls beyond Whisper model download and optional local Ollama HTTP.
* The system prompt overlay enforces a fixed instruction hierarchy and prevents capability claims.
* Review and delete `./recordings/*.md` and exported `.docx` if sensitive.