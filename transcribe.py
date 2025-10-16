"""Whisper model management and transcription workers."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
try:  # pragma: no cover - optional heavy dependency
    import whisper
except ModuleNotFoundError:  # pragma: no cover
    whisper = None
from PySide6.QtCore import QObject, Signal, Slot

from audio_io import SAMPLE_RATE


class ModelManager:
    _instance: "ModelManager" | None = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init()
            return cls._instance

    def _init(self) -> None:
        self._models: Dict[str, whisper.Whisper] = {}
        self._current: Optional[str] = None
        self._model_lock = threading.Lock()

    def load(self, model_name: str) -> whisper.Whisper:
        if whisper is None:  # pragma: no cover - runtime guard
            raise RuntimeError("openai-whisper is not installed")
        with self._model_lock:
            if model_name in self._models:
                self._current = model_name
                return self._models[model_name]
            model = whisper.load_model(model_name, device="cpu")
            self._models[model_name] = model
            self._current = model_name
            return model

    @property
    def current_model_name(self) -> Optional[str]:
        return self._current

    def detect_language(self, audio_f32_mono_16k: np.ndarray) -> str:
        if whisper is None:  # pragma: no cover
            raise RuntimeError("openai-whisper is not installed")
        model = self.load(self._current or "small")
        options = whisper.DecodingOptions(task="transcribe")
        result = whisper.transcribe.transcribe(model, audio_f32_mono_16k, decode_options=options)
        return result.get("language", "")


@dataclass
class TranscriptionResult:
    text: str
    segments: List[Dict[str, float]]
    language: str
    model: str
    created_iso: str


class TranscriberWorker(QObject):
    finished = Signal(object)
    errorOccurred = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._task = "transcribe"
        self._language: Optional[str] = None
        self._manager = ModelManager()

    def set_params(self, task: str, language: Optional[str]) -> None:
        self._task = task
        self._language = language

    @Slot(np.ndarray, str)
    def transcribe(self, audio: np.ndarray, model_name: str) -> None:
        def _run() -> None:
            try:
                model = self._manager.load(model_name)
                audio_f32 = audio.astype(np.float32) * (1.0 / 32768.0)
                start = time.monotonic()
                result = model.transcribe(
                    audio_f32,
                    language=self._language if self._language and self._language != "auto" else None,
                    task=self._task,
                    fp16=False,
                    temperature=0.0,
                    condition_on_previous_text=True,
                )
                duration = time.monotonic() - start
                segments = [
                    {
                        "id": seg.get("id"),
                        "start": float(seg.get("start", 0.0)),
                        "end": float(seg.get("end", 0.0)),
                        "text": seg.get("text", ""),
                    }
                    for seg in result.get("segments", [])
                ]
                res = TranscriptionResult(
                    text=result.get("text", ""),
                    segments=segments,
                    language=result.get("language") or (self._language or ""),
                    model=model_name,
                    created_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                )
                self.finished.emit({
                    "result": res,
                    "latency": duration,
                })
            except Exception as exc:  # pragma: no cover - heavy dependency
                self.errorOccurred.emit(str(exc))
        threading.Thread(target=_run, daemon=True).start()
