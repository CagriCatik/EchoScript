"""Application settings repository for EchoScript."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import QSettings


ORG = "CATIK"
APP = "EchoScript"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class LlmSettings:
    model: str
    temperature: float
    top_k: int
    top_p: float
    repeat_penalty: float
    num_predict: int
    num_ctx: int
    stop: str


class SettingsRepo:
    """Thin wrapper around :class:`QSettings` with typed helpers."""

    def __init__(self) -> None:
        self._settings = QSettings(ORG, APP)

    # --- core audio/transcription settings ---
    def get_microphone_index(self) -> Optional[int]:
        if not self._settings.contains("mic_index"):
            return None
        value = self._settings.value("mic_index", type=int)
        return value if value is not None and value >= 0 else None

    def set_microphone_index(self, index: Optional[int]) -> None:
        if index is None:
            self._settings.remove("mic_index")
        elif index >= 0:
            self._settings.setValue("mic_index", int(index))

    def get_whisper_model(self) -> str:
        value = self._settings.value("model", "small", type=str)
        return value or "small"

    def set_whisper_model(self, model: str) -> None:
        if model:
            self._settings.setValue("model", model)

    def get_language(self) -> str:
        return self._settings.value("language", "auto", type=str) or "auto"

    def set_language(self, language: str) -> None:
        self._settings.setValue("language", language or "auto")

    def get_translate_to_english(self) -> bool:
        return bool(self._settings.value("translate_to_en", False, type=bool))

    def set_translate_to_english(self, enabled: bool) -> None:
        self._settings.setValue("translate_to_en", bool(enabled))

    def get_save_directory(self) -> Optional[str]:
        value = self._settings.value("save_dir", type=str)
        return value if value else None

    def set_save_directory(self, path: Optional[str]) -> None:
        if path:
            self._settings.setValue("save_dir", path)
        else:
            self._settings.remove("save_dir")

    def get_buffer_minutes(self) -> int:
        value = self._settings.value("buffer_minutes", 10, type=int)
        return int(_clamp(value if value else 10, 1, 60))

    def set_buffer_minutes(self, minutes: int) -> None:
        minutes = int(_clamp(minutes or 10, 1, 60))
        self._settings.setValue("buffer_minutes", minutes)

    # --- LLM settings ---
    def get_llm_settings(self) -> LlmSettings:
        model = self._settings.value("llm_model", "", type=str) or ""
        temperature = float(_clamp(self._settings.value("llm_temperature", 0.2, type=float), 0.0, 2.0))
        top_k = max(1, int(self._settings.value("llm_top_k", 40, type=int)))
        top_p = float(_clamp(self._settings.value("llm_top_p", 0.9, type=float), 0.0, 1.0))
        repeat_penalty = float(_clamp(self._settings.value("llm_repeat_penalty", 1.05, type=float), 0.0, 10.0))
        num_predict = max(16, int(self._settings.value("llm_num_predict", 512, type=int)))
        num_ctx = max(128, int(self._settings.value("llm_num_ctx", 4096, type=int)))
        stop = self._settings.value("llm_stop", "", type=str) or ""
        return LlmSettings(
            model=model,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            num_predict=num_predict,
            num_ctx=num_ctx,
            stop=stop,
        )

    def set_llm_settings(self, settings: LlmSettings) -> None:
        self._settings.setValue("llm_model", settings.model)
        self._settings.setValue("llm_temperature", float(_clamp(settings.temperature, 0.0, 2.0)))
        self._settings.setValue("llm_top_k", max(1, int(settings.top_k)))
        self._settings.setValue("llm_top_p", float(_clamp(settings.top_p, 0.0, 1.0)))
        self._settings.setValue("llm_repeat_penalty", float(_clamp(settings.repeat_penalty, 0.0, 10.0)))
        self._settings.setValue("llm_num_predict", max(16, int(settings.num_predict)))
        self._settings.setValue("llm_num_ctx", max(128, int(settings.num_ctx)))
        self._settings.setValue("llm_stop", settings.stop or "")

    def sync(self) -> None:
        self._settings.sync()
