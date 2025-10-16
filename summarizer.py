"""Local LLM summarization via Ollama."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests


OLLAMA_URL = "http://localhost:11434"
TAGS_ENDPOINT = f"{OLLAMA_URL}/api/tags"
GENERATE_ENDPOINT = f"{OLLAMA_URL}/api/generate"


@dataclass
class LlmParams:
    temperature: float = 0.2
    top_k: int = 40
    top_p: float = 0.9
    repeat_penalty: float = 1.05
    num_predict: int = 512
    num_ctx: int = 4096
    stop: Optional[List[str]] = None

    def to_payload(self) -> Dict[str, object]:
        payload = {
            "temperature": float(self.temperature),
            "top_k": int(self.top_k),
            "top_p": float(self.top_p),
            "repeat_penalty": float(self.repeat_penalty),
            "num_predict": int(self.num_predict),
            "num_ctx": int(self.num_ctx),
        }
        if self.stop:
            payload["stop"] = self.stop
        return payload


@dataclass
class SummaryResult:
    summary_text: str
    model: str
    params: Dict[str, object]
    created_iso: str
    tokens_out: Optional[int]
    latency_sec: float


class OllamaUnavailable(Exception):
    pass


class Summarizer:
    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self._session = session or requests.Session()
        self._model_cache: List[Dict[str, object]] = []
        self._last_latency = 0.0
        self._last_error: Optional[str] = None

    @property
    def last_latency(self) -> float:
        return self._last_latency

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def list_models(self, force_refresh: bool = False) -> List[Dict[str, object]]:
        if self._model_cache and not force_refresh:
            return self._model_cache
        try:
            response = self._session.get(TAGS_ENDPOINT, timeout=2)
            response.raise_for_status()
            data = response.json()
            models = data.get("models", []) if isinstance(data, dict) else []
            self._model_cache = models
            return models
        except Exception:
            # fall back to CLI
            return self._fetch_via_cli()

    def _fetch_via_cli(self) -> List[Dict[str, object]]:
        import subprocess

        try:
            result = subprocess.run(["ollama", "list", "--format", "json"], capture_output=True, timeout=2, check=True)
            models = json.loads(result.stdout.decode("utf-8"))
            if isinstance(models, list):
                self._model_cache = models
                return models
        except Exception:
            self._last_error = "Ollama CLI not available"
        return []

    def summarize(self, text: str, params: LlmParams, model: str, language: str, translate_to_en: bool) -> SummaryResult:
        if not model:
            raise OllamaUnavailable("No model selected")
        prompt = self._build_prompt(text, language, translate_to_en)
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": params.to_payload(),
        }
        backoff = 0.2
        for attempt in range(3):
            try:
                start = time.monotonic()
                response = self._session.post(GENERATE_ENDPOINT, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                summary_text = data.get("response", "")
                tokens = data.get("eval_count") or data.get("eval_tokens")
                created = datetime.now(timezone.utc).isoformat()
                self._last_latency = time.monotonic() - start
                self._last_error = None
                return SummaryResult(
                    summary_text=summary_text.strip(),
                    model=model,
                    params=params.to_payload(),
                    created_iso=created,
                    tokens_out=tokens,
                    latency_sec=self._last_latency,
                )
            except Exception as exc:
                self._last_error = str(exc)
                time.sleep(backoff)
                backoff *= 2
        raise OllamaUnavailable(self._last_error or "Ollama summarization failed")

    def _build_prompt(self, text: str, language: str, translate_to_en: bool) -> str:
        target_language = "English" if translate_to_en else (language or "the language of the transcript")
        sections = [
            "You are EchoScript's summarizer.",
            "Create a crisp meeting-style summary with bullet points and action items.",
            "Include time-stamped highlights when timestamps are present.",
            f"Respond in {target_language}.",
            "Transcript:",
            text.strip(),
        ]
        return "\n\n".join(sections)
