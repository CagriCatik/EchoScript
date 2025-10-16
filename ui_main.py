"""PySide6 user interface wiring for EchoScript."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Q_ARG, QMetaObject, QObject, QSize, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from audio_io import AudioRecorderWorker, RingBuffer, list_input_devices
from markdown_io import MarkdownRenderer
from settings import SettingsRepo
from summarizer import LlmParams, OllamaUnavailable, Summarizer, SummaryResult
from transcribe import TranscriberWorker

WHISPER_MODELS = ["tiny", "base", "small", "medium"]


class SummaryWorker(QObject):
    finished = Signal(object)
    errorOccurred = Signal(str)

    def __init__(self, summarizer: Summarizer) -> None:
        super().__init__()
        self._summarizer = summarizer

    @Slot(str, dict)
    def summarize(self, text: str, payload: dict) -> None:
        try:
            params = payload["params"]
            llm_params = LlmParams(
                temperature=params["temperature"],
                top_k=params["top_k"],
                top_p=params["top_p"],
                repeat_penalty=params["repeat_penalty"],
                num_predict=params["num_predict"],
                num_ctx=params["num_ctx"],
                stop=params.get("stop"),
            )
            result = self._summarizer.summarize(
                text,
                llm_params,
                payload["model"],
                payload["language"],
                payload["translate_to_en"],
            )
            self.finished.emit(result)
        except Exception as exc:
            self.errorOccurred.emit(str(exc))


class DiagnosticsDialog(QDialog):
    def __init__(self, parent: QWidget, info_provider) -> None:
        super().__init__(parent)
        self.setWindowTitle("Diagnostics")
        layout = QFormLayout(self)
        self._info_provider = info_provider
        self._labels: Dict[str, QLabel] = {}
        fields = [
            "Whisper Model",
            "Sample Rate",
            "Chunk Size",
            "Buffer Duration",
            "Overruns",
            "Ollama Reachable",
            "Ollama Models",
            "Summary Latency",
            "Last Summary Error",
        ]
        for field in fields:
            label = QLabel("-")
            layout.addRow(f"{field}:", label)
            self._labels[field] = label
        self.refresh()

    def refresh(self) -> None:
        info = self._info_provider()
        for key, value in info.items():
            if key in self._labels:
                self._labels[key].setText(str(value))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("EchoScript")
        self.settings = SettingsRepo()
        self.renderer = MarkdownRenderer()
        self.summarizer = Summarizer()

        self.audio_thread: Optional[QThread] = None
        self.audio_worker: Optional[AudioRecorderWorker] = None
        self.transcriber_thread: Optional[QThread] = None
        self.transcriber_worker = TranscriberWorker()
        self.summary_thread: Optional[QThread] = None
        self.summary_worker = SummaryWorker(self.summarizer)

        self._ring_buffer: Optional[RingBuffer] = None
        self._start_time = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_elapsed)
        self._current_transcript: Optional[str] = None
        self._current_segments: List[Dict[str, object]] = []
        self._current_summary: Optional[SummaryResult] = None
        self._last_transcription_meta: Dict[str, str] = {}

        self._setup_ui()
        self._connect_workers()
        self._load_initial_settings()
        self._refresh_ollama_models()
        self._update_status_bar()

    def _setup_ui(self) -> None:
        central = QWidget()
        main_layout = QVBoxLayout(central)

        control_bar = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.pause_btn = QPushButton("Pause")
        self.resume_btn = QPushButton("Resume")
        self.stop_btn = QPushButton("Stop")
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        control_bar.addWidget(self.start_btn)
        control_bar.addWidget(self.pause_btn)
        control_bar.addWidget(self.resume_btn)
        control_bar.addWidget(self.stop_btn)

        self.device_combo = QComboBox()
        self.device_combo.addItems(list_input_devices())
        control_bar.addWidget(QLabel("Mic"))
        control_bar.addWidget(self.device_combo)

        self.model_combo = QComboBox()
        self.model_combo.addItems(WHISPER_MODELS)
        control_bar.addWidget(QLabel("Whisper"))
        control_bar.addWidget(self.model_combo)

        self.language_combo = QComboBox()
        languages = ["auto", "en", "es", "fr", "de", "it", "zh"]
        self.language_combo.addItems(languages)
        control_bar.addWidget(QLabel("Language"))
        control_bar.addWidget(self.language_combo)

        self.translate_checkbox = QCheckBox("Translate to English")
        control_bar.addWidget(self.translate_checkbox)

        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setMaximumWidth(120)
        control_bar.addWidget(QLabel("Level"))
        control_bar.addWidget(self.level_bar)

        self.elapsed_label = QLabel("00:00")
        control_bar.addWidget(QLabel("Elapsed"))
        control_bar.addWidget(self.elapsed_label)

        self.buffer_spin = QSpinBox()
        self.buffer_spin.setRange(1, 60)
        self.buffer_spin.setSuffix(" min")
        control_bar.addWidget(QLabel("Window"))
        control_bar.addWidget(self.buffer_spin)

        main_layout.addLayout(control_bar)

        llm_bar = QHBoxLayout()
        self.summary_checkbox = QCheckBox("Generate Summary")
        llm_bar.addWidget(self.summary_checkbox)
        self.llm_model_combo = QComboBox()
        llm_bar.addWidget(QLabel("LLM Model"))
        llm_bar.addWidget(self.llm_model_combo)
        self.reload_models_btn = QPushButton("Reload")
        llm_bar.addWidget(self.reload_models_btn)

        self.llm_params_group = QGroupBox("LLM Params")
        params_layout = QHBoxLayout()
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setToolTip("Sampling temperature")
        params_layout.addWidget(QLabel("T"))
        params_layout.addWidget(self.temp_spin)

        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(1, 2000)
        params_layout.addWidget(QLabel("top_k"))
        params_layout.addWidget(self.topk_spin)

        self.topp_spin = QDoubleSpinBox()
        self.topp_spin.setRange(0.0, 1.0)
        self.topp_spin.setSingleStep(0.05)
        params_layout.addWidget(QLabel("top_p"))
        params_layout.addWidget(self.topp_spin)

        self.repeat_spin = QDoubleSpinBox()
        self.repeat_spin.setRange(0.5, 10.0)
        self.repeat_spin.setSingleStep(0.05)
        params_layout.addWidget(QLabel("repeat"))
        params_layout.addWidget(self.repeat_spin)

        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(16, 8192)
        params_layout.addWidget(QLabel("max"))
        params_layout.addWidget(self.max_tokens_spin)

        self.ctx_spin = QSpinBox()
        self.ctx_spin.setRange(128, 32768)
        params_layout.addWidget(QLabel("ctx"))
        params_layout.addWidget(self.ctx_spin)

        self.stop_sequences = QLineEdit()
        self.stop_sequences.setPlaceholderText("comma separated stop tokens")
        params_layout.addWidget(QLabel("stop"))
        params_layout.addWidget(self.stop_sequences)
        self.llm_params_group.setLayout(params_layout)
        llm_bar.addWidget(self.llm_params_group)
        main_layout.addLayout(llm_bar)

        splitter = QSplitter()
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.transcript_edit = QTextEdit()
        left_layout.addWidget(QLabel("Transcript"))
        left_layout.addWidget(self.transcript_edit)
        export_btn = QPushButton("Export Markdown")
        left_layout.addWidget(export_btn)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        self.ollama_list = QListWidget()
        self.ollama_list.setSelectionMode(QListWidget.SingleSelection)
        right_layout.addWidget(QLabel("Ollama Models Installed"))
        right_layout.addWidget(self.ollama_list)

        self.tabs = QTabWidget()
        self.segments_view = QTextEdit()
        self.segments_view.setReadOnly(True)
        segments_tab = QWidget()
        seg_layout = QVBoxLayout(segments_tab)
        seg_layout.addWidget(self.segments_view)
        self.segments_footer = QLabel("Segments: 0")
        seg_layout.addWidget(self.segments_footer)

        self.summary_view = QTextEdit()
        self.summary_view.setReadOnly(True)
        summary_tab = QWidget()
        sum_layout = QVBoxLayout(summary_tab)
        sum_layout.addWidget(self.summary_view)
        self.summary_footer = QLabel("Tokens: -")
        sum_layout.addWidget(self.summary_footer)

        self.tabs.addTab(segments_tab, "Segments")
        self.tabs.addTab(summary_tab, "Summary")
        self.tabs.setTabEnabled(1, False)
        right_layout.addWidget(self.tabs)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        main_layout.addWidget(splitter)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.setCentralWidget(central)

        self.export_btn = export_btn

        self.start_btn.clicked.connect(self.start_recording)
        self.pause_btn.clicked.connect(self.pause_recording)
        self.resume_btn.clicked.connect(self.resume_recording)
        self.stop_btn.clicked.connect(self.stop_recording)
        self.reload_models_btn.clicked.connect(lambda: self._refresh_ollama_models(force=True))
        self.export_btn.clicked.connect(self._export_markdown)
        self.summary_checkbox.toggled.connect(self._on_summary_toggle)
        self.llm_model_combo.currentTextChanged.connect(self._persist_llm_model)
        self.device_combo.currentIndexChanged.connect(self._on_device_change)
        self.model_combo.currentTextChanged.connect(self._on_whisper_model_change)
        self.buffer_spin.valueChanged.connect(self._on_buffer_minutes_change)

        diag_action = QAction("Diagnostics", self)
        diag_action.triggered.connect(self._show_diagnostics)
        self.menuBar().addAction(diag_action)

    def _connect_workers(self) -> None:
        self.audio_thread = QThread(self)
        self.audio_worker = AudioRecorderWorker(self.settings.get_microphone_index())
        self.audio_worker.moveToThread(self.audio_thread)
        self.audio_worker.update_buffer_minutes(self.settings.get_buffer_minutes())
        self.audio_thread.start()
        self.audio_worker.levelChanged.connect(self.level_bar.setValue)
        self.audio_worker.chunkCaptured.connect(self._on_chunk)
        self.audio_worker.errorOccurred.connect(self._on_audio_error)
        self.audio_worker.statsChanged.connect(self._on_stats_change)
        self.audio_worker.stopped.connect(self._on_audio_stopped)

        self.summary_thread = QThread(self)
        self.summary_worker.moveToThread(self.summary_thread)
        self.summary_thread.start()
        self.summary_worker.finished.connect(self._on_summary_ready)
        self.summary_worker.errorOccurred.connect(self._on_summary_error)

        self.transcriber_thread = QThread(self)
        self.transcriber_worker.moveToThread(self.transcriber_thread)
        self.transcriber_thread.start()
        self.transcriber_worker.finished.connect(self._on_transcription_ready)
        self.transcriber_worker.errorOccurred.connect(self._on_transcription_error)

    def _load_initial_settings(self) -> None:
        llm_settings = self.settings.get_llm_settings()
        if llm_settings.model:
            self.llm_model_combo.addItem(llm_settings.model)
            self.llm_model_combo.setCurrentText(llm_settings.model)
        self.temp_spin.setValue(llm_settings.temperature)
        self.topk_spin.setValue(llm_settings.top_k)
        self.topp_spin.setValue(llm_settings.top_p)
        self.repeat_spin.setValue(llm_settings.repeat_penalty)
        self.max_tokens_spin.setValue(llm_settings.num_predict)
        self.ctx_spin.setValue(llm_settings.num_ctx)
        self.stop_sequences.setText(llm_settings.stop)

        whisper_model = self.settings.get_whisper_model()
        idx = self.model_combo.findText(whisper_model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)

        language = self.settings.get_language()
        idx = self.language_combo.findText(language)
        if idx >= 0:
            self.language_combo.setCurrentIndex(idx)
        self.translate_checkbox.setChecked(self.settings.get_translate_to_english())
        self.buffer_spin.setValue(self.settings.get_buffer_minutes())

        mic_idx = self.settings.get_microphone_index()
        if mic_idx is not None:
            for i in range(self.device_combo.count()):
                if self.device_combo.itemText(i).startswith(f"{mic_idx}:"):
                    self.device_combo.setCurrentIndex(i)
                    break

    def _on_summary_toggle(self, checked: bool) -> None:
        self.tabs.setTabEnabled(1, checked)
        self._update_status_bar()

    def _persist_llm_model(self) -> None:
        settings = self.settings.get_llm_settings()
        settings.model = self.llm_model_combo.currentText()
        self.settings.set_llm_settings(settings)

    def _on_device_change(self, idx: int) -> None:
        text = self.device_combo.itemText(idx)
        if ":" in text:
            device_index = int(text.split(":", 1)[0])
            self.settings.set_microphone_index(device_index)
            if self.audio_worker:
                self.audio_worker.set_device_index(device_index)

    def _on_whisper_model_change(self, model: str) -> None:
        self.settings.set_whisper_model(model)
        self._update_status_bar()

    def _on_buffer_minutes_change(self, value: int) -> None:
        self.settings.set_buffer_minutes(value)
        if self.audio_worker:
            self.audio_worker.update_buffer_minutes(value)
            self._ring_buffer = self.audio_worker.ring_buffer()

    def start_recording(self) -> None:
        if not self.audio_thread or not self.audio_worker:
            return
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.resume_btn.setEnabled(False)
        self._ring_buffer = self.audio_worker.ring_buffer()
        self._ring_buffer.clear()
        self._start_time = time.monotonic()
        self._timer.start(100)
        QMetaObject.invokeMethod(self.audio_worker, "start", Qt.QueuedConnection)

    def pause_recording(self) -> None:
        if self.audio_worker:
            QMetaObject.invokeMethod(self.audio_worker, "pause", Qt.QueuedConnection)
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(True)

    def resume_recording(self) -> None:
        if self.audio_worker:
            QMetaObject.invokeMethod(self.audio_worker, "resume", Qt.QueuedConnection)
        self.pause_btn.setEnabled(True)
        self.resume_btn.setEnabled(False)

    def stop_recording(self) -> None:
        if self.audio_worker:
            QMetaObject.invokeMethod(self.audio_worker, "stop", Qt.QueuedConnection)
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self._timer.stop()
        self.elapsed_label.setText("00:00")
        if self._ring_buffer:
            audio = self._ring_buffer.read_last(self._ring_buffer.duration_sec)
            if audio.size:
                language = self.language_combo.currentText()
                translate = self.translate_checkbox.isChecked()
                task = "translate" if translate else "transcribe"
                self.transcriber_worker.set_params(task=task, language=language)
                self.transcriber_worker.transcribe(audio, self.model_combo.currentText())

    def _on_chunk(self, frames: np.ndarray) -> None:
        self._current_transcript = None

    def _update_elapsed(self) -> None:
        if not self._start_time:
            return
        elapsed = time.monotonic() - self._start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        self.elapsed_label.setText(f"{minutes:02d}:{seconds:02d}")

    def _on_audio_error(self, message: str) -> None:
        self.status_bar.showMessage(message, 5000)

    def _on_audio_stopped(self) -> None:
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)

    def _on_stats_change(self, stats) -> None:
        self.status_bar.showMessage(
            f"Overruns: {stats.overruns} | Whisper: {self.model_combo.currentText()} (CPU) | "
            f"LLM: {self.llm_model_combo.currentText() or 'Disabled'}",
            3000,
        )

    def _on_transcription_ready(self, payload: dict) -> None:
        result = payload["result"]
        self._current_transcript = result.text
        self._current_segments = result.segments
        self.transcript_edit.setPlainText(result.text)
        segments_lines = [
            f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}" for seg in result.segments
        ]
        self.segments_view.setPlainText("\n".join(segments_lines))
        self.segments_footer.setText(f"Segments: {len(result.segments)} | Language: {result.language}")
        self._last_transcription_meta = {
            "title": "EchoScript Transcript",
            "created": result.created_iso,
            "model": result.model,
            "language": result.language,
            "translate_to_en": self.translate_checkbox.isChecked(),
        }
        self._autosave_markdown()
        if self.summary_checkbox.isChecked():
            self._trigger_summary(result.text, result.language)

    def _on_transcription_error(self, message: str) -> None:
        QMessageBox.critical(self, "Transcription Error", message)

    def _trigger_summary(self, text: str, language: str) -> None:
        settings = self.settings.get_llm_settings()
        settings.model = self.llm_model_combo.currentText()
        settings.temperature = self.temp_spin.value()
        settings.top_k = self.topk_spin.value()
        settings.top_p = self.topp_spin.value()
        settings.repeat_penalty = self.repeat_spin.value()
        settings.num_predict = self.max_tokens_spin.value()
        settings.num_ctx = self.ctx_spin.value()
        settings.stop = self.stop_sequences.text()
        self.settings.set_llm_settings(settings)
        params = {
            "model": settings.model,
            "params": {
                "temperature": settings.temperature,
                "top_k": settings.top_k,
                "top_p": settings.top_p,
                "repeat_penalty": settings.repeat_penalty,
                "num_predict": settings.num_predict,
                "num_ctx": settings.num_ctx,
                "stop": [s.strip() for s in settings.stop.split(",") if s.strip()] if settings.stop else None,
            },
            "language": language,
            "translate_to_en": self.translate_checkbox.isChecked(),
        }
        if self.summary_worker:
            QMetaObject.invokeMethod(
                self.summary_worker,
                "summarize",
                Qt.QueuedConnection,
                Q_ARG(str, text),
                Q_ARG(dict, params),
            )

    def _on_summary_ready(self, result: SummaryResult) -> None:
        self._current_summary = result
        params_str = ", ".join(f"{k}={v}" for k, v in result.params.items() if k != "stop")
        if result.params.get("stop"):
            params_str += f", stop={result.params['stop']}"
        header = f"Model: {result.model}, {params_str}"
        self.summary_view.setPlainText(f"{header}\n\n{result.summary_text}")
        tokens = result.tokens_out if result.tokens_out is not None else "-"
        self.summary_footer.setText(f"Tokens: {tokens}")
        self.tabs.setCurrentIndex(1)
        self._autosave_markdown()
        self._update_status_bar()

    def _on_summary_error(self, message: str) -> None:
        self.summary_view.setPlainText(f"Summary failed: {message}")
        self._current_summary = None
        self.summary_footer.setText("Tokens: -")
        self._update_status_bar()

    def _autosave_markdown(self) -> None:
        if not self._current_transcript or not self._last_transcription_meta:
            return
        summary = self._current_summary
        markdown = self.renderer.render(
            self._current_transcript,
            self._current_segments,
            self._last_transcription_meta,
            summary,
        )
        save_dir = self.settings.get_save_directory() or str(Path.cwd() / "recordings")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        filename = f"transcript_{int(time.time())}.md"
        path = Path(save_dir) / filename
        self.renderer.save_atomic(markdown, path)
        self.status_bar.showMessage(f"Autosaved to {path}", 5000)

    def _export_markdown(self) -> None:
        if not self._current_transcript:
            QMessageBox.information(self, "Nothing to export", "No transcript available")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Markdown", filter="Markdown (*.md)")
        if not path:
            return
        markdown = self.renderer.render(
            self._current_transcript,
            self._current_segments,
            self._last_transcription_meta,
            self._current_summary,
        )
        self.renderer.save_atomic(markdown, Path(path))
        QMessageBox.information(self, "Exported", f"Saved to {path}")

    def _refresh_ollama_models(self, force: bool = False) -> None:
        models = self.summarizer.list_models(force_refresh=force)
        self.ollama_list.clear()
        for model in models:
            name = model.get("name") or model.get("model") or "unknown"
            size = model.get("size", "")
            item = QListWidgetItem(f"{name} {size}")
            self.ollama_list.addItem(item)
            if name not in [self.llm_model_combo.itemText(i) for i in range(self.llm_model_combo.count())]:
                self.llm_model_combo.addItem(name)
        if models:
            self.status_bar.showMessage(f"Ollama models loaded: {len(models)}", 3000)
            self.summary_checkbox.setEnabled(True)
        else:
            self.status_bar.showMessage("Ollama unavailable", 3000)
            self.summary_checkbox.setChecked(False)
            self.summary_checkbox.setEnabled(False)

    def _show_diagnostics(self) -> None:
        dialog = DiagnosticsDialog(self, self._diagnostic_info)
        dialog.exec()

    def _diagnostic_info(self) -> Dict[str, str]:
        buffer_dur = 0.0
        overruns = 0
        if self.audio_worker:
            ring = self.audio_worker.ring_buffer()
            buffer_dur = ring.duration_sec
            overruns = ring.overrun_count
        info = {
            "Whisper Model": self.model_combo.currentText(),
            "Sample Rate": "16000",
            "Chunk Size": str(self.audio_worker._chunk_frames if self.audio_worker else 1024),
            "Buffer Duration": f"{buffer_dur:.1f}s",
            "Overruns": str(overruns),
            "Ollama Reachable": "yes" if self.ollama_list.count() else "no",
            "Ollama Models": str(self.ollama_list.count()),
            "Summary Latency": f"{self.summarizer.last_latency:.2f}s",
            "Last Summary Error": self.summarizer.last_error or "",
        }
        return info

    def _update_status_bar(self) -> None:
        llm_status = self.llm_model_combo.currentText() if self.summary_checkbox.isChecked() else "Disabled"
        self.status_bar.showMessage(
            f"Whisper: {self.model_combo.currentText()} (CPU) | LLM: {llm_status}",
            5000,
        )


def run_app() -> None:
    app = QApplication([])
    window = MainWindow()
    window.resize(QSize(1200, 700))
    window.show()
    app.exec()
