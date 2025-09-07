# main.py
# PySide6 GUI: real-time capture -> Whisper transcription -> Markdown save
# CPU-only, mic selection, pause/resume, segments view

import sys
import os
import time
import traceback
import numpy as np
import pyaudio
import whisper

from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot, QTimer, QSettings
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QTextEdit, QProgressBar, QFileDialog, QMessageBox, QCheckBox
)

# ===== Audio constants =====
FORMAT = pyaudio.paInt16
SAMPLE_WIDTH = 2  # bytes for int16
CHANNELS = 1
RATE = 16000
CHUNK = 1024  # frames per buffer

# ===== Helpers =====
def process_audio_data(raw_audio: np.ndarray) -> np.ndarray:
    return (raw_audio.astype(np.float32) / np.iinfo(np.int16).max).astype(np.float32)

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

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
    lines.append(text.strip())
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

# ===== Audio Recorder Worker =====
class AudioRecorder(QObject):
    levelUpdated = Signal(int)          # 0-100
    chunkReady = Signal(np.ndarray)     # int16
    error = Signal(str)
    stopped = Signal()

    def __init__(self, device_index=None, parent=None):
        super().__init__(parent)
        self.device_index = device_index
        self._running = False
        self._paused = False
        self._p = None
        self._stream = None

    @Slot()
    def start(self):
        if self._running:
            return
        try:
            self._p = pyaudio.PyAudio()
            kwargs = dict(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=None,
            )
            if self.device_index is not None:
                kwargs["input_device_index"] = self.device_index
            self._stream = self._p.open(**kwargs)
            self._running = True
        except Exception as e:
            self.error.emit(f"Audio open failed: {e}")
            self.cleanup()
            return

        try:
            while self._running:
                if self._paused:
                    QThread.msleep(50)
                    continue
                data = self._stream.read(CHUNK, exception_on_overflow=False)
                arr = np.frombuffer(data, dtype=np.int16)
                self.chunkReady.emit(arr)
                rms = np.sqrt(np.mean(np.square(arr.astype(np.float32))))
                level = int(min(100, (rms / 500.0) * 100))  # heuristic
                self.levelUpdated.emit(level)
        except Exception as e:
            self.error.emit(f"Audio read failed: {e}")
        finally:
            self.cleanup()
            self.stopped.emit()

    @Slot()
    def stop(self):
        self._running = False

    @Slot()
    def pause(self):
        self._paused = True

    @Slot()
    def resume(self):
        self._paused = False

    def cleanup(self):
        try:
            if self._stream is not None:
                self._stream.stop_stream()
                self._stream.close()
        except Exception:
            pass
        try:
            if self._p is not None:
                self._p.terminate()
        except Exception:
            pass
        self._stream = None
        self._p = None

# ===== Whisper Transcriber Worker (CPU only) =====
class Transcriber(QObject):
    status = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, model_name="tiny", translate_to_english=False, language="", parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.translate_to_english = translate_to_english
        self.language = language  # "" for auto
        self._model = None
        self._audio = None

    def set_audio(self, audio_float32: np.ndarray):
        self._audio = audio_float32

    @Slot()
    def run(self):
        if self._audio is None or len(self._audio) == 0:
            self.error.emit("No audio to transcribe.")
            return
        try:
            self.status.emit(f"Loading model (CPU): {self.model_name}")
            if self._model is None:
                self._model = whisper.load_model(self.model_name, device="cpu")
            self.status.emit("Transcribing...")
            result = self._model.transcribe(
                self._audio,
                task="translate" if self.translate_to_english else "transcribe",
                language=(None if not self.language else self.language),
                fp16=False,
                condition_on_previous_text=True,
                temperature=0.0
            )
            self.finished.emit(result)
        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f"Transcription failed: {e}\n{tb}")

# ===== Main Window =====
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EchoScript")
        self.setMinimumSize(900, 640)

        # Settings
        self.settings = QSettings("CATIK", "EchoScript")

        # Controls
        self.modelBox = QComboBox()
        self.modelBox.addItems(["tiny", "base", "small", "medium"])
        self.modelBox.setCurrentText(self.settings.value("model", "tiny"))

        self.langBox = QComboBox()
        langs = [""] + sorted(["en", "tr", "de", "fr", "es", "it", "ru", "ar", "zh", "ja", "ko", "pt"])
        self.langBox.addItems(langs)
        self.langBox.setCurrentText(self.settings.value("language", ""))

        self.translateCheck = QCheckBox("Translate to English")
        self.translateCheck.setChecked(self.settings.value("translate_to_en", "false") == "true")

        self.micBox = QComboBox()
        self._populate_mics()

        self.startBtn = QPushButton("Start")
        self.pauseBtn = QPushButton("Pause")
        self.resumeBtn = QPushButton("Resume")
        self.stopBtn = QPushButton("Stop & Transcribe")
        self.pauseBtn.setEnabled(False)
        self.resumeBtn.setEnabled(False)
        self.stopBtn.setEnabled(False)

        self.statusLabel = QLabel("Idle")
        self.levelBar = QProgressBar()
        self.levelBar.setRange(0, 100)
        self.levelBar.setValue(0)
        self.levelBar.setFormat("Input level: %p%")

        self.timeLabel = QLabel("00:00")
        self._elapsed_sec = 0
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._tick)

        self.transcript = QTextEdit()
        self.transcript.setReadOnly(False)

        self.segmentsText = QTextEdit()
        self.segmentsText.setReadOnly(True)

        self.saveMdBtn = QPushButton("Save .md")
        self.saveMdBtn.setEnabled(False)

        # Layout
        top = QHBoxLayout()
        top.addWidget(QLabel("Model:"))
        top.addWidget(self.modelBox)
        top.addWidget(QLabel("Language:"))
        top.addWidget(self.langBox)
        top.addWidget(self.translateCheck)
        top.addSpacing(12)
        top.addWidget(QLabel("Mic:"))
        top.addWidget(self.micBox, 1)

        controls = QHBoxLayout()
        controls.addWidget(self.startBtn)
        controls.addWidget(self.pauseBtn)
        controls.addWidget(self.resumeBtn)
        controls.addWidget(self.stopBtn)
        controls.addStretch(1)
        controls.addWidget(QLabel("Duration:"))
        controls.addWidget(self.timeLabel)

        meters = QHBoxLayout()
        meters.addWidget(self.statusLabel, 2)
        meters.addWidget(self.levelBar, 3)

        exports = QHBoxLayout()
        exports.addWidget(self.saveMdBtn)
        exports.addStretch(1)

        left = QVBoxLayout()
        left.addLayout(top)
        left.addLayout(controls)
        left.addLayout(meters)
        left.addWidget(QLabel("Transcript"))
        left.addWidget(self.transcript, 2)
        left.addLayout(exports)

        right = QVBoxLayout()
        right.addWidget(QLabel("Segments (timestamps)"))
        right.addWidget(self.segmentsText, 1)

        root = QHBoxLayout()
        root.addLayout(left, 3)
        root.addLayout(right, 2)
        self.setLayout(root)

        # State
        self._captured_chunks = []
        self.audioThread = None
        self.audioWorker = None
        self.transcribeThread = None
        self.transcribeWorker = None
        self._result = None
        self._default_dir = self.settings.value("save_dir", "./recordings")
        self._session_tag = time.strftime("%Y%m%d_%H%M%S")

        # Connect
        self.startBtn.clicked.connect(self.on_start)
        self.pauseBtn.clicked.connect(self.on_pause)
        self.resumeBtn.clicked.connect(self.on_resume)
        self.stopBtn.clicked.connect(self.on_stop)
        self.saveMdBtn.clicked.connect(self.on_save_md)

    # ===== Mic enumeration =====
    def _populate_mics(self):
        self.micBox.clear()
        p = pyaudio.PyAudio()
        items = []
        try:
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if int(info.get("maxInputChannels", 0)) > 0:
                    name = f"{i}: {info.get('name','Unknown')}"
                    items.append((name, i))
        except Exception:
            pass
        p.terminate()
        if not items:
            self.micBox.addItem("No input devices", -1)
        else:
            for label, idx in items:
                self.micBox.addItem(label, idx)
            sel = self.settings.value("mic_index", None)
            if sel is not None:
                idxs = [self.micBox.itemData(k) for k in range(self.micBox.count())]
                if int(sel) in idxs:
                    self.micBox.setCurrentIndex(idxs.index(int(sel)))

    # ===== Recording control =====
    @Slot()
    def on_start(self):
        self._result = None
        self.transcript.clear()
        self.segmentsText.clear()
        self.levelBar.setValue(0)
        self.statusLabel.setText("Recording...")
        self._captured_chunks.clear()
        self._elapsed_sec = 0
        self.timeLabel.setText("00:00")
        self.timer.start()
        self._session_tag = time.strftime("%Y%m%d_%H%M%S")

        mic_index = self.micBox.currentData()
        if mic_index is None or mic_index == -1:
            self._error("No input device.")
            return

        self._persist_settings()

        self.audioWorker = AudioRecorder(device_index=mic_index)
        self.audioThread = QThread()
        self.audioWorker.moveToThread(self.audioThread)

        self.audioThread.started.connect(self.audioWorker.start)
        self.audioWorker.levelUpdated.connect(self.levelBar.setValue)
        self.audioWorker.chunkReady.connect(self._on_chunk)
        self.audioWorker.error.connect(self._error)
        self.audioWorker.stopped.connect(self._on_rec_stopped)

        self.audioThread.start()

        self.startBtn.setEnabled(False)
        self.pauseBtn.setEnabled(True)
        self.resumeBtn.setEnabled(False)
        self.stopBtn.setEnabled(True)
        self.saveMdBtn.setEnabled(False)

    @Slot()
    def on_pause(self):
        if self.audioWorker:
            self.audioWorker.pause()
        self.pauseBtn.setEnabled(False)
        self.resumeBtn.setEnabled(True)
        self.statusLabel.setText("Paused")

    @Slot()
    def on_resume(self):
        if self.audioWorker:
            self.audioWorker.resume()
        self.pauseBtn.setEnabled(True)
        self.resumeBtn.setEnabled(False)
        self.statusLabel.setText("Recording...")

    @Slot()
    def on_stop(self):
        if self.audioWorker:
            self.audioWorker.stop()
        self.stopBtn.setEnabled(False)
        self.statusLabel.setText("Stopping recorder...")

    @Slot(np.ndarray)
    def _on_chunk(self, arr: np.ndarray):
        # keep up to ~10 minutes
        max_minutes = 10
        max_samples = int(RATE * 60 * max_minutes)
        current = sum(len(c) for c in self._captured_chunks)
        if current + len(arr) > max_samples:
            while self._captured_chunks and sum(len(c) for c in self._captured_chunks) > max_samples - len(arr):
                self._captured_chunks.pop(0)
        self._captured_chunks.append(arr.copy())

    @Slot()
    def _on_rec_stopped(self):
        try:
            if self.audioThread and self.audioThread.isRunning():
                self.audioThread.quit()
                self.audioThread.wait(3000)
        except Exception:
            pass

        self.timer.stop()
        self.statusLabel.setText("Preparing audio...")
        if not self._captured_chunks:
            self._error("No audio captured.")
            self._reset_buttons()
            return

        raw = np.concatenate(self._captured_chunks)
        audio_f32 = process_audio_data(raw)

        self.statusLabel.setText("Starting transcription (CPU)...")
        model_name = self.modelBox.currentText()
        translate_to_english = self.translateCheck.isChecked()
        language = self.langBox.currentText()

        self.transcribeWorker = Transcriber(
            model_name=model_name,
            translate_to_english=translate_to_english,
            language=language
        )
        self.transcribeWorker.set_audio(audio_f32)
        self.transcribeThread = QThread()
        self.transcribeWorker.moveToThread(self.transcribeThread)

        self.transcribeThread.started.connect(self.transcribeWorker.run)
        self.transcribeWorker.status.connect(self.statusLabel.setText)
        self.transcribeWorker.finished.connect(self._on_transcription_finished)
        self.transcribeWorker.error.connect(self._error)

        self.transcribeThread.start()

    @Slot(dict)
    def _on_transcription_finished(self, result: dict):
        try:
            if self.transcribeThread and self.transcribeThread.isRunning():
                self.transcribeThread.quit()
                self.transcribeThread.wait(3000)
        except Exception:
            pass

        self._result = result
        text = (result.get("text") or "").strip()
        self.transcript.setPlainText(text)

        seg_lines = []
        for i, seg in enumerate(result.get("segments", []), start=1):
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            seg_text = (seg.get("text") or "").strip()
            seg_lines.append(f"{i:03} [{start:7.2f} -> {end:7.2f}] {seg_text}")
        self.segmentsText.setPlainText("\n".join(seg_lines))

        self.statusLabel.setText("Done")
        self._reset_buttons()
        self.saveMdBtn.setEnabled(True)

        self._autosave_md(text)

    # ===== Markdown save =====
    def _autosave_md(self, text: str):
        created = time.strftime("%Y-%m-%d %H:%M:%S")
        md = render_markdown_document(
            text=text,
            result=(self._result or {}),
            meta={
                "title": "Transcription",
                "model": self.modelBox.currentText(),
                "language": self.langBox.currentText(),
                "translate_to_en": self.translateCheck.isChecked(),
                "created": created,
            },
        )
        path = os.path.join(self._default_dir, f"transcription_{self._session_tag}.md")
        try:
            ensure_dir(path)
            with open(path, "w", encoding="utf-8") as f:
                f.write(md)
            self.statusLabel.setText(f"Saved: {path}")
        except Exception as e:
            self._error(f"Auto-save failed: {e}")

    @Slot()
    def on_save_md(self):
        text = self.transcript.toPlainText()
        if not text:
            self._error("Nothing to save.")
            return
        created = time.strftime("%Y-%m-%d %H:%M:%S")
        md = render_markdown_document(
            text=text,
            result=(self._result or {}),
            meta={
                "title": "Transcription",
                "model": self.modelBox.currentText(),
                "language": self.langBox.currentText(),
                "translate_to_en": self.translateCheck.isChecked(),
                "created": created,
            },
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Markdown",
            os.path.join(self._default_dir, f"transcription_{self._session_tag}.md"),
            "Markdown Files (*.md)"
        )
        if not path:
            return
        try:
            ensure_dir(path)
            with open(path, "w", encoding="utf-8") as f:
                f.write(md)
            self.statusLabel.setText(f"Saved: {path}")
            self._default_dir = os.path.dirname(path)
            self._persist_settings()
        except Exception as e:
            self._error(f"Save failed: {e}")

    def _tick(self):
        self._elapsed_sec += 1
        mm = self._elapsed_sec // 60
        ss = self._elapsed_sec % 60
        self.timeLabel.setText(f"{mm:02}:{ss:02}")

    def _reset_buttons(self):
        self.startBtn.setEnabled(True)
        self.pauseBtn.setEnabled(False)
        self.resumeBtn.setEnabled(False)
        self.stopBtn.setEnabled(False)

    def _persist_settings(self):
        self.settings.setValue("model", self.modelBox.currentText())
        self.settings.setValue("language", self.langBox.currentText())
        self.settings.setValue("translate_to_en", "true" if self.translateCheck.isChecked() else "false")
        self.settings.setValue("mic_index", self.micBox.currentData())
        self.settings.setValue("save_dir", self._default_dir)

    def _error(self, msg: str):
        self.statusLabel.setText("Error")
        QMessageBox.critical(self, "Error", msg)
        self._reset_buttons()

    def closeEvent(self, event):
        try:
            if self.audioWorker:
                self.audioWorker.stop()
            if self.audioThread and self.audioThread.isRunning():
                self.audioThread.quit()
                self.audioThread.wait(1000)
        except Exception:
            pass
        try:
            if self.transcribeThread and self.transcribeThread.isRunning():
                self.transcribeThread.quit()
                self.transcribeThread.wait(1000)
        except Exception:
            pass
        super().closeEvent(event)

# ===== Main =====
def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
