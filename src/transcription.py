import traceback
import whisper
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

class Transcriber(QObject):
    status = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, model_name="tiny", translate_to_english=False, language=""):
        super().__init__()
        self.model_name = model_name
        self.translate_to_english = translate_to_english
        self.language = language
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
                temperature=0.0,
            )
            self.finished.emit(result)
        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f"Transcription failed: {e} {tb}")
