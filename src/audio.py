import numpy as np
import pyaudio
from PySide6.QtCore import QObject, QThread, Signal, Slot

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

class AudioRecorder(QObject):
    levelUpdated = Signal(int)        # 0..100
    chunkReady = Signal(np.ndarray)   # int16
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
                rms = float(np.sqrt(np.mean(np.square(arr.astype(np.float32))))) if arr.size else 0.0
                level = int(min(100, (rms / 500.0) * 100))
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
