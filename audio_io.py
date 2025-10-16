"""Audio IO utilities and workers for EchoScript."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
try:  # pragma: no cover - optional in test env
    import pyaudio
except ModuleNotFoundError:  # pragma: no cover
    pyaudio = None
from PySide6.QtCore import QObject, Signal, Slot

FORMAT = pyaudio.paInt16 if pyaudio else None
CHANNELS = 1
SAMPLE_RATE = 16_000
CHUNK_FRAMES = 1024


class RingBuffer:
    """Single-producer / single-consumer ring buffer for ``int16`` samples."""

    def __init__(self, sample_rate_hz: int, max_minutes: int) -> None:
        self._sample_rate = sample_rate_hz
        self._capacity_frames = int(sample_rate_hz * 60 * max(1, max_minutes))
        self._buffer = np.zeros(self._capacity_frames, dtype=np.int16)
        self._lock = threading.Lock()
        self._write_index = 0
        self._size = 0
        self._overrun_count = 0

    @property
    def size_frames(self) -> int:
        with self._lock:
            return self._size

    @property
    def capacity_frames(self) -> int:
        return self._capacity_frames

    @property
    def duration_sec(self) -> float:
        return self.size_frames / float(self._sample_rate)

    @property
    def overrun_count(self) -> int:
        with self._lock:
            return self._overrun_count

    def write(self, frames: np.ndarray) -> None:
        if frames.dtype != np.int16:
            raise TypeError("RingBuffer only supports int16 frames")
        n = int(frames.size)
        if n == 0:
            return
        with self._lock:
            if n >= self._capacity_frames:
                # keep the most recent chunk
                frames = frames[-self._capacity_frames :]
                n = frames.size
                self._write_index = 0
                self._buffer[:] = frames
                self._size = n
                self._overrun_count += 1
                return
            end = self._write_index + n
            if end <= self._capacity_frames:
                self._buffer[self._write_index:end] = frames
            else:
                first = self._capacity_frames - self._write_index
                self._buffer[self._write_index:] = frames[:first]
                self._buffer[: end - self._capacity_frames] = frames[first:]
            self._write_index = (self._write_index + n) % self._capacity_frames
            if self._size + n > self._capacity_frames:
                self._overrun_count += 1
            self._size = min(self._capacity_frames, self._size + n)

    def read_last(self, duration_sec: float) -> np.ndarray:
        frames = min(int(duration_sec * self._sample_rate), self._capacity_frames)
        with self._lock:
            frames = min(frames, self._size)
            if frames <= 0:
                return np.zeros(0, dtype=np.int16)
            start = (self._write_index - frames) % self._capacity_frames
            if start + frames <= self._capacity_frames:
                out = self._buffer[start : start + frames]
                return np.array(out, copy=True)
            else:
                first = self._capacity_frames - start
                out = np.concatenate((self._buffer[start:], self._buffer[: frames - first]))
                return out.astype(np.int16, copy=False)

    def clear(self) -> None:
        with self._lock:
            self._write_index = 0
            self._size = 0


@dataclass
class RecorderStats:
    overruns: int = 0
    underruns: int = 0
    last_level: int = 0


class AudioRecorderWorker(QObject):
    levelChanged = Signal(int)
    errorOccurred = Signal(str)
    chunkCaptured = Signal(np.ndarray)
    deviceChanged = Signal()
    statsChanged = Signal(RecorderStats)
    stopped = Signal()

    def __init__(self, device_index: Optional[int], chunk_frames: int = CHUNK_FRAMES) -> None:
        super().__init__()
        self._device_index = device_index
        self._chunk_frames = chunk_frames
        self._ring_buffer = RingBuffer(SAMPLE_RATE, 10)
        self._running = False
        self._paused = False
        self._stream = None
        self._pa = None
        self._stats = RecorderStats()

    def ring_buffer(self) -> RingBuffer:
        return self._ring_buffer

    def update_buffer_minutes(self, minutes: int) -> None:
        minutes = max(1, min(60, int(minutes)))
        self._ring_buffer = RingBuffer(SAMPLE_RATE, minutes)

    def set_device_index(self, device_index: Optional[int]) -> None:
        self._device_index = device_index
        if self._running:
            self._restart_stream()

    @Slot()
    def start(self) -> None:
        if self._running:
            return
        try:
            self._open_stream()
        except Exception as exc:  # pragma: no cover - PyAudio specific
            self.errorOccurred.emit(f"Audio open failed: {exc}")
            self._cleanup()
            return
        self._running = True
        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue
            try:
                data = self._stream.read(self._chunk_frames, exception_on_overflow=False)
            except Exception as exc:  # pragma: no cover - PyAudio specific
                self._stats.overruns += 1
                self.errorOccurred.emit(f"Audio read failed: {exc}")
                time.sleep(0.1)
                continue
            frames = np.frombuffer(data, dtype=np.int16)
            self._ring_buffer.write(frames)
            level = int(
                min(
                    100,
                    np.sqrt(np.mean(np.square(frames.astype(np.float32)))) / 327.0,
                )
            )
            self._stats.last_level = level
            self.levelChanged.emit(level)
            self.chunkCaptured.emit(frames)
            self.statsChanged.emit(self._stats)
        self._cleanup()
        self.stopped.emit()

    @Slot()
    def stop(self) -> None:
        self._running = False

    @Slot()
    def pause(self) -> None:
        self._paused = True

    @Slot()
    def resume(self) -> None:
        self._paused = False

    def _open_stream(self) -> None:
        if pyaudio is None:  # pragma: no cover - runtime guard
            raise RuntimeError("PyAudio is not installed")
        self._pa = pyaudio.PyAudio()
        kwargs = dict(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=self._chunk_frames,
        )
        if self._device_index is not None:
            kwargs["input_device_index"] = self._device_index
        self._stream = self._pa.open(**kwargs)

    def _restart_stream(self) -> None:
        self._cleanup()
        try:
            self._open_stream()
            self.deviceChanged.emit()
        except Exception as exc:  # pragma: no cover
            self.errorOccurred.emit(f"Device switch failed: {exc}")

    def _cleanup(self) -> None:
        try:
            if self._stream is not None:
                self._stream.stop_stream()
                self._stream.close()
        finally:
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None


def list_input_devices() -> List[str]:
    if pyaudio is None:  # pragma: no cover - runtime guard
        return []
    pa = pyaudio.PyAudio()
    try:
        count = pa.get_device_count()
        devices = []
        for i in range(count):
            info = pa.get_device_info_by_index(i)
            if int(info.get("maxInputChannels", 0)) > 0:
                devices.append(f"{i}: {info.get('name')}")
        return devices
    finally:
        pa.terminate()
