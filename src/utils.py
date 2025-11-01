import os
import time
import numpy as np

def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def process_audio_int16_to_float32(raw_audio: np.ndarray) -> np.ndarray:
    # int16 -> float32 in [-1, 1]
    return (raw_audio.astype(np.float32) / np.iinfo(np.int16).max).astype(np.float32)

def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def session_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def sec_to_mmss(s: float) -> str:
    try:
        s = float(s)
    except Exception:
        s = 0.0
    m = int(s // 60)
    ss = int(round(s % 60))
    return f"{m:02d}:{ss:02d}"
