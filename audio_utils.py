import io
import wave
from pathlib import Path
import numpy as np


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    return (audio.flatten() * 32767).clip(-32768, 32767).astype(np.int16).tobytes()


def to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(float32_to_pcm16(audio))
    return buf.getvalue()


def save_wav(audio: np.ndarray, path: Path, sample_rate: int = 16000, channels: int = 1):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(float32_to_pcm16(audio))
