import threading
import queue
import numpy as np
import sounddevice as sd
from config import config


class AudioCapture:
    def __init__(self):
        self.queue: queue.Queue[np.ndarray] = queue.Queue()
        self.is_recording = False
        self._stream: sd.InputStream | None = None

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            print(f"[audio] {status}")
        if self.is_recording:
            self.queue.put(indata.copy())

    def start(self):
        self.is_recording = True
        self._stream = sd.InputStream(
            samplerate=config.sample_rate,
            channels=config.channels,
            dtype="float32",
            blocksize=int(config.sample_rate * 0.1),  # 100ms chunks
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self):
        self.is_recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_audio(self, timeout: float = 0.5) -> np.ndarray | None:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> list[np.ndarray]:
        chunks = []
        while not self.queue.empty():
            try:
                chunks.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return chunks
