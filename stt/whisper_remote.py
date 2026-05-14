import io
import threading
import time
import wave
import numpy as np
import httpx
from stt.base import STTProvider
from config import config


class WhisperRemoteSTT(STTProvider):
    """Streaming STT via remote faster-whisper on 5090 GPU over LAN."""

    def __init__(self, on_partial, on_final):
        super().__init__(on_partial, on_final)
        self._buffer = np.array([], dtype=np.float32)
        self._lock = threading.Lock()
        self._last_process_time = 0.0
        self._step_seconds = config.stt_step_ms / 1000.0
        self._processing = False
        self._client = httpx.Client(timeout=30.0)

    def _ensure_model(self):
        pass  # Model is on the remote server

    def _audio_to_wav_bytes(self, audio: np.ndarray) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(config.sample_rate)
            pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()

    def feed_audio(self, audio: np.ndarray):
        with self._lock:
            self._buffer = np.concatenate([self._buffer, audio.flatten()])

        now = time.monotonic()
        if now - self._last_process_time >= self._step_seconds and not self._processing:
            self._last_process_time = now
            threading.Thread(target=self._process, args=(False,), daemon=True).start()

    def finalize(self):
        self._process(is_final=True)

    def _process(self, is_final: bool):
        if self._processing and not is_final:
            return
        self._processing = True

        try:
            with self._lock:
                if len(self._buffer) < config.sample_rate * 0.5:
                    return
                if is_final:
                    audio_data = self._buffer.copy()
                else:
                    max_samples = int(15 * config.sample_rate)
                    audio_data = self._buffer[-max_samples:].copy()

            wav_bytes = self._audio_to_wav_bytes(audio_data)

            lang = config.primary_language or "auto"
            response = self._client.post(
                f"{config.whisper_remote_url}/transcribe",
                files={"audio": ("chunk.wav", wav_bytes, "audio/wav")},
                data={
                    "language": lang,
                    "initial_prompt": config.whisper_prompt,
                },
            )
            response.raise_for_status()
            result = response.json()

            text = result.get("text", "").strip()
            if not text:
                return

            if is_final:
                self.on_final(text)
            else:
                self.on_partial(text)

        except Exception as e:
            print(f"[stt-remote] Error: {e}")
        finally:
            self._processing = False

    def reset(self):
        with self._lock:
            self._buffer = np.array([], dtype=np.float32)
        self._last_process_time = 0.0
