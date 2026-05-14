import io
import threading
import time
import wave
import numpy as np
import httpx
from stt.base import STTProvider
from config import config


class WhisperRemoteSTT(STTProvider):
    """Streaming STT via remote faster-whisper on 5090 GPU over LAN.

    Uses incremental processing: once text stabilizes, that audio region
    is "committed" and won't be re-processed. This prevents language
    flip-flopping when switching between Chinese and English.
    """

    OVERLAP_SECONDS = 3  # overlap with committed region for context
    WINDOW_SECONDS = 12  # max new audio to process at once

    def __init__(self, on_partial, on_final):
        super().__init__(on_partial, on_final)
        self._buffer = np.array([], dtype=np.float32)
        self._lock = threading.Lock()
        self._last_process_time = 0.0
        self._step_seconds = config.stt_step_ms / 1000.0
        self._processing = False
        self._client = httpx.Client(timeout=30.0)

        # Incremental tracking
        self._committed_samples = 0  # how many samples are "done"
        self._committed_text: list[str] = []  # finalized text segments
        self._last_partial = ""

    def _ensure_model(self):
        pass

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

    def _transcribe(self, audio_data: np.ndarray) -> str:
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
        return response.json().get("text", "").strip()

    def _process(self, is_final: bool):
        if self._processing and not is_final:
            return
        self._processing = True

        try:
            with self._lock:
                total_samples = len(self._buffer)
                if total_samples < config.sample_rate * 0.5:
                    return

                # For final: process only uncommitted audio
                # For partial: process from (committed - overlap) to end, capped at window
                overlap_samples = int(self.OVERLAP_SECONDS * config.sample_rate)
                start = max(0, self._committed_samples - overlap_samples)

                if is_final:
                    audio_data = self._buffer[start:].copy()
                else:
                    max_end = start + int(self.WINDOW_SECONDS * config.sample_rate)
                    audio_data = self._buffer[start:min(total_samples, max_end)].copy()

            if len(audio_data) < config.sample_rate * 0.3:
                if is_final and self._committed_text:
                    self.on_final(" ".join(self._committed_text))
                return

            new_text = self._transcribe(audio_data)

            if not new_text:
                if is_final and self._committed_text:
                    self.on_final(" ".join(self._committed_text))
                return

            if is_final:
                # Combine committed + new
                all_text = " ".join(self._committed_text + [new_text])
                self.on_final(all_text)
            else:
                # Check if partial text stabilized (same as last time)
                if new_text == self._last_partial and new_text:
                    # Text stabilized → commit it
                    self._committed_text.append(new_text)
                    with self._lock:
                        self._committed_samples = len(self._buffer)
                    self._last_partial = ""

                self._last_partial = new_text
                display = " ".join(self._committed_text + [new_text])
                self.on_partial(display)

        except Exception as e:
            print(f"[stt-remote] Error: {e}")
        finally:
            self._processing = False

    def reset(self):
        with self._lock:
            self._buffer = np.array([], dtype=np.float32)
        self._committed_samples = 0
        self._committed_text.clear()
        self._last_partial = ""
        self._last_process_time = 0.0
