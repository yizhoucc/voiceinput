import threading
import time
import numpy as np
from faster_whisper import WhisperModel
from stt.base import STTProvider
from config import config


class WhisperLocalSTT(STTProvider):
    """Sliding-window streaming STT using faster-whisper on local CPU/Metal."""

    def __init__(self, on_partial, on_final):
        super().__init__(on_partial, on_final)
        self._model: WhisperModel | None = None
        self._buffer = np.array([], dtype=np.float32)
        self._lock = threading.Lock()
        self._last_process_time = 0.0
        self._step_seconds = config.stt_step_ms / 1000.0
        self._max_buffer_seconds = 30.0
        self._loading = False

    def _ensure_model(self):
        if self._model is None and not self._loading:
            self._loading = True
            print(f"[stt] Loading whisper model: {config.whisper_model} ...")
            self._model = WhisperModel(
                config.whisper_model,
                device="cpu",
                compute_type="int8",
            )
            print("[stt] Model loaded.")
            self._loading = False

    def feed_audio(self, audio: np.ndarray):
        self._ensure_model()
        with self._lock:
            flat = audio.flatten()
            self._buffer = np.concatenate([self._buffer, flat])

            max_samples = int(self._max_buffer_seconds * config.sample_rate)
            if len(self._buffer) > max_samples:
                self._buffer = self._buffer[-max_samples:]

        now = time.monotonic()
        if now - self._last_process_time >= self._step_seconds:
            self._last_process_time = now
            self._process_buffer(is_final=False)

    def finalize(self):
        self._process_buffer(is_final=True)

    def _process_buffer(self, is_final: bool):
        with self._lock:
            if len(self._buffer) < config.sample_rate * 0.3:
                return
            audio_data = self._buffer.copy()

        if self._model is None:
            return

        segments, info = self._model.transcribe(
            audio_data,
            language=config.primary_language,
            beam_size=1,
            best_of=1,
            vad_filter=True,
            vad_parameters=dict(
                threshold=config.vad_threshold,
                min_silence_duration_ms=config.silence_duration_ms,
            ),
        )

        text_parts = []
        for seg in segments:
            text_parts.append(seg.text.strip())

        full_text = " ".join(text_parts)
        if not full_text:
            return

        if is_final:
            self.on_final(full_text)
        else:
            self.on_partial(full_text)

    def reset(self):
        with self._lock:
            self._buffer = np.array([], dtype=np.float32)
        self._last_process_time = 0.0
