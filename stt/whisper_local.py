import threading
import time
import numpy as np
from faster_whisper import WhisperModel
from stt.base import STTProvider
from config import config


class WhisperLocalSTT(STTProvider):
    """Sliding-window streaming STT using faster-whisper on local CPU."""

    def __init__(self, on_partial, on_final):
        super().__init__(on_partial, on_final)
        self._model: WhisperModel | None = None
        self._buffer = np.array([], dtype=np.float32)
        self._lock = threading.Lock()
        self._last_process_time = 0.0
        self._step_seconds = config.stt_step_ms / 1000.0
        self._processing = False
        self._finalized_text: list[str] = []

    def _ensure_model(self):
        if self._model is None:
            self._model = WhisperModel(
                config.whisper_model,
                device="cpu",
                compute_type="int8",
            )

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
                # Only process last 10 seconds for partials, full buffer for final
                if is_final:
                    audio_data = self._buffer.copy()
                else:
                    max_samples = int(10 * config.sample_rate)
                    audio_data = self._buffer[-max_samples:].copy()

            if self._model is None:
                return

            segments, _ = self._model.transcribe(
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

            text_parts = [seg.text.strip() for seg in segments if seg.text.strip()]
            full_text = " ".join(text_parts)
            if not full_text:
                return

            if is_final:
                self.on_final(full_text)
            else:
                self.on_partial(full_text)
        finally:
            self._processing = False

    def reset(self):
        with self._lock:
            self._buffer = np.array([], dtype=np.float32)
        self._last_process_time = 0.0
        self._finalized_text.clear()
