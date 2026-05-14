import threading
import time
import numpy as np
from faster_whisper import WhisperModel
from stt.base import STTProvider
from config import config


class WhisperLocalSTT(STTProvider):
    """Streaming STT using faster-whisper sliding window on local CPU.

    Uses incremental commit to prevent language flip-flopping.
    """

    OVERLAP_SECONDS = 3
    WINDOW_SECONDS = 12

    def __init__(self, on_partial, on_final):
        super().__init__(on_partial, on_final)
        self._model: WhisperModel | None = None
        self._buffer = np.array([], dtype=np.float32)
        self._lock = threading.Lock()
        self._last_process_time = 0.0
        self._step_seconds = config.stt_step_ms / 1000.0
        self._processing = False
        self._committed_samples = 0
        self._committed_text: list[str] = []
        self._last_partial = ""

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
                total_samples = len(self._buffer)
                if total_samples < config.sample_rate * 0.5:
                    return

                overlap_samples = int(self.OVERLAP_SECONDS * config.sample_rate)
                start = max(0, self._committed_samples - overlap_samples)

                if is_final:
                    audio_data = self._buffer[start:].copy()
                else:
                    max_end = start + int(self.WINDOW_SECONDS * config.sample_rate)
                    audio_data = self._buffer[start:min(total_samples, max_end)].copy()

            if self._model is None or len(audio_data) < config.sample_rate * 0.3:
                if is_final and self._committed_text:
                    self.on_final(" ".join(self._committed_text))
                return

            segments, _ = self._model.transcribe(
                audio_data,
                language=config.primary_language,
                initial_prompt=config.whisper_prompt,
                beam_size=1,
                best_of=1,
                vad_filter=True,
                vad_parameters=dict(
                    threshold=config.vad_threshold,
                    min_silence_duration_ms=config.silence_duration_ms,
                ),
            )

            text_parts = [seg.text.strip() for seg in segments if seg.text.strip()]
            new_text = " ".join(text_parts)

            if not new_text:
                if is_final and self._committed_text:
                    self.on_final(" ".join(self._committed_text))
                return

            if is_final:
                all_text = " ".join(self._committed_text + [new_text])
                self.on_final(all_text)
            else:
                if new_text == self._last_partial and new_text:
                    self._committed_text.append(new_text)
                    with self._lock:
                        keep_from = max(0, len(self._buffer) - int(self.OVERLAP_SECONDS * config.sample_rate))
                        self._buffer = self._buffer[keep_from:]
                        self._committed_samples = len(self._buffer)
                    self._last_partial = ""

                self._last_partial = new_text
                display = " ".join(self._committed_text + [new_text])
                self.on_partial(display)
        finally:
            self._processing = False

    def reset(self):
        with self._lock:
            self._buffer = np.array([], dtype=np.float32)
        self._committed_samples = 0
        self._committed_text.clear()
        self._last_partial = ""
        self._last_process_time = 0.0
