import threading
import time
import numpy as np
from faster_whisper import WhisperModel
from stt.base import STTProvider
from config import config


class WhisperLocalSTT(STTProvider):
    """Dual-pass streaming STT: run zh + en in parallel, merge by word confidence."""

    def __init__(self, on_partial, on_final):
        super().__init__(on_partial, on_final)
        self._model: WhisperModel | None = None
        self._buffer = np.array([], dtype=np.float32)
        self._lock = threading.Lock()
        self._last_process_time = 0.0
        self._step_seconds = config.stt_step_ms / 1000.0
        self._processing = False

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

    def _transcribe_language(self, audio_data, lang, prompt):
        """Run whisper for a single language, return list of (word, start, end, prob)."""
        segments, _ = self._model.transcribe(
            audio_data,
            language=lang,
            initial_prompt=prompt,
            beam_size=1,
            best_of=1,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(
                threshold=config.vad_threshold,
                min_silence_duration_ms=config.silence_duration_ms,
            ),
        )
        words = []
        for seg in segments:
            if seg.words:
                for w in seg.words:
                    words.append((w.word, w.start, w.end, w.probability))
        return words

    def _merge_dual_pass(self, zh_words, en_words):
        """Merge zh and en word lists by timestamp, picking higher confidence per region."""
        if not zh_words and not en_words:
            return ""
        if not zh_words:
            return "".join(w[0] for w in en_words)
        if not en_words:
            return "".join(w[0] for w in zh_words)

        # Build a timeline: for each time point, pick the language with higher confidence
        all_events = []
        for word, start, end, prob in zh_words:
            all_events.append((start, end, word, prob, "zh"))
        for word, start, end, prob in en_words:
            all_events.append((start, end, word, prob, "en"))

        all_events.sort(key=lambda x: (x[0], -x[3]))

        # Greedy merge: walk through time, skip overlapping words with lower confidence
        result = []
        covered_until = -1.0

        for start, end, word, prob, lang in all_events:
            # Skip if this word's start is already covered by a higher-confidence word
            if start < covered_until - 0.05:
                continue
            result.append(word)
            covered_until = end

        return "".join(result).strip()

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
                    max_samples = int(10 * config.sample_rate)
                    audio_data = self._buffer[-max_samples:].copy()

            if self._model is None:
                return

            zh_prompt = config.whisper_prompt
            en_prompt = "The following is a conversation about technology, AI, machine learning, transformers, QKV, attention, neural networks."

            zh_words = self._transcribe_language(audio_data, "zh", zh_prompt)
            en_words = self._transcribe_language(audio_data, "en", en_prompt)

            merged = self._merge_dual_pass(zh_words, en_words)

            if not merged:
                return

            if is_final:
                self.on_final(merged)
            else:
                self.on_partial(merged)
        finally:
            self._processing = False

    def reset(self):
        with self._lock:
            self._buffer = np.array([], dtype=np.float32)
        self._last_process_time = 0.0
