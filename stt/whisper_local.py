import threading
import time
import numpy as np
from faster_whisper import WhisperModel
from stt.base import STTProvider
from config import config


class WhisperLocalSTT(STTProvider):
    """Dual-pass streaming STT: run zh + en, merge by segment confidence."""

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
        """Run whisper for a single language, return segments with word details."""
        segments, info = self._model.transcribe(
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
        result = []
        for seg in segments:
            words = []
            if seg.words:
                words = [(w.word, w.start, w.end, w.probability) for w in seg.words]
            avg_prob = seg.avg_logprob
            result.append({
                "text": seg.text.strip(),
                "start": seg.start,
                "end": seg.end,
                "avg_logprob": avg_prob,
                "words": words,
            })
        return result, info.language_probability

    def _merge_dual_pass(self, zh_segs, zh_lang_prob, en_segs, en_lang_prob):
        """Merge zh and en by choosing the better language per time segment."""
        if not zh_segs and not en_segs:
            return ""
        if not zh_segs:
            return " ".join(s["text"] for s in en_segs)
        if not en_segs:
            return " ".join(s["text"] for s in zh_segs)

        # Build time-aligned chunks. For each zh segment, find overlapping en segments
        # and pick the one with better word-level confidence.
        all_chunks = []

        # Tag each segment with language
        for s in zh_segs:
            all_chunks.append(("zh", s))
        for s in en_segs:
            all_chunks.append(("en", s))

        # Sort by start time
        all_chunks.sort(key=lambda x: x[1]["start"])

        # Greedy: walk through, for overlapping segments pick the one with higher avg word confidence
        result = []
        covered_until = -1.0

        for lang, seg in all_chunks:
            seg_start = seg["start"]
            seg_end = seg["end"]

            # Compute overlap with already covered region
            overlap = max(0, min(covered_until, seg_end) - max(0, seg_start))
            seg_duration = seg_end - seg_start
            if seg_duration <= 0:
                continue

            # If >50% of this segment is already covered, skip it
            if overlap / seg_duration > 0.5:
                continue

            # Compute average word confidence for this segment
            if seg["words"]:
                avg_conf = sum(w[3] for w in seg["words"]) / len(seg["words"])
            else:
                avg_conf = 0.5

            # Check if there's a competing segment we already added that overlaps
            # If the last added segment overlaps significantly, compare and maybe replace
            if result and result[-1][2] > seg_start + 0.1:
                prev_lang, prev_text, prev_end, prev_conf = result[-1]
                prev_overlap = prev_end - seg_start
                if prev_overlap > 0.3:  # significant overlap
                    if avg_conf > prev_conf:
                        result[-1] = (lang, seg["text"], seg_end, avg_conf)
                        covered_until = seg_end
                    continue

            result.append((lang, seg["text"], seg_end, avg_conf))
            covered_until = max(covered_until, seg_end)

        return " ".join(item[1] for item in result if item[1]).strip()

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

            zh_segs, zh_prob = self._transcribe_language(audio_data, "zh", zh_prompt)
            en_segs, en_prob = self._transcribe_language(audio_data, "en", en_prompt)

            merged = self._merge_dual_pass(zh_segs, zh_prob, en_segs, en_prob)

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
