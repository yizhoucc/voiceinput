import io
import threading
import time
import wave
import numpy as np
import httpx
from stt.base import STTProvider
from config import config


class WhisperRemoteSTT(STTProvider):
    """Streaming STT with append-only commits, timestamp-tracked dedup.

    - Whisper processes full WINDOW_SECONDS for quality (no context cutting)
    - Buffer kept for whisper context, trimmed only for memory (>30s)
    - Global timestamps track which audio has been committed
    - Only Cmd+V append, never modify editor text
    """

    WINDOW_SECONDS = 15
    STABLE_SECONDS = 2.0

    def __init__(self, on_partial, on_final, on_commit=None):
        super().__init__(on_partial, on_final, on_commit)
        self._buffer = np.array([], dtype=np.float32)
        self._lock = threading.Lock()
        self._last_process_time = 0.0
        self._step_seconds = config.stt_step_ms / 1000.0
        self._processing = False
        self._total_samples_received = 0  # global sample counter
        self._committed_until_global = 0.0
        self._committed_text: list[str] = []
        self._recent_commits: list[str] = []  # last N committed texts for dedup
        self._last_speaker = None
        self._client = httpx.Client(timeout=30.0)

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
        flat = audio.flatten()
        with self._lock:
            self._buffer = np.concatenate([self._buffer, flat])
            self._total_samples_received += len(flat)
            # Trim buffer for memory: keep last 30s
            max_samples = int(30 * config.sample_rate)
            if len(self._buffer) > max_samples:
                self._buffer = self._buffer[-max_samples:]

        now = time.monotonic()
        if now - self._last_process_time >= self._step_seconds and not self._processing:
            self._last_process_time = now
            threading.Thread(target=self._process, args=(False,), daemon=True).start()

    def finalize(self):
        self._process(is_final=True)

    def _transcribe(self, audio_data: np.ndarray) -> list[dict]:
        wav_bytes = self._audio_to_wav_bytes(audio_data)
        lang = config.primary_language or "auto"
        response = self._client.post(
            f"{config.whisper_remote_url}/transcribe",
            files={"audio": ("chunk.wav", wav_bytes, "audio/wav")},
            data={
                "language": lang,
                "initial_prompt": config.whisper_prompt,
                "identify_speaker": "true",
            },
        )
        response.raise_for_status()
        return response.json().get("segments", [])

    def _fmt_seg(self, seg, for_commit=False):
        text = seg.get("text", "").strip()
        speaker = seg.get("speaker", "unknown")
        if for_commit and speaker == self._last_speaker:
            return text
        if for_commit:
            self._last_speaker = speaker
        if speaker == "me":
            return f"[我] {text}"
        elif speaker == "other":
            return f"[他] {text}"
        return text

    def _process(self, is_final: bool):
        if self._processing and not is_final:
            return
        self._processing = True

        try:
            with self._lock:
                buf_len = len(self._buffer)
                if buf_len < config.sample_rate * 0.5:
                    return
                window_samples = int(self.WINDOW_SECONDS * config.sample_rate)
                start = max(0, buf_len - window_samples)
                audio_data = self._buffer[start:].copy()
                audio_duration = len(audio_data) / config.sample_rate

                # Global time of window start
                total_time = self._total_samples_received / config.sample_rate
                window_start_global = total_time - audio_duration

            if len(audio_data) < config.sample_rate * 0.3:
                return

            segments = self._transcribe(audio_data)
            if not segments:
                return

            stable_cutoff = audio_duration - self.STABLE_SECONDS
            new_commits = []
            partial_texts = []

            for seg in segments:
                text = seg.get("text", "").strip()
                if not text:
                    continue
                seg_end = seg.get("end", 0)

                # Convert to global time
                seg_end_global = window_start_global + seg_end

                # Skip already committed segments
                if seg_end_global <= self._committed_until_global + 0.3:
                    continue

                if is_final or seg_end < stable_cutoff:
                    formatted = self._fmt_seg(seg, for_commit=True)
                    new_commits.append((formatted, seg_end_global))
                else:
                    partial_texts.append(self._fmt_seg(seg))

            if new_commits:
                for text, end_global in new_commits:
                    # Dedup: skip if same text was recently committed
                    if text in self._recent_commits:
                        self._committed_until_global = end_global
                        continue
                    self.on_commit(text)
                    self._committed_text.append(text)
                    self._recent_commits.append(text)
                    if len(self._recent_commits) > 10:
                        self._recent_commits.pop(0)
                    self._committed_until_global = end_global

            display = " ".join(self._committed_text + partial_texts)
            if is_final:
                self.on_final(display)
            else:
                self.on_partial(display)

        except Exception as e:
            print(f"\n[stt-remote] Error: {e}")
        finally:
            self._processing = False

    def reset(self):
        with self._lock:
            self._buffer = np.array([], dtype=np.float32)
            self._total_samples_received = 0
        self._committed_text.clear()
        self._committed_until_global = 0.0
        self._last_process_time = 0.0
        self._last_speaker = None
        self._recent_commits = []
