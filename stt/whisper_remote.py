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

    Commit strategy: segments in the stable zone (first half of window)
    are auto-committed. Segments near the end (unstable tail) stay as partial.
    Additionally, a pause > 1s between segments triggers commit.
    """

    OVERLAP_SECONDS = 3
    WINDOW_SECONDS = 12
    STABLE_ZONE_SECONDS = 6  # segments older than this within window get committed
    PAUSE_COMMIT_SECONDS = 1.0  # silence gap to trigger commit

    def __init__(self, on_partial, on_final):
        super().__init__(on_partial, on_final)
        self._buffer = np.array([], dtype=np.float32)
        self._lock = threading.Lock()
        self._last_process_time = 0.0
        self._step_seconds = config.stt_step_ms / 1000.0
        self._processing = False
        self._client = httpx.Client(timeout=30.0)
        self._committed_text: list[str] = []
        self._buffer_offset = 0  # total samples trimmed so far

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

    def _transcribe(self, audio_data: np.ndarray) -> list[dict]:
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
        return response.json().get("segments", [])

    def _process(self, is_final: bool):
        if self._processing and not is_final:
            return
        self._processing = True

        try:
            with self._lock:
                buf_len = len(self._buffer)
                if buf_len < config.sample_rate * 0.5:
                    return

                # Window: last WINDOW_SECONDS of buffer
                window_samples = int(self.WINDOW_SECONDS * config.sample_rate)
                if is_final:
                    audio_data = self._buffer.copy()
                    window_start_in_buf = 0
                else:
                    window_start_in_buf = max(0, buf_len - window_samples)
                    audio_data = self._buffer[window_start_in_buf:].copy()

                audio_duration = len(audio_data) / config.sample_rate

            if len(audio_data) < config.sample_rate * 0.3:
                if is_final and self._committed_text:
                    self.on_final(" ".join(self._committed_text))
                return

            segments = self._transcribe(audio_data)

            if not segments:
                if is_final and self._committed_text:
                    self.on_final(" ".join(self._committed_text))
                return

            if is_final:
                # Commit everything
                all_text = [s["text"] for s in segments if s.get("text", "").strip()]
                final = " ".join(self._committed_text + all_text)
                self.on_final(final)
                return

            # Split segments into stable (to commit) and unstable (partial)
            stable_cutoff = audio_duration - self.STABLE_ZONE_SECONDS
            commit_texts = []
            partial_texts = []
            last_end = 0.0

            for seg in segments:
                text = seg.get("text", "").strip()
                if not text:
                    continue
                seg_end = seg.get("end", 0)
                seg_start = seg.get("start", 0)

                # Pause-based commit: if gap > threshold before this segment
                if seg_start - last_end > self.PAUSE_COMMIT_SECONDS and commit_texts:
                    # Commit everything before this gap
                    pass  # already in commit_texts

                if seg_end < stable_cutoff:
                    commit_texts.append(text)
                else:
                    partial_texts.append(text)

                last_end = seg_end

            # Also check for pause-based commit within partial zone
            # If there's a big gap, commit everything before the gap
            if len(segments) >= 2:
                for i in range(1, len(segments)):
                    prev_end = segments[i - 1].get("end", 0)
                    curr_start = segments[i].get("start", 0)
                    if curr_start - prev_end > self.PAUSE_COMMIT_SECONDS:
                        # Move segments before the gap to commit
                        gap_segments_before = segments[:i]
                        gap_segments_after = segments[i:]
                        before_texts = [s["text"].strip() for s in gap_segments_before if s.get("text", "").strip()]
                        after_texts = [s["text"].strip() for s in gap_segments_after if s.get("text", "").strip()]
                        if before_texts:
                            commit_texts = before_texts
                            partial_texts = after_texts

            # Commit stable text and trim buffer
            if commit_texts:
                self._committed_text.extend(commit_texts)
                # Trim buffer: keep from stable_cutoff onwards
                trim_samples = int(stable_cutoff * config.sample_rate)
                actual_trim = window_start_in_buf + trim_samples
                with self._lock:
                    if actual_trim > 0 and actual_trim < len(self._buffer):
                        self._buffer = self._buffer[actual_trim:]
                        self._buffer_offset += actual_trim

            # Display
            display = " ".join(self._committed_text + partial_texts)
            self.on_partial(display)

        except Exception as e:
            print(f"\n[stt-remote] Error: {e}")
        finally:
            self._processing = False

    def reset(self):
        with self._lock:
            self._buffer = np.array([], dtype=np.float32)
        self._committed_text.clear()
        self._buffer_offset = 0
        self._last_process_time = 0.0
