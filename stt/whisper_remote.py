import io
import threading
import time
import wave
import numpy as np
import httpx
from stt.base import STTProvider
from config import config


class WhisperRemoteSTT(STTProvider):
    """Streaming STT: commit the stable prefix of transcription.

    Each run, whisper returns the full transcription of the window.
    Compare with previous run: the common prefix (characters that
    didn't change) is stable and can be committed.

    This handles whisper's varying segment boundaries naturally.
    """

    WINDOW_SECONDS = 15
    MIN_COMMIT_CHARS = 2  # don't commit fewer than this many new chars

    def __init__(self, on_partial, on_final, on_commit=None):
        super().__init__(on_partial, on_final, on_commit)
        self._buffer = np.array([], dtype=np.float32)
        self._lock = threading.Lock()
        self._last_process_time = 0.0
        self._step_seconds = config.stt_step_ms / 1000.0
        self._processing = False
        self._pending = False
        self._total_samples = 0
        self._committed_text_str = ""  # full committed text as string
        self._prev_full_text = ""  # full text from previous run
        self._last_speaker = None
        self._client = httpx.Client(timeout=30.0)

    def _ensure_model(self):
        pass

    def _to_wav(self, audio: np.ndarray) -> bytes:
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
            self._total_samples += len(flat)
            max_s = int(30 * config.sample_rate)
            if len(self._buffer) > max_s:
                self._buffer = self._buffer[-max_s:]

        now = time.monotonic()
        if now - self._last_process_time >= self._step_seconds:
            if not self._processing:
                self._last_process_time = now
                threading.Thread(target=self._process, args=(False,), daemon=True).start()
            else:
                self._pending = True

    def finalize(self):
        self._process(is_final=True)

    def _transcribe(self, audio_data):
        resp = self._client.post(
            f"{config.whisper_remote_url}/transcribe",
            files={"audio": ("c.wav", self._to_wav(audio_data), "audio/wav")},
            data={
                "language": config.primary_language or "auto",
                "initial_prompt": config.whisper_prompt,
                "identify_speaker": "true",
            },
        )
        resp.raise_for_status()
        return resp.json().get("segments", [])

    def _format_segments(self, segments):
        """Format all segments into text with speaker tags."""
        parts = []
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            speaker = seg.get("speaker", "unknown")
            if speaker == "me" and speaker != self._last_speaker:
                parts.append(f"[我] {text}")
            elif speaker == "other" and speaker != self._last_speaker:
                parts.append(f"[他] {text}")
            else:
                parts.append(text)
            # Don't update _last_speaker here - do it on commit
        return " ".join(parts)

    def _common_prefix_len(self, a: str, b: str) -> int:
        n = min(len(a), len(b))
        for i in range(n):
            if a[i] != b[i]:
                return i
        return n

    def _process(self, is_final):
        if self._processing and not is_final:
            return
        self._processing = True

        try:
            with self._lock:
                buf_len = len(self._buffer)
                if buf_len < config.sample_rate * 0.5:
                    return
                win_samples = int(self.WINDOW_SECONDS * config.sample_rate)
                start = max(0, buf_len - win_samples)
                audio_data = self._buffer[start:].copy()

            segments = self._transcribe(audio_data)
            if not segments:
                return

            current_full = self._format_segments(segments)

            if is_final:
                # Commit everything remaining
                to_commit = current_full.strip()
                cleaned = to_commit.replace("[我]", "").replace("[他]", "").strip()
                if cleaned:
                    self.on_commit(to_commit)
                self.on_final(current_full)
                return

            # Check if window slid past our committed content
            if self._committed_text_str and not current_full.startswith(self._committed_text_str[:20]):
                # Window slid. Commit any remaining text from previous window.
                remaining = self._prev_full_text[len(self._committed_text_str):].strip()
                cleaned = remaining.replace("[我]", "").replace("[他]", "").strip()
                if cleaned:
                    self.on_commit(remaining)
                # Reset for new window
                self._committed_text_str = ""
                self._prev_full_text = ""

            # Find stable prefix: chars same in prev and current
            prefix_len = self._common_prefix_len(self._prev_full_text, current_full)

            # Commit new stable text
            committed_len = len(self._committed_text_str)
            if prefix_len > committed_len + self.MIN_COMMIT_CHARS:
                commit_end = prefix_len
                last_space = current_full.rfind(" ", committed_len, commit_end)
                if last_space > committed_len:
                    commit_end = last_space

                new_committed = current_full[committed_len:commit_end].strip()
                # Skip if only speaker tags with no real content
                cleaned = new_committed.replace("[我]", "").replace("[他]", "").strip()
                if cleaned:
                    self.on_commit(new_committed)
                    self._committed_text_str = current_full[:commit_end]

            self._prev_full_text = current_full

            # Display: all committed (including from previous windows) + current partial
            partial = current_full[len(self._committed_text_str):].strip()
            display = current_full  # show everything whisper returns
            self.on_partial(display)

        except Exception as e:
            print(f"\n[stt-remote] Error: {e}")
        finally:
            self._processing = False
            if self._pending and not is_final:
                self._pending = False
                self._last_process_time = time.monotonic()
                threading.Thread(target=self._process, args=(False,), daemon=True).start()

    def reset(self):
        with self._lock:
            self._buffer = np.array([], dtype=np.float32)
            self._total_samples = 0
        self._committed_text_str = ""
        self._prev_full_text = ""
        self._last_process_time = 0.0
        self._pending = False
        self._last_speaker = None
