import io
import threading
import time
import wave
import numpy as np
import httpx
from stt.base import STTProvider
from config import config


class WhisperRemoteSTT(STTProvider):
    """Streaming STT. Simple and reliable.

    - Always processes full window (15s) for quality
    - on_partial: every 2s, show current transcription in terminal
    - on_commit: triggered by (a) long pause detected or (b) finalize
    - Committed text is inserted at cursor and never modified
    """

    WINDOW_SECONDS = 15
    PAUSE_SECONDS = 2.0  # silence this long triggers auto-commit

    def __init__(self, on_partial, on_final, on_commit=None):
        super().__init__(on_partial, on_final, on_commit)
        self._buffer = np.array([], dtype=np.float32)
        self._lock = threading.Lock()
        self._last_process_time = 0.0
        self._step_seconds = config.stt_step_ms / 1000.0
        self._processing = False
        self._pending = False
        self._total_samples = 0
        self._committed_audio_end = 0.0  # global time up to which audio is committed
        self._client = httpx.Client(timeout=30.0)

    def _ensure_model(self):
        pass

    def _to_wav(self, audio):
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

        now = time.monotonic()
        if now - self._last_process_time >= self._step_seconds:
            if not self._processing:
                self._last_process_time = now
                threading.Thread(target=self._process, args=(False,), daemon=True).start()
            else:
                self._pending = True

    def finalize(self):
        self._process(is_final=True)

    def _transcribe_range(self, start_sample, end_sample):
        """Transcribe a specific range of the buffer."""
        with self._lock:
            audio = self._buffer[start_sample:end_sample].copy()
        if len(audio) < config.sample_rate * 0.3:
            return ""
        resp = self._client.post(
            f"{config.whisper_remote_url}/transcribe",
            files={"audio": ("c.wav", self._to_wav(audio), "audio/wav")},
            data={
                "language": config.primary_language or "auto",
                "initial_prompt": config.whisper_prompt,
                "identify_speaker": "true",
            },
        )
        resp.raise_for_status()
        segments = resp.json().get("segments", [])
        parts = []
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            speaker = seg.get("speaker", "unknown")
            if speaker == "me":
                parts.append(f"[我] {text}")
            elif speaker == "other":
                parts.append(f"[他] {text}")
            else:
                parts.append(text)
        return " ".join(parts)

    def _process(self, is_final):
        if self._processing and not is_final:
            return
        self._processing = True

        try:
            with self._lock:
                buf_len = len(self._buffer)
                total_time = self._total_samples / config.sample_rate

            if buf_len < config.sample_rate * 0.5:
                return

            # Determine what to transcribe: from committed point to end (no overlap)
            committed_samples = int(self._committed_audio_end * config.sample_rate)
            uncommitted_start = committed_samples
            audio_chunk = None
            with self._lock:
                audio_chunk = self._buffer[uncommitted_start:buf_len].copy()

            if len(audio_chunk) < config.sample_rate * 0.3:
                return

            # Transcribe
            resp = self._client.post(
                f"{config.whisper_remote_url}/transcribe",
                files={"audio": ("c.wav", self._to_wav(audio_chunk), "audio/wav")},
                data={
                    "language": config.primary_language or "auto",
                    "initial_prompt": config.whisper_prompt,
                    "identify_speaker": "true",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            segments = data.get("segments", [])

            if not segments:
                return

            # Build text
            parts = []
            last_seg_end = 0
            for seg in segments:
                text = seg.get("text", "").strip()
                if not text:
                    continue
                speaker = seg.get("speaker", "unknown")
                if speaker == "me":
                    parts.append(f"[我] {text}")
                elif speaker == "other":
                    parts.append(f"[他] {text}")
                else:
                    parts.append(text)
                last_seg_end = seg.get("end", 0)

            full_text = " ".join(parts)
            audio_dur = len(audio_chunk) / config.sample_rate

            if is_final:
                # Commit everything
                if full_text.strip():
                    self.on_commit(full_text)
                self.on_final(full_text)
                return

            # Show partial in terminal
            self.on_partial(full_text)

            # Check for pause: if last segment ends >PAUSE_SECONDS before audio end
            silence_at_end = audio_dur - last_seg_end
            if silence_at_end > self.PAUSE_SECONDS and full_text.strip():
                # Long pause detected → auto-commit
                self.on_commit(full_text)
                with self._lock:
                    self._committed_audio_end = total_time

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
        self._committed_audio_end = 0.0
        self._last_process_time = 0.0
        self._pending = False
