import threading
import time
import numpy as np
import httpx
from stt.base import STTProvider
from audio_utils import to_wav_bytes
from config import config


class WhisperRemoteSTT(STTProvider):
    """Streaming STT via 5090 GPU. Append-only commits.

    Commit triggers: speaker change or manual stop (finalize).
    """

    WINDOW_SECONDS = 15
    SPEAKER_CHANGE_COMMIT = True

    def __init__(self, on_partial, on_final, on_commit=None):
        super().__init__(on_partial, on_final, on_commit)
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._last_process_time = 0.0
        self._step_seconds = config.stt_step_ms / 1000.0
        self._processing = False
        self._pending = False
        self._committed_audio_end = 0.0
        self._last_committed_text = ""
        self._client = httpx.Client(timeout=30.0)

    def prepare_finalize(self):
        self._committed_audio_end = 0.0

    def feed_audio(self, audio: np.ndarray):
        flat = audio.flatten()
        with self._lock:
            self._chunks.append(flat)

        now = time.monotonic()
        if now - self._last_process_time >= self._step_seconds:
            if not self._processing:
                self._last_process_time = now
                threading.Thread(target=self._process, args=(False,), daemon=True).start()
            else:
                self._pending = True

    def finalize(self):
        self._process(is_final=True)

    def _get_buffer(self):
        with self._lock:
            if not self._chunks:
                return np.array([], dtype=np.float32)
            buf = np.concatenate(self._chunks)
            # Trim for memory: keep last 30s
            max_s = int(30 * config.sample_rate)
            if len(buf) > max_s:
                buf = buf[-max_s:]
                self._chunks = [buf]
            return buf

    def _dedup(self, text: str) -> str:
        if not self._last_committed_text or not text:
            self._last_committed_text = text
            return text
        if text.strip() == self._last_committed_text.strip():
            return ""
        tail = self._last_committed_text
        for overlap_len in range(min(len(tail), len(text)), 2, -1):
            if text[:overlap_len] == tail[-overlap_len:]:
                text = text[overlap_len:].strip()
                break
        self._last_committed_text = text
        return text

    @staticmethod
    def _format_segments(segments):
        parts = []
        speakers = []
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            speaker = seg.get("speaker", "unknown")
            speakers.append(speaker)
            if speaker == "me":
                parts.append(f"[我] {text}")
            elif speaker == "other":
                parts.append(f"[他] {text}")
            else:
                parts.append(text)
        return parts, speakers

    def _transcribe(self, audio_data):
        resp = self._client.post(
            f"{config.whisper_remote_url}/transcribe",
            files={"audio": ("c.wav", to_wav_bytes(audio_data, config.sample_rate), "audio/wav")},
            data={
                "language": config.primary_language or "auto",
                "initial_prompt": config.whisper_prompt,
                "identify_speaker": "true",
            },
        )
        resp.raise_for_status()
        return resp.json().get("segments", [])

    def _process(self, is_final):
        if self._processing and not is_final:
            return
        self._processing = True

        try:
            buf = self._get_buffer()
            buf_len = len(buf)
            if buf_len < config.sample_rate * 0.5:
                return

            committed_samples = int(self._committed_audio_end * config.sample_rate)
            overlap_samples = int(1.0 * config.sample_rate)
            start = max(0, committed_samples - overlap_samples)
            audio_chunk = buf[start:buf_len]

            if len(audio_chunk) < config.sample_rate * 0.3:
                return

            segments = self._transcribe(audio_chunk)
            if not segments:
                return

            parts, speakers = self._format_segments(segments)
            full_text = " ".join(parts)
            last_seg_end = segments[-1].get("end", 0) if segments else 0

            if is_final:
                deduped = self._dedup(full_text)
                if deduped.strip():
                    self.on_commit(deduped)
                self.on_final(deduped)
                return

            self.on_partial(full_text)

            # Speaker change commit
            if self.SPEAKER_CHANGE_COMMIT and len(speakers) >= 2:
                unique = set(s for s in speakers if s != "unknown")
                if len(unique) >= 2:
                    first_speaker = speakers[0]
                    for idx, spk in enumerate(speakers[1:], 1):
                        if spk != first_speaker and spk != "unknown":
                            before = " ".join(parts[:idx])
                            deduped = self._dedup(before)
                            if deduped.strip():
                                self.on_commit(deduped)
                                committed_segs = [s for s in segments if s.get("text", "").strip()][:idx]
                                if committed_segs:
                                    seg_end = committed_segs[-1].get("end", 0)
                                    self._committed_audio_end = (start / config.sample_rate) + seg_end
                            break

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
            self._chunks.clear()
        self._committed_audio_end = 0.0
        self._last_process_time = 0.0
        self._pending = False
        self._last_committed_text = ""
