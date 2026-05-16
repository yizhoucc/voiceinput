"""Local STT using MLX Whisper on Apple Silicon GPU."""
import threading
import time
import numpy as np
import mlx_whisper
from audio_utils import save_wav
from stt.base import STTProvider
from config import config
from pathlib import Path


class WhisperLocalSTT(STTProvider):
    """Streaming STT using MLX Whisper on Mac GPU.

    mlx_whisper.transcribe() takes a file path, so we save audio chunks
    to a temp file and transcribe periodically.
    """

    WINDOW_SECONDS = 15

    def __init__(self, on_partial, on_final, on_commit=None):
        super().__init__(on_partial, on_final, on_commit)
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._last_process_time = 0.0
        self._step_seconds = config.stt_step_ms / 1000.0
        self._processing = False
        self._pending = False
        self._prev_text = ""
        self._committed_text = ""
        self._tmp_path = Path("/tmp/_voiceinput_mlx_chunk.wav")

    def warmup(self):
        # Trigger model download on first use
        dummy = np.zeros(16000, dtype=np.float32)
        save_wav(dummy, self._tmp_path, config.sample_rate)
        mlx_whisper.transcribe(str(self._tmp_path),
            path_or_hf_repo=config.whisper_model)

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

    def _get_audio(self):
        with self._lock:
            if not self._chunks:
                return np.array([], dtype=np.float32)
            buf = np.concatenate(self._chunks)
            max_s = int(30 * config.sample_rate)
            if len(buf) > max_s:
                buf = buf[-max_s:]
                self._chunks = [buf]
            return buf

    def _transcribe(self, audio: np.ndarray) -> str:
        save_wav(audio, self._tmp_path, config.sample_rate)
        result = mlx_whisper.transcribe(
            str(self._tmp_path),
            path_or_hf_repo=config.whisper_model,
            language=config.primary_language,
            initial_prompt=config.whisper_prompt,
        )
        return result.get("text", "").strip()

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
            buf = self._get_audio()
            if len(buf) < config.sample_rate * 0.5:
                return

            # Take last WINDOW_SECONDS for processing
            win = int(self.WINDOW_SECONDS * config.sample_rate)
            audio = buf[-win:] if len(buf) > win else buf

            text = self._transcribe(audio)
            if not text:
                return

            if is_final:
                self.on_commit(text)
                self.on_final(text)
                return

            self.on_partial(text)

            # Prefix stability commit
            prefix_len = self._common_prefix_len(self._prev_text, text)
            committed_len = len(self._committed_text)
            if prefix_len > committed_len + 3:
                commit_end = prefix_len
                last_space = text.rfind(" ", committed_len, commit_end)
                if last_space > committed_len:
                    commit_end = last_space
                new = text[committed_len:commit_end].strip()
                if new:
                    self.on_commit(new)
                    self._committed_text = text[:commit_end]

            self._prev_text = text

        except Exception as e:
            print(f"\n[stt-local] Error: {e}")
        finally:
            self._processing = False
            if self._pending and not is_final:
                self._pending = False
                self._last_process_time = time.monotonic()
                threading.Thread(target=self._process, args=(False,), daemon=True).start()

    def reset(self):
        with self._lock:
            self._chunks.clear()
        self._prev_text = ""
        self._committed_text = ""
        self._last_process_time = 0.0
        self._pending = False

    def prepare_finalize(self):
        self._committed_text = ""
