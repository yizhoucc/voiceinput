import threading
import signal
import sys
import wave
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from audio import AudioCapture
from output.terminal import TerminalOutput
from output.system_insert import SystemTextInserter
from hotkey import HotkeyListener, check_accessibility
from config import config


def create_stt(on_partial, on_final, on_commit=None):
    if config.stt_provider == "whisper_remote":
        from stt.whisper_remote import WhisperRemoteSTT
        return WhisperRemoteSTT(on_partial, on_final, on_commit)
    else:
        from stt.whisper_local import WhisperLocalSTT
        return WhisperLocalSTT(on_partial, on_final, on_commit)


def create_llm():
    if not config.llm_polish_enabled:
        return None
    if config.llm_provider == "vllm_remote":
        from llm.vllm_remote import VLLMPolisher
        return VLLMPolisher()
    return None


AUDIO_DIR = Path("recordings")
AUDIO_DIR.mkdir(exist_ok=True)


def main():
    output = TerminalOutput()
    inserter = SystemTextInserter()
    polisher = create_llm()

    # Commit buffer: store commits with overlap for LLM polish
    commit_buffer: list[str] = []
    polished_count = 0

    def polish_and_insert(idx: int):
        """Polish commit at idx using overlap context, then insert."""
        nonlocal polished_count
        raw = commit_buffer[idx]
        ctx_before = commit_buffer[idx - 1] if idx > 0 else ""
        ctx_after = commit_buffer[idx + 1] if idx + 1 < len(commit_buffer) else ""

        if polisher:
            polished = polisher.polish(raw, context_before=ctx_before, context_after=ctx_after)
        else:
            polished = raw

        inserter.append(polished + " ")
        polished_count += 1
        print(f"\n[polish #{polished_count}] {raw[:30]}... → {polished[:30]}...")

    def try_polish_pending():
        """Polish commits that have enough context (next commit available as overlap)."""
        # Polish all commits except the last one (which doesn't have context_after yet)
        while polished_count < len(commit_buffer) - 1:
            polish_and_insert(polished_count)

    def on_partial(text: str):
        output.show_partial(text)

    def on_final(text: str):
        output.show_final(text)
        # On final, polish any remaining unpolished commits
        nonlocal polished_count
        while polished_count < len(commit_buffer):
            polish_and_insert(polished_count)

    def on_commit(text: str):
        commit_buffer.append(text)
        try_polish_pending()

    stt = create_stt(on_partial=on_partial, on_final=on_final, on_commit=on_commit)
    audio = AudioCapture()
    recording = False
    feed_thread: threading.Thread | None = None
    stop_event = threading.Event()

    def feed_loop():
        while not stop_event.is_set():
            chunk = audio.get_audio(timeout=0.1)
            if chunk is not None:
                stt.feed_audio(chunk)

    def on_activate():
        nonlocal recording, feed_thread, stop_event, polished_count
        if recording:
            return
        recording = True
        stt.reset()
        inserter.reset()
        commit_buffer.clear()
        polished_count = 0
        stop_event.clear()
        audio.start()
        feed_thread = threading.Thread(target=feed_loop, daemon=True)
        feed_thread.start()
        print("[recording] Speak now... (Ctrl+Shift+R or Enter to stop)")

    def save_wav(raw_audio: np.ndarray) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = AUDIO_DIR / f"{ts}.wav"
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(config.channels)
            wf.setsampwidth(2)
            wf.setframerate(config.sample_rate)
            pcm = (raw_audio * 32767).clip(-32768, 32767).astype(np.int16)
            wf.writeframes(pcm.tobytes())
        return path

    def on_deactivate():
        nonlocal recording, feed_thread
        if not recording:
            return
        recording = False
        stop_event.set()
        audio.stop()
        if feed_thread:
            feed_thread.join(timeout=2)
        for chunk in audio.drain():
            stt.feed_audio(chunk)

        raw = audio.get_raw_audio()
        if raw is not None and len(raw) > 0:
            wav_path = save_wav(raw)
            print(f"[saved] {wav_path}")

        stt.finalize()
        print("[ready] Ctrl+Shift+R or Enter to record")

    hotkey = HotkeyListener(on_activate=on_activate, on_deactivate=on_deactivate)

    print("=== VoiceInput ===")
    provider_info = config.stt_provider
    if config.stt_provider == "whisper_remote":
        provider_info += f" ({config.whisper_remote_url})"
    else:
        provider_info += f" ({config.whisper_model})"
    print(f"STT: {provider_info}")
    print(f"LLM: {config.llm_provider} ({config.vllm_model})" if config.llm_polish_enabled else "LLM: disabled")
    print(f"Language: {config.primary_language or 'auto-detect'}")
    print()

    print("[stt] Loading...", end=" ", flush=True)
    stt._ensure_model()
    print("done.")
    print()
    if check_accessibility():
        print("Hotkey: Ctrl+Shift+R to toggle recording (global)")
    else:
        print("Hotkey: Enter to toggle (grant Accessibility for Ctrl+Shift+R global)")
    print("Press Ctrl+C to exit.")
    print()

    signal.signal(signal.SIGINT, lambda *_: (print("\n[exit]"), sys.exit(0)))
    hotkey.start()
    threading.Event().wait()


if __name__ == "__main__":
    main()
