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
from output.overlay import Overlay
from hotkey import HotkeyListener, check_accessibility
from config import config


def create_stt(on_partial, on_final, on_commit=None):
    if config.stt_provider == "whisper_remote":
        from stt.whisper_remote import WhisperRemoteSTT
        return WhisperRemoteSTT(on_partial, on_final, on_commit)
    else:
        from stt.whisper_local import WhisperLocalSTT
        return WhisperLocalSTT(on_partial, on_final, on_commit)

AUDIO_DIR = Path("recordings")
AUDIO_DIR.mkdir(exist_ok=True)


def main():
    output = TerminalOutput()
    inserter = SystemTextInserter()
    overlay = Overlay()

    def on_partial(text: str):
        output.show_partial(text)
        # Show uncommitted portion in floating overlay
        overlay.show(text)

    def on_final(text: str):
        output.show_final(text)
        overlay.hide()

    def on_commit(text: str):
        inserter.commit(text)

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
        nonlocal recording, feed_thread, stop_event
        if recording:
            return
        recording = True
        stt.reset()
        inserter.reset()
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
        # Don't reset inserter here — let the final text be inserted first
        # Reset happens in on_activate before next recording
        print("[ready] Ctrl+Shift+R or Enter to record")

    hotkey = HotkeyListener(on_activate=on_activate, on_deactivate=on_deactivate)

    print("=== VoiceInput ===")
    provider_info = config.stt_provider
    if config.stt_provider == "whisper_remote":
        provider_info += f" ({config.whisper_remote_url})"
    else:
        provider_info += f" ({config.whisper_model})"
    print(f"STT: {provider_info}")
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
    print("Text will be inserted at cursor position in any app.")
    print("Press Ctrl+C to exit.")
    print()

    signal.signal(signal.SIGINT, lambda *_: (print("\n[exit]"), sys.exit(0)))

    hotkey.start()

    threading.Event().wait()


if __name__ == "__main__":
    main()
