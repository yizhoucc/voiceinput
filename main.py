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
from hotkey import HotkeyListener
from config import config


def create_stt(on_partial, on_final):
    if config.stt_provider == "whisper_remote":
        from stt.whisper_remote import WhisperRemoteSTT
        return WhisperRemoteSTT(on_partial, on_final)
    else:
        from stt.whisper_local import WhisperLocalSTT
        return WhisperLocalSTT(on_partial, on_final)

AUDIO_DIR = Path("recordings")
AUDIO_DIR.mkdir(exist_ok=True)


def main():
    output = TerminalOutput()

    def on_partial(text: str):
        output.show_partial(text)

    def on_final(text: str):
        output.show_final(text)

    stt = create_stt(on_partial=on_partial, on_final=on_final)
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
        stop_event.clear()
        audio.start()
        feed_thread = threading.Thread(target=feed_loop, daemon=True)
        feed_thread.start()
        print("[recording] Speak now... (press Enter to stop)")

    def save_wav(raw_audio: np.ndarray) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = AUDIO_DIR / f"{ts}.wav"
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(config.channels)
            wf.setsampwidth(2)  # 16-bit
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
        print("[ready] Press Enter to record")

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
    print("Press Enter to start recording, Enter again to stop.")
    print("Press Ctrl+C to exit.")
    print()

    signal.signal(signal.SIGINT, lambda *_: (print("\n[exit]"), sys.exit(0)))

    hotkey.start()

    # Keep main thread alive
    threading.Event().wait()


if __name__ == "__main__":
    main()
