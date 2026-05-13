import threading
import signal
import sys

sys.stdout.reconfigure(line_buffering=True)

from audio import AudioCapture
from stt.whisper_local import WhisperLocalSTT
from output.terminal import TerminalOutput
from hotkey import HotkeyListener
from config import config


def main():
    output = TerminalOutput()

    def on_partial(text: str):
        output.show_partial(text)

    def on_final(text: str):
        output.show_final(text)

    stt = WhisperLocalSTT(on_partial=on_partial, on_final=on_final)
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
        stt.finalize()
        print("[ready] Press Enter to record")

    hotkey = HotkeyListener(on_activate=on_activate, on_deactivate=on_deactivate)

    print("=== VoiceInput ===")
    print(f"STT: whisper local | Model: {config.whisper_model}")
    print(f"Language: {config.primary_language or 'auto-detect'}")
    print()

    # Load model first (blocking), so user knows when ready
    print("[stt] Loading model...", end=" ", flush=True)
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
