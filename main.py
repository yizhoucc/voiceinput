import threading
import signal
import sys

# Unbuffered stdout
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
        output.show_status("[recording] Speak now... (double-tap Fn to stop)\n")

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
        output.show_status("[ready] Double-tap Fn to record\n")

    hotkey = HotkeyListener(on_activate=on_activate, on_deactivate=on_deactivate)

    def signal_handler(sig, frame):
        print("\n[exit] Shutting down...")
        hotkey.stop()
        if recording:
            audio.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print("=== VoiceInput ===")
    print(f"STT: {config.stt_provider} | Model: {config.whisper_model}")
    print(f"Language: {config.primary_language or 'auto-detect'}")
    print("Press Ctrl+Option+1 to toggle recording.")
    print("Press Ctrl+C to exit.\n")

    # Pre-load whisper model in background
    threading.Thread(target=stt._ensure_model, daemon=True).start()

    # Start hotkey listener (runs NSRunLoop on its thread)
    hotkey.start()

    # Keep main thread alive
    try:
        signal.pause()
    except AttributeError:
        threading.Event().wait()


if __name__ == "__main__":
    main()
