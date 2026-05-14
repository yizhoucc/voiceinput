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


AUDIO_DIR = Path("recordings")
AUDIO_DIR.mkdir(exist_ok=True)


def main():
    output = TerminalOutput()
    inserter = SystemTextInserter()

    # LLM polisher
    polisher = None
    if config.llm_polish_enabled:
        try:
            from llm.vllm_remote import VLLMPolisher
            polisher = VLLMPolisher()
            print("[llm] Polish enabled via vLLM")
        except Exception as e:
            print(f"[llm] Polish disabled: {e}")

    # State
    current_mode = [None]
    commit_history = []  # for overlap context

    def polish_text(text: str) -> str:
        if not polisher:
            return text
        ctx_before = commit_history[-1] if commit_history else ""
        try:
            result = polisher.polish(text, context_before=ctx_before)
            if result and result.strip():
                return result
        except Exception as e:
            print(f"\n[llm] Polish error: {e}")
        return text

    def on_partial(text: str):
        output.show_partial(text)

    def on_final(text: str):
        output.show_final(text)

    def on_commit(text: str):
        if current_mode[0] == "manual":
            return
        polished = polish_text(text)
        inserter.paste(polished + " ")
        commit_history.append(polished)

    stt = create_stt(on_partial=on_partial, on_final=on_final, on_commit=on_commit)
    audio_capture = AudioCapture()
    recording = [False]
    feed_thread = [None]
    stop_event = threading.Event()

    def feed_loop():
        while not stop_event.is_set():
            chunk = audio_capture.get_audio(timeout=0.1)
            if chunk is not None:
                stt.feed_audio(chunk)

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

    def start_recording(mode):
        if recording[0]:
            return
        recording[0] = True
        current_mode[0] = mode
        commit_history.clear()
        stt.reset()
        stop_event.clear()
        audio_capture.start()
        feed_thread[0] = threading.Thread(target=feed_loop, daemon=True)
        feed_thread[0].start()
        mode_label = "smart (speaker-change commit)" if mode == "smart" else "manual (commit on stop)"
        print(f"[recording:{mode}] Speak now...")

    def stop_recording():
        if not recording[0]:
            return
        mode = current_mode[0]
        recording[0] = False
        stop_event.set()
        audio_capture.stop()
        if feed_thread[0]:
            feed_thread[0].join(timeout=2)
        for chunk in audio_capture.drain():
            stt.feed_audio(chunk)

        raw = audio_capture.get_raw_audio()
        if raw is not None and len(raw) > 0:
            wav_path = save_wav(raw)
            print(f"[saved] {wav_path}")

        if mode == "manual":
            # In manual mode, finalize commits everything as one block
            stt._committed_audio_end = 0.0  # reset so finalize processes all
            # Temporarily enable commit for finalize
            current_mode[0] = "finalizing"

        stt.finalize()
        current_mode[0] = None
        print("[ready]")

    # Smart mode callbacks
    def on_activate():
        start_recording("smart")

    def on_deactivate():
        stop_recording()

    # Manual mode callbacks
    def on_activate_manual():
        start_recording("manual")

    def on_deactivate_manual():
        stop_recording()

    hotkey = HotkeyListener(
        on_activate=on_activate,
        on_deactivate=on_deactivate,
        on_activate_manual=on_activate_manual,
        on_deactivate_manual=on_deactivate_manual,
    )

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
        print("Ctrl+Shift+R = smart mode (speaker-change auto-commit)")
        print("Ctrl+Shift+E = manual mode (commit only on stop)")
    else:
        print("Enter = smart mode | type 'e'+Enter = manual mode")
    print("Press Ctrl+C to exit.")
    print()

    signal.signal(signal.SIGINT, lambda *_: (print("\n[exit]"), sys.exit(0)))
    hotkey.start()
    threading.Event().wait()


if __name__ == "__main__":
    main()
