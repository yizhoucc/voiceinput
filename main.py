import argparse
import threading
import signal
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from audio import AudioCapture
from audio_utils import save_wav
from output.terminal import TerminalOutput
from output.system_insert import SystemTextInserter
from hotkey import HotkeyListener, check_accessibility
from config import config


def parse_args():
    parser = argparse.ArgumentParser(description="VoiceInput: streaming voice input tool")
    parser.add_argument("--llm", nargs="?", const="default", default=None,
                        help="Enable LLM polish. Optionally specify model name (e.g. --llm Qwen/Qwen3-8B)")
    parser.add_argument("--local", action="store_true", help="Use local whisper (Mac CPU, no 5090)")
    parser.add_argument("--language", type=str, default=None, help="Force language: zh, en, or auto (default: auto)")
    parser.add_argument("--quantize", action="store_true",
                        help="Use quantized inference (whisper int8, LLM int4). Saves ~50%% VRAM")
    return parser.parse_args()


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
    args = parse_args()

    if args.local:
        config.stt_provider = "whisper_local"
    if args.llm:
        config.llm_polish_enabled = True
        if args.llm != "default":
            config.vllm_model = args.llm
    if args.language:
        config.primary_language = None if args.language == "auto" else args.language
    if args.quantize:
        config.quantize = True

    output = TerminalOutput()
    inserter = SystemTextInserter()

    polisher = None
    if config.llm_polish_enabled:
        try:
            from llm.vllm_remote import VLLMPolisher
            polisher = VLLMPolisher()
            print("[llm] Polish enabled via vLLM")
        except Exception as e:
            print(f"[llm] Polish disabled: {e}")

    mode = None
    recording = False
    last_commit = ""
    feed_thread = None
    stop_event = threading.Event()

    def polish_text(text: str) -> str:
        nonlocal last_commit
        if not polisher:
            return text
        try:
            result = polisher.polish(text, context_before=last_commit)
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
        nonlocal last_commit
        if mode == "manual":
            return
        polished = polish_text(text)
        if polished != text:
            print(f"\n[polish] {text[:30]}... → {polished[:30]}...")
        inserter.paste(polished + " ")
        last_commit = polished

    stt = create_stt(on_partial=on_partial, on_final=on_final, on_commit=on_commit)
    audio_capture = AudioCapture()

    def feed_loop():
        while not stop_event.is_set():
            chunk = audio_capture.get_audio(timeout=0.1)
            if chunk is not None:
                stt.feed_audio(chunk)

    def on_start(m):
        nonlocal mode, recording, last_commit, feed_thread
        if recording:
            return
        recording = True
        mode = m
        last_commit = ""
        stt.reset()
        stop_event.clear()
        audio_capture.start()
        feed_thread = threading.Thread(target=feed_loop, daemon=True)
        feed_thread.start()
        print(f"[recording:{m}] Speak now...")

    def on_stop():
        nonlocal recording, mode, feed_thread
        if not recording:
            return
        m = mode
        recording = False
        stop_event.set()
        audio_capture.stop()
        if feed_thread:
            feed_thread.join(timeout=2)
        for chunk in audio_capture.drain():
            stt.feed_audio(chunk)

        raw = audio_capture.get_raw_audio()
        if raw is not None and len(raw) > 0:
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            wav_path = AUDIO_DIR / f"{ts}.wav"
            save_wav(raw, wav_path, config.sample_rate, config.channels)
            print(f"[saved] {wav_path}")

        if m == "manual":
            stt.prepare_finalize()
            mode = "finalizing"

        stt.finalize()
        mode = None
        print("[ready]")

    hotkey = HotkeyListener(on_start=on_start, on_stop=on_stop)

    print("=== VoiceInput ===")
    if config.stt_provider == "whisper_remote":
        print(f"STT: whisper_remote ({config.whisper_remote_url})")
    else:
        print(f"STT: whisper_local ({config.whisper_model})")
    llm_info = "disabled"
    if config.llm_polish_enabled:
        llm_info = config.vllm_model
    print(f"LLM: {llm_info}")
    print(f"Language: {config.primary_language or 'auto-detect'}")
    if config.quantize:
        print("Quantize: ON (whisper int8, LLM requires quantized server)")
        print("  Start quantized vLLM: vllm serve MODEL --quantization fp8 --dtype float16")
    print()

    print("[stt] Loading...", end=" ", flush=True)
    stt.warmup()
    print("done.")
    print()
    if check_accessibility():
        print("Ctrl+Shift+R = smart | Ctrl+Shift+E = manual")
    else:
        print("Enter = smart | 'e'+Enter = manual")
    print("Ctrl+C to exit.\n")

    signal.signal(signal.SIGINT, lambda *_: (print("\n[exit]"), sys.exit(0)))
    hotkey.start()
    threading.Event().wait()


if __name__ == "__main__":
    main()
