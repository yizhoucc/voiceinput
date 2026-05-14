"""Simulate streaming: feed audio file to STT 1 second at a time."""
import sys
import time
import subprocess
import numpy as np
import soundfile as sf

sys.stdout.reconfigure(line_buffering=True)

DURATION = 60  # seconds to test

print(f"Converting roro.m4a (first {DURATION}s)...", flush=True)
subprocess.run([
    "ffmpeg", "-y", "-i", "/Users/yizhoucc/Documents/roro.m4a",
    "-t", str(DURATION), "-ar", "16000", "-ac", "1", "-f", "wav", "/tmp/roro_test.wav"
], capture_output=True)

audio, sr = sf.read("/tmp/roro_test.wav")
audio = audio.astype(np.float32)
print(f"Audio: {len(audio)/sr:.1f}s", flush=True)

from config import config

if config.stt_provider == "whisper_remote":
    from stt.whisper_remote import WhisperRemoteSTT as STTClass
else:
    from stt.whisper_local import WhisperLocalSTT as STTClass

commit_count = 0
partial_count = 0

def on_partial(text):
    global partial_count
    partial_count += 1
    t = time.monotonic() - t0
    short = text[-100:] if len(text) > 100 else text
    print(f"\r\033[K  [{t:5.1f}s] [...] {short}", end="", flush=True)

def on_final(text):
    global commit_count
    commit_count += 1
    t = time.monotonic() - t0
    print(f"\r\033[K  [{t:5.1f}s] [FINAL] {text}", flush=True)

stt = STTClass(on_partial=on_partial, on_final=on_final)
stt._ensure_model()
print("Ready.\n", flush=True)

t0 = time.monotonic()
chunk_size = sr  # 1s

for i in range(0, len(audio), chunk_size):
    chunk = audio[i:i + chunk_size]
    stt.feed_audio(chunk)
    elapsed = time.monotonic() - t0
    target = (i + chunk_size) / sr
    if target > elapsed:
        time.sleep(target - elapsed)

print(f"\n\n  --- Finalizing ---", flush=True)
stt.finalize()
time.sleep(1)

print(f"\n=== Stats: {partial_count} partials, {commit_count} finals ===", flush=True)
if hasattr(stt, '_committed_text'):
    print(f"Committed segments: {len(stt._committed_text)}", flush=True)
    for i, t in enumerate(stt._committed_text):
        print(f"  #{i+1}: {t}", flush=True)
