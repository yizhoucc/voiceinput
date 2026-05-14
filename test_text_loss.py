"""Debug test: trace committed_text evolution to find where text gets lost."""
import sys
import time
import subprocess
import numpy as np
import soundfile as sf

sys.stdout.reconfigure(line_buffering=True)

DURATION = 60

subprocess.run([
    "ffmpeg", "-y", "-i", "/Users/yizhoucc/Documents/roro.m4a",
    "-ss", "300", "-t", str(DURATION),
    "-ar", "16000", "-ac", "1", "-f", "wav", "/tmp/roro_debug.wav"
], capture_output=True)

audio, sr = sf.read("/tmp/roro_debug.wav")
audio = audio.astype(np.float32)
print(f"Audio: {len(audio)/sr:.1f}s\n", flush=True)

from stt.whisper_remote import WhisperRemoteSTT

partial_history = []
final_text = None

def on_partial(text):
    t = time.monotonic() - t0
    partial_history.append((t, len(text), text))
    short = text[-80:] if len(text) > 80 else text
    print(f"\r\033[K  [{t:5.1f}s] len={len(text):4d} | ...{short}", end="", flush=True)

def on_final(text):
    global final_text
    t = time.monotonic() - t0
    final_text = text
    print(f"\n  [{t:5.1f}s] FINAL len={len(text)}", flush=True)

stt = WhisperRemoteSTT(on_partial, on_final)
stt._ensure_model()

t0 = time.monotonic()
chunk_size = sr

for i in range(0, len(audio), chunk_size):
    stt.feed_audio(audio[i:i + chunk_size])
    elapsed = time.monotonic() - t0
    target = (i + chunk_size) / sr
    if target > elapsed:
        time.sleep(target - elapsed)

print("\n\nFinalizing...", flush=True)
stt.finalize()
time.sleep(1)

# Analysis
print("\n=== Analysis ===\n", flush=True)

print("Committed segments:", flush=True)
for i, seg in enumerate(stt._committed_text):
    print(f"  #{i+1}: {seg}", flush=True)

print(f"\nPartial text length over time:", flush=True)
prev_len = 0
for t, length, text in partial_history:
    direction = "+" if length >= prev_len else "SHRUNK!"
    if length < prev_len:
        print(f"  [{t:5.1f}s] len={length:4d} ({direction} lost {prev_len - length} chars)", flush=True)
        # Show what was lost
        prev_text = [p[2] for p in partial_history if p[1] == prev_len]
        if prev_text:
            old = prev_text[-1]
            # Find the missing prefix
            for j in range(min(len(old), len(text))):
                if j >= len(text) or old[j] != text[j]:
                    print(f"         OLD: {old[:j+20]}...", flush=True)
                    print(f"         NEW: {text[:j+20]}...", flush=True)
                    break
    prev_len = length

if final_text:
    print(f"\nFinal text ({len(final_text)} chars):", flush=True)
    print(f"  {final_text[:200]}...", flush=True)
