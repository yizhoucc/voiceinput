"""5 minute streaming test: track text length evolution and shrinkage."""
import sys
import time
import subprocess
import numpy as np
import soundfile as sf

sys.stdout.reconfigure(line_buffering=True)

DURATION = 300  # 5 minutes

print(f"Extracting {DURATION}s from roro.m4a (starting at 300s)...", flush=True)
subprocess.run([
    "ffmpeg", "-y", "-i", "/Users/yizhoucc/Documents/roro.m4a",
    "-ss", "300", "-t", str(DURATION),
    "-ar", "16000", "-ac", "1", "-f", "wav", "/tmp/roro_5min.wav"
], capture_output=True)

audio, sr = sf.read("/tmp/roro_5min.wav")
audio = audio.astype(np.float32)
print(f"Audio: {len(audio)/sr:.1f}s\n", flush=True)

from stt.whisper_remote import WhisperRemoteSTT

max_len = 0
shrink_events = []
partial_count = 0

def on_partial(text):
    global max_len, partial_count
    partial_count += 1
    t = time.monotonic() - t0
    tlen = len(text)

    if tlen < max_len:
        lost = max_len - tlen
        shrink_events.append((t, lost, max_len, tlen))
        marker = f" *** SHRUNK by {lost} ***"
    else:
        marker = ""
    max_len = max(max_len, tlen)

    if partial_count % 5 == 0 or marker:
        short = text[-60:] if len(text) > 60 else text
        print(f"  [{t:6.1f}s] len={tlen:5d} max={max_len:5d}{marker} | ...{short}", flush=True)

def on_final(text):
    t = time.monotonic() - t0
    print(f"\n  [{t:6.1f}s] FINAL len={len(text)} max_ever={max_len}", flush=True)

stt = WhisperRemoteSTT(on_partial, on_final)
stt._ensure_model()
print("Ready. Feeding audio...\n", flush=True)

t0 = time.monotonic()
chunk_size = sr

for i in range(0, len(audio), chunk_size):
    stt.feed_audio(audio[i:i + chunk_size])
    elapsed = time.monotonic() - t0
    target = (i + chunk_size) / sr
    if target > elapsed:
        time.sleep(target - elapsed)

    audio_pos = (i + chunk_size) / sr
    if audio_pos % 60 == 0:
        print(f"\n  === {audio_pos:.0f}s fed, {len(stt._committed_text)} committed segments ===\n", flush=True)

print("\nFinalizing...", flush=True)
stt.finalize()
time.sleep(1)

print(f"\n{'='*60}", flush=True)
print(f"=== 5-MINUTE TEST RESULTS ===", flush=True)
print(f"{'='*60}", flush=True)
print(f"Total partials: {partial_count}", flush=True)
print(f"Committed segments: {len(stt._committed_text)}", flush=True)
print(f"Max text length seen: {max_len}", flush=True)
print(f"Shrink events: {len(shrink_events)}", flush=True)
for t, lost, old, new in shrink_events:
    print(f"  [{t:.1f}s] lost {lost} chars ({old} → {new})", flush=True)
print(f"\nFirst 5 committed:", flush=True)
for i, seg in enumerate(stt._committed_text[:5]):
    print(f"  #{i+1}: {seg}", flush=True)
print(f"\nLast 5 committed:", flush=True)
for i, seg in enumerate(stt._committed_text[-5:]):
    print(f"  #{len(stt._committed_text)-4+i}: {seg}", flush=True)
