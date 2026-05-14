"""Simulate streaming: feed roro.m4a to STT 1 second at a time, as if speaking live."""
import sys
import time
import subprocess
import numpy as np
import soundfile as sf

sys.stdout.reconfigure(line_buffering=True)

# Convert m4a to wav first
print("Converting roro.m4a to wav...", flush=True)
subprocess.run([
    "ffmpeg", "-y", "-i", "/Users/yizhoucc/Documents/roro.m4a",
    "-t", "120",  # first 2 minutes
    "-ar", "16000", "-ac", "1", "-f", "wav", "/tmp/roro_test.wav"
], capture_output=True)

audio, sr = sf.read("/tmp/roro_test.wav")
audio = audio.astype(np.float32)
total_seconds = len(audio) / sr
print(f"Audio: {total_seconds:.1f}s, {sr}Hz", flush=True)

# Import our STT
from config import config

print(f"Provider: {config.stt_provider}", flush=True)

if config.stt_provider == "whisper_remote":
    from stt.whisper_remote import WhisperRemoteSTT as STTClass
else:
    from stt.whisper_local import WhisperLocalSTT as STTClass

partial_log = []
final_log = []

def on_partial(text):
    t = time.monotonic() - start_time
    # Only show last 80 chars to keep terminal clean
    display = text[-80:] if len(text) > 80 else text
    print(f"\r\033[K  [{t:5.1f}s] [...] {display}", end="", flush=True)
    partial_log.append((t, text))

def on_final(text):
    t = time.monotonic() - start_time
    print(f"\r\033[K  [{t:5.1f}s] [done] {text}", flush=True)
    final_log.append((t, text))

print(f"\nLoading model...", flush=True)
stt = STTClass(on_partial=on_partial, on_final=on_final)
stt._ensure_model()
print("Ready.\n", flush=True)

# Simulate streaming: feed 1 second of audio at a time, with real-time pacing
print("=== Simulated streaming (1s chunks, real-time pace) ===\n", flush=True)
chunk_size = sr  # 1 second
start_time = time.monotonic()

for i in range(0, len(audio), chunk_size):
    chunk = audio[i:i + chunk_size]
    stt.feed_audio(chunk)

    # Wait to simulate real-time (but account for processing time)
    elapsed = time.monotonic() - start_time
    target = (i + chunk_size) / sr
    if target > elapsed:
        time.sleep(target - elapsed)

    audio_pos = (i + chunk_size) / sr
    if audio_pos % 10 == 0:
        print(f"\n  --- {audio_pos:.0f}s of audio fed ---", flush=True)

# Finalize
print("\n\n  --- Finalizing ---", flush=True)
stt.finalize()
time.sleep(1)

# Summary
print("\n\n=== Summary ===\n", flush=True)
print(f"Audio duration: {total_seconds:.1f}s", flush=True)
print(f"Partial updates: {len(partial_log)}", flush=True)
print(f"Commits (finals during stream): {len(final_log)}", flush=True)

if final_log:
    print("\n--- Final transcripts ---", flush=True)
    for t, text in final_log:
        print(f"  [{t:.1f}s] {text}", flush=True)

# Save full log
with open("test_streaming_log.txt", "w") as f:
    f.write("=== Partial updates ===\n")
    for t, text in partial_log:
        f.write(f"[{t:.1f}s] {text}\n")
    f.write("\n=== Final transcripts ===\n")
    for t, text in final_log:
        f.write(f"[{t:.1f}s] {text}\n")
print("\nFull log saved to test_streaming_log.txt", flush=True)
