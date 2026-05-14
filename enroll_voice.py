"""Record your voice for speaker enrollment. Speak for 10-15 seconds."""
import sys
import time
from pathlib import Path
import numpy as np
import sounddevice as sd
import httpx

from audio_utils import save_wav
from config import config

sys.stdout.reconfigure(line_buffering=True)

DURATION = 15
WAV_PATH = Path("my_voice.wav")

print("=== Voice Enrollment ===")
print(f"Press Enter to start recording ({DURATION}s).")
print("Speak naturally in Chinese and/or English.")
input()

print(f"Recording for {DURATION} seconds...", flush=True)
audio = sd.rec(int(DURATION * config.sample_rate), samplerate=config.sample_rate, channels=config.channels, dtype="float32")
for i in range(DURATION, 0, -1):
    print(f"  {i}s remaining...", flush=True)
    time.sleep(1)
sd.wait()
print("Recording done.", flush=True)

save_wav(audio.flatten(), WAV_PATH, config.sample_rate, config.channels)
print(f"Saved to {WAV_PATH}", flush=True)

print("\nSending to server for enrollment...", flush=True)
try:
    with open(WAV_PATH, "rb") as f:
        response = httpx.post(
            f"{config.whisper_remote_url}/enroll",
            files={"audio": ("my_voice.wav", f, "audio/wav")},
            timeout=30.0,
        )
    print(f"Enrollment result: {response.json()}", flush=True)
    print("\nDone! Your voice is now registered.")
except Exception as e:
    print(f"Error: {e}")
    print("Make sure the whisper server is running and SSH tunnel is active.")
