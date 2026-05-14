"""Record your voice for speaker enrollment. Speak for 10-15 seconds."""
import sys
import time
import wave
import numpy as np
import sounddevice as sd
import httpx

sys.stdout.reconfigure(line_buffering=True)

SAMPLE_RATE = 16000
DURATION = 15  # seconds
WAV_PATH = "my_voice.wav"

print("=== Voice Enrollment ===")
print(f"Press Enter to start recording ({DURATION}s).")
print("Speak naturally in Chinese and/or English.")
input()

print(f"Recording for {DURATION} seconds...", flush=True)
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
for i in range(DURATION, 0, -1):
    print(f"  {i}s remaining...", flush=True)
    time.sleep(1)
sd.wait()
print("Recording done.", flush=True)

# Save WAV
audio_flat = audio.flatten()
with wave.open(WAV_PATH, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    pcm = (audio_flat * 32767).clip(-32768, 32767).astype(np.int16)
    wf.writeframes(pcm.tobytes())
print(f"Saved to {WAV_PATH}", flush=True)

# Enroll with server
print("\nSending to server for enrollment...", flush=True)
try:
    with open(WAV_PATH, "rb") as f:
        response = httpx.post(
            "http://localhost:8787/enroll",
            files={"audio": ("my_voice.wav", f, "audio/wav")},
            timeout=30.0,
        )
    result = response.json()
    print(f"Enrollment result: {result}", flush=True)
    print("\nDone! Your voice is now registered.")
except Exception as e:
    print(f"Error: {e}")
    print("Make sure the whisper server is running and SSH tunnel is active.")
