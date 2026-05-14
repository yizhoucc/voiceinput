"""Test Qwen2-Audio-7B: direct transcription vs polish-only vs whisper baseline.

Tests on all recordings:
  A: Whisper large-v3-turbo (ground truth baseline)
  B: Qwen2-Audio direct transcription (audio → text, no whisper)
  C: Whisper + Qwen2-Audio polish (whisper粗转录 + audio → corrected text)
"""
import sys
import os
import time
import subprocess
import httpx

sys.stdout.reconfigure(line_buffering=True)

MAX_DURATION = 180  # 3 minutes max
SR = 16000


def words(text):
    return set(text.replace(",", "").replace("，", "").replace("。", "")
               .replace("?", "").replace("？", "").replace("!", "")
               .replace("[我]", "").replace("[他]", "").split())


def get_recordings():
    files = sorted([f"recordings/{f}" for f in os.listdir("recordings") if f.endswith(".wav")])
    return [f for f in files if 5 < (os.path.getsize(f) - 44) / (SR * 2) <= MAX_DURATION]


def prepare_audio(filepath):
    """Ensure audio is ≤3min WAV at 16kHz."""
    size = os.path.getsize(filepath)
    dur = (size - 44) / (SR * 2)
    if dur <= MAX_DURATION:
        return filepath, dur
    tmp = f"/tmp/voiceinput_trim_{os.path.basename(filepath)}"
    subprocess.run([
        "ffmpeg", "-y", "-i", filepath, "-t", str(MAX_DURATION),
        "-ar", "16000", "-ac", "1", "-f", "wav", tmp
    ], capture_output=True)
    return tmp, MAX_DURATION


def whisper_transcribe(filepath):
    """Ground truth: full file via whisper server."""
    r = httpx.post("http://localhost:8787/transcribe",
        files={"audio": open(filepath, "rb")},
        data={"language": "auto", "identify_speaker": "false"},
        timeout=300.0)
    r.raise_for_status()
    return " ".join(s["text"] for s in r.json()["segments"])


def qwen2audio_transcribe(filepath):
    """Direct transcription via Qwen2-Audio server."""
    r = httpx.post("http://localhost:8786/transcribe",
        files={"audio": open(filepath, "rb")},
        timeout=300.0)
    r.raise_for_status()
    return r.json().get("text", "")


def qwen2audio_polish(filepath, whisper_text):
    """Polish whisper output using Qwen2-Audio (audio + text → corrected)."""
    r = httpx.post("http://localhost:8786/polish",
        files={"audio": open(filepath, "rb")},
        data={"whisper_text": whisper_text},
        timeout=300.0)
    r.raise_for_status()
    return r.json().get("text", "")


def main():
    recordings = get_recordings()
    print(f"Testing {len(recordings)} recordings\n")

    print(f"{'File':<28} {'Dur':>4} | {'A:whisper':>9} {'B:qwen2直接':>11} {'C:whisper+qwen2润色':>19}")
    print(f"{'':<28} {'':>4} | {'(GT)':>9} {'overlap%':>11} {'overlap%':>19}")
    print("=" * 80)

    results_b = []
    results_c = []

    for filepath in recordings:
        wav_path, dur = prepare_audio(filepath)
        fname = os.path.basename(filepath)[:27]

        try:
            # A: Whisper ground truth
            gt = whisper_transcribe(wav_path)
            gt_w = words(gt)
            if len(gt_w) < 2:
                continue

            # B: Qwen2-Audio direct
            try:
                b_text = qwen2audio_transcribe(wav_path)
                b_w = words(b_text)
                b_pct = len(gt_w & b_w) / len(gt_w) * 100
                results_b.append(b_pct)
                b_str = f"{b_pct:.0f}%"
            except Exception as e:
                b_str = f"ERR"

            # C: Whisper + Qwen2-Audio polish
            try:
                c_text = qwen2audio_polish(wav_path, gt)
                c_w = words(c_text)
                c_pct = len(gt_w & c_w) / len(gt_w) * 100
                results_c.append(c_pct)
                c_str = f"{c_pct:.0f}%"
            except Exception as e:
                c_str = f"ERR"

            print(f"{fname:<28} {dur:4.0f}s | {'100%':>9} {b_str:>11} {c_str:>19}")

        except Exception as e:
            print(f"{fname:<28} ERROR: {e}")

        if wav_path != filepath:
            os.unlink(wav_path)

    print("\n" + "=" * 80)
    if results_b:
        print(f"B (Qwen2-Audio direct):      avg {sum(results_b)/len(results_b):.1f}% ({len(results_b)} files)")
    if results_c:
        print(f"C (Whisper+Qwen2-Audio polish): avg {sum(results_c)/len(results_c):.1f}% ({len(results_c)} files)")
    print(f"A (Whisper large-v3-turbo):    100% (ground truth baseline)")


if __name__ == "__main__":
    main()
