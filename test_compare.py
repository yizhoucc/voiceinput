"""Compare ground truth (full-file) vs streaming transcription."""
import sys
import time
import numpy as np
import soundfile as sf
import httpx

sys.stdout.reconfigure(line_buffering=True)

from stt.whisper_remote import WhisperRemoteSTT

test_files = [
    "recordings/20260513_233924.wav",  # 32s
    "recordings/20260513_234126.wav",  # 72s
]

for filepath in test_files:
    audio, sr = sf.read(filepath)
    audio = audio.astype(np.float32)
    dur = len(audio) / sr
    print(f"\n{'='*60}")
    print(f"=== {filepath} ({dur:.1f}s) ===")

    # 1. Ground truth
    r = httpx.post("http://localhost:8787/transcribe",
        files={"audio": open(filepath, "rb")},
        data={"language": "auto", "identify_speaker": "false"},
        timeout=60.0)
    gt_segs = r.json()["segments"]
    gt_text = " ".join(s["text"] for s in gt_segs)
    print(f"\nGround truth ({len(gt_text)} chars):")
    print(f"  {gt_text[:200]}...")

    # 2. Streaming simulation
    commits = []

    def on_partial(text):
        pass

    def on_final(text):
        pass

    def on_commit(text):
        t = time.monotonic() - t0
        commits.append((t, text))

    stt = WhisperRemoteSTT(on_partial, on_final, on_commit)
    t0 = time.monotonic()

    for i in range(0, len(audio), sr):
        stt.feed_audio(audio[i:i + sr])
        elapsed = time.monotonic() - t0
        target = (i + sr) / sr
        if target > elapsed:
            time.sleep(target - elapsed)

    stt.finalize()
    time.sleep(1)

    streaming_text = " ".join(c[1] for c in commits)
    print(f"\nStreaming ({len(streaming_text)} chars, {len(commits)} commits):")
    print(f"  {streaming_text[:200]}...")

    # 3. Comparison
    print(f"\n--- Commits timeline ---")
    for t, text in commits:
        print(f"  [{t:5.1f}s] {text[:60]}")

    # Simple similarity: what % of ground truth words are in streaming
    gt_words = set(gt_text.replace(",", " ").replace("。", " ").split())
    st_words = set(streaming_text.replace(",", " ").replace("。", " ").split())
    if gt_words:
        overlap = len(gt_words & st_words) / len(gt_words)
        missing = gt_words - st_words
        extra = st_words - gt_words
        print(f"\n  Word overlap: {overlap:.0%}")
        if missing:
            print(f"  Missing from streaming: {' '.join(list(missing)[:10])}")
        if extra:
            print(f"  Extra in streaming: {' '.join(list(extra)[:10])}")
