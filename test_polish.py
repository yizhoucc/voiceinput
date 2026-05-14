"""Compare: ground truth vs streaming raw vs streaming+LLM polish."""
import sys
import time
import numpy as np
import soundfile as sf
import httpx

sys.stdout.reconfigure(line_buffering=True)

from stt.whisper_remote import WhisperRemoteSTT
from llm.vllm_remote import VLLMPolisher

test_files = [
    "recordings/20260513_233924.wav",  # 32s
    "recordings/20260513_234126.wav",  # 72s
]

polisher = VLLMPolisher()

for filepath in test_files:
    audio, sr = sf.read(filepath)
    audio = audio.astype(np.float32)
    dur = len(audio) / sr
    print(f"\n{'='*70}")
    print(f"=== {filepath} ({dur:.1f}s) ===")

    # 1. Ground truth (full file, no streaming)
    r = httpx.post("http://localhost:8787/transcribe",
        files={"audio": open(filepath, "rb")},
        data={"language": "auto", "identify_speaker": "false"},
        timeout=60.0)
    gt_text = " ".join(s["text"] for s in r.json()["segments"])

    # 2. Streaming simulation
    commits_raw = []

    def on_partial(text):
        pass

    def on_final(text):
        pass

    def on_commit(text):
        commits_raw.append(text)

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

    streaming_raw = " ".join(commits_raw)

    # 3. LLM polish on streaming commits
    polished_commits = []
    for idx in range(len(commits_raw)):
        raw = commits_raw[idx]
        ctx_b = commits_raw[idx - 1] if idx > 0 else ""
        ctx_a = commits_raw[idx + 1] if idx + 1 < len(commits_raw) else ""
        p = polisher.polish(raw, context_before=ctx_b, context_after=ctx_a)
        polished_commits.append(p)

    streaming_polished = " ".join(polished_commits)

    # Print comparison
    print(f"\n1. GROUND TRUTH ({len(gt_text)} chars):")
    print(f"   {gt_text[:300]}")

    print(f"\n2. STREAMING RAW ({len(streaming_raw)} chars, {len(commits_raw)} commits):")
    print(f"   {streaming_raw[:300]}")

    print(f"\n3. STREAMING + POLISH ({len(streaming_polished)} chars):")
    print(f"   {streaming_polished[:300]}")

    # Word-level comparison
    def word_set(text):
        return set(text.replace(",", " ").replace("，", " ").replace("。", " ")
                   .replace("?", " ").replace("？", " ").replace("!", " ").split())

    gt_words = word_set(gt_text)
    raw_words = word_set(streaming_raw)
    pol_words = word_set(streaming_polished)

    if gt_words:
        raw_overlap = len(gt_words & raw_words) / len(gt_words)
        pol_overlap = len(gt_words & pol_words) / len(gt_words)
        print(f"\n   Word overlap with ground truth:")
        print(f"     Raw streaming:    {raw_overlap:.0%}")
        print(f"     After LLM polish: {pol_overlap:.0%}")

        # Key improvements
        fixed = (pol_words & gt_words) - (raw_words & gt_words)
        if fixed:
            print(f"     Fixed by LLM: {' '.join(list(fixed)[:10])}")
