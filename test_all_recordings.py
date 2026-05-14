"""Test all recordings: ground truth vs streaming raw vs streaming+polish."""
import sys
import time
import os
import numpy as np
import soundfile as sf
import httpx

sys.stdout.reconfigure(line_buffering=True)

from stt.whisper_remote import WhisperRemoteSTT
from llm.vllm_remote import VLLMPolisher

polisher = VLLMPolisher()

recordings = sorted([f"recordings/{f}" for f in os.listdir("recordings") if f.endswith(".wav")])
print(f"Found {len(recordings)} recordings\n")

results = []

for filepath in recordings:
    size = os.path.getsize(filepath)
    dur = (size - 44) / (16000 * 2)
    if dur < 3 or dur > 130:
        continue

    try:
        # Ground truth
        r = httpx.post("http://localhost:8787/transcribe",
            files={"audio": open(filepath, "rb")},
            data={"language": "auto", "identify_speaker": "false"},
            timeout=60.0)
        gt = " ".join(s["text"] for s in r.json()["segments"])
        if not gt.strip():
            continue

        # Streaming
        raw_commits = []
        def on_partial(text): pass
        def on_final(text): pass
        def on_commit(text): raw_commits.append(text)

        stt = WhisperRemoteSTT(on_partial, on_final, on_commit)
        audio, sr = sf.read(filepath)
        audio = audio.astype(np.float32)
        t0 = time.monotonic()
        for i in range(0, len(audio), sr):
            stt.feed_audio(audio[i:i + sr])
            elapsed = time.monotonic() - t0
            target = (i + sr) / sr
            if target > elapsed:
                time.sleep(target - elapsed)
        stt.finalize()
        time.sleep(0.5)
        raw_text = " ".join(raw_commits)

        # Polish
        polished_commits = []
        for idx, raw in enumerate(raw_commits):
            ctx = polished_commits[-1] if polished_commits else ""
            p = polisher.polish(raw, context_before=ctx)
            polished_commits.append(p)
        polished_text = " ".join(polished_commits)

        # Word overlap
        def words(t):
            return set(t.replace(",", "").replace("，", "").replace("。", "")
                       .replace("?", "").replace("？", "").replace("!", "").split())
        gt_w = words(gt)
        raw_w = words(raw_text)
        pol_w = words(polished_text)
        raw_pct = len(gt_w & raw_w) / len(gt_w) * 100 if gt_w else 0
        pol_pct = len(gt_w & pol_w) / len(gt_w) * 100 if gt_w else 0
        delta = pol_pct - raw_pct

        fname = os.path.basename(filepath)
        print(f"{fname} ({dur:5.0f}s) | GT:{len(gt):3d}ch {len(gt_w):2d}w | Raw:{raw_pct:4.0f}% → Polish:{pol_pct:4.0f}% ({delta:+4.0f}%)")

        if abs(delta) > 2:
            improved = (pol_w & gt_w) - (raw_w & gt_w)
            degraded = (raw_w & gt_w) - (pol_w & gt_w)
            if improved:
                print(f"  + {' '.join(list(improved)[:5])}")
            if degraded:
                print(f"  - {' '.join(list(degraded)[:5])}")

        results.append((fname, dur, raw_pct, pol_pct, delta))

    except Exception as e:
        print(f"{filepath}: ERROR {e}")

print(f"\n{'='*50}")
print(f"SUMMARY: {len(results)} recordings tested")
if results:
    avg_raw = sum(r[2] for r in results) / len(results)
    avg_pol = sum(r[3] for r in results) / len(results)
    avg_delta = sum(r[4] for r in results) / len(results)
    print(f"Average word overlap: Raw {avg_raw:.0f}% → Polish {avg_pol:.0f}% ({avg_delta:+.0f}%)")
    improved = sum(1 for r in results if r[4] > 0)
    same = sum(1 for r in results if r[4] == 0)
    degraded = sum(1 for r in results if r[4] < 0)
    print(f"Improved: {improved} | Same: {same} | Degraded: {degraded}")
