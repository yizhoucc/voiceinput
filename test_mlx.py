"""Benchmark: MLX Whisper (Mac GPU) vs faster-whisper (Mac CPU) vs 5090 GT."""
import sys, os, time, re
import httpx
import mlx_whisper
from faster_whisper import WhisperModel

sys.stdout.reconfigure(line_buffering=True)

MAX_DURATION = 180
SR = 16000

def char_overlap(gt, pred):
    gt_c = set(re.sub(r"[,，。？?！!\s\[\]我他]", "", gt))
    pred_c = set(re.sub(r"[,，。？?！!\s\[\]我他]", "", pred))
    return len(gt_c & pred_c) / len(gt_c) * 100 if gt_c else 0

recordings = sorted([f"recordings/{f}" for f in os.listdir("recordings") if f.endswith(".wav")])
recordings = [f for f in recordings if 5 < (os.path.getsize(f)-44)/(SR*2) <= MAX_DURATION]
print(f"Testing {len(recordings)} recordings\n")

print("Loading faster-whisper small (CPU int8)...", flush=True)
fw_model = WhisperModel("small", device="cpu", compute_type="int8")

print(f"\n{'File':<28} {'Dur':>4} | {'B:MLX%':>6} {'C:FW%':>5} | {'B time':>6} {'C time':>6}")
print("="*70)

results_b, results_c = [], []

for filepath in recordings:
    dur = (os.path.getsize(filepath)-44)/(SR*2)
    fname = os.path.basename(filepath)[:27]
    try:
        # A: 5090 GT
        r = httpx.post("http://localhost:8787/transcribe",
            files={"audio": open(filepath,"rb")},
            data={"language":"auto","identify_speaker":"false"}, timeout=120.0)
        gt = " ".join(s["text"] for s in r.json()["segments"])
        if len(gt) < 5: continue

        # B: MLX
        t0 = time.monotonic()
        mlx_r = mlx_whisper.transcribe(filepath,
            path_or_hf_repo="mlx-community/whisper-large-v3-turbo")
        b_time = time.monotonic() - t0
        b_pct = char_overlap(gt, mlx_r["text"])

        # C: faster-whisper
        t0 = time.monotonic()
        segs, _ = fw_model.transcribe(filepath, language="zh", beam_size=1, vad_filter=True)
        c_text = " ".join(s.text.strip() for s in segs if s.text.strip())
        c_time = time.monotonic() - t0
        c_pct = char_overlap(gt, c_text)

        results_b.append((b_pct, b_time, dur))
        results_c.append((c_pct, c_time, dur))
        print(f"{fname:<28} {dur:4.0f}s | {b_pct:5.0f}% {c_pct:4.0f}% | {b_time:5.1f}s {c_time:5.1f}s")
    except Exception as e:
        print(f"{fname:<28} ERROR: {e}")

print(f"\n{'='*70}")
if results_b:
    ab = sum(r[0] for r in results_b)/len(results_b)
    ac = sum(r[0] for r in results_c)/len(results_c)
    tb = sum(r[1] for r in results_b)/len(results_b)
    tc = sum(r[1] for r in results_c)/len(results_c)
    print(f"B: MLX large-v3-turbo (Mac GPU):  avg {ab:.0f}% char overlap, {tb:.1f}s/file")
    print(f"C: FW small (Mac CPU):            avg {ac:.0f}% char overlap, {tc:.1f}s/file")
    print(f"A: 5090 large-v3-turbo (GT):      100%")
