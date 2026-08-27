#!/usr/bin/env python3
"""Batch-transcribe staged Voice Memos on WSL and optionally polish transcripts."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path


AUDIO_EXTENSIONS = {".m4a", ".qta", ".wav", ".mp3", ".aac", ".flac", ".ogg", ".mov", ".mp4"}
TIMESTAMP_RE = re.compile(r"^\[[0-9:.]+-[0-9:.]+\]", re.MULTILINE)
SPEAKER_RE = re.compile(r"\[Speaker \d+\]")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def seconds_to_stamp(value: float) -> str:
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    seconds = value % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def load_manifest(path: str | None) -> dict[str, dict]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {row.get("path", ""): row for row in data if row.get("path")}
    return data


def convert_to_wav(source: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="voice-memo-", suffix=".wav", delete=False)
    handle.close()
    output = Path(handle.name)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-vn", "-ar", "16000", "-ac", "1", str(output)],
        check=True,
    )
    return output


def load_diarizer():
    import torch
    from pyannote.audio import Pipeline
    from pyannote.audio.models.embedding.wespeaker import WeSpeakerResNet34

    def patched_compute_fbank(self, waveforms):
        waveforms = waveforms.squeeze(1)
        device = waveforms.device
        features = []
        for idx in range(waveforms.shape[0]):
            features.append(self._fbank(waveforms[idx].unsqueeze(0).cpu()))
        return torch.stack(features).to(device)

    WeSpeakerResNet34.compute_fbank = patched_compute_fbank
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    pipeline.to(torch.device("cuda"))
    return pipeline


def diarize(pipeline, wav_path: Path) -> list[dict]:
    import soundfile as sf
    import torch

    audio, sample_rate = sf.read(wav_path, dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    result = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    annotation = result.speaker_diarization
    raw = [(turn.start, turn.end, label) for turn, _, label in annotation.itertracks(yield_label=True)]
    labels = []
    for _, _, label in raw:
        if label not in labels:
            labels.append(label)
    label_map = {label: f"Speaker {idx}" for idx, label in enumerate(labels)}
    return [{"start": start, "end": end, "speaker": label_map[label]} for start, end, label in raw]


def assign_speaker(start: float, end: float, diar_segments: list[dict]) -> str:
    if not diar_segments:
        return "Speaker 0"
    best = max(
        diar_segments,
        key=lambda seg: max(0.0, min(end, seg["end"]) - max(start, seg["start"])),
    )
    overlap = max(0.0, min(end, best["end"]) - max(start, best["start"]))
    if overlap == 0:
        midpoint = (start + end) / 2
        best = min(diar_segments, key=lambda seg: abs(midpoint - (seg["start"] + seg["end"]) / 2))
    return best["speaker"]


def transcribe_command(args) -> None:
    from faster_whisper import BatchedInferencePipeline, WhisperModel
    from opencc import OpenCC

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    files = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)
    print(f"[setup] files={len(files)} model={args.model}", flush=True)

    model = WhisperModel(args.model, device="cuda", compute_type="float16")
    batched = BatchedInferencePipeline(model=model)
    diarizer = None if args.no_speakers else load_diarizer()
    converter = OpenCC("t2s")

    completed = 0
    failed = 0
    started = time.time()
    for index, source in enumerate(files, 1):
        raw_path = output_dir / f"{source.name}.raw.txt"
        json_path = output_dir / f"{source.name}.json"
        if raw_path.exists() and json_path.exists() and not args.force:
            completed += 1
            print(f"[skip] {index}/{len(files)} {source.name}", flush=True)
            continue

        item_started = time.time()
        wav_path = None
        try:
            wav_path = convert_to_wav(source)
            diar_segments = [] if diarizer is None else diarize(diarizer, wav_path)
            language = None if args.language == "auto" else args.language
            segments_iter, info = batched.transcribe(
                str(wav_path),
                language=language,
                task="transcribe",
                beam_size=args.beam_size,
                batch_size=args.batch_size,
                vad_filter=True,
                condition_on_previous_text=True,
                without_timestamps=False,
            )
            segments = []
            for segment in segments_iter:
                text = converter.convert(segment.text.strip())
                if not text:
                    continue
                segments.append({
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "speaker": assign_speaker(float(segment.start), float(segment.end), diar_segments),
                    "text": text,
                })

            meta = manifest.get(source.name, {})
            title = meta.get("title") or source.stem
            duration = float(meta.get("duration") or (segments[-1]["end"] if segments else 0.0))
            header = [
                f"# Title: {title}",
                f"# Source: {source.name}",
                f"# Duration: {duration:.3f} seconds",
                f"# Language: {info.language}",
                "",
            ]
            lines = header + [
                f"[{seconds_to_stamp(seg['start'])}-{seconds_to_stamp(seg['end'])}] [{seg['speaker']}] {seg['text']}"
                for seg in segments
            ]
            atomic_write(raw_path, "\n".join(lines).rstrip() + "\n")
            payload = {
                "title": title,
                "source": source.name,
                "duration": duration,
                "language": info.language,
                "language_probability": info.language_probability,
                "segments": segments,
            }
            atomic_write(json_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            completed += 1
            elapsed = time.time() - item_started
            total_elapsed = time.time() - started
            rate = completed / total_elapsed if total_elapsed else 0
            eta = (len(files) - completed) / rate if rate else 0
            print(f"[done] {index}/{len(files)} {source.name} segments={len(segments)} elapsed={elapsed:.1f}s eta={eta/60:.1f}m", flush=True)
        except Exception as exc:
            failed += 1
            print(f"[error] {index}/{len(files)} {source.name}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            if wav_path and wav_path.exists():
                wav_path.unlink()

    print(f"[summary] completed={completed} failed={failed} total={len(files)} elapsed={(time.time()-started)/60:.1f}m", flush=True)
    if failed:
        raise SystemExit(2)


POLISH_PROMPT = """你是语音转录校对助手。只修正高度确定的语音识别错误、错别字、繁简体和标点。
每行开头的标题、时间戳和说话人标签必须原样保留。
严格保持原意、行数和信息量，不概括、不扩写、不删除；无法确定时必须保留原文。
只输出校对后的全文。"""


def split_chunks(text: str, max_chars: int) -> list[str]:
    lines = text.splitlines(keepends=True)
    chunks = []
    current = []
    size = 0
    for line in lines:
        if current and size + len(line) > max_chars:
            chunks.append("".join(current))
            current = []
            size = 0
        current.append(line)
        size += len(line)
    if current:
        chunks.append("".join(current))
    return chunks


def polish_chunk(url: str, model: str, text: str, timeout: int) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": POLISH_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0.0,
        "max_tokens": min(3600, max(512, int(len(text) * 1.35))),
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    polished = result["choices"][0]["message"]["content"].strip() + "\n"
    original_timestamps = TIMESTAMP_RE.findall(text)
    polished_timestamps = TIMESTAMP_RE.findall(polished)
    original_speakers = SPEAKER_RE.findall(text)
    polished_speakers = SPEAKER_RE.findall(polished)
    ratio = len(polished) / max(1, len(text))
    if original_timestamps != polished_timestamps or original_speakers != polished_speakers or not 0.65 <= ratio <= 1.35:
        raise ValueError("polished chunk failed structure/length validation")
    return polished


def polish_one(source: Path, output_dir: Path, args) -> tuple[str, bool, str]:
    output = output_dir / source.name.replace(".raw.txt", ".polished.txt")
    if output.exists() and not args.force:
        return source.name, True, "skip"
    try:
        raw = source.read_text(encoding="utf-8")
        chunks = split_chunks(raw, args.max_chars)
        polished = "".join(polish_chunk(args.url, args.model, chunk, args.timeout) for chunk in chunks)
        atomic_write(output, polished)
        return source.name, True, f"chunks={len(chunks)}"
    except Exception as exc:
        return source.name, False, f"{type(exc).__name__}: {exc}"


def polish_command(args) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("*.raw.txt"))
    print(f"[setup] files={len(files)} concurrency={args.concurrency}", flush=True)
    completed = 0
    failed = 0
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(polish_one, source, output_dir, args): source for source in files}
        for future in concurrent.futures.as_completed(futures):
            name, ok, message = future.result()
            completed += int(ok)
            failed += int(not ok)
            print(f"[{'done' if ok else 'error'}] {completed+failed}/{len(files)} {name} {message}", flush=True)
    print(f"[summary] completed={completed} failed={failed} total={len(files)} elapsed={(time.time()-started)/60:.1f}m", flush=True)
    if failed:
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    transcribe = sub.add_parser("transcribe")
    transcribe.add_argument("input_dir")
    transcribe.add_argument("output_dir")
    transcribe.add_argument("--manifest")
    transcribe.add_argument("--model", default=str(Path.home() / "models/faster-whisper-large-v3"))
    transcribe.add_argument("--language", default="auto", choices=["auto", "zh", "en"])
    transcribe.add_argument("--batch-size", type=int, default=16)
    transcribe.add_argument("--beam-size", type=int, default=5)
    transcribe.add_argument("--no-speakers", action="store_true")
    transcribe.add_argument("--force", action="store_true")
    transcribe.set_defaults(func=transcribe_command)

    polish = sub.add_parser("polish")
    polish.add_argument("input_dir")
    polish.add_argument("output_dir")
    polish.add_argument("--url", default="http://127.0.0.1:8000")
    polish.add_argument("--model", default="qwen-polish")
    polish.add_argument("--concurrency", type=int, default=8)
    polish.add_argument("--max-chars", type=int, default=3000)
    polish.add_argument("--timeout", type=int, default=300)
    polish.add_argument("--force", action="store_true")
    polish.set_defaults(func=polish_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
