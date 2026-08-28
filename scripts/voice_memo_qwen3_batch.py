#!/usr/bin/env python3
"""High-quality Voice Memo transcription with Qwen3-ASR and word-level diarization."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path


AUDIO_EXTENSIONS = {".m4a", ".qta", ".wav", ".mp3", ".aac", ".flac", ".ogg", ".mov", ".mp4"}
SENTENCE_END_RE = re.compile(r"[。！？!?；;](?:[\"'”’）)])?$|(?<!\b[A-Z])[.](?:[\"'”’）)])?$")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def stamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remainder = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remainder:06.3f}"


def load_manifest(path: str | None) -> dict[str, dict]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {row["path"]: row for row in data if row.get("path")}
    return data


def convert_to_wav(source: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="voice-qwen3-", suffix=".wav", delete=False)
    handle.close()
    output = Path(handle.name)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-vn", "-ar", "16000", "-ac", "1", str(output)],
        check=True,
    )
    return output


def load_audio(path: Path):
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    return audio, sample_rate


def build_windows(audio, sample_rate: int, max_seconds: float) -> list[tuple[int, int]]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(
        min_speech_duration_ms=200,
        min_silence_duration_ms=550,
        speech_pad_ms=350,
        max_speech_duration_s=max_seconds - 5,
    )
    speech = get_speech_timestamps(audio, options)
    if not speech:
        return []

    maximum = int(max_seconds * sample_rate)
    windows = []
    start = speech[0]["start"]
    end = speech[0]["end"]
    for segment in speech[1:]:
        if segment["end"] - start <= maximum:
            end = segment["end"]
        else:
            windows.append((start, end))
            start, end = segment["start"], segment["end"]
    windows.append((start, end))
    return windows


def load_diarizer():
    import torch
    from pyannote.audio import Pipeline
    from pyannote.audio.models.embedding.wespeaker import WeSpeakerResNet34

    def patched_compute_fbank(self, waveforms):
        waveforms = waveforms.squeeze(1)
        device = waveforms.device
        features = [self._fbank(waveforms[i].unsqueeze(0).cpu()) for i in range(waveforms.shape[0])]
        return torch.stack(features).to(device)

    WeSpeakerResNet34.compute_fbank = patched_compute_fbank
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
    pipeline.to(torch.device("cuda"))
    return pipeline


def diarize(pipeline, audio, sample_rate: int, min_speakers: int, max_speakers: int) -> list[dict]:
    import torch

    waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    result = pipeline(
        {"waveform": waveform, "sample_rate": sample_rate},
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    annotation = result.speaker_diarization
    raw = [(turn.start, turn.end, label) for turn, _, label in annotation.itertracks(yield_label=True)]
    durations = {}
    for start, end, label in raw:
        durations[label] = durations.get(label, 0.0) + end - start
    ordered = sorted(durations, key=lambda label: (-durations[label], label))
    label_map = {label: f"Speaker {index}" for index, label in enumerate(ordered)}
    return [
        {"start": float(start), "end": float(end), "speaker": label_map[label]}
        for start, end, label in raw
    ]


def speaker_for(start: float, end: float, turns: list[dict]) -> str:
    if not turns:
        return "Unknown"
    best = max(turns, key=lambda turn: max(0.0, min(end, turn["end"]) - max(start, turn["start"])))
    if min(end, best["end"]) <= max(start, best["start"]):
        midpoint = (start + end) / 2
        best = min(turns, key=lambda turn: abs(midpoint - (turn["start"] + turn["end"]) / 2))
    return best["speaker"]


def timestamp_dict(item) -> dict:
    if isinstance(item, dict):
        return item
    result = {}
    for key in ("text", "start_time", "end_time", "start", "end"):
        if hasattr(item, key):
            result[key] = getattr(item, key)
    return result


def decorate_tokens(full_text: str, timestamp_items, offset: float, turns: list[dict]) -> list[dict]:
    stamps = [timestamp_dict(item) for item in timestamp_items]
    positions = []
    cursor = 0
    lower = full_text.lower()
    for item in stamps:
        token = str(item.get("text", ""))
        position = full_text.find(token, cursor)
        if position < 0:
            position = lower.find(token.lower(), cursor)
        if position < 0:
            position = cursor
        positions.append(position)
        cursor = max(cursor, position + len(token))

    tokens = []
    for index, item in enumerate(stamps):
        token = str(item.get("text", ""))
        start_position = positions[index]
        if index + 1 < len(stamps) and positions[index + 1] >= start_position:
            display = full_text[start_position:positions[index + 1]]
        else:
            display = full_text[start_position:]
        if token and token not in display and token.lower() not in display.lower():
            display = token
        start = offset + float(item.get("start_time", item.get("start", 0.0)))
        end = offset + float(item.get("end_time", item.get("end", start - offset)))
        tokens.append(
            {
                "start": start,
                "end": end,
                "text": display,
                "speaker": speaker_for(start, end, turns),
            }
        )
    return tokens


def regroup(tokens: list[dict], respect_speaker: bool) -> list[dict]:
    sentences = []
    current = None

    def flush():
        nonlocal current
        if current and current["text"].strip():
            current["text"] = current["text"].strip()
            sentences.append(current)
        current = None

    for token in tokens:
        gap = token["start"] - current["end"] if current else 0.0
        if current and ((respect_speaker and token["speaker"] != current["speaker"]) or gap >= 1.2):
            flush()
        if current is None:
            current = {
                "start": token["start"],
                "end": token["end"],
                "speaker": token["speaker"],
                "text": token["text"],
            }
        else:
            current["end"] = token["end"]
            current["text"] += token["text"]

        duration = current["end"] - current["start"]
        if SENTENCE_END_RE.search(current["text"].rstrip()) or duration >= 20 or len(current["text"]) >= 180:
            flush()
    flush()
    return sentences


def write_outputs(source: Path, output_dir: Path, meta: dict, payload: dict) -> None:
    title = meta.get("title") or source.stem
    duration = float(meta.get("duration") or payload["duration"])
    header = [
        f"# Title: {title}",
        f"# Source: {source.name}",
        f"# Duration: {duration:.3f} seconds",
        "# Model: Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B",
        "",
    ]
    clean_lines = header + [
        f"[{stamp(sentence['start'])}-{stamp(sentence['end'])}] {sentence['text']}"
        for sentence in payload["clean_sentences"]
    ]
    speaker_lines = header + [
        f"[{stamp(sentence['start'])}-{stamp(sentence['end'])}] [{sentence['speaker']}] {sentence['text']}"
        for sentence in payload["speaker_sentences"]
    ]
    atomic_write(output_dir / "clean" / f"{source.name}.qwen3.txt", "\n".join(clean_lines).rstrip() + "\n")
    atomic_write(output_dir / "speakers" / f"{source.name}.qwen3.speakers.txt", "\n".join(speaker_lines).rstrip() + "\n")
    atomic_write(
        output_dir / "metadata" / f"{source.name}.qwen3.json",
        json.dumps({"title": title, "source": source.name, **payload}, ensure_ascii=False, indent=2) + "\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--manifest")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--aligner", default="Qwen/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--chunk-seconds", type=float, default=240.0)
    parser.add_argument("--min-speakers", type=int, default=1)
    parser.add_argument("--max-speakers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import soundfile as sf
    import torch
    from qwen_asr import Qwen3ASRModel

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    manifest = load_manifest(args.manifest)
    files = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS)
    print(f"[setup] files={len(files)} model={args.model}", flush=True)

    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=1,
        max_new_tokens=4096,
        forced_aligner=args.aligner,
        forced_aligner_kwargs={"dtype": torch.bfloat16, "device_map": "cuda:0"},
    )
    diarizer = load_diarizer()

    completed = 0
    failed = 0
    started = time.time()
    for file_index, source in enumerate(files, 1):
        required = [
            output_dir / "clean" / f"{source.name}.qwen3.txt",
            output_dir / "speakers" / f"{source.name}.qwen3.speakers.txt",
            output_dir / "metadata" / f"{source.name}.qwen3.json",
        ]
        if all(path.exists() and path.stat().st_size for path in required) and not args.force:
            completed += 1
            print(f"[skip] {file_index}/{len(files)} {source.name}", flush=True)
            continue

        item_started = time.time()
        wav = None
        try:
            wav = convert_to_wav(source)
            audio, sample_rate = load_audio(wav)
            duration = len(audio) / sample_rate
            windows = build_windows(audio, sample_rate, args.chunk_seconds)
            turns = diarize(diarizer, audio, sample_rate, args.min_speakers, args.max_speakers)
            tokens = []
            window_data = []
            for window_index, (start_sample, end_sample) in enumerate(windows, 1):
                offset = start_sample / sample_rate
                clip_handle = tempfile.NamedTemporaryFile(prefix="qwen-window-", suffix=".wav", delete=False)
                clip_handle.close()
                clip_path = Path(clip_handle.name)
                try:
                    sf.write(clip_path, audio[start_sample:end_sample], sample_rate)
                    result = model.transcribe(audio=str(clip_path), language=None, return_time_stamps=True)[0]
                    chunk_tokens = decorate_tokens(result.text, result.time_stamps, offset, turns)
                    tokens.extend(chunk_tokens)
                    window_data.append(
                        {
                            "start": offset,
                            "end": end_sample / sample_rate,
                            "language": result.language,
                            "text": result.text,
                            "token_count": len(chunk_tokens),
                        }
                    )
                    print(
                        f"[chunk] {file_index}/{len(files)} {source.name} "
                        f"{window_index}/{len(windows)} language={result.language} tokens={len(chunk_tokens)}",
                        flush=True,
                    )
                finally:
                    clip_path.unlink(missing_ok=True)

            clean_sentences = regroup(tokens, respect_speaker=False)
            speaker_sentences = regroup(tokens, respect_speaker=True)
            payload = {
                "duration": duration,
                "windows": window_data,
                "diarization_turns": turns,
                "tokens": tokens,
                "clean_sentences": clean_sentences,
                "speaker_sentences": speaker_sentences,
            }
            write_outputs(source, output_dir, manifest.get(source.name, {}), payload)
            completed += 1
            elapsed = time.time() - item_started
            total_elapsed = time.time() - started
            rate = completed / total_elapsed if total_elapsed else 0
            eta = (len(files) - completed) / rate if rate else 0
            print(
                f"[done] {file_index}/{len(files)} {source.name} windows={len(windows)} "
                f"sentences={len(clean_sentences)} elapsed={elapsed:.1f}s eta={eta/60:.1f}m",
                flush=True,
            )
        except Exception as exc:
            failed += 1
            print(f"[error] {file_index}/{len(files)} {source.name}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            if wav:
                wav.unlink(missing_ok=True)

    print(f"[summary] completed={completed} failed={failed} total={len(files)} elapsed={(time.time()-started)/60:.1f}m", flush=True)
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
