"""Batch transcription CLI: transcribe audio files to text documents.

Usage:
  python transcribe.py recording.wav              # single file
  python transcribe.py recordings/                 # entire directory
  python transcribe.py recordings/ -o output.txt   # custom output path
  python transcribe.py recording.wav --local       # use Mac CPU instead of 5090
  python transcribe.py recording.wav --language zh  # force language
"""
import argparse
import sys
import os
import subprocess
from pathlib import Path
from datetime import timedelta

sys.stdout.reconfigure(line_buffering=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe audio files to text")
    parser.add_argument("path", help="Audio file or directory of audio files")
    parser.add_argument("-o", "--output", help="Output file path (default: <input>.transcript.txt)")
    parser.add_argument("--local", action="store_true", help="Use local Mac CPU whisper (no 5090)")
    parser.add_argument("--language", type=str, default=None, help="Force language: zh, en, or auto")
    parser.add_argument("--speaker", action="store_true", help="Enable speaker identification")
    parser.add_argument("--format", choices=["txt", "srt", "tsv"], default="txt",
                        help="Output format: txt (plain), srt (subtitles), tsv (timestamps)")
    return parser.parse_args()


def transcribe_remote(filepath, language=None, speaker=False):
    import httpx
    from server_manager import ensure_servers, ensure_ssh_tunnel

    ensure_ssh_tunnel(8787)
    if not ensure_servers(need_llm=False):
        print("[error] Cannot reach 5090. Use --local.", flush=True)
        sys.exit(1)

    lang = language or "auto"
    r = httpx.post("http://localhost:8787/transcribe",
        files={"audio": open(filepath, "rb")},
        data={
            "language": lang,
            "identify_speaker": "true" if speaker else "false",
        },
        timeout=300.0)
    r.raise_for_status()
    return r.json()


def transcribe_local(filepath, language=None):
    import numpy as np
    import soundfile as sf
    from faster_whisper import WhisperModel

    audio, sr = sf.read(filepath)
    audio = audio.astype(np.float32)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        audio, language=language, beam_size=5, vad_filter=True,
        vad_parameters=dict(threshold=0.5, min_silence_duration_ms=500))

    result_segments = []
    for seg in segments:
        result_segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "speaker": "unknown",
        })
    return {
        "segments": result_segments,
        "language": info.language,
        "language_probability": info.language_probability,
    }


def format_time_srt(seconds):
    td = timedelta(seconds=seconds)
    hours = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    secs = int(td.total_seconds() % 60)
    ms = int((td.total_seconds() % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def format_output(data, fmt, speaker_enabled):
    segments = data["segments"]
    lang = data.get("language", "?")
    lang_prob = data.get("language_probability", 0)
    lines = []

    if fmt == "txt":
        lines.append(f"# Language: {lang} ({lang_prob:.0%})\n")
        for seg in segments:
            text = seg["text"]
            if not text:
                continue
            if speaker_enabled and seg.get("speaker") in ("me", "other"):
                tag = "[我]" if seg["speaker"] == "me" else "[他]"
                lines.append(f"{tag} {text}")
            else:
                lines.append(text)
        return "\n".join(lines) + "\n"

    elif fmt == "srt":
        idx = 1
        for seg in segments:
            if not seg["text"]:
                continue
            start = format_time_srt(seg["start"])
            end = format_time_srt(seg["end"])
            text = seg["text"]
            if speaker_enabled and seg.get("speaker") in ("me", "other"):
                tag = "[我]" if seg["speaker"] == "me" else "[他]"
                text = f"{tag} {text}"
            lines.append(f"{idx}")
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")
            idx += 1
        return "\n".join(lines)

    elif fmt == "tsv":
        lines.append("start\tend\tspeaker\ttext")
        for seg in segments:
            if not seg["text"]:
                continue
            spk = seg.get("speaker", "unknown")
            lines.append(f"{seg['start']:.2f}\t{seg['end']:.2f}\t{spk}\t{seg['text']}")
        return "\n".join(lines) + "\n"


def get_audio_files(path):
    p = Path(path)
    exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".mp4", ".webm"}
    if p.is_file():
        return [p]
    elif p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in exts)
        return files
    else:
        print(f"[error] Not found: {path}")
        sys.exit(1)


def convert_to_wav(filepath):
    """Convert non-wav to temp wav using ffmpeg."""
    if filepath.suffix.lower() == ".wav":
        return filepath, False
    tmp = Path(f"/tmp/voiceinput_convert_{filepath.stem}.wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(filepath),
        "-ar", "16000", "-ac", "1", "-f", "wav", str(tmp)
    ], capture_output=True)
    return tmp, True


def main():
    args = parse_args()
    files = get_audio_files(args.path)

    if not files:
        print("[error] No audio files found.")
        sys.exit(1)

    print(f"Transcribing {len(files)} file(s)...\n", flush=True)

    all_outputs = []

    for filepath in files:
        wav_path, converted = convert_to_wav(filepath)
        dur = (os.path.getsize(wav_path) - 44) / (16000 * 2) if wav_path.suffix == ".wav" else 0
        print(f"  {filepath.name} ({dur:.0f}s)...", end=" ", flush=True)

        if args.local:
            data = transcribe_local(str(wav_path), args.language)
        else:
            data = transcribe_remote(str(wav_path), args.language, args.speaker)

        output_text = format_output(data, args.format, args.speaker)
        all_outputs.append((filepath.name, output_text))

        seg_count = len([s for s in data["segments"] if s.get("text", "").strip()])
        print(f"{seg_count} segments", flush=True)

        if converted:
            wav_path.unlink()

    # Write output
    if args.output:
        out_path = Path(args.output)
    elif len(files) == 1:
        out_path = files[0].with_suffix(f".transcript.{args.format}")
    else:
        out_path = Path(args.path) / f"transcripts.{args.format}"

    with open(out_path, "w") as f:
        for name, text in all_outputs:
            if len(all_outputs) > 1:
                f.write(f"\n{'='*60}\n")
                f.write(f"# {name}\n")
                f.write(f"{'='*60}\n\n")
            f.write(text)
            f.write("\n")

    print(f"\nOutput: {out_path}", flush=True)


if __name__ == "__main__":
    main()
