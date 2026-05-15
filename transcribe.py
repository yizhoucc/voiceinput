"""Batch transcription CLI: transcribe audio files to polished text documents.

Usage:
  python transcribe.py recording.wav              # single file → .transcript.txt
  python transcribe.py recordings/                 # entire directory
  python transcribe.py recording.wav --format srt  # SRT subtitles
  python transcribe.py recording.wav --no-polish   # skip LLM polish
  python transcribe.py recording.wav --local       # Mac CPU only
"""
import argparse
import sys
import os
import subprocess
from pathlib import Path
from datetime import timedelta

sys.stdout.reconfigure(line_buffering=True)

PARAGRAPH_PAUSE = 2.0  # seconds of silence to start a new paragraph
SPEAKER_CHANGE_PARAGRAPH = True  # new paragraph on speaker change


def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe audio files to text")
    parser.add_argument("path", help="Audio file or directory of audio files")
    parser.add_argument("-o", "--output", help="Output file path (default: <input>.transcript.txt)")
    parser.add_argument("--local", action="store_true", help="Use local Mac CPU whisper (no 5090)")
    parser.add_argument("--no-polish", action="store_true", help="Skip LLM polish")
    parser.add_argument("--language", type=str, default=None, help="Force language: zh, en, or auto")
    parser.add_argument("--format", choices=["txt", "srt", "tsv"], default="txt",
                        help="Output format: txt (plain), srt (subtitles), tsv (timestamps)")
    return parser.parse_args()


def transcribe_remote(filepath, language=None):
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
            "identify_speaker": "true",
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


def polish_segments(segments):
    """Polish text with LLM, using overlap context."""
    from llm.vllm_remote import VLLMPolisher
    from server_manager import ensure_servers, ensure_ssh_tunnel

    ensure_ssh_tunnel(8000)
    if not ensure_servers(need_llm=True):
        print("[polish] vLLM not available, skipping polish.", flush=True)
        return segments

    polisher = VLLMPolisher()
    polished = []
    prev_text = ""
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            polished.append(seg)
            continue
        try:
            fixed = polisher.polish(text, context_before=prev_text)
            seg = dict(seg)
            seg["text"] = fixed
            prev_text = fixed
        except Exception:
            prev_text = text
        polished.append(seg)
    return polished


def format_time_srt(seconds):
    td = timedelta(seconds=seconds)
    hours = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    secs = int(td.total_seconds() % 60)
    ms = int((td.total_seconds() % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def speaker_label(speaker):
    if speaker == "me":
        return "[我]"
    elif speaker == "other":
        return "[他]"
    return None


def format_txt(segments, lang, lang_prob):
    lines = [f"# Language: {lang} ({lang_prob:.0%})\n"]
    prev_end = 0.0
    prev_speaker = None

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        start = seg.get("start", 0)
        speaker = seg.get("speaker", "unknown")
        gap = start - prev_end

        # New paragraph on long pause or speaker change
        need_paragraph = False
        if gap > PARAGRAPH_PAUSE and prev_end > 0:
            need_paragraph = True
        if SPEAKER_CHANGE_PARAGRAPH and speaker != prev_speaker and prev_speaker is not None:
            need_paragraph = True

        if need_paragraph:
            lines.append("")

        label = speaker_label(speaker)
        if label and speaker != prev_speaker:
            lines.append(f"{label} {text}")
        else:
            lines.append(text)

        prev_end = seg.get("end", start)
        prev_speaker = speaker

    return "\n".join(lines) + "\n"


def format_srt(segments):
    lines = []
    idx = 1
    prev_speaker = None
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        start = format_time_srt(seg["start"])
        end = format_time_srt(seg["end"])
        speaker = seg.get("speaker", "unknown")
        label = speaker_label(speaker)
        if label and speaker != prev_speaker:
            text = f"{label} {text}"
        prev_speaker = speaker
        lines.append(f"{idx}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
        idx += 1
    return "\n".join(lines)


def format_tsv(segments):
    lines = ["start\tend\tspeaker\ttext"]
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        spk = seg.get("speaker", "unknown")
        lines.append(f"{seg['start']:.2f}\t{seg['end']:.2f}\t{spk}\t{text}")
    return "\n".join(lines) + "\n"


def format_output(data, fmt):
    segments = data["segments"]
    lang = data.get("language", "?")
    lang_prob = data.get("language_probability", 0)

    if fmt == "txt":
        return format_txt(segments, lang, lang_prob)
    elif fmt == "srt":
        return format_srt(segments)
    elif fmt == "tsv":
        return format_tsv(segments)


def get_audio_files(path):
    p = Path(path)
    exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".mp4", ".webm"}
    if p.is_file():
        return [p]
    elif p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in exts)
    else:
        print(f"[error] Not found: {path}")
        sys.exit(1)


def convert_to_wav(filepath):
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

    # Load dictionary
    from custom_dict import load_dictionary
    from config import config
    dict_terms, dict_corrections = load_dictionary()
    if dict_terms:
        config.whisper_prompt += ", " + ", ".join(dict_terms[:100])
        config.custom_corrections.update(dict_corrections)
        print(f"[dict] {len(dict_terms)} terms, {len(dict_corrections)} corrections")

    print(f"Transcribing {len(files)} file(s)...\n", flush=True)

    all_outputs = []

    for filepath in files:
        wav_path, converted = convert_to_wav(filepath)
        dur = (os.path.getsize(wav_path) - 44) / (16000 * 2) if wav_path.suffix == ".wav" else 0
        print(f"  {filepath.name} ({dur:.0f}s)...", end=" ", flush=True)

        if args.local:
            data = transcribe_local(str(wav_path), args.language)
        else:
            data = transcribe_remote(str(wav_path), args.language)

        seg_count = len([s for s in data["segments"] if s.get("text", "").strip()])
        print(f"{seg_count} segments", end="", flush=True)

        # LLM polish
        if not args.no_polish and not args.local:
            print(" → polishing...", end="", flush=True)
            data["segments"] = polish_segments(data["segments"])
            print(" done", end="", flush=True)

        print(flush=True)

        output_text = format_output(data, args.format)
        all_outputs.append((filepath.name, output_text))

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
