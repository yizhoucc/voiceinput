# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Build a Typeless-like streaming voice input tool for macOS. Core experience: press hotkey, speak naturally, see polished text appear in real-time and get inserted at cursor position in any app.

Architecture: "STT instant display + async LLM streaming polish & replace"
- STT partial results → show immediately (user sees text growing)
- STT final results → send to LLM for async streaming polish → replace raw text

## Build & Run

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
brew install portaudio  # required by sounddevice
pip install -r requirements.txt

# Run (default: 5090 GPU remote, auto-fallback to Mac local)
python main.py

# Run local only (MLX Whisper on Apple Silicon GPU)
python main.py --local

# Run with LLM polish
python main.py --llm

# Batch transcribe
python transcribe.py recording.wav
```

## Architecture

```
Hotkey → Audio Capture → STT (sliding window) → partial text → display immediately
                                               → commit       → LLM polish → Cmd+V insert
```

### STT Providers (pluggable via config.stt_provider)
- `whisper_local` — MLX Whisper `large-v3-turbo` on Mac Apple Silicon GPU (91% accuracy)
- `whisper_remote` — faster-whisper `large-v3-turbo` on 5090 GPU via SSH tunnel (100% accuracy)

### LLM Provider
- `vllm_remote` — Qwen3-8B via vLLM on 5090 GPU, with custom correction dictionary

## Key Design Decisions

- **All local, no cloud dependency.** 5090 LAN is optional acceleration, not required.
- **Chinese-English mixed recognition:** `language=None` auto-detects, restricted to zh/en only (other languages fall back to zh). opencc converts traditional→simplified Chinese.
- **Append-only text insertion:** editor text is only appended via Cmd+V, never modified. All approaches to modify in-place (Cmd+Z, Shift+Left, diff) proved unreliable due to async key event timing.
- **Speaker identification:** speechbrain ECAPA-TDNN on 5090. User enrolls voice via `enroll_voice.py`, then each segment tagged `[我]`/`[他]`.
- **Screen context:** On recording start, screenshot → OCR → keywords injected into whisper prompt + LLM context for better domain term recognition.
- **Custom dictionary:** `dictionary.txt` provides terms for whisper prompt and correction mappings for LLM polish.
- **Audio saved:** every recording session saved as WAV in `recordings/` for quality validation.

## Project Structure

```
main.py              — entry point, CLI args, wires modules together
transcribe.py        — batch transcription CLI (txt/srt/tsv output)
enroll_voice.py      — record 15s voice for speaker enrollment
config.py            — all configuration
server_manager.py    — auto-manage 5090 whisper/vLLM servers
audio.py             — microphone capture via sounddevice
audio_utils.py       — WAV conversion utilities
hotkey.py            — global hotkey (Ctrl+Shift+R/E)
screen_context.py    — OCR screenshot for context keywords
custom_dict.py       — load dictionary.txt
dictionary.txt       — custom terms + correction mappings
stt/
  base.py            — STT provider interface
  whisper_local.py   — MLX Whisper on Mac GPU
  whisper_remote.py  — faster-whisper on 5090 GPU via HTTP
llm/
  base.py            — LLM provider interface
  vllm_remote.py     — Qwen3-8B polish via vLLM
output/
  terminal.py        — terminal display + transcript logging
  system_insert.py   — macOS text insertion (Cmd+V)
```

## 5090 Whisper Server

Located at `/tmp/whisper_server.py` on WSL. Endpoints:
- `POST /transcribe` — audio file → segments with text, timestamps, speaker tags
- `POST /enroll` — audio file → register speaker embedding

Features: large-v3-turbo, speaker identification (speechbrain), opencc t2s, language restricted to zh/en.

## Testing & Quality Validation

```bash
python test_compare.py       # ground truth vs streaming comparison
python test_benchmark.py     # 4-config benchmark (5090/local × quantize)
python test_mlx.py           # MLX Whisper vs faster-whisper comparison
python test_all_recordings.py # LLM polish effect on all recordings
```

## Hardware

- **Primary**: Mac (Apple Silicon) — runs the app, audio capture, MLX Whisper (local mode)
- **Optional**: RTX 5090 on LAN (WSL, 10.0.0.145) — faster-whisper + speaker recognition + vLLM
  - SSH tunnel: `ssh -f -N -L 8787:localhost:8787 wsl`
