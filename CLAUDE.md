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

# Run (default: 5090 remote, auto-fallback to local MLX)
python main.py

# Run with LLM polish
python main.py --llm

# Run local only (MLX Whisper on Apple Silicon GPU)
python main.py --local
```

## Architecture

```
Default: Mac mic → 5090 whisper large-v3-turbo → LLM polish → Cmd+V insert
Local:   Mac mic → MLX Whisper large-v3-turbo (Apple Silicon GPU) → Cmd+V insert
```

### STT Providers (pluggable via config.stt_provider)
- `whisper_remote` — faster-whisper large-v3-turbo on 5090 GPU via SSH tunnel
- `whisper_local` — MLX Whisper large-v3-turbo on Mac Apple Silicon GPU

### LLM Provider
- `vllm_remote` — Qwen3-8B via vLLM on 5090 (optional, `--llm` flag)

## Key Design Decisions

- **All local, no cloud dependency.** 5090 LAN is optional acceleration.
- **MLX Whisper for local:** Uses Apple Silicon GPU (Metal), 91% accuracy vs 78% for CPU-only faster-whisper.
- **Chinese-English mixed recognition:** `language=None` auto-detects zh/en. opencc converts traditional→simplified on server.
- **Prefix stability commit:** Compare consecutive transcription runs, commit matching prefix. Speaker change also triggers commit.
- **Speaker identification:** speechbrain ECAPA-TDNN on 5090. User enrolls voice via `enroll_voice.py`.
- **Screen context:** OCR screenshot on recording start, inject keywords into whisper prompt.
- **Custom dictionary:** `dictionary.txt` for domain terms and correction mappings.
- **Audio saved:** every recording session saved as WAV in `recordings/`.

## Project Structure

```
main.py              — entry point, CLI args, wires modules
transcribe.py        — batch transcription CLI
enroll_voice.py      — speaker voice enrollment
config.py            — all configuration
dictionary.txt       — custom terms and correction mappings
server_manager.py    — 5090 server auto-management
audio.py             — microphone capture via sounddevice
audio_utils.py       — WAV conversion utilities
hotkey.py            — global hotkey (Ctrl+Shift+R/E)
screen_context.py    — screen OCR keyword extraction
custom_dict.py       — dictionary loader
stt/
  base.py            — STT provider interface
  whisper_local.py   — MLX Whisper (Mac Apple Silicon GPU)
  whisper_remote.py  — 5090 GPU faster-whisper via HTTP
llm/
  base.py            — LLM provider interface
  vllm_remote.py     — Qwen3-8B via vLLM
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

When modifying STT or insertion logic, always run ground truth comparison:

```bash
python test_compare.py
```

Key quality metrics:
- Character overlap should be >90% with ground truth (MLX local: 91%, 5090: 100%)
- No duplicate commits (same text committed twice)
- All terminal output should match editor output (no data loss in insertion)

## Hardware

- **Primary**: Mac (Apple Silicon) — runs the app, audio capture, MLX Whisper
- **Optional**: RTX 5090 on LAN (WSL, 10.0.0.145) — faster-whisper + speaker recognition + vLLM
  - SSH tunnel: `ssh -f -N -L 8787:localhost:8787 wsl`
