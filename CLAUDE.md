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

# Run (local whisper)
python main.py
# Press Enter to start/stop recording

# Run with 5090 GPU (much faster + better quality)
# 1. Start whisper server on 5090 (see whisper_server.py on WSL at /tmp/whisper_server.py)
# 2. SSH tunnel: ssh -f -N -L 8787:localhost:8787 wsl
# 3. Set config.stt_provider = "whisper_remote" in config.py
# 4. python main.py
```

## Architecture

```
Enter key → Audio Capture → STT (sliding window) → partial text → display immediately
                                                  → final text  → LLM stream polish → replace
```

### STT Providers (pluggable via config.stt_provider)
- `whisper_local` — faster-whisper `small` on Mac CPU, sliding window
- `whisper_remote` — faster-whisper `large-v3-turbo` on 5090 GPU via SSH tunnel (http://localhost:8787)
- `apple_speech` — (planned) macOS SFSpeechRecognizer

### LLM Providers (pluggable via config.llm_provider)
- `ollama` — local Ollama, streaming API
- `vllm_remote` — (planned) vLLM on 5090 via LAN HTTP

All providers implement abstract interfaces in `stt/base.py` and `llm/base.py`.

## Key Design Decisions

- **All local, no cloud dependency.** 5090 LAN is optional acceleration, not required.
- **Chinese-English mixed recognition:** `language=None` auto-detects, restricted to zh/en only (other languages fall back to zh). opencc converts traditional→simplified Chinese.
- **Incremental commit:** segments in the stable zone (>6s old in window) auto-commit. Pause >1s also triggers commit. Committed audio is trimmed from buffer → O(1) memory.
- **Speaker identification:** speechbrain ECAPA-TDNN on 5090. User enrolls voice via `enroll_voice.py`, then each segment tagged `[我]`/`[他]`.
- **Audio saved:** every recording session saved as WAV in `recordings/` for future audio-LLM correction.

## Project Structure

```
main.py              — entry point, wires modules together
config.py            — all configuration (STT/LLM provider, model, hotkey, etc.)
audio.py             — microphone capture via sounddevice, saves raw audio
hotkey.py            — Enter key toggle (placeholder for global hotkey)
enroll_voice.py      — record 15s of your voice to register speaker embedding
test_streaming.py    — simulate streaming with audio file for testing
stt/
  base.py            — STT provider interface (on_partial, on_final callbacks)
  whisper_local.py   — faster-whisper sliding window on Mac CPU
  whisper_remote.py  — faster-whisper on 5090 GPU via HTTP
llm/
  base.py            — LLM provider interface
output/
  terminal.py        — terminal display + transcript logging
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

This compares full-file transcription (ground truth) vs streaming transcription on saved recordings.
- Use recordings from `recordings/` directory
- Ground truth: send full WAV to 5090 server in one request
- Streaming: feed audio 1s at a time, collect commits
- Compare word overlap, missing words, extra words
- Log detailed commit timeline and text diffs

Key quality metrics:
- Word overlap should be >70% with ground truth
- No duplicate commits (same text committed twice)
- Commit interval: 3-8 seconds average
- All terminal output should match editor output (no data loss in insertion)

Known trade-off: streaming quality < full-file quality due to limited whisper context window.
Transformer/QKV recognition sometimes fails in short chunks but succeeds in longer context.

## Hardware

- **Primary**: Mac (Apple Silicon) — runs the app, audio capture, output
- **Optional**: RTX 5090 on LAN (WSL, 10.0.0.145) — faster-whisper + speaker recognition
  - SSH tunnel: `ssh -f -N -L 8787:localhost:8787 wsl`
  - SSH tunnel: `ssh -f -N -L 8787:localhost:8787 wsl`
