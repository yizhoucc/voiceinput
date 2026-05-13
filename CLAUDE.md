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

# Run
python main.py
# Press Option+Space to start, release Option to stop
```

Requires Ollama running locally for LLM polish:
```bash
brew install ollama
ollama pull qwen2.5:7b
```

## Architecture

```
Hotkey → Audio Capture → STT (sliding window) → partial text → display immediately
                                                → final text  → LLM stream polish → replace
```

### STT Providers (pluggable via config.stt_provider)
- `whisper_local` — faster-whisper on Mac CPU, sliding window (~1-3s chunks)
- `apple_speech` — macOS SFSpeechRecognizer, true word-by-word streaming
- `whisper_remote` — faster-whisper on 5090 GPU via LAN HTTP

### LLM Providers (pluggable via config.llm_provider)
- `ollama` — local Ollama, streaming API
- `vllm_remote` — vLLM on 5090 via LAN HTTP

All providers implement abstract interfaces in `stt/base.py` and `llm/base.py`.

## Key Design Decisions

- **All local, no cloud dependency.** 5090 LAN is optional acceleration, not required.
- **Chinese-English mixed recognition:** set `config.primary_language = "zh"` for Chinese-primary with English mixed in. `None` for auto-detect.
- **Async polish:** LLM runs in parallel with continued STT, never blocks new speech input.
- **Segment state machine** in `output/manager.py`: partial → raw → polishing → done.

## Project Structure

```
main.py          — entry point, wires modules together
config.py        — all configuration (STT/LLM provider, model, hotkey, etc.)
audio.py         — microphone capture via sounddevice
hotkey.py        — global hotkey listener (Option+Space) via pynput
stt/base.py      — STT provider interface (on_partial, on_final callbacks)
stt/whisper_local.py — faster-whisper sliding window implementation
llm/base.py      — LLM provider interface
llm/ollama.py    — Ollama streaming client
output/terminal.py    — terminal display (partial overwrites line, final appends)
output/manager.py     — segment state machine for async polish & replace
output/system_insert.py — macOS system-level text insertion
```

## Hardware

- **Primary**: Mac (Apple Silicon)
- **Optional**: RTX 5090 on LAN (WSL, 10.0.0.145) for faster-whisper + vLLM
