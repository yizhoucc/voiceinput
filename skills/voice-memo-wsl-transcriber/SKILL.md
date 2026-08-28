---
name: voice-memo-wsl-transcriber
description: Transcribe Apple Voice Memos stored on the user's remote MacBook with the WSL RTX 5090, using Qwen3-ASR for accurate multilingual text, forced alignment, optional speaker labels, and resumable outputs. Use for one-file tests or complete Voice Memos archive runs; do not use for live dictation.
---

# Voice Memo WSL Transcriber

Use the remote MacBook as the source and final storage location, and WSL only as temporary GPU compute.

## Fixed topology

- Remote MacBook: `ssh yc@10.0.0.98`
- WSL is reachable from that MacBook as `ssh wsl`.
- Voice Memos media: `~/Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings`
- Voice Memos database: `.../Recordings/CloudRecordings.db`
- Canonical high-quality worker: `~/repo/voiceinput/scripts/voice_memo_qwen3_batch.py`
- Legacy worker: `~/repo/voiceinput/scripts/voice_memo_batch.py`
- WSL Qwen3 overlay environment: `~/.venvs/voice-memo-qwen3-asr`
- Qwen3 models: `Qwen/Qwen3-ASR-1.7B` and `Qwen/Qwen3-ForcedAligner-0.6B`

Never recursively search `~/Library`; access only the explicit Voice Memos container. Never inspect Chrome or Google browser data.

## Workflow

1. Query `CloudRecordings.db` for `ZPATH`, title, and duration. Treat both `.m4a` and `.qta` paths as recordings.
2. Before a new archive run, select one short recording and validate the complete pipeline. Keep the raw and polished versions for comparison.
3. Stage source media to a dated WSL job directory with `rsync`. Do not modify the Voice Memos container.
4. Check `/usr/lib/wsl/lib/nvidia-smi`. If another large-model process is using the GPU, wait and check once per minute.
5. Run long-lived WSL commands in named `tmux` sessions so they continue if the initiating Mac disconnects. Use the environment's absolute Python path inside `tmux`; its non-login shell may not resolve `python` from Conda.
6. Run `voice_memo_qwen3_batch.py`. It converts input to 16kHz mono WAV, uses VAD to form windows shorter than the aligner's five-minute limit, transcribes with Qwen3-ASR, and obtains word/character timestamps from Qwen3-ForcedAligner.
7. Produce three resumable outputs per recording: `clean/` for the primary readable transcript without speaker labels, `speakers/` for an experimental diarized view, and `metadata/` for timestamps, chunks, and diarization data.
8. Treat the clean transcript as authoritative. Speaker labels are anonymous clustering results and must not influence clean sentence boundaries; do not present them as real identities without an enrolled voice sample.
9. Copy outputs and logs back to the remote MacBook. Verify counts, nonempty outputs, valid JSON, and monotonic timestamps before deleting the WSL staging directory.

If a database row has `ZLOCALDURATION=0`, or the media file is tiny and fails decoding with a missing `moov` atom, treat it as an iCloud placeholder rather than an ASR failure. Record it in the final report and ask the user to open or play it in Voice Memos so macOS downloads the audio; a later resumable run should process only those missing files.

## Output requirements

- Preserve original source filenames in output names.
- Include the Voice Memos title, source filename, duration, model, and timestamps.
- Keep the clean and speaker-labeled views separate. Do not contaminate the primary text with uncertain diarization.
- Keep JSON token, sentence, chunk-language, and diarization metadata alongside text.
- Do not use text-only LLM rewriting as the primary correction mechanism: testing showed it can replace ASR errors with plausible but false words. Prefer improving the acoustic model.
- Use a dated folder under `~/Documents/Voice Memo Transcripts/` on the remote MacBook.

## Operations

This is a long remote workflow. Send one Bark notification at start and one at completion. During active WSL work, inspect the log and GPU once per minute. Notify immediately on an error; otherwise only send an intermediate notification when the ETA changes by more than two minutes. Every ETA notification title must include the concrete remaining time, for example `Voice Memo Polish ETA 35 minutes`; never send a generic `ETA updated` title. Estimate from a sufficiently long throughput window because concurrent workers can finish several files at once and make short-window ETAs unstable.
