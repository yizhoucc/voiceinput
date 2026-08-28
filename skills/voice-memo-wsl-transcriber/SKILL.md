---
name: voice-memo-wsl-transcriber
description: Transcribe Apple Voice Memos stored on the user's remote MacBook with the WSL RTX 5090, including timestamps, speaker labels, resumable raw outputs, and optional local Qwen polishing. Use for one-file tests or complete Voice Memos archive runs; do not use for live dictation.
---

# Voice Memo WSL Transcriber

Use the remote MacBook as the source and final storage location, and WSL only as temporary GPU compute.

## Fixed topology

- Remote MacBook: `ssh yc@10.0.0.98`
- WSL is reachable from that MacBook as `ssh wsl`.
- Voice Memos media: `~/Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings`
- Voice Memos database: `.../Recordings/CloudRecordings.db`
- Canonical worker: `~/repo/voiceinput/scripts/voice_memo_batch.py`
- WSL Python environment: `~/miniconda3/envs/medllm`
- faster-whisper model: `~/models/faster-whisper-large-v3`
- Qwen polish model: `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct-AWQ`

Never recursively search `~/Library`; access only the explicit Voice Memos container. Never inspect Chrome or Google browser data.

## Workflow

1. Query `CloudRecordings.db` for `ZPATH`, title, and duration. Treat both `.m4a` and `.qta` paths as recordings.
2. Before a new archive run, select one short recording and validate the complete pipeline. Keep the raw and polished versions for comparison.
3. Stage source media to a dated WSL job directory with `rsync`. Do not modify the Voice Memos container.
4. Check `/usr/lib/wsl/lib/nvidia-smi`. If another large-model process is using the GPU, wait and check once per minute.
5. Run long-lived WSL commands in named `tmux` sessions so they continue if the initiating Mac disconnects. Use the environment's absolute Python path inside `tmux`; its non-login shell may not resolve `python` from Conda.
6. Run the worker's `transcribe` command first. It loads faster-whisper large-v3 and pyannote once, writes one `.raw.txt` and one `.json` per source, and safely skips completed files on retry.
7. Copy raw outputs back to the remote MacBook before polishing.
8. Release the ASR process. Start the cached Qwen2.5-32B-AWQ model with `scripts/start_voice_memo_vllm.sh` in its own `tmux` session, then run the worker's `polish` command in a second session. Keep raw outputs permanently; polished files are an additional view, never a replacement.
9. Copy polished files and logs back to the remote MacBook. Verify counts, nonempty outputs, valid JSON, and exact preservation of timestamp/speaker prefixes before deleting the WSL staging directory.

If a database row has `ZLOCALDURATION=0`, or the media file is tiny and fails decoding with a missing `moov` atom, treat it as an iCloud placeholder rather than an ASR failure. Record it in the final report and ask the user to open or play it in Voice Memos so macOS downloads the audio; a later resumable run should process only those missing files.

## Output requirements

- Preserve original source filenames in output names.
- Include the Voice Memos title, source filename, duration, detected language, timestamps, and speaker labels.
- Keep JSON segment metadata alongside text.
- If polish changes timestamp/speaker structure or changes length excessively, reject that chunk and retain the raw text.
- Use a dated folder under `~/Documents/Voice Memo Transcripts/` on the remote MacBook.

## Operations

This is a long remote workflow. Send one Bark notification at start and one at completion. During active WSL work, inspect the log and GPU once per minute. Notify immediately on an error; otherwise only send an intermediate notification when the ETA changes by more than two minutes. Every ETA notification title must include the concrete remaining time, for example `Voice Memo Polish ETA 35 minutes`; never send a generic `ETA updated` title. Estimate from a sufficiently long throughput window because concurrent workers can finish several files at once and make short-window ETAs unstable.
