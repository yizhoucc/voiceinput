#!/usr/bin/env bash
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate medllm

model="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct-AWQ/snapshots/5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c"

exec vllm serve "$model" \
  --served-model-name qwen-polish \
  --port 8000 \
  --quantization awq_marlin \
  --dtype half \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.88 \
  --enable-prefix-caching
