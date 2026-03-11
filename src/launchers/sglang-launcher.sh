#!/bin/bash
set -eux

export HF_HOME=/ssd

if [ -z "$MODEL_NAME" ]; then
  echo "Error: MODEL_NAME environment variable is not set."
  exit 1
fi

echo "Launching SGLang for Wan2.2 on 4x B200 GPUs"

# We set --tp 4 to match the 4 GPUs
# We keep Ulysses and Ring but ensure SGLang recognizes the total SP degree
sglang serve \
  --model-path "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port 8000 \
  --tp 4 \
  --trust-remote-code \
  --ulysses-degree 2 \
  --ring-degree 2 \
  --dist-backend nccl \
  --text-encoder-cpu-offload
