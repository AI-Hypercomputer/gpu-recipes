#!/bin/bash
# Updated for SGLang Diffusion / Wan2.2

set -eux

export HF_HOME=/ssd

# Check for Model Name
if [ -z "$MODEL_NAME" ]; then
  echo "Error: MODEL_NAME environment variable is not set."
  exit 1
fi

echo "--------------------------------------------------"
echo "Launching SGLang Serve for Wan2.2 (Blackwell)"
echo "--------------------------------------------------"

# Use 'sglang serve' instead of 'python3 -m sglang.launch_server'
# This backend is specifically designed for Wan2.2 arguments.
sglang serve \
  --model-path "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port 8000 \
  --tp 4 \
  --trust-remote-code \
  --ulysses-degree 2 \
  --ring-degree 2 \
  --text-encoder-cpu-offload \
  "$@"
