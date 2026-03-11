#!/bin/bash

# Copyright 2025 Google LLC
# Licensed under the Apache License, Version 2.0

set -eux # Exit immediately if a command exits with a non-zero status.

echo "SGLang server arguments received: $@"

export HF_HOME=/ssd

# MODEL_NAME and CONFIG_FILE should be passed from the Helm deployment
if [ -z "$MODEL_NAME" ]; then
  echo "Error: MODEL_NAME environment variable is not set."
  exit 1
fi

# Default to the path where Helm mounts the serving-args.yaml
CONFIG_FILE=${CONFIG_FILE:-"/workload/configs/serving-args.yaml"}

echo "--------------------------------------------------"
echo "Launching SGLang Server for Blackwell (A4X)"
echo "Model Path:  $MODEL_NAME"
echo "Config File: $CONFIG_FILE"
echo "--------------------------------------------------"

# Launch the server using the corrected --model-path argument
# and linking the external configuration file.
python3 -m sglang.launch_server \
  --model-path "$MODEL_NAME" \
  --config-file "$CONFIG_FILE" \
  "$@"

echo "Server bringup is complete. SGLang server command finished."
