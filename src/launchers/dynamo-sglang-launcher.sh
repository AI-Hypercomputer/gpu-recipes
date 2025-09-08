#!/bin/bash

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -eux # Exit immediately if a command exits with a non-zero status.

echo "Dynamo SGLang launcher starting"
echo "Arguments received: $@"

# MODEL_NAME should be passed as an environment variable from deployment
if [ -z "$MODEL_NAME" ]; then
  echo "Error: MODEL_NAME environment variable is not set."
  exit 1
fi
echo "Using MODEL_NAME: $MODEL_NAME"

# Launch the Dynamo SGLang server
echo "Launching Dynamo SGLang server with model: $MODEL_NAME"
python3 -m dynamo.sglang \
  --model "$MODEL_NAME" \
  "$@"

echo "Dynamo SGLang server command finished."