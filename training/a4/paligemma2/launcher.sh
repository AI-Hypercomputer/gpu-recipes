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

export HF_TOKEN=
export PYTHONUNBUFFERED=1

pip3 install \
  transformers==4.46.3 \
  datasets \
  accelerate \
  peft \
  bitsandbytes \
  pillow \
  tensorboard

export LD_LIBRARY_PATH="$NCCL_PLUGIN_PATH"
ldconfig "$LD_LIBRARY_PATH"
echo "Added $LD_LIBRARY_PATH to ldconfig:"
ldconfig -p | grep libcuda | sed 's/^/  /'
echo ""

export NODE_RANK=$JOB_COMPLETION_INDEX
export HYDRA_FULL_ERROR=1
export NVIDIA_VISIBLE_DEVICES=0

echo "Launching Torch distributed as node rank $NODE_RANK out of $NNODES nodes"

mkdir /app
cat > /app/train.sh << 'EOF'
#!/bin/bash

python "$PYTHON_MAIN"
EOF

export TOKENIZERS_PARALLELISM=false
export NVTE_UB_SOCKET_IFNAME="eth1"

# Training parameters
export NUM_TRAIN_EPOCHS=1
export PER_DEVICE_TRAIN_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=2

chmod +x /app/train.sh

accelerate launch --no_python /app/train.sh
