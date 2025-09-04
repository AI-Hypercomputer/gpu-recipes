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

export PYTHONUNBUFFERED=1

export LD_LIBRARY_PATH="$NCCL_PLUGIN_PATH"
ldconfig "$LD_LIBRARY_PATH"
echo "Added $LD_LIBRARY_PATH to ldconfig:"
ldconfig -p | grep libcuda | sed 's/^/  /'
echo ""

current_directory="$PWD"

export NODE_RANK=$JOB_COMPLETION_INDEX
export HYDRA_FULL_ERROR=1
export PYTHON_MAIN="${current_directory}/paligemma.py"
export TRAIN_SCRIPT="${current_directory}/train.sh"

echo "Launching Torch distributed as node rank $NODE_RANK out of $NNODES nodes"

cat > $TRAIN_SCRIPT << 'EOF'
#!/bin/bash

nsys profile -s none -t nvtx,cuda --gpu-metrics-devices=all --enable nvml_metrics \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o /gcs/nemo-experiments/$JOB_IDENTIFIER/rank-$NODE_RANK-0 \
  --force-overwrite true --session-new "nsys-$JOB_IDENTIFIER-$NODE_RANK-0" \
  python "$PYTHON_MAIN"
EOF

export TOKENIZERS_PARALLELISM=false
export NVTE_UB_SOCKET_IFNAME="eth1"
export NUM_TRAIN_EPOCHS=1
export PER_DEVICE_TRAIN_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=2
chmod +x $TRAIN_SCRIPT

accelerate launch --no_python $TRAIN_SCRIPT