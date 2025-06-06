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
#
# This script is used to run NeMo 1.x workloads.
#


usage()
{
cat << EOF
usage: bash ./grl-nemo-20-launcher.sh [--config-override  [--config-override ...]]
config-override  (Optional) A  NeMo configuration override. E.g.--max-steps=100.
EOF
}

parse_args() {
  while [ "$1" != "" ]; do
    local arg="$1"
    if [[ "$arg" == --* ]]; then
      config_overrides+=("$arg")
    else
      echo "Invalid argument: $arg. Arguments must start with --."
      usage
      exit 1
    fi

    shift
  done
  config_overrides="${config_overrides[*]}"
}

echo "JOB settings:"
echo "  JOB_IDENTIFIER: $JOB_IDENTIFIER"
echo "  JOB_TIMESTAMP: $JOB_TIMESTAMP"
echo "  JOB_UUID: $JOB_UUID"
echo "  JOB_ORCHESTRATOR: $JOB_ORCHESTRATOR"
echo "  JOB_COMPLETION_INDEX: $JOB_COMPLETION_INDEX"
echo "  REPLICATED_JOB_NAME: $REPLICATED_JOB_NAME"
echo "  JOBSET_NAME: $JOBSET_NAME"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"

config_overrides=()
parse_args "$@"

echo "Config overrides:"
echo "${config_overrides[@]}"

echo "NeMo Training file:"
cat "${NEMO_CONFIG_PATH}/${NEMO_CONFIG_NAME}" | sed 's/^/| /'
echo ""

if [ -z "${config_overrides}" ]; then
  echo "No NeMo config overrides specified"
else
  echo "NeMo config overrides:"
  echo "  ${config_overrides}"
fi

export LD_LIBRARY_PATH="$NCCL_PLUGIN_PATH"
ldconfig $LD_LIBRARY_PATH
echo "Added $LD_LIBRARY_PATH to ldconfig:"
ldconfig -p | grep libcuda | sed 's/^/  /'
echo ""
export NCCL_DEBUG=VERSION
echo "Launching Torch distributed on the node rank $JOB_COMPLETION_INDEX out of $NNODES nodes"

CMD="exec python3 resiliency/launcher.py --use-supervisor \
--ft-param-initial_rank_heartbeat_timeout="${FT_PARAM_INITIAL_RANK_HEARTBEAT_TIMEOUT}" \
--ft-param-rank_heartbeat_timeout="${FT_PARAM_RANK_HEARTBEAT_TIMEOUT}" \
--nproc-per-node="${GPUS_PER_NODE}" \
--nnodes="${NNODES}" \
--node_rank="${JOB_COMPLETION_INDEX}" \
--rdzv_id="${JOB_IDENTIFIER}" \
--master_addr="${MASTER_ADDR}" \
--master_port="${MASTER_PORT}" \
--max-restarts="${MAX_IN_JOB_RESTARTS}" \
${NEMO_CONFIG_PATH}/${NEMO_CONFIG_NAME} \
${config_overrides}"

echo "Running command:"
echo "$CMD"
eval "$CMD"

echo "Training completed"
echo "Pod on $(hostname --fqdn) is exiting"