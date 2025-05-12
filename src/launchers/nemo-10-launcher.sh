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
usage: bash ./nemo-10-launcher.sh [config-override  [config-override ...]]
config-override  (Optional) A  NeMo configuration override. E.g. trainer.max_steps=10000.
EOF
}

parse_args() {
  while [ "$1" != "" ]; do
    case $(grep -o "=" <<< "$1" | wc -l) in
        1  )
        config_overrides+=("$1")
        ;;
        * )
            echo "Invalid config override: $1"
            usage
            exit 1
    esac
    shift
  done
  config_overrides="${config_overrides[*]}"
}


config_overrides=()
parse_args "$@"

echo "NeMo configuration file:"
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

if [[ -n "${EXPLICIT_LOG_DIR}" ]]; then
  explicit_log_dir=${EXPLICIT_LOG_DIR}
else
  explicit_log_dir="${EXPERIMENT_ROOT_DIR}/${EXPERIMENT_NAME}/${JOB_IDENTIFIER}"
fi

dllogger_json_file="${explicit_log_dir}/dllogger/rank-${JOB_COMPLETION_INDEX}/dllogger.json"

echo "Logging to ${explicit_log_dir}"
echo "DLLogger logging to ${dllogger_json_file}"
if [[ -n "${CHECKPOINTS_ROOT_DIR}" ]]; then
  checkpoints_dir="${CHECKPOINTS_ROOT_DIR}/${EXPERIMENT_NAME}/${JOB_IDENTIFIER}"
  echo "Checkpoints will be saved to ${checkpoints_dir}"
  config_overrides="${config_overrides} ++exp_manager.checkpoint_callback_params.dirpath=${checkpoints_dir}"
fi

if [[ -n "${TOKENIZER_PATH}" ]]; then
  echo "Getting tokenizer files"
  cp ${TOKENIZER_PATH}/* .
  echo ""
fi

echo "Launching Torch distributed on the node rank $JOB_COMPLETION_INDEX out of $NNODES nodes"

OMP_NUM_THREADS=12 torchrun \
--nproc-per-node="${GPUS_PER_NODE}" \
--nnodes="${NNODES}" \
--node_rank="${JOB_COMPLETION_INDEX}" \
--rdzv_id="${JOB_IDENTIFIER}" \
--master_addr="${MASTER_ADDR}" \
--master_port="${MASTER_PORT}" \
${NEMO_LAUNCH_SCRIPT} \
--config-path="${NEMO_CONFIG_PATH}" \
--config-name="${NEMO_CONFIG_NAME}" \
++trainer.num_nodes="${NNODES}" \
++exp_manager.name="${EXPERIMENT_NAME}" \
++exp_manager.version="${JOB_IDENTIFIER}" \
++exp_manager.explicit_log_dir="${explicit_log_dir}" \
++exp_manager.create_tensorboard_logger=true \
++exp_manager.create_dllogger_logger=true \
++exp_manager.dllogger_logger_kwargs.verbose=true \
++exp_manager.dllogger_logger_kwargs.stdout=true \
++exp_manager.dllogger_logger_kwargs.json_file="${dllogger_json_file}" \
${config_overrides}

echo "Training completed"
echo "Pod on $(hostname --fqdn) is exiting"