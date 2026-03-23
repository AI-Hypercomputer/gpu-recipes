usage()
{
cat << EOF
usage: bash ./launcher.sh [config-override  [config-override ...]]
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

if [ -z "${config_overrides}" ]; then
  echo "No NeMo config overrides specified"
else
  echo "NeMo config overrides:"
  echo "  ${config_overrides}"
fi

if [[ -n "${NCCL_PLUGIN_PATH}" ]]; then
  export LD_LIBRARY_PATH="$NCCL_PLUGIN_PATH"
  ldconfig $LD_LIBRARY_PATH
  echo "Added $LD_LIBRARY_PATH to ldconfig:"
  ldconfig -p | grep libcuda | sed 's/^/  /'
  echo ""
fi

if [[ -n "${EXPLICIT_LOG_DIR}" ]]; then
  explicit_log_dir=${EXPLICIT_LOG_DIR}
else
  explicit_log_dir=workload_logs
fi
echo "Logging to ${explicit_log_dir}"

if [[ -n "${TOKENIZER_PATH}" ]]; then
  echo "Getting tokenizer files"
  cp ${TOKENIZER_PATH}/* .
  echo ""
fi

echo "Launching Torch distributed on the node rank $JOB_COMPLETION_INDEX out of $NNODES nodes"


pip install git+https://github.com/NVIDIA/dllogger#egg=dllogger


# Export the nemo2 config to yaml.
python ${NEMO_LAUNCH_SCRIPT} --factory "recipe()" \
trainer.num_nodes="$NNODES" \
log.explicit_log_dir="${explicit_log_dir}" \
trainer.max_steps=30 \
trainer.num_nodes=16 \
trainer.devices=8 \
${config_overrides} \
--to-yaml exported_nemo_config.yaml

# Create the nsys directory.
mkdir -p ${explicit_log_dir}/nsys

OMP_NUM_THREADS=12 NSYS_CONFIG_DIRECTIVES="AgentLaunchTimeoutSec=240;AppLaunchTimeoutSec=240" TORCH_NCCL_ENABLE_MONITORING=0 \
/usr/local/bin/nsys profile -s none -t nvtx,cuda --capture-range=cudaProfilerApi --capture-range-end=stop \
-o ${explicit_log_dir}/nsys/noderank-${JOB_COMPLETION_INDEX} \
--session-new "nemo-rank${JOB_COMPLETION_INDEX}"-$RANDOM \
--wait all \
torchrun \
--nproc-per-node="8" \
--nnodes="${NNODES}" \
--node_rank="${JOB_COMPLETION_INDEX}" \
--rdzv_id="${JOB_IDENTIFIER}" \
--master_addr="${MASTER_ADDR}" \
--master_port="${MASTER_PORT}" \
${NEMO_LAUNCH_SCRIPT} --factory "recipe()" \
trainer.num_nodes="$NNODES" \
log.explicit_log_dir="${explicit_log_dir}" \
trainer.max_steps=30 \
trainer.num_nodes=16 \
trainer.devices=8 \
${config_overrides}

if [[ "$JOB_COMPLETION_INDEX" == "0" ]]; then
  mkdir -p ${ARTIFACT_DIR}
  cp -r ${explicit_log_dir}/* ${ARTIFACT_DIR}/
  cp ${NEMO_LAUNCH_SCRIPT} ${ARTIFACT_DIR}/run-cli.py
  cp dllogger.json ${ARTIFACT_DIR}/dllogger.json
  cp exported_nemo_config.yaml ${ARTIFACT_DIR}/nemo-configuration.yaml
  env > ${ARTIFACT_DIR}/environ.txt
  ls ${ARTIFACT_DIR}
fi
echo "Training completed"
echo "Pod on $(hostname --fqdn) is exiting"