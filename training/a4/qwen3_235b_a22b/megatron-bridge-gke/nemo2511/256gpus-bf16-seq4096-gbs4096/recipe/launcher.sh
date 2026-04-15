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

export LD_LIBRARY_PATH="$NCCL_PLUGIN_PATH"
ldconfig $LD_LIBRARY_PATH
echo "Added $LD_LIBRARY_PATH to ldconfig:"
ldconfig -p | grep libcuda | sed 's/^/  /'
echo ""

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

# Create the nsys directory.
mkdir -p ${explicit_log_dir}/nsys

cd /opt
rm -rf Megatron-Bridge
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
cd Megatron-Bridge
git checkout 7695d4acbfac19353d20e456509117efe4733d6b
ls

cp $CUSTOM_SETUP_EXPERIMENT_SCRIPT_PATH scripts/performance/

worker_command=$(cat <<- EOM
  if [ "\$RANK" -eq "0" ]; then
    echo "Worker 0 is stalling for a few seconds.." ;
    sleep 3 ;
    echo "The detected environment within worker rank 0 is:" ;
    env | sed 's/^/  /' ;
  fi ;

  cd /opt/Megatron-Bridge ;

  numactl \
    --cpunodebind=\$((LOCAL_RANK/4)) \
    --membind=\$((LOCAL_RANK/4))           nsys profile \
    -t nvtx,cuda \
    --cuda-event-trace=false \
    --sample=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --kill none \
    -o /${explicit_log_dir}/$JOB_IDENTIFIER/rank-\$RANK \
    --force-overwrite true \
    --session-new "nsys-\$RANDOM-\$RANK" \
  nice -10 \
  python scripts/performance/custom_setup_experiment.py \
    --gpu b200 \
    --model_family_name qwen \
    --model_recipe_name qwen3_235b_a22b \
    --gpus_per_node 8 \
    --num_gpus 256 \
    --seq_length 4096 \
    --compute_dtype bf16 \
    --global_batch_size 4096 \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 8 \
    --virtual_pipeline_model_parallel_size 4 \
    --expert_model_parallel_size 8 \
    --expert_tensor_parallel_size 1 \
    --moe_a2a_overlap True \
    --max_steps 30

EOM
)

echo "$worker_command" > worker_command.sh
chmod 777 worker_command.sh

torchrun \
--nproc-per-node="8" \
--nnodes="32" \
--node_rank="${JOB_COMPLETION_INDEX}" \
--rdzv_id="${JOB_IDENTIFIER}" \
--master_addr="${MASTER_ADDR}" \
--master_port="${MASTER_PORT}" \
--no-python bash worker_command.sh


if [[ "$JOB_COMPLETION_INDEX" == "0" ]]; then
  mkdir -p ${ARTIFACT_DIR}
  cp -r ${explicit_log_dir}/* ${ARTIFACT_DIR}/
  env > ${ARTIFACT_DIR}/environ.txt
  ls ${ARTIFACT_DIR}
fi
echo "Training completed"
echo "Pod on $(hostname --fqdn) is exiting"