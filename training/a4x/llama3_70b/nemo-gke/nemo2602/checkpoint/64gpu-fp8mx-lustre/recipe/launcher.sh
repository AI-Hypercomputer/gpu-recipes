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
  explicit_log_dir=/workspace/workload_logs
fi
echo "Logging to ${explicit_log_dir}"

if [[ -n "${TOKENIZER_PATH}" ]]; then
  echo "Getting tokenizer files"
  cp ${TOKENIZER_PATH}/* .
  echo ""
fi

if [[ -n "${DATASET_PATHS}" ]]; then
  dataload_args="--data ${DATASET_TYPE} \
  --dataset_paths ${DATASET_PATHS} \
  --index_mapping_dir ${INDEX_MAPPING_DIR}"
fi

if [[ -n "${CKPT_SAVE_DIR}" ]]; then
  ckpt_save_args="--save_dir "${CKPT_SAVE_DIR}/${JOB_IDENTIFIER}" \
  --save_interval ${CKPT_SAVE_INTERVAL} \
  --most_recent_k 10 \
  checkpoint.async_save=True"
fi

if [[ -n "${CKPT_LOAD_DIR}" ]]; then
  ckpt_load_args="--load_dir "${CKPT_LOAD_DIR}" \
    checkpoint.ckpt_step=${CKPT_LOAD_STEP}"
fi

echo "Launching Torch distributed on the node rank $JOB_COMPLETION_INDEX out of $NNODES nodes"

pip install git+https://github.com/NVIDIA/dllogger#egg=dllogger

# Create the nsys directory.
mkdir -p ${explicit_log_dir}/nsys

cd /opt
rm -rf Megatron-Bridge
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
cd Megatron-Bridge
git checkout 9c9dd848966322fc3ed7706747ec13219ef49dda

git clone -b core_r0.16.0 https://github.com/NVIDIA/Megatron-LM.git /opt/Megatron-Bridge/3rdparty/Megatron-LM
export HF_TOKEN=${HF_TOKEN}

cp $CUSTOM_SETUP_EXPERIMENT_SCRIPT_PATH scripts/performance/
cp $CUSTOM_RUN_SCRIPT_PATH scripts/performance/run_script.py

worker_command=$(cat <<- EOM
  if [ "\$RANK" -eq "0" ]; then
    echo "Worker 0 is stalling for a few seconds.." ;
    sleep 10 ;
    echo "The detected environment within worker rank 0 is:" ;
    env | sed 's/^/  /' ;
  fi ;

  cd /opt/Megatron-Bridge ;
  #sleep inf

  numactl \
    --cpunodebind=\$((LOCAL_RANK/2)) \
    --membind=\$((LOCAL_RANK/2))           nsys profile \
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
    --gpu gb200 \
    --account asq_google_com --partition a4xmaxpartition \
    --model_family_name llama \
    --model_recipe_name llama3_70b \
    --list_config_variants \
    --gpus_per_node 4 \
    --num_gpus ${WORLD_SIZE} \
    --use_megatron_fsdp 0 \
    --compute_dtype fp8_mx \
    --tensor_model_parallel_size 2 \
    --pipeline_model_parallel_size 4 \
    --virtual_pipeline_model_parallel_size 5 \
    --global_batch_size 128 \
    ${dataload_args} \
    ${ckpt_load_args} \
    ${ckpt_save_args} \
    --max_steps 30 \
    -hf $HF_TOKEN

EOM
)

echo "$worker_command" > worker_command.sh
chmod 777 worker_command.sh

torchrun \
--nproc-per-node="4" \
--nnodes="${NNODES}" \
--node_rank="${JOB_COMPLETION_INDEX}" \
--rdzv_id="${JOB_IDENTIFIER}" \
--master_addr="${MASTER_ADDR}" \
--master_port="${MASTER_PORT}" \
--no-python bash worker_command.sh | tee -a ${explicit_log_dir}/workload_run.log


if [[ "$JOB_COMPLETION_INDEX" == "0" ]]; then
  mkdir -p ${ARTIFACT_DIR}
  mkdir -p ${ARTIFACT_DIR}/${JOB_IDENTIFIER}
  cp -r ${explicit_log_dir}/* ${ARTIFACT_DIR}/${JOB_IDENTIFIER}
  env > ${ARTIFACT_DIR}/environ.txt
  ls ${ARTIFACT_DIR}
fi
echo "Training completed"
echo "Pod on $(hostname --fqdn) is exiting"
