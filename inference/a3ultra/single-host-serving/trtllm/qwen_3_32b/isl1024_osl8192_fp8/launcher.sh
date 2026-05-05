if [[ -n "${TRTLLM_ENV_FILE}" ]]; then source "${TRTLLM_ENV_FILE}"; fi
#!/bin/bash
set -e

# Function to run benchmarks
run_benchmark() {
  local model_name=$1
  local isl=$2
  local osl=$3
  local num_requests=$4
  local tp_size=$5
  local pp_size=$6
  local ep_size=$7
  local concurrency=$8
  local kv_cache_free_gpu_mem_fraction=$9
  local quantization_precision=${10}
  local quantize_model=${11}
  local max_batch_size=${12}
  local latency_requests=${13}

  # Default to false if not set by environment
  : "${USE_LEGACY_TRT_BACKEND:=false}"

  echo "Running benchmark for $model_name with ISL=$isl, OSL=$osl, TP=$tp_size, PP=$pp_size, EP=$ep_size, Concurrency=$concurrency, KV Cache Fraction=$kv_cache_free_gpu_mem_fraction, Max Batch Size=$max_batch_size, Latency Requests=$latency_requests"

  dataset_file="${mount_base_path}/token-norm-dist_${model_name##*/}_${isl}_${osl}.json"

  python benchmarks/cpp/prepare_dataset.py --tokenizer="$model_name" --stdout token-norm-dist --num-requests="$num_requests" --input-mean="$isl" --output-mean="$osl" --input-stdev=0 --output-stdev=0 > "$dataset_file"




  use_legacy_trt_backend=false
  # shellcheck disable=SC2153
  if [[ "${USE_LEGACY_TRT_BACKEND}" == "true" ]]; then
    echo "Legacy TRT backend forced via USE_LEGACY_TRT_BACKEND=true."
    use_legacy_trt_backend=true
  fi

  if [[ "$use_legacy_trt_backend" == "false" ]]; then
    echo "Using PyTorch backend (default). Skipping engine build."

    # Run throughput directly using --model instead of --engine_dir, passing max_batch_size if specified
    local extra_args=()
    if [[ -n "$max_batch_size" ]]; then
      extra_args+=(--max_batch_size "$max_batch_size")
    fi
    trtllm-bench --model "$model_name" throughput --dataset "$dataset_file" --kv_cache_free_gpu_mem_fraction "${kv_cache_free_gpu_mem_fraction}" --concurrency "$concurrency" --tp "$tp_size" --pp "$pp_size" --ep "$ep_size" --backend pytorch "${extra_args[@]}" | tee "${results_directory}/output_${model_name##*/}_isl${isl}_osl${osl}_tp${tp_size}_pp${pp_size}_ep${ep_size}_throughput.txt"

    # Run latency directly using --model instead of --engine_dir, using full dataset but capped by num_requests
    trtllm-bench --model "$model_name" latency --dataset "$dataset_file" --num_requests "$latency_requests" --kv_cache_free_gpu_mem_fraction "${kv_cache_free_gpu_mem_fraction}" --concurrency "$concurrency" --tp "$tp_size" --pp "$pp_size" --ep "$ep_size" --backend pytorch | tee "${results_directory}/output_${model_name##*/}_isl${isl}_osl${osl}_tp${tp_size}_pp${pp_size}_ep${ep_size}_latency.txt"

  else
    # Legacy engine-based flow (used when user explicitly requests it via USE_LEGACY_TRT_BACKEND=true)
    # Build the engine
    local build_args=()
    if [[ "${quantize_model}" == "true" ]]; then
      build_args+=(--quantization "${quantization_precision}")
    fi

    echo "Building engine with args: ${build_args[*]}"
    trtllm-bench --model "$model_name" build --tp_size "$tp_size" --pp_size "$pp_size" "${build_args[@]}" --dataset "$dataset_file"

    # Find engine dir
    engine_dir=$(find /tmp -maxdepth 3 -type d -name "tp_${tp_size}_pp_${pp_size}" | head -n 1)
    if [[ -z "$engine_dir" ]]; then
      echo "Error: Engine directory not found in /tmp"
      exit 1
    fi
    echo "Found engine directory: $engine_dir"

    # Run throughput
    trtllm-bench --model "$model_name" throughput --dataset "$dataset_file" --engine_dir "$engine_dir" --kv_cache_free_gpu_mem_fraction "${kv_cache_free_gpu_mem_fraction}" --concurrency "$concurrency" --tp "$tp_size" --pp "$pp_size" --ep "$ep_size" --backend tensorrt | tee "${results_directory}/output_${model_name##*/}_isl${isl}_osl${osl}_tp${tp_size}_pp${pp_size}_ep${ep_size}_throughput.txt"

    # Run latency
    trtllm-bench --model "$model_name" latency --dataset "$dataset_file" --num_requests "$latency_requests" --engine_dir "$engine_dir" --kv_cache_free_gpu_mem_fraction "${kv_cache_free_gpu_mem_fraction}" --concurrency "$concurrency" --tp "$tp_size" --pp "$pp_size" --ep "$ep_size" --backend tensorrt | tee "${results_directory}/output_${model_name##*/}_isl${isl}_osl${osl}_tp${tp_size}_pp${pp_size}_ep${ep_size}_latency.txt"

    # Clean up engine dir
    rm -rf "$engine_dir"
  fi

  # Clean up dataset file
  rm -f "$dataset_file"
}

function main() {
  model_name="${MODEL_NAME:-"meta-llama/Llama-3.1-8B"}"
  quantization_precision="${QUANTIZATION_PRECISION:-"fp8"}"
  quantization_precision=$(echo "$quantization_precision" | tr '[:lower:]' '[:upper:]')
  quantize_model="${QUANTIZE_MODEL:-"true"}"
  isl="${ISL:-128}"
  osl="${OSL:-4096}"
  num_requests="${NUM_REQUESTS:-5}"
  tp_size="${TP_SIZE:-1}"
  pp_size="${PP_SIZE:-1}"
  ep_size="${EP_SIZE:-1}"
  concurrency="${CONCURRENCY:-128}"
  kv_cache_free_gpu_mem_fraction="${KV_CACHE_FREE_GPU_MEM_FRACTION:-0.8}"
  max_batch_size="${MAX_BATCH_SIZE:-}"
  latency_requests="${LATENCY_REQUESTS:-20}"

  ## get the benchmark scripts directory from TRT-LLM github repo.
  git clone --depth=1 --no-checkout https://github.com/NVIDIA/TensorRT-LLM.git
  cd TensorRT-LLM/
  git sparse-checkout set benchmarks
  git checkout

  ## update gib lib path
  export LD_LIBRARY_PATH="/usr/local/gib/lib64:$LD_LIBRARY_PATH"

  run_benchmark "$model_name" "$isl" "$osl" "$num_requests" "$tp_size" "$pp_size" "$ep_size" "$concurrency" "$kv_cache_free_gpu_mem_fraction" "$quantization_precision" "$quantize_model" "$max_batch_size" "$latency_requests"
}

mount_base_path="${CONTAINER_RECIPE_DIRECTORY:-/tmp/ubench}"
results_relative_path="${RESULTS_RELATIVE_PATH:-results}"
results_directory="${mount_base_path}/${results_relative_path}"
mkdir -p "${results_directory}"
chmod 777 "${results_directory}"

export MODEL_NAME=Qwen/Qwen3-32B
export ISL=1024
export OSL=8192
export NUM_REQUESTS=1000
export TP_SIZE=4
export PP_SIZE=1
export EP_SIZE=1
export QUANTIZATION_PRECISION=fp8
export QUANTIZE_MODEL=false
export CONCURRENCY=128
export LATENCY_REQUESTS=20
export HF_TOKEN="<YOUR_HF_TOKEN_HERE>"
main