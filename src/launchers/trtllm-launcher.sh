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

#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -eux

echo "TensorRT-LLM benchmark arguments received:"
echo "  $@"
echo ""

# Define model download directory if not set
MODEL_DOWNLOAD_DIR=${MODEL_DOWNLOAD_DIR:-/ssd}

# Function to validate model name
validate_model_name() {
    if [ -z "$MODEL_NAME" ]; then
        echo "Error: MODEL_NAME environment variable is not set."
        exit 1
    fi
    echo "Using MODEL_NAME: $MODEL_NAME"
}

# Function to parse arguments
parse_arguments() {
    model_name=$MODEL_NAME
    isl_list="128"
    osl_list="128"
    num_requests=1024

    # Parse known arguments and check for unknown option or missing argument
    PARSED_OPTIONS=$(getopt -o "" -l model_name:,isl:,osl:,num_requests: -- "$@")
    if [ $? -ne 0 ]; then
        echo "Error: Failed to parse arguments. Check for invalid options or missing values."
        exit 1
    fi

    # set the shell's positional parameters
    eval set -- "$PARSED_OPTIONS"

    while true; do
        case "$1" in
        --model_name)
            model_name="$2"
            shift 2
            ;;
        --isl)
            isl_list="$2"
            shift 2
            ;;
        --osl)
            osl_list="$2"
            shift 2
            ;;
        --num_requests)
            num_requests="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Internal error: Argument parsing issue. Unexpected option: $1"
            exit 1
            ;;
        esac
    done

    SERVING_CONFIG=("$@")
}

# Function to parse serving config
parse_serving_config() {
    declare -g -A SERVING_CONFIG_DICT

    for ((index = 0; index < ${#SERVING_CONFIG[@]}; )); do
        current_arg="${SERVING_CONFIG[$index]}"
        next_arg=${SERVING_CONFIG[$((index + 1))]:-}

        # Handle --key=value format
        if [[ "$current_arg" =~ ^--[^=]+=.+ ]]; then
            key=$(echo "$current_arg" | cut -d'=' -f1 | sed 's/--//')
            value=$(echo "$current_arg" | cut -d'=' -f2-)
            SERVING_CONFIG_DICT["$key"]="$value"
            ((index++))
        # Handle --key value format
        elif [[ "$current_arg" =~ ^--[^=]+$ && -n "$next_arg" && ! "$next_arg" =~ ^-- ]]; then
            # Check if:
            # 1. Current arg starts with -- and has no '=' (e.g., --key)
            # 2. There IS a next argument (`-n "$next_arg"`)
            # 3. The next argument does NOT start with -- (meaning it's a value, not another option)
            key=$(echo "$current_arg" | sed 's/--//')
            value="$next_arg"
            SERVING_CONFIG_DICT["$key"]="$value"
            ((index += 2))
        # Handle --flag (boolean flag without a value)
        elif [[ "$current_arg" =~ ^--[^=]+$ ]]; then
            # If the key was pre-defined with a default, this will overwrite it to 'true'.
            # If not pre-defined, it will create it.
            key=$(echo "$current_arg" | sed 's/--//')
            SERVING_CONFIG_DICT["$key"]="true"
            ((index++))
        else
            ((index++))
        fi
    done

    tp_size=${SERVING_CONFIG_DICT["tp_size"]:=8}
    pp_size=${SERVING_CONFIG_DICT["pp_size"]:=1}
    ep_size=${SERVING_CONFIG_DICT["ep_size"]:=1}
    backend=${SERVING_CONFIG_DICT["backend"]:="tensorrt"}
    kv_cache_free_gpu_mem_fraction=${SERVING_CONFIG_DICT["kv_cache_free_gpu_mem_fraction"]:=0.70}
    modality=${SERVING_CONFIG_DICT["modality"]:=""}
    streaming=${SERVING_CONFIG_DICT["streaming"]:="false"}
    max_input_len=${SERVING_CONFIG_DICT["max_input_len"]:=""}
    max_batch_size=${SERVING_CONFIG_DICT["max_batch_size"]:=""}
    custom_dataset=${SERVING_CONFIG_DICT["dataset"]:=""}
    quantization=${SERVING_CONFIG_DICT["quantization"]:="FP8"}
}

print_configuration() {
    echo "TensorRT-LLM benchmark arguments received:"
    echo "  $@"
    echo ""
    echo "--------------------------------"
    echo "--- Parsed Arguments Summary ---"
    echo "model name:              $model_name"
    echo "input seq lengths:       $isl_list"
    echo "output seq lengths:      $osl_list"
    echo "number of requests:      $num_requests"
    echo "tensor parallel size:    $tp_size"
    echo "pipeline parallel size:  $pp_size"
    echo "expert parallel size:    $ep_size"
    echo "backend:                 $backend"
    echo "modality:                $modality"
    echo "streaming:               $streaming"
    echo "max input length:        $max_input_len"
    echo "quantization:            $quantization"
    echo "max batch size:          $max_batch_size"
    echo "kv_cache_free_gpu_mem_fraction: $kv_cache_free_gpu_mem_fraction"
    echo "--------------------------------"
}

download_model() {
    echo "Downloading model from HuggingFace... This may take a while when downloading for the first time."
    echo "NOTE: by default, huggingface-cli response can be verbose."
    # Ensure we download into a model-specific subdirectory
    huggingface-cli download $model_name --exclude="*original*" --local-dir ${MODEL_DOWNLOAD_DIR}/${model_name##*/} --local-dir-use-symlinks False
}

# Function to run benchmarks
run_benchmark() {
    local model_name=$1
    local isl=$2
    local osl=$3
    local num_requests=$4
    local tp_size=$5
    local pp_size=$6
    local ep_size=$7
    local backend=$8
    local kv_cache_free_gpu_mem_fraction=$9

    local engine_dir=""
    local generated_dataset=""
    local output_file="${MODEL_DOWNLOAD_DIR}/output_${model_name##*/}_isl${isl}_osl${osl}_tp${tp_size}.txt"
    local benchmark_rc=0

    export TOKENIZERS_PARALLELISM=false
    
    # Emergency cleanup on crash
    cleanup_on_exit() {
        local exit_code=$?
        if [[ $exit_code -ne 0 ]]; then
            echo "Benchmark interrupted (Exit code: $exit_code). Cleaning up temporary files..."
            [[ -n "${engine_dir:-}" && -d "${engine_dir:-}" ]] && rm -rf "${engine_dir}" || true
            [[ -n "${generated_dataset:-}" && -f "${generated_dataset:-}" ]] && rm -f "${generated_dataset}" || true
        fi
    }
    trap cleanup_on_exit EXIT SIGINT SIGTERM

    echo "Running benchmark for $model_name with ISL=$isl, OSL=$osl, TP=$tp_size, PP=$pp_size, EP=$ep_size, backend=$backend"

    vl_args=""
    if [ -n "$modality" ]; then vl_args="$vl_args --modality $modality"; fi
    if [ "$streaming" == "true" ]; then vl_args="$vl_args --streaming"; fi
    if [ -n "$max_input_len" ]; then vl_args="$vl_args --max_input_len $max_input_len"; fi
    if [ -n "$max_batch_size" ]; then vl_args="$vl_args --max_batch_size $max_batch_size"; fi

    dataset_file=$custom_dataset
    local tokenizer_arg=$model_name
    if [ -d "/serving-model" ]; then
        tokenizer_arg="/serving-model"
    fi

    # Point 3 & 4: Tokenizer logic
    if [ -z "$dataset_file" ]; then
        dataset_file="${MODEL_DOWNLOAD_DIR}/token-norm-dist_${model_name##*/}_${isl}_${osl}_tp${tp_size}.json"
        generated_dataset=$dataset_file
        echo "Preparing dataset"
        python3 $TRTLLM_DIR/benchmarks/cpp/prepare_dataset.py \
            --tokenizer=$tokenizer_arg \
            --stdout token-norm-dist \
            --num-requests=$num_requests \
            --input-mean=$isl \
            --output-mean=$osl \
            --input-stdev=0 \
            --output-stdev=0 >$dataset_file
    fi 

    # Clean up any leftover extra args to prevent "invalid argument: enable_cuda_graph"
    rm -f /tmp/extra_llm_api_args.yaml || true

    extra_args_file="/tmp/extra_llm_api_args.yaml"
    extra_args=""
    if [ -f "$extra_args_file" ]; then
        extra_args="--extra_llm_api_options $extra_args_file"
    fi

    # Point 5: Determine model path
    local model_path_arg="${MODEL_DOWNLOAD_DIR}/${model_name}"
    if [ -d "/serving-model" ]; then
        model_path_arg="/serving-model"
    fi

    # Point 6, 7, 8: Throughput call
    if [[ $backend == "pytorch" ]]; then
        echo "Running throughput benchmark"
        set +e
        trtllm-bench \
        --model $model_name \
        --model_path $model_path_arg throughput \
        --dataset $dataset_file \
        --num_requests $num_requests \
        --tp $tp_size \
        --pp $pp_size \
        --ep $ep_size \
        --backend $backend \
        --kv_cache_free_gpu_mem_fraction $kv_cache_free_gpu_mem_fraction \
        $extra_args $vl_args | tee "$output_file"
        benchmark_rc=${PIPESTATUS[0]}
        set -e
    else
        # Point 9: Build engine
        trtllm-bench \
            --model $model_name \
            --model_path $model_path_arg \
            --workspace ${MODEL_DOWNLOAD_DIR} build \
            --tp_size $tp_size \
            --pp_size $pp_size \
            --quantization $quantization \
            --dataset $dataset_file

        # Point 10: Engine Dir path
        engine_dir="${model_path_arg}/tp_${tp_size}_pp_${pp_size}"

        # Point 11 & 12: Run throughput with engine
        echo "Running throughput benchmark"
        set +e
        trtllm-bench \
            --model $model_name \
            --model_path $model_path_arg throughput \
            --dataset $dataset_file \
            --engine_dir $engine_dir \
            --tp $tp_size \
            --pp $pp_size \
            --ep $ep_size \
            --backend $backend \
            --kv_cache_free_gpu_mem_fraction $kv_cache_free_gpu_mem_fraction \
            $extra_args | tee "$output_file"
        benchmark_rc=${PIPESTATUS[0]}
        set -e
    fi

    # Point 13: Immediate sync of results to GCS
    mkdir -p /gcs/benchmark_logs/trtllm || true
    cp "$output_file" /gcs/benchmark_logs/trtllm/ || true

    # Standard cleanup after successful iteration
    [[ -n "${engine_dir:-}" && -d "${engine_dir:-}" ]] && rm -rf "${engine_dir}" || true
    [[ -n "${dataset_file:-}" ]] && rm -f "${dataset_file}" || true
    trap - EXIT SIGINT SIGTERM

    if [ $benchmark_rc -ne 0 ]; then
        echo "Error: Benchmark command failed with exit code $benchmark_rc."
        exit $benchmark_rc
    fi
}

# Main function to run the benchmark
main() {
    # parse arguments
    validate_model_name
    parse_arguments "$@"
    parse_serving_config

    # download model
    download_model

    # Convert comma-separated lists to arrays
    # Remove spaces and split by comma
    IFS=',' read -ra ISL_ARRAY <<< "${isl_list// /}"
    IFS=',' read -ra OSL_ARRAY <<< "${osl_list// /}"

    # Validate array lengths match
    if [[ ${#ISL_ARRAY[@]} -ne ${#OSL_ARRAY[@]} ]]; then
        echo "Error: The number of values in --isl and --osl must be the same."
        echo "ISL count: ${#ISL_ARRAY[@]}, OSL count: ${#OSL_ARRAY[@]}"
        exit 1
    fi

    print_configuration "$@"

    # run benchmark
    mkdir -p /gcs/benchmark_logs/trtllm
    echo "Running benchmarks"
    for i in "${!ISL_ARRAY[@]}"; do
        current_isl="${ISL_ARRAY[$i]}"
        current_osl="${OSL_ARRAY[$i]}"
        
        echo "Starting iteration $((i+1))/${#ISL_ARRAY[@]}: ISL=$current_isl, OSL=$current_osl, REQ=$num_requests"
        run_benchmark "$model_name" "$current_isl" "$current_osl" "$num_requests" $tp_size $pp_size $ep_size $backend $kv_cache_free_gpu_mem_fraction
    done

    echo "-----------------------------------------------------------"
    echo "Benchmarks complete. Keeping container alive for result inspection."
    sleep infinity
}

# Force load the container's internal NCCL to resolve symbol mismatches on GKE
for loc in "/usr/local/lib/python3.12/dist-packages/torch/lib/libnccl.so.2" "/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2" "/usr/local/cuda/lib64/libnccl.so.2" "/usr/lib/x86_64-linux-gnu/libnccl.so.2"; do
    if [ -f "$loc" ]; then
        if nm -D "$loc" 2>/dev/null | grep -q "ncclCommWindowDeregister"; then
            export LD_PRELOAD="$loc"
            break
        fi
    fi
done

# Run the main function
main "$@"