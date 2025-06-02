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
    isl=128
    osl=128
    num_requests=30000

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
            isl="$2"
            shift 2
            ;;
        --osl)
            osl="$2"
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
        next_arg="${SERVING_CONFIG[$((index + 1))]}"

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
}

print_configuration() {
    echo "TensorRT-LLM benchmark arguments received:"
    echo "  $@"
    echo ""
    echo "--------------------------------"
    echo "--- Parsed Arguments Summary ---"
    echo "model name:              $model_name"
    echo "input seq length:        $isl"
    echo "output seq length:       $osl"
    echo "number of requests:      $num_requests"
    echo "tensor parallel size:    $tp_size"
    echo "pipeline parallel size:  $pp_size"
    echo "--------------------------------"
}

download_model() {
    echo "Downloading model from HuggingFace"
    huggingface-cli download $model_name --exclude="*original*" --local-dir ${MODEL_DOWNLOAD_DIR} --local-dir-use-symlinks False
}

# Function to run benchmarks
run_benchmark() {
    local model_name=$1
    local isl=$2
    local osl=$3
    local num_requests=$4
    local tp_size=$5
    local pp_size=$6

    echo "Running benchmark for $model_name with ISL=$isl, OSL=$osl, TP=$tp_size, PP=$pp_size"

    dataset_file="/ssd/token-norm-dist_${model_name##*/}_${isl}_${osl}_tp${tp_size}.json"
    output_file="/ssd/output_${model_name##*/}_isl${isl}_osl${osl}_tp${tp_size}.txt"

    echo "Preparing dataset"
    python3 /workspace/tensorrtllm_backend/tensorrt_llm/benchmarks/cpp/prepare_dataset.py \
        --tokenizer=$model_name \
        --stdout token-norm-dist \
        --num-requests=$num_requests \
        --input-mean=$isl \
        --output-mean=$osl \
        --input-stdev=0 \
        --output-stdev=0 >$dataset_file

    echo "Building engine"
    trtllm-bench \
        --model $model_name \
        --model_path /ssd/${model_name} \
        --workspace /ssd build \
        --tp_size $tp_size \
        --quantization FP8 \
        --dataset $dataset_file

    engine_dir="/ssd/${model_name}/tp_${tp_size}_pp_${pp_size}"

    # Save throughput output to a file
    echo "Running throughput benchmark"
    trtllm-bench \
        --model $model_name \
        --model_path /ssd/${model_name} throughput \
        --dataset $dataset_file \
        --engine_dir $engine_dir \
        --kv_cache_free_gpu_mem_fraction 0.95 >$output_file

    cat $output_file
    gsutil cp $output_file /gcs/benchmark_logs/trtllm/

    rm -rf $engine_dir
    rm -f $dataset_file
}

# Main function to run the benchmark
main() {
    # parse arguments
    validate_model_name
    parse_arguments "$@"
    parse_serving_config
    print_configuration "$@"

    # download model
    download_model

    # run benchmark
    mkdir -p /gcs/benchmark_logs/trtllm
    echo "Running benchmarks"
    run_benchmark "$model_name" $isl $osl $num_requests $tp_size $pp_size
}

# Set environment variables
export HF_HOME=/ssd
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/tensorrt/lib

# Run the main function
main "$@"