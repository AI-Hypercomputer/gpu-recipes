# Copyright 2026 Google LLC
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
    kv_cache_free_gpu_mem_fraction=${SERVING_CONFIG_DICT["kv_cache_free_gpu_mem_fraction"]:=0.95}
    modality=${SERVING_CONFIG_DICT["modality"]:=""}
    streaming=${SERVING_CONFIG_DICT["streaming"]:="false"}
    max_input_len=${SERVING_CONFIG_DICT["max_input_len"]:=""}
    max_batch_size=${SERVING_CONFIG_DICT["max_batch_size"]:=""}
    max_num_tokens=${SERVING_CONFIG_DICT["max_num_tokens"]:=""}
    custom_dataset=${SERVING_CONFIG_DICT["dataset"]:=""}
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
    echo "expert parallel size:    $ep_size"
    echo "backend:                 $backend"
    echo "modality:                $modality"
    echo "streaming:               $streaming"
    echo "max input length:        $max_input_len"
    echo "max batch size:          $max_batch_size"
    echo "max num tokens:          $max_num_tokens"
    echo "kv_cache_free_gpu_mem_fraction: $kv_cache_free_gpu_mem_fraction"
    echo "--------------------------------"
}

download_model() {
    echo "Downloading model from HuggingFace... This may take a while when downloading for the first time."
    echo "NOTE: by default, huggingface-cli response can be verbose."
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
    local ep_size=$7
    local backend=$8
    local kv_cache_free_gpu_mem_fraction=$9

    echo "Running benchmark for $model_name with ISL=$isl, OSL=$osl, TP=$tp_size, PP=$pp_size, EP=$ep_size, backend=$backend"

    vl_args=""
    if [ -n "$modality" ]; then vl_args="$vl_args --modality $modality"; fi
    if [ "$streaming" == "true" ]; then vl_args="$vl_args --streaming"; fi
    if [ -n "$max_input_len" ]; then vl_args="$vl_args --max_input_len $max_input_len"; fi
    if [ -n "$max_batch_size" ]; then vl_args="$vl_args --max_batch_size $max_batch_size"; fi
    if [ -n "$max_num_tokens" ]; then vl_args="$vl_args --max_num_tokens $max_num_tokens"; fi

    dataset_file=$custom_dataset
    # If custom_dataset is not set, generate a textual dataset with tokens sampled in normal distribution
    if [ -z "$dataset_file" ]; then
        dataset_file="/scratch/token-norm-dist_${model_name##*/}_${isl}_${osl}_tp${tp_size}.json"
        echo "Preparing dataset"
        python3 $TRTLLM_DIR/benchmarks/cpp/prepare_dataset.py \
            --tokenizer=$model_name \
            --stdout token-norm-dist \
            --num-requests=$num_requests \
            --input-mean=$isl \
            --output-mean=$osl \
            --input-stdev=0 \
            --output-stdev=0 >$dataset_file
    fi 

    output_file="/scratch/output_${model_name##*/}_isl${isl}_osl${osl}_tp${tp_size}.txt"
    extra_args_file="/tmp/extra_llm_api_args.yaml"
    extra_args=""
    if [ -f "$extra_args_file" ]; then
        extra_args="--extra_llm_api_options $extra_args_file"
    fi

    export TOKENIZERS_PARALLELISM=false
    echo "enable_cuda_graph: false" > /tmp/extra_llm_api_args.yaml

    if [[ $backend == "pytorch" ]]; then
        echo "Running throughput benchmark"
        export NCCL_P2P_LEVEL=PHB
        trtllm-bench \
        --model $model_name \
        --model_path /scratch/${model_name} throughput --tp $tp_size --pp $pp_size \
        --dataset $dataset_file \
        --num_requests $num_requests \
        --tp $tp_size \
        --pp $pp_size \
        --ep $ep_size \
        --backend "pytorch" \
        --kv_cache_free_gpu_mem_fraction $kv_cache_free_gpu_mem_fraction \
        $extra_args $vl_args > $output_file
    else
        echo "Building engine"
        trtllm-bench \
            --model $model_name \
            --model_path /scratch/${model_name} \
            --workspace /scratch build \
            --tp_size $tp_size \
            --pp_size $pp_size \
            --quantization FP8 \
            --dataset $dataset_file

        engine_dir="/scratch/${model_name}/tp_${tp_size}_pp_${pp_size}"

        # Save throughput output to a file
        echo "Running throughput benchmark"
        trtllm-bench \
            --model $model_name \
            --model_path /scratch/${model_name} throughput --tp $tp_size --pp $pp_size \
            --dataset $dataset_file \
            --engine_dir $engine_dir \
            --kv_cache_free_gpu_mem_fraction $kv_cache_free_gpu_mem_fraction $extra_args $vl_args >$output_file
    fi

    cat $output_file
    gcloud storage cp $output_file /gcs/benchmark_logs/trtllm/

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
    run_benchmark "$model_name" $isl $osl $num_requests $tp_size $pp_size $ep_size $backend $kv_cache_free_gpu_mem_fraction
}

# Set environment variables
export HF_HOME=/scratch
export LD_LIBRARY_PATH=/usr/local/gib/lib64:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/tensorrt/lib

# Run the main function
main "$@"
