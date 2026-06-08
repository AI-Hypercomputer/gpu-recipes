# Cloud GPU performance benchmark recipes

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

This repository contains recipes that provide instructions to reproduce specific
workload performance measurements, which are part of a confidential benchmarking
program. These recipes focus on helping you reliably achieve performance metrics,
such as throughput, that demonstrate the combined hardware and software stack on GPUs.

**Note:** The recipes in this repository are not designed as general-purpose code
samples or tutorials for using Compute Engine-based products.

## Intended audience

This content is for you if you are a customer or partner who needs to:

- Validate hardware performance with your suppliers.
- Inform purchasing decisions using the benchmarking data.
- Reproduce optimal performance scenarios before you customize workflows for your
  own requirements.
  
## How to use these recipes

To reproduce a benchmark, follow these steps:

1. **Identify your requirements:** determine the model, GPU type, workload, framework,
   and orchestrator that you are interested in.
2. **Select a recipe:** based on your requirements use the
   [Benchmark support matrix](#benchmarks-support-matrix) to find a recipe that meets your needs.
3. **Follow the recipe:** each recipe will provide you with procedures to complete the following tasks:
   * prepare your environment.
   * run the benchmark.
   * analyze the benchmarks results. This includes not just the results but detailed logs for further analysis.
   You can automate your infrastructure setup using Cluster Toolkit. For more information, see
   [Automated GPU environment deployment with Cluster Toolkit](https://cloud.google.com/cluster-toolkit/docs/deploy/gke/gke-gpu-overview).

## Benchmarks support matrix

### Training benchmarks A3 Mega

Models            | GPU Machine Type                                                                                          | Framework | Workload Type | Orchestrator | Link to the recipe
----------------- | --------------------------------------------------------------------------------------------------------- | --------- | ------------- | ------------ | ------------------
**GPT3-175B**     | [A3 Mega (NVIDIA H100)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-mega-vms) | NeMo (25.07) | Pre-training  | GKE          | [Link](./training/a3mega/gpt3_175b/nemo-gke/nemo2507/recipe/)
**Llama-3-70B**   | [A3 Mega (NVIDIA H100)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-mega-vms) | NeMo (25.07) | Pre-training  | GKE          | [Link](./training/a3mega/llama3_70b/nemo-gke/nemo2507/128gpus-bf16/recipe/)
**Mixtral-8-7B**  | [A3 Mega (NVIDIA H100)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-mega-vms) | NeMo (25.07) | Pre-training  | GKE          | [Link](./training/a3mega/mixtral_8x7b/nemo-gke/nemo2507/recipe/)


### Training benchmarks A3 Ultra

Models             | GPU Machine Type                                                                                            | Framework | Workload Type | Orchestrator | Link to the recipe
------------------ | ----------------------------------------------------------------------------------------------------------- | --------- | ------------- | ------------ | ------------------
**Llama-3.1-70B**  | [A3 Ultra (NVIDIA H200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-ultra-vms) | MaxText | Pre-training  | GKE          | [Link](./training/a3ultra/llama3-1-70b/maxtext-pretraining-gke/README.md)
**Llama-3.1-70B**  | [A3 Ultra (NVIDIA H200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-ultra-vms) | NeMo (24.07) | Pre-training  | GKE          | [Link](./training/a3ultra/llama3_70b/nemo-gke/nemo2407/recipe/)
**Llama-3-70B**    | [A3 Ultra (NVIDIA H200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-ultra-vms) | Megatron-Bridge (26.02) | Pre-training  | GKE          | [Link](./training/a3ultra/llama3_70b/megatron-bridge-gke/nemo2602/)
**Llama-3-70B**    | [A3 Ultra (NVIDIA H200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-ultra-vms) | Megatron-Bridge (25.11) | Pre-training  | Slurm          | [Link](./training/a3ultra/llama3_70b/megatron-bridge-slurm/nemo2511/)
**Llama-3-8B**     | [A3 Ultra (NVIDIA H200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-ultra-vms) | Megatron-Bridge (25.11) | Pre-training  | Slurm          | [Link](./training/a3ultra/llama3_8b/megatron-bridge-slurm/nemo2511/)
**Llama-3.1-405B** | [A3 Ultra (NVIDIA H200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-ultra-vms) | MaxText | Pre-training  | GKE          | [Link](./training/a3ultra/llama3-1-405b/maxtext-pretraining-gke/README.md)
**Llama-3.1-405B** | [A3 Ultra (NVIDIA H200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-ultra-vms) | NeMo (24.12) | Pre-training  | GKE          | [Link](./training/a3ultra/llama31_405b/nemo-gke/nemo2412/recipe/)
**Mixtral-8-7B**   | [A3 Ultra (NVIDIA H200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-ultra-vms) | NeMo (24.07) | Pre-training  | GKE          | [Link](./training/a3ultra/mixtral_8x7b/nemo-gke/nemo2407/recipe/)
**DeepSeek-V3**    | [A3 Ultra (NVIDIA H200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-ultra-vms) | Megatron-Bridge (26.02) | Pre-training  | GKE          | [Link](./training/a3ultra/deepseek_v3/megatron-bridge-gke/nemo2602/)
**GPT OSS 120B**   | [A3 Ultra (NVIDIA H200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-ultra-vms) | NeMo (26.02) | Pre-training  | GKE          | [Link](./training/a3ultra/gpt_oss_120b/nemo-gke/nemo2602/)
**Qwen-3-30B**     | [A3 Ultra (NVIDIA H200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-ultra-vms) | NeMo (26.02) | Pre-training  | GKE          | [Link](./training/a3ultra/qwen3_30b_a3b/nemo-gke/nemo2602/)
**Wan-2.1**        | [A3 Ultra (NVIDIA H200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-ultra-vms) | Megatron-Bridge (26.02) | Pre-training  | GKE          | [Link](./training/a3ultra/wan/megatron-bridge-gke/nemo2602/)


### Training benchmarks A4

Models             | GPU Machine Type                                                                                     | Framework / Library | Workload Type | Orchestrator | Link to the recipe
------------------ | ---------------------------------------------------------------------------------------------------- | ------------------- | ------------- | ------------ | ------------------
**Llama-3.1-70B**  | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)      | MaxText | Pre-training  | GKE          | [Link](./training/a4/llama3-1-70b/maxtext-pretraining-gke/README.md)
**Llama-3.1-70B**  | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)      | NeMo (25.07) | Pre-training  | GKE          | [Link](./training/a4/llama3_70b/nemo-gke/nemo2507/)
**Llama-3.1-70B**  | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)      | NeMo (26.02) | Pre-training  | GKE          | [Link](./training/a4/llama3_70b/nemo-gke/nemo2602/)
**Llama-3.1-70B**  | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)      | Megatron-Bridge (25.09) | Pre-training  | Slurm          | [Link](./training/a4/llama3_70b/megatron-bridge-slurm/nemo2509/)
**Llama-3.1-405B** | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)      | MaxText | Pre-training  | GKE          | [Link](./training/a4/llama3-1-405b/maxtext-pretraining-gke/README.md)
**Llama-3.1-405B** | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)      | NeMo (25.07) | Pre-training  | GKE          | [Link](./training/a4/llama31_405b/nemo-gke/nemo2507/)
**Llama-3.1-405B** | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)      | NeMo (26.02) | Pre-training  | GKE          | [Link](./training/a4/llama31_405b/nemo-gke/nemo2602/)
**Llama-3.1-405B** | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)      | Megatron-Bridge (25.09) | Pre-training  | Slurm          | [Link](./training/a4/llama31_405b/megatron-bridge-slurm/nemo2509/)
**Mixtral-8-7B**   | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)      | NeMo (25.07) | Pre-training  | GKE          | [Link](./training/a4/mixtral_8x7b/nemo-gke/nemo2507/recipe/)
**PaliGemma2**     | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)      | Hugging Face Accelerate | Finetuning | GKE          | [Link](./training/a4/paligemma2/README.md)
**DeepSeek-V3**    | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)      | Megatron-Bridge (25.11) | Pre-training  | GKE          | [Link](./training/a4/deepseek_v3/megatron-bridge-gke/nemo2511/)
**DeepSeek-V3**    | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)      | Megatron-Bridge (26.02) | Pre-training  | GKE          | [Link](./training/a4/deepseek_v3/megatron-bridge-gke/nemo2602/)
**GPT OSS 120B**   | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)      | Megatron-Bridge (26.02) | Pre-training  | GKE          | [Link](./training/a4/gpt_oss_120b/megatron-bridge-gke/nemo2602/)
**Llama-3-8B**     | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)      | Megatron-Bridge (26.02) | Pre-training  | GKE          | [Link](./training/a4/llama3-8b/megatron-bridge-gke/nemo2602/)
**Qwen-3-235B**    | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)      | Megatron-Bridge (25.11) | Pre-training  | GKE          | [Link](./training/a4/qwen3_235b_a22b/megatron-bridge-gke/nemo2511/)
**Qwen-3-235B**    | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)      | Megatron-Bridge (26.02) | Pre-training  | GKE          | [Link](./training/a4/qwen3_235b_a22b/megatron-bridge-gke/nemo2602/)
**Qwen-3-235B**    | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)      | Megatron-Bridge (25.11) | Pre-training  | Slurm          | [Link](./training/a4/qwen3_235b_a22b/megatron-bridge-slurm/nemo2511/)
**Qwen-3-30B**     | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)      | NeMo (26.02) | Pre-training  | GKE          | [Link](./training/a4/qwen3_30b_a3b/nemo-gke/nemo2602/)
**Wan-2.1-14B**    | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)      | NeMo (25.11) | Pre-training  | GKE          | [Link](./training/a4/wan_14b/nemo-gke/nemo2511/)


### Training benchmarks A4X

Models             | GPU Machine Type                                                                                     | Framework | Workload Type | Orchestrator | Link to the recipe
------------------ | ---------------------------------------------------------------------------------------------------- | --------- | ------------- | ------------ | ------------------
**Llama-3.1-8B** | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms) | NeMo (25.07) | Pre-training | GKE | [Link](./training/a4x/llama3_8b/nemo-gke/nemo2507/)
**Llama-3.1-8B** | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms) | Megatron-Bridge (25.11) | Pre-training | GKE | [Link](./training/a4x/llama3_8b/megatron-bridge-gke/nemo2511/)
**Llama-3.1-8B** | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms) | Megatron-Bridge (25.11) | Pre-training | Slurm | [Link](./training/a4x/llama3_8b/megatron-bridge-slurm/nemo2511/)
**Llama-3.1-70B** | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms) | NeMo (25.07) | Pre-training | GKE | [Link](./training/a4x/llama3_70b/nemo-gke/nemo2507/)
**Llama-3.1-70B** | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms) | Megatron-Bridge (26.02) | Pre-training | GKE | [Link](./training/a4x/llama3_70b/nemo-gke/nemo2602/)
**Llama-3.1-405B** | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms) | NeMo (25.07) | Pre-training | GKE | [Link](./training/a4x/llama31_405b/nemo-gke/nemo2507/)
**Llama-3.1-405B** | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms) | NeMo (26.02) | Pre-training | GKE | [Link](./training/a4x/llama31_405b/nemo-gke/nemo2602/)
**Llama-3.1-405B** | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms) | Megatron-Bridge (26.02) | Pre-training | GKE | [Link](./training/a4x/llama31_405b/megatron-bridge-gke/nemo2602/)
**Llama-3.1-405B** | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms) | Megatron-Bridge (25.09) | Pre-training | Slurm | [Link](./training/a4x/llama31_405b/megatron-bridge-slurm/nemo2509/)
**Nemotron-4-340B** | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms) | NeMo (25.09) | Pre-training | GKE | [Link](./training/a4x/nemotron4_340b/nemo-gke/nemo2509/)
**Wan-2.1-14B** | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms) | NeMo (25.11) | Pre-training | GKE | [Link](./training/a4x/wan_14b/nemo-gke/nemo2511/)
**Wan-2.1-14B** | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms) | NeMo (26.02) | Pre-training | GKE | [Link](./training/a4x/wan_14b/nemo-gke/nemo2602/)
**Wan-2.1-14B** | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms) | NeMo (25.11) | Pre-training | Slurm | [Link](./training/a4x/wan_14b/nemo-slurm/nemo2511/)
**DeepSeek-V3** | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms) | Megatron-Bridge (25.11) | Pre-training | GKE | [Link](./training/a4x/deepseek_v3/megatron-bridge-gke/nemo2511/)
**Qwen-3-235B** | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms) | Megatron-Bridge (25.11) | Pre-training | GKE | [Link](./training/a4x/qwen3_235b_a22b/megatron-bridge-gke/nemo2511/)
**Qwen-3-235B** | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms) | Megatron-Bridge (25.11) | Pre-training | Slurm | [Link](./training/a4x/qwen3_235b_a22b/megatron-bridge-slurm/nemo2511/)
**Qwen-3-30B** | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms) | Megatron-Bridge (25.11) | Pre-training | GKE | [Link](./training/a4x/qwen3_30b_a3b/megatron-bridge-gke/nemo2511/)
**Qwen-3-30B** | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms) | Megatron-Bridge (25.11) | Pre-training | Slurm | [Link](./training/a4x/qwen3_30b_a3b/megatron-bridge-slurm/nemo2511/)


### Inference benchmarks A3 Mega

| Models           | GPU Machine Type | Framework | Workload Type       | Orchestrator | Link to the recipe |
| ---------------- | ---------------- | --------- | ------------------- | ------------ | ------------------ |
| **Llama-4**      | [A3 Mega (NVIDIA H100)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-mega-vms)    | SGLang  | Inference   | GKE          | [Link](./inference/a3mega/llama-4/vllm-serving-gke/README.md)
| **DeepSeek R1 671B**     | [A3 Mega (NVIDIA H100)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-mega-vms)    | SGLang  | Inference   | GKE          | [Link](./inference/a3mega/deepseek-r1-671b/sglang-serving-gke/README.md)
| **DeepSeek R1 671B**     | [A3 Mega (NVIDIA H100)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-mega-vms)    | vLLM  | Inference   | GKE          | [Link](./inference/a3mega/deepseek-r1-671b/vllm-serving-gke/README.md)

### Inference benchmarks A3 Ultra

| Models           | GPU Machine Type | Framework | Workload Type       | Orchestrator | Link to the recipe |
| ---------------- | ---------------- | --------- | ------------------- | ------------ | ------------------ |
| **GPT OSS 120B**     | [A3 Ultra (NVIDIA H200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-ultra-vms)    | vLLM  | Inference   | GKE          | [Link](./inference/a3ultra/single-host-serving/vllm/README.md)
| **Llama-4**      | [A3 Ultra (NVIDIA H200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-ultra-vms)    | vLLM  | Inference   | GKE          | [Link](./inference/a3ultra/single-host-serving/vllm/README.md)
| **Llama-3.1-405B**     | [A3 Ultra (NVIDIA H200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-ultra-vms)    | TensorRT-LLM  | Inference   | GKE          | [Link](./inference/a3ultra/single-host-serving/trtllm/README.md)
| **DeepSeek R1 671B**     | [A3 Ultra (NVIDIA H200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-ultra-vms)    | SGLang  | Inference   | GKE          | [Link](./inference/a3ultra/single-host-serving/sglang/README.md)
| **DeepSeek R1 671B**     | [A3 Ultra (NVIDIA H200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-ultra-vms)    | vLLM  | Inference   | GKE          | [Link](./inference/a3ultra/single-host-serving/vllm/README.md)

### Inference benchmarks A4

| Models           | GPU Machine Type | Framework | Workload Type       | Orchestrator | Link to the recipe |
| ---------------- | ---------------- | --------- | ------------------- | ------------ | ------------------ |
| **DeepSeek R1 671B**     | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)    | vLLM  | Inference   | GKE          | [Link](./inference/a4/single-host-serving/vllm/README.md)
| **DeepSeek R1 671B**     | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)    | SGLang  | Inference   | GKE          | [Link](./inference/a4/single-host-serving/sglang/README.md)
| **DeepSeek R1 671B**     | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)    | TensorRT-LLM   | Inference   | GKE          | [Link](./inference/a4/single-host-serving/tensorrt-llm/README.md)
| **Llama 3.1 405B**     | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)    | TensorRT-LLM   | Inference   | GKE          | [Link](./inference/a4/single-host-serving/tensorrt-llm/README.md)
| **Qwen 2.5 VL 7B**     | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)    | TensorRT-LLM   | Inference   | GKE          | [Link](./inference/a4/single-host-serving/tensorrt-llm/README.md)
| **Qwen 3 235B A22B**     | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)    | TensorRT-LLM   | Inference   | GKE          | [Link](./inference/a4/single-host-serving/tensorrt-llm/README.md)
| **Qwen 3 32B**     | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)    | TensorRT-LLM   | Inference   | GKE          | [Link](./inference/a4/single-host-serving/tensorrt-llm/README.md)

### Inference benchmarks A4X

| Models           | GPU Machine Type | Framework | Workload Type       | Orchestrator | Link to the recipe |
| ---------------- | ---------------- | --------- | ------------------- | ------------ | ------------------ |
| **DeepSeek R1 671B**     | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms)    | vLLM (v0.14.0rc1)  | Inference   | GKE          | [Link](./inference/a4x/single-host-serving/vllm/README.md)
| **Wan2.2 T2V A14B Diffusers**     | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms)    | SGLang (latest)  | Inference   | GKE          | [Link](./inference/a4x/single-host-serving/sglang/README.md)
| **Wan2.2 I2V A14B Diffusers**     | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms)    | SGLang (latest)  | Inference   | GKE          | [Link](./inference/a4x/single-host-serving/sglang/README.md)
| **DeepSeek R1 671B**     | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms)    | TensorRT-LLM (1.3.0rc5)   | Inference   | GKE          | [Link](./inference/a4x/single-host-serving/tensorrt-llm/README.md) <br> <br> [Link for Using Google Cloud Storage (GCS) as Storage Option]((./inference/a4x/single-host-serving/tensorrt-llm-gcs/README.md)) <br> <br> [Link for Using Lustre as Storage Option]((./inference/a4x/single-host-serving/tensorrt-llm-lustre/README.md))
| **Llama 3.1 405B**      | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms)    | TensorRT-LLM (1.3.0rc5)   | Inference   | GKE          | [Link](./inference/a4x/single-host-serving/tensorrt-llm/README.md)
| **Llama 3.1 70B**      | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms)    | TensorRT-LLM (1.3.0rc5)   | Inference   | GKE          | [Link](./inference/a4x/single-host-serving/tensorrt-llm/README.md)
| **Llama 3.1 8B**      | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms)    | TensorRT-LLM (1.3.0rc5)   | Inference   | GKE          | [Link](./inference/a4x/single-host-serving/tensorrt-llm/README.md)
| **Qwen 2.5 VL 7B**      | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms)    | TensorRT-LLM (1.3.0rc5)   | Inference   | GKE          | [Link](./inference/a4x/single-host-serving/tensorrt-llm/README.md)
| **Qwen 3 235B A22B**      | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms)    | TensorRT-LLM (1.3.0rc5)   | Inference   | GKE          | [Link](./inference/a4x/single-host-serving/tensorrt-llm/README.md)
| **Qwen 3 32B**      | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms)    | TensorRT-LLM (1.3.0rc5)   | Inference   | GKE          | [Link](./inference/a4x/single-host-serving/tensorrt-llm/README.md)
| **Qwen 3 4B**      | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms)    | TensorRT-LLM (1.3.0rc5)   | Inference   | GKE          | [Link](./inference/a4x/single-host-serving/tensorrt-llm/README.md)


### Inference benchmarks G4

| Models           | GPU Machine Type | Framework | Workload Type       | Orchestrator | Link to the recipe |
| ---------------- | ---------------- | --------- | ------------------- | ------------ | ------------------ |
| **Qwen3 8B**     | [G4 (NVIDIA RTX PRO 6000 Blackwell)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#g4-series)    | vLLM  | Inference   | GCE          | [Link](./inference/g4/qwen-8b/single-host-serving/vllm/README.md)
| **Qwen3 30B A3B**| [G4 (NVIDIA RTX PRO 6000 Blackwell)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#g4-series)    | TensorRT-LLM  | Inference   | GCE          | [Link](./inference/g4/qwen3_30b_a3b/single-host-serving/tensorrt-llm/README.md)
| **Qwen3 4B**     | [G4 (NVIDIA RTX PRO 6000 Blackwell)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#g4-series)    | TensorRT-LLM  | Inference   | GCE          | [Link](./inference/g4/qwen3_4b/single-host-serving/tensorrt-llm/README.md)
| **Qwen3 8B**     | [G4 (NVIDIA RTX PRO 6000 Blackwell)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#g4-series)    | TensorRT-LLM  | Inference   | GCE          | [Link](./inference/g4/qwen3_8b/single-host-serving/tensorrt-llm/README.md)
| **Qwen3 32B**    | [G4 (NVIDIA RTX PRO 6000 Blackwell)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#g4-series)    | TensorRT-LLM  | Inference   | GCE          | [Link](./inference/g4/qwen3_32b/single-host-serving/tensorrt-llm/README.md)
| **Qwen3 32B**     | [G4 (NVIDIA RTX PRO 6000 Blackwell)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#g4-series)    | vLLM  | Inference   | GCE          | [Link](./inference/g4/single-host-serving/vllm/README.md)
| **Llama3.1 70B** | [G4 (NVIDIA RTX PRO 6000 Blackwell)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#g4-series)    | TensorRT-LLM  | Inference   | GCE          | [Link](./inference/g4/llama3_1_70b/single-host-serving/tensorrt-llm/README.md)
| **DeepSeek R1**  | [G4 (NVIDIA RTX PRO 6000 Blackwell)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#g4-series)    | TensorRT-LLM  | Inference   | GCE          | [Link](./inference/g4/deepseek_r1/single-host-serving/tensorrt-llm/README.md)
| **Qwen3 235B**  | [G4 (NVIDIA RTX PRO 6000 Blackwell)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#g4-series)    | TensorRT-LLM  | Inference   | GCE          | [Link](./inference/g4/qwen3_235b/single-host-serving/tensorrt-llm/README.md)
| **Wan2.2 14B**  | [G4 (NVIDIA RTX PRO 6000 Blackwell)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#g4-series)    | SGLang  | Inference   | GCE          | [Link](./inference/g4/wan2.2/sglang/README.md)

### Checkpointing benchmarks

Models            | GPU Machine Type                                                                                          | Framework | Workload Type                                                   | Orchestrator | Link to the recipe
----------------- | --------------------------------------------------------------------------------------------------------- | --------- | --------------------------------------------------------------- | ------------ | ------------------
**Llama-3.1-70B** | [A3 Mega (NVIDIA H100)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-mega-vms) | NeMo      | Pre-training using Google Cloud Storage buckets for checkpoints | GKE          | [Link](./training/a3mega/llama3-1-70b/nemo-pretraining-gke-gcs/README.md)

### Goodput benchmarks

Models            | GPU Machine Type                                                                                          | Framework | Workload Type | Orchestrator | Link to the recipe
----------------- | --------------------------------------------------------------------------------------------------------- | --------- | ------------- | ------------ | ------------------
**Llama-3.1-70B** | [A3 Mega (NVIDIA H100)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-mega-vms) | NeMo      | Pre-training using  the Google Cloud Resiliency library  | GKE          | [Link](./training/a3mega/llama3-1-70b/nemo-pretraining-gke-resiliency/README.md)
**Llama-3.1-405B** | [A3 Ultra (NVIDIA H200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-ultra-vms) | NeMo      | Pre-training using  the Google Cloud Resiliency library  | GKE          | [Link](./training/a3ultra/llama3-1-405b/nemo-pretraining-gke-resiliency/README.md)
**Mixtral-8x7B** | [A3 Ultra (NVIDIA H200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-ultra-vms) | NeMo      | Pre-training using  the Google Cloud Resiliency library  | GKE          | [Link](./training/a3ultra/mixtral-8x7b/nemo-pretraining-gke-resiliency/README.md)

## Repository organization

- `./training`: this directory contains recipes with instructions to reproduce
  training benchmarks with GPUs.
- `./inference`: this directory contains recipes with instructions to reproduce
  inference benchmarks with GPUs.
- `./src`: this directory contains the shared dependencies required to run
  benchmarks, such as Docker images and Helm charts.
- `./docs`: this directory contains supporting documentation for explanations
  of benchmark methodologies or configurations.

## Repository scope

This repository provides the steps that you can use to reproduce a specific benchmark.
The actual performance measurements and the complete, confidential benchmark report are
not included.

## Methodology

Performance benchmarks measure the performance of various workloads on the platform.
These benchmarks are primarily used to validate performance with hardware suppliers and
to provide you with data for purchasing decisions.

## Maintenance policy

Benchmark data is considered a point-in-time measurement and completed benchmarks
are not repeated. We maintain and update the recipes in this repository on a best-effort
basis.

## Resources

For general guidance on how to get started using Compute products, refer to the official
documentation and tutorials:

- [Compute Engine overview](https://docs.cloud.google.com/compute/docs/overview)
- [Compute Engine samples](https://docs.cloud.google.com/compute/docs/samples)
- [Cloud GPU documentation](https://cloud.google.com/compute/docs/gpus)
- [AI Hypercomputer documentation](https://docs.cloud.google.com/ai-hypercomputer/docs)
- [Automated GPU environment deployment with Cluster Toolkit](https://docs.cloud.google.com/cluster-toolkit/docs/deploy/gke/gke-gpu-overview)

## Report issues
If you have questions or encounter problems with this repository, report them through
[GitHub Issues](https://github.com/AI-Hypercomputer/gpu-recipes/issues) or reach out to
your Google Cloud account team for assistance.

## Contributor notes

Note: This is not an officially supported Google product. This project is not eligible
for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).

