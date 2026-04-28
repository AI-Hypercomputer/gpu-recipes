# Cloud GPU performance benchmark recipes

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

The reproducible benchmark recipes repository for GPUs contains instructions 
necessary to reproduce specific training and serving performance measurements, 
which are part of a confidential benchmarking program. This repository focuses
on helping users reliably achieve performance metrics, such as throughput, that
demonstrate the combined hardware and software stack on GPUs.

**Note:** The content in this repository is not designed as a set of general-purpose 
code samples or tutorials for using Compute Engine-based products.

## Intended audience

This content is for you if you are a customer or partner who needs to:

- validate hardware performance with your suppliers.
- inform purchasing decisions using the benchmarking data.
- reproduce optimal performance scenarios before you customize workflows
  for your own requirements.
  
## How to use these recipes

To reproduce a benchmark, follow these steps:

1. **Identify your requirements:** determine the model, GPU type, workload, framework,
   and orchestrator you are interested in.
2. **Select a recipe:** based on your requirements use the
   [Benchmark support matrix](#benchmarks-support-matrix) to find a recipe that meets your needs.
3. **Follow the recipe:** each recipe will provide you with procedures to complete the following tasks:
   * prepare your environment.
   * run the benchmark.
   * analyze the benchmarks results. This includes not just the results but detailed logs for further analysis.

## Benchmarks support matrix

### Training benchmarks A3 Mega

Models            | GPU Machine Type                                                                                          | Framework | Workload Type | Orchestrator | Link to the recipe
----------------- | --------------------------------------------------------------------------------------------------------- | --------- | ------------- | ------------ | ------------------
**GPT3-175B**     | [A3 Mega (NVIDIA H100)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-mega-vms) | NeMo      | Pre-training  | GKE          | [Link](./training/a3mega/gpt3-175b/nemo-pretraining-gke/README.md)
**Llama-3-70B**   | [A3 Mega (NVIDIA H100)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-mega-vms) | NeMo      | Pre-training  | GKE          | [Link](./training/a3mega/llama3-70b/nemo-pretraining-gke/README.md)
**Llama-3.1-70B** | [A3 Mega (NVIDIA H100)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-mega-vms) | NeMo      | Pre-training  | GKE          | [Link](./training/a3mega/llama3-1-70b/nemo-pretraining-gke/README.md)
**Mixtral-8-7B**  | [A3 Mega (NVIDIA H100)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-mega-vms) | NeMo      | Pre-training  | GKE          | [Link](./training/a3mega/mixtral-8x7b/nemo-pretraining-gke/README.md)


### Training benchmarks A3 Ultra

Models             | GPU Machine Type                                                                                            | Framework | Workload Type | Orchestrator | Link to the recipe
------------------ | ----------------------------------------------------------------------------------------------------------- | --------- | ------------- | ------------ | ------------------
**Llama-3.1-70B**  | [A3 Ultra (NVIDIA H200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-ultra-vms) | MaxText   | Pre-training  | GKE          | [Link](./training/a3ultra/llama3-1-70b/maxtext-pretraining-gke/README.md)
**Llama-3.1-70B**  | [A3 Ultra (NVIDIA H200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-ultra-vms) | NeMo      | Pre-training  | GKE          | [Link](./training/a3ultra/llama3-1-70b/nemo-pretraining-gke/README.md)
**Llama-3.1-405B** | [A3 Ultra (NVIDIA H200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-ultra-vms) | MaxText   | Pre-training  | GKE          | [Link](./training/a3ultra/llama3-1-405b/maxtext-pretraining-gke/README.md)
**Llama-3.1-405B** | [A3 Ultra (NVIDIA H200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-ultra-vms) | NeMo.     | Pre-training  | GKE          | [Link](./training/a3ultra/llama3-1-405b/nemo-pretraining-gke/README.md)
**Mixtral-8-7B**   | [A3 Ultra (NVIDIA H200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-ultra-vms) | NeMo      | Pre-training  | GKE          | [Link](./training/a3ultra/mixtral-8x7b/nemo-pretraining-gke/README.md)

### Training benchmarks A4

Models             | GPU Machine Type                                                                                     | Framework / Library | Workload Type | Orchestrator | Link to the recipe
------------------ | ---------------------------------------------------------------------------------------------------- | --------- | ------------- | ------------ | ------------------
**Llama-3.1-70B** | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)      | MaxText   | Pre-training  | GKE          | [Link](./training/a4/llama3-1-70b/maxtext-pretraining-gke/README.md)
**Llama-3.1-70B** | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)      | NeMo      | Pre-training  | GKE          | [Link](./training/a4/llama3-1-70b/nemo-pretraining-gke)
**Llama-3.1-405B** | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)      | MaxText   | Pre-training  | GKE          | [Link](./training/a4/llama3-1-405b/maxtext-pretraining-gke/README.md)
**Llama-3.1-405B** | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)      | NeMo      | Pre-training  | GKE          | [Link](./training/a4/llama3-1-405b/nemo-pretraining-gke/README.md)
**Mixtral-8-7B**   | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)      | NeMo      | Pre-training  | GKE          | [Link](./training/a4/mixtral-8x7b/nemo-pretraining-gke/README.md)
**PaliGemma2**     | [A4 (NVIDIA B200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms)      | Hugging Face Accelerate | Finetuning | GKE     | [Link](./training/a4/paligemma2/README.md)

### Training benchmarks A4X

Models             | GPU Machine Type                                                                                     | Framework | Workload Type | Orchestrator | Link to the recipe
------------------ | ---------------------------------------------------------------------------------------------------- | --------- | ------------- | ------------ | ------------------
**Llama-3.1-8B** | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms)      | NeMo   | Pre-training  | GKE          | [Link](./training/a4x/llama3-1-8b/nemo-pretraining-gke/)
**Llama-3.1-70B** | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms)      | NeMo      | Pre-training  | GKE          | [Link](./training/a4x/llama3-1-70b/nemo-pretraining-gke/)
**Llama-3.1-405B** | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms)      | NeMo      | Pre-training  | GKE          | [Link](./training/a4x/llama3-1-405b/nemo-pretraining-gke/)
**Nemotron-4-340B** | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms)      | NeMo      | Pre-training  | GKE          | [Link](./training/a4x/nemotron4-340B/nemo-pretraining-gke/)
**Wan-2.1-14B** | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms)      | NeMo      | Pre-training  | GKE          | [Link](./training/a4x/wan2-1-14b/nemo-pretraining-gke/)
**Wan-2.1-14B** | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms)      | NeMo      | Pre-training  | Slurm          | [Link](./training/a4x/wan2-1-14b/nemo-pretraining-slurm/)

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
| **DeepSeek R1 671B**     | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms)    | vLLM  | Inference   | GKE          | [Link](./inference/a4x/single-host-serving/vllm/README.md)
| **Wan2.2 T2V A14B Diffusers**     | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms)    | SGLang  | Inference   | GKE          | [Link](./inference/a4x/single-host-serving/sglang/README.md)
| **Wan2.2 I2V A14B Diffusers**     | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms)    | SGLang  | Inference   | GKE          | [Link](./inference/a4x/single-host-serving/sglang/README.md)
| **DeepSeek R1 671B**     | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms)    | TensorRT-LLM   | Inference   | GKE          | [Link](./inference/a4x/single-host-serving/tensorrt-llm/README.md) <br> <br> [Link for Using Google Cloud Storage (GCS) as Storage Option]((./inference/a4x/single-host-serving/tensorrt-llm-gcs/README.md)) <br> <br> [Link for Using Lustre as Storage Option]((./inference/a4x/single-host-serving/tensorrt-llm-lustre/README.md))
| **Llama 3.1 405B**      | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms)    | TensorRT-LLM   | Inference   | GKE          | [Link](./inference/a4x/single-host-serving/tensorrt-llm/README.md)
| **Llama 3.1 70B**      | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms)    | TensorRT-LLM   | Inference   | GKE          | [Link](./inference/a4x/single-host-serving/tensorrt-llm/README.md)
| **Llama 3.1 8B**      | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms)    | TensorRT-LLM   | Inference   | GKE          | [Link](./inference/a4x/single-host-serving/tensorrt-llm/README.md)
| **Qwen 2.5 VL 7B**      | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms)    | TensorRT-LLM   | Inference   | GKE          | [Link](./inference/a4x/single-host-serving/tensorrt-llm/README.md)
| **Qwen 3 235B A22B**      | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms)    | TensorRT-LLM   | Inference   | GKE          | [Link](./inference/a4x/single-host-serving/tensorrt-llm/README.md)
| **Qwen 3 32B**      | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms)    | TensorRT-LLM   | Inference   | GKE          | [Link](./inference/a4x/single-host-serving/tensorrt-llm/README.md)
| **Qwen 3 4B**      | [A4X (NVIDIA GB200)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4x-vms)    | TensorRT-LLM   | Inference   | GKE          | [Link](./inference/a4x/single-host-serving/tensorrt-llm/README.md)


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

- `./training`: use these recipes to reproduce training benchmarks with GPUs.
- `./inference`: use these recipes to reproduce inference benchmarks with GPUs.
- `./src`: shared dependencies required to run benchmarks, such as Docker
  images and Helm charts.
- `./docs`: supporting documentation for explanations of benchmark methodologies
  or configurations.

## Repository scope

This repository provides the steps that you can use to reproduce a specific
benchmark. The actual performance measurements or the complete, confidential
benchmark report are not included.

## Methodology

Performance benchmarks measure the performance of various workloads on the 
platform. These benchmarks are primarily used to validate performance with
hardware suppliers and to provide you with data for purchasing decisions.

### Maintenance policy

Benchmark data is considered a point-in-time measurement and completed
benchmarks are not repeated. As such, there is no intent to maintain or
update the reproducibility steps provided in this repository.

## Resources

If you are looking for general guidance on how to get started using
Compute products, refer to the official documentation and tutorials:

- [Official Compute Engine tutorials and samples](https://docs.cloud.google.com/compute/docs/overview)
- [Cloud TPU documentation](https://docs.cloud.google.com/tpu/docs)
- [AI Hypercomputer documentation](https://docs.cloud.google.com/ai-hypercomputer/docs)

## Getting help

If you have any questions or if you encounter any problems with this repository,
report them through https://github.com/AI-Hypercomputer/tpu-recipes/issues.

## Contributor notes

Note: This is not an officially supported Google product. This project is not
eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).

