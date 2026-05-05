# Qwen3-32B Inference Recipe on A3 Ultra

This recipe demonstrates how to run inference for the Qwen3-32B model on A3 Ultra (H200) GPUs using TensorRT-LLM.

## Prerequisites

- Access to A3 Ultra hardware
- HuggingFace API token (for downloading the model)

## Setup

**Important:** Before launching, you must update the `launcher.sh` script to include your HuggingFace API token. Look for the following line and replace the placeholder:

```bash
export HF_TOKEN="<YOUR_HF_TOKEN_HERE>"
```

## Workload Configuration
- **Model:** Qwen/Qwen3-32B
- **Precision:** fp8
- **Input Sequence Length (ISL):** 1024
- **Output Sequence Length (OSL):** 8192
- **Tensor Parallelism (TP):** 4
- **Pipeline Parallelism (PP):** 1
