# Qwen3 235B A22B on 16-node A4 (B200)

This recipe runs pretraining for Qwen3 235B A22B on 16 nodes of A4 (B200) GPUs using Megatron-Bridge on GKE.

## Prerequisites
Please replace the placeholder `YOUR_HF_TOKEN` in `launcher.sh` with your huggingface token before launching. You may also need to update other configurations such as your GCS buckets.

## Launch
You can launch this job using Helm or kubectl as per the standard GKE workflow for Megatron-Bridge workloads.
