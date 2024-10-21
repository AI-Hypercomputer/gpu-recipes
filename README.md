
# Reproducible benchmark recipes for GPUs

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Welcome to the reproducible benchmark recipes repository for GPUs! This repository contains recipes for reproducing training and serving benchmarks for large machine learning models using GPUs on Google Cloud.

## Overview

1. **Identify your requirements:** Determine the model, GPU type, workload, framework, and orchestrator you are interested in.
2. **Select a recipe:** Based on your requirements use the [Benchmark support matrix](#benchmarks-support-matrix) to find a recipe that meets your needs.
3. Follow the recipe: each recipe will provide you with procedures to complete the following tasks:
   * Prepare your environment
   * Run the benchmark
   * Analyze the benchmarks results. This includes not just the results but detailed logs for further analysis

## Benchmarks support matrix

### Training benchmarks

| Models           | GPU Machine Type | Framework | Workload Type       | Orchestrator | Link to the recipe |
| ---------------- | ---------------- | --------- | ------------------- | ------------ | ------------------ |
| **GPT3-175B**       | [A3 Mega (NVIDIA H100)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-mega-vms)    | NeMo  | Pre-training   | GKE          | [Link](./training/a3mega/GPT3-175B/nemo-pretraining-gke/README.md)              |
| **Llama-3-70B**     | [A3 Mega (NVIDIA H100)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-mega-vms)    | NeMo  | Pre-training   | GKE          | [Link](./training/a3mega/llama-3-70b/nemo-pretraining-gke/README.md)            |

## Repository structure

* **[training/](./training)**: Contains recipes to reproduce training benchmarks with GPUs.
* **[src/](./src)**: Contains shared dependencies required to run benchmarks, such as Docker and Helm charts.
* **[docs/](./docs)**: Contains supporting documentation for the recipes, such as explanation of benchmark methodologies or configurations.

## Getting help

If you have any questions or if you found any problems with this repository, please report through GitHub issues.

## Disclaimer

This is not an officially supported Google product. The code in this repository is for demonstrative purposes only.

