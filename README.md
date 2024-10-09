
# Reproducible benchmark recipes for GPUs

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Welcome to the reproducible benchmark recipes repository for GPUs! This repository contains recipes for reproducing training and serving benchmarks for large machine learning models using GPUs on Google Cloud. 

## Getting Started

1. **Identify the model:** Determine the model, GPU type, workload, framework, and orchestrator you are interested in.
2. **Select a recipe:** Refer to the [Support Matrix](#benchmarks-support-matrix) and find the recipe that matches your needs.
3. **Prepare your environment:**  Each recipe has instructions on setting up environment to run the benchmark.
4. **Run the benchmark:** Follow the steps in the recipe to execute the benchmark.
5. **Analyze the results:**  At the end of the benchmark run, you'll get the resultant metrics and detailed logs for further analysis.

## Benchmarks Support matrix

### Training benchmarks

| Models           | GPU Machine Type | Framework | Workload Type       | Orchestrator | Link to the recipe |
| ---------------- | ---------------- | --------- | ------------------- | ------------ | ------------------ | 
| [GPT3-175B](./training/a3mega/GPT3-175B/nemo-pretraining-gke/README.md)        | [A3 Mega (NVIDIA H100)](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-mega-vms)    | NeMo  | Pre-training   | GKE          | [Link](./training/a3mega/GPT3-175B/nemo-pretraining-gke/README.md)              | 

## Repository structure 

* **[training/](./training)**: Contains recipes to reproduce training benchmarks with GPUs.
* **[src/](./src)**: Contains shared dependencies required to run benchmarks, such as docker files, helm charts.
* **[docs/](./docs)**: Contains documentation referred to in the recipes, such as explanation of benchmark methodologies or configurations.

## Getting help

If you have any questions or if you found any problems with this repository, please report through GitHub issues.

## Disclaimer

This is not an officially supported Google product. The code in this repository is for demonstrative purposes only.
