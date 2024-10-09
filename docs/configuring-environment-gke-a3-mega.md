# Configuring Environment for Running Benchmark Recipes on a GKE Cluster with A3 Mega Node Pools

This guide outlines the steps to configure the environment required to run benchmark recipes on a [Google Kubernetes Engine (GKE) cluster](https://cloud.google.com/kubernetes-engine/docs/concepts/kubernetes-engine-overview) with [A3 Mega](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-mega-vms) node pools.

## Prerequisites

Before you begin, ensure you have:

- A Google Cloud project with billing enabled and required APIs enabled.
  1. [Select or create a Google Cloud project](https://console.cloud.google.com/cloud-resource-manager).
  2. [Make sure that billing is enabled for your project](https://cloud.google.com/billing/docs/how-to/modify-project).
  3. [Enable the Service Usage API](https://console.cloud.google.com/apis/library/serviceusage.googleapis.com).
  4. [Enable the Google Kubernetes Engine API](https://console.cloud.google.com/flows/enableapi?apiid=container.googleapis.com).
  5. [Enable the Cloud Storage API](https://console.cloud.google.com/flows/enableapi?apiid=storage.googleapis.com).
  6. [Enable the Artifact Registry API](https://console.cloud.google.com/flows/enableapi?apiid=artifactregistry.googleapis.com).

- Ensure that you have enough GPU quotas. Each `a3-megagpu-8g` machine has 8 H100 80GB GPUs attached.
  1. To view quotas, see [View the quotas for your project](/docs/quotas/view-manage).
     In the Filter field, select **Dimensions(e.g location)** and specify [`gpu_family:NVIDIA_H100_MEGA`](https://cloud.google.com/compute/resource-usage#gpu_quota).
  1. If you don't have enough quota, [request a higher quota](https://cloud.google.com/docs/quotas/view-manage#requesting_higher_quota).

## Environment Overview

The environment comprises the following components:

- Client workstation: A user sets up the development environment to prepare, submit, and monitor ML workloads.
- [Google Cloud Storage (GCS) Bucket](https://cloud.google.com/storage/docs): Used for storing datasets and logs.
- [Artifact Registry](https://cloud.google.com/artifact-registry/docs/overview): Serves as a private container registry for storing and managing Docker images used in the deployment.
- [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine/docs/concepts/kubernetes-engine-overview) Cluster with A3 Mega Node Pools: Provides a managed Kubernetes environment to run benchmark recipes.

## Setting up the Client Workstation

We recommend using [Google Cloud Shell](https://cloud.google.com/shell/docs) as it comes with all necessary components pre-installed. If you prefer to use your local machine, ensure you have the following components installed. 

1. Google Cloud SDK
2. kubectl
3. Helm
4. Docker

To install these components locally, follow these steps:

### 1. Install Google Cloud SDK

Follow the installation instructions for your operating system:
[https://cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install)

### 2. Install kubectl

You can install kubectl by following [the documentation](https://kubernetes.io/docs/tasks/tools/#kubectl):

### 3. Install Helm

Install Helm by following [the documentation](https://helm.sh/docs/intro/quickstart/):

### 4. Install Docker

Install Docker by following [this document](https://docs.docker.com/engine/install/).

## Setting up a Google Cloud Storage bucket

```bash
gcloud storage buckets create gs://BUCKET_NAME --location=BUCKET_LOCATION --no-public-access-prevention
```

Replace the following:
- `BUCKET_NAME` is the name of your bucket. The name must comply with the [Cloud Storage bucket naming conventions](https://cloud.google.com/storage/docs/buckets#naming).
- `BUCKET_LOCATION` is the location of your bucket. The bucket must be located in  the same region as the GKE cluster.

## Setting up an Artifact Registry

```bash
  gcloud artifacts repositories create REPOSITORY \
      --repository-format=docker \
      --location=LOCATION \
      --description="DESCRIPTION" \
```
Replace the following:
- `REPOSITORY`: the name of the repository. For each repository location in a project, repository names must be unique.
- `LOCATION`: the regional or multi-regional location for the repository. You can omit this flag if you set a default. 
- `DESCRIPTION`: a description of the repository. Do not include sensitive data, since repository descriptions are not encrypted.

If you use Cloud KMS for repository encryption, please use the command in [this documentation](https://cloud.google.com/artifact-registry/docs/repositories/create-repos#create-repo-gcloud-docker). 

## Setting up a GKE Cluster with A3 Mega Node Pools

To set up a GKE cluster with A3 Mega node pools, GPUDirect-TCPXO, gVNIC, and multi-networking, please follow the official Google Cloud documentation:

[Configure GPUDirect TCPX on GKE](https://cloud.google.com/kubernetes-engine/docs/how-to/gpu-bandwidth-gpudirect-tcpx)

This documentation provides detailed instructions to:
- Create the necessary VPC networks and subnets.
- Create a GKE cluster with multi-networking enabled.
- Create an A3 Mega node pool with NVIDIA H100 GPUs.
- Install the required components for GPUDirect and NCCL plugin.

## Next Steps

Once you have set up your GKE cluster with A3 Mega node pools, you can proceed to deploy and run your [benchmark recipes](../README.md#benchmarks-support-matrix). Refer to the specific instructions for each benchmark in their respective directories within this repository.

## Getting Help

If you encounter any issues or have questions about this setup, please:
- Consult the [official GKE documentation](https://cloud.google.com/kubernetes-engine/docs).
- Reach out to Google Cloud support.
- Check the issues section of this repository for known problems and solutions.