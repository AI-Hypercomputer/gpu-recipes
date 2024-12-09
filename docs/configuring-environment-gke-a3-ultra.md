# Configuring the environment for running benchmark recipes on a GKE Cluster with A3 Ultra Node Pools

This [guide](https://cloud.google.com/ai-hypercomputer/docs/create/gke-ai-hypercompute) outlines the steps to configure the environment required to run benchmark recipes on a [Google Kubernetes Engine (GKE) cluster](https://cloud.google.com/kubernetes-engine/docs/concepts/kubernetes-engine-overview) with [A3 Ultra](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-vms) node pools.

## Prerequisites

Before you begin, ensure you have completed the following:

1. Create a Google Cloud project with billing enabled.

   a. To create a project, see [Creating and managing projects](https://cloud.google.com/resource-manager/docs/creating-managing-projects).
   b. To enable billing, see [Verify the billing status of your projects](https://cloud.google.com/billing/docs/how-to/verify-billing-enabled).

2. Enabled the following APIs:

   - [Service Usage API](https://console.cloud.google.com/apis/library/serviceusage.googleapis.com).
   - [Google Kubernetes Engine API](https://console.cloud.google.com/flows/enableapi?apiid=container.googleapis.com).
   - [Cloud Storage API](https://console.cloud.google.com/flows/enableapi?apiid=storage.googleapis.com).
   - [Artifact Registry API](https://console.cloud.google.com/flows/enableapi?apiid=artifactregistry.googleapis.com).

3. Requested enough GPU quotas. Each `a3-ultragpu-8g` machine has 8 H200 GPUs attached.
  1. To view quotas, see [View the quotas for your project](/docs/quotas/view-manage).
     In the Filter field, select **Dimensions(e.g location)** and 
     specify [`gpu_family:NVIDIA_H200`](https://cloud.google.com/compute/resource-usage#gpu_quota).
  1. If you don't have enough quota, [request a higher quota](https://cloud.google.com/docs/quotas/view-manage#requesting_higher_quota).

## The environment

The environment comprises of the following components:

- Client workstation: this is used to prepare, submit, and monitor ML workloads.
- [Google Cloud Storage (GCS) Bucket](https://cloud.google.com/storage/docs): used for storing 
  datasets and logs.
- [Artifact Registry](https://cloud.google.com/artifact-registry/docs/overview): serves as a
  private container registry for storing and managing Docker images used in the deployment.
- [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine/docs/concepts/kubernetes-engine-overview)
  Cluster with A3 Ultra Node Pools: provides a managed Kubernetes environment to run benchmark
  recipes.

## Set up the client workstation

You have two options, you can use either a local machine or Google Cloud Shell.

### Google Cloud Shell

We recommend using [Google Cloud Shell](https://cloud.google.com/shell/docs) as it
comes with all necessary components pre-installed.

### Local client
If you prefer to use your local machine, ensure your local machine has the following
components installed. 

1. Google Cloud SDK. To install, see 
   [Install the gcloud CLI](https://cloud.google.com/sdk/docs/install).
2. kubectl. To install, see the
   [kuberenetes documentation](https://kubernetes.io/docs/tasks/tools/#kubectl).
3. Helm. To install, see the [Helm documentation](https://helm.sh/docs/intro/quickstart/).
4. Docker. To install, see the [Docker documentation](https://docs.docker.com/engine/install/).


## Set up a Google Cloud Storage bucket

```bash
gcloud storage buckets create gs://<BUCKET_NAME> --location=<BUCKET_LOCATION> --no-public-access-prevention
```

Replace the following:

- `BUCKET_NAME`: the name of your bucket. The name must comply with the
   [Cloud Storage bucket naming conventions](https://cloud.google.com/storage/docs/buckets#naming).
- `BUCKET_LOCATION`: the location of your bucket. The bucket must be located in
   the same region as the GKE cluster.

## Set up an Artifact Registry

- If you use Cloud KMS for repository encryption, create your artifact registry by using the
  [instructions here](https://cloud.google.com/artifact-registry/docs/repositories/create-repos#create-repo-gcloud-docker).
- If you don't use Cloud KMS, you can create your repository by using the following command:

  ```bash
    gcloud artifacts repositories create <REPOSITORY> \
        --repository-format=docker \
        --location=<LOCATION> \
        --description="<DESCRIPTION>" \
  ```
  Replace the following:
  
  - `REPOSITORY`: the name of the repository. For each repository location in a project,
     repository names must be unique.
  - `LOCATION`: the regional or multi-regional location for the repository. You can omit this
     flag if you set a default region. 
  - `DESCRIPTION`: a description of the repository. Don't include sensitive data because
     repository descriptions are not encrypted.


## Create a GKE Cluster with A3 Ultra Node Pools

Follow [this guide](https://cloud.google.com/ai-hypercomputer/docs/create/gke-ai-hypercompute) for 
detailed instructions to create a GKE cluster with A3 Ultra node pools, GPUDirect-RDMA and required GPU driver versions. 

The documentation uses [ Cluster Toolkit](https://cloud.google.com/cluster-toolkit/docs/overview) to create your GKE cluster quickly while incorporating best practices:

- Creation of the necessary VPC networks and subnets.
- Creation of a GKE cluster with multi-networking enabled.
- Creation of an A3 Ultra node pool with NVIDIA H200 GPUs.
- Installation of the required components for GPUDirect-RDMA and NCCL plugin.

## What's next

Once you have set up your GKE cluster with A3 Ultra node pools, you can proceed to deploy and
run your [benchmark recipes](../README.md#benchmarks-support-matrix). 

## Get Help

If you encounter any issues or have questions about this setup, use one of the following 
resources:

- Consult the [official GKE documentation](https://cloud.google.com/kubernetes-engine/docs).
- Check the issues section of this repository for known problems and solutions.
- Reach out to Google Cloud support.

