# Configuring the environment for running benchmark recipes on a GKE Cluster with A4 Node Pools

This [guide](https://cloud.google.com/ai-hypercomputer/docs/create/gke-ai-hypercompute) outlines the steps to configure the environment required to run benchmark recipes on a [Google Kubernetes Engine (GKE) cluster](https://cloud.google.com/kubernetes-engine/docs/concepts/kubernetes-engine-overview) with [A4](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-vms) node pools.

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

3. Requested enough GPU quotas. Each `a4-highgpu-8g` machine has 8 B200 GPUs attached.
  1. To view quotas, see [View the quotas for your project](/docs/quotas/view-manage).
     In the Filter field, select **Dimensions(e.g location)** and
     specify [`gpu_family:NVIDIA_B200`](https://cloud.google.com/compute/resource-usage#gpu_quota).
  1. If you don't have enough quota, [request a higher quota](https://cloud.google.com/docs/quotas/view-manage#requesting_higher_quota).

## Reserve capacity

To ensure that your workloads have the A4 GPU resources required for these
instructions, you can create a [future reservation request](https://cloud.google.com/compute/docs/instances/future-reservations-overview).
With this request, you can reserve blocks of capacity for a defined duration in the
future. At that date and time in the future, Compute Engine automatically
provisions the blocks of capacity by creating on-demand reservations that you
can immediately consume by provisioning node pools for this cluster.

Additionally, as your reserved capacity might span multiple
[blocks](https://cloud.google.com/ai-hypercomputer/docs/terminology#block), we recommend that you create
GKE nodes on a specific block within your reservation.

Do the following steps to request capacity and gather the required information
to create nodes on a specific block within your reservation:

1. [Request capacity](https://cloud.google.com/ai-hypercomputer/docs/request-capacity).

1. To get the name of the blocks that are available for your reservation,
   run the following command:

   ```sh
   gcloud beta compute reservations blocks list RESERVATION_NAME \
       --zone=ZONE --format "value(name)"
   ```
   Replace the following:

   * `RESERVATION_NAME`: the name of your reservation.
   * `ZONE`: the compute zone of your reservation.

   The output has the following format: `BLOCK_NAME`.
   For example the output might be similar to the following: `example-res1-block-0001`.

1. If you want to target specific blocks within a reservation when
   provisioning GKE node pools, you must specify the full reference
   to your block as follows:

    ```none
   RESERVATION_NAME/reservationBlocks/BLOCK_NAME
   ```

   For example, using the example output in the preceding step, the full path is as follows: `example-res1/reservationBlocks/example-res1-block-0001`

## The environment

The environment comprises of the following components:

- Client workstation: this is used to prepare, submit, and monitor ML workloads.
- [Google Cloud Storage (GCS) Bucket](https://cloud.google.com/storage/docs): used for storing
  datasets and logs.
- [Artifact Registry](https://cloud.google.com/artifact-registry/docs/overview): serves as a
  private container registry for storing and managing Docker images used in the deployment.
- [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine/docs/concepts/kubernetes-engine-overview)
  Cluster with A4 Node Pools: provides a managed Kubernetes environment to run benchmark
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

Add IAM binding to allow workloads authenticated via a workload identity (with the default service account) to access Cloud Storage objects.

   ```bash
   PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
   gcloud storage buckets add-iam-policy-binding gs://<BUCKET_NAME> \
   --role=roles/storage.objectUser \
   --member=principal://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$PROJECT_ID.svc.id.goog/subject/ns/default/sa/default \
   --condition=None
   ```

Replace the following:

- `BUCKET_NAME`: the name of your bucket created in the previous step

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


## Create a GKE Cluster with A4 Node Pools

Follow [this guide]() for
detailed instructions to create a GKE cluster with A4 node pools and required GPU driver versions.

The documentation uses [ Cluster Toolkit](https://cloud.google.com/cluster-toolkit/docs/overview) to create your GKE cluster quickly while incorporating best practices:

- Creation of the necessary VPC networks and subnets.
- Creation of a GKE cluster with multi-networking enabled.
- Creation of an A4 node pool with NVIDIA B200 GPUs.
- Installation of the required components for GPUDirect-RDMA and NCCL plugin.

1.  [Launch Cloud Shell](https://cloud.google.com/shell/docs/launching-cloud-shell). You can use a
    different environment; however, we recommend Cloud Shell because the
    dependencies are already pre-installed for Cluster Toolkit. If you
    don't want to use Cloud Shell, follow the instructions to [install
    dependencies](/cluster-toolkit/docs/setup/install-dependencies) to prepare a
    different environment.

1. Clone the Cluster Toolkit from the git repository:

   ```sh
   cd ~
   git clone https://github.com/GoogleCloudPlatform/cluster-toolkit.git
   ```
1. Install the Cluster Toolkit:

   ```sh
   cd cluster-toolkit && git checkout main && make
   ```

1. Create a Cloud Storage bucket to store the state of the Terraform
   deployment:

   ```sh
   gcloud storage buckets create gs://TF_STATE_BUCKET_NAME \
    --default-storage-class=STANDARD \
    --location=COMPUTE_REGION \
    --uniform-bucket-level-access
   gcloud storage buckets update gs://TF_STATE_BUCKET_NAME --versioning
   ```

   Replace the following variables:

    * `TF_STATE_BUCKET_NAME`: the name of the new Cloud Storage bucket.
    * `COMPUTE_REGION`: the compute region where you want to store the state of the Terraform deployment.

1. In the [`examples/gke-a4-highgpu/gke-a4-highgpu-deployment.yaml`](https://github.com/GoogleCloudPlatform/cluster-toolkit/blob/main/examples/gke-a4-highgpu/gke-a4-highgpu-deployment.yaml)
   file, replace the following variables in the `terraform_backend_defaults` and
   `vars` sections to match the specific values for your deployment:

   * `BUCKET_NAME`: the name of the Cloud Storage bucket you created in the
      previous step to store the state of Terraform deployment.
   * `PROJECT_ID`: your Google Cloud project ID.
   * `COMPUTE_REGION`: the compute region for the cluster.
   * `COMPUTE_ZONE`: the compute zone for the node pool of A4 machines.
   * `IP_ADDRESS/SUFFIX`: The IP address range that you want to allow to
      connect with the cluster. This CIDR block must include the IP address of
      the machine to call Terraform.
   * `RESERVATION_NAME`: the name of your reservation.
   * `BLOCK_NAME`: the name of a specific block within the reservation.
   * `NODE_COUNT`: the number of A4 nodes in your cluster.

  To modify advanced settings, edit
  `examples/gke-a4-highgpu/gke-a4-highgpu.yaml`.

1. Generate [Application Default Credentials (ADC)](/docs/authentication/provide-credentials-adc#google-idp)
   to provide access to Terraform.

1.  Deploy the blueprint to provision the GKE infrastructure
    using A4 machine types:

   ```sh
   cd ~/cluster-toolkit
   ./gcluster deploy -d \
    examples/gke-a4-highgpu/gke-a4-highgpu-deployment.yaml \
    examples/gke-a4-highgpu/gke-a4-highgpu.yaml
   ```

## Clean up {:#clean-up}

To avoid recurring charges for the resources used on this page, clean up the
resources provisioned by Cluster Toolkit, including the
VPC networks and GKE cluster:

   ```sh
   ./gcluster destroy gke-a4-high/
   ```


## What's next

Once you have set up your GKE cluster with A4 node pools, you can proceed to deploy and
run your [benchmark recipes](../README.md#benchmarks-support-matrix).

## Get Help

If you encounter any issues or have questions about this setup, use one of the following
resources:

- Consult the [official GKE documentation](https://cloud.google.com/kubernetes-engine/docs).
- Check the issues section of this repository for known problems and solutions.
- Reach out to Google Cloud support.
