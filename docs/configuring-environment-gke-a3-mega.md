# Configuring the environment for running benchmark recipes on a GKE Cluster with A3 Mega Node Pools

This guide outlines the steps to configure the environment required to run benchmark recipes
on a [Google Kubernetes Engine (GKE) cluster](https://cloud.google.com/kubernetes-engine/docs/concepts/kubernetes-engine-overview)
with [A3 Mega](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-mega-vms) node pools.

## Prerequisites

Before you begin, complete the following:

1. Create a Google Cloud project with billing enabled.

   a. To create a project, see [Creating and managing projects](https://cloud.google.com/resource-manager/docs/creating-managing-projects).
   b. To enable billing, see [Verify the billing status of your projects](https://cloud.google.com/billing/docs/how-to/verify-billing-enabled).

2. Enable the following APIs:

   - [Service Usage API](https://console.cloud.google.com/apis/library/serviceusage.googleapis.com).
   - [Google Kubernetes Engine API](https://console.cloud.google.com/flows/enableapi?apiid=container.googleapis.com).
   - [Cloud Storage API](https://console.cloud.google.com/flows/enableapi?apiid=storage.googleapis.com).
   - [Artifact Registry API](https://console.cloud.google.com/flows/enableapi?apiid=artifactregistry.googleapis.com).

3. Request enough GPU quotas. Each `a3-megagpu-8g` machine has 8 H100 80GB GPUs attached.
  1. To view quotas, see [View the quotas for your project](/docs/quotas/view-manage).
     In the Filter field, select **Dimensions(e.g., location)** and
     specify [`gpu_family:NVIDIA_H100_MEGA`](https://cloud.google.com/compute/resource-usage#gpu_quota).
  1. If you don't have enough quota, [request a higher quota](https://cloud.google.com/docs/quotas/view-manage#requesting_higher_quota).

## Environment

The environment comprises of the following components:

- Client workstation: used to prepare, submit, and monitor ML workloads.
- [Google Cloud Storage (GCS) Bucket](https://cloud.google.com/storage/docs): used for logs.
- [Artifact Registry](https://cloud.google.com/artifact-registry/docs/overview): serves as a
  private container registry for storing and managing Docker images used in the deployment.
- [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine/docs/concepts/kubernetes-engine-overview)
  Cluster with A3 Mega Node Pools: provides a managed Kubernetes environment to run benchmark
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


## Set up a Google Cloud Storage bucket for logging

The recipes use a Google Cloud Storage bucket to maintain workload logs.

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


## Create a GKE Cluster with A3 Mega Node Pools

Follow [this guide](https://cloud.google.com/cluster-toolkit/docs/deploy/deploy-a3-mega-gke-cluster) for
detailed instructions to create a GKE cluster with A3 Mega node pools.

The documentation uses [Cluster Toolkit](https://cloud.google.com/cluster-toolkit/docs/overview) to create your GKE cluster quickly while incorporating best practices:

- Creation of the necessary VPC networks and subnets.
- Creation of a GKE cluster with multi-networking enabled.
- Creation of an A3 Mega node pool with NVIDIA H100 GPUs.
- Installation of the required components for the GPUDirect-TCPXO networking stack.

## (Optional) Create Google Cloud Storage buckets with hierarchical namespace.

Some recipes require Google Cloud Storage buckets with [hierarchical namespace](https://cloud.google.com/storage/docs/hns-overview#:~:text=Buckets%20with%20hierarchical%20namespace%20enabled%20offer%20granular%20access%20control%20through,exist%20without%20the%20corresponding%20folder.) enabled to manage data and checkpoints.

You can create a bucket with hierarchical namespace enabled using the following command.

```bash
gcloud storage buckets create gs://<BUCKET_NAME> --location=<BUCKET_LOCATION> \
--no-public-access-prevention \
--uniform-bucket-level-access \
--enable-hierarchical-namespace
```

Replace the following:

- `BUCKET_NAME`: the name of your bucket. The name must comply with the
   [Cloud Storage bucket naming conventions](https://cloud.google.com/storage/docs/buckets#naming).
- `BUCKET_LOCATION`: the location of your bucket. The bucket must be located in
   the same region as the GKE cluster.

## (Optional) Create a Parallestore instance

Some recipes require a [Google Cloud Parallelstore](https://cloud.google.com/parallelstore/docs/overview) instance for data and checkpointing. You can create and configure the instance using the following steps.

### Required IAM permissions

You must be granted the following roles:
- `roles/parallelstore.admin`
- `roles/compute.networkAdmin` or `roles/roles/servicenetworking.networksAdmin`

### Enable the Parallestore API

```
gcloud services enable parallelstore.googleapis.com --project=<PROJECT_ID>
```

Replace the following:
- `PROJECT_ID`: the project ID of your project

### Configure a VPC network

Parallelstore runs within a Virtual Private Cloud (VPC), which provides networking functionality to Compute Engine virtual machine (VM) instances, Google Kubernetes Engine (GKE) clusters, and serverless workloads.

You must use the same VPC network when creating the Parallelstore instance that you used for your Google Kubernetes Engine cluster.

You must also configure private services access within this VPC.

#### Enable service networking

```
gcloud services enable servicenetworking.googleapis.com
```

#### Get the VPC name of your cluster

```
NETWORK_NAME=$(
gcloud container clusters describe <CLUSTER_NAME> \
--location <CLUSTER_REGION> \
--format="value(network)"
)
```

Replace the following:
- `CLUSTER_NAME`: the name of your GKE cluster
- `CLUSTER_REGION`: the region of you GKE cluster


#### Create an IP range
Private services access requires a prefix-length of at least /24 (256 addresses). Parallelstore reserves 64 addresses per instance, which means that you can re-use this IP range with other services or other Parallelstore instances if needed.

```
IP_RANGE_NAME=<IP_RANGE_NAME>

gcloud compute addresses create $IP_RANGE_NAME \
  --global \
  --purpose=VPC_PEERING \
  --prefix-length=24 \
  --description="Parallelstore VPC Peering" \
  --network=$NETWORK_NAME
```

Replace the following:
- `IP_RANGE_NAME` - the name of the IP range. You can use any name that hasn't been already used.

#### Get the CIDR range associated with the range you created in the previous step.

```
CIDR_RANGE=$(
  gcloud compute addresses describe $IP_RANGE_NAME \
  --global \
  --format="value[separator=/](address, prefixLength)"
)
```


#### Create a firewall rule to allow TCP traffic from the IP range you created.

```
FIREWALL_RULE_NAME=<FIREWALL_RULE_NAME>

gcloud compute firewall-rules create $FIREWALL_RULE_NAME \
  --allow=tcp \
  --network=$NETWORK_NAME \
  --source-ranges=$CIDR_RANGE
```

Replace the following:
- `FIREWALL_RULE_NAME`: the name of the firewall rule. You can use any name that hasn't been already used.

### Connect the peering

```
gcloud services vpc-peerings connect \
  --network=$NETWORK_NAME \
  --ranges=$IP_RANGE_NAME \
  --service=servicenetworking.googleapis.com
```


### Create Parallestore instance

After the VPC network used by your GKE cluster is configured, you can create a Parallestore instance.

When creating a Parallelstore instance, you must define the following properties:

- The instance's name.
- The storage capacity. Capacity can range from 12TiB (tebibytes) to 100TiB, in multiples of 4. For example, 16TiB, 20TiB, 24TiB.
- The location.
- File and directory striping settings.

Your instance must be located in the same zone as your cluster's A3 Mega node pool. The storage capacity and file and directory striping settings depend on the recipe and will be specified in the recipe's instructions.



```
gcloud beta parallelstore instances create <INSTANCE_ID> \
  --capacity-gib=<CAPACITY_GIB> \
  --location=<LOCATION> \
  --network=<NETWORK_NAME> \
  --project=<PROJECT_ID> \
  --directory-stripe-level=<DIRECTORY_STRIPE_LEVEL> \
  --file-stripe-level=<FILE_STRIPE_LEVEL>
```

Replace the following with:
- `PROJECT_ID` - the name of your project.
- `LOCATION` - the zone where your cluster's A3 Mega node pool is located.
- `NETWORK_NAME` - the name of your cluster VPC.
- `CAPACITY_GIB` - the storage capacity of the instance in Gibibytes (GiB). Allowed values are from 12000 to 100000, in multiples of 4000.
- `DIRECTORY_STRIPE_LEVEL` - the [striping level for directories](https://cloud.google.com/parallelstore/docs/performance#directory_striping_setting) . Allowed values are:
    - directory-stripe-level-balanced
    - directory-stripe-level-max
    - directory-stripe-level-min
- `FILE_STRIPE_LEVEL` - the [file striping settings](https://cloud.google.com/parallelstore/docs/performance#file_striping_setting). Allowed values are:
    - file-stripe-level-balanced
    - file-stripe-level-max
    - file-stripe-level-min


## What's next

Once you've set up your GKE cluster with A3 Mega node pools, you can proceed to deploy and
run your [benchmark recipes](../README.md#benchmarks-support-matrix).

## Get Help

If you encounter any issues or have questions about this setup, use one of the following
resources:

- Consult the [official GKE documentation](https://cloud.google.com/kubernetes-engine/docs).
- Check the issues section of this repository for known problems and solutions.
- Reach out to Google Cloud support.

