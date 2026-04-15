<!-- mdformat global-off -->
# Pretrain Wan workloads on a3ultra GKE Node pools with Nvidia Megatron-Bridge Framework

This recipe outlines the steps for running a Wan pretraining
workload on [a3ultra GKE Node pools](https://cloud.google.com/kubernetes-engine) by using the
[Megatron-Bridge pretraining workload](https://github.com/NVIDIA-NeMo/Megatron-Bridge) and [DFM](https://github.com/NVIDIA-NeMo/DFM).

## Orchestration and deployment tools

For this recipe, the following setup is used:

- Orchestration - [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine)
- Pretraining job configuration and deployment - A Helm chart is used to
  configure and deploy the [Kubernetes Jobset](https://kubernetes.io/blog/2025/03/23/introducing-jobset) resource which manages the execution of the
  workload.

## Test environment

This recipe has been optimized for and tested with the following configuration:

- GKE cluster
Please follow Cluster Toolkit [instructions](https://github.com/GoogleCloudPlatform/cluster-toolkit/)
to create your a3ultra GKE cluster.

## Training dataset

This recipe uses a mock pretraining dataset.

## Docker container image

This recipe uses the following docker images:

- `nvcr.io/nvidia/nemo:26.02`
- `us-docker.pkg.dev/gce-ai-infra/gpudirect-gib/nccl-plugin-gib:v1.1.0`

## Run the recipe

From your client workstation, complete the following steps:

### Configure environment settings

Set the environment variables to match your environment:

 ```bash
 export PROJECT_ID=<PROJECT_ID>
 export CLUSTER_REGION=<CLUSTER_REGION>
 export CLUSTER_NAME=<CLUSTER_NAME>
 export GCS_BUCKET=<GCS_BUCKET> # Note: path should not be prefixed with gs://
 export KUEUE_NAME=<KUEUE_NAME>
 export HF_TOKEN=<YOUR_HF_TOKEN>
 ```

Replace the following values:

 - `<PROJECT_ID>`: your Google Cloud project ID.
 - `<CLUSTER_REGION>`: the region where your cluster is located.
 - `<CLUSTER_NAME>`: the name of your GKE cluster.
 - `<GCS_BUCKET>`: the name of your Cloud Storage bucket. Don't include the `gs://` prefix.
 - `<KUEUE_NAME>`: the name of the Kueue local queue. The default queue created by the cluster toolkit is `a3ultra`. Make sure to verify the name of the local queue in your cluster.
 - `<YOUR_HF_TOKEN>`: Your HuggingFace token.

Set the default project:

 ```bash
 gcloud config set project $PROJECT_ID
 ```

### Get the recipe

Clone the `gpu-recipes` repository and set a reference to the recipe folder.

```
git clone https://github.com/ai-hypercomputer/gpu-recipes.git
cd gpu-recipes
export REPO_ROOT=`git rev-parse --show-toplevel`
export RECIPE_ROOT=$REPO_ROOT/training/a3ultra/wan/megatron-bridge-gke/nemo2602/32gpus-bf16/recipe
cd $RECIPE_ROOT
```

### Get cluster credentials

```
gcloud container clusters get-credentials $CLUSTER_NAME --region $CLUSTER_REGION
```

### Configure and submit a pretraining job

#### Using 4 node (32 gpus) bf16 precision
To execute the job with the default settings, run the following command from
your client:

```bash
cd $RECIPE_ROOT
export WORKLOAD_NAME=$USER-a3ultra-wan-4node
helm install $WORKLOAD_NAME . -f values.yaml \
--set-file workload_launcher=launcher.sh \
--set workload.image=nvcr.io/nvidia/nemo:26.02 \
--set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
--set volumes.gcsMounts[0].mountPath=/job-logs \
--set workload.envs[0].value=/job-logs/$WORKLOAD_NAME \
--set queue=${KUEUE_NAME}
```

### Monitor the job

To check the status of pods in your job, run the following command:

```
kubectl get pods | grep $USER-a3ultra-wan-4node
```

### Uninstall the Helm release

You can delete the job and other resources created by the Helm chart. To
uninstall Helm, run the following command from your client:

```bash
helm uninstall $USER-a3ultra-wan-4node
```
