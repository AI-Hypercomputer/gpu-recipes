<!-- mdformat global-off -->
# Pretrain llama3-1-70b-gpus128 workloads on A4 GKE Node pools with Nvidia NeMo Framework  using Google Cloud Storage for training data and checkpoints

This recipe outlines the steps for running a llama3-1-70b-gpus128 pretraining
workload on [A4 GKE Node pools](https://cloud.google.com/kubernetes-engine) by using the
[NVIDIA NeMo framework](https://github.com/NVIDIA/nemo).

## Orchestration and deployment tools

For this recipe, the following setup is used:

- Orchestration - [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine)
- Pretraining job configuration and deployment - A Helm chart is used to
  configure and deploy the [Kubernetes Jobset](https://kubernetes.io/blog/2025/03/23/introducing-jobset) resource which manages the execution of the
  [NeMo pretraining workload](https://github.com/NVIDIA/nemo). The chart generates the job's manifest, adhering to best practices for using GPUDirect-TCPXO with Google Kubernetes Engine (GKE), which includes setting optimal values for NVIDIA NCCL and the TCPXO NCCL plugin.


## Test environment

This recipe has been optimized for and tested with the following configuration:

- A  standard GKE cluster:
  - GKE version: 1.33.5-gke.1162000 or later
  - A GPU node pool with 16 [a4-highgpu-8g](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-machine-type) machines
  - Workload Identity Federation for GKE enabled
  - Cloud Storage FUSE CSI driver enabled
  - DCGM metrics enabled
  - Kueue and JobSet APIs installed
  - Kueue configured to support Topology Aware Scheduling
- A regional Google Cloud Storage (GCS) bucket to store logs.
- A regional Google Cloud Storage (GCS) bucket with [hierarchical](https://cloud.google.com/storage/docs/hns-overview)) namespace to store the Pile dataset
- A regional Google Cloud Storage (GCS) bucket with [hierarchical](https://cloud.google.com/storage/docs/hns-overview)) namespace to store checkpoints
- A client workstation with the following pre-installed:
   - Google Cloud SDK
   - Helm
   - kubectl

*Important: All GCS buckets must be in the same region as the GKE cluster*.

To prepare the required environment, see
[GKE environment setup guide](../../../../docs/configuring-environment-gke-a4.md).

## Training dataset

The recipe uses the [Pile dataset](https://pile.eleuther.ai/) converted to NeMo memory map (mmap) format.

## Docker container image

This recipe uses the following docker images:

- `nvcr.io/nvidia/nemo:25.07`
- `us-docker.pkg.dev/gce-ai-infra/gpudirect-gib/nccl-plugin-gib-arm64:v1.0.6`

## Run the recipe

From your client workstation, complete the following steps:

### Configure environment settings

Set the environment variables to match your environment:

 ```bash
 export PROJECT_ID=<PROJECT_ID>
 export CLUSTER_REGION=<CLUSTER_REGION>
 export CLUSTER_NAME=<CLUSTER_NAME>
 export GCS_BUCKET_LOGS=<GCS_BUCKET_LOGS>
 export GCS_BUCKET_DATA=<GCS_BUCKET_DATA>
 export GCS_BUCKET_CHECKPOINTS=<GCS_BUCKET_CHECKPOINTS>
 export ENABLE_DATALOADING=<ENABLE_DATALOADING>
 export ENABLE_CHECKPOINT_WRITE=<ENABLE_CHECKPOINT_WRITE>
 export CHECKPOINT_WRITE_INTERVAL=<CHECKPOINT_WRITE_INTERVAL>
 export ENABLE_CHECKPOINT_LOAD=<ENABLE_CHECKPOINT_LOAD>
 export RESTORE_PATH=<RESTORE_PATH>
 export TOKEN_PATH=<TOKEN_PATH>
 export DATASET_PATH=<DATASET_PATH>
 ```

Replace the following values:

  - `<PROJECT_ID>`: your Google Cloud project ID
  - `<CLUSTER_REGION>`: the region where your cluster is located
  - `<CLUSTER_NAME>`: the name of your GKE cluster
  - `<GCS_BUCKET_LOGS>`: the name of a Cloud Storage bucket for logs. Do not include the `gs://` prefix
  - `<GCS_BUCKET_DATA>`: the name of a Cloud Storage bucket for training data. Do not include the `gs://` prefix
  - `<GCS_BUCKET_CHECKPOINTS>`: the name of a Cloud Storage bucket for checkpoints. Do not include the `gs://` prefix
  * `<ENABLE_DATALOADING>`: The recipe has an option to use real dataset for dataloading.
*   `<ENABLE_CHECKPOINT_WRITE>`: To enable checkpoint write.
*   `<CHECKPOINT_WRITE_INTERVAL>`: Step interval at which checkpoint will be written.
*   `<ENABLE_CHECKPOINT_LOAD>`: To enable checkpoint restore.
*   `<RESTORE_PATH>`: Path to a specific checkpoint to restore from. The mount point of checkpoint_bucket is `/checkpoints` and hence the path should start with `/checkpoints`. 
*   `<TOKEN_PATH>`: tokenizer model file of sentencepiece.
*   `<DATASET_PATH>`: Path in dataset_bucket for dataloading. The path should contain only the dataloading objects. The mount point of dataset_bucket is `/data` and hence the path should start with `/data`. 

Set the default project:

 ```bash
 gcloud config set project $PROJECT_ID
 ```
### Upload the training dataset

The Pile dataset in the NVIDIA NeMo *mmap* format is staged in the public GCS bucket. You need to upload the dataset to your GCS bucket with hierarchical namespace enabled.

1. Create a folder for the dataset

```
gcloud storage folders create gs://${GCS_BUCKET_DATA}/data
```

2. Upload the dataset

```
gcloud storage cp gs://cloud-samples-data/third-party/pile/*.* gs://${GCS_BUCKET_DATA}/data
```

### Get the recipe

Clone the `gpu-recipes` repository and set a reference to the recipe folder.

```
git clone https://github.com/ai-hypercomputer/gpu-recipes.git
cd gpu-recipes
export REPO_ROOT=`git rev-parse --show-toplevel`
export RECIPE_ROOT=$REPO_ROOT/training/a4/llama3-1-70b/nemo-pretraining-gke/16node-bf16-seq8192-gbs512-gcs
cd $RECIPE_ROOT
```

### Get cluster credentials

```
gcloud container clusters get-credentials $CLUSTER_NAME --region $CLUSTER_REGION
```

### Create Persistent Volumes and Persistent Volume Claims

The pretraining job accesses GCS buckets for training data and checkpoints through [the Cloud Storage FUSE CSI driver](https://cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/cloud-storage-fuse-csi-driver) configured using Kubernetes Persistent Volumes (PV) and Persistent Volume Claims (PVC). You must generate PVs and PVCs for both data and checkpoint buckets using the [gcs-fuse helper Helm chart](../../../../src/helm-charts/storage/gcs-fuse). The chart configures the FUSE driver settings following the best practices for optimizing access to buckets for training data and checkpoints.

```
helm install -f $REPO_ROOT/src/helm-charts/storage/gcs-fuse/values.yaml \
--set gcsVolumes[0].bucketName=${GCS_BUCKET_DATA} \
--set gcsVolumes[1].bucketName=${GCS_BUCKET_CHECKPOINTS} \
$USER-gcs-pv-pvc \
$REPO_ROOT/src/helm-charts/storage/gcs-fuse
```

### Configure and submit a pretraining job

#### Using 16 node (64 gpus) bf16-mixed precision
To execute the job with the default settings, run the following command from
your client:

    ```bash
    cd $RECIPE_ROOT
    export WORKLOAD_NAME=a4-llama3-1-70b-gpus128
    helm install $WORKLOAD_NAME . -f values.yaml \
    --set workload_launcher=launcher.sh \
    --set workload_config=llama3-1-70b.py \
    --set workload.image=nvcr.io/nvidia/nemo:25.07 \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET_LOGS} \
    --set volumes.gcsMounts[0].mountPath=/job-logs
    ```

**Examples**

-   To set the number of training steps to 100, run the following command from
    your client:

    ```bash
    cd $RECIPE_ROOT
    export WORKLOAD_NAME=a4-llama3-1-70b-gpus128
    helm install $WORKLOAD_NAME . -f values.yaml \
    --set workload_launcher=launcher.sh \
    --set workload_config=llama3-1-70b.py \
    --set workload.image=nvcr.io/nvidia/nemo:25.07 \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET_LOGS} \
    --set volumes.gcsMounts[0].mountPath=/job-logs \
    --set workload.step_count=100
    ```
-   To enable dataloading, checkpoint restore and checkpoint write at every 25 steps, run the following command from
    your client:

    ```bash
    cd $RECIPE_ROOT
    export WORKLOAD_NAME=a4-llama3-1-70b-gpus128
    helm install $WORKLOAD_NAME . -f values.yaml \
    --set workload_launcher=launcher.sh \
    --set workload_config=llama3-1-70b.py \
    --set workload.image=nvcr.io/nvidia/nemo:25.07 \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET_LOGS} \
    --set volumes.gcsMounts[0].mountPath=/job-logs \
    --set workload.enable_dataloading=$ENABLE_DATALOADING \
    --set workload.enable_ckpt_write=$ENABLE_CHECKPOINT_WRITE \
    --set workload.enable_ckpt_load=$ENABLE_CHECKPOINT_LOAD \
    --set workload.ckpt_write_interval=$CHECKPOINT_WRITE_INTERVAL \
    --set workload.token_path=$TOKEN_PATH \
    --set workload.dataset_path=$DATASET_PATH \
    --set workload.restore_path=$RESTORE_PATH 
    ```

### Monitor the job

To check the status of pods in your job, run the following command:

```
kubectl get pods | grep a4-llama3-1-70b-gpus128
```

Replace the following:

- JOB_NAME_PREFIX - your job name prefix. For example a4-llama3-1-70b-gpus128.

To get the logs for one of the pods, run the following command:

```
kubectl logs POD_NAME
```

Information about the training job's progress, including crucial details such as
loss, step count, and step time, is generated by the rank 0 process.
This process runs on the pod whose name begins with
`JOB_NAME_PREFIX-workload-0-0`.
For example: `a4-llama3-1-70b-gpus128-workload-0-0-s9zrv`.

### Analyze results

When completed, the job creates several artifacts, including logs and traces, and places them
in the  Google Cloud Storage logs bucket as follows:

```
gs://${GCS_BUCKET_LOGS}/nemo-experiments-storage/<JOB_ID>
├── nemo-configuration.yaml
├── lightning_logs.txt
├── nemo_error_logs.txt
├── nemo_log_globalrank-[RANK]_localrank-[LOCAL].txt
├── dllogger
│   ├── rank-0
│   │   ├── dllogger.json
...
```

- `nemo-configuration.yaml`: the NeMo configuration used by the pretraining script. This includes
   the combined [configuration file](../16node-bf16-seq8192-gbs512/llama3-1-70b.py)
   and the command line overrides
- `lightning_logs.txt`: the log files generated by PyTorch Lightning, which is used by NeMo
- `nemo_error_logs.txt`: the warning and error logs generated by NeMo
- `nemo_log_globalrank-[RANK]_localrank-[LOCAL].txt`: the NeMo logs for each rank
- `dllogger/`: The log captured by [NVIDIA DLLogger](https://github.com/NVIDIA/dllogger):
   DLLogger is configured to store logs on the rank 0 node. The log is in JSON format
   and includes loss, step_time, and other key metrics for each training step


The NeMo log files include information about checkpoint operations on each rank. 

### Uninstall the Helm release

You can delete the job and other resources created by the Helm chart. To
uninstall Helm, run the following command from your client:

```bash
helm uninstall $WORKLOAD_NAME
```

### Uninstall PVCs and PVs

To uninstall Persistent Volume and Persistent Volume Claim resources for Parallelstore execute the following command:

```
helm uninstall $USER-gcs-pv-pvc
```
