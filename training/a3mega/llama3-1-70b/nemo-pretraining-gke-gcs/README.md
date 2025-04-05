# Pretrain Llama-3.1-70B workloads on A3 Mega GKE Node pools using Google Cloud Storage for training data and checkpoints.

This recipe outlines the steps for running a Llama-3.1-70B pretraining workload on
[A3 Mega GKE Node pools](https://cloud.google.com/kubernetes-engine) by using the
[NVIDIA NeMo framework](https://github.com/NVIDIA/nemo). Google Cloud Storage buckets with [hierarchical namespace](https://cloud.google.com/storage/docs/hns-overview)  are used to manage training data and checkpoints.

## Orchestration and deployment tools

For this recipe, the following setup is used:

- Orchestration - [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine)
- Job configuration and deployment - Helm chart is used to configure and deploy the
  [Kubernetes Index Job](https://kubernetes.io/blog/2021/04/19/introducing-indexed-jobs).
  This job encapsulates the
  [NVIDIA NeMo Megatron GPT pretraining workload](https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/megatron_gpt_pretraining.py).
  The chart generates the job's manifest, adhering to best practices for using GPUDirect-TCPXO
  with Google Kubernetes Engine (GKE), which includes setting optimal values for NVIDIA NCCL
  and the TCPXO NCCL plugin.


## Test environment

This recipe has been tested on the following environment configuration:

- A  standard GKE cluster:
  - GKE version: 1.32.2-gke.1182001 or later
  - A GPU node pool with 32 [a3-megagpu-8g](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3-mega-vms) machines provisioned using a [compact placement policy](https://cloud.google.com/kubernetes-engine/docs/how-to/compact-placement)
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
[GKE environment setup guide](../../../../docs/configuring-environment-gke-a3-mega.md).


## Training dataset

The recipe uses the [Pile dataset](https://pile.eleuther.ai/) converted to NeMo memory map (mmap) format.

## Run the recipe

It's recommended to use Cloud Shell as your client to complete the steps.
Cloud Shell comes pre-installed with the necessary utilities, including
`kubectl`, `the Google Cloud SDK`, and `Helm`.

### Launch Cloud Shell

In the Google Cloud console, start a [Cloud Shell Instance](https://console.cloud.google.com/?cloudshell=true).

### Configure environment settings

From your client, complete the following steps:

1. Set the environment variables to match your environment:

  ```bash
  export PROJECT_ID=<PROJECT_ID>
  export CLUSTER_REGION=<CLUSTER_REGION>
  export CLUSTER_NAME=<CLUSTER_NAME>
  export GCS_BUCKET_LOGS=<GCS_BUCKET>
  export GCS_BUCKET_DATA=<GCS_BUCKET_DATA>
  export GCS_BUCKET_CHECKPOINTS=<GCS_BUCKET_CHECKPOINTS>
  ```
  Replace the following values:

  - `<PROJECT_ID>`: your Google Cloud project ID
  - `<CLUSTER_REGION>`: the region where your cluster is located
  - `<CLUSTER_NAME>`: the name of your GKE cluster
  - `<GCS_BUCKET_LOGS>`: the name of a Cloud Storage bucket for logs. Do not include the `gs://` prefix
  - `<GCS_BUCKET_DATA>`: the name of a Cloud Storage bucket for training data. Do not include the `gs://` prefix
  - `<GCS_BUCKET_CHECKPOINTS>`: the name of a Cloud Storage bucket for checkpoints. Do not include the `gs://` prefix

2. Set the default project:

  ```bash
  gcloud config set project $PROJECT_ID
  ```

### Upload the training dataset

The Pile dataset in the NVIDIA NeMo *mmap* format is staged in the public GCS bucket. You need to upload the dataset to your GCS bucket with hierarchical namespace enabled.

1. Create a folder for the dataset

```
gcloud storage folders create gs://${GCS_BUCKET_DATA}/pile
```

2. Upload the dataset

```
gcloud storage cp gs://cloud-samples-data/third-party/pile/*.* gs://${GCS_BUCKET_DATA}/pile/
```

### Get the recipe

From your client, clone the `gpu-recipes` repository and set a reference to the recipe folder.

```
git clone https://github.com/ai-hypercomputer/gpu-recipes.git
cd gpu-recipes
export REPO_ROOT=`git rev-parse --show-toplevel`
export RECIPE_ROOT=$REPO_ROOT/training/a3mega/llama3-1-70b/nemo-pretraining-gke-gcs
```

### Get cluster credentials

From your client, get the credentials for your cluster.

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

By default the pre-training job will run 100 steps and generate a checkpoint every 25 steps.
To execute the job with the default settings, run the following command from your client:

```bash
cd $RECIPE_ROOT
helm install -f values.yaml \
    --set-file nemo_config=$REPO_ROOT/src/frameworks/a3mega/nemo-configs/llama3-1-70b-256gpus-bf16-pile-checkpointing.yaml \
    --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET_LOGS} \
    $USER-llama31-70b-gcs \
    $REPO_ROOT/src/helm-charts/a3mega/nemo-training-v2
```

#### Configure job settings

You can overwrite any of the default
[NeMo configurations](../../../../src/frameworks/a3mega/nemo-configs/llama3-1-70b-256gpus-bf16-pile-checkpointing.yaml)
for this job. To do this, we can set the new arguments using `--set workload.nemoConfigOverrides`.

**Examples**

- To set the number of training steps to 500, and the interval at which a checkpoint is taken to 50, run the following command:

    ```bash
    cd $RECIPE_ROOT
    helm install -f values.yaml \
        --set-file nemo_config=$REPO_ROOT/src/frameworks/a3mega/nemo-configs/llama3-1-70b-256gpus-bf16-pile-checkpointing.yaml \
        --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET_LOGS} \
        --set workload.nemoConfigOverrides[0]="trainer.max_steps=100" \
        --set workload.nemoConfigOverrides[1]="exp_manager.checkpoint_callback_params.every_n_train_steps=50" \
        $USER-llama31-70b-gcs \
        $REPO_ROOT/src/helm-charts/a3mega/nemo-training-v2
    ```


### Monitor the job

To check the status of pods in the indexed job, run the following command from your client:

```
kubectl get pods | grep $USER-llama-3-1-70b-256-nemo
```

To get the logs for one of the pods, run the following command from your client:

```
kubectl logs "<pod_name>"
```

### Analyze results

When completed, the job creates several artifacts, including logs and traces, and places them
in the  Google Cloud Storage logs bucket as follows:

```
gs://${GCS_BUCKET_LOGS}/nemo-experiments-storage/<JOB_ID>
├── hparams.yaml
├── lightning_logs.txt
├── nemo_error_logs.txt
├── nemo_log_globalrank-[RANK]_localrank-[LOCAL].txt
├── dllogger
│   ├── rank-0
│   │   ├── dllogger.json
...
```

- `hparams.yaml`: the NeMo configuration used by the pretraining script. This includes
   the combined [configuration file](../../../../src/frameworks/a3mega/nemo-configs/llama3.1-70b-256gpus-bf16.yaml)
   and the command line overrides
- `lightning_logs.txt`: the log files generated by PyTorch Lightning, which is used by NeMo
- `nemo_error_logs.txt`: the warning and error logs generated by NeMo
- `nemo_log_globalrank-[RANK]_localrank-[LOCAL].txt`: the NeMo logs for each rank
- `dllogger/`: The log captured by [NVIDIA DLLogger](https://github.com/NVIDIA/dllogger):
   DLLogger is configured to store logs on the rank 0 node. The log is in JSON format
   and includes loss, step_time, and other key metrics for each training step

The `<JOB_ID>` has the following format:
- `$USER--llama31-70b-gcs-[YYYY]-[MM]-[DD]-[hh]-[mm]-[ss]`, where the suffix of the ID is a day and time when the job was started.


The NeMo log files include information about checkpoint operations on each rank. You can use the [checkpointing_metrics](../../../../src/utils/checkpointint_metrics) utility to calculate statistics for checkpoint write times.

To calculate statistics:


1. Set a path to the NeMo logs.

```
export JOB_ID=<JOB_ID>
export GCS_LOGS_PATH="gs://${GCS_BUCKET_LOGS}/nemo-experiments-storage/${JOB_ID}"
```

Replace `<JOB_ID>` with the ID of your job.

2. Run the utility

```
cd $REPO_ROOT/src/utils/checkpointing_metrics
python3 calculate_checkpoint_metrics.py --gcs_logs_path=${GCS_LOGS_PATH}
```

You should see the output similar to the following:

```
...
Analyzing file: nemo-experiments-storage/job_id/nemo_log_globalrank-8_localrank-0.txt, Global rank: 8, Local rank: 0
Analyzing file: nemo-experiments-storage/job_id/run_0/nemo_log_globalrank-7_localrank-7.txt, Global rank: 7, Local rank: 7
min checkpoint write duration: 172.0s
max checkpoint write duration: 323.0s
average checkpoint write duration: 257.75s
checkpoint write time standard deviation: 63.03107699116894
```



### Uninstall the Helm release

You can delete the job and other resources created by the Helm chart.
To uninstall Helm, run the following command from your client:

```bash
helm uninstall $USER-llama31-70b-gcs
```

### Uninstall PVCs and PVs

To uninstall Persistent Volume and Persistent Volume Claim resources for Parallelstore execute the following command:

```
helm uninstall $USER-gcs-pv-pvc
```

