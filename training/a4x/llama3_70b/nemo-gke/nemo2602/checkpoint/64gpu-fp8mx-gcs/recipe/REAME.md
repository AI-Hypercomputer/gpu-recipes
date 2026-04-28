<!-- mdformat global-off -->
# Pretrain llama3-1-70b workloads on a4x GKE Node pools with Nvidia NeMo Framework

This recipe outlines the steps for running a llama3-1-70b pretraining
workload on [a4x GKE Node pools](https://cloud.google.com/kubernetes-engine) by using the
[NVIDIA NeMo framework](https://github.com/NVIDIA/nemo).

## Orchestration and deployment tools

For this recipe, the following setup is used:

- Orchestration - [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine)
- Pretraining job configuration and deployment - A Helm chart is used to
  configure and deploy the [Kubernetes Jobset](https://kubernetes.io/blog/2025/03/23/introducing-jobset) resource which manages the execution of the
  [NeMo pretraining workload](https://github.com/NVIDIA/nemo).

## Test environment

This recipe has been optimized for and tested with the following configuration:

- GKE cluster
Please follow Cluster Toolkit [instructions](https://github.com/GoogleCloudPlatform/cluster-toolkit/tree/main/examples/gke-a4x)
to create your a4x GKE cluster.

## Training dataset

This recipe uses a mock pretraining dataset provided by the NeMo framework.

## Docker container image

This recipe uses the following docker images:

- `nvcr.io/nvidia/nemo:26.02.01`
- `us-docker.pkg.dev/gce-ai-infra/gpudirect-gib/nccl-plugin-gib-diagnostic-arm64:v1.1.0`

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
 ```

Replace the following values:

 - `<PROJECT_ID>`: your Google Cloud project ID.
 - `<CLUSTER_REGION>`: the region where your cluster is located.
 - `<CLUSTER_NAME>`: the name of your GKE cluster.
 - `<GCS_BUCKET>`: the name of your Cloud Storage bucket. Don't include the `gs://` prefix.
 - `<KUEUE_NAME>`: the name of the Kueue local queue. The default queue created by the cluster toolkit is `a4x`. Make sure to verify the name of the local queue in your cluster.

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
export RECIPE_ROOT=$REPO_ROOT/training/a4x/llama3_70b/nemo-gke/nemo2602/64gpus-fp8mx-gcs/recipe
cd $RECIPE_ROOT
```

### Get cluster credentials

```
gcloud container clusters get-credentials $CLUSTER_NAME --region $CLUSTER_REGION
```
### Setup GCS bucket PVCs
1) Create GCS buckets for dataload and checkpointing.

Ensure the buckets have `--uniform-bucket-level-access` and `--enable-hirearchical-namespace`

```
export CKPT_BUCKET_NAME=a4x-storage-ckpt
export DL_BUCKET_NAME=a4x-storage-data

gcloud storage buckets create "gs://${CKPT_BUCKET_NAME}" --location=$CLUSTER_REGION --uniform-bucket-level-access --enable-hierarchical-namespace

gcloud storage buckets create "gs://${DL_BUCKET_NAME}" --location=$CLUSTER_REGION --uniform-bucket-level-access --enable-hierarchical-namespace
```

2) Upload training dataset to dataload bucket
For commands, see: [Upload opjects from a file system](https://docs.cloud.google.com/storage/docs/uploading-objects)

3) Create PersistentVolumes for the buckets and claim these volumes

Replace `<DL_BUCKET_NAME>` and `<CKPT_BUCKET_NAME>` in gcs_pv.yamk and run the following command. 
Ensure file cache is enabled for optimal performance.

```
kubectl apply -f ./gcs_pv.yaml
```

4) Add volume claims to values.yaml

```
...
gcsVolumes: true
psVolumes: false
pvcMounts:
- claimName: "a4x-storage-ckpt-pvc"
  mountPath: "/gcsckpt"
- claimName: "a4x-storage-data-pvc"
  mountPath: "/gcsdata"
...
```

Using PVC optimizes performance, but you can also mount the GCS bucket using gcsMounts option.

### Configure and submit a pretraining job

#### Using 32 node (64 gpus) fp8 precision with GCS dataload and checkpointing
To execute the job with dataloading and checkpoint saving, run the following:

```bash
cd $RECIPE_ROOT
export WORKLOAD_NAME=$USER-a4x-llama3-1-70b-32node
export DATASET_TYPE=<DATASET_TYPE>
export DATASET_PATHS=<DATASET_PATHS>
export INDEX_MAPPING_DIR=<INDEX_MAPPING_DIR>
export CKPT_SAVE_DIR=<CKPT_SAVE_DIR>
export CKPT_SAVE_INTERVAL=<CKPT_SAVE_INTERVAL>
helm install $WORKLOAD_NAME . -f values.yaml \
--set-file workload_launcher=launcher.sh \
--set-file workload_config=llama3-1-70b-fp8cs-gbs2048-gpus64.py \
--set workload.image=nvcr.io/nvidia/nemo:26.02.01 \
--set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
--set volumes.gcsMounts[0].mountPath=/job-logs \
--set workload.envs[0].value=/job-logs/$WORKLOAD_NAME \
--set workload.envs[5].value=${DATASET_TYPE} \
--set workload.envs[6].value="/gcsdata/${DATASET_PATHS}" \
--set workload.envs[7].value=/gcsdata/${INDEX_MAPPING_DIR} \
--set workload.envs[8].value=/gcsckpt/${CKPT_SAVE_DIR} \
--set workload.envs[9].value=/gcsckpt/${CKPT_SAVE_INTERVAL} \
--set queue=${KUEUE_NAME}
```

Replace the following values:
 - `<DATASET_TYPE>`: The type of dataset used (see [Megatron-Bridge data arguments](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/r0.3.0/scripts/performance#data-arguments))
 - `<DATASET_PATHS>`: Paths to your dataset (for rp2 dataset)
 - `<INDEX_MAPPING_DIR>`: Index mapping dir (for rp2 dataset)
 - `<CKPT_SAVE_DIR>`: The directory where you wish you save checkpoints
 - `<CKPT_SAVE_INTERVAL>`: Save checkpoint every CKPT_SAVE_INTERVAL train steps

Note: Edit `recipe.dataset.num_workers = 8` in `run_script.py` to change the number of dataloading workers.

This recipe uses Nvidia NeMo 26.02.01 container, which uses Megatron-Bridge for checkpointing and dataloading. See [checkpointing arguments](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/r0.3.0/scripts/performance#checkpointing-arguments) and [data arguments](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/r0.3.0/scripts/performance#data-arguments) for explanations on dataset types.

You can also edit values.yaml directly.
```
- name: DATASET_TYPE
  value: "rp2"
- name: DATASET_PATHS
  value: "/gcsdata/${DATASET_PATHS}"
- name: INDEX_MAPPING_DIR
  value: "/gcsdata/${INDEX_MAPPING_DIR}"
- name: CKPT_SAVE_DIR
  value: "/gcsckpt/${CKPT_SAVE_DIR}"
- name: CKPT_SAVE_INTERVAL
  value: "10"
```

To load a checkpoint, add the following:
```bash
cd $RECIPE_ROOT
export WORKLOAD_NAME=$USER-a4x-llama3-1-70b-32node
export CKPT_LOAD_DIR=<CKPT_LOAD_DIR>
export CKPT_LOAD_STEP=<CKPT_LOAD_STEP>
helm install $WORKLOAD_NAME . -f values.yaml \
--set-file workload_launcher=launcher.sh \
--set-file workload_config=llama3-1-70b-fp8cs-gbs2048-gpus64.py \
--set workload.image=nvcr.io/nvidia/nemo:26.02.01 \
--set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
--set volumes.gcsMounts[0].mountPath=/job-logs \
--set workload.envs[0].value=/job-logs/$WORKLOAD_NAME \
--set workload.envs[10].value=/gcsckpt/${CKPT_LOAD_DIR} \
--set workload.envs[11].value="/gcsckpt/${CKPT_LOAD_STEP}" \
--set queue=${KUEUE_NAME}
```

Or edit values.yaml directly
```
- name: CKPT_LOAD_DIR
  value: "/gcsckpt/${CKPT_LOAD_DIR}"
- name: CKPT_LOAD_STEP
  value: "10"
```

**Examples**

-   To set the number of training steps to 100, run the following command from your client:

```bash
cd $RECIPE_ROOT
export WORKLOAD_NAME=$USER-a4x-llama3-1-70b-32node
helm install $WORKLOAD_NAME . -f values.yaml \
--set-file workload_launcher=launcher.sh \
--set-file workload_config=llama3-1-70b-fp8cs-gbs2048-gpus64.py \
--set workload.image=nvcr.io/nvidia/nemo:26.02.01 \
--set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
--set volumes.gcsMounts[0].mountPath=/job-logs \
--set workload.envs[0].value=/job-logs/$WORKLOAD_NAME \
--set workload.envs[5].value=${DATASET_TYPE} \
--set workload.envs[6].value="/gcsdata/${DATASET_PATHS}" \
--set workload.envs[7].value=/gcsdata/${INDEX_MAPPING_DIR} \
--set workload.envs[8].value=/gcsckpt/${CKPT_SAVE_DIR} \
--set workload.envs[9].value=/gcsckpt/${CKPT_SAVE_INTERVAL} \
--set queue=${KUEUE_NAME} \
--set workload.arguments[0]="trainer.max_steps=100"
```


### Monitor the job

To check the status of pods in your job, run the following command:

```
kubectl get pods | grep $USER-a4x-llama3-1-70b-32node
```

Replace the following:

- JOB_NAME_PREFIX - your job name prefix. For example $USER-a4x-llama3-1-70b-32node.

To get the logs for one of the pods, run the following command:

```
kubectl logs POD_NAME
```

Information about the training job's progress, including crucial details such as
loss, step count, and step time, is generated by the rank 0 process.
This process runs on the pod whose name begins with
`JOB_NAME_PREFIX-workload-0-0`.
For example: `$USER-a4x-llama3-1-70b-32node-workload-0-0-s9zrv`.

Logs will display checkpoint save directory:
```
28_worker0/0   successfully saved checkpoint from iteration      40 to /gcsckpt/ckpt/asq-llama70b-ckpt-64gpu-gcs-2026-04-03-03-51-29 [ t 1/2, p 1/4 ] 
```

### Uninstall the Helm release

You can delete the job and other resources created by the Helm chart. To
uninstall Helm, run the following command from your client:

```bash
helm uninstall $USER-a4x-llama3-1-70b-32node
```