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
export RECIPE_ROOT=$REPO_ROOT/training/a4x/llama3_70b/nemo-gke/nemo2602/128gpus-fp8mx-lustre/recipe
cd $RECIPE_ROOT
```

### Get cluster credentials

```
gcloud container clusters get-credentials $CLUSTER_NAME --region $CLUSTER_REGION
```
### Setup Lustre PVCs
1) Create Lustre instance

Follow [Create Lustre Instance](https://docs.cloud.google.com/managed-lustre/docs/create-instance) to create a lustre instance.

**Create the instance in the same zone and network as your cluster.**

Run `gcloud lustre instances list --location $ZONE --project $PROJECT_ID` and take note of the filesystem, name and network. It will look something like this:
```
capacityGib: '126000'
createTime: '2026-01-07T00:24:57.572296415Z'
filesystem: <FILESYSTEM>
mountPoint: <LUSTRE_IP>@tcp:/<FILESYSTEM>
name: projects/<PROJECT_ID>/locations/<ZONE>/instances/<NAME>
network: projects/<PROJECT_ID>/global/networks/<NETWORK>
perUnitStorageThroughput: '1000'
state: ACTIVE
uid: 9b9400c4-a669-48d0-89de-fe4bce4664fb
updateTime: '2026-04-27T20:38:55.794373239Z'
```

2) Upload training dataset to the instance
For commands, see: TODO

3) Create PersistentVolumes for the instance and claim the volume

Replace the following variables in `lustre_pv.yaml`:
 - `<FILESYSTEM>`: Filesystem name of the instance
 - `<LUSTRE_IP>`: IP of Lustre instance
 - `<PROJECT_ID>`: Project ID
 - `<ZONE>`: Zone of your compute nodes and lustre instance
 - `<NAME>`: Lustre instance name
 - `<NETWORK>`: Network name of your cluster and instance

```
kubectl apply -f ./lustre_pv.yaml
```

4) Add volume claims to values.yaml

```
...
pvcMounts:
  - claimName: "asq-0106-lustre-pvc"
    mountPath: "/lustrefs"
...
```

### Configure and submit a pretraining job

#### Using 32 node (128 gpus) fp8 precision with Lustre dataload and checkpointing
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
--set-file workload_config=llama3-1-70b-fp8cs-gbs2048-gpus128.py \
--set workload.image=nvcr.io/nvidia/nemo:26.02.01 \
--set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
--set volumes.gcsMounts[0].mountPath=/job-logs \
--set workload.envs[0].value=/job-logs/$WORKLOAD_NAME \
--set workload.envs[5].value=${DATASET_TYPE} \
--set workload.envs[6].value="/lustrefs/data/${DATASET_PATHS}" \
--set workload.envs[7].value=/lustrefs/data/${INDEX_MAPPING_DIR} \
--set workload.envs[8].value=/lustrefs/ckpt/${CKPT_SAVE_DIR} \
--set workload.envs[9].value=/lustrefs/ckpt/${CKPT_SAVE_INTERVAL} \
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
  value: "/lustrefs/data/${DATASET_PATHS}"
- name: INDEX_MAPPING_DIR
  value: "/lustrefs/data/${INDEX_MAPPING_DIR}"
- name: CKPT_SAVE_DIR
  value: "/lustrefs/ckpt/${CKPT_SAVE_DIR}"
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
--set-file workload_config=llama3-1-70b-fp8cs-gbs2048-gpus128.py \
--set workload.image=nvcr.io/nvidia/nemo:26.02.01 \
--set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
--set volumes.gcsMounts[0].mountPath=/job-logs \
--set workload.envs[0].value=/job-logs/$WORKLOAD_NAME \
--set workload.envs[10].value=/lustrefs/ckpt/${CKPT_LOAD_DIR} \
--set workload.envs[11].value="${CKPT_LOAD_STEP}" \
--set queue=${KUEUE_NAME}
```

Or edit values.yaml directly
```
- name: CKPT_LOAD_DIR
  value: "/lustrefs/ckpt/${CKPT_LOAD_DIR}"
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
--set-file workload_config=llama3-1-70b-fp8cs-gbs2048-gpus128.py \
--set workload.image=nvcr.io/nvidia/nemo:25.07 \
--set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
--set volumes.gcsMounts[0].mountPath=/job-logs \
--set workload.envs[0].value=/job-logs/$WORKLOAD_NAME \
--set workload.envs[5].value=${DATASET_TYPE} \
--set workload.envs[6].value="/lustrefs/data/${DATASET_PATHS}" \
--set workload.envs[7].value=/lustrefs/data/${INDEX_MAPPING_DIR} \
--set workload.envs[8].value=/lustrefs/ckpt/${CKPT_SAVE_DIR} \
--set workload.envs[9].value=/lustrefs/ckpt/${CKPT_SAVE_INTERVAL} \
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
28_worker0/0   successfully saved checkpoint from iteration      40 to /lustrefs/ckpt/asq-llama70b-ckpt-128gpu-lustre-2026-04-03-03-51-29 [ t 1/2, p 1/4 ] 
```

### Uninstall the Helm release

You can delete the job and other resources created by the Helm chart. To
uninstall Helm, run the following command from your client:

```bash
helm uninstall $USER-a4x-llama3-1-70b-32node
```