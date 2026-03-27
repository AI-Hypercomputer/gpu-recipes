<!-- mdformat global-off -->
# Pretrain Qwen 3 235B A22B workloads on A4X Slurm Cluster with Nvidia Megatron-Bridge

This recipe outlines the steps for running a Qwen 3 235B A22B pretraining workload on [Google Cloud A4X Slurm clusters](https://docs.cloud.google.com/ai-hypercomputer/docs/create/create-slurm-cluster) by using [NVIDIA Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge).

## Orchestration and deployment tools

For this recipe, the following setup is used:

- Orchestration - [Slurm Workload Manager](https://slurm.schedmd.com/)
- Deployment - [Cluster Toolkit](https://cloud.google.com/cluster-toolkit/docs/overview)

## Test environment

This recipe has been optimized for and tested with the following configuration:

- A4X Slurm Cluster (32 nodes, 128 GPUs)
- Machine Type: A4X (GB200)
- Lustre Filesystem

Please follow the instructions in the [Cluster Toolkit A4X Example README](https://github.com/GoogleCloudPlatform/cluster-toolkit/tree/main/examples/machine-learning) to provision an A4X Slurm cluster.

## Docker container image

This recipe uses the following container images:

- `nvcr.io/nvidia/nemo:25.11`

## Run the recipe

### Configure environment settings

Set the environment variables to match your environment:

 ```bash
 export PROJECT_ID=<PROJECT_ID>
 export CLUSTER_REGION=<CLUSTER_REGION>
 export CLUSTER_NAME=<CLUSTER_NAME>
 gcloud compute ssh $CLUSTER_NAME --project <project-name> --zone $CLUSTER_REGION -- -o Hostname=nic0.$CLUSTER_NAME.$CLUSTER_REGION.c.$PROJECT_ID$.internal.gcpnode.com
 ```

Replace the following values:

 - `<PROJECT_ID>`: your Google Cloud project ID.
 - `<CLUSTER_REGION>`: the region where your cluster is located.
 - `<CLUSTER_NAME>`: the name of your SLURM cluster.

Set the default project:

 ```bash
 gcloud config set project $PROJECT_ID
 ```

From your cluster login node, complete the following steps:

### Get the recipe

Clone the `gpu-recipes` repository and set a reference to the recipe folder.

```
git clone https://github.com/ai-hypercomputer/gpu-recipes.git
cd gpu-recipes
export REPO_ROOT=`git rev-parse --show-toplevel`
export RECIPE_ROOT=$REPO_ROOT/training/a4x/qwen3-235b-a22b/megatron-bridge-pretraining-slurm/32node-BF16-GBS2048/recipe
cd $RECIPE_ROOT
```

### Submit a pretraining job

**Note:** Before running the recipe, please ensure you replace `<YOUR_HF_TOKEN>` with your actual Hugging Face token in `launch_script.sh`.

```
cd ..
sbatch ./recipe/sbatch_script.sh
```

### Monitor the job

To check the status of pods in your job, run the following command:

```
squeue --me
```

To get the logs for the job, run the following command:

```
tail -f slurm_{jobID}.out
```

### Uninstall the job

```bash
scancel -u $USER
```
