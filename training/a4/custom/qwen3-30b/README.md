# A4 GPU Training Guide for Qwen3-30B

## Usage

### 1. Connect to GKE Cluster

```bash
export PROJECT=your-project-id
export REGION=us-central1
export ZONE=us-central1-b
export CLUSTER_NAME=your-gke-cluster

gcloud config set project ${PROJECT}
gcloud config set compute/zone ${ZONE}
gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION
```

### 2. Build Docker Image

The entry point is customized `run.py` in the recipe folder.
Need to build your own image with the recipe.

```bash
export RECIPE_ROOT=$(pwd)
export IMAGE=us-central1-docker.pkg.dev/${PROJECT}/${USER}/nemo1:25.04

# Rootless docker
sudo usermod -aG docker $USER
newgrp docker

# The dockerfile needs to copy files in the recipe root
cd ${RECIPE_ROOT}
docker build -t $IMAGE -f docker/Dockerfile .
docker push $IMAGE
```

### 3. Select and Copy Recipe

Select the recipe and override the files in helm-context.

```bash
export RECIPE_ROOT=$(pwd)
cd ${RECIPE_ROOT}
export RECIPE_NAME=qwen2-5-7b-8gpu-spfalse-tp1-pp1-cp1-vpnull-gbs512-mbs4
cp ./recipes/${RECIPE_NAME}.yaml ./helm_context/selected-configuration.yaml
cp ./recipes/values-${RECIPE_NAME}.yaml ./helm_context/values.yaml
```

### 4. Launch Workload

```bash
export RECIPE_ROOT=$(pwd)
export WORKLOAD_NAME=nemo2504-qwen2-5-7b
cd ${RECIPE_ROOT}
helm install $WORKLOAD_NAME --set workload.image=$IMAGE --set queue=a4-high helm_context/
```

### 5. Check Workload Status

```bash
kubectl get pods | grep $WORKLOAD_NAME
kubectl logs <pod-name>
```

To continuously view logs, you can use the `-f` parameter:

```bash
kubectl logs -f <pod-name>
```

### 6. Uninstall Workload

```bash
helm uninstall $WORKLOAD_NAME
```

## Directory Structure

- `command.sh` - Demo script for running training
- `docker/` - Docker configurations
- `helm-context/` - Helm chart configurations
  - `templates/nemo-example.yaml` - Containers settings. Command to start the distributed training process in here.
  - `values.yaml` - Default configurations for the Helm chart, the number of gpus is here
  - `selected-configuration.yaml` - Model and training parameters.
- `recipes` - Containing different selected-configuration and values used in helm.
- `logs/` - Training log files
- `run.py` - torchDistributedTarget build into the container image
