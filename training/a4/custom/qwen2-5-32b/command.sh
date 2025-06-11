# 1. Connect to cluster
export PROJECT=your-project-id
export REGION=us-east4
export ZONE=us-east4-c
export CLUSTER_NAME=your-gek-cluster

gcloud config set project ${PROJECT}
gcloud config set compute/zone ${ZONE}
gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION

# 2. Build the docker images and push to gcr
export RECIPE_ROOT=$(pwd)
export IMAGE=us-central1-docker.pkg.dev/${PROJECT}/${USER}/nemo1:25.04

# Rootless docker
sudo usermod -aG docker $USER
newgrp docker

# The dockerfile needs to copy files in the recipe root
cd ${RECIPE_ROOT}
docker build -t $IMAGE -f docker/Dockerfile .
docker push $IMAGE

# 3. Select and copy recipe
export RECIPE_ROOT=$(pwd)
cd ${RECIPE_ROOT}
export RECIPE_NAME=qwen2-5-32b-16gpu-sptrue-tp2-pp2-cp1-vp32-gbs512-mbs1
cp ./recipes/${RECIPE_NAME}.yaml ./helm_context/selected-configuration.yaml
cp ./recipes/values-${RECIPE_NAME}.yaml ./helm_context/values.yaml

# 4. Helm install the workload and get the log
export RECIPE_ROOT=$(pwd)
export WORKLOAD_NAME=nemo2504-qwen25-32b-16gpu-sq1-tp2-pp2-vp32-gbs512
cd ${RECIPE_ROOT}
helm install $WORKLOAD_NAME --set workload.image=$IMAGE helm_context/

# List Kubernetes pods and filter for those containing the WORKLOAD_NAME
kubectl get pods | grep $WORKLOAD_NAME

kubectl logs -f nemo2504-qwen25-32b-16gpu-sq1-tp2-pp2-vp32-gbs512-0-89fpt

# Fetch the full logs from a specific pod and save them to a file in the ./logs directory
kubectl logs nemo2504-qwen25-32b-16gpu-sq1-tp2-pp2-vp32-gbs512-0-mn96v > ./logs/nemo2504-qwen25-32b-16gpu-sq1-tp2-pp2-vp32-gbs512-full-logs

# Uninstall the Helm release associated with the WORKLOAD_NAME
helm uninstall $WORKLOAD_NAME
