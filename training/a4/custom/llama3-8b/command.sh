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

# 3. Copy recipes
export RECIPE_ROOT=$(pwd)
cd ${RECIPE_ROOT}
cp ./recipes/llama3-8b-8gpu-tp1-pp1-cp1-vp1-gbs128-mbs2-fp8.yaml ./helm_context/selected-configuration.yaml
cp ./recipes/values-llama3-8b-8gpu-tp1-pp1-cp1-vp1-gbs128-mbs2-fp8.yaml ./helm_context/values.yaml

# 4. Helm install the workload and get the log
export RECIPE_ROOT=$(pwd)
export WORKLOAD_NAME=llama3-70b-64gpu-sq1-tp2-pp4-cp2-vp5-gbs128-mbs1-fp8
cd ${RECIPE_ROOT}
helm install $WORKLOAD_NAME --set workload.image=$IMAGE helm_context/

# List Kubernetes pods and filter for those containing the WORKLOAD_NAME
kubectl get pods | grep $WORKLOAD_NAME

kubectl logs -f llama3-70b-64gpu-sq1-tp2-pp4-cp2-vp5-gbs128-mbs1-fp8-0-qdrr4

# Fetch the full logs from a specific pod and save them to a file in the ./logs directory
kubectl logs llama3-70b-64gpu-sq1-tp2-pp4-cp2-vp5-gbs128-mbs1-fp8-0-qdrr4 > ./logs/llama3-70b-64gpu-sq1-tp2-pp4-cp2-vp5-gbs128-mbs1-fp8-full-logs

# Uninstall the Helm release associated with the WORKLOAD_NAME
helm uninstall $WORKLOAD_NAME