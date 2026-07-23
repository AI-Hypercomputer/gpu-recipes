# Disaggregated Inference with llm-d on GKE (No helm version)

This document outlines the steps to deploy an llm-d inference server on GKE without using helm.

## 1. Environment Setup (One-Time)

1.1. If using A3U or A4, create an RDMA cluster following [this guide](https://docs.cloud.google.com/ai-hypercomputer/docs/create/gke-ai-hypercompute-custom#create-with-rdma); if using A4X, create an RDMA cluster following [this guide](https://docs.cloud.google.com/ai-hypercomputer/docs/create/gke-ai-hypercompute-custom-a4x).

1.2. Clone the repository

```bash
git clone [https://github.com/ai-hypercomputer/gpu-recipes.git](https://github.com/ai-hypercomputer/gpu-recipes.git)
cd gpu-recipes/inference/llm-d
````

1.3. Configure environment variables

``` bash
export PROJECT_ID=<PROJECT_ID>
export CLUSTER_REGION=<REGION_of_your_cluster>
export CLUSTER_NAME=<YOUR_GKE_CLUSTER_NAME>
export NAMESPACE=<YOUR_k8s_NAMESPACE>
export HF_TOKEN=<YOUR_HF_TOKEN>
```

1.4. Connect to your GKE cluster

``` bash
gcloud container clusters get-credentials $CLUSTER_NAME --region $CLUSTER_REGION

kubectl create namespace ${NAMESPACE}

kubectl config set-context --current --namespace=$NAMESPACE
```

1.5. Create secrets

``` bash
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${NAMESPACE}
```

## 2\. Set up GKE Gateway

2.1. [Enable Gateway API in your Cluster](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/deploying-gateways#enable-gateway)

``` bash
gcloud container clusters update $CLUSTER_NAME \
    --location=$CLUSTER_REGION \
    --gateway-api=standard
```

2.2. [Verify your cluster](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/deploying-gateways#verify-internal)

``` bash
gcloud container clusters describe $CLUSTER_NAME \
  --location=$CLUSTER_REGION \
  --format json
```

The output is similar to the following:

``` json
"networkConfig": {
  ...
  "gatewayApiConfig": {
    "channel": "CHANNEL_STANDARD"
  },
  ...
},
```

Confirm the `GatewayClasses` are installed in your cluster:

``` bash
kubectl get gatewayclass
```

The output is similar to the following:

``` 
NAME                             CONTROLLER                  ACCEPTED   AGE
gke-l7-global-external-managed   networking.gke.io/gateway   True       16h
gke-l7-regional-external-managed networking.gke.io/gateway   True       16h
gke-l7-gxlb                      networking.gke.io/gateway   True       16h
gke-l7-rilb                      networking.gke.io/gateway   True       16h
```

2.3. [Configure a proxy-only subnet](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/deploying-gateways#configure_a_proxy-only_subnet)

``` bash
export SUBNET_NAME=<YOUR_NAME_OF_THE_PROXY_ONLY_SUBNET> (e.g. gateway-proxy-only-subnet)
export VPC_NETWORK_NAME=<YOUR_NAME_OF_THE_VPC_NETWORK_IN_WHICH_YOU_CREATE_THE_SUBNET> (e.g. default)
export CIDR_RANGE=<YOUR_PRIMARY_IP_ADDRESS_RANGE_OF_THE_SUBNET> (e.g. 10.1.1.0/24)

gcloud compute networks subnets create $SUBNET_NAME \
    --purpose=REGIONAL_MANAGED_PROXY \
    --role=ACTIVE \
    --region=$CLUSTER_REGION \
    --network=$VPC_NETWORK_NAME \
    --range=$CIDR_RANGE
```

2.4. Verify your proxy-only subnet:

``` bash
gcloud compute networks subnets describe $SUBNET_NAME \
    --region=$CLUSTER_REGION
```

The output is similar to the following:

``` 
...
gatewayAddress: 10.1.1.1
ipCidrRange: 10.1.1.0/24
kind: compute#subnetwork
name: proxy-subnet
network: [https://www.googleapis.com/compute/v1/projects/PROJECT_NAME/global/networks/default](https://www.googleapis.com/compute/v1/projects/PROJECT_NAME/global/networks/default)
privateIpGoogleAccess: false
privateIpv6GoogleAccess: DISABLE_GOOGLE_ACCESS
purpose: REGIONAL_MANAGED_PROXY
region: [https://www.googleapis.com/compute/v1/projects/PROJECT_NAME/regions/REGION](https://www.googleapis.com/compute/v1/projects/PROJECT_NAME/regions/REGION)
role: ACTIVE
selfLink: [https://www.googleapis.com/compute/v1/projects/PROJECT_NAME/regions/REGION/subnetworks/proxy-subnet](https://www.googleapis.com/compute/v1/projects/PROJECT_NAME/regions/REGION/subnetworks/proxy-subnet)
state: READY
```

2.5. [Install needed Custom Resource Definitions (CRDs) in your GKE cluster](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/deploy-gke-inference-gateway#prepare-environment):

  * For GKE versions `1.34.0-gke.1626000` or later, install only the alpha `InferenceObjective` CRD:

<!-- end list -->

``` bash
kubectl apply -f [https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/v1.0.0/config/crd/bases/inference.networking.x-k8s.io_inferenceobjectives.yaml](https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/v1.0.0/config/crd/bases/inference.networking.x-k8s.io_inferenceobjectives.yaml)
```

  * For GKE versions earlier than `1.34.0-gke.1626000`, install both the `v1 InferencePool` and alpha `InferenceObjective` CRDs:

<!-- end list -->

``` bash
kubectl apply -f [https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/v1.0.0/manifests.yaml](https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/v1.0.0/manifests.yaml)
```

2.6. [Deploy GKE gateway](https://github.com/llm-d/llm-d/blob/main/guides/recipes/gateway/README.md)

``` bash
kubectl apply -f gateway.yaml -n ${NAMESPACE}
```

Clone the repository:

``` bash
git clone [https://github.com/llm-d/llm-d.git](https://github.com/llm-d/llm-d.git)

cd guides/recipes/gateway
```

Deploy a gateway suitable for GKE

``` bash
kubectl apply -k ./gke-l7-regional-external-managed -n ${NAMESPACE}
```

2.7. Deploy the InferencePool

``` bash
kubectl apply -f inference-pool.yaml -n ${NAMESPACE}
```

## 3\. Deploy the model

Install LeaderWorkerSet:

``` bash
VERSION=v0.8.0
kubectl apply --server-side -f [https://github.com/kubernetes-sigs/lws/releases/download/$VERSION/manifests.yaml](https://github.com/kubernetes-sigs/lws/releases/download/$VERSION/manifests.yaml)
```

To wait for LeaderWorkerSet to be fully available, run:

``` bash
kubectl wait deploy/lws-controller-manager -n lws-system --for=condition=available --timeout=5m
```

### H200:

``` bash
kubectl apply -f a3ultra/disaggregated-serving.yaml -n ${NAMESPACE}
```

### B200:

``` bash
kubectl apply -f a4/disaggregated-serving.yaml -n ${NAMESPACE}
```

## 4\. Verify the deployment

``` bash
export GATEWAY_IP=$(kubectl get gateway llm-d-inference-gateway -n default -o jsonpath='{.status.addresses[0].value}')

curl http://$GATEWAY_IP/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-0528",
    "messages": [
      {"role": "user", "content": "Explain quantum computing in one sentence."}
    ],
    "max_tokens": 50
  }'

```
