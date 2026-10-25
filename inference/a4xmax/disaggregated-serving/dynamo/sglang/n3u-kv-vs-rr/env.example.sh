#!/usr/bin/env bash
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Copy to env.sh and edit. Scripts never change your current kubectl context.
export PROJECT_ID=YOUR_PROJECT
export CLUSTER_NAME=YOUR_CLUSTER
export CLUSTER_REGION=YOUR_REGION
export KUBE_CONTEXT=gke_YOUR_PROJECT_YOUR_REGION_YOUR_CLUSTER
export NAMESPACE=agentx-routing
export GPU_NODEPOOL=YOUR_A4XMAX_POOL
export CPU_NODEPOOL=YOUR_CPU_POOL
export CPU_ARCH=amd64
export SERVICE_ACCOUNT=agentx
export MODEL_PVC=model-cache
export ARTIFACT_PVC=agentx-artifacts
export MODEL_ID=nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4
# Set both to the same full Hugging Face commit SHA before staging weights.
export MODEL_REVISION=REPLACE_WITH_40_CHARACTER_COMMIT_SHA
export TOKENIZER_REVISION=$MODEL_REVISION
export TOKENIZER=$MODEL_ID
export MODEL_PATH=/model-cache/nemotron/$MODEL_REVISION
export SERVING_IMAGE=YOUR_REGISTRY/agentx-serving@sha256:REPLACE_WITH_DIGEST
export CLIENT_IMAGE=YOUR_REGISTRY/agentx-client@sha256:REPLACE_WITH_DIGEST
# Existing etcd and NATS endpoints, or the services created in the quickstart.
export ETCD_ENDPOINTS=http://agentx-etcd.agentx-routing.svc.cluster.local:2379
export NATS_SERVER=nats://agentx-nats.agentx-routing.svc.cluster.local:4222
export HF_SECRET=hf-token-secret
# Leave empty for images your nodes can pull without an extra registry secret.
export IMAGE_PULL_SECRET=
export RDMA_CLAIM_TEMPLATE=agentx-mrdma
export CLIENT_CPUS=32
export CLIENT_MEMORY=128Gi
export AIPERF_WORKERS=200
export AIPERF_RECORD_PROCESSORS=8
