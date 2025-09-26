#!/bin/bash
REPLICA_COUNT=2

helm install ray-cluster ../qwen2.5-1.5b \
  --set additionalWorkerGroups.worker-grp-0.replicas=$REPLICA_COUNT