# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Create one component at a time with explicit kubectl create/patch/scale steps."""

import argparse
import os
import sys

from common import Kube, Settings, name_check, required, router_args


def deployment(s, role, policy="kv"):
    worker = role in s.recipe["workers"]
    client = role == "client"
    labels = {"app": s.component(role), "agentx-fleet": s.fleet,
              "app.kubernetes.io/part-of": "agentx-routing"}
    if worker:
        labels["agentx-worker"] = "true"
    env = s.runtime_env() if not client else {
        "HF_HOME": "/artifacts/hf-cache", "PYTHONUNBUFFERED": "1",
        "AIPERF_DATASET_CONFIGURATION_TIMEOUT": "3600",
        "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT": "3600",
        "AIPERF_HTTP_CONNECTION_LIMIT": "2000",
        "MODEL_REVISION": s.revision,
    }
    volumes = [{"name": "model-cache", "persistentVolumeClaim": {"claimName": s.model_pvc, "readOnly": True}}]
    mounts = [{"name": "model-cache", "mountPath": "/model-cache", "readOnly": True}]
    annotations = {"gke-gcsfuse/volumes": "true", "gke-gcsfuse/memory-limit": "8Gi"}
    container = {"name": role, "image": s.client_image if client else s.image, "volumeMounts": mounts}
    pod = {
        "serviceAccountName": os.environ.get("SERVICE_ACCOUNT", "agentx"),
        "automountServiceAccountToken": False,
        "nodeSelector": {"cloud.google.com/gke-nodepool": s.cpu_pool if client else s.gpu_pool,
                         "kubernetes.io/arch": os.environ.get("CPU_ARCH", "amd64") if client else "arm64"},
        "tolerations": [{"key": "kubernetes.io/arch", "operator": "Exists", "effect": "NoSchedule"}],
        "containers": [container], "volumes": volumes,
        "terminationGracePeriodSeconds": 60,
    }
    if not client:
        pod["tolerations"].append({"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"})
    pull_secret = os.environ.get("IMAGE_PULL_SECRET")
    if pull_secret:
        pod["imagePullSecrets"] = [{"name": name_check(pull_secret)}]
    if worker:
        config = s.recipe["workers"][role]
        args = [a.replace("${MODEL_PATH}", s.model_path).replace("${MODEL_ID}", s.model) for a in config["args"]]
        container.update(command=["python3", "-m", "dynamo.sglang"], args=args)
        env.update(config["env"])
        container["resources"] = {"requests": {"cpu": "8", "memory": "128Gi"}, "limits": {"nvidia.com/gpu": "4"}}
        container["securityContext"] = {"runAsUser": 0, "capabilities": {"add": ["IPC_LOCK"]}}
        probe = {"httpGet": {"path": "/live", "port": 9090}, "periodSeconds": 20, "timeoutSeconds": 10}
        container["startupProbe"] = {**probe, "failureThreshold": 720}
        container["readinessProbe"] = {**probe, "failureThreshold": 3}
        volumes.append({"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "250Gi"}})
        mounts.append({"name": "shm", "mountPath": "/dev/shm"})
        claim = name_check(required("RDMA_CLAIM_TEMPLATE")) if s.arch == "agg" else f"{s.fleet}-channel"
        pod["resourceClaims"] = [{"name": "transport", "resourceClaimTemplateName": claim}]
        container["resources"]["claims"] = [{"name": "transport"}]
    elif client:
        container["command"] = ["sleep", "infinity"]
        volumes.append({"name": "artifacts", "persistentVolumeClaim": {"claimName": s.artifact_pvc}})
        mounts.append({"name": "artifacts", "mountPath": "/artifacts"})
        container["resources"] = {"requests": {"cpu": os.environ.get("CLIENT_CPUS", "32"),
                                               "memory": os.environ.get("CLIENT_MEMORY", "128Gi")}}
    else:
        container.update(command=["python3", "-m", "dynamo.frontend"], args=router_args(policy))
        container["resources"] = {"requests": {"cpu": "2", "memory": "8Gi"}}
        container["readinessProbe"] = {"tcpSocket": {"port": 8000}, "periodSeconds": 10}
    container["env"] = [{"name": k, "value": v} for k, v in env.items()]
    if client and os.environ.get("HF_SECRET"):
        container["env"].append({"name": "HF_TOKEN", "valueFrom": {"secretKeyRef": {
            "name": name_check(os.environ["HF_SECRET"]), "key": "HF_TOKEN"}}})
    return {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": s.component(role), "labels": labels},
            "spec": {"replicas": s.recipe["workers"][role]["replicas"] if worker else 1,
                     "strategy": {"type": "Recreate", "rollingUpdate": None}, "progressDeadlineSeconds": 14400,
                     "selector": {"matchLabels": {"app": s.component(role)}},
                     "template": {"metadata": {"labels": labels, "annotations": annotations}, "spec": pod}}}


def create_deployment(k, s, role, policy):
    obj = deployment(s, role, policy)
    if role == "client":
        # This pod only checks reachability and reads retained artifacts. The
        # benchmark Job reserves CLIENT_CPUS / CLIENT_MEMORY separately.
        obj["spec"]["template"]["spec"]["containers"][0]["resources"] = {
            "requests": {"cpu": "1", "memory": "2Gi"}}
    replicas = obj["spec"].pop("replicas")
    # Fail if a deployment already exists; never overwrite someone else's fleet.
    k.run("create", "deployment", s.component(role), "--image", obj["spec"]["template"]["spec"]["containers"][0]["image"], "--replicas=0")
    obj.pop("apiVersion")
    obj.pop("kind")
    k.run("patch", "deployment", s.component(role), "--type=merge", "--patch-file=/dev/stdin", payload=obj)
    k.run("scale", f"deployment/{s.component(role)}", f"--replicas={replicas}")


def domain(s):
    return {"apiVersion": "resource.nvidia.com/v1beta1", "kind": "ComputeDomain",
            "metadata": {"name": f"{s.fleet}-domain", "namespace": s.namespace},
            "spec": {"numNodes": 16, "channel": {"resourceClaimTemplate": {"name": f"{s.fleet}-channel"}}}}


def rdma(s):
    return {"apiVersion": "resource.k8s.io/v1", "kind": "ResourceClaimTemplate",
            "metadata": {"name": name_check(required("RDMA_CLAIM_TEMPLATE")), "namespace": s.namespace},
            "spec": {"spec": {"devices": {"requests": [{"name": "rdma", "exactly": {
                "deviceClassName": "mrdma.google.com", "allocationMode": "ExactCount", "count": 8}}]}}}}


def wait_workers(k, s):
    for role in s.recipe["workers"]:
        k.run("rollout", "status", f"deployment/{s.component(role)}", "--timeout=14400s")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--architecture", choices=["agg", "disagg"], required=True)
    p.add_argument("step", choices=["rdma", "domain", "worker", "prefill", "decode", "frontend", "service", "client", "wait", "status"])
    p.add_argument("--policy", choices=["kv", "rr"], default="kv")
    p.add_argument("--dry-run", action="store_true", help="Print commands and patches; do not contact Kubernetes")
    a = p.parse_args()
    s = Settings(a.architecture)
    k = Kube(s, a.dry_run)
    if a.step in ("worker", "prefill", "decode") and a.step not in s.recipe["workers"]:
        p.error(f"{a.step} is not a component of {s.arch}")
    if a.step == "rdma":
        if s.arch != "agg":
            p.error("D88 uses the ComputeDomain channel; create it with disagg/deploy.sh domain")
        k.run("create", "-f", "-", payload=rdma(s))
    elif a.step == "domain":
        if s.arch != "disagg":
            p.error("Only disagg needs a ComputeDomain")
        k.run("create", "-f", "-", payload=domain(s))
    elif a.step == "service":
        k.run("expose", "deployment", s.component("frontend"), "--port=8000", "--target-port=8000", "--type=ClusterIP")
    elif a.step == "wait":
        wait_workers(k, s)
        for role in ("frontend", "client"):
            k.run("rollout", "status", f"deployment/{s.component(role)}", "--timeout=1800s")
    elif a.step == "status":
        k.run("get", "deployments,pods,services", "-l", f"agentx-fleet={s.fleet}", "-o", "wide")
    else:
        create_deployment(k, s, a.step, a.policy)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError) as e:
        sys.exit(str(e))
