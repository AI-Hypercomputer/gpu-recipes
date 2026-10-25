# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Small, dependency-free helpers shared by the two GKE recipes."""

import json
import os
import re
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A4XMAX_ROOT = (ROOT / "../../../..").resolve()
RECIPE_DIRS = {
    "agg": A4XMAX_ROOT / "aggregated-serving/dynamo/sglang/n3u-kv-vs-rr",
    "disagg": ROOT,
}


def required(name):
    value = os.environ.get(name, "")
    if not value or "YOUR_" in value or "REPLACE_WITH" in value:
        raise ValueError(f"Set {name} in env.sh before running this command")
    return value


def name_check(value):
    if not re.fullmatch(r"[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?", value):
        raise ValueError(f"Expected a Kubernetes DNS label, got {value!r}")
    return value


class Settings:
    def __init__(self, architecture):
        self.arch = architecture
        self.recipe = json.loads((RECIPE_DIRS[architecture] / "recipe.json").read_text())
        self.context = required("KUBE_CONTEXT")
        self.namespace = name_check(required("NAMESPACE"))
        self.fleet = name_check(os.environ.get("FLEET", f"agentx-{architecture}"))
        # Keep room for component names and benchmark job suffixes.
        if len(self.fleet) > 35:
            raise ValueError("FLEET must be at most 35 characters")
        self.image = required("SERVING_IMAGE")
        self.client_image = required("CLIENT_IMAGE")
        for value in (self.image, self.client_image):
            if not re.fullmatch(r"[^\s]+@sha256:[a-f0-9]{64}", value):
                raise ValueError("SERVING_IMAGE and CLIENT_IMAGE must use @sha256 digests")
        self.model = required("MODEL_ID")
        self.model_path = required("MODEL_PATH")
        if not self.model_path.startswith("/model-cache/"):
            raise ValueError("MODEL_PATH must be under /model-cache/")
        self.revision = required("MODEL_REVISION")
        self.tokenizer = os.environ.get("TOKENIZER", self.model)
        self.tokenizer_revision = required("TOKENIZER_REVISION")
        for revision in (self.revision, self.tokenizer_revision):
            if not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError("Model and tokenizer revisions must be full 40-character SHAs")
        if self.tokenizer == "builtin" or self.tokenizer.startswith("/"):
            raise ValueError("TOKENIZER must be a Hugging Face repo ID with a pinned revision")
        self.model_pvc = name_check(required("MODEL_PVC"))
        self.artifact_pvc = name_check(required("ARTIFACT_PVC"))
        self.gpu_pool = required("GPU_NODEPOOL")
        self.cpu_pool = required("CPU_NODEPOOL")
        if self.gpu_pool == self.cpu_pool:
            raise ValueError("Use a separate CPU_NODEPOOL for the benchmark client")
        self.etcd = required("ETCD_ENDPOINTS")
        self.nats = required("NATS_SERVER")
        self.gpus = 4 * sum(w["replicas"] for w in self.recipe["workers"].values())

    def component(self, role):
        return f"{self.fleet}-{role}"

    @property
    def worker_selector(self):
        return f"agentx-fleet={self.fleet},agentx-worker=true"

    def runtime_env(self):
        return {
            "ETCD_ENDPOINTS": self.etcd,
            "NATS_SERVER": self.nats,
            "DYN_NAMESPACE": self.fleet,
            "DYN_DISCOVERY_BACKEND": "etcd",
            "DYN_SYSTEM_PORT": "9090",
            "HF_HOME": "/tmp/hf",
            "HF_HUB_OFFLINE": "1",
            "HF_MODULES_CACHE": "/tmp/hf_modules",
        }


class Kube:
    def __init__(self, settings, dry_run=False):
        self.base = ["kubectl", "--context", settings.context, "-n", settings.namespace]
        self.dry_run = dry_run

    def run(self, *args, payload=None, capture=False, check=True):
        cmd = self.base + list(args)
        data = json.dumps(payload) if payload is not None else None
        print("+ " + shlex.join(cmd), flush=True)
        if self.dry_run:
            if data:
                print(json.dumps(payload, indent=2))
            return ""
        result = subprocess.run(cmd, input=data, text=True, capture_output=capture, check=check)
        return result.stdout if capture else result.returncode

    def get(self, *args):
        return json.loads(self.run("get", *args, "-o", "json", capture=True))

    def exec_client(self, s, *args, capture=False):
        return self.run("exec", f"deployment/{s.component('client')}", "-c", "client", "--", *args, capture=capture)


def router_args(policy, *, temperature=None, load_scale=None, credit=None, decay=None):
    if policy == "rr":
        if any(v is not None for v in (temperature, load_scale, credit, decay)):
            raise ValueError("KV tuning arguments cannot be applied to RR")
        return ["--router-mode", "round-robin", "--request-plane", "nats"]
    args = ["--router-mode", "kv", "--router-temperature", str(temperature or 0),
            "--router-queue-policy", "fcfs", "--request-plane", "nats"]
    for flag, value in [("prefill-load-scale", load_scale), ("kv-overlap-score-credit", credit),
                        ("kv-overlap-score-credit-decay", decay)]:
        if value is not None:
            if value < 0:
                raise ValueError(f"{flag} must be nonnegative")
            args += [f"--router-{flag}", str(value)]
    if temperature is not None and temperature < 0:
        raise ValueError("temperature must be nonnegative")
    return args
