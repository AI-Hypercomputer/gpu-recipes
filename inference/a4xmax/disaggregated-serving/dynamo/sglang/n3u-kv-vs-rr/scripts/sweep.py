# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Sweep session concurrency for KV and RR on one dedicated serving fleet."""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from common import Kube, Settings, name_check, router_args
from deploy import deployment, wait_workers

LADDERS = {"agg": "48,96,192", "disagg": "72,96,192,480"}


def positive_ints(value):
    result = [int(x) for x in value.split(",")]
    if not result or any(x <= 0 for x in result) or len(set(result)) != len(result):
        raise ValueError("Concurrency values must be distinct positive integers")
    return result


def cells(a):
    policies = a.policies.split(",")
    if not policies or any(x not in ("kv", "rr") for x in policies) or len(set(policies)) != len(policies):
        raise ValueError("--policies must be kv,rr, rr,kv, kv, or rr")
    keys = ["load_scale", "credit", "decay", "temperature"]
    if any(getattr(a, k) is not None for k in keys) and policies != ["kv"]:
        raise ValueError("Use --policies kv with tuning flags; run baseline RR separately")
    flags = {key: getattr(a, key) for key in keys}
    if any(v is not None and (not math.isfinite(v) or v < 0) for v in flags.values()):
        raise ValueError("Router flag values must be finite and nonnegative")
    if a.repeat < 1 or a.duration < 1 or a.job_timeout < a.duration + 60:
        raise ValueError("Check repeat, duration, and job-timeout (must exceed duration + grace)")
    if a.duration < 900 and not a.smoke:
        raise ValueError("Duration below 900 seconds requires --smoke")
    conc = positive_ints(a.concurrencies or ("2" if a.smoke else LADDERS[a.architecture]))
    for c in conc:
        for repeat in range(1, a.repeat + 1):
            # Reverse A/B order on the second repeat to reduce order confounding.
            order = policies if repeat % 2 else list(reversed(policies))
            for policy in order:
                yield {"id": f"{policy}-c{c}-r{repeat}", "policy": policy,
                       "concurrency": c, "repeat": repeat, "router_args": router_args(policy, **flags)}


def patch_frontend(k, s, cell):
    k.run("patch", "deployment", s.component("frontend"), "--type=strategic", "--patch-file=/dev/stdin",
          payload={"spec": {"template": {"spec": {"containers": [{"name": "frontend",
                    "command": ["python3", "-m", "dynamo.frontend"], "args": cell["router_args"]}]}}}})
    # Restart even if the arguments are unchanged: old index state must not persist
    # after a worker reset, and every cell follows the same startup procedure.
    k.run("rollout", "restart", f"deployment/{s.component('frontend')}")
    k.run("rollout", "status", f"deployment/{s.component('frontend')}", "--timeout=1800s")


def reset_workers(k, s):
    for role in s.recipe["workers"]:
        k.run("scale", f"deployment/{s.component(role)}", "--replicas=0")
    k.run("wait", "--for=delete", "pod", "-l", s.worker_selector, "--timeout=1800s")
    for role, config in s.recipe["workers"].items():
        k.run("scale", f"deployment/{s.component(role)}", f"--replicas={config['replicas']}")
    wait_workers(k, s)


HTTP_CHECK = """
import json, sys, time, urllib.request
url, model, *metrics = sys.argv[1:]
deadline = time.monotonic() + 600
while True:
    try:
        with urllib.request.urlopen(url + '/v1/models', timeout=20) as response:
            assert any(x['id'] == model for x in json.load(response)['data']), 'model not registered'
        for endpoint in metrics:
            with urllib.request.urlopen(endpoint, timeout=20) as response:
                assert 'sglang' in response.read().decode(), 'engine metrics absent at ' + endpoint
        break
    except Exception:
        if time.monotonic() > deadline: raise
        time.sleep(5)
print('Model registration and all engine metric endpoints are reachable.')
"""

VERSIONS = """
import importlib.metadata as m, json
expected = {'ai-dynamo':'1.4.2','sglang':'0.5.16','flashinfer-python':'0.6.18'}
actual = {k:m.version(k) for k in expected}
print(json.dumps(actual))
assert actual == expected, (actual, expected)
"""


def capture_fleet(k, s, folder):
    workers = k.get("pods", "-l", s.worker_selector)["items"]
    expected = sum(x["replicas"] for x in s.recipe["workers"].values())
    if len(workers) != expected:
        raise ValueError(f"Expected {expected} worker Pods, found {len(workers)}")
    metrics = []
    role_counts = {role: 0 for role in s.recipe["workers"]}
    for pod in workers:
        if not any(c["type"] == "Ready" and c["status"] == "True" for c in pod["status"].get("conditions", [])):
            raise ValueError(f"Worker is not Ready: {pod['metadata']['name']}")
        metrics.append(f"http://{pod['status']['podIP']}:9090/metrics")
        container = next(c for c in pod["spec"]["containers"] if c["name"] in s.recipe["workers"])
        worker_container = container["name"]
        role_counts[worker_container] += 1
        expected_container = deployment(s, worker_container)["spec"]["template"]["spec"]["containers"][0]
        for field in ("image", "command", "args"):
            if container[field] != expected_container[field]:
                raise ValueError(f"Live worker {field} differs from this recipe: {pod['metadata']['name']}")
        gpu_limit = container.get("resources", {}).get("limits", {}).get("nvidia.com/gpu")
        if gpu_limit is None or int(gpu_limit) != 4:
            raise ValueError(f"Every worker must request four GPUs: {pod['metadata']['name']}")
        versions = k.run("exec", pod["metadata"]["name"], "-c", worker_container,
                         "--", "python3", "-c", VERSIONS, capture=True)
        (folder / f"{pod['metadata']['name']}-versions.json").write_text(versions)
    if role_counts != {role: value["replicas"] for role, value in s.recipe["workers"].items()}:
        raise ValueError(f"Unexpected prefill/decode worker counts: {role_counts}")
    (folder / "worker-pods.json").write_text(json.dumps(workers, indent=2) + "\n")
    (folder / "deployments.json").write_text(json.dumps(k.get("deployments", "-l", f"agentx-fleet={s.fleet}"), indent=2) + "\n")
    return metrics


def point_job(s, a, cell, index, metrics):
    # Reuse the client pod shape: same image, resources, node pool, and caches.
    template = deployment(s, "client")["spec"]["template"]
    template["metadata"]["labels"]["app"] = f"{s.fleet}-benchmark"
    template["spec"]["restartPolicy"] = "Never"
    container = template["spec"]["containers"][0]
    container["name"] = "perf"
    container["command"] = ["python3", "/workspace/scripts/run_point.py"]
    path = f"/artifacts/{a.run_id}/{cell['id']}"
    args = ["--model", s.model, "--tokenizer", s.tokenizer, "--tokenizer-revision", s.tokenizer_revision,
            "--url", f"http://{s.component('frontend')}:8000", "--artifact-dir", path,
            "--concurrency", str(cell["concurrency"]), "--gpus", str(s.gpus),
            "--duration", str(a.duration), "--seed", str(a.seed), "--request-timeout", str(a.request_timeout),
            "--workers", os.environ.get("AIPERF_WORKERS", "200"),
            "--record-processors", os.environ.get("AIPERF_RECORD_PROCESSORS", "8")]
    for url in metrics:
        args += ["--metrics-url", url]
    if a.smoke:
        args += ["--smoke"]
    container["args"] = args
    # The helper stays small while each benchmark Job reserves the full client CPU.
    return {"apiVersion": "batch/v1", "kind": "Job", "metadata": {
                "name": name_check(f"{a.run_id}-{index:03}"), "labels": {"agentx-fleet": s.fleet}},
            "spec": {"backoffLimit": 0, "activeDeadlineSeconds": a.job_timeout, "template": template}}


def wait_job(k, name, timeout):
    deadline = time.monotonic() + timeout + 120
    while time.monotonic() < deadline:
        job = k.get("job", name)
        conditions = job.get("status", {}).get("conditions", [])
        if any(c["type"] == "Complete" and c["status"] == "True" for c in conditions):
            return True
        if any(c["type"] == "Failed" and c["status"] == "True" for c in conditions):
            return False
        time.sleep(15)
    raise TimeoutError(f"Job {name} did not finish before its deadline; inspect it before starting another sweep")


def collect(k, s, a, cell, name, folder):
    # The helper uses the same PVC, so Job termination does not lose artifacts.
    log = k.run("logs", f"job/{name}", "-c", "perf", capture=True, check=False)
    (folder / "job.log").write_text(log or "")
    for file in ("status.json", "summary.json", "command.json", "packages.json"):
        try:
            text = k.exec_client(s, "cat", f"/artifacts/{a.run_id}/{cell['id']}/{file}", capture=True)
            (folder / file).write_text(text)
        except subprocess.CalledProcessError:
            print(f"Missing {file}; retained Job logs and cluster artifacts", file=sys.stderr)
    (folder / "job-final.json").write_text(json.dumps(k.get("job", name), indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--architecture", choices=["agg", "disagg"], required=True)
    p.add_argument("--concurrencies", help="Comma list of live session counts")
    p.add_argument("--policies", default="kv,rr")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--duration", type=int, default=3600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--request-timeout", type=int, default=1200)
    p.add_argument("--job-timeout", type=int, default=28800, help="Whole Job deadline, including dataset setup and warmup")
    p.add_argument("--cache-reset", choices=["restart", "reuse"], default="restart")
    p.add_argument("--run-id", help="New DNS label; never reuses/overwrites a campaign")
    p.add_argument("--output", type=Path, default=Path("artifacts"))
    for name in ("load-scale", "credit", "decay", "temperature"):
        p.add_argument(f"--{name}", type=float, help="Optional KV setting for this campaign; requires --policies kv")
    p.add_argument("--smoke", action="store_true", help="Explicitly invalid scenario smoke; use --duration 120")
    p.add_argument("--dry-run", action="store_true", help="Print the plan without cluster calls or files")
    a = p.parse_args()
    if a.smoke and a.duration == 3600:
        a.duration = 120
    s = Settings(a.architecture)
    a.run_id = name_check(a.run_id or f"{s.arch}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}")
    if len(a.run_id) > 55:
        p.error("run-id must be at most 55 characters")
    plan = list(cells(a))
    if not plan:
        p.error("No cells selected")
    k = Kube(s)
    info = {"architecture": s.arch, "fleet": s.fleet, "gpus": s.gpus,
            "run_id": a.run_id, "model": s.model, "model_revision": s.revision,
            "tokenizer": s.tokenizer, "tokenizer_revision": s.tokenizer_revision,
            "serving_image": s.image, "client_image": s.client_image, "recipe": s.recipe,
            "cache_reset": a.cache_reset, "duration_s": a.duration, "seed": a.seed,
            "smoke": a.smoke, "cells": plan}
    print(json.dumps({key: info[key] for key in ("run_id", "architecture", "gpus", "cache_reset", "cells")}, indent=2))
    print(f"{len(plan)} cells; {len(plan) * a.duration / 3600:g} profiling hours, plus setup, warmup, drain, and resets.")
    if a.dry_run:
        # A single example is enough: only policy/concurrency vary across cells.
        print(json.dumps(point_job(s, a, plan[0], 0, ["http://WORKER_POD_IP:9090/metrics"])
                         ["spec"]["template"]["spec"]["containers"][0]["args"], indent=2))
        return
    output = a.output / a.run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(info, indent=2) + "\n")
    # Refuse overlapping campaigns on the same fleet. A ConfigMap is an atomic
    # namespace-scoped lock; SIGKILL leaves it for the operator to inspect.
    lock = f"{s.fleet}-sweep-lock"
    k.run("create", "configmap", lock, f"--from-literal=run-id={a.run_id}")
    active_job = None
    try:
        active = k.get("jobs", "-l", f"agentx-fleet={s.fleet}")["items"]
        if any(not any(c["status"] == "True" and c["type"] in ("Complete", "Failed")
                       for c in j.get("status", {}).get("conditions", [])) for j in active):
            raise ValueError("An unfinished benchmark Job is already using this fleet")
        k.run("rollout", "status", f"deployment/{s.component('client')}", "--timeout=1800s")
        # Verify storage access and refuse a reused artifact path before changing serving state.
        k.exec_client(s, "python3", "-c", "import pathlib,sys; pathlib.Path(sys.argv[1]).mkdir(exist_ok=False)", f"/artifacts/{a.run_id}")
        for i, cell in enumerate(plan):
            folder = output / cell["id"]
            folder.mkdir()
            (folder / "cell.json").write_text(json.dumps(cell, indent=2) + "\n")
            if a.cache_reset == "restart":
                reset_workers(k, s)
            else:
                wait_workers(k, s)
            patch_frontend(k, s, cell)
            urls = capture_fleet(k, s, folder)
            k.exec_client(s, "python3", "-c", HTTP_CHECK, f"http://{s.component('frontend')}:8000", s.model, *urls)
            job = point_job(s, a, cell, i, urls)
            (folder / "job.json").write_text(json.dumps(job, indent=2) + "\n")
            active_job = job["metadata"]["name"]
            k.run("create", "-f", "-", payload=job)
            okay = wait_job(k, active_job, a.job_timeout)
            collect(k, s, a, cell, active_job, folder)
            active_job = None
            status_file = folder / "status.json"
            status = json.loads(status_file.read_text()) if status_file.exists() else {}
            if not okay or (not a.smoke and status.get("comparison_eligible") is not True):
                raise ValueError(f"Cell {cell['id']} failed validation. Inspect {folder}; sweep stopped.")
        print(f"Completed {len(plan)} cells. Summaries: {output}; raw exports: {s.artifact_pvc}:/artifacts/{a.run_id}")
    finally:
        if active_job is None:
            k.run("delete", "configmap", lock)
        else:
            print(f"Job {active_job} may still be running. Lock {lock} retained; inspect both before resuming.", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TimeoutError, subprocess.CalledProcessError) as e:
        sys.exit(str(e))
