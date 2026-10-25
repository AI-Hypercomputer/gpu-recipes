# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Offline checks for deployment boundaries, replay fidelity, and metric math."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import RECIPE_DIRS, Settings
from deploy import deployment, domain
from run_point import aiperf_command, parser
from run_point import main as run_point_main
from summarize import percentile, summarize
from sweep import cells, point_job

ENV = {
    "KUBE_CONTEXT": "test-cluster", "NAMESPACE": "test-agentx", "GPU_NODEPOOL": "gpu", "CPU_NODEPOOL": "cpu",
    "SERVING_IMAGE": "registry/serving@sha256:" + "1" * 64,
    "CLIENT_IMAGE": "registry/client@sha256:" + "2" * 64,
    "MODEL_ID": "test/model", "MODEL_PATH": "/model-cache/model",
    "MODEL_REVISION": "a" * 40, "TOKENIZER_REVISION": "a" * 40,
    "MODEL_PVC": "models", "ARTIFACT_PVC": "artifacts", "ETCD_ENDPOINTS": "http://etcd:2379",
    "NATS_SERVER": "nats://nats:4222", "RDMA_CLAIM_TEMPLATE": "test-rdma",
}


def sweep_args(**kwargs):
    values = {"architecture": "agg", "policies": "kv,rr", "concurrencies": "48,96", "repeat": 2,
                  "duration": 3600, "seed": 42, "job_timeout": 28800, "request_timeout": 1200,
                  "smoke": False, "load_scale": None, "credit": None, "decay": None, "temperature": None,
                  "run_id": "unit-test"}
    return argparse.Namespace(**(values | kwargs))


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, ENV, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_gpu_counts_admission_and_transport_are_architecture_specific(self):
        for arch, count in [("agg", 24), ("disagg", 64)]:
            s = Settings(arch)
            total = 0
            for role, expected_limit in ({"worker": 16} if arch == "agg" else {"prefill": 8, "decode": 64}).items():
                d = deployment(s, role)
                pod = d["spec"]["template"]["spec"]
                c = pod["containers"][0]
                total += int(c["resources"]["limits"]["nvidia.com/gpu"]) * d["spec"]["replicas"]
                self.assertEqual(int(c["args"][c["args"].index("--max-running-requests") + 1]), expected_limit)
                self.assertNotIn("--disable-radix-cache", c["args"])
                self.assertIn("--enable-metrics", c["args"])
                if arch == "disagg":
                    self.assertIn("mooncake", c["args"])
                    self.assertEqual(pod["resourceClaims"][0]["resourceClaimTemplateName"], domain(s)["spec"]["channel"]["resourceClaimTemplate"]["name"])
                else:
                    self.assertNotIn("--disaggregation-mode", c["args"])
            self.assertEqual(total, count)

    def test_rr_changes_only_frontend_arguments(self):
        s = Settings("agg")
        kv, rr = deployment(s, "frontend", "kv"), deployment(s, "frontend", "rr")
        kv_container = kv["spec"]["template"]["spec"]["containers"][0]
        rr_container = rr["spec"]["template"]["spec"]["containers"][0]
        self.assertIn("round-robin", rr_container.pop("args"))
        self.assertIn("kv", kv_container.pop("args"))
        self.assertEqual(kv, rr)

    def test_baseline_order_and_invalid_tuning(self):
        planned = list(cells(sweep_args()))
        self.assertEqual([(c["concurrency"], c["policy"]) for c in planned],
                         [(48, "kv"), (48, "rr"), (48, "rr"), (48, "kv"),
                          (96, "kv"), (96, "rr"), (96, "rr"), (96, "kv")])
        for changes in [{"credit": 1.5}, {"policies": "bogus"}, {"concurrencies": "0"},
                        {"concurrencies": "48,48"}, {"duration": 120},
                        {"policies": "kv", "decay": float("nan")}]:
            with self.assertRaises(ValueError):
                list(cells(sweep_args(**changes)))

    def test_job_preserves_replay_and_collects_every_engine(self):
        s = Settings("disagg")
        a = sweep_args(architecture="disagg", concurrencies="480", policies="kv", credit=1.5)
        cell = next(cells(a))
        urls = [f"http://10.0.0.{n}:9090/metrics" for n in range(1, 17)]
        job = point_job(s, a, cell, 0, urls)
        spec = job["spec"]
        self.assertEqual(spec["backoffLimit"], 0)
        c = spec["template"]["spec"]["containers"][0]
        parsed = parser().parse_args(c["args"])
        command = aiperf_command(parsed)
        self.assertEqual(parsed.gpus, 64)
        self.assertEqual(parsed.metrics_url, urls)
        self.assertIn("inferencex-agentx-mvp", command)
        self.assertIn("semianalysis_cc_traces_weka_062126_256k", command)
        for forbidden in ("--ignore-trace-delays", "--request-rate", "--unsafe-override"):
            self.assertNotIn(forbidden, command)
        self.assertEqual(command[command.index("--benchmark-duration") + 1], "3600")
        self.assertEqual(command[command.index("--tokenizer-revision") + 1], "a" * 40)

    def test_short_smoke_keeps_scenario_and_is_explicitly_unsafe(self):
        s = Settings("agg")
        a = sweep_args(duration=120, smoke=True, concurrencies="2")
        c = point_job(s, a, next(cells(a)), 0, ["http://engine/metrics"])["spec"]["template"]["spec"]["containers"][0]
        cmd = aiperf_command(parser().parse_args(c["args"]))
        self.assertIn("--scenario", cmd)
        self.assertIn("--unsafe-override", cmd)

    def test_dry_runs_do_not_contact_cluster(self):
        with tempfile.TemporaryDirectory() as directory:
            trap = Path(directory) / "kubectl"
            marker = Path(directory) / "called"
            trap.write_text(f"#!/bin/sh\ntouch '{marker}'\nexit 99\n")
            trap.chmod(0o755)
            env = dict(os.environ, PATH=f"{directory}:/usr/bin:/bin")
            for arch in ("agg", "disagg"):
                for file, args in [("sweep.sh", ["--dry-run"]), ("deploy.sh", ["frontend", "--dry-run"])]:
                    result = subprocess.run(["bash", str(RECIPE_DIRS[arch] / file), *args], env=env, cwd=directory, text=True, capture_output=True, check=False)
                    self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(marker.exists())
            self.assertFalse((Path(directory) / "artifacts").exists())

    def test_summary_filters_warmup_errors_and_inverts_request_ratios(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            data = {"metadata": {"submission_valid": True},
                    "request_count": {"unit": "requests", "avg": 3},
                    "request_throughput": {"unit": "requests/sec", "avg": 1},
                    "input_token_throughput": {"unit": "tokens/sec", "avg": 800},
                    "output_token_throughput": {"unit": "tokens/sec", "avg": 200}}
            (folder / "profile_export_aiperf.json").write_text(json.dumps(data))
            def record(ms, tokens, **meta):
                return {"metadata": {"benchmark_phase": "profiling", **meta}, "metrics": {
                    "request_latency": {"unit": "ms", "value": ms},
                    "output_sequence_length": {"unit": "tokens", "value": tokens},
                    "time_to_first_token": {"unit": "ms", "value": 1000}}}
            records = [record(1000, 10), record(10000, 1000), record(999999, 1, benchmark_phase="warmup"),
                       record(1, 0), record(999999, 1, was_cancelled=True),
                       record(999999, 1) | {"error": {"message": "failure"}}]
            (folder / "profile_export.jsonl").write_text("\n".join(json.dumps(r) for r in records))
            result = summarize(folder, 4)
            self.assertEqual(result["successful_requests"], 3)
            self.assertEqual(result["request_errors"], 1)
            self.assertEqual(result["cancelled_requests"], 1)
            self.assertEqual(result["interactivity_excluded_successful_requests"], 1)
            self.assertAlmostEqual(result["e2e_normalized_interactivity_p90_tps"], 1 / 0.091)
            self.assertEqual(result["total_tok_s_gpu"], 250)
            self.assertEqual(result["ttft_p95_s"], 1)
            self.assertFalse(result["combined_slo_pass"])
            self.assertEqual(percentile([1, 2, 3], 0.95), 2.9)
            data["request_count"]["avg"] = 100
            (folder / "profile_export_aiperf.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "export may be incomplete"):
                summarize(folder, 4)

    def test_successful_process_with_invalid_scenario_fails_the_job(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cell"
            argv = ["run_point.py", "--model", "test/model", "--tokenizer", "test/model",
                    "--tokenizer-revision", "a" * 40, "--url", "http://example.invalid",
                    "--artifact-dir", str(output), "--concurrency", "48", "--gpus", "24"]
            with patch.object(sys, "argv", argv), \
                 patch("run_point.subprocess.run", return_value=SimpleNamespace(returncode=0)), \
                 patch("run_point.importlib.metadata.version", return_value="0.12.0"), \
                 patch("run_point.summarize", return_value={"submission_valid": False, "successful_requests": 10}), \
                 redirect_stdout(StringIO()):
                self.assertEqual(run_point_main(), 1)
            self.assertFalse(json.loads((output / "status.json").read_text())["comparison_eligible"])


if __name__ == "__main__":
    unittest.main()
