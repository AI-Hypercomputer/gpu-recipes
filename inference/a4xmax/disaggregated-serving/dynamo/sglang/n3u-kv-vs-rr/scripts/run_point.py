# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Run exactly one AgentX cell inside the benchmark Job and retain its status."""

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

from summarize import summarize


def aiperf_command(a):
    command = [
        "aiperf", "profile", "--model", a.model, "--tokenizer", a.tokenizer,
        "--tokenizer-revision", a.tokenizer_revision, "--tokenizer-trust-remote-code",
        "--scenario", "inferencex-agentx-mvp", "--endpoint-type", "chat",
        "--public-dataset", "semianalysis_cc_traces_weka_062126_256k",
        "--cache-bust", "first_turn_prefix", "--system-idle-gap-cap-seconds", "10",
        "--trajectory-start-min-ratio", "0.25", "--trajectory-start-max-ratio", "0.75",
        "--use-server-token-count", "--num-dataset-entries", "393",
        "--slice-duration", "1.0", "--max-context-length", "262144",
        "--url", a.url, "--streaming", "--extra-inputs", "ignore_eos:true",
        "--concurrency", str(a.concurrency), "--random-seed", str(a.seed),
        "--benchmark-duration", str(a.duration), "--benchmark-grace-period", "60",
        "--request-timeout-seconds", str(a.request_timeout),
        "--workers-max", str(a.workers), "--record-processors", str(a.record_processors),
        "--profile-export-level", "raw", "--ui", "simple",
        "--artifact-dir", str(Path(a.artifact_dir) / "profile"),
    ]
    if a.metrics_url:
        command += ["--server-metrics", *a.metrics_url]
    if a.smoke:
        command += ["--unsafe-override"]
    return command


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ["model", "tokenizer", "tokenizer-revision", "url", "artifact-dir"]:
        p.add_argument(f"--{arg}", required=True)
    for arg, default in [("concurrency", None), ("gpus", None), ("duration", 3600), ("seed", 42),
                         ("workers", 200), ("record-processors", 8), ("request-timeout", 1200)]:
        p.add_argument(f"--{arg}", type=int, default=default, required=default is None)
    p.add_argument("--metrics-url", action="append", default=[])
    p.add_argument("--smoke", action="store_true")
    return p


def main():
    a = parser().parse_args()
    if a.duration < 900 and not a.smoke:
        raise ValueError("A measured AgentX run requires at least 900 seconds")
    root = Path(a.artifact_dir)
    root.mkdir(parents=True, exist_ok=False)
    command = aiperf_command(a)
    (root / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    (root / "packages.json").write_text(json.dumps(packages, indent=2) + "\n")
    if importlib.metadata.version("aiperf") != "0.12.0":
        raise ValueError("This recipe requires AIPerf 0.12.0")
    # No synthetic warmup and no session-affinity TTL are added to this baseline.
    # AIPerf performs the scenario's trajectory warmup itself.
    exit_code = subprocess.run(command, check=False).returncode
    status = {"aiperf_exit_code": exit_code, "smoke": a.smoke, "comparison_eligible": False}
    try:
        metrics = summarize(root / "profile", a.gpus)
        (root / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
        status.update(submission_valid=metrics["submission_valid"],
                      comparison_eligible=exit_code == 0 and metrics["submission_valid"] is True and not a.smoke)
        if a.smoke:
            # Unsafe smoke runs must honestly advertise the scenario violation.
            okay = exit_code == 0 and metrics["successful_requests"] > 0 and metrics["submission_valid"] is False
        else:
            okay = status["comparison_eligible"] and metrics["successful_requests"] > 0
    except (OSError, ValueError, KeyError) as e:
        status["validation_error"] = str(e)
        okay = False
    finally:
        (root / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2), flush=True)
    return 0 if okay else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as e:
        sys.exit(str(e))
