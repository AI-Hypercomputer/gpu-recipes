# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Summarize successful profiling records, keeping failures and exclusions visible."""

import argparse
import json
import math
from pathlib import Path


def percentile(values, q):
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sample")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    fraction = pos - lower
    return ordered[lower] * (1 - fraction) + ordered[min(lower + 1, len(ordered) - 1)] * fraction


def metric(record, name, unit):
    field = record.get("metrics", {}).get(name)
    if field is None:
        return None
    if field["unit"] != unit:
        raise ValueError(f"Unexpected {name} unit: {field['unit']}")
    value = field["value"]
    return float(value) if value is not None and math.isfinite(value) else None


def summarize(directory, gpus):
    if gpus <= 0:
        raise ValueError("GPU count must be positive")
    data = json.loads((directory / "profile_export_aiperf.json").read_text())
    success = errors = cancelled = overflows = excluded = 0
    normalized, ttft, e2e = [], [], []
    traces, sessions, turns, isl, osl = set(), set(), [], [], []
    with (directory / "profile_export.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            meta = record["metadata"]
            if meta.get("benchmark_phase") != "profiling":
                continue
            if record.get("error"):
                errors += 1
                continue
            if meta.get("was_cancelled"):
                cancelled += 1
                continue
            if meta.get("context_overflow_skip"):
                overflows += 1
                continue
            success += 1
            latency = metric(record, "request_latency", "ms")
            output = metric(record, "output_sequence_length", "tokens")
            first = metric(record, "time_to_first_token", "ms")
            prompt = metric(record, "input_sequence_length", "tokens")
            if latency is not None and output is not None and output > 0 and latency > 0:
                normalized.append(latency / 1000 / output)
            else:
                excluded += 1
            if first is not None:
                ttft.append(first / 1000)
            if latency is not None:
                e2e.append(latency / 1000)
            if prompt is not None:
                isl.append(prompt)
            if output is not None:
                osl.append(output)
            if meta.get("source_trace_id") is not None:
                traces.add(str(meta["source_trace_id"]))
            if meta.get("root_correlation_id") is not None:
                sessions.add(str(meta["root_correlation_id"]))
            if meta.get("turn_index") is not None:
                turns.append(meta["turn_index"])
    if not success or not ttft or not normalized:
        raise ValueError("No usable successful profiling request metrics")
    if data["request_count"]["unit"] != "requests" or success != data["request_count"]["avg"]:
        raise ValueError("Successful request count differs between summary and JSONL; export may be incomplete")
    # Read AIPerf's throughput, whose denominator includes its recorded window.
    # Dividing completed token sums by the requested duration would be different.
    def average(name, unit):
        field = data[name]
        if field["unit"] != unit:
            raise ValueError(f"Unexpected {name} unit: {field['unit']}")
        return field["avg"]
    output_rate = average("output_token_throughput", "tokens/sec")
    input_rate = average("input_token_throughput", "tokens/sec")
    p95 = percentile(ttft, 0.95)
    i90 = 1 / percentile(normalized, 0.9)
    return {
        "submission_valid": data.get("metadata", {}).get("submission_valid"),
        "submission_invalid_reasons": data.get("metadata", {}).get("submission_invalid_reasons", []),
        "gpus": gpus, "total_tok_s_gpu": (input_rate + output_rate) / gpus,
        "output_tok_s_gpu": output_rate / gpus,
        "requests_s": average("request_throughput", "requests/sec"),
        "ttft_p95_s": p95, "e2e_p95_s": percentile(e2e, 0.95),
        "e2e_normalized_interactivity_p90_tps": i90,
        "ttft_slo_pass": p95 < 10, "interactivity_slo_pass": i90 >= 20,
        "combined_slo_pass": p95 < 10 and i90 >= 20,
        "successful_requests": success, "request_errors": errors,
        "cancelled_requests": cancelled, "context_overflow_skips": overflows,
        "interactivity_excluded_successful_requests": excluded,
        "ttft_excluded_successful_requests": success - len(ttft),
        "source_trace_ids": sorted(traces), "distinct_sessions": len(sessions),
        "turn_index_p50": percentile(turns, 0.5) if turns else None,
        "input_tokens_p50": percentile(isl, 0.5) if isl else None,
        "output_tokens_p90": percentile(osl, 0.9) if osl else None,
        "output_tokens_p99": percentile(osl, 0.99) if osl else None,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("profile_directory", type=Path)
    p.add_argument("--gpus", type=int, required=True)
    a = p.parse_args()
    print(json.dumps(summarize(a.profile_directory, a.gpus), indent=2))
