#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Standalone /v1/systemone and /v1/decisions benchmark and boundary probe.

Runs warmed, zero-error concurrency sweeps against a deployed Nemotron 3.5
Decision vLLM endpoint, reporting requests/s, questions/s, amortized ms/q,
client p50/p95/p99 latency, and error counts. Uses Python stdlib only so it
runs in any container without external packages.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import math
import statistics
import threading
import time
import urllib.parse


def _make_questions(count: int) -> dict[str, dict]:
  base = [
      (
          "team",
          {
              "type": "choice",
              "instructions": "Which team should handle this ticket?",
              "criteria": {
                  "billing": "payments, invoices, and refunds",
                  "shipping": "delivery delays and tracking",
                  "tech": "bugs, outages, and login failures",
              },
          },
      ),
      (
          "refund",
          {
              "type": "noul",
              "instructions": "Is the customer asking for a refund?",
          },
      ),
      (
          "severity",
          {
              "type": "score",
              "instructions": "How severe is the customer impact?",
              "criteria": ["Low impact", "Degraded service", "Service outage"],
          },
      ),
  ]
  questions: dict[str, dict] = {}
  for idx in range(count):
    prefix, template = base[idx % len(base)]
    q = dict(template)
    if idx >= len(base):
      q["instructions"] = f"[Check {idx + 1}] {template['instructions']}"
      key = f"{prefix}_{idx}"
    else:
      key = prefix
    questions[key] = q
  return questions


def _make_state(target_tokens: int, variant_id: int) -> str:
  core = (
      f"Ticket #{variant_id}: Customer reports duplicate credit card charge on"
      " their enterprise plan and requests an immediate refund because the"
      " billing portal showed an error during checkout."
  )
  if target_tokens <= 250:
    return core
  filler = (
      "Account audit log entry showing normal service operation, routine"
      " health check passed, regional latency within nominal range, and no"
      " security incidents recorded in the billing ledger. "
  )
  target_chars = max(0, int(target_tokens * 5.9) - len(core))
  repeats = max(1, target_chars // len(filler))
  blob = filler * repeats
  half = len(blob) // 2
  return blob[:half] + "\n" + core + "\n" + blob[half:]


class _ConnPool:
  """Thread-local HTTP keep-alive connection pool."""

  def __init__(self, host: str, port: int, is_https: bool, timeout_s: float):
    self._host = host
    self._port = port
    self._is_https = is_https
    self._timeout_s = timeout_s
    self._local = threading.local()

  def post_json(self, path: str, payload_bytes: bytes) -> tuple[int, bytes, float]:
    conn = getattr(self._local, "conn", None)
    for attempt in range(2):
      if conn is None:
        cls = (
            http.client.HTTPSConnection
            if self._is_https
            else http.client.HTTPConnection
        )
        conn = cls(self._host, self._port, timeout=self._timeout_s)
        self._local.conn = conn
      t0 = time.perf_counter()
      try:
        conn.request(
            "POST",
            path,
            body=payload_bytes,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        body = resp.read()
        dt = time.perf_counter() - t0
        return resp.status, body, dt
      except (http.client.HTTPException, OSError):
        try:
          conn.close()
        except OSError:
          pass
        conn = None
        self._local.conn = None
        if attempt == 1:
          raise
    raise RuntimeError("unreachable")


def _percentile(sorted_vals: list[float], pct: float) -> float:
  if not sorted_vals:
    return float("nan")
  idx = (len(sorted_vals) - 1) * (pct / 100.0)
  lo = math.floor(idx)
  hi = math.ceil(idx)
  if lo == hi:
    return sorted_vals[lo]
  return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def run_tier(
    pool: _ConnPool,
    path: str,
    model: str,
    concurrency: int,
    num_requests: int,
    questions_per_req: int,
    state_tokens: int,
    warmup: int,
) -> dict:
  questions = _make_questions(questions_per_req)
  payloads = [
      json.dumps(
          {
              "model": model,
              "state": _make_state(state_tokens, i),
              "questions": questions,
          },
          separators=(",", ":"),
      ).encode("utf-8")
      for i in range(max(warmup, num_requests))
  ]

  warmup_latencies = []
  for i in range(warmup):
    status, body, dt = pool.post_json(path, payloads[i])
    if status != 200:
      raise RuntimeError(
          f"Warmup {i} failed HTTP {status}: {body.decode('utf-8', 'replace')[:300]}"
      )
    warmup_latencies.append(dt * 1000.0)

  latencies_ms: list[float] = []
  input_tokens_list: list[int] = []
  errors: list[str] = []
  lock = threading.Lock()

  def _one(idx: int) -> None:
    try:
      status, body, dt = pool.post_json(path, payloads[idx])
      if status != 200:
        with lock:
          errors.append(f"HTTP {status}: {body.decode('utf-8', 'replace')[:120]}")
        return
      data = json.loads(body)
      answers = data.get("answers", {})
      if len(answers) != questions_per_req:
        with lock:
          errors.append(
              f"expected {questions_per_req} answers, got {len(answers)}"
          )
        return
      usage = data.get("usage") or {}
      with lock:
        latencies_ms.append(dt * 1000.0)
        if "input_tokens" in usage:
          input_tokens_list.append(int(usage["input_tokens"]))
    except Exception as exc:  # pylint: disable=broad-except
      with lock:
        errors.append(f"{type(exc).__name__}: {exc}")

  t_start = time.perf_counter()
  with ThreadPoolExecutor(max_workers=concurrency) as ex:
    list(ex.map(_one, range(num_requests)))
  wall_s = time.perf_counter() - t_start

  latencies_ms.sort()
  ok = len(latencies_ms)
  req_per_s = ok / wall_s if wall_s > 0 else 0.0
  q_per_s = req_per_s * questions_per_req
  return {
      "concurrency": concurrency,
      "questions_per_req": questions_per_req,
      "state_tokens_target": state_tokens,
      "input_tokens_mean": (
          round(statistics.mean(input_tokens_list), 1)
          if input_tokens_list
          else None
      ),
      "requests_total": num_requests,
      "requests_ok": ok,
      "errors": len(errors),
      "error_samples": errors[:3],
      "wall_s": round(wall_s, 3),
      "warmup_first_ms": round(warmup_latencies[0], 1) if warmup_latencies else None,
      "warmup_last_ms": round(warmup_latencies[-1], 1) if warmup_latencies else None,
      "req_per_s": round(req_per_s, 2),
      "questions_per_s": round(q_per_s, 2),
      "amortized_ms_per_q": round(1000.0 / q_per_s, 2) if q_per_s > 0 else None,
      "p50_ms": round(_percentile(latencies_ms, 50), 2),
      "p95_ms": round(_percentile(latencies_ms, 95), 2),
      "p99_ms": round(_percentile(latencies_ms, 99), 2),
  }


def run_boundary_checks(pool: _ConnPool, model: str) -> dict:
  """Verifies /v1/decisions, long context scaling, and 256/257 options."""
  out: dict = {}

  dec_body = json.dumps({
      "model": model,
      "input": "Customer says: I was charged twice for my order and want a refund.",
      "questions": [
          {
              "type": "choice",
              "instructions": "Which team should handle this ticket?",
              "choices": [
                  {"value": "billing", "description": "payments and refunds"},
                  {"value": "shipping", "description": "delivery problems"},
              ],
          }
      ],
  }).encode("utf-8")
  status, body, dt = pool.post_json("/v1/decisions", dec_body)
  out["decisions_api"] = {
      "status": status,
      "latency_ms": round(dt * 1000.0, 1),
      "snippet": body.decode("utf-8", "replace")[:260],
  }

  ctx_results = []
  for tok in (2000, 8500, 16000, 32000):
    req = json.dumps({
        "model": model,
        "state": _make_state(tok, 999),
        "questions": {
            "refund": {
                "type": "noul",
                "instructions": "Is the customer asking for a refund?",
            }
        },
    }).encode("utf-8")
    st, bd, dt = pool.post_json("/v1/systemone", req)
    rec = {"target_tokens": tok, "status": st, "latency_ms": round(dt * 1000.0, 1)}
    if st == 200:
      d = json.loads(bd)
      rec["input_tokens"] = (d.get("usage") or {}).get("input_tokens")
      rec["noul"] = round(d["answers"]["refund"]["noul"], 4)
    else:
      rec["error"] = bd.decode("utf-8", "replace")[:200]
    ctx_results.append(rec)
  out["context_scaling"] = ctx_results

  opt_results = []
  for n_opts in (256, 257):
    criteria = {f"dept_{i}": f"Department {i} general queue" for i in range(n_opts)}
    criteria["dept_137"] = "Duplicate billing charges and credit card refunds"
    req = json.dumps({
        "model": model,
        "state": "Customer was charged twice on their card and wants a refund.",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Pick the exact department for this ticket.",
                "criteria": criteria,
            }
        },
    }).encode("utf-8")
    st, bd, dt = pool.post_json("/v1/systemone", req)
    rec = {"options": n_opts, "status": st, "latency_ms": round(dt * 1000.0, 1)}
    if st == 200:
      d = json.loads(bd)
      rec["choice"] = d["answers"]["route"]["choice"]
      rec["confidence"] = round(d["answers"]["route"].get("confidence", 0.0), 4)
    else:
      rec["error"] = bd.decode("utf-8", "replace")[:200]
    opt_results.append(rec)
  out["option_boundary"] = opt_results
  return out


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--base-url", default="http://127.0.0.1:8000")
  ap.add_argument("--model", default="nemotron-decide")
  ap.add_argument("--concurrencies", default="1,4,8,16,32,64")
  ap.add_argument("--questions-per-req", type=int, default=3)
  ap.add_argument("--state-tokens", type=int, default=200)
  ap.add_argument("--min-requests", type=int, default=120)
  ap.add_argument("--warmup", type=int, default=5)
  ap.add_argument("--timeout", type=float, default=180.0)
  ap.add_argument("--boundaries", action="store_true")
  ap.add_argument("--output", default="")
  args = ap.parse_args()

  parsed = urllib.parse.urlparse(args.base_url)
  host = parsed.hostname or "127.0.0.1"
  is_https = parsed.scheme == "https"
  port = parsed.port or (443 if is_https else 80)
  pool = _ConnPool(host, port, is_https, args.timeout)

  report: dict = {"base_url": args.base_url, "model": args.model, "tiers": []}
  concurrencies = [
      int(c_str.strip()) for c_str in args.concurrencies.split(",") if c_str.strip()
  ]

  for c in concurrencies:
    n_req = max(args.min_requests, 10 * c)
    row = run_tier(
        pool=pool,
        path="/v1/systemone",
        model=args.model,
        concurrency=c,
        num_requests=n_req,
        questions_per_req=args.questions_per_req,
        state_tokens=args.state_tokens,
        warmup=args.warmup,
    )
    report["tiers"].append(row)
    print(json.dumps(row), flush=True)

  if args.boundaries:
    # Ensure at least one warmup call ran before timing boundaries.
    if not concurrencies and args.warmup > 0:
      warm_payload = json.dumps({
          "model": args.model,
          "state": _make_state(200, 0),
          "questions": _make_questions(1),
      }).encode("utf-8")
      pool.post_json("/v1/systemone", warm_payload)
    report["boundaries"] = run_boundary_checks(pool, args.model)
    print(json.dumps({"boundaries": report["boundaries"]}), flush=True)

  if args.output:
    with open(args.output, "w", encoding="utf-8") as f:
      json.dump(report, f, indent=2)


if __name__ == "__main__":
  main()
