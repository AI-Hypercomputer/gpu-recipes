#!/usr/bin/env bash
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
RECIPE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SHARED_DIR=$(cd -- "$RECIPE_DIR/../../../../disaggregated-serving/dynamo/sglang/n3u-kv-vs-rr" && pwd)
exec python3 "$SHARED_DIR/scripts/sweep.py" --architecture agg "$@"
