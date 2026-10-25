# N3U KV vs. RR: disaggregated serving

This recipe runs Nemotron-3-Ultra-550B-A55B-NVFP4 on **64 A4X Max GPUs**:
eight TP4/EP4 prefill workers and eight TP4/EP4 decode workers in one NVLink
domain. The admission limits are 8 requests per prefill worker and 64 per
decode worker, with Mooncake MNNVL transfer between the stages.

- [Worker configuration](recipe.json)
- [Component deployment script](deploy.sh)
- [Concurrency sweep script](sweep.sh)
- [Comparison, prerequisites, dataset replay, and quickstart](../README.md)

This directory also holds the shared image builds, environment example,
benchmark helpers, tests, and measured results used by both architectures.
Complete the shared setup in the comparison guide first. From this directory:

```bash
source env.sh
./deploy.sh domain --dry-run
./deploy.sh prefill --dry-run
./deploy.sh decode --dry-run
./sweep.sh --concurrencies 72,96,192,480 --dry-run
```

The [aggregated N3U recipe](../../../../aggregated-serving/dynamo/sglang/n3u-kv-vs-rr/README.md)
has its own worker configuration and entry points.
