# N3U KV vs. RR: aggregated serving

This recipe runs Nemotron-3-Ultra-550B-A55B-NVFP4 on **24 A4X Max GPUs**:
six TP4/EP4 workers, each handling prefill and decode, with a 262,144-token
context and 16 running requests per worker.

- [Worker configuration](recipe.json)
- [Component deployment script](deploy.sh)
- [Concurrency sweep script](sweep.sh)
- [Comparison, prerequisites, dataset replay, and quickstart](../../../../disaggregated-serving/dynamo/sglang/README.md)

Complete the shared setup in the comparison guide first. Keep the repository
checkout intact: these entry points use the shared helpers and environment
file in the [disaggregated N3U directory](../../../../disaggregated-serving/dynamo/sglang/n3u-kv-vs-rr).
From this directory, inspect the deployment and sweep commands with:

```bash
source ../../../../disaggregated-serving/dynamo/sglang/n3u-kv-vs-rr/env.sh
./deploy.sh worker --dry-run
./sweep.sh --concurrencies 48,96,192 --dry-run
```

The sweep compares KV and RR while keeping the worker recipe fixed. Each
measured point uses the same AgentX corpus and replay controls from the
comparison guide.
