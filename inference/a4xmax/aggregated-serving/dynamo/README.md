# Dynamo aggregated serving on A4X Max

Use the [Nemotron-3-Ultra KV/RR benchmarking recipe](../../disaggregated-serving/dynamo/sglang/README.md)
for the 24-GPU aggregated configuration (six TP4/EP4 workers), AgentX dataset
replay, measured performance curves, and SLO comparisons.

The [agg recipe](sglang/n3u-kv-vs-rr/recipe.json),
[deployment steps](sglang/n3u-kv-vs-rr/deploy.sh), and
[agg sweep script](sglang/n3u-kv-vs-rr/sweep.sh) are separate from the
64-GPU disaggregated configuration.
