# DeepSeek-R1 (NVFP4): Values Files

Helm values files for the DeepSeek-R1 (NVFP4) operating points on A4X-MAX. Each file sets the topology and worker environment of one point on the Pareto curve.

Deployment steps, point details and the Pareto map are in the [pareto-sweep recipe](../README.md#points-dsr1-fp4).

| Point | Pareto position | Values file | Serving configs |
| :--- | :--- | :--- | :--- |
| **low_latency** | Low latency | [`dsr1_fp4_low_latency.yaml`](dsr1_fp4_low_latency.yaml) | [prefill](../../../../../../src/frameworks/a4xmax/dynamo-configs/dsr1-fp4-low-latency-1p4d-tp4-tp4-prefill.yaml), [decode](../../../../../../src/frameworks/a4xmax/dynamo-configs/dsr1-fp4-low-latency-1p4d-tp4-tp4-decode.yaml) |
| **mid_curve** | Mid-curve | [`dsr1_fp4_mid_curve.yaml`](dsr1_fp4_mid_curve.yaml) | [prefill](../../../../../../src/frameworks/a4xmax/dynamo-configs/dsr1-fp4-mid-curve-6p1d-tp4-dep48-prefill.yaml), [decode](../../../../../../src/frameworks/a4xmax/dynamo-configs/dsr1-fp4-mid-curve-6p1d-tp4-dep48-decode.yaml) |
| **max_tpt** | Max throughput | [`dsr1_fp4_max_throughput.yaml`](dsr1_fp4_max_throughput.yaml) | [prefill](../../../../../../src/frameworks/a4xmax/dynamo-configs/dsr1-fp4-max-throughput-10p1d-tp4-dep32-prefill.yaml), [decode](../../../../../../src/frameworks/a4xmax/dynamo-configs/dsr1-fp4-max-throughput-10p1d-tp4-dep32-decode.yaml) |
