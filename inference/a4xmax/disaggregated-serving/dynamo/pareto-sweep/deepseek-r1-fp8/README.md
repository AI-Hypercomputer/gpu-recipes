# DeepSeek-R1 (FP8): Values Files

Helm values files for the DeepSeek-R1 (FP8) operating points on A4X-MAX. Each file sets the topology and worker environment of one point on the Pareto curve.

Deployment steps, point details and the Pareto map are in the [pareto-sweep recipe](../README.md#points-dsr1-fp8).

| Point | Pareto position | Values file | Serving configs |
| :--- | :--- | :--- | :--- |
| **low-latency** | Low latency | [`dsr1_fp8_low_latency.yaml`](dsr1_fp8_low_latency.yaml) | [prefill](../../../../../../src/frameworks/a4xmax/dynamo-configs/dsr1-fp8-low-latency-1p1d-tp4-tp4-prefill.yaml), [decode](../../../../../../src/frameworks/a4xmax/dynamo-configs/dsr1-fp8-low-latency-1p1d-tp4-tp4-decode.yaml) |
| **mid** | Mid-curve | [`dsr1_fp8_mid_curve.yaml`](dsr1_fp8_mid_curve.yaml) | [prefill](../../../../../../src/frameworks/a4xmax/dynamo-configs/dsr1-fp8-mid-curve-5p1d-dep8-dep32-prefill.yaml), [decode](../../../../../../src/frameworks/a4xmax/dynamo-configs/dsr1-fp8-mid-curve-5p1d-dep8-dep32-decode.yaml) |
| **max** | Max throughput | [`dsr1_fp8_max_throughput.yaml`](dsr1_fp8_max_throughput.yaml) | [prefill](../../../../../../src/frameworks/a4xmax/dynamo-configs/dsr1-fp8-max-throughput-6p1d-dep8-dep24-prefill.yaml), [decode](../../../../../../src/frameworks/a4xmax/dynamo-configs/dsr1-fp8-max-throughput-6p1d-dep8-dep24-decode.yaml) |
