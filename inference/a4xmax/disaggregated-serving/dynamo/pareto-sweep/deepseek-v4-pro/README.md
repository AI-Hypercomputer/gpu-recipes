# DeepSeek-V4-Pro (MXFP4 + EAGLE MTP): Values Files

Helm values files for the DeepSeek-V4-Pro (MXFP4 + EAGLE MTP) operating points on A4X-MAX. Each file sets the topology and worker environment of one point on the Pareto curve.

Deployment steps, point details and the Pareto map are in the [pareto-sweep recipe](../README.md#points-dsv4).

| Point | Pareto position | Values file | Serving configs |
| :--- | :--- | :--- | :--- |
| **p1** | Low latency | [`dsv4_p1_low_latency.yaml`](dsv4_p1_low_latency.yaml) | [prefill](../../../../../../src/frameworks/a4xmax/dynamo-configs/dsv4-p1-low-latency-1p1d-tp4-tp4-prefill.yaml), [decode](../../../../../../src/frameworks/a4xmax/dynamo-configs/dsv4-p1-low-latency-1p1d-tp4-tp4-decode.yaml) |
| **p2** | Low latency | [`dsv4_p2_low_latency.yaml`](dsv4_p2_low_latency.yaml) | [prefill](../../../../../../src/frameworks/a4xmax/dynamo-configs/dsv4-p2-low-latency-1p6d-dep4-tp4-prefill.yaml), [decode](../../../../../../src/frameworks/a4xmax/dynamo-configs/dsv4-p2-low-latency-1p6d-dep4-tp4-decode.yaml) |
| **p3** | Mid-curve | [`dsv4_p3_mid_curve.yaml`](dsv4_p3_mid_curve.yaml) | [prefill](../../../../../../src/frameworks/a4xmax/dynamo-configs/dsv4-p3-mid-curve-1p1d-dep4-dep8-prefill.yaml), [decode](../../../../../../src/frameworks/a4xmax/dynamo-configs/dsv4-p3-mid-curve-1p1d-dep4-dep8-decode.yaml) |
| **p4** | Mid-curve | [`dsv4_p4_mid_curve.yaml`](dsv4_p4_mid_curve.yaml) | [prefill](../../../../../../src/frameworks/a4xmax/dynamo-configs/dsv4-p4-mid-curve-1p1d-dep4-dep16-prefill.yaml), [decode](../../../../../../src/frameworks/a4xmax/dynamo-configs/dsv4-p4-mid-curve-1p1d-dep4-dep16-decode.yaml) |
| **p5** | Mid-curve | [`dsv4_p5_mid_curve.yaml`](dsv4_p5_mid_curve.yaml) | [prefill](../../../../../../src/frameworks/a4xmax/dynamo-configs/dsv4-p5-mid-curve-2p1d-dep4-dep8-prefill.yaml), [decode](../../../../../../src/frameworks/a4xmax/dynamo-configs/dsv4-p5-mid-curve-2p1d-dep4-dep8-decode.yaml) |
| **p6** | Mid-curve | [`dsv4_p6_mid_curve.yaml`](dsv4_p6_mid_curve.yaml) | [prefill](../../../../../../src/frameworks/a4xmax/dynamo-configs/dsv4-p6-mid-curve-4p1d-dep4-dep8-prefill.yaml), [decode](../../../../../../src/frameworks/a4xmax/dynamo-configs/dsv4-p6-mid-curve-4p1d-dep4-dep8-decode.yaml) |
| **p7** | Max throughput | [`dsv4_p7_max_throughput.yaml`](dsv4_p7_max_throughput.yaml) | [prefill](../../../../../../src/frameworks/a4xmax/dynamo-configs/dsv4-p7-max-throughput-6p1d-dep4-dep8-prefill.yaml), [decode](../../../../../../src/frameworks/a4xmax/dynamo-configs/dsv4-p7-max-throughput-6p1d-dep4-dep8-decode.yaml) |
| **p8** | Max throughput (end of curve) | [`dsv4_p8_max_throughput.yaml`](dsv4_p8_max_throughput.yaml) | [prefill](../../../../../../src/frameworks/a4xmax/dynamo-configs/dsv4-p8-max-throughput-8p1d-dep4-dep8-prefill.yaml), [decode](../../../../../../src/frameworks/a4xmax/dynamo-configs/dsv4-p8-max-throughput-8p1d-dep4-dep8-decode.yaml) |
