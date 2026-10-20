# Independent profile evidence

A100 80GB PCIe, cu129/R550. Diagnostic torch/CUPTI profiles, separate from B1 throughput.

C8 uses the extended stage-1 capture (attempt 2). All rows use complete interior steps. GPU span, GPU kernel-duration sum and inclusive CPU scope are different quantities.

| C | Phase | Steps | Actual generation requests min/p50/max | Context tokens min/p50/max | GPU span p50/p95 ms | Kernel sum p50 ms | CPU execute p50 ms | CPU GDN union p50 ms |
| ---: | --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| 8 | decode_only | 232 | 6/8/8 | 0/0/0 | 13.744/17.648 | 16.357 | 4.865 | - |
| 8 | mixed | 22 | 6/7/7 | 126/169.5/227 | 49.067/55.238 | 36.969 | 58.954 | 40.011 |
| 16 | decode_only | 194 | 13/16/16 | 0/0/0 | 18.063/23.317 | 20.365 | 6.278 | - |
| 16 | mixed | 29 | 13/15/15 | 148/176/552 | 46.324/59.306 | 39.618 | 58.110 | 38.297 |
| 64 | decode_only | 89 | 57/62/64 | 0/0/0 | 33.179/40.421 | 33.784 | 6.592 | - |
| 64 | mixed | 29 | 56/61/63 | 157/358/1299 | 66.695/106.022 | 58.909 | 60.268 | 39.611 |

Kernel-family shares below use summed kernel durations, **not wall time**. Dense GEMM names cannot distinguish backbone from codebook tower.

| C | Phase | Marlin % | Dense GEMM/GEMV % | BLAS reductions % | GDN/helper % |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8 | decode_only | 30.94 | 29.71 | 8.59 | 3.23 |
| 8 | mixed | 51.97 | 21.65 | 1.67 | 7.53 |
| 16 | decode_only | 38.60 | 25.78 | 7.35 | 3.89 |
| 16 | mixed | 49.76 | 22.53 | 1.50 | 8.41 |
| 64 | decode_only | 46.12 | 19.40 | 3.92 | 8.51 |
| 64 | mixed | 38.05 | 28.06 | 0.48 | 11.67 |

| C | Capture SM min/p50/max MHz | Temperature min/p50/max °C | SW thermal samples | Common capture s | GPU overlap ms |
| ---: | --- | --- | --- | ---: | ---: |
| 8 | 1395/1410/1410 | 82/82/83 | 0/5 | 4.423 | 393.104 |
| 16 | 1275/1305/1365 | 84/84/84 | 4/5 | 5.036 | 626.441 |
| 64 | 1080/1155/1350 | 84/84/85 | 6/6 | 5.109 | 1016.121 |

These are diagnostic observations under unmatched thermal conditions. Differences in mixed-step times do not establish an improvement; no Amdahl or speedup claim is made. CPU GDN durations include nested execution and profiler overhead. They must not be added to their nested FLA function times. Cross-stage overlap establishes concurrent device activity, not a percentage slowdown caused by MPS.

Exact distributions, correlation coverage, hardware samples and measurement definitions are retained in [INDEPENDENT_PROFILE_FINDINGS.json](INDEPENDENT_PROFILE_FINDINGS.json).
