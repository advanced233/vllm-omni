# Service controls: descriptive results

C16 × 256 requests per fresh service, plus 8 excluded warmups and 33 C1 drain requests outside the main window. No torch profiler, no audio saving, no explicit request seed. Each control has one run.

| Control | Complete probe steps | GDN host p50 ms | Stepwise GDN sum/span p50 % | Audio s/s | TTFA p50/p95 ms | Mean RTF | SM median MHz |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| baseline_laps | 220 | 28.969 | 89.556 | 32.68 | 200.0/363.3 | 0.4690 | 1395 |
| baseline_no_laps | 219 | 29.093 | 90.791 | 33.18 | 202.6/355.6 | 0.4739 | 1380 |
| installed | — | — | — | 31.07 | 204.8/360.2 | 0.4833 | 1305 |
| direct_out_laps | 217 | 28.822 | 89.700 | 30.93 | 209.0/360.8 | 0.4899 | 1290 |

The sum/span denominator is only the host span from first to last GDN call. It is neither complete-step wall time nor GPU time.

| Control | Core range °C | HBM range °C | SM median MHz | SW thermal samples % | SW power-cap samples % |
| --- | --- | --- | ---: | ---: | ---: |
| baseline_laps | 60–74 | 68–79 | 1395 | 0.00 | 77.14 |
| baseline_no_laps | 69–82 | 75–85 | 1380 | 0.00 | 96.88 |
| installed | 78–85 | 82–89 | 1305 | 50.00 | 50.00 |
| direct_out_laps | 78–85 | 81–89 | 1290 | 57.14 | 45.71 |

These thermal/clock conditions were not matched; end-to-end differences cannot be assigned solely to the probe or direct-output change.

| Lapped control | Chunk mean ms | Gather mean ms | Scatter mean ms | Copy mean ms |
| --- | ---: | ---: | ---: | ---: |
| baseline_laps | 15.204 | 2.295 | 1.475 | 0.557 |
| direct_out_laps | 15.752 | 2.316 | 1.496 | 0.026 |

Three-way common shapes: **70**; **57** have fewer than 3 samples in at least one group.

- baseline_laps: 147/220 steps covered (66.8%).
- baseline_no_laps: 151/219 steps covered (68.9%).
- direct_out_laps: 149/217 steps covered (68.7%).

[Per-shape counts and host p50](common_shape_host.csv) preserve sparse cells; no significance claim is made. Same shape does not mean the same request or matched hardware conditions. See [complete comparison](SERVICE_CONTROLS_COMPARISON.json) for temperature, clocks, long-output contribution and section timings. Single runs cannot establish stable gains or zero measurement overhead.
