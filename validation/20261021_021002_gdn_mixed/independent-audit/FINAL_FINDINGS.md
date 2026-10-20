# GDN mixed-core experiment: final evidence

**The corrected 37-case correctness gate passed. A stable net host speedup from `direct_out` was not established.**

The recorded execution order was fingerprint → 37 correctness tests →microbenchmark →four fresh service controls. User benchmark/probe/server computation remained fixed; one external runner update added foreign-GPU watchdog checks and post-cleanup export of the unchanged official summary. Its previous version and both hashes are retained.

Correctness:37 passed, 0 failed, 0 errors, 0 skipped; original tolerances unchanged. The fixture now reserves state row 0 and adds finiteness/null-row checks.

Microbenchmark:200 iterations per case, randomized installed/baseline/direct order. Baseline minus installed host medians were +0.034%, +0.250%, +0.111%. Direct versus baseline was +0.167%, −0.390%, −0.329%. These are single-run descriptive differences, not confidence intervals. CUDA-event span and separate-profiler device-time estimates are distinct quantities; host×30 is not a measured service gain.

In service, direct-output GDN host p50 was 28.822 ms versus 28.969 ms baseline (−0.509%), while its mean was 29.217 ms versus 29.075 ms (+0.489%). The opposite directions do not demonstrate stable net improvement. Mean copy-section time fell 0.557 → 0.026 ms, but the chunk interval rose 15.204 → 15.752 ms. All 1024 main requests and 132 out-of-window drain requests succeeded.

## Four service controls

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


[Full independent evidence](FINAL_FINDINGS.json) includes gates, source checks, exact windows and post-main flush validation. Original user summaries remain unchanged.
