# GPU 2 cold-start C1 observation

1088/1088 requests succeeded; no saved audio, no explicit seed, 8 warmups. Private MPS maps to physical GPU 2. Live server and client CPU affinity matched B1.

First observed SW thermal flag: 2026-10-21T00:36:02.421+08:00; 380.300s after service launch, 129.951s after reconstructed first timed HTTP send. Core 84°C, HBM 89°C, SM 1305MHz.

Operating-limit query: GPU max operating 85°C, GPU slowdown 89°C, HBM max operating 95°C. One-second observations cannot identify the exact trigger instant or the responsible sensor.

| Metric | GPU 2 cold-start C1 | Prior GPU 1 B1 C1 |
| --- | ---: | ---: |
| Timed duration (s) | 1100.82 | 1165.36 |
| Audio throughput (audio s/s) | 4.34 | 4.10 |
| Timed SM median (MHz) | 1200.00 | 1110.00 |
| Timed SW thermal samples (%) | 88.22 | 93.63 |
| Timed core maximum (°C) | 86.00 | 86.00 |
| Timed HBM maximum (°C) | 92.00 | not recorded |
| Last 5 min SM median (MHz) | 1200.00 | 1110.00 |
| Last 5 min core median (°C) | 84.00 | 84.00 |
| Last 5 min HBM median (°C) | 90.00 | not recorded |
| Last 5 min SW thermal samples (%) | 100.00 | 100.00 |

The physical GPU, starting temperatures and thermal history differ. This is not a controlled causal performance comparison. GPU 1 did not record HBM temperature. Telemetry rates are sample fractions, and old GPU 1 timed-window alignment has <1s uncertainty.

NVIDIA reports GPU Fan Speed as N/A; this does not show that chassis fans have stopped. Actual chassis fan RPM/mode was unavailable to this account. No fan or BMC setting was changed.

[Timeline](timeline.png) · [One-second CSV](timeline.csv) · [Detailed evidence](COLD_GPU2_FINDINGS.json)
