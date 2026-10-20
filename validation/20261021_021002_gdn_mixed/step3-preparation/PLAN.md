# Service-only low-overhead GDN probe experiment preparation

CPU preparation only. No service/GPU launch is authorized until correctness and microbenchmark gates both pass and the root agent explicitly hands over GPU1.

Four fresh services, in this order: baseline with default laps; baseline with DRAMA3_GDN_PROBE_LAPS=0; installed function with no probe environment/PYTHONPATH injection; direct_out with default laps. Each uses unchanged default BF16 deployment, C16 first256 codes, no explicit seed, no audio, eight warmups, same affinity/server0–27,32–59 and client60–63. No torch.profiler, NVTX/custom profiling scopes, or unvalidated thermal gate. Record clocks.sm, clocks.mem, temperature, utilization, power, and every relevant throttle reason at1s intervals.

Probe installation variables and probe_site PYTHONPATH go only into API/stage subtree environment through Popen(env=...). The supervisor, correctness tests, microbenchmark and every benchmark client receive a cleaned environment without DRAMA3_GDN_* and without the probe_site path. Use an absolute private NVMe DRAMA3_GDN_PROBE_DIR per run. Any NOT patched/not installed log stops work. A stage0 PID-specific patched line is mandatory; API's patched line is not proof.

## Prevent loss of final probe records without changing user tools

StepRecorder flushes sequential batches of32 completed steps before the next step, and also registers atexit.close. It has no signal handler. API-only SIGINT does not prove stage0 atexit ran: the engine's own shutdown may terminate/forcekill stage0 or kill during finally, losing buffered tail. Do not interpret missing tail as a performance improvement.

Root-approved external method: after the main C16×256 completes, issue33 additional normal C1 codes requests into separate flush_drain output. Keep these outside the main benchmark timing and perform the same drain in all four controls. Each request has a new ForwardContext and produces at least one prefilling step, causing the probe's unmodified32-record periodic append to flush the entire earlier timed tail. Verify stage0 JSONL has a durable record with timestamp strictly AFTER the actual main client's completion walltime, all records have30 calls and correct variant/laps. Ordered JSONL append then proves all preceding main-window records persisted. Only then API-only SIGINT; losing final post-window drain records cannot affect reported main data. If this check fails, stop/report; do not silently depend on atexit.

Original summarize_probe.py defines its conservative timed window using started_at rounded to seconds, start+1 through start+duration. Reuse that exact function, record its window, and independently require per-step30 calls and expected variant/laps. Do not pool API/probe processes with stage0. The 33 post-window requests and8 warmups are never included in the main summary.

No changes to the user's probe, function variants, service sources or venv. External orchestration inherits the proven API-only SIGINT supervisor, with leftover UID/create_time verification and private MPS socket scope. Cleanup metadata cannot by itself establish probe data integrity; validate durable post-window records before stopping and again after shutdown.

## Corrected tool revision

Use only `${NVME_WORKDIR}/gdn-mixed/20261021_021002/tool_snapshot/drama3_bench`, exact user commit `7aeaa5b46ec1cdbdb7e5d43b396b4d05af87fa37`. The fixture now assigns real state slots starting at1, reserves NULL_BLOCK_ID0, verifies null/spare rows unchanged, checks finiteness and retains detailed standard-path mismatch errors. Old failed daac4cf results never authorize Step3. Root must provide STEP3_AUTHORIZED.json with all37correctness pass, Step2complete, exact evidence paths and GPU handover.
