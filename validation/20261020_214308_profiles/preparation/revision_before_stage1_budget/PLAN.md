# A100 Drama 3 profiling preparation

Prepared outside both repositories. No GPU was used during preparation. Do not run until B1 finishes and GPU 1 is handed over by the root agent.

## Revisions and source evidence

Server `${WORKSPACE}/vllm-omni` at `da572cddb38383deae9caef33b8d490e292ef071`; installed vLLM 0.32.0+cu129. Benchmark tool source at `3b116ea1a7f39323e517804900aa7bb016bafdf1` was inspected. The exact execution revision/hashes must be recorded again when profiling starts.

- `vllm_omni/worker/base.py:97–157` creates per-stage Omni profilers and supports delayed start/max iterations. Both stage 0 GPUARWorker and stage 1 GPUGenerationWorker inherit upstream Worker.execute_model.
- Installed `vllm/v1/worker/gpu_worker.py:1179–1300` calls profiler.step and creates `execute_context_N(tokens)_generation_M(tokens)` scopes from SchedulerOutput. It wraps model_runner.execute_model at approximately line 1331. This includes MR V2 for stage 0.
- Installed `vllm/v1/utils.py:835` counts new requests and cached requests whose output token count remains zero as context. Chunked prefill remains context. Decode has zero context and positive generation; mixed has both positive. Pure prefill is reported separately.
- `vllm_omni/entrypoints/openai/api_server.py:3035–3084` accepts POST `/start_profile` and `/stop_profile`, body `{"stages":[0,1]}`. The Omni request schema does not expose upstream per-session max/delay override, so set limits per stage in YAML.
- `vllm_omni/profiler/omni_torch_profiler.py` exports stage-named Chrome traces; both CPU and CUDA are defaults. It ignores torch warmup/active schedule options; use WorkerProfiler max_iterations plus wall-time endpoint stop instead.
- `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1` enables extra native record_function scopes. Stack/shape/memory collection stays off.

## Execution after B1 only

1. Activate the existing private cu129 venv plus `~/drama3-a100/activate_cuda129.sh`. Set physical GPU 1 and the same server/client CPU affinity used for B1. Never alter system driver/CUDA, MPS topology, clocks, or deploy compute settings.
2. With GPU 1 idle and ownership confirmed, run `graph_trace_probe.py --out <new probe dir>`. It captures a graph *before* starting torch profiler, then replays three times. Require internal matrix-multiply and sine CUDA kernel events, not merely cudaGraphLaunch. Save trace/probe JSON. Stop if no kernels or CUPTI errors. This uses a few tens of MB for seconds.
3. For each C8, C16, C64, create a new run directory. Use `make_config.py --source <default yaml> --out <run>/deploy.yaml --trace-dir <run>/traces`. It adds only profiler_config to the two stage dictionaries, verifies all other YAML values remain equal. Defaults: torch, max_iterations=256, delay=0, gzip=true, stack/shapes/memory/flops=false.
4. Start a fresh service with the usual audited own-process lifecycle runner, bound to `taskset -c 0-27,32-59`. Set `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1`. Retain server log, GPU CSV, revision/hash/config metadata. Use no source patch. Require both profiler stages and endpoints enabled.
5. Run client on `taskset -c 60-63`: `python capture_client.py --codes <existing 1088-row codes> --ref-mode codes --concurrency C --num-prompts 1088 --num-warmups 8 --out <run>/client`. No --seed and no --save-audio. This imports existing stream_bench.py unchanged and wraps only its timed-phase call with a 20-second delayed profile trigger, 5-second trace window, and finally stop. Existing eight warmups happen before this. Use all 1088 original rows at every concurrency. The C128 short run completed 256 rows in 14.545 seconds, so 256/512 rows cannot safely guarantee a 20-second steady-state delay plus capture. Recheck each corresponding B1 duration before execution and shorten the delay only if required to retain a full bounded capture window; record any adjustment. Environment knobs: PROFILE_START_AFTER_S and PROFILE_DURATION_S. Timing from these runs is diagnostic, never B1.
6. Stop only our server, wait for gzip completion, verify valid JSON, analyze both stage traces with `analyze_trace.py ... --out <run>/analysis`. Require at least 5 native stage-0 decode steps AND at least 3 mixed steps with correlated CUDA work, actual CUDA kernel events from graph replay, and useful correlation coverage. Inspect complete CPU scopes and trace boundaries; do not use truncated boundary steps. If mixed steps are missing, collect an additional bounded window from another complete closed-loop pass; do not relabel pure-prefill or infer phase from kernel names.
7. Write a short findings table per C: counts of decode/mixed/prefill scopes; actual generation-batch distribution; CPU dispatch scope durations; correlated GPU step span and kernel sum; top raw kernel names and heuristic families; unassigned activity; stage-1 DAC timeline and overlap. Keep CPU scope time distinct from GPU span and kernel sum. Plot profiles independently of B1 throughput.

## Bounds and limitations

Three fresh services likely require 15–25 minutes including ~4-minute starts, plus profiling/trace export and client drain. First use a small graph probe. Each window is bounded to 5 wall seconds and at most 256 execute calls per stage. Both traces can still be large (rough initial allowance 1–4 GB raw per C, often much smaller compressed), so record actual disk/RSS and reduce to 64–128 iterations if needed. No long full-run trace.

Native worker scopes do not wrap sample_tokens outside execute_model. Such launches remain explicitly unassigned unless actual profiler correlation identifies them. Stage 0 fused/codebook sampling inside the full graph is naturally part of execute scope, but same-named backbone/tower kernels cannot be separated by names alone. The analyzer never assigns phase by GPU event timestamp: asynchronous GPU execution can follow CPU scope completion; it requires runtime correlation to the CPU launch inside a real scheduler scope.

Cross-stage overlap uses each Chrome trace's baseTimeNanoseconds plus event timestamp, only if anchors exist; verify same-host overlapping windows before interpreting. Kernel-time sums can double-count overlap, so union and overlap are reported separately. Observing overlap does not establish an MPS slowdown percentage; that requires a separate controlled experiment and is not claimed here.

Installed nsys is `/usr/sbin/nsys`, version `2023.4.4.54`. It advertises `--cuda-graph-trace=node`, `--trace-fork-before-exec=true`, and cudaProfilerApi capture. Its CUDA12.9 compatibility is unverified. If torch fails graph coverage, a tiny nsys graph probe may be tried using existing nsys, never upgrading system tools. For full fallback, each stage uses profiler_config `{profiler: cuda}` and NVTX scopes, nsys launches the service with graph=node and bounded capture-range; preserve failures and do not claim a successful profile without internal kernels and per-step annotations.

CPU-only analyzer validation is recorded in cpu_validation.json. No service/GPU profiling has yet been executed by this preparation agent.

## Lifecycle audit

Installed WorkerProfiler.step automatically calls its public stop() after the iteration limit. A later HTTP stop_profile calls WorkerProfiler.stop(), which returns safely when _active is false (wrapper.py approximately lines 173–190). Thus the endpoint stop after max_iterations is intentionally idempotent at worker level. The HTTP response still must be checked and archived. This has been source-verified, not yet live-verified.

## B1 handover context

B1 root: `${NVME_WORKDIR}/baseline/20261020_194158`; points C1/4/8/12/16/24/32/48/64/96/128 with C64 ×3 and C128 ×5, total17 runs. Profile execution waits for root-agent GPU handover. Each profile client uses all1088 rows, no audio and no explicit seed.

## Executable runner handover

Execution root: `${NVME_WORKDIR}/profiling/20261020_214308`. External runner `preparation/run.sh` launches three fresh services C8/C16/C64, verifies source/tool/helper hashes and exclusive GPU 1, records thermal CSV every second, and uses the audited API-only SIGINT supervisor with UID/create_time checks for any leftover own stage/MPS processes. Private TMPDIR is `${NVME_WORKDIR}/tmp/pr-214308`. It does not alter B1 environment files.

No GPU work begins until root creates `GPU_HANDOVER_AUTHORIZED`; then run the graph probe separately into `graph_probe`, inspect graph-node coverage and correlations, and launch runner in a dedicated tmux session. Native stage0/stage1 MPS trace coverage must be audited separately: a successful standalone graph probe is necessary but insufficient. Each stage has an independent max_iterations counter, so its active window may differ. Original raw traces are retained even for insufficient coverage attempts.

Example detached launch *after* handover and probe: `tmux new-session -d -s drama3-profile 'bash ${NVME_WORKDIR}/profiling/20261020_214308/preparation/run.sh > ${NVME_WORKDIR}/profiling/20261020_214308/runner.log 2>&1; printf "%s\n" "$?" > ${NVME_WORKDIR}/profiling/20261020_214308/runner.exit'`. No such launch has occurred during preparation.

The current B1 has substantial software thermal slowdown at 84C: observed SM medians around C8 735MHz, C16 600MHz and C64 570–592MHz. Profile durations must be interpreted with the simultaneously captured profile thermal/SM data. Absolute kernel durations must not be presented as equal-temperature comparisons against other runs.
