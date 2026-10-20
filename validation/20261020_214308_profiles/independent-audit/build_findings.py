"""Build a compact, publication-safe summary after all three profile audits."""
import json
from pathlib import Path
from audit_trace import stats

ROOT = Path(__file__).parent
SELECTED = [(8, "c8-attempt2"), (16, "c16"), (64, "c64")]


def read(path):
    return json.loads(path.read_text())


def ms(stats_us):
    return {k: (v/1000 if k != "n" and v is not None else v) for k,v in stats_us.items()}


rows = []
captures = []
for concurrency, name in SELECTED:
    directory = ROOT / name
    trace = read(directory / "trace_0_independent.json")
    codec = read(directory / "trace_1_independent.json")
    family = read(directory / "family_summary.json")
    cpu = read(directory / "cpu_waits_stage0.json")
    timing = read(directory / "summary.json")
    hardware = read(directory / "capture_hardware.json")
    assert family["classification_version"] == 2
    assert trace["phase_summary"]["decode_only"]["conservative_complete_interior_steps"] >= 5
    assert trace["phase_summary"]["mixed"]["conservative_complete_interior_steps"] >= 3
    assert trace["exact_duplicate_kernel_records"] == 0
    assert not trace["runtime_correlation_collisions"]
    overlap = timing["cross_stage_overlap"][0]
    assert overlap.get("overlap_verified")
    # Hardware sample rows contain raw timestamps but no identity fields.
    captures.append({
        "requested_concurrency": concurrency, "selected_attempt": name,
        "stage0_trace_sha256": trace["sha256"], "stage1_trace_sha256": codec["sha256"],
        "stage0_total_kernel_events": trace["kernel_events"],
        "stage0_unassigned_kernel_count": sum(trace["unassigned_kernel_reasons"].values()),
        "stage0_unassigned_kernel_time_pct": 100*trace["unassigned_kernel_sum_us"]/trace["kernel_sum_us_all"],
        "stage0_graph_launches": len(trace["graph_launches"]),
        "stage0_graph_launches_without_correlated_kernels": sum(x["correlated_kernel_count"] == 0 for x in trace["graph_launches"]),
        "complete_codec_nonempty_scopes": sum(v["conservative_complete_interior_steps"] for v in codec["phase_summary"].values()),
        "same_host_and_clock_anchor_check": overlap["same_host_and_epoch_anchors"],
        "common_capture_seconds": overlap["common_capture_window_us"]/1e6,
        "kernel_busy_overlap_ms": overlap["kernel_busy_overlap_us"]/1000,
        "overlap_pct_of_common_capture": overlap["overlap_pct_of_common_capture"],
        "stage0_capture_hardware": {k:v for k,v in hardware["windows"]["0"].items() if k!="sample_rows"},
    })
    for phase in ("decode_only", "mixed"):
        s = trace["phase_summary"][phase]
        c = cpu["phases"][phase]
        batches = [int(batch) for batch,count in s["generation_request_histogram"].items() for _ in range(count)]
        gdn_op = [o for o in c["cpu_op_inclusive_table_not_additive"] if o["name"]=="vllm::qwen_gdn_attention_core"]
        rows.append({
            "requested_concurrency": concurrency, "phase": phase,
            "complete_interior_steps": s["conservative_complete_interior_steps"],
            "actual_generation_requests": stats(batches),
            "actual_context_tokens": s["context_tokens"],
            "actual_generation_request_histogram": s["generation_request_histogram"],
            "actual_context_request_histogram": s["context_request_histogram"],
            "graph_launches_per_step_histogram": s["graph_launches_per_step"],
            "gpu_span_ms": ms(s["gpu_span_us"]),
            "gpu_kernel_sum_ms_per_step": ms(s["gpu_kernel_sum_us_per_step"]),
            "gpu_busy_union_ms_per_step": ms(s["gpu_busy_union_us_per_step"]),
            "cpu_execute_scope_ms": ms(c["cpu_scope_us"]),
            "cpu_gdn_scope_union_ms": ms(c["selected_cpu_op_union_us_per_step"]["vllm::qwen_gdn_attention_core"]) if gdn_op else None,
            "cpu_runtime_api_union_ms": ms(c["runtime_all_union_us_per_step"]),
            "cpu_synchronization_api_union_ms": ms(c["runtime_group_union_us_per_step"]["synchronization_API"]),
            "GPU_DtoH_transfers": {k:v for k,v in c["correlated_GPU_copy_directions"].items() if "DtoH" in k},
            "kernel_time_share_pct_by_name_heuristic": {k:v["kernel_time_share_pct"] for k,v in family["phases"][phase]["families"].items()},
        })
result = {
    "scope": "A100 80GB PCIe, cu129/R550. Diagnostic torch/CUPTI profiles, separate from B1 throughput.",
    "selected_profiles": captures,
    "phase_rows": rows,
    "measurement_definitions": {
        "complete_steps": "Native CPU execute scopes inside profiler capture; first/last CPU-thread scopes removed conservatively; all attributed kernels must be capture-contained.",
        "GPU_span": "First to last correlated kernel in one step; includes gaps and can outlive CPU dispatch.",
        "GPU_kernel_sum": "Sum of correlated kernel durations; overlapping kernels count separately. Family fractions use this denominator, not walltime.",
        "GPU_busy_union": "Union of kernel intervals within a step, with overlap counted once.",
        "CPU_scope": "Inclusive host execute or GDN scope, including nested dispatch and profiler overhead. GDN CPU scope is unavailable when only graph replay is visible; not proof of no GDN work.",
        "CPU_runtime_union": "Nonduplicated runtime/driver API intervals inside each complete CPU scope. Parent/child CPU operation times must not be added.",
        "cross_stage_overlap": "Kernel interval overlap only inside the validated common capture window. Does not estimate MPS slowdown.",
    },
    "interpretation_limits": [
        "Concurrency labels are request limits, not constant generation batch size; actual generation requests and prefill-token distributions differ.",
        "Mixed-step latency at C8 versus C16/C64 is not a speed comparison without matching token mix, batch, clock and profiling overhead.",
        "Capture temperatures/clocks differ from B1 and across profiles. Twenty seconds of requests is not thermal steady state.",
        "No explicit synchronize API or DtoH in a scope does not prove all execution is wait-free; only observed evidence is reported.",
        "No speedup, removable-time estimate, Amdahl bound or bottleneck causality is inferred from kernel names or inclusive CPU time.",
        "Capture-contained steps cannot exclude silent CUPTI activity loss; retained logs and correlation checks supply additional evidence.",
    ],
}
(ROOT / "INDEPENDENT_PROFILE_FINDINGS.json").write_text(json.dumps(result, indent=2)+"\n")
lines = [
    "# Independent profile evidence", "",
    result["scope"], "",
    "C8 uses the extended stage-1 capture (attempt 2). All rows use complete interior steps. "
    "GPU span, GPU kernel-duration sum and inclusive CPU scope are different quantities.", "",
    "| C | Phase | Steps | Actual generation requests min/p50/max | Context tokens min/p50/max | GPU span p50/p95 ms | Kernel sum p50 ms | CPU execute p50 ms | CPU GDN union p50 ms |",
    "| ---: | --- | ---: | --- | --- | --- | ---: | ---: | ---: |",
]
def triple(s):
    return f'{s["min"]:g}/{s["p50"]:g}/{s["max"]:g}'
for r in rows:
    gdn = "-" if r["cpu_gdn_scope_union_ms"] is None else f'{r["cpu_gdn_scope_union_ms"]["p50"]:.3f}'
    lines.append(f'| {r["requested_concurrency"]} | {r["phase"]} | {r["complete_interior_steps"]} | '
                 f'{triple(r["actual_generation_requests"])} | {triple(r["actual_context_tokens"])} | '
                 f'{r["gpu_span_ms"]["p50"]:.3f}/{r["gpu_span_ms"]["p95"]:.3f} | '
                 f'{r["gpu_kernel_sum_ms_per_step"]["p50"]:.3f} | {r["cpu_execute_scope_ms"]["p50"]:.3f} | {gdn} |')
lines += ["", "Kernel-family shares below use summed kernel durations, **not wall time**. Dense GEMM names cannot distinguish backbone from codebook tower.", "",
          "| C | Phase | Marlin % | Dense GEMM/GEMV % | BLAS reductions % | GDN/helper % |",
          "| ---: | --- | ---: | ---: | ---: | ---: |"]
for r in rows:
    families = r["kernel_time_share_pct_by_name_heuristic"]
    keys = ("Marlin_MoE_GEMM","dense_GEMM_GEMV_heuristic","BLAS_splitK_and_GEMV_reductions","GDN_and_prefill_helpers_heuristic")
    lines.append(f'| {r["requested_concurrency"]} | {r["phase"]} | ' + " | ".join(f"{families.get(k,0):.2f}" for k in keys) + " |")
lines += ["", "| C | Capture SM min/p50/max MHz | Temperature min/p50/max °C | SW thermal samples | Common capture s | GPU overlap ms |",
          "| ---: | --- | --- | --- | ---: | ---: |"]
for c in captures:
    h=c["stage0_capture_hardware"];t=h["reason_flags"]["sw_thermal_slowdown"]
    lines.append(f'| {c["requested_concurrency"]} | {triple(h["sm_clock_mhz"])} | {triple(h["temperature_c"])} | '
                 f'{t["active_samples"]}/{t["valid_samples"]} | {c["common_capture_seconds"]:.3f} | {c["kernel_busy_overlap_ms"]:.3f} |')
lines += ["", "These are diagnostic observations under unmatched thermal conditions. "
          "Differences in mixed-step times do not establish an improvement; no Amdahl or speedup claim is made. "
          "CPU GDN durations include nested execution and profiler overhead. They must not be added to their nested FLA function times. "
          "Cross-stage overlap establishes concurrent device activity, not a percentage slowdown caused by MPS.", "",
          "Exact distributions, correlation coverage, hardware samples and measurement definitions are retained in "
          "[INDEPENDENT_PROFILE_FINDINGS.json](INDEPENDENT_PROFILE_FINDINGS.json)."]
(ROOT / "INDEPENDENT_PROFILE_FINDINGS.md").write_text("\n".join(lines)+"\n")
print(json.dumps({"phase_rows":len(rows),"profile_runs":len(captures),"out":str(ROOT)}))
