"""CPU-only descriptive four-control table plus common-shape probe comparison."""
import csv
import hashlib
import json
from pathlib import Path
from audit_service_control import records, stats

ROOT=Path(__file__).parent.parent
OUT=ROOT/"independent-audit"
MODES=["baseline_laps","baseline_no_laps","installed","direct_out_laps"]
PROBES=[m for m in MODES if m!="installed"]
audits={m:json.loads((OUT/f"step3_{m}_cpu_audit.json").read_text()) for m in MODES}
assert all(a["valid"] for a in audits.values())
metadata={m:json.loads((ROOT/"service_controls"/m/"run_metadata.json").read_text()) for m in MODES}
reference=metadata["baseline_laps"]
matched_fields={k:all(v[k]==reference[k] for v in metadata.values()) for k in
                ("tool_snapshot_commit","server_commit","function_sha256",
                 "client_affinity","server_affinity","packages","server_argv")}
assert all(matched_fields.values())
assert all(set(v["files"])==set(reference["files"]) for v in metadata.values())
runner_path=str(ROOT/"step3-preparation/run_service_controls.py")
archive=ROOT/"step3-preparation/revisions/before_control_watchdog/run_service_controls.py"
archived_runner_sha=hashlib.sha256(archive.read_bytes()).hexdigest()
assert archived_runner_sha==reference["files"][runner_path]
file_differences={}
for path,digest in reference["files"].items():
    values={mode:metadata[mode]["files"][path] for mode in MODES}
    if len(set(values.values()))>1:
        assert path==runner_path, "Unexpected computation/source difference across controls"
        file_differences["external_service_runner"]=values
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==values["direct_out_laps"]
    else:
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest, "Source bytes changed after execution"
assert set(file_differences)=={"external_service_runner"}
matched_env={k:len({m["environment"].get(k) for m in metadata.values()})==1 for k in
             ("CUDA_VISIBLE_DEVICES","CUDA_HOME","LD_LIBRARY_PATH","OMP_NUM_THREADS","MAX_JOBS",
              "TRITON_CACHE_DIR","TORCHINDUCTOR_CACHE_DIR","VLLM_CACHE_ROOT","CUDA_CACHE_PATH")}
assert all(matched_env.values())
grouped={}
for mode in PROBES:
    a=audits[mode];lo,hi=a["official_conservative_window"];groups={}
    for path in sorted((ROOT/"service_controls"/mode/"probe").glob("gdn_probe_*.jsonl")):
        for r in records(path):
            if r["pid"]==a["stage0_pid"] and lo<=r["t"] and r["t"]+r["span_ns"]/1e9<=hi:
                assert r["calls"]==30
                key=(r["num_prefills"],r["prefill_tokens"],r["decode_tokens"])
                groups.setdefault(key,[]).append(r)
    grouped[mode]=groups
common=set.intersection(*(set(grouped[m]) for m in PROBES))
shape_rows=[]
for shape in sorted(common):
    row={"num_prefills":shape[0],"prefill_tokens":shape[1],"decode_tokens":shape[2],"groups":{}}
    for mode in PROBES:
        rr=grouped[mode][shape]
        row["groups"][mode]={
            "n":len(rr),"host_ms":stats([r["sum_ns"]/1e6 for r in rr]),
            "first_to_last_call_host_span_ms":stats([r["span_ns"]/1e6 for r in rr]),
            "sum_over_span_pct":stats([100*r["sum_ns"]/r["span_ns"] for r in rr]),
        }
    row["any_group_below_3_samples"]=any(v["n"]<3 for v in row["groups"].values())
    shape_rows.append(row)
coverage={}
for mode in PROBES:
    total=sum(len(v) for v in grouped[mode].values())
    retained=sum(len(grouped[mode][key]) for key in common)
    coverage[mode]={"total_complete_steps":total,"unique_shapes":len(grouped[mode]),
                    "steps_in_three_way_common_shapes":retained,
                    "common_shape_coverage_pct":100*retained/total}
table=[]
for mode in MODES:
    a=audits[mode];p=a["performance"];h=a["timed_hardware"];s=a.get("official_complete_scope")
    table.append({
        "control":mode,"requests":a["requests"],"failed":a["failed"],
        "GDN_host_ms_p50":s["GDN_host_ms"]["p50"] if s else None,
        "GDN_sum_over_span_pct_p50":s["GDN_sum_over_first_to_last_call_host_span_pct"]["p50"] if s else None,
        "complete_probe_steps":s["steps"] if s else None,
        "prefill_tokens_p50":s["prefill_tokens"]["p50"] if s else None,
        "decode_tokens_p50":s["decode_tokens"]["p50"] if s else None,
        "sections_mean_ms":s["sections_mean_ms"] if s else {},
        "throughput_audio_s_per_s":p["throughput_audio_s_per_s"],
        "TTFA_p50_ms":p["ttfa_ms"]["p50"],"TTFA_p95_ms":p["ttfa_ms"]["p95"],
        "RTF_mean":p["rtf"]["mean"],"stall_0ms_pct":p["stall"]["0ms"]["streams_stalled_pct"],
        "long_take_audio_share_pct":p["long_takes"]["audio_share_pct"],
        "SM_MHz_p50":h["SM_MHz"]["p50"],"core_C_range":[h["core_C"]["min"],h["core_C"]["max"]],
        "HBM_C_range":[h["HBM_C"]["min"],h["HBM_C"]["max"]],
        "sw_thermal_sample_pct":h["sw_thermal_slowdown"]["sample_pct"],
        "sw_power_cap_sample_pct":h["sw_power_cap"]["sample_pct"],
    })
result={
    "provenance":{"tool_commit":reference["tool_snapshot_commit"],"server_commit":reference["server_commit"],
                  "function_sha256":reference["function_sha256"],"packages":reference["packages"],
                  "matched_metadata_fields":matched_fields,"matched_environment_fields":matched_env,
                  "source_files_rehashed":len(reference["files"]),"server_CPU_affinity":reference["server_affinity"],
                  "client_CPU_affinity":reference["client_affinity"],"physical_GPU":"1",
                  "user_computation_files_identical":True,
                  "external_runner_revisions":file_differences,
                  "external_runner_revision_review":"After baseline_laps, external supervisor added BLOCK_NEXT existence checks in its 1s startup/client polls and automatic original-summary export after cleanup. Archived prior runner hash verified; user benchmark/probe/server computation unchanged."},
    "runs":table,"common_shape_coverage":coverage,"common_shapes":shape_rows,
    "limits":[
        "Original user summarize_probe outputs are unchanged. Independent summaries additionally require complete30-call scopes and validate flushed post-main records.",
        "Each control was run once. Do not claim stable speedup, statistical equivalence, or zero probe overhead.",
        "Common-shape comparison uses the intersection of(num_prefills,prefill_tokens,decode_tokens) across all three probe controls. Sparse cells are retained with counts, not treated as significant.",
        "Equal shape is not a pairing of identical requests, random outputs, expert routing or thermal conditions.",
        "sum/span is computed per step then summarized. Span covers the host interval from first to last GDN call, not a complete mixed step and not GPU time.",
        "Host measurements include launch-side blocking when present; host≈GPU event span alone is not definitive causality.",
        "End-to-end throughput includes long generated audio. No audio was saved, so this experiment supplies no WER/SIM comparison.",
    ],
}
(OUT/"SERVICE_CONTROLS_COMPARISON.json").write_text(json.dumps(result,indent=2)+"\n")
with (OUT/"common_shape_host.csv").open("w",newline="") as f:
    keys=["num_prefills","prefill_tokens","decode_tokens"]+[f"{m}_{metric}" for m in PROBES for metric in ("n","host_ms_p50","sum_over_span_pct_p50")]
    writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
    for r in shape_rows:
        flat={k:r[k] for k in keys[:3]}
        for m in PROBES:
            v=r["groups"][m];flat.update({f"{m}_n":v["n"],f"{m}_host_ms_p50":v["host_ms"]["p50"],f"{m}_sum_over_span_pct_p50":v["sum_over_span_pct"]["p50"]})
        writer.writerow(flat)
md=["# Service controls: descriptive results","",
    "C16 ×256 requests per fresh service, plus8 excluded warmups and33 C1 drain requests outside the main window. "
    "No torch profiler, no audio saving, no explicit request seed. Each control has one run.","",
    "| Control | Complete probe steps | GDN host p50 ms | Stepwise GDN sum/span p50 % | Audio s/s | TTFA p50/p95 ms | Mean RTF | SM median MHz |",
    "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |"]
for r in table:
    f=lambda x:"—" if x is None else f"{x:.3f}"
    md.append(f'| {r["control"]} | {r["complete_probe_steps"] or "—"} | {f(r["GDN_host_ms_p50"])} | {f(r["GDN_sum_over_span_pct_p50"])} | {r["throughput_audio_s_per_s"]:.2f} | {r["TTFA_p50_ms"]:.1f}/{r["TTFA_p95_ms"]:.1f} | {r["RTF_mean"]:.4f} | {r["SM_MHz_p50"]:g} |')
md+=["","The sum/span denominator is only the host span from first to last GDN call. "
     "It is neither complete-step wall time nor GPU time.","",
     "| Control | Core range °C | HBM range °C | SM median MHz | SW thermal samples % | SW power-cap samples % |",
     "| --- | --- | --- | ---: | ---: | ---: |"]
for r in table:
    md.append(f'| {r["control"]} | {r["core_C_range"][0]:g}–{r["core_C_range"][1]:g} | '
              f'{r["HBM_C_range"][0]:g}–{r["HBM_C_range"][1]:g} | {r["SM_MHz_p50"]:g} | '
              f'{r["sw_thermal_sample_pct"]:.2f} | {r["sw_power_cap_sample_pct"]:.2f} |')
md+=["","These thermal/clock conditions were not matched; end-to-end differences cannot be assigned solely to the probe or direct-output change.","",
     "| Lapped control | Chunk mean ms | Gather mean ms | Scatter mean ms | Copy mean ms |",
     "| --- | ---: | ---: | ---: | ---: |"]
for r in table:
    sec=r["sections_mean_ms"]
    if sec:
        md.append(f'| {r["control"]} | {sec["chunk"]:.3f} | {sec["gather"]:.3f} | {sec["scatter"]:.3f} | {sec["copy"]:.3f} |')
md+=["",
     f"Three-way common shapes: **{len(common)}**; "
     f"**{sum(r['any_group_below_3_samples'] for r in shape_rows)}** have fewer than3 samples in at least one group.",""]
for mode,c in coverage.items():
    md.append(f'- {mode}: {c["steps_in_three_way_common_shapes"]}/{c["total_complete_steps"]} steps covered ({c["common_shape_coverage_pct"]:.1f}%).')
md+=["","[Per-shape counts and host p50](common_shape_host.csv) preserve sparse cells; no significance claim is made. "
     "Same shape does not mean the same request or matched hardware conditions. "
     "See [complete comparison](SERVICE_CONTROLS_COMPARISON.json) for temperature, clocks, long-output contribution and section timings. "
     "Single runs cannot establish stable gains or zero measurement overhead."]
(OUT/"SERVICE_CONTROLS_COMPARISON.md").write_text("\n".join(md)+"\n")
print(json.dumps({"controls":len(table),"common_shapes":len(common),"coverage":coverage}))
