"""Independent CPU-only audit of real torch/CUPTI profiles.

No imported production analyzer. Preserve unassigned activity, reject ambiguous
correlations, exclude each CPU thread's first/last execute scopes conservatively.
"""
import argparse
import bisect
import collections
import gzip
import hashlib
import json
from pathlib import Path
import re

PATTERN = re.compile(r"^execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)$")


def merged(intervals):
    result = []
    for a, b in sorted(intervals):
        if result and a <= result[-1][1]:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return result


def overlap(a, b):
    i = j = 0
    total = 0.
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def stats(values):
    if not values:
        return {"n": 0, "p50": None, "p95": None}
    values = sorted(values)
    def q(frac):
        p = (len(values) - 1) * frac
        lo = int(p)
        return values[lo] + (values[min(lo + 1, len(values)-1)] - values[lo]) * (p-lo)
    return {"n": len(values), "p50": q(.5), "p95": q(.95), "min": values[0], "max": values[-1]}


def audit(path):
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as f:
        trace = json.load(f)
    events = trace["traceEvents"]
    capture = [e for e in events if e.get("cat") == "Trace" and e.get("ph") == "X" and e.get("name", "").startswith("PyTorch Profiler")]
    bounds = [(e["ts"], e["ts"]+e["dur"]) for e in capture]
    if len(bounds) != 1:
        raise ValueError(f"Need one unambiguous profiler capture scope: {bounds}")
    capture_start, capture_end = bounds[0]
    steps = []
    gpu_clones = 0
    for event in events:
        match = PATTERN.match(event.get("name", ""))
        if not match:
            continue
        if event.get("cat") == "gpu_user_annotation":
            gpu_clones += 1
        if event.get("ph") != "X" or event.get("cat") != "user_annotation":
            continue
        context, context_tokens, generation, generation_tokens = map(int, match.groups())
        steps.append({
            "ts": event["ts"], "end": event["ts"]+event["dur"], "cpu_scope_us": event["dur"],
            "pid": event["pid"], "tid": event["tid"],
            "phase": "mixed" if context and generation else "prefill_only" if context else "decode_only" if generation else "empty",
            "context_requests": context, "context_tokens": context_tokens,
            "generation_requests": generation, "generation_tokens": generation_tokens,
            "kernels": [], "graph_launches": [],
        })
    steps.sort(key=lambda s: s["ts"])
    threads = collections.defaultdict(list)
    for index, step in enumerate(steps):
        threads[(step["pid"], step["tid"])].append(index)
    starts = {key: [steps[i]["ts"] for i in ids] for key, ids in threads.items()}
    boundary = {i for ids in threads.values() for i in (ids[0], ids[-1])}
    def containing(event):
        key = (event.get("pid"), event.get("tid"))
        if key not in threads:
            return None
        pos = bisect.bisect_right(starts[key], event["ts"]) - 1
        if pos >= 0:
            i = threads[key][pos]
            if event["ts"] <= steps[i]["end"]:
                return i
        return None
    runtime = collections.defaultdict(list)
    graph_events = []
    for event in events:
        if event.get("ph") == "X" and event.get("cat") in ("cuda_runtime", "cuda_driver"):
            corr = event.get("args", {}).get("correlation")
            if corr is not None:
                runtime[corr].append((containing(event), event))
            if "GraphLaunch" in event.get("name", ""):
                graph_events.append(event)
                i = containing(event)
                if i is not None:
                    steps[i]["graph_launches"].append(corr)
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    kernel_by_corr = collections.defaultdict(list)
    signatures = collections.Counter()
    unassigned = collections.Counter()
    kernel_step = []
    for event in kernels:
        args = event.get("args", {})
        corr = args.get("correlation")
        kernel_by_corr[corr].append(event)
        signatures[(event["name"], event["ts"], event.get("dur"), event.get("pid"), event.get("tid"), corr, args.get("graph node id"))] += 1
        choices = {i for i, _ in runtime.get(corr, [])}
        if not choices:
            index = None
            unassigned["no_runtime_correlation"] += 1
        elif len(choices) != 1:
            index = None
            unassigned["ambiguous_runtime_scope"] += 1
        else:
            index = next(iter(choices))
            if index is None:
                unassigned["runtime_launch_outside_execute_scope"] += 1
        kernel_step.append(index)
        if index is not None:
            steps[index]["kernels"].append(event)
    duplicate_runtime = [
        {"correlation": corr, "events": [{"name": e["name"], "pid": e["pid"], "tid": e["tid"], "scope": i} for i, e in values]}
        for corr, values in runtime.items() if len(values) > 1
    ]
    reasons = collections.Counter()
    complete = set()
    for i, step in enumerate(steps):
        excluded = []
        if i in boundary:
            excluded.append("first_or_last_cpu_thread_scope_conservatively_excluded")
        if step["ts"] < capture_start or step["end"] > capture_end:
            excluded.append("cpu_scope_outside_capture")
        if not step["kernels"]:
            excluded.append("no_correlated_kernels")
        if any(e["ts"] < capture_start or e["ts"]+e["dur"] > capture_end for e in step["kernels"]):
            excluded.append("correlated_kernel_outside_capture")
        step["excluded_reasons"] = excluded
        reasons.update(excluded)
        if not excluded:
            complete.add(i)
    phases = {}
    tables = {}
    for phase in ("decode_only", "mixed", "prefill_only", "empty"):
        selected = [i for i in complete if steps[i]["phase"] == phase]
        phase_kernels = [e for i in selected for e in steps[i]["kernels"]]
        names = collections.defaultdict(lambda: {"count": 0, "kernel_sum_us": 0.})
        for event in phase_kernels:
            names[event["name"]]["count"] += 1
            names[event["name"]]["kernel_sum_us"] += event["dur"]
        table = [dict(name=name, **values) for name, values in sorted(names.items(), key=lambda kv: -kv[1]["kernel_sum_us"])]
        total = sum(e["dur"] for e in phase_kernels)
        spans, busy_sums, kernel_sums = [], [], []
        for i in selected:
            ks = steps[i]["kernels"]
            spans.append(max(e["ts"]+e["dur"] for e in ks)-min(e["ts"] for e in ks))
            busy_sums.append(sum(b-a for a,b in merged([(e["ts"],e["ts"]+e["dur"]) for e in ks])))
            kernel_sums.append(sum(e["dur"] for e in ks))
        phases[phase] = {
            "all_cpu_scopes": sum(s["phase"] == phase for s in steps),
            "conservative_complete_interior_steps": len(selected),
            "kernel_count": len(phase_kernels), "kernel_sum_us": total,
            "gpu_span_us": stats(spans), "gpu_busy_union_us_per_step": stats(busy_sums),
            "gpu_kernel_sum_us_per_step": stats(kernel_sums),
            "cpu_scope_us": stats([steps[i]["cpu_scope_us"] for i in selected]),
            "generation_request_histogram": dict(collections.Counter(steps[i]["generation_requests"] for i in selected)),
            "context_request_histogram": dict(collections.Counter(steps[i]["context_requests"] for i in selected)),
            "context_tokens": stats([steps[i]["context_tokens"] for i in selected]),
            "graph_launches_per_step": dict(collections.Counter(len(steps[i]["graph_launches"]) for i in selected)),
            "raw_top10_kernel_time_coverage_pct": 100*sum(t["kernel_sum_us"] for t in table[:10])/total if total else None,
            "gpu_span_by_generation_requests": {
                str(batch): stats([span for i,span in zip(selected,spans) if steps[i]["generation_requests"] == batch])
                for batch in sorted({steps[i]["generation_requests"] for i in selected})
            },
        }
        tables[phase] = table
    all_busy = merged([(e["ts"], e["ts"]+e["dur"]) for e in kernels])
    base = trace.get("baseTimeNanoseconds")
    result = {
        "trace": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "host_name": trace.get("host_name"), "baseTimeNanoseconds": base,
        "deviceProperties": trace.get("deviceProperties"),
        "capture_local_us": [capture_start, capture_end],
        "capture_epoch_us": [capture_start+base/1000, capture_end+base/1000] if isinstance(base, (int, float)) else None,
        "cpu_execute_scopes": len(steps), "gpu_cloned_scopes_excluded": gpu_clones,
        "boundary_exclusion_counts": dict(reasons), "phase_summary": phases,
        "raw_kernel_tables_complete_interior_only": tables,
        "kernel_events": len(kernels), "kernel_sum_us_all": sum(e["dur"] for e in kernels),
        "gpu_busy_union_us_all": sum(b-a for a,b in all_busy),
        "unassigned_kernel_reasons": dict(unassigned),
        "unassigned_kernel_sum_us": sum(e["dur"] for e,i in zip(kernels,kernel_step) if i is None),
        "exact_duplicate_kernel_records": sum(n-1 for n in signatures.values() if n > 1),
        "runtime_correlation_collisions": duplicate_runtime,
        "graph_launches": [{
            "correlation": e.get("args",{}).get("correlation"),
            "native_scope_index": containing(e),
            "correlated_kernel_count": len(kernel_by_corr[e.get("args",{}).get("correlation")]),
            "distinct_graph_nodes": len({k.get("args",{}).get("graph node id") for k in kernel_by_corr[e.get("args",{}).get("correlation")] if k.get("args",{}).get("graph node id") is not None}),
        } for e in graph_events],
        "steps": [{k:v for k,v in step.items() if k!="kernels"} | {"index":i, "kernel_count":len(step["kernels"])} for i,step in enumerate(steps)],
        "limits": [
            "Complete means capture-contained correlated work after conservatively excluding first and last CPU-thread steps; it cannot rule out silent CUPTI drops.",
            "CPU scope is dispatch duration; GPU span includes gaps; kernel sum counts overlapping kernels more than once; union removes overlap.",
            "Kernel category/name alone does not establish backbone/tower ownership, bottleneck causality or a speedup.",
            "GPU cloned annotations never count as CPU steps or kernel events.",
            "Cross-stage timing needs the same host, explicit epoch anchors and an intersecting capture interval.",
            "Native execute scopes exclude sample_tokens outside execute_model; unassigned activity is retained.",
        ],
    }
    return result, all_busy if isinstance(base,(int,float)) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("traces", nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results, intervals = [], []
    for index, path in enumerate(args.traces):
        result, busy = audit(path)
        (args.out / f"trace_{index}_independent.json").write_text(json.dumps(result, indent=2)+"\n")
        results.append({k:v for k,v in result.items() if k not in ("steps","raw_kernel_tables_complete_interior_only","graph_launches","deviceProperties")})
        intervals.append(busy)
    comparisons = []
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            a,b = results[i],results[j]
            verified = bool(a["host_name"] and a["host_name"]==b["host_name"] and a["capture_epoch_us"] and b["capture_epoch_us"])
            record = {"trace_a":i,"trace_b":j,"same_host_and_epoch_anchors":verified}
            if verified:
                # Keep interval arithmetic near local trace times. Adding Unix
                # epoch microseconds to every short kernel loses precision.
                reference_base = min(a["baseTimeNanoseconds"], b["baseTimeNanoseconds"])
                offsets = [(v["baseTimeNanoseconds"] - reference_base)/1000 for v in (a,b)]
                start = max(a["capture_local_us"][0]+offsets[0],b["capture_local_us"][0]+offsets[1])
                end = min(a["capture_local_us"][1]+offsets[0],b["capture_local_us"][1]+offsets[1])
                record["arithmetic_reference_base_ns"] = reference_base
                record["common_capture_window_us"] = max(0,end-start)
                record["overlap_verified"] = end > start
                if end > start:
                    shifted = [[(x+offset,y+offset) for x,y in intervals[k]] for k,offset in zip((i,j),offsets)]
                    clipped = [[(max(x,start),min(y,end)) for x,y in values if max(x,start)<min(y,end)] for values in shifted]
                    shared = overlap(*clipped)
                    record.update(kernel_busy_overlap_us=shared, overlap_pct_of_common_capture=100*shared/(end-start),
                                  each_trace_busy_union_us_in_common_capture=[sum(y-x for x,y in v) for v in clipped],
                                  interpretation="Observed device activity overlap; not evidence for how much MPS slows either stage.")
            comparisons.append(record)
    (args.out/"summary.json").write_text(json.dumps({"traces":results,"cross_stage_overlap":comparisons},indent=2)+"\n")
    print(json.dumps({"traces":len(results),"out":str(args.out),"phase_counts":[{p:v["conservative_complete_interior_steps"] for p,v in r["phase_summary"].items()} for r in results]}))


if __name__ == "__main__":
    main()
