"""CPU-only runtime API accounting within independently validated execute scopes."""
import argparse
import bisect
import collections
import gzip
import json
from pathlib import Path
import re
from audit_trace import merged, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--independent", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    independent = json.loads(args.independent.read_text())
    steps = [s for s in independent["steps"] if not s["excluded_reasons"]]
    bythread = collections.defaultdict(list)
    for i,s in enumerate(steps):
        bythread[(s["pid"],s["tid"])].append(i)
    starts = {k:[steps[i]["ts"] for i in ids] for k,ids in bythread.items()}
    def scope(e):
        key = (e.get("pid"),e.get("tid"))
        if key not in starts:
            return None
        j = bisect.bisect_right(starts[key],e["ts"])-1
        if j >= 0:
            i = bythread[key][j]
            if e["ts"]+e.get("dur",0) <= steps[i]["end"]:
                return i
        return None
    with (gzip.open(args.trace,"rt") if args.trace.suffix==".gz" else args.trace.open()) as f:
        events = json.load(f)["traceEvents"]
    runtime, cpuops, correlations = [], [], {}
    runtime_by_step = collections.defaultdict(list)
    cpu_by_step = collections.defaultdict(list)
    for e in events:
        if e.get("ph")!="X":
            continue
        if e.get("cat") in ("cuda_runtime","cuda_driver"):
            i = scope(e)
            corr = e.get("args",{}).get("correlation")
            if corr is not None:
                correlations[corr] = i
            if i is not None:
                runtime.append((i,e))
                runtime_by_step[i].append(e)
        elif e.get("cat")=="cpu_op":
            i = scope(e)
            if i is not None:
                cpuops.append((i,e))
                cpu_by_step[i].append(e)
    copies = []
    for e in events:
        if e.get("ph")=="X" and e.get("cat")=="gpu_memcpy":
            i = correlations.get(e.get("args",{}).get("correlation"))
            if i is not None:
                copies.append((i,e))
    def group_name(name):
        if "Synchronize" in name:
            return "synchronization_API"
        if "Memcpy" in name:
            return "memory_copy_API"
        if "Launch" in name:
            return "kernel_or_graph_launch_API"
        return "other_runtime_API"
    def union_duration(es):
        return sum(b-a for a,b in merged([(e["ts"],e["ts"]+e["dur"]) for e in es]))
    result = {}
    for phase in ("decode_only","mixed","prefill_only"):
        selected = [i for i,s in enumerate(steps) if s["phase"]==phase]
        selected_set = set(selected)
        apis = [(i,e) for i,e in runtime if i in selected_set]
        ops = [(i,e) for i,e in cpuops if i in selected_set]
        memcpy = [(i,e) for i,e in copies if i in selected_set]
        api_names = collections.defaultdict(list)
        op_names = collections.defaultdict(list)
        api_groups = collections.defaultdict(list)
        for _,e in apis:
            api_names[e["name"]].append(e)
            api_groups[group_name(e["name"])].append(e)
        for _,e in ops:
            op_names[e["name"]].append(e)
        api_table = [
            {"name":name,"count":len(es),"inclusive_duration_us":sum(e["dur"] for e in es),
             "same_name_interval_union_us":union_duration(es),"call_duration_us":stats([e["dur"] for e in es])}
            for name,es in sorted(api_names.items(),key=lambda kv:-sum(e["dur"] for e in kv[1]))
        ]
        op_table = [
            {"name":name,"count":len(es),"inclusive_duration_us":sum(e["dur"] for e in es),
             "call_duration_us":stats([e["dur"] for e in es])}
            for name,es in sorted(op_names.items(),key=lambda kv:-sum(e["dur"] for e in kv[1]))
        ]
        group_per_step = {
            g:stats([union_duration([e for e in runtime_by_step[i] if group_name(e["name"])==g]) for i in selected])
            for g in ("synchronization_API","memory_copy_API","kernel_or_graph_launch_API","other_runtime_API")
        }
        sync_details = []
        for i,e in sorted([(i,e) for i,e in apis if group_name(e["name"])=="synchronization_API"],key=lambda v:-v[1]["dur"])[:20]:
            enclosing = [o for o in cpu_by_step[i] if o["ts"]<=e["ts"] and o["ts"]+o["dur"]>=e["ts"]+e["dur"]]
            enclosing.sort(key=lambda o:o["dur"])
            sync_details.append({"step_original_index":steps[i]["index"],"api":e["name"],"duration_us":e["dur"],
                                 "ts":e["ts"],"nearest_enclosing_cpu_ops":[{"name":o["name"],"duration_us":o["dur"]} for o in enclosing[:5]]})
        transfer_names = collections.defaultdict(lambda:{"count":0,"bytes":0,"gpu_copy_duration_us":0.})
        for _,e in memcpy:
            entry=transfer_names[e["name"]]
            entry["count"]+=1
            entry["bytes"]+=e.get("args",{}).get("bytes",0)
            entry["gpu_copy_duration_us"]+=e["dur"]
        result[phase] = {
            "complete_steps":len(selected),"cpu_scope_us":stats([steps[i]["cpu_scope_us"] for i in selected]),
            "runtime_all_union_us_per_step":stats([union_duration(runtime_by_step[i]) for i in selected]),
            "cpu_ops_all_union_us_per_step":stats([union_duration(cpu_by_step[i]) for i in selected]),
            "selected_cpu_op_union_us_per_step": {
                name:stats([union_duration([e for e in cpu_by_step[i] if e["name"]==name]) for i in selected])
                for name in ("vllm::qwen_gdn_attention_core","ChunkGatedDeltaRuleFunction",
                             "vllm::unified_attention_with_output","aten::index","aten::index_put_")
            },
            "runtime_group_union_us_per_step":group_per_step,
            "runtime_API_table":api_table,
            "cpu_op_inclusive_table_not_additive":op_table,
            "suspect_copy_scalar_index_ops": [v for v in op_table if re.search(r"item|_local_scalar_dense|nonzero|copy|to$|index|masked_fill",v["name"])],
            "largest_sync_calls_with_enclosing_ops":sync_details,
            "correlated_GPU_copy_directions":dict(transfer_names),
        }
    args.out.write_text(json.dumps({
        "trace":str(args.trace),"independent_scopes":str(args.independent),
        "denominator":"Complete interior CPU execute scopes; same-thread runtime API intervals are unioned to avoid nested driver/runtime double counting.",
        "limits":[
            "CPU operation durations are inclusive and nested; never add them across names.",
            "Runtime groups may overlap one another; all-runtime union is the non-duplicated total.",
            "Synchronize API duration is observed CPU blocking, not proof all of it is necessary or independently removable.",
            "GPU memcpy direction comes from correlated CUPTI gpu_memcpy events, not inference from aten::to.",
            "Unexplained CPU scope time is not automatically CPU idle time; includes Python/operator dispatch and profiler overhead.",
        ],"phases":result},indent=2)+"\n")
    print(json.dumps({p:{"steps":v["complete_steps"],"cpu_scope_us":v["cpu_scope_us"],"runtime_groups":v["runtime_group_union_us_per_step"],"gpu_copies":v["correlated_GPU_copy_directions"]} for p,v in result.items()}))


if __name__=="__main__":
    main()
