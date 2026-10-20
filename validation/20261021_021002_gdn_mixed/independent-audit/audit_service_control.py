"""Independent CPU-only audit; never edits the user's probe or its summaries."""
import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
import re
import statistics
import time


def load(path):
    return json.loads(path.read_text())


def records(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def stats(values):
    values=sorted(values)
    if not values:
        return {"n":0,"mean":None,"p50":None,"p90":None,"min":None,"max":None}
    def q(x):
        return values[round(x*(len(values)-1))]
    return {"n":len(values),"mean":statistics.fmean(values),"p50":q(.5),"p90":q(.9),"min":values[0],"max":values[-1]}


def summarize(recs):
    return {
        "steps":len(recs),"calls_per_step":sorted({r["calls"] for r in recs}),
        "mixed_steps":sum(r["decode_tokens"]>0 for r in recs),
        "prefill_only_steps":sum(r["decode_tokens"]==0 for r in recs),
        "GDN_host_ms":stats([r["sum_ns"]/1e6 for r in recs]),
        "first_to_last_GDN_call_host_span_ms":stats([r["span_ns"]/1e6 for r in recs]),
        "GDN_sum_over_first_to_last_call_host_span_pct":stats([100*r["sum_ns"]/r["span_ns"] for r in recs]),
        "decode_tokens":stats([r["decode_tokens"] for r in recs]),
        "prefill_tokens":stats([r["prefill_tokens"] for r in recs]),
        "prefill_requests":stats([r["num_prefills"] for r in recs]),
        "sections_mean_ms":{k:statistics.fmean(r["sections_ns"].get(k,0)/1e6 for r in recs)
                            for k in ("setup","conv","decode","prep","gather","chunk","scatter","copy")} if recs and recs[0]["laps"] else {},
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",required=True,type=Path)
    ap.add_argument("--mode",required=True,choices=["baseline_laps","baseline_no_laps","installed","direct_out_laps"])
    args=ap.parse_args();root=args.root;run=root/"service_controls"/args.mode
    out=root/"independent-audit"
    s=load(run/"client/summary.json");reqs=records(run/"client/requests.jsonl")
    assert (run/"client/client.exit").read_text().strip()=="0"
    assert [s[k] for k in ("requests","succeeded","failed")]==[256,256,0]
    assert len(reqs)==256 and sorted(r["index"] for r in reqs)==list(range(256))
    assert all(r["status"]==200 and r["ok"] and r.get("seed") is None for r in reqs)
    assert s["concurrency"]==16 and s["workload"]["warmups"]==8
    assert s["workload"]["ref_mode"]=="codes" and s["workload"]["extra_body"]=={}
    assert s["workload"]["seed"] is None and s["workload"]["shuffle_seed"] is None
    assert not (run/"client/audio").exists()
    assert s["client"]["loop_lag_ms"]["p99"]<10
    settings=load(run/"service_settings.json")
    assert settings["no_torch_profiler"] and settings["no_thermal_gate"]
    assert settings["static_installed_function_sha256"]=="9af0a98c23727bea4e7f34c1cf3283d838ad7a84a741b2fa8f946d823a4292a9"
    cleanup=load(run/"supervisor_cleanup.json")
    assert cleanup["clean"] and cleanup["port_free"] and not cleanup["own_processes_remaining"]
    text=(run/"server.log").read_text(errors="replace")
    pids={int(x) for x in re.findall(r"StageEngineCoreProc_stage0_replica0 pid=(\d+)",text)}
    assert len(pids)==1
    stage0=pids.pop()
    assert "NOT patched" not in text and "[drama3-gdn-probe] not installed:" not in text
    a=time.time_ns();p=time.perf_counter_ns();b=time.time_ns()
    after={"wall_before_ns":a,"perf_counter_ns":p,"wall_after_ns":b}
    (out/f"{args.mode}_clock_after.json").write_text(json.dumps(after,indent=2)+"\n")
    before=load(out/"step3_clock_before.json")
    p0,p1=before["perf_counter_ns"]/1e9,p/1e9
    offset0=((before["wall_before_ns"]+before["wall_after_ns"])/2-before["perf_counter_ns"])/1e9
    offset1=((a+b)/2-p)/1e9
    def wall(perf):
        return perf+offset0+(offset1-offset0)*(perf-p0)/(p1-p0)
    actual_lo=wall(min(r["sent_s"] for r in reqs))
    actual_hi=wall(max(r.get("done_s",r["sent_s"]+r["e2e_s"]) for r in reqs))
    second=dt.datetime.strptime(s["started_at"],"%Y-%m-%dT%H:%M:%S%z").timestamp()
    official_lo,official_hi=second+1,second+float(s["duration_s"])
    report={"mode":args.mode,"stage0_pid":stage0,"valid":True,"requests":256,"failed":0,
            "loop_lag_p99_ms":s["client"]["loop_lag_ms"]["p99"],
            "actual_HTTP_window_estimate":[actual_lo,actual_hi],"official_conservative_window":[official_lo,official_hi],
            "clock_offset_change_s":offset1-offset0,"performance":{k:s[k] for k in ("duration_s","throughput_audio_s_per_s","ttfa_ms","rtf","stall","long_takes")},
            "p50_method":"Nearest order statistic, matching user summarize_probe; not interpolated percentile.",
            "boundary_note":"Probe t precedes the first perf_counter timestamp slightly; t+span is an estimated scope end. Actual-window subset uses1ms interior guard; official conservative subset additionally verifies t+span <= hi."}
    paths=sorted((run/"probe").glob("gdn_probe_*.jsonl"))
    if args.mode=="installed":
        assert not paths and not re.search(r"\[drama3-gdn-probe pid=\d+\] patched",text)
        assert not settings["probe_enabled"]
        actual_env=load(run/"actual_stage0_environment.json")
        assert actual_env["stage0_pid"]==stage0 and not actual_env["DRAMA3_GDN_environment"]
        assert actual_env["probe_absent_in_actual_stage0_environment"]
        report["unpatched_control_verified"]=True
        report["actual_stage0_environment_no_probe_verified"]=True
        drain_paths=sorted(run.glob("flush_drain_*"))
        assert drain_paths
        for path in drain_paths:
            d=load(path/"summary.json");rr=records(path/"requests.jsonl")
            assert [d[k] for k in ("requests","succeeded","failed")]==[33,33,0]
            assert d["concurrency"]==1 and d["workload"]["warmups"]==0
            assert len(rr)==33 and all(r["ok"] and r["status"]==200 for r in rr)
        report["drain_batches"]=[p.name for p in drain_paths]
    else:
        variant="direct_out" if args.mode=="direct_out_laps" else "baseline"
        laps=args.mode!="baseline_no_laps"
        assert f"[drama3-gdn-probe pid={stage0}] patched mixed_forward_core: variant={variant} laps={laps}" in text
        byfile={str(path):records(path) for path in paths}
        all_records=[r for rows in byfile.values() for r in rows]
        stage=[r for r in all_records if r["pid"]==stage0]
        assert stage and all(r["variant"]==variant and r["laps"]==laps for r in stage)
        assert all(x["t"]<=y["t"] for x,y in zip(stage,stage[1:]))
        selected=[r for r in stage if official_lo<=r["t"]<=official_hi]
        full=[r for r in selected if r["t"]+r["span_ns"]/1e9<=official_hi]
        actual=[r for r in stage if actual_lo+.001<=r["t"] and r["t"]+r["span_ns"]/1e9<=actual_hi-.001]
        assert selected and full and actual
        assert all(r["calls"]==30 for r in selected+actual)
        for r in selected+actual:
            assert math.isfinite(r["sum_ns"]) and r["span_ns"]>=r["sum_ns"]>0
            assert r["num_prefills"]>0 and r["prefill_tokens"]>0 and r["decode_tokens"]>=0
            assert sum(r["sections_ns"].values())==r["sum_ns"] if laps else not r["sections_ns"]
        completed=load(run/"bench_boundaries.json")["main_client_completed_wall_s"]
        post=[r for r in stage if r["t"]>completed]
        assert post and all(r["calls"]==30 for r in post)
        drains=[]
        for path in sorted(run.glob("flush_drain_*")):
            drain=load(path/"summary.json");rr=records(path/"requests.jsonl")
            assert [drain[k] for k in ("requests","succeeded","failed")]==[33,33,0]
            assert drain["concurrency"]==1 and drain["workload"]["warmups"]==0
            assert len(rr)==33 and all(r["ok"] and r["status"]==200 for r in rr)
            drains.append(str(path.name))
        assert drains
        official=load(run/"official_probe_summary.json")
        assert len(official)==1
        derived=summarize(selected);o=official[0]
        assert o["steps"]==derived["steps"] and o["calls_per_step"]==[30]
        assert o["variants"]==[variant] and o["laps"]==[laps]
        for q in ("mean","p50","p90"):
            assert math.isclose(o["gdn_ms_per_step"][q],derived["GDN_host_ms"][q],rel_tol=1e-12)
        for q in ("mean","p50"):
            assert math.isclose(o["span_ms_per_step"][q],derived["first_to_last_GDN_call_host_span_ms"][q],rel_tol=1e-12)
        if laps:
            assert all(math.isclose(o["section_ms_per_step"][k],v,rel_tol=1e-12) for k,v in derived["sections_mean_ms"].items())
        report.update(expected_variant=variant,expected_laps=laps,
                      original_official_summary_independently_reproduced=True,
                      all_stage0_records=len(stage),other_PID_records=len(all_records)-len(stage),
                      official_start_only=summarize(selected),official_complete_scope=summarize(full),
                      actual_complete_scope=summarize(actual),
                      official_boundary_crossing_scopes=len(selected)-len(full),
                      persisted_post_client_completion_records=len(post),drain_batches=drains,
                      flush_proof="Sequential stage0 append contains complete records after main client completion, proving main-window buffered records crossed a flush boundary. Drain's entire tail need not be persisted.")
    samples=[]
    with (run/"gpu_thermal.csv").open() as f:
        for raw in csv.DictReader(f,skipinitialspace=True):
            t=dt.datetime.strptime(raw["timestamp"],"%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=dt.timezone(dt.timedelta(hours=8))).timestamp()
            if actual_lo<=t<=actual_hi:
                samples.append(raw)
    assert samples
    report["timed_hardware"]={"samples":len(samples)}
    for key,outkey in [("clocks.current.sm [MHz]","SM_MHz"),("temperature.gpu","core_C"),("temperature.memory","HBM_C")]:
        report["timed_hardware"][outkey]=stats([float(r[key].split()[0]) for r in samples])
    for reason in ("sw_thermal_slowdown","hw_thermal_slowdown","sw_power_cap"):
        active=sum(r.get("clocks_event_reasons."+reason,r.get("clocks_throttle_reasons."+reason))=="Active" for r in samples)
        report["timed_hardware"][reason]={"active_samples":active,"sample_pct":100*active/len(samples)}
    (out/f"step3_{args.mode}_cpu_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"mode":args.mode,"valid":True,"out":str(out/f"step3_{args.mode}_cpu_audit.json"),
                      "official_complete":report.get("official_complete_scope"),"hardware":report["timed_hardware"]}))


if __name__=="__main__":
    main()
