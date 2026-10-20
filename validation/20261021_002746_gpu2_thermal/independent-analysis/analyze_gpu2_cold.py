"""Independent completed-run CPU analysis of GPU2 cold-start C1 telemetry."""
import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import statistics

ZONE = dt.timezone(dt.timedelta(hours=8))
OLD = Path("${NVME_WORKDIR}/baseline/20261020_194158/b1/c1/r1")


def read(path):
    return json.loads(path.read_text())


def stat(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {"n": 0, "min": None, "p50": None, "p95": None, "max": None}
    def q(frac):
        index = (len(values)-1)*frac
        low = int(index)
        return values[low]+(values[min(low+1,len(values)-1)]-values[low])*(index-low)
    return {"n":len(values),"min":values[0],"p50":q(.5),"p95":q(.95),"max":values[-1]}


def iso(epoch):
    return dt.datetime.fromtimestamp(epoch,ZONE).isoformat(timespec="milliseconds")


def load_csv(path):
    rows = []
    with path.open() as f:
        for raw in csv.DictReader(f,skipinitialspace=True):
            raw = {re.sub(r"\s*\[.*\]$","",k.strip()):(v or "").strip() for k,v in raw.items() if k}
            epoch = dt.datetime.strptime(raw["timestamp"],"%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=ZONE).timestamp()
            row = {"timestamp":raw["timestamp"],"epoch_s":epoch}
            for name,aliases in {
                "core_c":["temperature.gpu"],"hbm_c":["temperature.memory"],
                "sm_mhz":["clocks.current.sm","clocks.sm"],"power_w":["power.draw"],
                "memory_used_mib":["memory.used"],"gpu_utilization_pct":["utilization.gpu"],
            }.items():
                value = next((raw[k] for k in aliases if k in raw), None)
                try:
                    row[name] = float(value.split()[0]) if value is not None else None
                except ValueError:
                    row[name] = None
            mask = raw.get("clocks_event_reasons.active",raw.get("clocks_throttle_reasons.active"))
            row["active_mask"] = int(mask,16) if mask and mask.startswith("0x") else None
            for name,bit in (("sw_thermal_slowdown",0x20),("hw_thermal_slowdown",0x40),
                             ("sw_power_cap",0x4),("hw_power_brake_slowdown",0x80)):
                flag=raw.get("clocks_event_reasons."+name,raw.get("clocks_throttle_reasons."+name))
                row[name]=(flag=="Active") if flag in ("Active","Not Active") else (
                    bool(row["active_mask"] & bit) if row["active_mask"] is not None else None)
                if flag in ("Active","Not Active") and row["active_mask"] is not None:
                    assert row[name]==bool(row["active_mask"] & bit), "Reason field disagrees with active bitmask"
            rows.append(row)
    assert rows and all(a["epoch_s"]<b["epoch_s"] for a,b in zip(rows,rows[1:]))
    return rows


def summarize(rows):
    result = {key:stat([r[key] for r in rows]) for key in ("core_c","hbm_c","sm_mhz","power_w","memory_used_mib","gpu_utilization_pct")}
    result["samples"]=len(rows)
    result["sample_gap_s"]=stat([b["epoch_s"]-a["epoch_s"] for a,b in zip(rows,rows[1:])])
    for key in ("sw_thermal_slowdown","hw_thermal_slowdown","sw_power_cap","hw_power_brake_slowdown"):
        values=[r[key] for r in rows if r[key] is not None]
        result[key]={"valid_samples":len(values),"active_samples":sum(values),"sample_pct":100*sum(values)/len(values) if values else None}
    return result


def episodes(rows):
    output=[]; current=None
    for i,row in enumerate(rows):
        active=row["sw_thermal_slowdown"] is True
        gap=i>0 and row["epoch_s"]-rows[i-1]["epoch_s"]>2.5
        if current is not None and (not active or gap):
            current["first_inactive_after_s"]=row["epoch_s"] if not active and not gap else None
            current["end_censored_by_gap"]=gap
            output.append(current); current=None
        if active:
            if current is None:
                current={"first_active_s":row["epoch_s"],"last_active_s":row["epoch_s"],"active_samples":0,
                         "previous_inactive_s":rows[i-1]["epoch_s"] if i and not gap and rows[i-1]["sw_thermal_slowdown"] is False else None}
            current["active_samples"]+=1;current["last_active_s"]=row["epoch_s"]
    if current:
        current["first_inactive_after_s"]=None
        current["end_censored_by_gap"]=False
        output.append(current)
    for ep in output:
        ep["observed_active_span_s"]=ep["last_active_s"]-ep["first_active_s"]
    return output


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,required=True)
    args=ap.parse_args();root=args.root;out=root/"independent-analysis";run=root/"sweep/c1/r1"
    summary=read(run/"summary.json");meta=read(run/"run_metadata.json")
    cleanup=read(run/"supervisor_cleanup.json")
    process_samples=[json.loads(line) for line in (run/"gpu-process-observations.jsonl").read_text().splitlines() if line.strip()]
    assert all(p.get("owned_by_this_run") for sample in process_samples for p in sample["processes"])
    assert not (root/"sweep/BLOCK_NEXT").exists()
    requests=[json.loads(line) for line in (run/"requests.jsonl").read_text().splitlines() if line.strip()]
    assert cleanup["clean"] and cleanup["port_free"] and not cleanup["own_processes_remaining"]
    assert [summary[k] for k in ("requests","succeeded","failed")]==[1088,1088,0]
    assert len(requests)==1088 and sorted(r["index"] for r in requests)==list(range(1088))
    assert len({r["utt_id"] for r in requests})==1088
    assert all(r["ok"] and r["status"]==200 and r.get("seed") is None and not r.get("error") for r in requests)
    assert all(r["sample_rate"]==44100 and r["channels"]==1 and r["bytes"]>0 and r["bytes"]%2==0 for r in requests)
    assert all(abs(r["audio_s"]-r["bytes"]/88200)<.00006 and r["chunk_log"][-1][1]==r["bytes"] for r in requests)
    w=summary["workload"]
    assert all(w.get(k)==v for k,v in {"seed":None,"warmups":8,"shuffle_seed":None,"ref_mode":"codes","rows_timed":1088,"extra_body":{}}.items())
    assert not (run/"audio").exists() and not any("wav" in r for r in requests)
    assert summary["concurrency"]==1 and summary["client"]["loop_lag_ms"]["p99"]<10
    assert meta["client_affinity"]=="60-63" and meta["server_affinity"]=="0-27,32-59"
    affinity=read(out/"live_cpu_affinity.json")
    assert len(affinity["api_and_stage_processes"])==3
    assert all(p["cpu_allowed"]=="0-27,32-59" for p in affinity["api_and_stage_processes"])
    assert len(affinity["benchmark_client_processes"])==1
    assert affinity["benchmark_client_processes"][0]["allowed_cpus"]==[60,61,62,63]
    assert meta["environment"]["CUDA_VISIBLE_DEVICES"]=="2"
    experiment=read(root/"experiment.json");uuid=experiment["gpu_uuid"]
    assert experiment["gpu_index"]==2 and meta["gpu_before"].split(",")[0]==uuid
    log=(run/"server.log").read_text(errors="replace")
    assert f"Started private MPS for {uuid}" in log
    old_summary=read(OLD/"summary.json");old_meta=read(OLD/"run_metadata.json")
    matching={k:meta[k]==old_meta[k] for k in ("server_commit","tool_sha256","model_revision","deploy_sha256","codes_sha256","seed","save_audio","warmups","client_affinity","server_affinity","packages")}
    assert all(matching.values())
    anchors=[read(root/name) for name in ("clock_anchor_before.json","clock_anchor_after.json")]
    perf=[x["perf_counter_ns"]/1e9 for x in anchors]
    offsets=[(x["wall_before_ns"]+x["wall_after_ns"])/2e9-x["perf_counter_ns"]/1e9 for x in anchors]
    def bridge(value):
        weight=(value-perf[0])/(perf[1]-perf[0])
        return value+offsets[0]+weight*(offsets[1]-offsets[0])
    timed_start=bridge(min(r["sent_s"] for r in requests))
    last_audio=bridge(max(r["sent_s"]+r["e2e_s"] for r in requests))
    service_start=read(run/"service_start.json")["epoch_s"]
    observations=[json.loads(line) for line in (root/"phase_observations.jsonl").read_text().splitlines()]
    def observation_time(event):
        a=event["clock_anchor"]
        return (a["wall_before_ns"]+a["wall_after_ns"])/2e9
    warm=next(observation_time(e) for e in observations if e["event"]=="warmup_print_observed")
    independent_health=read(root/"health_ready.json")["epoch_s"]
    rows=load_csv(root/"thermal_continuous.csv")
    # Global runner-ready time has only second precision and precedes the
    # independent health observer; preserve that distinction.
    ready_line=next(line for line in (root/"sweep/sweep.log").read_text().splitlines() if "ready after" in line)
    ready_hms=re.search(r"\[(\d\d:\d\d:\d\d)\]",ready_line)[1]
    ready=dt.datetime.combine(dt.datetime.fromtimestamp(warm,ZONE).date(),dt.time.fromisoformat(ready_hms)).replace(tzinfo=ZONE).timestamp()
    phases=[("startup",service_start,ready),("ready / warmup",ready,timed_start),
            ("timed",timed_start,last_audio),("shutdown / finalization",last_audio,rows[-1]["epoch_s"])]
    for row in rows:
        row["minutes_since_service_start"]=(row["epoch_s"]-service_start)/60
        row["seconds_since_timed_start"]=row["epoch_s"]-timed_start
        row["phase"]=next((name for name,start,end in phases if start<=row["epoch_s"]<end),"before service" if row["epoch_s"]<service_start else "shutdown / finalization")
    timed=[r for r in rows if timed_start<=r["epoch_s"]<=last_audio]
    last5=[r for r in rows if max(timed_start,last_audio-300)<=r["epoch_s"]<=last_audio]
    last3=[r for r in rows if max(timed_start,last_audio-180)<=r["epoch_s"]<=last_audio]
    thermal=episodes(rows)
    first=next((r for r in rows if r["sw_thermal_slowdown"]),None)
    previous=(rows[rows.index(first)-1] if first is not None and rows.index(first)>0 else None)
    threshold_text=(root/"nvidia-smi-before.txt").read_text()
    assert uuid in threshold_text, "Raw pre-run NVIDIA query does not match the selected GPU UUID"
    fan=read(root/"fan_readonly_check.json")
    thresholds={key:int(re.search(pattern+r"\s*:\s*(\d+) C",threshold_text)[1]) for key,pattern in {
        "core_max_operating_c":"GPU Max Operating Temp","core_slowdown_c":"GPU Slowdown Temp",
        "hbm_max_operating_c":"Memory Max Operating Temp"}.items()}
    coincident=[r for r in timed if r["sw_thermal_slowdown"]]
    first_threshold={}
    for key,limit in (("core_c",thresholds["core_max_operating_c"]),("hbm_c",thresholds["hbm_max_operating_c"])):
        found=next((r for r in rows if r[key] is not None and r[key]>=limit),None)
        first_threshold[key]=found
    old_rows=load_csv(OLD/"gpu.csv")
    old_start=dt.datetime.strptime(old_summary["started_at"],"%Y-%m-%dT%H:%M:%S%z").timestamp()
    old_timed=[r for r in old_rows if old_start<=r["epoch_s"]<=old_start+old_summary["duration_s"]]
    old_last5=[r for r in old_timed if r["epoch_s"]>=old_start+old_summary["duration_s"]-300]
    old_first=next((r for r in old_rows if r["sw_thermal_slowdown"]),None)
    def performance(s):
        return {k:s[k] for k in ("duration_s","throughput_audio_s_per_s","ttfa_ms","rtf","stall","long_takes")}
    report={
        "checks":{"requests":1088,"succeeded":1088,"failed":0,"no_audio_saved":True,"seed":None,"warmups":8,
                  "cpu_affinity_matches_B1":True,"CUDA_VISIBLE_DEVICES":"2","private_MPS_matches_physical_GPU2":True,
                  "live_API_stage_client_affinity_independently_verified":True,
                  "cleanup_clean":True,"loop_lag_p99_ms":summary["client"]["loop_lag_ms"]["p99"],
                  "observed_unowned_GPU_processes":0,"GPU_process_observation_samples":len(process_samples),
                  "software_workload_match_fields":matching,
                  "oom_cuda_preemption_log_matches":[line for line in log.splitlines() if re.search(r"outofmemory|out of memory|CUDA error|resuming preempted",line,re.I)]},
        "timing":{"service_start":iso(service_start),"runner_health_ready_second":iso(ready),
                  "independent_health_observation":iso(independent_health),"warmup_print_observed":iso(warm),
                  "estimated_first_timed_HTTP_send":iso(timed_start),"estimated_last_audio_arrival":iso(last_audio),
                  "clock_anchor_offset_change_s":offsets[1]-offsets[0],
                  "anchor_wall_bracket_ns":[x["wall_after_ns"]-x["wall_before_ns"] for x in anchors],
                  "HTTP_span_s":last_audio-timed_start,"benchmark_duration_s":summary["duration_s"],
                  "boundary_note":"Runner-ready has 1s precision; warmup log is observed by 0.1s polling. Independent health observer was later, not the first successful health. HTTP times use linear wall/perf clock-anchor interpolation; last arrival uses rounded e2e_s and excludes trailing client housekeeping."},
        "operating_thresholds":thresholds,"full_run":summarize(rows),"timed":summarize(timed),
        "timed_last_5_minutes":summarize(last5),
        "timed_last_3_minutes":summarize(last3),
        "first_SW_thermal_sample":None if first is None else dict(first,relative_service_start_s=first["epoch_s"]-service_start,relative_timed_start_s=first["epoch_s"]-timed_start),
        "first_SW_thermal_onset_bracket":None if first is None else {
            "previous_sample":previous,"first_active_sample":first,
            "relative_service_start_s":[previous["epoch_s"]-service_start if previous else None,first["epoch_s"]-service_start],
            "relative_timed_start_s":[previous["epoch_s"]-timed_start if previous else None,first["epoch_s"]-timed_start],
            "note":"Flag onset lies after the previous inactive observation and no later than the first active observation; sensor/internal sampling delays remain unknown."
        },
        "SW_thermal_episodes":thermal,
        "threshold_first_samples":first_threshold,
        "coincidence_during_timed":{"thermal_active_samples":len(coincident),
            "thermal_with_core_below_85_HBM_at_least_95":sum(r["core_c"]<85 and r["hbm_c"]>=95 for r in coincident),
            "thermal_with_core_at_least_85":sum(r["core_c"]>=85 for r in coincident),
            "thermal_with_HBM_at_least_95":sum(r["hbm_c"]>=95 for r in coincident)},
        "fan_observation":{"nvidia_GPU_fan_speed":fan["gpu_fan_speed"],
                           "chassis_fan_RPM_or_mode_verified":False,
                           "interpretation":fan["interpretation"],
                           "fan_or_BMC_settings_changed":fan["fan_or_bmc_settings_changed"]},
        "performance":{"GPU2_cold_C1":performance(summary),"prior_GPU1_B1_C1":performance(old_summary)},
        "prior_GPU1_hardware":{"timed":summarize(old_timed),"timed_last_5_minutes":summarize(old_last5),
                              "first_telemetry_sample":old_rows[0],"first_SW_thermal_sample":old_first,
                              "first_SW_thermal_relative_timed_start_s":old_first["epoch_s"]-old_start if old_first else None,
                              "HBM_temperature_available":False,"timed_alignment_uncertainty_s":1},
        "limits":[
            "Changed physical GPU and starting/thermal conditions; this is not a controlled causal comparison against GPU1.",
            "Core and HBM temperatures are sampled once per second and rounded by the sensor. First active sample brackets an event; it is not the exact onset.",
            "SW thermal slowdown does not identify which sensor triggered it. Threshold coincidence is evidence of association, not proof of the exclusive cause.",
            "Do not infer core-temperature triggering from a reading of 84C. Old GPU1 telemetry contains no HBM temperature.",
            "Sample proportions are not exact time-weighted throttling shares. Episode spans are between observed active samples, with censoring recorded.",
            "No driver/CUDA/power/clock/fan settings were changed by this analysis."
        ],
    }
    out.mkdir(exist_ok=True)
    (out/"COLD_GPU2_FINDINGS.json").write_text(json.dumps(report,indent=2)+"\n")
    new_hw=report["timed"];old_hw=report["prior_GPU1_hardware"]["timed"]
    new_tail=report["timed_last_5_minutes"];old_tail=report["prior_GPU1_hardware"]["timed_last_5_minutes"]
    def value(x):
        return "not recorded" if x is None else f"{x:.2f}"
    comparison=[
        ("Timed duration (s)",summary["duration_s"],old_summary["duration_s"]),
        ("Audio throughput (audio s/s)",summary["throughput_audio_s_per_s"],old_summary["throughput_audio_s_per_s"]),
        ("Timed SM median (MHz)",new_hw["sm_mhz"]["p50"],old_hw["sm_mhz"]["p50"]),
        ("Timed SW thermal samples (%)",new_hw["sw_thermal_slowdown"]["sample_pct"],old_hw["sw_thermal_slowdown"]["sample_pct"]),
        ("Timed core maximum (°C)",new_hw["core_c"]["max"],old_hw["core_c"]["max"]),
        ("Timed HBM maximum (°C)",new_hw["hbm_c"]["max"],None),
        ("Last 5 min SM median (MHz)",new_tail["sm_mhz"]["p50"],old_tail["sm_mhz"]["p50"]),
        ("Last 5 min core median (°C)",new_tail["core_c"]["p50"],old_tail["core_c"]["p50"]),
        ("Last 5 min HBM median (°C)",new_tail["hbm_c"]["p50"],None),
        ("Last 5 min SW thermal samples (%)",new_tail["sw_thermal_slowdown"]["sample_pct"],old_tail["sw_thermal_slowdown"]["sample_pct"]),
    ]
    note=(f'First observed SW thermal flag: {iso(first["epoch_s"])}; '
          f'{first["epoch_s"]-service_start:.3f}s after service launch, '
          f'{first["epoch_s"]-timed_start:.3f}s after reconstructed first timed HTTP send. '
          f'Core {first["core_c"]:g}°C, HBM {first["hbm_c"]:g}°C, SM {first["sm_mhz"]:g}MHz.'
          if first else "No SW thermal flag was observed in the recorded samples.")
    md=["# GPU 2 cold-start C1 observation","",
        "1088/1088 requests succeeded; no saved audio, no explicit seed, 8 warmups. "
        "Private MPS maps to physical GPU 2. Live server and client CPU affinity matched B1.","",
        note,"",
        "Operating-limit query: GPU max operating 85°C, GPU slowdown 89°C, HBM max operating 95°C. "
        "One-second observations cannot identify the exact trigger instant or the responsible sensor.","",
        "| Metric | GPU 2 cold-start C1 | Prior GPU 1 B1 C1 |","| --- | ---: | ---: |"]
    md += [f"| {label} | {value(new)} | {value(old)} |" for label,new,old in comparison]
    md += ["","The physical GPU, starting temperatures and thermal history differ. "
           "This is not a controlled causal performance comparison. GPU 1 did not record HBM temperature. "
           "Telemetry rates are sample fractions, and old GPU 1 timed-window alignment has <1s uncertainty.","",
           "NVIDIA reports GPU Fan Speed as N/A; this does not show that chassis fans have stopped. "
           "Actual chassis fan RPM/mode was unavailable to this account. No fan or BMC setting was changed.","",
           "[Timeline](timeline.png) · [One-second CSV](timeline.csv) · [Detailed evidence](COLD_GPU2_FINDINGS.json)"]
    (out/"COLD_GPU2_FINDINGS.md").write_text("\n".join(md)+"\n")
    with (out/"timeline.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"svg.fonttype":"none",
                         "axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    fig.subplots_adjust(left=.14,right=.975,top=.84,bottom=.15,hspace=.2)
    fig.text(.14,.96,"GPU 2 cold-start C1: temperature, clocks and throttle reasons",fontsize=16,weight="bold")
    fig.text(.14,.92,"A100 80GB PCIe · 1088 codes requests · 8 warmups · cu129 / R550 · observed host conditions",fontsize=10.5)
    colors={"startup":"#D9DDE3","ready / warmup":"#F6E8BB","timed":"#DDECF6","shutdown / finalization":"#E8E8E8"}
    for ax in axes:
        for name,start,end in phases:
            ax.axvspan((start-service_start)/60,(end-service_start)/60,color=colors[name],alpha=.55,zorder=0)
        ax.grid(axis="y",color="#D5DADF",linewidth=.7)
    x=[r["minutes_since_service_start"] for r in rows]
    axes[0].plot(x,[r["core_c"] for r in rows],label="GPU core",color="#215E96",lw=1.5)
    axes[0].plot(x,[r["hbm_c"] for r in rows],label="HBM",color="#B85D16",lw=1.5)
    axes[0].axhline(85,color="#215E96",ls=":",lw=1,label="Core max operating 85°C")
    axes[0].axhline(95,color="#B85D16",ls=":",lw=1,label="HBM max operating 95°C")
    axes[0].set_ylabel("Temperature (°C)");axes[0].set_ylim(25,105)
    axes[0].legend(loc="lower right",frameon=False,ncol=2,fontsize=8.5)
    axes[1].plot(x,[r["sm_mhz"] for r in rows],color="#215E96",lw=1.5)
    axes[1].set_ylabel("SM clock (MHz)");axes[1].set_ylim(bottom=0)
    reason_keys=["sw_thermal_slowdown","hw_thermal_slowdown","sw_power_cap","hw_power_brake_slowdown"]
    reason_labels=["SW thermal","HW thermal","SW power cap","HW power brake"]
    for i,key in enumerate(reason_keys):
        active_x=[r["minutes_since_service_start"] for r in rows if r[key]]
        axes[2].scatter(active_x,[3-i]*len(active_x),marker="|",s=40,color="#B85D16" if i==0 else "#354052",linewidth=.8)
    axes[2].set_yticks([3,2,1,0],reason_labels);axes[2].set_ylim(-.5,3.5)
    axes[2].set_xlabel("Minutes since service launch")
    if first:
        first_x=first["minutes_since_service_start"]
        for ax in axes:
            ax.axvline(first_x,color="#7A3D54",lw=1,ls="--")
        axes[0].annotate(f'First observed SW thermal: +{first["seconds_since_timed_start"]:.1f}s from timed start',
                         xy=(first_x,first["hbm_c"]),xytext=(.3,.97),textcoords="axes fraction",
                         va="top",fontsize=9,arrowprops={"arrowstyle":"->","color":"#7A3D54"},color="#7A3D54")
    fig.legend(handles=[Patch(facecolor=color,label=name) for name,color in colors.items()],loc="upper left",
               bbox_to_anchor=(.135,.895),ncol=4,frameon=False,fontsize=9)
    fig.text(.14,.045,
             "One-second samples; stage boundaries combine runner logs and reconstructed client timestamps. Shutdown includes client drain.\n"
             "Temperature/flag coincidence does not identify the triggering sensor. GPU 1 had no HBM telemetry.\n"
             "Different physical cards and starting thermal states: this is not a controlled causal performance comparison.",
             fontsize=9,linespacing=1.35)
    for ext in ("png","svg"):
        fig.savefig(out/f"timeline.{ext}",dpi=180)
    plt.close(fig)
    print(json.dumps({"checks":report["checks"],"first_SW_thermal":report["first_SW_thermal_sample"],
                      "timed":report["timed"],"last5":report["timed_last_5_minutes"],"out":str(out)}))


if __name__=="__main__":
    main()
