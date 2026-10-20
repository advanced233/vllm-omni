"""Align recorded GPU samples to actual profiler captures, CPU-only."""
import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import re
from audit_trace import stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-dir", required=True, type=Path)
    ap.add_argument("--gpu-csv", required=True, type=Path)
    args = ap.parse_args()
    traces = [json.loads(p.read_text()) for p in sorted(args.audit_dir.glob("trace_*_independent.json"))]
    zone = dt.timezone(dt.timedelta(hours=8))
    rows = []
    with args.gpu_csv.open() as f:
        for raw in csv.DictReader(f, skipinitialspace=True):
            row = {re.sub(r"\s*\[.*\]$", "", k.strip()): v.strip() for k, v in raw.items()}
            row["_epoch_us"] = dt.datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=zone).timestamp()*1e6
            rows.append(row)
    windows = [(str(i), r["capture_epoch_us"]) for i, r in enumerate(traces)]
    if len(traces) > 1 and len({r["host_name"] for r in traces}) == 1:
        windows.append(("common", [max(r["capture_epoch_us"][0] for r in traces), min(r["capture_epoch_us"][1] for r in traces)]))
    result = {}
    for name, (start,end) in windows:
        selected = [r for r in rows if start <= r["_epoch_us"] <= end]
        def numeric(key):
            values = []
            for r in selected:
                try:
                    values.append(float(r[key].split()[0]))
                except (ValueError, KeyError):
                    pass
            return stats(values)
        reasons = {}
        for reason in ("sw_thermal_slowdown", "hw_thermal_slowdown", "sw_power_cap", "hw_power_brake_slowdown"):
            values = [r.get("clocks_event_reasons."+reason, r.get("clocks_throttle_reasons."+reason)) for r in selected]
            valid = [v for v in values if v in ("Active","Not Active")]
            reasons[reason] = {"valid_samples": len(valid), "active_samples": valid.count("Active"),
                               "active_sample_pct": 100*valid.count("Active")/len(valid) if valid else None}
        result[name] = {
            "capture_epoch_us": [start,end], "duration_s": max(0,end-start)/1e6,
            "sample_count": len(selected), "sm_clock_mhz": numeric("clocks.current.sm"),
            "temperature_c": numeric("temperature.gpu"), "power_draw_w": numeric("power.draw"),
            "utilization_pct": numeric("utilization.gpu"), "reason_flags": reasons,
            "sample_rows": [{k:v for k,v in r.items() if not k.startswith("_")} for r in selected],
        }
    out = args.audit_dir / "capture_hardware.json"
    out.write_text(json.dumps({
        "source": str(args.gpu_csv), "timezone": "Asia/Shanghai UTC+08:00",
        "interpretation": "Actual profiler-capture intervals only; excludes later native export and client drain. GPU samples are 1s snapshots, not continuous proof. A 20s workload warmup is not thermal steady state. Do not substitute whole-run hardware medians or compare absolute kernel latency to thermally throttled B1.",
        "trace_paths": [r["trace"] for r in traces], "windows": result,
    }, indent=2)+"\n")
    print(json.dumps({"out": str(out), "windows": {k:{x:v[x] for x in ("sample_count","sm_clock_mhz","temperature_c","reason_flags")} for k,v in result.items()}}))


if __name__ == "__main__":
    main()
