"""Summarize recorded nvidia-smi telemetry using each timed client interval.

CPU-only artifact analysis. No live queries or machine configuration changes.
"""
import argparse
import collections
import csv
import datetime as dt
import json
import math
import re
from pathlib import Path


EVIDENCE = Path(__file__).with_name("thermal_reason_evidence.json")
FIELDS = {
    "sm_clock_mhz": ("clocks.current.sm", "clocks.sm"),
    "memory_clock_mhz": ("clocks.current.memory", "clocks.mem"),
    "temperature_c": ("temperature.gpu",),
    "power_draw_w": ("power.draw",),
    "power_limit_w": ("power.limit",),
    "gpu_utilization_pct": ("utilization.gpu",),
    "memory_used_mib": ("memory.used",),
}


def quantile(values, q):
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] * (hi - k) + values[hi] * (k - lo) if lo != hi else values[lo]


def stats(values):
    if not values:
        return {"n": 0, "min": None, "p50": None, "p95": None, "max": None}
    return {
        "n": len(values),
        "min": min(values),
        "p50": quantile(values, .5),
        "p95": quantile(values, .95),
        "max": max(values),
    }


def load_csv(path, zone):
    rows, errors = [], []
    if not path.exists():
        return rows, ["missing_file"]
    with path.open() as f:
        for lineno, raw in enumerate(csv.DictReader(f, skipinitialspace=True), 2):
            raw = {re.sub(r"\s*\[.*?\]\s*$", "", k.strip()): (v or "").strip() for k, v in raw.items() if k}
            try:
                stamp = dt.datetime.strptime(raw["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=zone)
            except (ValueError, KeyError) as exc:
                errors.append(f"line {lineno}: {type(exc).__name__}: {exc}")
                continue
            row = {"timestamp": stamp.isoformat(), "_time": stamp.timestamp()}
            for name, aliases in FIELDS.items():
                value = next((raw[k] for k in aliases if k in raw), None)
                try:
                    row[name] = float(value.split()[0]) if value is not None else None
                except ValueError:
                    row[name] = None
            active = next((raw[k] for k in ("clocks_event_reasons.active", "clocks_throttle_reasons.active") if k in raw), None)
            try:
                row["active_mask"] = int(active, 16)
            except (TypeError, ValueError):
                row["active_mask"] = None
            row["explicit_reasons"] = {
                k.split(".", 1)[1]: v
                for k, v in raw.items()
                if k.startswith(("clocks_event_reasons.", "clocks_throttle_reasons.")) and not k.endswith(".active")
            }
            rows.append(row)
    return rows, errors


def summarize(rows, reasons):
    masks = [r["active_mask"] for r in rows if r["active_mask"] is not None]
    distribution = collections.Counter(masks)
    bit_counts = {name: sum(bool(mask & bit) for mask in masks) for name, bit in reasons.items()}
    explicit = collections.defaultdict(collections.Counter)
    for row in rows:
        for key, value in row["explicit_reasons"].items():
            explicit[key][value] += 1
    gaps = [b["_time"] - a["_time"] for a, b in zip(rows, rows[1:])]
    return {
        "sample_count": len(rows),
        "first_timestamp": rows[0]["timestamp"] if rows else None,
        "last_timestamp": rows[-1]["timestamp"] if rows else None,
        "sample_gap_seconds": stats(gaps),
        "hardware": {name: stats([r[name] for r in rows if r[name] is not None]) for name in FIELDS},
        "active_mask_valid_samples": len(masks),
        "exact_active_mask_distribution": {hex(mask): {"n": n, "sample_pct": 100*n/len(masks)} for mask, n in sorted(distribution.items())},
        "active_reason_bits": {name: {"mask": hex(reasons[name]), "n": n, "sample_pct": 100*n/len(masks) if masks else None} for name, n in bit_counts.items()},
        "explicit_reason_fields": dict(explicit),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    evidence = json.loads(EVIDENCE.read_text())
    reasons = {}
    for line in evidence["definition_lines"]:
        match = re.match(r"nvmlClocksEventReason(\w+)\s*=\s*(0x[0-9a-fA-F]+)", line["text"])
        if match and int(match[2], 16):
            reasons[match[1]] = int(match[2], 16)
    assert reasons["SwThermalSlowdown"] == 0x20
    runs = []
    for summary_file in sorted(args.root.glob("c*/r*/summary.json")):
        s = json.loads(summary_file.read_text())
        start = dt.datetime.strptime(s["started_at"], "%Y-%m-%dT%H:%M:%S%z")
        start_sec = start.timestamp()
        duration = s["duration_s"]
        rec = {
            "directory": str(summary_file.parent),
            "concurrency": s["concurrency"],
            "repeat": s.get("meta", {}).get("repeat"),
            "timed_start_second": start.isoformat(),
            "duration_s": duration,
            "timed_end_nominal": dt.datetime.fromtimestamp(start_sec + duration, start.tzinfo).isoformat(),
            "nominal_interval_definition": "started_at <= timestamp <= started_at + duration_s",
            "strict_interior_definition": "started_at + 1 second <= timestamp <= started_at + duration_s",
        }
        for filename, name in (("gpu.csv", "gpu_500ms"), ("thermal_reasons.csv", "thermal_explicit_10s")):
            rows, errors = load_csv(summary_file.parent / filename, start.tzinfo)
            nominal = [r for r in rows if start_sec <= r["_time"] <= start_sec + duration]
            interior = [r for r in rows if start_sec + 1 <= r["_time"] <= start_sec + duration]
            envelope = [r for r in rows if start_sec <= r["_time"] <= start_sec + duration + 1]
            rec[name] = {
                "parse_errors": errors,
                "file_samples_total": len(rows),
                "nominal_timed_window": summarize(nominal, reasons),
                "strict_interior_window": summarize(interior, reasons),
                "possible_timed_envelope_sample_count": len(envelope),
            }
        runs.append(rec)
    result = {
        "completed_runs_seen": len(runs),
        "reason_mapping_evidence": evidence,
        "interpretation": [
            "Only runs with summary.json are included; rerunning updates the report as more runs complete.",
            "Client started_at is truncated to whole seconds. Actual timed start is in [started_at, started_at+1s); duration_s has 1ms rounding.",
            "Nominal selection excludes startup except possible <1s leading-boundary contamination, and can miss <1s of the timed tail.",
            "Strict interior removes the uncertain first second; compare its statistics to nominal selection for boundary sensitivity.",
            "nvidia-smi timestamps have no zone; interpret them in the offset recorded by summary.started_at on this same host.",
            "Reason percentages are percentages of valid sampled records, not exact time-weighted duration.",
            "SwThermalSlowdown uses bitwise mask & 0x20; the exact-mask distribution separately preserves combinations.",
            "The 10s explicit-reason file started partway through C1, so C1 coverage is partial. The original 500ms GPU log covers the run.",
            "Thermal throttling documents actual machine state and does not automatically invalidate a run. Future comparisons need matched hardware/thermal conditions.",
            "No settings were changed; CPU-only analysis of existing files."
        ],
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"completed_runs_seen": len(runs), "out": str(args.out)}))


if __name__ == "__main__":
    main()
