#!/usr/bin/env python3
"""Per-mixed-step host time of the GDN core from drama3_gdn_probe records, one row per run.

A run is a probe directory (``gdn_probe_<pid>.jsonl`` files). With
``--bench-summary`` (``stream_bench.py``'s summary.json, one per run, in the
same order) only the steps inside that run's timed window count, which leaves
out start-up (CUDA graph capture, warmup) and the warmup requests.

Every number is host time from perf_counter_ns: what stage 0's Python thread
spends issuing the GDN work of a step, not GPU time.

Usage:
  python summarize_probe.py probe_baseline probe_direct --bench-summary c16_base/summary.json c16_direct/summary.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
from pathlib import Path
from typing import Any

SECTIONS = ("setup", "conv", "decode", "prep", "gather", "chunk", "scatter", "copy")


def load(run: Path) -> list[dict[str, Any]]:
    recs = []
    for path in sorted(run.glob("gdn_probe_*.jsonl")):
        recs.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return sorted(recs, key=lambda r: r["t"])


def bench_window(summary: Path) -> tuple[float, float]:
    s = json.loads(summary.read_text(encoding="utf-8"))
    start = dt.datetime.strptime(s["started_at"], "%Y-%m-%dT%H:%M:%S%z").timestamp()
    wall = (s.get("thermal_gate") or {}).get("wall_s") or s["duration_s"]
    # started_at has one-second resolution: start one second later rather than catch warmup steps.
    return start + 1.0, start + float(wall)


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, round(q / 100 * (len(values) - 1)))]


def summarize(recs: list[dict[str, Any]]) -> dict[str, Any]:
    ms = [r["sum_ns"] / 1e6 for r in recs]
    span = [r["span_ns"] / 1e6 for r in recs]
    out: dict[str, Any] = {
        "steps": len(recs),
        "variants": sorted({r["variant"] for r in recs}),
        "laps": sorted({r["laps"] for r in recs}),
        "calls_per_step": sorted({r["calls"] for r in recs}),
        "gdn_ms_per_step": {"mean": statistics.fmean(ms) if ms else None, "p50": pct(ms, 50), "p90": pct(ms, 90)},
        "span_ms_per_step": {"mean": statistics.fmean(span) if span else None, "p50": pct(span, 50)},
        "prefill_tokens_p50": pct([r["prefill_tokens"] for r in recs], 50),
        "decode_tokens_p50": pct([r["decode_tokens"] for r in recs], 50),
        "prefills_p50": pct([r["num_prefills"] for r in recs], 50),
    }
    if recs and recs[0]["laps"]:
        out["section_ms_per_step"] = {
            name: statistics.fmean(r["sections_ns"].get(name, 0) / 1e6 for r in recs) for name in SECTIONS
        }
    return out


def fmt(x: float | None, digits: int = 2) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("runs", nargs="+", help="Probe directories (DRAMA3_GDN_PROBE_DIR of each run)")
    p.add_argument("--bench-summary", nargs="*", default=None, help="stream_bench summary.json per run")
    p.add_argument("--skip-first", type=int, default=0, help="Without --bench-summary: drop this many steps")
    p.add_argument("--json", default=None)
    args = p.parse_args()
    if args.bench_summary and len(args.bench_summary) != len(args.runs):
        p.error("give one --bench-summary per run, in the same order")

    rows = []
    for i, run in enumerate(args.runs):
        recs = load(Path(run))
        if args.bench_summary:
            lo, hi = bench_window(Path(args.bench_summary[i]))
            recs = [r for r in recs if lo <= r["t"] <= hi]
        else:
            recs = recs[args.skip_first :]
        rows.append({"run": run, **summarize(recs)})

    print(
        "| run | variant | steps | calls/step | GDN host ms/step p50 (p90) | span ms p50 "
        "| prefill tok p50 | decode tok p50 |"
    )
    print("| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |")
    for r in rows:
        g = r["gdn_ms_per_step"]
        print(
            f"| {Path(r['run']).name} | {','.join(r['variants']) or '-'} | {r['steps']} "
            f"| {r['calls_per_step']} | {fmt(g['p50'])} ({fmt(g['p90'])}) | {fmt(r['span_ms_per_step']['p50'])} "
            f"| {fmt(r['prefill_tokens_p50'], 0)} | {fmt(r['decode_tokens_p50'], 0)} |"
        )
    lapped = [r for r in rows if "section_ms_per_step" in r]
    if lapped:
        print("\nmean host ms per mixed step, by section")
        print("| run | " + " | ".join(SECTIONS) + " |")
        print("| --- | " + " | ".join("---:" for _ in SECTIONS) + " |")
        for r in lapped:
            sec = r["section_ms_per_step"]
            print(f"| {Path(r['run']).name} | " + " | ".join(fmt(sec[s]) for s in SECTIONS) + " |")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
