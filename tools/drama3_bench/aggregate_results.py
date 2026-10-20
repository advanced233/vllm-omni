#!/usr/bin/env python3
"""Table of ``stream_bench.py`` runs: one row per (label, concurrency), repeats averaged.

PR #8933 reports the mean over repeats (3 at 64 streams, 5 at 128). Each cell
here is that mean, with the min-max over repeats when there is more than one.

Usage:
  python aggregate_results.py results/sweep_*/**/summary.json [--json table.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

COLUMNS = [
    ("Audio s/s", lambda s: s["throughput_audio_s_per_s"], 1),
    ("Audio s/s (full window)", lambda s: s["full_concurrency_window"]["throughput_audio_s_per_s"], 1),
    ("TTFA p50 (ms)", lambda s: s["ttfa_ms"]["p50"], 0),
    ("TTFA p95 (ms)", lambda s: s["ttfa_ms"]["p95"], 0),
    ("RTF (mean)", lambda s: s["rtf"]["mean"], 3),
    ("Stall % (>0 ms)", lambda s: s["stall"].get("0ms", {}).get("streams_stalled_pct"), 2),
    ("Stall % (>100 ms)", lambda s: s["stall"].get("100ms", {}).get("streams_stalled_pct"), 2),
    ("Failed", lambda s: s["failed"], 0),
]


def cell(values: list[float | None], digits: int) -> str:
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    mean = statistics.fmean(values)
    text = f"{mean:.{digits}f}"
    if len(values) > 1:
        text += f" ({min(values):.{digits}f}-{max(values):.{digits}f})"
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("summaries", nargs="+", help="summary.json files (or directories holding them)")
    parser.add_argument("--json", default=None, help="Also write the aggregated rows here")
    args = parser.parse_args()

    paths: dict[Path, None] = {}
    for item in args.summaries:
        p = Path(item)
        for found in sorted(p.rglob("summary.json")) if p.is_dir() else [p]:
            paths.setdefault(found.resolve())
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for p in paths:
        s = json.loads(p.read_text(encoding="utf-8"))
        groups[(s.get("label", ""), int(s["concurrency"]))].append(s)

    header = ["Label", "Concurrency", "Runs"] + [name for name, _, _ in COLUMNS]
    print("| " + " | ".join(header) + " |")
    print("| --- | ---: | ---: | " + " | ".join("---:" for _ in COLUMNS) + " |")
    rows = []
    for (label, concurrency), runs in sorted(groups.items()):
        cells = [cell([get(s) for s in runs], digits) for _, get, digits in COLUMNS]
        print(f"| {label or '-'} | {concurrency} | {len(runs)} | " + " | ".join(cells) + " |")
        rows.append({"label": label, "concurrency": concurrency, "runs": len(runs), **dict(zip(header[3:], cells))})
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
