"""Export final, audited B1 curves as PNG/SVG plus exact numeric inputs.

Does not generate provisional plots. Requires all 17 specified runs and
a passing independent artifact audit. No server or GPU access.
"""

import argparse
import collections
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


LEVELS = [1, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
PLAN = [(c, r) for c in LEVELS for r in range(1, {64: 3, 128: 5}.get(c, 1) + 1)]
BLUE, ORANGE, GREY = "#215E96", "#B85D16", "#354052"
CAPTION = (
    "1088 prompts/run; fresh server; codes mode; no saved audio. "
    "C64: 3 runs, C128: 5 runs; others: 1 run.\n"
    "Points: mean of run-level metrics. Bars: observed min–max across runs, not confidence intervals.\n"
    "Actual host conditions, including thermal throttling. Server da572cdd; benchmark tools 3b116ea."
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_audit(audit, b1):
    if audit.get("artifact_checks_pass") is not True or audit.get("planned_runs") != len(PLAN):
        raise SystemExit("Refusing final plots: all 17 runs must pass the independent artifact audit.")
    actual = {(r["concurrency"], r["repeat"]) for r in audit["runs"]}
    if actual != set(PLAN) or len(audit["runs"]) != len(PLAN):
        raise SystemExit("Audit run plan differs from the final 17-run plan.")
    for run in audit["runs"]:
        expected = b1 / f'c{run["concurrency"]}' / f'r{run["repeat"]}'
        if Path(run["directory"]).resolve() != expected.resolve():
            raise SystemExit("Audit refers to a different run directory.")


def aggregate(runs):
    grouped = collections.defaultdict(list)
    for run in runs:
        grouped[run["concurrency"]].append(run)
    result = []
    keys = list(runs[0]["metrics"])
    for concurrency in LEVELS:
        group = grouped[concurrency]
        metrics = {}
        for key in keys:
            values = [r["metrics"][key] for r in group]
            if any(v is None or not math.isfinite(v) for v in values):
                raise SystemExit(f"Missing/nonfinite metric {key} at C{concurrency}.")
            metrics[key] = {
                "n": len(values), "mean": statistics.fmean(values),
                "min": min(values), "max": max(values),
                "sample_stdev": statistics.stdev(values) if len(values) > 1 else None,
                "values_by_repeat": values,
            }
        result.append({"concurrency": concurrency, "runs": len(group), "metrics": metrics})
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--b1", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--hardware", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log-review", type=Path, help="JSON {reviewed: true, note: ...} if audit flags phase-aware log review")
    args = ap.parse_args()
    audit = json.loads(args.audit.read_text())
    validate_audit(audit, args.b1)
    if audit.get("phase_aware_log_review_pending"):
        review = json.loads(args.log_review.read_text()) if args.log_review else {}
        if review.get("reviewed") is not True or not review.get("note"):
            raise SystemExit("Artifact audit requests phase-aware log review; supply its recorded conclusion.")
    hardware = json.loads(args.hardware.read_text())
    hw = {Path(r["directory"]).resolve(): r for r in hardware["runs"]}
    if len(hw) != 17:
        raise SystemExit("Hardware report must cover exactly the same 17 completed runs.")
    sources = [
        {"path": str(p.resolve()), "sha256": sha(p)}
        for p in [args.audit, args.hardware, Path(__file__)] + ([args.log_review] if args.log_review else [])
    ]
    runs = []
    for c, rep in PLAN:
        directory = args.b1 / f"c{c}" / f"r{rep}"
        summary_path = directory / "summary.json"
        s = json.loads(summary_path.read_text())
        if [s.get("requests"), s.get("succeeded"), s.get("failed")] != [1088, 1088, 0]:
            raise SystemExit(f"Counts changed after audit: {directory}")
        if s.get("concurrency") != c or s.get("workload", {}).get("seed") is not None:
            raise SystemExit(f"Workload changed after audit: {directory}")
        h = hw[directory.resolve()]["gpu_500ms"]["nominal_timed_window"]
        if h["sample_count"] <= 0:
            raise SystemExit(f"Empty timed telemetry: {directory}")
        metrics = {
            "throughput_audio_s_per_s": s["throughput_audio_s_per_s"],
            "full_window_audio_s_per_s": s["full_concurrency_window"]["throughput_audio_s_per_s"],
            "rtf_mean": s["rtf"]["mean"],
            "ttfa_p50_ms": s["ttfa_ms"]["p50"],
            "ttfa_p95_ms": s["ttfa_ms"]["p95"],
            "stall_0ms_pct": s["stall"]["0ms"]["streams_stalled_pct"],
            "stall_100ms_pct": s["stall"]["100ms"]["streams_stalled_pct"],
            "sm_clock_run_median_mhz": h["hardware"]["sm_clock_mhz"]["p50"],
            "thermal_sample_pct": h["active_reason_bits"]["SwThermalSlowdown"]["sample_pct"],
            "long_take_audio_share_pct": s["long_takes"]["audio_share_pct"],
        }
        runs.append({"concurrency": c, "repeat": rep, "directory": str(directory.resolve()), "metrics": metrics})
        for filename in ("summary.json", "requests.jsonl", "gpu.csv"):
            p = directory / filename
            sources.append({"path": str(p.resolve()), "sha256": sha(p)})
    grouped = aggregate(runs)
    numeric = {
        "title": "Drama 3 B1 — A100 80GB PCIe, cu129 / driver R550",
        "scope": "Actual machine/environment; not official cu130/R580+ compliance.",
        "aggregation": "Arithmetic mean of run-level values. For TTFA this is the mean of each run's p50/p95, not pooled request percentiles.",
        "error_bars": "Minimum and maximum across observed repeats; no confidence intervals. Single runs have no estimated spread.",
        "axes": "Concurrency on a base-2 logarithmic quantitative axis; exact measured levels shown as ticks. Lines only connect observations, not a fitted model.",
        "hardware_alignment": "Nominal timed window from summary.started_at (whole-second truncation) plus duration_s. Boundary alignment may differ by <1s; thermal values are sample proportions.",
        "quality_limitation": "Audio >=20s is a duration proxy; no saved audio, so no noise-run or WER inference.",
        "sources": sources,
        "runs": runs,
        "by_concurrency": grouped,
    }
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.labelcolor": GREY, "text.color": GREY, "axes.edgecolor": "#A5ABB3",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "semibold", "axes.titlesize": 14,
        "svg.fonttype": "none", "savefig.facecolor": "white",
    })
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "plot_data.json").write_text(json.dumps(numeric, indent=2) + "\n")
    with (args.out / "plot_data_runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["concurrency", "repeat"] + list(runs[0]["metrics"]))
        writer.writeheader()
        for run in runs:
            writer.writerow({"concurrency": run["concurrency"], "repeat": run["repeat"], **run["metrics"]})

    def axis(ax, ylabel):
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(LEVELS))
        ax.xaxis.set_major_formatter(FixedFormatter([str(c) for c in LEVELS]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(.86, 147)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Concurrency (requests; log₂ scale)")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E4E7EA", linewidth=.7)
        ax.set_axisbelow(True)

    def series(ax, key, label, color, marker, linestyle="-"):
        values = [r["metrics"][key] for r in grouped]
        means = [v["mean"] for v in values]
        errors = [[v["mean"] - v["min"] for v in values], [v["max"] - v["mean"] for v in values]]
        ax.errorbar(LEVELS, means, yerr=errors, color=color, marker=marker,
                    linestyle=linestyle, linewidth=1.7, markersize=5, capsize=3,
                    elinewidth=1.2, label=label)

    def finish(fig, name, extra=None):
        caption = CAPTION + ("\n" + extra if extra else "")
        fig.text(.075, .035, caption, fontsize=8.5, va="bottom", linespacing=1.35)
        fig.savefig(args.out / f"{name}.png", dpi=180)
        fig.savefig(args.out / f"{name}.svg")
        plt.close(fig)

    def figure(title):
        fig, ax = plt.subplots(figsize=(10.8, 6.4))
        fig.subplots_adjust(left=.09, right=.975, bottom=.24, top=.84)
        fig.text(.09, .945, title, fontsize=16, weight="semibold")
        fig.text(.09, .905, "Drama 3 · A100 80GB PCIe · cu129 / R550", fontsize=11)
        return fig, ax

    fig, ax = figure("Streaming audio throughput")
    series(ax, "throughput_audio_s_per_s", "Whole timed run", BLUE, "o")
    series(ax, "full_window_audio_s_per_s", "Full-concurrency window", ORANGE, "s", "--")
    axis(ax, "Generated audio seconds / wall second")
    ax.legend(loc="best", frameon=False)
    finish(fig, "throughput", "Full window ends at last dispatch; it includes initial filling and excludes the drain.")

    fig, ax = figure("Real-time factor")
    series(ax, "rtf_mean", "Mean request RTF", BLUE, "o")
    ax.axhline(1, color=GREY, linewidth=1, linestyle=":", label="RTF = 1")
    axis(ax, "Request elapsed time / generated audio duration")
    ax.legend(loc="best", frameon=False)
    finish(fig, "rtf")

    fig, ax = figure("Time to first audio")
    series(ax, "ttfa_p50_ms", "Run-level p50", BLUE, "o")
    series(ax, "ttfa_p95_ms", "Run-level p95", ORANGE, "s", "--")
    axis(ax, "Client-observed TTFA (ms)")
    ax.legend(loc="best", frameon=False)
    finish(fig, "ttfa")

    fig, ax = figure("Streams with playback stalls")
    series(ax, "stall_0ms_pct", "Any wait > 0 ms", BLUE, "o")
    series(ax, "stall_100ms_pct", "Any wait > 100 ms", ORANGE, "s", "--")
    axis(ax, "Streams with at least one qualifying wait (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="best", frameon=False)
    finish(fig, "stalls")

    fig, axes = plt.subplots(3, 1, figsize=(10.8, 11.2))
    fig.subplots_adjust(left=.105, right=.975, bottom=.16, top=.89, hspace=.52)
    fig.text(.105, .96, "Observed hardware state and long outputs", fontsize=16, weight="semibold")
    fig.text(.105, .925, "Drama 3 · A100 80GB PCIe · cu129 / R550", fontsize=11)
    for ax, key, ylabel, color, marker in [
        (axes[0], "sm_clock_run_median_mhz", "SM clock: run median (MHz)", BLUE, "o"),
        (axes[1], "thermal_sample_pct", "SW thermal slowdown samples (%)", ORANGE, "s"),
        (axes[2], "long_take_audio_share_pct", "Audio from takes ≥ 20 s (%)", BLUE, "o"),
    ]:
        series(ax, key, key, color, marker)
        axis(ax, ylabel)
    axes[1].set_ylim(0, 105)
    finish(fig, "hardware_and_long_outputs",
           "Thermal rates use 500 ms samples in the timed window (alignment uncertainty <1 s).\n"
           "Long-output share is a duration proxy, not a measurement of noise or transcription quality.")
    print(json.dumps({"completed_runs": len(runs), "figure_pairs_png_svg": 5, "out": str(args.out)}))


if __name__ == "__main__":
    main()
