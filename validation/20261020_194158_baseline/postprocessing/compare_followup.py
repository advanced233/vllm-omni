"""Separate C10/C12/C16 follow-up comparison; never changes the 17-run B1 table."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root = args.root
    inputs = []

    def read(path):
        inputs.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        return json.loads(path.read_text())

    for name, count in (("b1", 17), ("followup", 3)):
        audit = read(root / f"{name}-independent-audit.json")
        if audit.get("artifact_checks_pass") is not True or audit["planned_runs"] != count:
            raise SystemExit(f"{name} requires completed passing artifact audit.")
    hardware = {}
    for name in ("b1", "followup"):
        hardware[name] = {
            Path(r["directory"]).resolve(): r["gpu_500ms"]["nominal_timed_window"]
            for r in read(root / f"{name}-hardware-summary.json")["runs"]
        }
    rows, metadata = [], []
    for group, levels in (("b1", [8, 12, 16]), ("followup", [10, 12, 16])):
        for concurrency in levels:
            directory = root / group / f"c{concurrency}" / "r1"
            s = read(directory / "summary.json")
            meta = read(directory / "run_metadata.json")
            h = hardware[group][directory.resolve()]
            if [s.get("requests"), s.get("succeeded"), s.get("failed")] != [1088, 1088, 0]:
                raise SystemExit(f"Unexpected request counts in {directory}")
            w = s["workload"]
            if any(w.get(k) != v for k, v in {"seed": None, "warmups": 8, "ref_mode": "codes", "extra_body": {}, "shuffle_seed": None}.items()):
                raise SystemExit(f"Workload mismatch in {directory}")
            metadata.append((str(directory), meta))
            rows.append({
                "group": group, "concurrency": concurrency, "repeat": 1,
                "started_at": s["started_at"],
                "throughput_audio_s_per_s": s["throughput_audio_s_per_s"],
                "full_window_audio_s_per_s": s["full_concurrency_window"]["throughput_audio_s_per_s"],
                "rtf_mean": s["rtf"]["mean"],
                "ttfa_p50_ms": s["ttfa_ms"]["p50"], "ttfa_p95_ms": s["ttfa_ms"]["p95"],
                "stall_0ms_pct": s["stall"]["0ms"]["streams_stalled_pct"],
                "stall_100ms_pct": s["stall"]["100ms"]["streams_stalled_pct"],
                "sm_median_mhz": h["hardware"]["sm_clock_mhz"]["p50"],
                "temperature_median_c": h["hardware"]["temperature_c"]["p50"],
                "power_median_w": h["hardware"]["power_draw_w"]["p50"],
                "sw_thermal_sample_pct": h["active_reason_bits"]["SwThermalSlowdown"]["sample_pct"],
                "long_take_audio_share_pct": s["long_takes"]["audio_share_pct"],
                "client_loop_lag_p99_ms": s["client"]["loop_lag_ms"]["p99"],
            })
    # Only specific experimental parameters are declared matched; dynamic
    # timestamps/temperature/clock values are intentionally not equated.
    keys = [
        "server_commit", "tool_commit", "tool_sha256", "model_revision",
        "deploy_sha256", "codes_sha256", "seed", "warmups", "save_audio",
        "client_affinity", "server_affinity", "packages", "server_argv",
    ]
    reference = metadata[0][1]
    parameter_check = {
        k: {"all_match": all(m.get(k) == reference.get(k) for _, m in metadata), "reference": reference.get(k)}
        for k in keys
    }
    for key in ("OMP_NUM_THREADS", "MAX_JOBS", "CUDA_VISIBLE_DEVICES"):
        values = [m["environment"].get(key) for _, m in metadata]
        parameter_check[f"environment.{key}"] = {"all_match": len(set(values)) == 1, "reference": values[0]}
    if not all(v["all_match"] for v in parameter_check.values()):
        raise SystemExit("Experimental metadata differs. Review before plotting; do not claim matched parameters.")
    deltas = []
    for c in (12, 16):
        before = next(r for r in rows if r["group"] == "b1" and r["concurrency"] == c)
        after = next(r for r in rows if r["group"] == "followup" and r["concurrency"] == c)
        deltas.append({
            "concurrency": c,
            "followup_minus_b1": {k: after[k] - before[k] for k in before if isinstance(before[k], float)},
            "interpretation": "Observed repeat difference; not a causal improvement. Percentage differences are percentage points.",
        })
    note = (
        "Separate follow-up: B1 C8/C12/C16 compared with C10/C12/C16 after the full sweep. "
        "Each plotted point is one complete 1088-prompt run; no confidence intervals. "
        "Software, workload and recorded CPU affinity match, but thermal state and clocks were not controlled. "
        "Differences cannot be assigned to an optimization or saving audio. "
        "C10 adds an observation between C8 and C12; it does not establish a precise universal threshold."
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.json").write_text(json.dumps({
        "note": note, "parameters": parameter_check, "rows": rows,
        "paired_observed_deltas": deltas, "sources": inputs,
    }, indent=2) + "\n")
    with (args.out / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    columns = ["group", "concurrency", "throughput_audio_s_per_s", "rtf_mean", "ttfa_p95_ms",
               "stall_0ms_pct", "stall_100ms_pct", "sm_median_mhz", "sw_thermal_sample_pct", "long_take_audio_share_pct"]
    labels = ["Run", "C", "Audio s/s", "RTF", "TTFA p95 ms", "Stall >0 %", "Stall >100ms %",
              "SM median MHz", "Thermal samples %", "Long audio %"]
    md = [note, "", "| " + " | ".join(labels) + " |", "| " + " | ".join(["---"] * len(labels)) + " |"]
    for r in rows:
        md.append("| " + " | ".join(f"{r[k]:.2f}" if isinstance(r[k], float) else str(r[k]) for k in columns) + " |")
    (args.out / "comparison.md").write_text("\n".join(md) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))
    fig.subplots_adjust(left=.08, right=.98, bottom=.2, top=.84, hspace=.42, wspace=.28)
    fig.text(.08, .945, "B1 and subsequent C10/C12/C16 checks", fontsize=17, weight="semibold")
    fig.text(.08, .9, "Drama 3 · A100 80GB PCIe · cu129 / R550 · actual host conditions", fontsize=11)
    for group, label, color, marker in [("b1", "B1", "#215E96", "o"), ("followup", "Follow-up", "#B85D16", "s")]:
        subset = [r for r in rows if r["group"] == group]
        x = [r["concurrency"] for r in subset]
        for ax, key in [(axes[0, 0], "throughput_audio_s_per_s"), (axes[0, 1], "stall_0ms_pct"),
                        (axes[1, 0], "sm_median_mhz"), (axes[1, 1], "sw_thermal_sample_pct")]:
            ax.plot(x, [r[key] for r in subset], marker=marker, color=color, linewidth=1.5, label=label)
        axes[0, 1].plot(x, [r["stall_100ms_pct"] for r in subset], marker=marker, color=color,
                        linestyle="--", linewidth=1.3, label=f"{label}: >100 ms")
    for ax, title, unit in [
        (axes[0, 0], "Whole-run throughput", "Audio seconds / wall second"),
        (axes[0, 1], "Streams with stalls", "Streams (%)"),
        (axes[1, 0], "SM clock during timed work", "Run median (MHz)"),
        (axes[1, 1], "Software thermal slowdown", "Timed GPU samples (%)"),
    ]:
        ax.set_title(title, fontsize=12)
        ax.set_xticks([8, 10, 12, 16])
        ax.set_xlim(7.5, 16.5)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Concurrency (requests)")
        ax.set_ylabel(unit)
        ax.grid(axis="y", color="#E4E7EA", linewidth=.7)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9)
    axes[0, 1].set_ylim(0, 105)
    axes[1, 1].set_ylim(0, 105)
    fig.text(.08, .045,
             "Each point: one 1088-prompt run, same workload and CPU affinity; no saved audio. Solid stall lines: >0 ms.\n"
             "Thermal conditions were not controlled; hardware panels document observed state, not a causal correction.\n"
             "Lines connect observations only. Follow-up results remain separate from the original 17-run B1 table.",
             fontsize=9, linespacing=1.4)
    for ext in ("png", "svg"):
        fig.savefig(args.out / f"comparison.{ext}", dpi=180)
    plt.close(fig)
    print(json.dumps({"rows": len(rows), "parameters_match": True, "out": str(args.out)}))


if __name__ == "__main__":
    main()
