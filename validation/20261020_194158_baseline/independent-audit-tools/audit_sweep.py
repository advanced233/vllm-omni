"""CPU-only independent audit of completed Drama 3 sweep artifacts.

This does not launch, stop, change or connect to any GPU/server process.
Exit 0 means artifact checks pass; it is not a claim of audio quality,
official cu130 compatibility, or absence of unlogged preemption.
"""

import argparse
import collections
import json
import math
import re
import statistics
from pathlib import Path


def numeric_summary(values):
    values = [v for v in values if isinstance(v, (float, int)) and math.isfinite(v)]
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "sample_stdev": statistics.stdev(values) if len(values) > 1 else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--levels", default="1 4 8 12 16 24 32 48 64 96 128")
    ap.add_argument("--repeats-at", default="64:3 128:5")
    ap.add_argument("--num-prompts", type=int, default=1088)
    ap.add_argument("--label", default="a100-main")
    args = ap.parse_args()
    repeats = dict(map(lambda s: tuple(map(int, s.split(":"))), args.repeats_at.split()))
    plan = [(c, r) for c in map(int, args.levels.split()) for r in range(1, repeats.get(c, 1) + 1)]
    runs = []
    for concurrency, repeat in plan:
        directory = args.root / f"c{concurrency}" / f"r{repeat}"
        rec = {"concurrency": concurrency, "repeat": repeat, "directory": str(directory), "errors": [], "review": []}
        errors = rec["errors"]
        try:
            summary = json.loads((directory / "summary.json").read_text())
            requests = [json.loads(line) for line in (directory / "requests.jsonl").read_text().splitlines() if line.strip()]
        except (OSError, ValueError) as exc:
            errors.append(f"cannot_read_required_artifacts: {type(exc).__name__}: {exc}")
            runs.append(rec)
            continue
        n = args.num_prompts
        if [summary.get(k) for k in ("requests", "succeeded", "failed")] != [n, n, 0]:
            errors.append("summary_request_counts")
        if summary.get("concurrency") != concurrency or summary.get("label") != args.label:
            errors.append("concurrency_or_label_mismatch")
        if len(requests) != n or sorted(r.get("index", -1) for r in requests) != list(range(n)):
            errors.append("request_count_or_index_coverage")
        if len({r.get("utt_id") for r in requests}) != n:
            errors.append("utterance_ids_not_unique")
        if any(r.get("ok") is not True or r.get("status") != 200 or r.get("error") for r in requests):
            errors.append("unsuccessful_request")
        if any(r.get("seed") is not None for r in requests):
            errors.append("unexpected_request_seed")
        if any(r.get("sample_rate") != 44100 or r.get("channels") != 1 for r in requests):
            errors.append("unexpected_audio_format")
        for r in requests:
            if r.get("bytes", 0) <= 0 or r.get("bytes", 0) % 2:
                errors.append("empty_or_partial_pcm_sample")
                break
            if abs(r["audio_s"] - r["bytes"] / 88200) > 0.00006:
                errors.append("pcm_duration_inconsistent")
                break
            if not r.get("chunk_log") or r["chunk_log"][-1][1] != r["bytes"]:
                errors.append("chunk_log_byte_total")
                break
        w = summary.get("workload", {})
        expected = {"ref_mode": "codes", "rows_timed": n, "warmups": 8, "seed": None, "shuffle_seed": None, "extra_body": {}}
        if any(w.get(k) != v for k, v in expected.items()):
            errors.append("workload_differs_from_baseline")
        if (directory / "audio").exists() or any("wav" in r for r in requests):
            errors.append("unexpected_saved_audio")
        lag = summary.get("client", {}).get("loop_lag_ms", {}).get("p99")
        if lag is None or lag >= 10:
            errors.append("loop_lag_p99_not_below_10ms")
        cleanup = directory / "cleanup.json"
        if not cleanup.exists():
            errors.append("missing_cleanup")
        else:
            cleanup_data = json.loads(cleanup.read_text())
            rec["cleanup"] = cleanup_data
            if cleanup_data.get("gpu_mem_after_mib", 99999) >= 128:
                errors.append("cleanup_gpu_memory_not_below_128MiB")
        for required in ("server.log", "gpu.csv", "client.log", "startup_seconds.txt"):
            if not (directory / required).is_file():
                errors.append(f"missing_{required}")
        log = (directory / "server.log").read_text(errors="replace") if (directory / "server.log").exists() else ""
        # Broad evidence collection only. Shutdown cancellation traces need phase-aware review.
        rec["log_evidence"] = [
            {"line": i, "text": line}
            for i, line in enumerate(log.splitlines(), 1)
            if re.search(r"preempt|outofmemory|out of memory|CUDA error|Traceback|ERROR|KV cache size|Maximum concurrency", line, re.I)
        ]
        if any(re.search(r"outofmemory|out of memory|CUDA error|resuming preempted", e["text"], re.I) for e in rec["log_evidence"]):
            errors.append("oom_cuda_or_preemption_requires_review")
        if any(re.search(r"Traceback|ERROR|preempt", e["text"], re.I) for e in rec["log_evidence"]):
            rec["review"].append("phase_aware_server_log_review")
        ok = [r for r in requests if r.get("ok")]
        long = [r for r in ok if r.get("audio_s", 0) >= 20]
        total_audio = sum(r.get("audio_s", 0) for r in ok)
        long_audio = sum(r["audio_s"] for r in long)
        at_cap = sum(r.get("bytes", 0) // (2 * r.get("channels", 1) * 2048) >= 500 for r in ok)
        derived_long = {
            "at_least_20s": len(long),
            "at_least_20s_pct": round(100 * len(long) / len(ok), 2) if ok else None,
            "at_frame_cap": at_cap,
            "audio_share_pct": round(100 * long_audio / total_audio, 2) if total_audio else None,
        }
        if summary.get("long_takes") != derived_long:
            errors.append("long_take_summary_mismatch")
        duration = summary.get("duration_s", 0)
        if abs(total_audio - summary.get("audio_s_total", 0)) > n * 0.00006 + 0.001:
            errors.append("audio_total_inconsistent")
        rec["metrics"] = {
            "duration_s": duration,
            "throughput_audio_s_per_s": summary.get("throughput_audio_s_per_s"),
            "full_window_audio_s_per_s": summary.get("full_concurrency_window", {}).get("throughput_audio_s_per_s"),
            "full_window_seconds": summary.get("full_concurrency_window", {}).get("seconds"),
            "ttfa_p50_ms": summary.get("ttfa_ms", {}).get("p50"),
            "ttfa_p95_ms": summary.get("ttfa_ms", {}).get("p95"),
            "rtf_mean": summary.get("rtf", {}).get("mean"),
            "stall_0ms_pct": summary.get("stall", {}).get("0ms", {}).get("streams_stalled_pct"),
            "stall_100ms_pct": summary.get("stall", {}).get("100ms", {}).get("streams_stalled_pct"),
            "loop_lag_p99_ms": lag,
            "audio_s_total": total_audio,
            **derived_long,
            "long_take_audio_seconds": long_audio,
            "short_take_audio_s_per_same_wall_s": (total_audio - long_audio) / duration if duration else None,
        }
        rec["long_take_indices"] = [r["index"] for r in long]
        runs.append(rec)
    grouped = collections.defaultdict(list)
    for run in runs:
        if not run["errors"] and "metrics" in run:
            grouped[run["concurrency"]].append(run)
    numerical = [
        {"concurrency": c, "accepted_artifact_runs": len(group), "metrics": {k: numeric_summary([r["metrics"][k] for r in group]) for k in group[0]["metrics"]}}
        for c, group in sorted(grouped.items())
    ]
    sweep_errors = []
    sweep_log_path = args.root / "sweep.log"
    client_exits = []
    if sweep_log_path.exists():
        client_exits = [
            {"concurrency": int(c), "repeat": int(r), "exit": int(status)}
            for c, r, status in re.findall(
                r"c=(\d+) r=(\d+): client exit (\d+)", sweep_log_path.read_text(errors="replace")
            )
        ]
        if {(r["concurrency"], r["repeat"]) for r in client_exits} != set(plan):
            sweep_errors.append("client_exit_plan_coverage")
        if len(client_exits) != len(plan) or any(r["exit"] != 0 for r in client_exits):
            sweep_errors.append("client_exit_nonzero_or_duplicates")
    else:
        sweep_errors.append("missing_sweep_log")
    for filename in ("runner.exit", "run.exit"):
        exit_file = args.root / filename
        if exit_file.exists() and exit_file.read_text().strip() != "0":
            sweep_errors.append(f"nonzero_{filename}")
    if (args.root / "BLOCK_NEXT").exists():
        sweep_errors.append("executor_BLOCK_NEXT_exists")
    report = {
        "planned_runs": len(plan),
        "artifact_checks_pass": not sweep_errors and all(not r["errors"] for r in runs),
        "sweep_errors": sweep_errors,
        "recorded_client_exits": client_exits,
        "phase_aware_log_review_pending": any(r["review"] for r in runs),
        "notes": [
            "Run-level metric means/min/max/stdev; not pooled request percentiles.",
            "GPU telemetry, source hashes, own-PID lifecycle and other-user interference require executor-manifest review.",
            "Full-concurrency window includes initial filling and excludes draining; 256 prompts at C128 is a short capacity check.",
            "Long/capped outputs are duration proxies, not noise-run, WER or finish_reason evidence.",
            "Short-take audio divided by unchanged wall time is a numerator sensitivity only, not a counterfactual speed estimate.",
            "No audio means no quality verification or WER/SIM can be recovered from these artifacts.",
        ],
        "runs": runs,
        "numeric_by_concurrency": numerical,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"planned": len(plan), "artifacts_pass": report["artifact_checks_pass"], "needs_log_review": report["phase_aware_log_review_pending"], "out": str(args.out)}))
    return 0 if report["artifact_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
