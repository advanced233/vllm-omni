#!/usr/bin/env python3
"""Stuck takes, speaking rate and pauses of saved Drama 3 replies, set against the codec windows.

Stuck takes: Drama 3 sometimes stops speaking and emits steady low-level noise
for seconds, often up to the ``max_tokens`` cap (500 frames, 23.22 s). That
noise can be as loud as quiet speech (-38 to -50 dBFS), so no energy threshold
separates it; what does is that it is flat. Every 1 s window (0.25 s hop) whose
10 ms frame levels span less than ``--flat-db`` (p90 - p10) is noise, and a
reply whose noise runs ``--noise-run`` seconds or more is flagged
``noise-run``; one that reached the frame cap is flagged ``max-frames``.
Speech moves by 10-30 dB every syllable: on the first 84 A100 validation
replies the stuck ones ran 16.5-22.8 s of noise, the rest at most 1.25 s.

Slow speech cannot come from the codec: every frame decodes to 2048 samples
(~46.4 ms), so the rate is set by how many frames stage 0 samples per word. A
pause can come from either side, but one made by splicing streamed codec
windows sits on a window boundary. For every reply that ``stream_bench.py
--save-audio`` kept, this reports words per second of speech (and the
reference's), the pauses inside the speech, and how far each pause edge is from
the nearest window boundary: the observed one when ``requests.jsonl`` has a
``chunk_log`` (a new burst of bytes), else the one the deploy's ramp implies.

A single pause edge within the tolerance can be chance (boundaries are 0.3-3 s
apart); many flagged pauses on boundaries point at the codec stage. Words per
second is the target text's word count over the detected speech span, not a
transcript's, and ``slow`` compares it with the median over every reply given.

Usage:
  python analyze_audio.py results/check_codes results/check_audio --codes ref_codes_en.jsonl
  python analyze_audio.py results/row18_seed* --seed-tts-root /data/seed-tts-eval --all
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import wave
from pathlib import Path
from typing import Any

import numpy as np

FRAME_S = 0.01


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        channels, sr = w.getnchannels(), w.getframerate()
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    return x, sr


def speech_and_pauses(
    x: np.ndarray, sr: int, min_pause_s: float, rel_db: float, floor_db: float
) -> tuple[float, float, list[tuple[float, float]]]:
    """Start and end of the speech, and the silent stretches inside it, in seconds.

    A 10 ms frame is quiet when its RMS is ``rel_db`` below the clip's loud
    frames (95th percentile), and never louder than ``floor_db``. Drama 3's
    pauses are low-level noise (around -45 dBFS under speech at -20 to -25),
    not digital silence, so the default margin is 17 dB.
    """
    hop = max(1, int(sr * FRAME_S))
    n = len(x) // hop
    if n == 0:
        return 0.0, 0.0, []
    rms = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1))
    db = 20 * np.log10(rms + 1e-9)
    voiced = db > max(float(np.percentile(db, 95)) - rel_db, floor_db)
    idx = np.flatnonzero(voiced)
    if idx.size == 0:
        return 0.0, 0.0, []
    first, last = int(idx[0]), int(idx[-1])
    pauses, run_start = [], None
    for i in range(first, last + 1):
        if not voiced[i] and run_start is None:
            run_start = i
        elif voiced[i] and run_start is not None:
            if (i - run_start) * FRAME_S >= min_pause_s:
                pauses.append((run_start * FRAME_S, i * FRAME_S))
            run_start = None
    return first * FRAME_S, (last + 1) * FRAME_S, pauses


def noise_run(x: np.ndarray, sr: int, flat_db: float, win_s: float = 1.0, hop_s: float = 0.25) -> tuple[float, float]:
    """Length and start (s) of the longest stretch covered by flat 1 s windows: noise, not speech."""
    hop = max(1, int(sr * FRAME_S))
    n = len(x) // hop
    if n == 0:
        return 0.0, 0.0
    db = 20 * np.log10(np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-9)
    win, step = int(win_s / FRAME_S), int(hop_s / FRAME_S)
    flat = np.zeros(n, dtype=bool)
    for s in range(0, max(1, n - win + 1), step):
        seg = db[s : s + win]
        if np.percentile(seg, 90) - np.percentile(seg, 10) < flat_db:
            flat[s : s + win] = True
    best = best_start = run = 0
    for i, f in enumerate(flat):
        run = run + 1 if f else 0
        if run > best:
            best, best_start = run, i - run + 1
    return best * FRAME_S, best_start * FRAME_S


def ramp_boundaries(ramp: list[int], window: int, hop: int, sr: int, total_s: float) -> list[float]:
    bounds, frames, i = [], 0, 0
    while True:
        frames += ramp[i] if i < len(ramp) else window
        i += 1
        t = frames * hop / sr
        if t >= total_s:
            return bounds
        bounds.append(t)


def observed_boundaries(chunk_log: list[list[float]], bytes_per_s: float, gap_ms: float) -> list[float]:
    """Audio time where each burst of bytes after the first starts."""
    return [prev[1] / bytes_per_s for prev, cur in zip(chunk_log, chunk_log[1:]) if cur[0] - prev[0] > gap_ms]


def words(text: str) -> int:
    return len(re.findall(r"[\w']+", text))


def nearest_ms(t: float, bounds: list[float]) -> float | None:
    return round(min(abs(t - b) for b in bounds) * 1000, 1) if bounds else None


def analyze_dir(run: Path, args: argparse.Namespace, ref_info: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    audio = run / "audio"
    manifest = {}
    for line in (audio / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            manifest[entry["wav"]] = entry
    out = []
    for line in (run / "requests.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if not rec.get("wav") or (args.rows and rec["index"] not in args.rows):
            continue
        entry = manifest[rec["wav"]]
        x, sr = read_wav(audio / rec["wav"])
        start, end, pauses = speech_and_pauses(x, sr, args.min_pause, args.rel_db, args.floor_db)
        total = len(x) / sr
        if rec.get("chunk_log"):
            bounds = observed_boundaries(rec["chunk_log"], 2 * rec.get("channels", 1) * sr, args.burst_gap_ms)
            source = "observed"
        else:
            bounds = ramp_boundaries(args.ramp, args.window, args.hop, sr, total)
            source = "ramp"
        speech_s = end - start
        n_words = words(entry["target_text"])
        noise_s, noise_at = noise_run(x, sr, args.flat_db)
        row = {
            "run": run.name,
            "index": rec["index"],
            "utt_id": rec["utt_id"],
            "seed": rec.get("seed"),
            "audio_s": round(total, 3),
            "frames": round(len(x) / args.hop),
            "noise_run_s": round(noise_s, 2),
            "noise_run_at_s": round(noise_at, 2),
            "speech_s": round(speech_s, 3),
            "lead_s": round(start, 3),
            "tail_s": round(total - end, 3),
            "words": n_words,
            "wps": round(n_words / speech_s, 2) if speech_s > 0 else None,
            "pauses": [
                {
                    "start_s": round(a, 2),
                    "len_s": round(b - a, 2),
                    "start_to_boundary_ms": nearest_ms(a, bounds),
                    "end_to_boundary_ms": nearest_ms(b, bounds),
                }
                for a, b in pauses
            ],
            "boundaries": source,
        }
        ref = ref_info.get(entry["prompt_wav"])
        if ref:
            row["ref_wps"] = round(words(entry["ref_text"]) / ref["speech_s"], 2) if ref["speech_s"] else None
            if "longest_pause_s" in ref:
                row["ref_longest_pause_s"] = ref["longest_pause_s"]
        out.append(row)
    return out


def flag(row: dict[str, Any], median_wps: float | None, args: argparse.Namespace) -> list[str]:
    flags = []
    if row["noise_run_s"] >= args.noise_run:
        flags.append("noise-run")
    if row["frames"] >= args.max_frames:
        flags.append("max-frames")
    if row["wps"] is not None and median_wps and row["wps"] < args.slow * median_wps:
        flags.append("slow")
    long = [p for p in row["pauses"] if p["len_s"] >= args.long_pause]
    if long:
        flags.append("long-pause")
    tol = args.boundary_tol_ms
    if any(
        (p["start_to_boundary_ms"] is not None and p["start_to_boundary_ms"] <= tol)
        or (p["end_to_boundary_ms"] is not None and p["end_to_boundary_ms"] <= tol)
        for p in long
    ):
        flags.append("on-boundary")
    if row["tail_s"] >= args.long_tail:
        flags.append("long-tail")
    return flags


def reference_info(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """Speech span (and longest pause) of each run's references, keyed by their meta.lst path."""
    prompt_wavs = set()
    for run in args.runs:
        for line in (Path(run) / "audio" / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                prompt_wavs.add(json.loads(line)["prompt_wav"])
    info: dict[str, dict[str, Any]] = {}
    if args.codes:
        for line in Path(args.codes).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r["prompt_wav"] in prompt_wavs:
                    info[r["prompt_wav"]] = {"speech_s": r["ref_audio_s"]}
    if args.seed_tts_root:
        locale_dir = Path(args.seed_tts_root) / args.locale
        for prompt_wav in prompt_wavs:
            path = locale_dir / prompt_wav
            if path.is_file():
                x, sr = read_wav(path)
                start, end, pauses = speech_and_pauses(x, sr, args.min_pause, args.rel_db, args.floor_db)
                info[prompt_wav] = {
                    "speech_s": end - start,
                    "longest_pause_s": round(max((b - a for a, b in pauses), default=0.0), 2),
                }
    return info


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("runs", nargs="+", help="stream_bench.py --out directories run with --save-audio")
    p.add_argument("--codes", default=None, help="encode_ref_codes.py output, for the references' rate")
    p.add_argument("--seed-tts-root", default=None, help="Measure the reference wavs themselves")
    p.add_argument("--locale", default="en")
    p.add_argument("--rows", type=lambda v: {int(x) for x in v.split(",")}, default=None)
    p.add_argument("--all", action="store_true", help="Print every reply, not only flagged ones")
    p.add_argument("--json", default=None, help="Write every row here")
    p.add_argument("--min-pause", type=float, default=0.25, help="Shortest silence counted as a pause (s)")
    p.add_argument("--long-pause", type=float, default=0.7, help="Pause length flagged (s)")
    p.add_argument("--long-tail", type=float, default=1.5, help="Trailing silence flagged (s)")
    p.add_argument("--slow", type=float, default=0.65, help="Flag below this fraction of the median rate")
    p.add_argument("--flat-db", type=float, default=10.0, help="A 1 s window spanning less than this (dB) is noise")
    p.add_argument("--noise-run", type=float, default=2.0, help="Noise stretch flagged (s)")
    p.add_argument("--max-frames", type=int, default=500, help="Frame cap (the deploy's max_tokens)")
    p.add_argument("--rel-db", type=float, default=17.0, help="Quiet: this many dB under the loud frames")
    p.add_argument("--floor-db", type=float, default=-60.0, help="Quiet whatever the margin below this level")
    p.add_argument("--boundary-tol-ms", type=float, default=25.0)
    p.add_argument("--burst-gap-ms", type=float, default=5.0, help="Gap between reads that starts a new window")
    p.add_argument("--ramp", type=lambda v: [int(x) for x in v.split(",")], default=[6, 8, 12, 16, 24, 32])
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--hop", type=int, default=2048, help="Samples per codec frame")
    args = p.parse_args()

    ref_info = reference_info(args)
    per_run = [(run, analyze_dir(Path(run), args, ref_info)) for run in args.runs]
    rows = [r for _, run_rows in per_run for r in run_rows]
    # Pooled: a run of one reply is its own median.
    rates = [r["wps"] for r in rows if r["wps"] is not None and r["noise_run_s"] < args.noise_run]
    median_wps = statistics.median(rates) if rates else None
    for r in rows:
        r["flags"] = flag(r, median_wps, args)

    def counts(group: list[dict[str, Any]]) -> str:
        n = len(group)
        stuck = sum("noise-run" in r["flags"] for r in group)
        capped = sum("max-frames" in r["flags"] for r in group)
        flagged = sum(bool(r["flags"]) for r in group)
        pct = f" ({100 * stuck / n:.1f}%)" if n else ""
        return f"{n} replies, {stuck} noise-run{pct}, {capped} at the {args.max_frames}-frame cap, {flagged} flagged"

    print(f"median {median_wps} words/s over replies without a noise run")
    for run, run_rows in per_run:
        print(f"{run}: {counts(run_rows)}")
    if len(per_run) > 1:
        print(f"all: {counts(rows)}")

    print(
        "| run | row | seed | audio s | frames | noise run (s @ t) | words/s | ref words/s "
        "| longest pause (s @ t) | to boundary (ms) | flags |"
    )
    print("| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | --- |")
    for r in sorted(rows, key=lambda r: (r["index"], r["run"])):
        if not (args.all or r["flags"]):
            continue
        longest = max(r["pauses"], key=lambda q: q["len_s"], default=None)
        pause = f"{longest['len_s']} @ {longest['start_s']}" if longest else "-"
        edges = f"{longest['start_to_boundary_ms']} / {longest['end_to_boundary_ms']}" if longest else "-"
        noise = f"{r['noise_run_s']} @ {r['noise_run_at_s']}" if r["noise_run_s"] else "-"
        print(
            f"| {r['run']} | {r['index']} | {r['seed']} | {r['audio_s']} | {r['frames']} | {noise} | {r['wps']} "
            f"| {r.get('ref_wps', '-')} | {pause} | {edges} ({r['boundaries']}) | {', '.join(r['flags']) or '-'} |"
        )
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
