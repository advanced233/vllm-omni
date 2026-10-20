#!/usr/bin/env python3
"""Closed-loop streaming benchmark for Drama 3 on ``/v1/audio/speech``, measured as in PR #8933.

Each Seed-TTS-Eval row clones its own reference, sent pre-encoded as
``ref_audio_codes`` (from ``encode_ref_codes.py``) with its transcript, and the
reply is streamed as raw PCM with the server's default sampling. ``--concurrency``
requests stay in flight until the rows run out; warmup requests run first and
are not measured. Restart the server for every concurrency level (``run_sweep.sh``).

Per request, with every timestamp taken when the bytes reach the client:

* TTFA: request sent to the first audio sample;
* RTF: request sent to the last audio chunk, over the audio duration;
* stall: playback starts at the first audio and a stream stalls when its next
  audio arrives after the audio already received has played out (the player
  then waits). A stall counts when the wait exceeds a threshold: 0 ms is
  PR #8933's definition, 100 ms the ``vllm bench serve`` default.

Throughput is seconds of audio over wall-clock seconds, from the first request
to the last response; it is also reported over the window where all
``--concurrency`` slots are busy (until the last row is dispatched), which
leaves out the drain at the end of the run.

Writes ``summary.json`` and ``requests.jsonl`` into ``--out``; ``--save-audio``
also keeps each reply as a WAV plus ``audio/manifest.jsonl`` for WER / SIM.

Usage:
  python stream_bench.py --codes ref_codes_en.jsonl --concurrency 64 --out results/c64
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import gc
import json
import platform
import random
import re
import socket
import sys
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np

SAMPLE_WIDTH = 2  # pcm s16le
DEFAULT_SAMPLE_RATE = 44100
SAMPLE_RATE_HEADER = "X-Audio-Sample-Rate"
CHANNELS_HEADER = "X-Audio-Channels"
JSON_HEADERS = {"Content-Type": "application/json"}


@dataclass
class Row:
    index: int
    utt_id: str
    text: str
    ref_text: str
    prompt_wav: str
    ref_audio_codes: list[list[int]] | None


@dataclass
class Request:
    row: Row
    payload: bytes
    seed: int | None


def load_rows(path: Path) -> list[Row]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows.append(
                    Row(len(rows), r["utt_id"], r["target_text"], r["ref_text"], r["prompt_wav"], r["ref_audio_codes"])
                )
    return rows


def build_requests(rows: list[Row], args: argparse.Namespace, seed_offset: int = 0) -> list[Request]:
    """JSON bodies serialized before the timed phase, so the loop only sends bytes."""
    extra = json.loads(args.extra_body) if args.extra_body else {}
    locale_dir = Path(args.seed_tts_root).expanduser() / args.locale if args.seed_tts_root else None
    requests = []
    for row in rows:
        body: dict[str, Any] = {
            "model": args.model,
            "input": row.text,
            "ref_text": row.ref_text,
            "response_format": "pcm",
            "stream": True,
            "stream_format": "audio",
        }
        if args.ref_mode == "codes":
            body["ref_audio_codes"] = row.ref_audio_codes
        else:
            wav = (locale_dir / row.prompt_wav).read_bytes()
            body["ref_audio"] = "data:audio/wav;base64," + base64.b64encode(wav).decode()
        seed = None if args.seed is None else args.seed + seed_offset + row.index
        if seed is not None:
            body["seed"] = seed
        body.update(extra)
        requests.append(Request(row, json.dumps(body, separators=(",", ":")).encode(), seed))
    return requests


def playback(
    arrival_s: list[float], arrival_bytes: list[int], sample_rate: int, channels: int
) -> tuple[float | None, list[float], list[float]]:
    """When playback starts, the waits a player hits, and the audio seconds each arrival adds.

    Playback starts at the first whole sample. An arrival after everything
    received before it has played out is a wait of that gap, and the queued
    audio then plays from the arrival.
    """
    frame_bytes = SAMPLE_WIDTH * channels
    received = 0
    samples_before = 0
    start = None
    play_end = 0.0
    waits: list[float] = []
    added: list[float] = []
    for t, n in zip(arrival_s, arrival_bytes):
        received += n
        samples = received // frame_bytes
        seconds = (samples - samples_before) / sample_rate
        samples_before = samples
        added.append(seconds)
        if seconds == 0.0:
            continue
        if start is None:
            start = play_end = t
        elif t > play_end:
            waits.append(t - play_end)
            play_end = t
        play_end += seconds
    return start, waits, added


def write_wav(path: Path, pcm: bytes, sample_rate: int, channels: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(sample_rate)
        w.writeframes(pcm[: len(pcm) // (SAMPLE_WIDTH * channels) * SAMPLE_WIDTH * channels])


async def run_one(
    session: aiohttp.ClientSession, url: str, req: Request, args: argparse.Namespace, audio_dir: Path | None
) -> dict[str, Any]:
    row = req.row
    rec: dict[str, Any] = {"index": row.index, "utt_id": row.utt_id, "seed": req.seed, "ok": False}
    arrival_s: list[float] = []
    arrival_bytes: list[int] = []
    pcm: list[bytes] | None = [] if audio_dir is not None else None
    sample_rate, channels = args.sample_rate, 1
    sent = time.perf_counter()
    rec["sent_s"] = sent
    try:
        async with session.post(url, data=req.payload, headers=JSON_HEADERS) as resp:
            rec["status"] = resp.status
            if resp.status != 200:
                rec["error"] = (await resp.text())[:500]
                return rec
            sample_rate = int(resp.headers.get(SAMPLE_RATE_HEADER, sample_rate))
            channels = int(resp.headers.get(CHANNELS_HEADER, channels))
            async for chunk in resp.content.iter_any():
                if chunk:
                    arrival_s.append(time.perf_counter())
                    arrival_bytes.append(len(chunk))
                    if pcm is not None:
                        pcm.append(chunk)
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"[:500]
    rec.update(sample_rate=sample_rate, channels=channels, bytes=sum(arrival_bytes), n_chunks=len(arrival_s))
    start, waits, added = playback(arrival_s, arrival_bytes, sample_rate, channels)
    audio_s = float(sum(added))
    rec["audio_s"] = round(audio_s, 4)
    if start is None:
        rec.setdefault("error", "no audio in the response")
        return rec
    rec["ttfa_ms"] = round((start - sent) * 1000, 2)
    rec["e2e_s"] = round(arrival_s[-1] - sent, 4)
    rec["rtf"] = round((arrival_s[-1] - sent) / audio_s, 4)
    rec["waits_ms"] = [round(w * 1000, 2) for w in waits]
    # [ms since sent, bytes so far] per read: bursts mark the codec windows (analyze_audio.py).
    received = 0
    rec["chunk_log"] = []
    for t, n in zip(arrival_s, arrival_bytes):
        received += n
        rec["chunk_log"].append([round((t - sent) * 1000, 2), received])
    # Kept for the full-concurrency throughput window; dropped before writing requests.jsonl.
    rec["_arrivals"] = (arrival_s, added)
    if "error" not in rec:
        rec["ok"] = True
    if pcm is not None:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", row.utt_id)
        name = f"{row.index:04d}_{safe_id}.wav"
        rec["wav"] = name
        await asyncio.get_running_loop().run_in_executor(
            None, write_wav, audio_dir / name, b"".join(pcm), sample_rate, channels
        )
    return rec


async def run_phase(
    session: aiohttp.ClientSession,
    url: str,
    requests: list[Request],
    concurrency: int,
    args: argparse.Namespace,
    audio_dir: Path | None,
) -> tuple[list[dict[str, Any]], float, float, float]:
    """``concurrency`` workers draining ``requests``; returns records, start, last dispatch, end."""
    pending = deque(requests)
    records: list[dict[str, Any]] = []
    last_dispatch = 0.0

    async def worker() -> None:
        nonlocal last_dispatch
        while pending:
            req = pending.popleft()
            if not pending:
                last_dispatch = time.perf_counter()
            records.append(await run_one(session, url, req, args, audio_dir))

    start = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(min(concurrency, len(requests)))))
    return records, start, last_dispatch, time.perf_counter()


async def loop_lag(stop: asyncio.Event, lags: list[float], interval: float = 0.01) -> None:
    """How late the event loop wakes: the error it adds to every client timestamp."""
    while not stop.is_set():
        t = time.perf_counter()
        await asyncio.sleep(interval)
        lags.append(time.perf_counter() - t - interval)


def stats(values: list[float], digits: int = 2) -> dict[str, float | None]:
    if not values:
        return {k: None for k in ("mean", "p50", "p90", "p95", "p99", "max")}
    a = np.asarray(values, dtype=np.float64)
    out = {"mean": float(a.mean())}
    out.update({f"p{q}": float(np.percentile(a, q)) for q in (50, 90, 95, 99)})
    out["max"] = float(a.max())
    return {k: round(v, digits) for k, v in out.items()}


def summarize(
    records: list[dict[str, Any]], start: float, last_dispatch: float, end: float, thresholds_ms: list[float]
) -> dict[str, Any]:
    ok = [r for r in records if r["ok"]]
    total_audio = sum(r["audio_s"] for r in ok)
    # Audio that reached the client while every slot was busy.
    window_audio = 0.0
    for r in ok:
        times, added = r["_arrivals"]
        window_audio += sum(s for t, s in zip(times, added) if t <= last_dispatch)
    # Shorter windows (rows barely outnumber the slots) say nothing about the steady state.
    window = last_dispatch - start
    window_throughput = round(window_audio / window, 2) if window >= 1.0 else None
    stall = {}
    for threshold in thresholds_ms:
        stalled = [r for r in ok if any(w > threshold for w in r["waits_ms"])]
        events = sum(sum(w > threshold for w in r["waits_ms"]) for r in ok)
        stall[f"{threshold:g}ms"] = {
            "streams_stalled": len(stalled),
            "streams_stalled_pct": round(100.0 * len(stalled) / len(ok), 2) if ok else None,
            "events": events,
        }
    failures = [r for r in records if not r["ok"]]
    return {
        "requests": len(records),
        "succeeded": len(ok),
        "failed": len(failures),
        "failure_examples": [
            {"utt_id": r["utt_id"], "status": r.get("status"), "error": r.get("error")} for r in failures[:5]
        ],
        "duration_s": round(end - start, 3),
        "audio_s_total": round(total_audio, 3),
        "throughput_audio_s_per_s": round(total_audio / (end - start), 2) if end > start else None,
        "full_concurrency_window": {
            "seconds": round(window, 3),
            "throughput_audio_s_per_s": window_throughput,
        },
        "ttfa_ms": stats([r["ttfa_ms"] for r in ok], 1),
        "rtf": stats([r["rtf"] for r in ok], 4),
        "audio_s": stats([r["audio_s"] for r in ok], 3),
        "underrun_s_total": round(sum(sum(r["waits_ms"]) for r in ok) / 1000, 3),
        "longest_wait_ms": stats([max(r["waits_ms"], default=0.0) for r in ok], 1),
        "stall": stall,
    }


async def fetch_json(session: aiohttp.ClientSession, url: str) -> Any:
    async with session.get(url) as resp:
        resp.raise_for_status()
        return await resp.json(content_type=None)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_rows(Path(args.codes))
    if args.ref_mode == "audio" and not args.seed_tts_root:
        raise SystemExit("--ref-mode audio needs --seed-tts-root to read the prompt wavs")
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(rows)
    timed_rows = rows
    if args.rows:
        missing = sorted(set(args.rows) - {row.index for row in rows})
        if missing:
            raise SystemExit(f"--rows {missing} not in {args.codes} ({len(rows)} rows)")
        timed_rows = [row for row in rows if row.index in set(args.rows)]
    if args.num_prompts:
        timed_rows = timed_rows[: args.num_prompts]
    warmups = build_requests(rows[: args.num_warmups], args, seed_offset=1_000_000)
    timed = build_requests(timed_rows, args)
    if not timed:
        raise SystemExit(f"no rows in {args.codes}")
    for row in rows:
        row.ref_audio_codes = None  # serialized into the payloads; spare the collector these lists

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    audio_dir = out / "audio" if args.save_audio else None
    if audio_dir is not None:
        audio_dir.mkdir(exist_ok=True)

    base = args.base_url.rstrip("/")
    url = base + "/v1/audio/speech"
    connector = aiohttp.TCPConnector(limit=0, force_close=False)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async with session.get(base + "/health") as resp:
            if resp.status != 200:
                raise SystemExit(f"{base}/health answered {resp.status}")
        try:
            models = await fetch_json(session, base + "/v1/models")
        except aiohttp.ClientError as exc:
            models = {"error": str(exc)}

        if warmups:
            print(f"warmup: {len(warmups)} requests", flush=True)
            warm, *_ = await run_phase(session, url, warmups, min(args.concurrency, len(warmups)), args, None)
            bad = [r for r in warm if not r["ok"]]
            if bad:
                raise SystemExit(
                    f"{len(bad)} warmup requests failed, e.g. {bad[0].get('status')}: {bad[0].get('error')}"
                )

        print(f"timed: {len(timed)} requests at concurrency {args.concurrency}", flush=True)
        # Keep collector pauses out of the timestamps: everything allocated so far is frozen.
        gc.collect()
        gc.freeze()
        stop = asyncio.Event()
        lags: list[float] = []
        lag_task = asyncio.create_task(loop_lag(stop, lags))
        started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        records, start, last_dispatch, end = await run_phase(session, url, timed, args.concurrency, args, audio_dir)
        stop.set()
        await lag_task

    records.sort(key=lambda r: r["index"])
    summary = summarize(records, start, last_dispatch, end, args.stall_thresholds_ms)
    lag_ms = [x * 1000 for x in lags]
    summary = {
        "label": args.label,
        "concurrency": args.concurrency,
        "started_at": started_at,
        **summary,
        "client": {
            "loop_lag_ms": stats(lag_ms, 2),
            "event_loop": type(asyncio.get_running_loop()).__module__,
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
            "aiohttp": aiohttp.__version__,
        },
        "workload": {
            "codes_file": str(Path(args.codes).resolve()),
            "codes_meta": _codes_meta(Path(args.codes)),
            "ref_mode": args.ref_mode,
            "rows_timed": len(timed),
            "warmups": len(warmups),
            "shuffle_seed": args.shuffle_seed,
            "seed": args.seed,
            "extra_body": json.loads(args.extra_body) if args.extra_body else {},
            "stream": "raw pcm (stream_format=audio)",
            "sampling": "server defaults unless set in extra_body",
        },
        "server": {"base_url": base, "models": models},
        "meta": dict(kv.split("=", 1) for kv in args.meta),
        "argv": sys.argv,
    }
    with (out / "requests.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            r = {k: v for k, v in r.items() if not k.startswith("_")}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if audio_dir is not None:
        by_index = {row.index: row for row in timed_rows}
        with (audio_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                if r.get("wav"):
                    row = by_index[r["index"]]
                    entry = {"utt_id": row.utt_id, "wav": r["wav"], "ok": r["ok"], "target_text": row.text}
                    entry.update(ref_text=row.ref_text, prompt_wav=row.prompt_wav, sample_rate=r["sample_rate"])
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def _codes_meta(codes: Path) -> dict[str, Any] | None:
    meta = codes.with_name(codes.name + ".meta.json")
    if not meta.is_file():
        return None
    m = json.loads(meta.read_text(encoding="utf-8"))
    keys = ("model", "codec_sha256", "locale", "meta_lst_sha256", "rows_written", "device_name", "vllm_omni_commit")
    return {k: m.get(k) for k in keys}


def print_row(s: dict[str, Any]) -> None:
    stalls = " / ".join(f"{v['streams_stalled_pct']}% (>{k})" for k, v in s["stall"].items())
    print(
        "| Concurrency | Audio s/s | TTFA p50 | TTFA p95 | RTF (mean) | Streams that stall | Failed |\n"
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"| {s['concurrency']} | {s['throughput_audio_s_per_s']} | {s['ttfa_ms']['p50']} ms "
        f"| {s['ttfa_ms']['p95']} ms | {s['rtf']['mean']} | {stalls} | {s['failed']} |"
    )
    lag = s["client"]["loop_lag_ms"]
    if lag["p99"] is not None and lag["p99"] > 10:
        print(f"WARNING: client event-loop lag p99 {lag['p99']} ms; timestamps carry that error", file=sys.stderr)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--codes", required=True, help="JSON Lines from encode_ref_codes.py")
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--out", required=True, help="Result directory")
    p.add_argument("--base-url", default="http://127.0.0.1:8091")
    p.add_argument("--model", default="fishaudio/drama-3", help="Served model name")
    p.add_argument("--num-prompts", type=int, default=None, help="Time only the first N rows (default: all)")
    p.add_argument(
        "--rows",
        type=lambda v: [int(x) for x in v.split(",")],
        default=None,
        help="Time only these rows: 0-based line numbers of --codes, the prefix of --save-audio WAV names",
    )
    p.add_argument("--num-warmups", type=int, default=8, help="Unmeasured requests before the timed run")
    p.add_argument(
        "--ref-mode",
        choices=["codes", "audio"],
        default="codes",
        help="codes: ref_audio_codes (PR #8933); audio: the wav inline as ref_audio",
    )
    p.add_argument("--seed-tts-root", default=None, help="Seed-TTS root, for --ref-mode audio")
    p.add_argument("--locale", default="en", choices=["en", "zh"])
    p.add_argument("--seed", type=int, default=None, help="Request seed = this + row index (default: none)")
    p.add_argument("--shuffle-seed", type=int, default=None, help="Shuffle rows with this seed (default: file order)")
    p.add_argument("--extra-body", default=None, help="JSON merged into every request body")
    p.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Used only when the reply has no {SAMPLE_RATE_HEADER} header",
    )
    p.add_argument("--timeout", type=float, default=900.0, help="Per-request timeout (s)")
    p.add_argument("--stall-thresholds-ms", type=lambda s: [float(x) for x in s.split(",")], default=[0.0, 100.0])
    p.add_argument("--save-audio", action="store_true", help="Keep every reply as a WAV (for WER / SIM)")
    p.add_argument("--label", default="", help="Free-form run label (e.g. baseline / a100-tuned)")
    p.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE", help="Provenance kept in summary.json")
    args = p.parse_args(argv)
    if args.concurrency < 1:
        p.error("--concurrency must be >= 1")
    for kv in args.meta:
        if "=" not in kv:
            p.error(f"--meta expects KEY=VALUE, got {kv!r}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import uvloop

        runner = uvloop.run
    except ImportError:
        runner = asyncio.run
    summary = runner(run(args))
    print_row(summary)
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
