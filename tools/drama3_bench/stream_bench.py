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

Thermal gate (``--thermal-gate --gpu-index N``), for a GPU that throttles under
sustained load: the rows run in segments. A segment starts once the GPU has
cooled to ``--resume-temp``; when it reaches ``--pause-temp`` (or reports a
gated clock limit) no new request is sent, the ones in flight finish, and the
next segment waits for the GPU to cool again. If the GPU limits its clocks
during a segment anyway, the segment ends at that moment: requests that
finished before stand, the ones in flight run again later. Every number then
comes from unthrottled time and the cooling pauses are left out of the
durations; each segment ramps up and drains, so compare the full-concurrency
throughput and the ``steady`` latencies (requests that ran entirely while every
slot was busy) rather than the wall-clock rate.

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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

SAMPLE_WIDTH = 2  # pcm s16le
FRAME_SAMPLES = 2048  # one Drama 3 codec frame (~46.4 ms at 44.1 kHz)
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


@dataclass
class Phase:
    records: list[dict[str, Any]]
    dispatched: list[Request]
    start: float
    window_start: float  # every slot busy from here...
    last_dispatch: float  # ...until the last request was sent
    end: float
    # The full-concurrency window ends at the last dispatch, or earlier when the GPU limited
    # its clocks; its audio counts every request of the phase, also those run again later.
    window_end: float | None = None
    window_records: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.window_end is None:
            self.window_end = self.last_dispatch
        if self.window_records is None:
            self.window_records = self.records


async def run_phase(
    session: aiohttp.ClientSession,
    url: str,
    pending: deque[Request],
    concurrency: int,
    args: argparse.Namespace,
    audio_dir: Path | None,
    *,
    segment: int = 0,
    should_stop: Callable[[], bool] | None = None,
    limit: int | None = None,
    ramp_s: float = 0.0,
) -> Phase:
    """``concurrency`` workers taking requests off ``pending`` until it is empty, ``limit``
    requests were sent or ``should_stop()``; requests left are still in ``pending``.

    With ``ramp_s`` the workers start one after another over that many seconds,
    so the phase does not open with ``concurrency`` prompts at once.
    """
    records: list[dict[str, Any]] = []
    dispatched: list[Request] = []
    first_dispatch: list[float] = []
    last_dispatch = 0.0
    halted = False
    workers = min(concurrency, len(pending), limit or len(pending))

    async def worker(i: int) -> None:
        nonlocal last_dispatch, halted
        if ramp_s > 0:
            await asyncio.sleep(i * ramp_s / workers)
        first = True
        while pending and not halted and (limit is None or len(dispatched) < limit):
            if should_stop is not None and should_stop():
                halted = True  # for every worker: the phase drains, it does not run on with fewer slots
                break
            req = pending.popleft()
            last_dispatch = time.perf_counter()
            dispatched.append(req)
            if first:
                first_dispatch.append(last_dispatch)
            rec = await run_one(session, url, req, args, audio_dir)
            rec["done_s"] = time.perf_counter()
            rec["segment"] = segment
            if first:
                rec["first_in_segment"] = True
                first = False
            records.append(rec)

    start = time.perf_counter()
    await asyncio.gather(*(worker(i) for i in range(workers)))
    # Full concurrency only once every worker is sending (the last one, with a ramp).
    window_start = max(first_dispatch) if len(first_dispatch) == workers else last_dispatch
    return Phase(records, dispatched, start, window_start, last_dispatch, time.perf_counter())


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


def stall_counts(ok: list[dict[str, Any]], thresholds_ms: list[float]) -> dict[str, Any]:
    out = {}
    for threshold in thresholds_ms:
        stalled = [r for r in ok if any(w > threshold for w in r["waits_ms"])]
        events = sum(sum(w > threshold for w in r["waits_ms"]) for r in ok)
        out[f"{threshold:g}ms"] = {
            "streams_stalled": len(stalled),
            "streams_stalled_pct": round(100.0 * len(stalled) / len(ok), 2) if ok else None,
            "events": events,
        }
    return out


def summarize(
    records: list[dict[str, Any]],
    phases: list[Any],
    thresholds_ms: list[float],
    max_frames: int = 500,
) -> dict[str, Any]:
    """Totals over ``records``, each tagged with the index of its phase (segment) in ``phases``."""
    ok = [r for r in records if r["ok"]]
    total_audio = sum(r["audio_s"] for r in ok)
    duration = sum(p.end - p.start for p in phases)
    # Audio that reached the client while every slot was busy; windows under 1 s (rows
    # barely outnumber the slots) say nothing about the steady state and are left out.
    window = window_audio = 0.0
    for p in phases:
        if p.window_end - p.window_start < 1.0:
            continue
        window += p.window_end - p.window_start
        for r in p.window_records:
            if r["ok"]:
                times, added = r["_arrivals"]
                window_audio += sum(s for t, s in zip(times, added) if p.window_start <= t <= p.window_end)
    window_throughput = round(window_audio / window, 2) if window >= 1.0 else None
    stall = stall_counts(ok, thresholds_ms)

    # Requests sent and finished inside their phase's full-concurrency window.
    def steady(r: dict[str, Any]) -> bool:
        p = phases[r.get("segment", 0)]
        return r["sent_s"] >= p.window_start and r["sent_s"] + r["e2e_s"] <= p.window_end

    steady_ok = [r for r in ok if steady(r)]
    failures = [r for r in records if not r["ok"]]
    return {
        "requests": len(records),
        "succeeded": len(ok),
        "failed": len(failures),
        "failure_examples": [
            {"utt_id": r["utt_id"], "status": r.get("status"), "error": r.get("error")} for r in failures[:5]
        ],
        "segments": len(phases),
        "duration_s": round(duration, 3),
        "audio_s_total": round(total_audio, 3),
        "throughput_audio_s_per_s": round(total_audio / duration, 2) if duration > 0 else None,
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
        "steady": {
            "requests": len(steady_ok),
            "ttfa_ms": stats([r["ttfa_ms"] for r in steady_ok], 1),
            "rtf": stats([r["rtf"] for r in steady_ok], 4),
            "stall": stall_counts(steady_ok, thresholds_ms),
        },
        "long_takes": long_takes(ok, total_audio, max_frames),
    }


def long_takes(ok: list[dict[str, Any]], total_audio: float, max_frames: int) -> dict[str, Any]:
    """Takes that ran long (most stopped speaking): they add audio seconds without speech.

    20 s is #8933's runaway rule; ``max_frames`` is the deploy's max_tokens.
    """
    at_cap = [r for r in ok if r["bytes"] // (SAMPLE_WIDTH * r["channels"] * FRAME_SAMPLES) >= max_frames]
    long = [r for r in ok if r["audio_s"] >= 20.0]
    long_audio = sum(r["audio_s"] for r in long)
    return {
        "at_least_20s": len(long),
        "at_least_20s_pct": round(100 * len(long) / len(ok), 2) if ok else None,
        "at_frame_cap": len(at_cap),
        "audio_share_pct": round(100 * long_audio / total_audio, 2) if total_audio else None,
    }


async def wait_cool(monitor: Any, args: argparse.Namespace) -> float:
    """Seconds spent waiting for the GPU to cool to the resume temperature."""
    t0 = last_print = time.perf_counter()
    while not monitor.cool():
        monitor.check_alive()
        now = time.perf_counter()
        if now - t0 > args.max_cool_wait_s:
            raise SystemExit(f"GPU still above {monitor.resume_temp} C after {args.max_cool_wait_s:.0f} s")
        if now - last_print >= 10:
            last = monitor.latest()
            print(f"cooling: {last.temp_c} C, waiting for {monitor.resume_temp} C", flush=True)
            last_print = now
        await asyncio.sleep(0.25)
    return time.perf_counter() - t0


async def run_gated(
    session: aiohttp.ClientSession,
    url: str,
    timed: list[Request],
    args: argparse.Namespace,
    audio_dir: Path | None,
    monitor: Any,
) -> tuple[list[Phase], list[dict[str, Any]], dict[str, Any]]:
    """The timed rows in segments between cooling pauses (see the module docstring)."""
    from thermal_gate import reason_names

    pending = deque(timed)
    clean: list[Phase] = []
    discarded: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    cooling = 0.0
    fruitless = 0
    initial_pause = monitor.pause_temp
    wall0 = time.perf_counter()

    def should_stop() -> bool:
        monitor.check_alive()
        return monitor.hot()

    while pending:
        cooling += await wait_cool(monitor, args)
        phase = await run_phase(
            session,
            url,
            pending,
            args.concurrency,
            args,
            audio_dir,
            segment=len(clean),
            should_stop=should_stop,
            limit=args.segment_requests or None,
            ramp_s=args.ramp_s,
        )
        if not pending:
            ended_by = "rows done"
        elif args.segment_requests and len(phase.dispatched) >= args.segment_requests:
            ended_by = "segment limit"
        else:
            ended_by = f"pause at {monitor.latest().temp_c} C"
        info: dict[str, Any] = {
            "requests": len(phase.dispatched),
            "seconds": round(phase.end - phase.start, 2),
            "ended_by": ended_by,
            "pause_temp_c": monitor.pause_temp,
            **monitor.describe(phase.start, phase.end),
        }
        onset = monitor.first_limited(phase.start, phase.end)
        if onset is not None:
            # The limit may have begun up to one sample before it was seen. Requests that
            # finished before then stand; the rest run again, in the order they were taken.
            onset -= monitor.interval_ms / 1000
            names = reason_names(monitor.limited(phase.start, phase.end))
            keep = [r for r in phase.records if r["done_s"] <= onset]
            drop = [r for r in phase.records if r["done_s"] > onset]
            order = {req.row.index: i for i, req in enumerate(phase.dispatched)}
            by_index = {req.row.index: req for req in phase.dispatched}
            redo = sorted((by_index[r["index"]] for r in drop), key=lambda req: order[req.row.index])
            pending.extendleft(reversed(redo))
            for r in drop:
                r["discarded"] = names
            discarded.extend(drop)
            phase.window_records = phase.records
            phase.records = keep
            phase.window_end = min(phase.last_dispatch, onset)
            phase.end = max(phase.start, onset)
            info.update(limited=names, limit_onset_s=round(onset - phase.start, 2), kept=len(keep), rerun=len(drop))
            monitor.pause_temp = max(monitor.resume_temp + 3, monitor.pause_temp - 2)
            print(
                f"clock limit ({', '.join(names)}) {info['limit_onset_s']} s into the segment: kept {len(keep)}, "
                f"{len(drop)} rows queued again; pause now at {monitor.pause_temp} C",
                flush=True,
            )
        info["window_s"] = round(max(0.0, phase.window_end - phase.window_start), 2)
        segments.append(info)
        if not phase.records:
            fruitless += 1
            if fruitless >= args.max_tainted:
                raise SystemExit(f"{fruitless} segments in a row kept nothing; lower --pause-temp")
            continue
        fruitless = 0
        clean.append(phase)
        print(
            f"segment {len(clean)}: {info['requests']} requests in {info['seconds']} s "
            f"(window {info['window_s']} s), {info['temp_c'][0]}-{info['temp_c'][1]} C, "
            f"SM median {info['sm_mhz_median']} MHz, {ended_by}; {len(pending)} rows left",
            flush=True,
        )
    gate = {
        "gpu": args.gpu_index,
        "pause_temp_c": initial_pause,
        "pause_temp_final_c": monitor.pause_temp,
        "resume_temp_c": monitor.resume_temp,
        "cool_hold_s": monitor.hold_s,
        "gate_reasons": hex(monitor.gate_mask),
        "segment_requests": args.segment_requests,
        "ramp_s": args.ramp_s,
        "segments_kept": len(clean),
        "segments_limited": sum("limited" in i for i in segments),
        "rerun_requests": len(discarded),
        "cooling_wait_s": round(cooling, 1),
        "wall_s": round(time.perf_counter() - wall0, 1),
        "segments": segments,
    }
    return clean, discarded, gate


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
    monitor = None
    if args.thermal_gate:
        from thermal_gate import ThermalMonitor

        monitor = ThermalMonitor(
            args.gpu_index,
            pause_temp=args.pause_temp,
            resume_temp=args.resume_temp,
            hold_s=args.cool_hold_s,
            gate_mask=args.gate_reasons,
        )
        monitor.start()
    try:
        return await _run(args, base, url, timed_rows, warmups, timed, out, audio_dir, monitor)
    finally:
        if monitor is not None:
            monitor.stop()


async def _run(
    args: argparse.Namespace,
    base: str,
    url: str,
    timed_rows: list[Row],
    warmups: list[Request],
    timed: list[Request],
    out: Path,
    audio_dir: Path | None,
    monitor: Any,
) -> dict[str, Any]:
    connector = aiohttp.TCPConnector(limit=0, force_close=False)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    gate = None
    discarded: list[dict[str, Any]] = []
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async with session.get(base + "/health") as resp:
            if resp.status != 200:
                raise SystemExit(f"{base}/health answered {resp.status}")
        try:
            models = await fetch_json(session, base + "/v1/models")
        except aiohttp.ClientError as exc:
            models = {"error": str(exc)}

        if monitor is not None:
            await wait_cool(monitor, args)
        if warmups:
            print(f"warmup: {len(warmups)} requests", flush=True)
            warm = (await run_phase(session, url, deque(warmups), args.concurrency, args, None)).records
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
        if monitor is None:
            phase = await run_phase(session, url, deque(timed), args.concurrency, args, audio_dir, ramp_s=args.ramp_s)
            phases = [phase]
        else:
            phases, discarded, gate = await run_gated(session, url, timed, args, audio_dir, monitor)
        stop.set()
        await lag_task

    records = sorted((r for p in phases for r in p.records), key=lambda r: r["index"])
    summary = summarize(records, phases, args.stall_thresholds_ms, args.max_frames)
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
        "thermal_gate": gate,
        "meta": dict(kv.split("=", 1) for kv in args.meta),
        "argv": sys.argv,
    }
    for name, group in (("requests.jsonl", records), ("requests_discarded.jsonl", discarded)):
        if group or name == "requests.jsonl":
            with (out / name).open("w", encoding="utf-8") as f:
                for r in group:
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
    gate = s.get("thermal_gate")
    if gate:
        print(
            f"thermal gate: {gate['segments_kept']} segments kept, {gate['segments_limited']} hit a clock limit "
            f"({gate['rerun_requests']} requests rerun), {gate['cooling_wait_s']} s cooling of "
            f"{gate['wall_s']} s; full-window rate {s['full_concurrency_window']['throughput_audio_s_per_s']} "
            f"audio-s/s, steady TTFA p95 {s['steady']['ttfa_ms']['p95']} ms over {s['steady']['requests']} requests"
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
    p.add_argument("--max-frames", type=int, default=500, help="The deploy's max_tokens, for counting capped takes")
    p.add_argument("--ramp-s", type=float, default=0.0, help="Start the workers one by one over this many seconds")
    g = p.add_argument_group("thermal gate")
    g.add_argument("--thermal-gate", action="store_true", help="Pause between segments while the GPU cools")
    g.add_argument("--gpu-index", default=None, help="The served GPU as nvidia-smi -i takes it (index or UUID)")
    g.add_argument("--pause-temp", type=float, default=80.0, help="Stop sending new requests at this temperature")
    g.add_argument("--resume-temp", type=float, default=60.0, help="Start the next segment at or below this")
    g.add_argument("--cool-hold-s", type=float, default=3.0, help="...for this many seconds without a clock limit")
    g.add_argument(
        "--gate-reasons",
        type=lambda v: int(v, 0),
        default=0x68,
        help="Clock-limit reasons that pause and taint (0x68: thermal and hw slowdown; add 0x4 for power cap)",
    )
    g.add_argument("--segment-requests", type=int, default=0, help="Also end a segment after this many (0: no cap)")
    g.add_argument("--max-cool-wait-s", type=float, default=1800.0)
    g.add_argument("--max-tainted", type=int, default=5, help="Give up after this many segments in a row keep nothing")
    p.add_argument("--save-audio", action="store_true", help="Keep every reply as a WAV (for WER / SIM)")
    p.add_argument("--label", default="", help="Free-form run label (e.g. baseline / a100-tuned)")
    p.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE", help="Provenance kept in summary.json")
    args = p.parse_args(argv)
    if args.concurrency < 1:
        p.error("--concurrency must be >= 1")
    for kv in args.meta:
        if "=" not in kv:
            p.error(f"--meta expects KEY=VALUE, got {kv!r}")
    if args.thermal_gate and not args.gpu_index:
        p.error("--thermal-gate needs --gpu-index (the served GPU, as nvidia-smi -i takes it)")
    if args.thermal_gate and args.resume_temp >= args.pause_temp:
        p.error("--resume-temp must be below --pause-temp")
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
