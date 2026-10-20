#!/usr/bin/env python3
"""Microbenchmark of one Drama 3 GDN layer's mixed-step core on the A100 (FLA/Triton) path.

Calls ``mixed_forward_core`` on a synthetic step shaped like the profiled mixed
steps (``drama3_gdn_harness.PRESETS``), with no server, CUDA graphs or second
stage around it. Variants:

* ``installed``: vllm_omni's own ``mixed_forward_core``;
* ``baseline``: the instrumented copy, same work (its numbers should match ``installed``);
* ``direct_out``: the copy with FLA writing the prefill output straight into
  ``core_attn_out`` (no output allocation and copy).

Each iteration restores the state pools, synchronizes, then times one call per
variant, in a fresh random order, so the variants share the GPU's thermal
state. Per call it reports:

* host us: perf_counter_ns around the call, with an idle GPU and no
  synchronization inside it: what the CPU spends issuing the work;
* GPU span us: CUDA events around the call; it covers GPU idle time while the
  GPU waits for the CPU, so span close to host time means the call is host-bound;
* kernel us (``--kernel-time``): the summed device time of its kernels under
  torch.profiler, in a separate pass (the profiler inflates host time, not
  kernel durations).

A mixed step runs 30 GDN layers, so "x30" is the step's estimate.

Usage (from the vllm-omni checkout, on the GPU you will profile):
  CUDA_VISIBLE_DEVICES=1 python tools/drama3_bench/gdn_mixed/bench_mixed_core.py --cases c8 c16 c64 \
      --state-dtype bf16 --iters 200 --kernel-time --json bench.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import drama3_gdn_harness as harness  # noqa: E402

GDN_LAYERS = 30
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def variant_fns():
    import drama3_gdn_variants as variants
    from vllm_omni.model_executor.models.fish_omni.gdn import mixed_forward_core as installed

    def copy(direct_out: bool, timer=None):
        def fn(layer, mixed_qkv, b, a, out, md):
            variants.mixed_forward_core(layer, mixed_qkv, b, a, out, md, direct_out=direct_out, timer=timer)

        return fn

    return {"installed": installed, "baseline": copy(False), "direct_out": copy(True)}, variants


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, round(q / 100 * (len(values) - 1)))]


def kernel_us(fn, args, restore, iters: int) -> float:
    """Summed device time of the kernels of one call, under torch.profiler."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        restore()
        fn(*args)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            restore()
            torch.cuda.synchronize()
            fn(*args)
        torch.cuda.synchronize()
    restore_time = 0.0
    total = 0.0
    for evt in prof.key_averages():
        t = getattr(evt, "self_device_time_total", None)
        if t is None:
            t = getattr(evt, "self_cuda_time_total", 0.0)
        # The pool restores are copies too: measured separately and subtracted below.
        total += t
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            restore()
            torch.cuda.synchronize()
        torch.cuda.synchronize()
    for evt in prof.key_averages():
        t = getattr(evt, "self_device_time_total", None)
        restore_time += t if t is not None else getattr(evt, "self_cuda_time_total", 0.0)
    return (total - restore_time) / iters


def run_case(spec: str, state_dtype: torch.dtype, args, fns, variants_mod) -> list[dict]:
    device = torch.device("cuda")
    step = harness.parse_step(spec)
    conv0, ssm0 = harness.make_pools(device, step.rows, state_dtype)
    pools = (conv0.clone(), ssm0.clone())
    layer, dim = harness.make_layer(device, pools)
    md = harness.make_metadata(device, step)
    mixed_qkv, b, a, out = harness.make_inputs(device, step.tokens, dim)
    call_args = (layer, mixed_qkv, b, a, out, md)

    def restore():
        pools[0].copy_(conv0)
        pools[1].copy_(ssm0)

    names = [v for v in args.variants if v in fns]
    for _ in range(args.warmup):
        for name in names:
            restore()
            fns[name](*call_args)
    torch.cuda.synchronize()

    host: dict[str, list[float]] = {n: [] for n in names}
    span: dict[str, list[float]] = {n: [] for n in names}
    rng = random.Random(0)
    start_ev, end_ev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(args.iters):
        order = names[:]
        rng.shuffle(order)
        for name in order:
            restore()
            torch.cuda.synchronize()
            start_ev.record()
            t0 = time.perf_counter_ns()
            fns[name](*call_args)
            t1 = time.perf_counter_ns()
            end_ev.record()
            torch.cuda.synchronize()
            host[name].append((t1 - t0) / 1e3)
            span[name].append(start_ev.elapsed_time(end_ev) * 1e3)

    # Section laps of the copy, in their own pass so they do not touch the timings above.
    sections: dict[str, dict[str, float]] = {}
    for name, direct_out in (("baseline", False), ("direct_out", True)):
        if name not in names:
            continue
        acc: dict[str, list[float]] = {}
        for _ in range(min(args.iters, 100)):
            restore()
            torch.cuda.synchronize()
            timer = variants_mod.LapTimer()
            variants_mod.mixed_forward_core(*call_args, direct_out=direct_out, timer=timer)
            torch.cuda.synchronize()
            for sec, ns in timer.laps.items():
                acc.setdefault(sec, []).append(ns / 1e3)
        sections[name] = {sec: statistics.median(v) for sec, v in acc.items()}

    rows = []
    for name in names:
        row = {
            "case": spec,
            "step": harness.PRESETS.get(spec, spec),
            "state_dtype": str(state_dtype).removeprefix("torch."),
            "variant": name,
            "decode_tokens": step.decodes,
            "prefill_tokens": step.tokens - step.decodes,
            "host_us_p50": statistics.median(host[name]),
            "host_us_p90": pct(host[name], 90),
            "gpu_span_us_p50": statistics.median(span[name]),
            "sections_host_us_p50": sections.get(name),
        }
        if args.kernel_time:
            row["kernel_us"] = kernel_us(fns[name], call_args, restore, min(args.iters, 50))
        rows.append(row)
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cases", nargs="+", default=["c8", "c16", "c64"], help=f"Presets {list(harness.PRESETS)} or D:P")
    p.add_argument("--state-dtype", choices=["bf16", "fp32", "both"], default="bf16")
    p.add_argument("--variants", nargs="+", default=["installed", "baseline", "direct_out"])
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--kernel-time", action="store_true", help="Also sum kernel device time under torch.profiler")
    p.add_argument("--json", default=None)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    fns, variants_mod = variant_fns()
    matches, digest = variants_mod.installed_matches()
    print(f"installed mixed_forward_core sha256 {digest}: {'matches' if matches else 'DIFFERS FROM'} the copy")
    if not matches:
        print("WARNING: baseline / direct_out copy a different version than the installed one", file=sys.stderr)
    print(f"GPU {torch.cuda.get_device_name()}, torch {torch.__version__}")

    dtypes = ["bf16", "fp32"] if args.state_dtype == "both" else [args.state_dtype]
    rows = []
    for spec in args.cases:
        for d in dtypes:
            rows.extend(run_case(spec, DTYPES[d], args, fns, variants_mod))

    kt = " | kernel us" if args.kernel_time else ""
    print(f"\n| case | state | variant | host us p50 (p90) | GPU span us p50{kt} | host ms x{GDN_LAYERS} |")
    print("| --- | --- | --- | ---: | ---:" + (" | ---:" if args.kernel_time else "") + " | ---: |")
    for r in rows:
        k = f" | {r['kernel_us']:.0f}" if args.kernel_time else ""
        print(
            f"| {r['case']} ({r['decode_tokens']}+{r['prefill_tokens']}) | {r['state_dtype']} | {r['variant']} "
            f"| {r['host_us_p50']:.0f} ({r['host_us_p90']:.0f}) | {r['gpu_span_us_p50']:.0f}{k} "
            f"| {r['host_us_p50'] * GDN_LAYERS / 1e3:.2f} |"
        )
    print("\nhost us per section (p50), instrumented copy")
    for r in rows:
        if r["sections_host_us_p50"]:
            secs = ", ".join(f"{k} {v:.0f}" for k, v in r["sections_host_us_p50"].items())
            print(f"  {r['case']} {r['state_dtype']} {r['variant']}: {secs}")
    if args.json:
        meta = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "installed_sha256": digest}
        Path(args.json).write_text(json.dumps({"meta": meta, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
