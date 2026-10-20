# SPDX-License-Identifier: Apache-2.0
"""Host-time probe of Drama 3's mixed-step GDN core in a running server, without torch.profiler.

``probe_site/sitecustomize.py`` calls :func:`install` in every Python process
started with ``PYTHONPATH=<this dir>/probe_site`` and ``DRAMA3_GDN_PROBE=1``.
An import hook then replaces ``mixed_forward_core`` in
``vllm_omni.model_executor.models.fish_omni.gdn`` right after that module
loads, with the instrumented copy in :mod:`drama3_gdn_variants` (only when the
installed function is the one the copy was made from). Stage 0 calls it
30 times per step that holds prefills, outside the CUDA graphs.

Per mixed step it keeps: the number of calls, their summed host time, the span
from the first call's start to the last call's end, the time per section
(conv, decode, prep, gather, chunk, scatter, copy), and the step's shape. The
records stay in memory and are appended to ``<DRAMA3_GDN_PROBE_DIR>/
gdn_probe_<pid>.jsonl`` every 32 steps (before the next step is timed) and at
exit; a process killed with SIGKILL loses at most the last 31 steps.

Environment:
  DRAMA3_GDN_PROBE=1              turn the probe on
  DRAMA3_GDN_PROBE_DIR=<dir>      output directory (default ./gdn_probe)
  DRAMA3_GDN_VARIANT=baseline     baseline (the installed code) or direct_out
  DRAMA3_GDN_PROBE_LAPS=1         0: time only whole calls (bounds the laps' own cost)
  DRAMA3_GDN_PROBE_FORCE=1        patch even if the installed function differs
"""

from __future__ import annotations

import atexit
import importlib.abc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

TARGET = "vllm_omni.model_executor.models.fish_omni.gdn"
FLUSH_EVERY = 32


def _log(msg: str) -> None:
    print(f"[drama3-gdn-probe pid={os.getpid()}] {msg}", file=sys.stderr, flush=True)


class StepRecorder:
    """Aggregates the calls of one forward pass (one ForwardContext) into a step record."""

    def __init__(self, path: Path, variant: str, laps: bool) -> None:
        self.path, self.variant, self.laps = path, variant, laps
        self.buffer: list[dict[str, Any]] = []
        self.ctx: Any = None  # held, so a new context can never reuse its id
        self.rec: dict[str, Any] | None = None
        atexit.register(self.close)

    def enter(self, ctx: Any, md: Any) -> None:
        if ctx is self.ctx and self.rec is not None:
            return
        self._finish()
        if len(self.buffer) >= FLUSH_EVERY:
            self.flush()
        self.ctx = ctx
        self.rec = {
            "t": time.time(),
            "pid": os.getpid(),
            "variant": self.variant,
            "laps": self.laps,
            "calls": 0,
            "sum_ns": 0,
            "first_ns": None,
            "last_ns": None,
            "sections_ns": {},
            "num_prefills": int(md.num_prefills),
            "prefill_tokens": int(md.num_actual_tokens) - int(md.num_decode_tokens),
            "decode_tokens": int(md.num_decode_tokens),
        }

    def leave(self, start_ns: int, end_ns: int, laps: dict[str, int] | None) -> None:
        rec = self.rec
        assert rec is not None
        rec["calls"] += 1
        rec["sum_ns"] += end_ns - start_ns
        if rec["first_ns"] is None:
            rec["first_ns"] = start_ns
        rec["last_ns"] = end_ns
        if laps:
            sections = rec["sections_ns"]
            for name, ns in laps.items():
                sections[name] = sections.get(name, 0) + ns

    def _finish(self) -> None:
        rec = self.rec
        if rec is None or rec["calls"] == 0:
            return
        rec["span_ns"] = rec.pop("last_ns") - rec.pop("first_ns")
        self.buffer.append(rec)
        self.rec = None

    def flush(self) -> None:
        if not self.buffer:
            return
        with self.path.open("a", encoding="utf-8") as f:
            for rec in self.buffer:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self.buffer.clear()

    def close(self) -> None:
        self._finish()
        self.flush()


def _patch(module: Any) -> None:
    from vllm.forward_context import get_forward_context

    import drama3_gdn_variants as variants

    matches, digest = variants.installed_matches()
    if not matches and os.environ.get("DRAMA3_GDN_PROBE_FORCE") != "1":
        _log(
            f"NOT patched: installed mixed_forward_core sha256 {digest} is not the copied "
            f"{variants.EXPECTED_SOURCE_SHA256} (set DRAMA3_GDN_PROBE_FORCE=1 to patch anyway)"
        )
        return
    variant = os.environ.get("DRAMA3_GDN_VARIANT", "baseline")
    if variant not in ("baseline", "direct_out"):
        _log(f"NOT patched: unknown DRAMA3_GDN_VARIANT={variant!r}")
        return
    laps = os.environ.get("DRAMA3_GDN_PROBE_LAPS", "1") != "0"
    out_dir = Path(os.environ.get("DRAMA3_GDN_PROBE_DIR", "gdn_probe"))
    out_dir.mkdir(parents=True, exist_ok=True)
    recorder = StepRecorder(out_dir / f"gdn_probe_{os.getpid()}.jsonl", variant, laps)
    direct_out = variant == "direct_out"
    copy = variants.mixed_forward_core
    clock = time.perf_counter_ns

    def probed_mixed_forward_core(layer, mixed_qkv, b, a, core_attn_out, attn_metadata):
        recorder.enter(get_forward_context(), attn_metadata)
        if laps:
            timer = variants.LapTimer()
            copy(layer, mixed_qkv, b, a, core_attn_out, attn_metadata, direct_out=direct_out, timer=timer)
            recorder.leave(timer.t0, timer.last, timer.laps)
        else:
            start = clock()
            copy(layer, mixed_qkv, b, a, core_attn_out, attn_metadata, direct_out=direct_out)
            recorder.leave(start, clock(), None)

    module.__drama3_original_mixed_forward_core__ = module.mixed_forward_core
    module.mixed_forward_core = probed_mixed_forward_core
    _log(f"patched mixed_forward_core: variant={variant} laps={laps} -> {recorder.path}")


class _PatchAfterImport(importlib.abc.MetaPathFinder):
    """Finds ``TARGET`` with the other finders and patches it once its code has run."""

    def find_spec(self, fullname, path, target=None):
        if fullname != TARGET:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        loader = spec.loader
        if loader is None or not hasattr(loader, "exec_module"):
            return None
        exec_module = loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            try:
                _patch(module)
            except Exception as exc:  # noqa: BLE001 - a probe must never take the server down
                _log(f"NOT patched: {type(exc).__name__}: {exc}")

        loader.exec_module = exec_and_patch
        return spec


def install() -> None:
    if TARGET in sys.modules:
        _patch(sys.modules[TARGET])
    elif not any(isinstance(f, _PatchAfterImport) for f in sys.meta_path):
        sys.meta_path.insert(0, _PatchAfterImport())
