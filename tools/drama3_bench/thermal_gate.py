"""GPU temperature and clock-limit monitor for stream_bench.py's thermal gate.

A persistent ``nvidia-smi -lms`` process samples the served GPU; a reader
thread keeps the samples with ``time.perf_counter()`` stamps, the clock the
benchmark's own timestamps use. The gate is "hot" once the GPU reaches the
pause temperature or reports a gated clock-limit reason, and "cool" once it has
stayed at or below the resume temperature, without one, for the hold time.

Clock-limit (throttle) reason bits, as nvidia-smi / NVML report them:
0x4 software power cap, 0x8 hardware slowdown, 0x20 software thermal slowdown,
0x40 hardware thermal slowdown. The default gate is 0x68 (the thermal and
hardware ones); add 0x4 to also gate on the power cap.
"""

from __future__ import annotations

import statistics
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

REASON_NAMES = {0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x20: "sw_thermal", 0x40: "hw_thermal"}
DEFAULT_GATE_MASK = 0x68


@dataclass
class Sample:
    t: float
    temp_c: int
    sm_mhz: int | None
    power_w: float | None
    reasons: int


def reason_names(mask: int) -> list[str]:
    return [name for bit, name in REASON_NAMES.items() if mask & bit]


def _number(text: str, kind: type) -> float | int | None:
    try:
        return kind(text)
    except ValueError:
        return None  # "[N/A]" and friends


def parse_line(line: str, t: float) -> Sample | None:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 4 or not parts[0].isdigit():
        return None
    reasons = int(parts[3], 16) if parts[3].lower().startswith("0x") else 0
    return Sample(t, int(parts[0]), _number(parts[1], int), _number(parts[2], float), reasons)


def reasons_field(nvidia_smi: str, gpu: str) -> str:
    """The clock-limit field this driver knows: renamed clocks_event_reasons in newer drivers."""
    for field in ("clocks_event_reasons.active", "clocks_throttle_reasons.active"):
        probe = subprocess.run(
            [nvidia_smi, "-i", gpu, f"--query-gpu={field}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0 and probe.stdout.strip().lower().startswith("0x"):
            return field
    raise RuntimeError(f"{nvidia_smi} reports neither clock-limit reasons field for GPU {gpu}")


class ThermalMonitor:
    def __init__(
        self,
        gpu: str,
        *,
        pause_temp: float,
        resume_temp: float,
        hold_s: float = 3.0,
        gate_mask: int = DEFAULT_GATE_MASK,
        interval_ms: int = 200,
        nvidia_smi: str = "nvidia-smi",
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if resume_temp >= pause_temp:
            raise ValueError(f"resume temperature {resume_temp} must be below the pause temperature {pause_temp}")
        self.gpu, self.pause_temp, self.resume_temp, self.hold_s = gpu, pause_temp, resume_temp, hold_s
        self.gate_mask, self.interval_ms, self.nvidia_smi, self.clock = gate_mask, interval_ms, nvidia_smi, clock
        self.samples: list[Sample] = []
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        field = reasons_field(self.nvidia_smi, self.gpu)
        cmd = [
            self.nvidia_smi,
            "-i",
            self.gpu,
            f"--query-gpu=temperature.gpu,clocks.sm,power.draw,{field}",
            "--format=csv,noheader,nounits",
            "-lms",
            str(self.interval_ms),
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._thread = threading.Thread(target=self._read, name="thermal-monitor", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 10
        while self.latest() is None:
            if time.monotonic() > deadline or self._proc.poll() is not None:
                raise RuntimeError(f"no samples from {' '.join(cmd)}")
            time.sleep(0.05)

    def _read(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            sample = parse_line(line, self.clock())
            if sample is not None:
                with self._lock:
                    self.samples.append(sample)

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def latest(self) -> Sample | None:
        with self._lock:
            return self.samples[-1] if self.samples else None

    def between(self, t0: float, t1: float) -> list[Sample]:
        with self._lock:
            return [s for s in self.samples if t0 <= s.t <= t1]

    def check_alive(self) -> None:
        last = self.latest()
        stale = last is None or self.clock() - last.t > max(2.0, 10 * self.interval_ms / 1000)
        if stale or (self._proc is not None and self._proc.poll() is not None):
            raise RuntimeError("the nvidia-smi monitor stopped sending samples")

    def hot(self) -> bool:
        """Stop dispatching: at the pause temperature, or a gated clock limit is active."""
        last = self.latest()
        return last is not None and (last.temp_c >= self.pause_temp or bool(last.reasons & self.gate_mask))

    def cool(self) -> bool:
        """At or below the resume temperature, with no gated limit, for the last ``hold_s`` seconds."""
        now = self.clock()
        recent = self.between(now - self.hold_s, now)
        with self._lock:
            covered = bool(self.samples) and self.samples[0].t <= now - self.hold_s
        return (
            covered
            and bool(recent)
            and all(s.temp_c <= self.resume_temp and not s.reasons & self.gate_mask for s in recent)
        )

    def limited(self, t0: float, t1: float) -> int:
        """OR of the gated reasons seen in [t0, t1]: non-zero taints a segment."""
        mask = 0
        for s in self.between(t0, t1):
            mask |= s.reasons & self.gate_mask
        return mask

    def first_limited(self, t0: float, t1: float) -> float | None:
        """Stamp of the first sample in [t0, t1] with a gated reason."""
        return next((s.t for s in self.between(t0, t1) if s.reasons & self.gate_mask), None)

    def describe(self, t0: float, t1: float) -> dict[str, object]:
        """Hardware state of [t0, t1] for the summary."""
        samples = self.between(t0, t1)
        if not samples:
            return {"samples": 0}
        seen = 0
        for s in samples:
            seen |= s.reasons
        clocks = [s.sm_mhz for s in samples if s.sm_mhz is not None]
        power_capped = sum(bool(s.reasons & 0x4) for s in samples)
        return {
            "samples": len(samples),
            "temp_c": [min(s.temp_c for s in samples), max(s.temp_c for s in samples)],
            "sm_mhz_median": statistics.median(clocks) if clocks else None,
            "sm_mhz_min": min(clocks) if clocks else None,
            "reasons_seen": reason_names(seen),
            "power_cap_sample_pct": round(100 * power_capped / len(samples), 1),
        }
