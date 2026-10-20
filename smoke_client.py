"""Drama 3 smoke client for vLLM-Omni /v1/audio/speech (keep outside the repo).

Runs a fixed set of requests against a running server and writes every output,
plus summary.json, into --out. Request bodies mirror
tests/e2e/online_serving/test_drama_3.py.

Usage:
  python smoke_client.py --out <RUN_DIR> \
      --ref-wav <vllm-omni>/tests/assets/qwen3_tts/clone_2.wav
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

SAMPLE_RATE = 44100
MODEL = "fishaudio/drama-3"
TEXT_PROMPT = "The weather is nice today, perfect for a walk in the park."
CLONE_PROMPT = "The quarterly report is due on Thursday, so please finish it by noon."
REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. "
    "But you know what? You blew it! And thanks to you."
)
TIMEOUT_S = 600


def pcm_stats(pcm: bytes) -> dict:
    """Duration and loudness of mono int16 PCM at 44.1 kHz."""
    x = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0
    dur = len(x) / SAMPLE_RATE
    if len(x) == 0:
        return {"audio_s": 0.0, "rms_dbfs": None, "last_1s_rms_dbfs": None}

    def dbfs(v: np.ndarray) -> float:
        return round(float(20 * np.log10(np.sqrt(np.mean(v**2)) + 1e-12)), 1)

    return {
        "audio_s": round(dur, 3),
        "rms_dbfs": dbfs(x),
        "last_1s_rms_dbfs": dbfs(x[-SAMPLE_RATE:]),
    }


def wav_to_pcm(data: bytes) -> tuple[bytes, int]:
    with wave.open(io.BytesIO(data)) as w:
        return w.readframes(w.getnframes()), w.getframerate()


def save_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)


def post(url: str, body: dict, stream: bool) -> requests.Response:
    return requests.post(url, json=body, stream=stream, timeout=TIMEOUT_S)


def run_non_stream(url: str, body: dict, out: Path, name: str) -> dict:
    t0 = time.perf_counter()
    r = post(url, body, stream=False)
    elapsed = time.perf_counter() - t0
    rec = {"case": name, "status": r.status_code, "latency_s": round(elapsed, 3)}
    if r.status_code != 200:
        rec["error"] = r.text[:500]
        return rec
    (out / f"{name}.wav").write_bytes(r.content)
    pcm, sr = wav_to_pcm(r.content)
    rec["sample_rate"] = sr
    rec.update(pcm_stats(pcm))
    if rec["audio_s"]:
        rec["rtf"] = round(elapsed / rec["audio_s"], 3)
    return rec


def run_stream(url: str, body: dict, out: Path, name: str) -> dict:
    t0 = time.perf_counter()
    ttfa = None
    chunks: list[bytes] = []
    r = post(url, body, stream=True)
    rec = {"case": name, "status": r.status_code}
    if r.status_code != 200:
        rec["error"] = r.text[:500]
        return rec
    for chunk in r.iter_content(chunk_size=None):
        if chunk:
            if ttfa is None:
                ttfa = time.perf_counter() - t0
            chunks.append(chunk)
    total = time.perf_counter() - t0
    pcm = b"".join(chunks)
    save_wav(out / f"{name}.wav", pcm)
    rec.update(
        {
            "ttfa_ms": round(ttfa * 1000, 1) if ttfa is not None else None,
            "total_s": round(total, 3),
            "n_chunks": len(chunks),
        }
    )
    rec.update(pcm_stats(pcm))
    if rec["audio_s"]:
        rec["rtf"] = round(total / rec["audio_s"], 3)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8091")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-wav", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    url = f"{args.base_url}/v1/audio/speech"
    ref_url = "data:audio/wav;base64," + base64.b64encode(Path(args.ref_wav).read_bytes()).decode()

    text_body = {"model": MODEL, "input": TEXT_PROMPT, "response_format": "wav"}
    clone_body = {
        "model": MODEL,
        "input": CLONE_PROMPT,
        "ref_audio": ref_url,
        "ref_text": REF_TEXT,
        "response_format": "wav",
    }

    results = []
    # The first request also pays for any Triton kernels compiled on first use.
    results.append(run_non_stream(url, text_body, out, "00_warmup_text"))
    results.append(run_non_stream(url, text_body, out, "01_text"))
    results.append(run_non_stream(url, clone_body, out, "02_clone"))
    for seed in (0, 42):
        body = dict(clone_body, response_format="pcm", stream=True, stream_format="audio", seed=seed)
        results.append(run_stream(url, body, out, f"03_clone_stream_seed{seed}"))

    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(run_non_stream, url, clone_body, out, f"04_concurrent_{i}") for i in range(4)]
        results.extend(f.result() for f in futs)

    bad = {k: v for k, v in clone_body.items() if k != "ref_text"}
    r = post(url, bad, stream=False)
    results.append(
        {
            "case": "05_missing_ref_text",
            "status": r.status_code,
            "expected": "4xx",
            "ok": 400 <= r.status_code < 500,
        }
    )

    (out / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    for rec in results:
        print(json.dumps(rec, ensure_ascii=False))


if __name__ == "__main__":
    main()
