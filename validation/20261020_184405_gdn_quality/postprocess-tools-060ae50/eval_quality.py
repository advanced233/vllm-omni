#!/usr/bin/env python3
"""Seed-TTS WER and speaker similarity of replies saved by ``stream_bench.py --save-audio``.

Scores each saved WAV the way ``vllm bench serve --seed-tts-wer-eval`` scores a
reply, by calling the same helpers of
``vllm_omni.benchmarks.data_modules.seed_tts_eval``:

* WER: the reply resampled to 16 kHz (``_pcm_s16le_to_f32_16k``), transcribed
  by ``openai/whisper-large-v3`` in fp32 (English) or Paraformer (Chinese), and
  scored with seed-tts-eval's normalization (``process_one_official``); the run's
  WER is the mean over utterances, as in seed-tts-eval's ``run_wer.py``;
* SIM: cosine of mean-pooled WavLM embeddings of the reference prompt and the
  reply. This is the repository's proxy (``microsoft/wavlm-base-plus`` unless
  ``--wavlm-model``), not seed-tts-eval's fine-tuned WavLM-large speaker
  verification model, so it is comparable between runs here, not with
  published SIM numbers.

Every reply is also checked for a stuck take (``analyze_audio.noise_run``), so a
run reports its WER with and without the takes that stopped speaking. An empty
transcript scores WER 1 (every word deleted); ``wer_mean_nonempty`` leaves them
out, as the repository's evaluator does, for comparison with ``vllm bench serve``.

Writes ``quality.json`` (totals and every utterance) into each run directory.
Run it on a GPU the server does not use, or after the server has stopped.

Usage:
  python eval_quality.py results/quality_c16 --seed-tts-root /data/seed-tts-eval
  python eval_quality.py results/default results/fp32_state --seed-tts-root /data/seed-tts-eval --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_audio import noise_run, words  # noqa: E402

PROTOCOL_HELPERS = (
    "_missing_deps_message",
    "_pcm_s16le_to_f32_16k",
    "_audio_path_to_f32_16k",
    "_transcribe_en_f32_16k",
    "_transcribe_zh_wav_path",
    "process_one_official",
    "_wavlm_mean_embedding_f32_16k",
    "_cosine_similarity_unit_vectors",
    "OFFICIAL_WHISPER_HF_ID",
)


def load_protocol() -> Any:
    """The repository's Seed-TTS evaluator, checked for the helpers this script calls."""
    from vllm_omni.benchmarks.data_modules import seed_tts_eval

    missing = [name for name in PROTOCOL_HELPERS if not hasattr(seed_tts_eval, name)]
    if missing:
        raise SystemExit(f"{seed_tts_eval.__file__} no longer has {missing}; update eval_quality.py to match")
    return seed_tts_eval


def read_pcm(path: Path) -> tuple[bytes, int, int]:
    with wave.open(str(path)) as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
        return w.readframes(w.getnframes()), w.getframerate(), w.getnchannels()


def transcribe(ev: Any, wav_16k: np.ndarray, lang: str) -> str:
    if lang == "en":
        return ev._transcribe_en_f32_16k(wav_16k)
    import soundfile as sf

    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(tmp, wav_16k, 16000, subtype="PCM_16")
        return ev._transcribe_zh_wav_path(tmp)
    finally:
        os.unlink(tmp)


def evaluate_run(run: Path, ev: Any, args: argparse.Namespace, ref_cache: dict[str, Any]) -> dict[str, Any]:
    audio = run / "audio"
    manifest = (audio / "manifest.jsonl").read_text(encoding="utf-8")
    entries = [json.loads(line) for line in manifest.splitlines() if line.strip()]
    if args.limit:
        entries = entries[: args.limit]
    locale_dir = Path(args.seed_tts_root) / args.locale if args.seed_tts_root else None
    items: list[dict[str, Any]] = []
    started = time.perf_counter()
    for n, entry in enumerate(entries, 1):
        item: dict[str, Any] = {"utt_id": entry["utt_id"], "wav": entry["wav"], "target_text": entry["target_text"]}
        items.append(item)
        if not entry.get("ok", True):
            item["error"] = "request_failed"
            continue
        pcm, sr, channels = read_pcm(audio / entry["wav"])
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            x = x[: len(x) // channels * channels].reshape(-1, channels).mean(axis=1)
        item["audio_s"] = round(len(x) / sr, 3)
        item["frames"] = round(len(x) / args.hop)
        item["noise_run_s"] = round(noise_run(x, sr, args.flat_db)[0], 2)
        item["stuck"] = item["noise_run_s"] >= args.noise_run
        item["words"] = words(entry["target_text"])
        item["silence_run_s"] = round(silence_run(x, sr), 2)
        wav_16k = ev._pcm_s16le_to_f32_16k(pcm, pcm_sample_rate=sr, channels=channels)
        if len(wav_16k) == 0:
            item["error"] = "empty_audio"
            continue
        try:
            hyp = transcribe(ev, wav_16k, args.locale)
        except Exception as exc:  # noqa: BLE001 - one bad utterance must not end a 1088-row evaluation
            item["error"] = f"asr_exception: {type(exc).__name__}: {exc}"[:300]
            continue
        item["asr"] = hyp
        if hyp:
            item["wer"] = round(ev.process_one_official(hyp, entry["target_text"], args.locale)[0], 4)
        else:
            # Every word deleted. The repository's evaluator drops these instead (wer_mean_nonempty).
            item["wer"] = 1.0
            item["asr_empty"] = True
        if not args.no_sim and locale_dir is not None:
            ref_path = locale_dir / entry["prompt_wav"]
            if entry["prompt_wav"] not in ref_cache:
                ref_cache[entry["prompt_wav"]] = (
                    ev._wavlm_mean_embedding_f32_16k(ev._audio_path_to_f32_16k(str(ref_path)))
                    if ref_path.is_file()
                    else None
                )
            e_ref = ref_cache[entry["prompt_wav"]]
            e_hyp = ev._wavlm_mean_embedding_f32_16k(wav_16k) if e_ref is not None else None
            if e_hyp is not None:
                item["sim"] = round(ev._cosine_similarity_unit_vectors(e_ref, e_hyp), 4)
            else:
                item["sim_skipped"] = "no reference wav" if e_ref is None else "empty embedding"
        if n % 50 == 0 or n == len(entries):
            rate = n / max(time.perf_counter() - started, 1e-9)
            print(f"{run.name}: [{n}/{len(entries)}] {rate:.2f} utt/s", flush=True)
    return {"run": str(run), "totals": totals(items, args), "items": items}


def silence_run(x: np.ndarray, sr: int, window_s: float = 0.05, rms: float = 1e-3) -> float:
    """Longest run of 50 ms windows with RMS under 1e-3 (about -60 dBFS): #9044's silence."""
    hop = max(1, int(sr * window_s))
    n = len(x) // hop
    if n == 0:
        return 0.0
    quiet = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1)) < rms
    best = run = 0
    for q in quiet:
        run = run + 1 if q else 0
        best = max(best, run)
    return best * window_s


def upstream_failures(items: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """The run's failures counted the way #8933, #9157 and #9044 count them, for comparison."""
    timed = [i for i in items if "audio_s" in i and i.get("words")]
    n = len(timed)
    # #9157: the median seconds per word of the run predicts each line's duration.
    spw = statistics.median(i["audio_s"] / i["words"] for i in timed) if timed else None
    issue_9157 = []
    for i in timed:
        ratio = i["audio_s"] / (i["words"] * spw)
        i["duration_ratio"] = round(ratio, 2)
        if i["frames"] >= args.max_frames or i.get("wer", 0.0) > 0.5 or not 0.3 <= ratio <= 3.0:
            issue_9157.append(i)

    def share(count: int) -> dict[str, Any]:
        return {"count": count, "pct": round(100 * count / n, 2) if n else None}

    return {
        "takes": n,
        "pr8933_at_least_20s": share(sum(i["audio_s"] >= 20.0 for i in timed)),
        "issue9157_failed": share(len(issue_9157)),
        "pr9044_silence_over_1s": share(sum(i["silence_run_s"] > 1.0 for i in timed)),
        "definitions": {
            "pr8933_at_least_20s": "generated audio of 20 s or more (#8933 description)",
            "issue9157_failed": "ran to the frame cap, or WER over 50%, or duration outside 0.3-3x the "
            "run's median seconds per word times the line's words (#9157, Seed-TTS rule)",
            "pr9044_silence_over_1s": "longest run of 50 ms windows with RMS under 1e-3 over 1 s (#9044)",
        },
    }


def mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def totals(items: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    scored = [i for i in items if "wer" in i]
    stuck = [i for i in items if i.get("stuck")]
    errors: dict[str, int] = {}
    for i in items:
        if "error" in i:
            key = i["error"].split(":")[0]
            errors[key] = errors.get(key, 0) + 1
    wers = [i["wer"] for i in scored]
    sims = [i["sim"] for i in scored if "sim" in i]
    return {
        "utterances": len(items),
        "scored": len(scored),
        "errors": errors,
        "wer_mean": mean(wers),
        "wer_median": round(statistics.median(wers), 4) if wers else None,
        "wer_mean_nonempty": mean([i["wer"] for i in scored if not i.get("asr_empty")]),
        "asr_empty": sum(bool(i.get("asr_empty")) for i in scored),
        "wer_mean_without_stuck": mean([i["wer"] for i in scored if not i["stuck"]]),
        "wer_over_0.5": sum(w > 0.5 for w in wers),
        "wer_zero_pct": round(100 * sum(w == 0 for w in wers) / len(wers), 2) if wers else None,
        "sim_mean": mean(sims),
        "sim_median": round(statistics.median(sims), 4) if sims else None,
        "sim_scored": len(sims),
        "sim_mean_without_stuck": mean([i["sim"] for i in scored if "sim" in i and not i["stuck"]]),
        "stuck": len(stuck),
        "stuck_pct": round(100 * len(stuck) / len(items), 2) if items else None,
        "stuck_scored_wer_mean": mean([i["wer"] for i in stuck if "wer" in i]),
        "at_frame_cap": sum(i.get("frames", 0) >= args.max_frames for i in items),
        "upstream": upstream_failures(items, args),
    }


def provenance(ev: Any, args: argparse.Namespace) -> dict[str, Any]:
    from importlib import metadata

    versions = {}
    for name in ("vllm-omni", "torch", "transformers", "jiwer", "scipy"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "evaluator": ev.__file__,
        "asr_model": os.environ.get("SEED_TTS_HF_WHISPER_MODEL") or ev.OFFICIAL_WHISPER_HF_ID,
        "sim_model": None if args.no_sim else os.environ.get("SEED_TTS_WAVLM_MODEL", "microsoft/wavlm-base-plus"),
        "sim_note": "mean-pooled WavLM cosine (repository proxy), not seed-tts-eval's WavLM-large SV model",
        "device": os.environ.get("SEED_TTS_EVAL_DEVICE") or "evaluator default (cuda:0 if available)",
        "locale": args.locale,
        "stuck_rule": f"noise run >= {args.noise_run} s of 1 s windows spanning < {args.flat_db} dB",
        "versions": versions,
        "argv": sys.argv,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("runs", nargs="+", help="stream_bench.py --out directories run with --save-audio")
    p.add_argument("--seed-tts-root", default=None, help="Seed-TTS root holding the reference wavs (for SIM)")
    p.add_argument("--locale", default="en", choices=["en", "zh"])
    p.add_argument("--no-sim", action="store_true", help="Skip speaker similarity")
    p.add_argument("--device", default=None, help="Device for ASR and SIM (default: cuda:0 if available)")
    p.add_argument("--whisper-model", default=None, help="Override the protocol's whisper-large-v3 (debug only)")
    p.add_argument("--wavlm-model", default=None, help="HF id of the SIM embedding model")
    p.add_argument("--limit", type=int, default=None, help="Score only the first N utterances per run")
    p.add_argument("--worst", type=int, default=10, help="Print this many highest-WER utterances per run")
    p.add_argument("--flat-db", type=float, default=10.0)
    p.add_argument("--noise-run", type=float, default=2.0)
    p.add_argument("--max-frames", type=int, default=500)
    p.add_argument("--hop", type=int, default=2048)
    args = p.parse_args()
    if not args.no_sim and not args.seed_tts_root:
        p.error("SIM needs --seed-tts-root (or pass --no-sim)")

    # The evaluator reads these when it first loads its models.
    if args.device:
        os.environ["SEED_TTS_EVAL_DEVICE"] = args.device
        os.environ.setdefault("SEED_TTS_SIM_DEVICE", args.device)
    if args.whisper_model:
        os.environ["SEED_TTS_HF_WHISPER_MODEL"] = args.whisper_model
    if args.wavlm_model:
        os.environ["SEED_TTS_WAVLM_MODEL"] = args.wavlm_model
    ev = load_protocol()
    deps = ev._missing_deps_message(args.locale)
    if deps:
        raise SystemExit(deps)

    meta = provenance(ev, args)
    ref_cache: dict[str, Any] = {}
    results = []
    for run in args.runs:
        result = evaluate_run(Path(run), ev, args, ref_cache)
        result["provenance"] = meta
        out = Path(run) / "quality.json"
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        results.append(result)

    print(
        "| run | scored | WER mean | WER w/o stuck | WER > 0.5 | empty ASR | SIM mean | stuck | at cap | errors |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    )
    for r in results:
        t = r["totals"]
        print(
            f"| {Path(r['run']).name} | {t['scored']}/{t['utterances']} | {t['wer_mean']} "
            f"| {t['wer_mean_without_stuck']} | {t['wer_over_0.5']} | {t['asr_empty']} | {t['sim_mean']} "
            f"| {t['stuck']} ({t['stuck_pct']}%) | {t['at_frame_cap']} | {t['errors'] or '-'} |"
        )
    print(
        "\n| run | takes | >= 20 s (#8933) | #9157 failed | silence > 1 s at -60 dBFS (#9044) |\n"
        "| --- | ---: | ---: | ---: | ---: |"
    )
    for r in results:
        u = r["totals"]["upstream"]
        cells = [f"{u[k]['count']} ({u[k]['pct']}%)" for k in ("pr8933_at_least_20s", "issue9157_failed")]
        cells.append(f"{u['pr9044_silence_over_1s']['count']} ({u['pr9044_silence_over_1s']['pct']}%)")
        print(f"| {Path(r['run']).name} | {u['takes']} | " + " | ".join(cells) + " |")
    for r in results:
        worst = sorted((i for i in r["items"] if "wer" in i), key=lambda i: -i["wer"])[: args.worst]
        if worst:
            print(f"\n{Path(r['run']).name}: highest WER")
            for i in worst:
                tag = " [stuck]" if i["stuck"] else ""
                print(f"  {i['wer']:.3f}{tag}  {i['wav']}  asr={i['asr']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
