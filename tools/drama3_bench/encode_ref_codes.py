#!/usr/bin/env python3
"""Pre-encode Seed-TTS-Eval voice-clone references into Drama 3 ``ref_audio_codes``.

PR #8933 measured with every reference sent pre-encoded (``ref_audio_codes``),
so neither the baseline nor the head server encodes audio in stage 0. This
script produces those codes offline with the loader and encoder the server
itself uses for ``ref_audio``:

* the wav is decoded by vLLM's ``MediaConnector`` (as
  ``OmniOpenAIServingSpeech._resolve_ref_audio_array`` does), averaged to mono
  and held to the server's 1-30 s reference bounds;
* ``fish_speech.dac_encoder.encode_reference_audio_codes`` encodes it with the
  checkpoint's ``codec.pth`` in fp32 (what ``FishOmniForConditionalGeneration``
  calls for a ``ref_audio`` request) into ``[frames, codebooks]``, written out
  transposed as the API's ``[codebooks, frames]``.

Run it in the vLLM-Omni environment on a CUDA GPU: stage 0 encodes on its GPU,
and the CPU can differ from it in rare nearest-code ties. The GPU must have
room for the codec (a few GiB), so run it before starting the server.

Output: one JSON object per Seed-TTS row (``--out``, JSON Lines) and a
``<out>.meta.json`` with the provenance (checkpoint, codec and meta.lst
digests, versions, device, skipped rows).

Usage:
  python encode_ref_codes.py --model /models/fishaudio/drama-3 \
      --seed-tts-root /data/seed-tts-eval --locale en --out ref_codes_en.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# The server's bounds on a reference clip (serving_speech._check_ref_audio_duration).
REF_AUDIO_MIN_S = 1.0
REF_AUDIO_MAX_S = 30.0


@dataclass
class MetaRow:
    utt_id: str
    ref_text: str
    prompt_wav: str
    target_text: str


def parse_meta_lst(path: Path) -> list[MetaRow]:
    """Rows of a Seed-TTS ``meta.lst``, filtered like ``seed_tts_dataset._parse_meta_line``."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) < 4 or not parts[3].strip():
            continue
        rows.append(MetaRow(parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()))
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_codec_path(model: str) -> Path | None:
    local = Path(model).expanduser() / "codec.pth"
    if local.is_file():
        return local.resolve()
    try:
        from transformers.utils.hub import cached_file

        cached = cached_file(model, "codec.pth")
    except Exception:  # noqa: BLE001 - provenance only; the encoder reports a missing codec itself
        return None
    return Path(cached).resolve() if cached else None


class ReferenceLoader:
    """Decodes a reference wav the way the speech server decodes ``ref_audio``."""

    def __init__(self, root: Path) -> None:
        from vllm.multimodal.media import MediaConnector

        self._connector = MediaConnector(allowed_local_media_path=str(root))

    def __call__(self, path: Path) -> tuple[np.ndarray, int]:
        wav, sr = self._connector.fetch_audio(path.as_uri())
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=-1)
        return np.ascontiguousarray(wav, dtype=np.float32), int(sr)


class ReferenceEncoder:
    """``encode_reference_audio_codes`` returning the API's ``[codebooks, frames]`` grid."""

    def __init__(self, model: str, device: str) -> None:
        from vllm_omni.model_executor.models.fish_speech.dac_encoder import encode_reference_audio_codes
        from vllm_omni.model_executor.models.fish_speech.dac_utils import dac_codebook_sizes

        self._encode = encode_reference_audio_codes
        self.model = model
        self.device = device
        self.codebook_sizes = tuple(dac_codebook_sizes())

    def __call__(self, wav: np.ndarray, sr: int) -> list[list[int]]:
        codes_fq = self._encode(self.model, wav, sr, device=self.device)
        return codes_fq.transpose(0, 1).cpu().tolist()


def check_codes(codes: list[list[int]], codebook_sizes: tuple[int, ...]) -> str | None:
    """What the fish_omni adapter would reject in these codes, or None."""
    if len(codes) != len(codebook_sizes):
        return f"{len(codes)} codebooks, the model has {len(codebook_sizes)}"
    frames = len(codes[0])
    if frames == 0:
        return "no frames"
    for row, (values, size) in enumerate(zip(codes, codebook_sizes)):
        if len(values) != frames:
            return f"codebook {row} has {len(values)} frames, codebook 0 has {frames}"
        low, high = min(values), max(values)
        if low < 0 or high >= size:
            return f"codebook {row} holds values in [{low}, {high}], outside [0, {size})"
    return None


def encode_rows(
    rows: list[MetaRow],
    locale_dir: Path,
    load,
    encode,
    codebook_sizes: tuple[int, ...],
    out,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Encode each row's reference (once per wav) and write one record per row to ``out``.

    Returns the skipped rows (with reasons) and the frame count of every written row.
    """
    cache: dict[Path, tuple[list[list[int]], float, int] | str] = {}
    skipped: list[dict[str, Any]] = []
    frames: list[int] = []
    started = time.perf_counter()
    for i, row in enumerate(rows):
        wav_path = (locale_dir / row.prompt_wav).resolve()
        if wav_path not in cache:
            if not wav_path.is_file():
                cache[wav_path] = "missing prompt wav"
            else:
                wav, sr = load(wav_path)
                duration = len(wav) / sr if sr > 0 else 0.0
                if not REF_AUDIO_MIN_S <= duration <= REF_AUDIO_MAX_S:
                    cache[wav_path] = (
                        f"reference is {duration:.2f} s, the server takes {REF_AUDIO_MIN_S}-{REF_AUDIO_MAX_S} s"
                    )
                else:
                    codes = encode(wav, sr)
                    error = check_codes(codes, codebook_sizes)
                    cache[wav_path] = error if error else (codes, duration, sr)
        entry = cache[wav_path]
        if isinstance(entry, str):
            skipped.append({"utt_id": row.utt_id, "prompt_wav": row.prompt_wav, "reason": entry})
        else:
            codes, duration, sr = entry
            record = {
                "utt_id": row.utt_id,
                "ref_text": row.ref_text,
                "target_text": row.target_text,
                "prompt_wav": row.prompt_wav,
                "ref_audio_s": round(duration, 4),
                "ref_sample_rate": sr,
                "frames": len(codes[0]),
                "ref_audio_codes": codes,
            }
            out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            frames.append(len(codes[0]))
        if (i + 1) % 100 == 0 or i + 1 == len(rows):
            rate = (i + 1) / max(time.perf_counter() - started, 1e-9)
            print(f"[{i + 1}/{len(rows)}] {len(skipped)} skipped, {rate:.1f} rows/s", flush=True)
    return skipped, frames


def package_versions() -> dict[str, str | None]:
    from importlib import metadata

    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in ("torch", "torchaudio", "vllm", "vllm-omni", "librosa", "soundfile"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def vllm_omni_commit() -> str | None:
    try:
        import vllm_omni
    except ImportError:
        return None
    source = Path(vllm_omni.__file__).resolve().parent
    try:
        result = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def device_name(device: str) -> str:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.device(device))
    return device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="Served checkpoint (local dir or HF id) holding codec.pth")
    parser.add_argument("--seed-tts-root", required=True, help="Directory holding <locale>/meta.lst and prompt wavs")
    parser.add_argument("--locale", default="en", choices=["en", "zh"])
    parser.add_argument("--out", required=True, help="Output JSON Lines file")
    parser.add_argument("--device", default="cuda", help="Encoder device (default cuda, as stage 0)")
    parser.add_argument("--limit", type=int, default=None, help="Encode only the first N rows (quick checks)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.seed_tts_root).expanduser().resolve()
    locale_dir = root / args.locale
    meta_lst = locale_dir / "meta.lst"
    if not meta_lst.is_file():
        parser.error(f"{meta_lst} not found (expected the seed-tts-eval layout)")
    out_path = Path(args.out).expanduser().resolve()
    meta_path = out_path.with_name(out_path.name + ".meta.json")
    if out_path.exists() and not args.overwrite:
        parser.error(f"{out_path} exists; pass --overwrite to replace it")

    rows = parse_meta_lst(meta_lst)
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"{len(rows)} rows from {meta_lst}", flush=True)

    load = ReferenceLoader(root)
    encode = ReferenceEncoder(args.model, args.device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    started = time.time()
    with tmp_path.open("w", encoding="utf-8") as out:
        skipped, frames = encode_rows(rows, locale_dir, load, encode, encode.codebook_sizes, out)
    os.replace(tmp_path, out_path)

    codec_path = resolve_codec_path(args.model)
    meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        "encode_seconds": round(time.time() - started, 1),
        "model": args.model,
        "codec_path": str(codec_path) if codec_path else None,
        "codec_sha256": sha256_file(codec_path) if codec_path else None,
        "seed_tts_root": str(root),
        "locale": args.locale,
        "meta_lst_sha256": sha256_file(meta_lst),
        "rows_in_meta": len(parse_meta_lst(meta_lst)),
        "rows_requested": len(rows),
        "rows_written": len(frames),
        "rows_skipped": skipped,
        "frames": {
            "total": int(sum(frames)),
            "min": int(min(frames)) if frames else None,
            "mean": round(float(np.mean(frames)), 1) if frames else None,
            "max": int(max(frames)) if frames else None,
        },
        "layout": "ref_audio_codes is [codebooks, frames]; codebook 0 is the semantic one",
        "codebook_sizes": list(encode.codebook_sizes),
        "encoder": "vllm_omni.model_executor.models.fish_speech.dac_encoder.encode_reference_audio_codes (fp32)",
        "loader": "vllm.multimodal.media.MediaConnector.fetch_audio, mean over channels",
        "device": args.device,
        "device_name": device_name(args.device),
        "vllm_omni_commit": vllm_omni_commit(),
        "versions": package_versions(),
        "argv": sys.argv,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(frames)} rows to {out_path} ({len(skipped)} skipped); provenance in {meta_path}")
    return 0 if frames else 1


if __name__ == "__main__":
    sys.exit(main())
