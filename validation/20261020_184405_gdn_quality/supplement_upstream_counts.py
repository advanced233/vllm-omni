"""CPU-only new failure definitions; preserve original ASR/SIM and quality.json."""
import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
snapshot = ROOT / "postprocess-tools-060ae50"
sys.path.insert(0, str(snapshot))
spec = importlib.util.spec_from_file_location("quality_060ae50", snapshot / "eval_quality.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
report = {}
for mode in ("default", "fp32_state"):
    run = ROOT / mode
    source = run / "quality.json"
    original_bytes = source.read_bytes()
    original = json.loads(original_bytes)
    enriched = copy.deepcopy(original)
    assert len(enriched["items"]) == 1088
    for item in enriched["items"]:
        pcm, sr, channels = module.read_pcm(run / "audio" / item["wav"])
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            x = x[:len(x) // channels * channels].reshape(-1, channels).mean(axis=1)
        item["words"] = module.words(item["target_text"])
        item["silence_run_s"] = round(module.silence_run(x, sr), 2)
    new_totals = module.totals(enriched["items"], argparse.Namespace(max_frames=500))
    for key, value in original["totals"].items():
        assert new_totals[key] == value, (mode, key, value, new_totals[key])
    enriched["totals"] = new_totals
    enriched["postprocess_provenance"] = {
        "method": "CPU-only words/silence_run/totals from 060ae50; no ASR/SIM rerun",
        "tool_snapshot": json.loads((snapshot / "provenance.json").read_text()),
        "original_quality_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "original_evaluation_commit": "4d87d0f35b77b5d14dd5b381d2fde18f5cad26ab",
        "original_totals_preserved": True,
    }
    for before, after in zip(original["items"], enriched["items"], strict=True):
        assert all(after[k] == v for k, v in before.items())
    assert source.read_bytes() == original_bytes
    (run / "quality_upstream.json").write_text(json.dumps(enriched, indent=2) + "\n")
    report[mode] = new_totals["upstream"]
(ROOT / "upstream_failure_summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
