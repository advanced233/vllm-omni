"""Name-based descriptive groups for independently attributed kernel tables.

Norm is explicitly separated from GEMM (gemma is not gemm). Raw names and
duration-sum denominator are retained; groups do not establish causal ownership.
"""
import argparse
import collections
import json
from pathlib import Path
import re


def family(name):
    if "marlin_moe_wna16::Marlin" in name:
        return "Marlin_MoE_GEMM"
    if "vllm::moe::" in name:
        return "MoE_routing_alignment_sum"
    if re.search(r"gated_delta|causal_conv|chunk_fwd_kernel_o|merge_16x16_to_64x64_inverse|recompute_w_u|chunk_scaled_dot_kkt|chunk_local_cumsum|fused_post_conv", name):
        return "GDN_and_prefill_helpers_heuristic"
    if "flash::" in name or "fused_qk_rmsnorm_rope_gate" in name or "reshape_and_cache_flash" in name:
        return "attention_and_QK_KV_helpers"
    if name == "_draw_kernel" or "gatherTopK" in name or "warpMergeSortKV" in name:
        return "sampling_topk_draw"
    if re.search(r"gemma_norm|table_norm|mix_norm|rms_norm|rmsnorm|rsqrt", name):
        return "normalization_and_norm_fusions"
    if re.search(r"splitKreduce|reduce_1Block_kernel|cublasDot", name):
        return "BLAS_splitK_and_GEMV_reductions"
    if re.search(r"gemm(?!a)|gemv", name, re.I):
        return "dense_GEMM_GEMV_heuristic"
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("independent_trace", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    r = json.loads(args.independent_trace.read_text())
    phases = {}
    for phase in ("decode_only", "mixed", "prefill_only"):
        total = r["phase_summary"][phase]["kernel_sum_us"]
        groups = collections.defaultdict(lambda: {"kernel_sum_us": 0., "count": 0, "raw_names": []})
        for kernel in r["raw_kernel_tables_complete_interior_only"][phase]:
            g = groups[family(kernel["name"])]
            g["kernel_sum_us"] += kernel["kernel_sum_us"]
            g["count"] += kernel["count"]
            g["raw_names"].append(kernel["name"])
        phases[phase] = {
            "step_count": r["phase_summary"][phase]["conservative_complete_interior_steps"],
            "generation_batch_histogram": r["phase_summary"][phase]["generation_request_histogram"],
            "families": {name: dict(values, kernel_time_share_pct=100*values["kernel_sum_us"]/total if total else None)
                         for name, values in sorted(groups.items(), key=lambda kv: -kv[1]["kernel_sum_us"])},
        }
    result = {
        "classification_version": 2,
        "source": str(args.independent_trace),
        "denominator": "Sum of correlated kernel durations in complete interior steps, not walltime.",
        "heuristics": "Dense GEMM names cannot attribute backbone versus codebook tower. Normalization and BLAS reduction groups are separate; raw names retained.",
        "change_from_v1": "Explicitly exclude Gemma norm from GEMM matching; additionally separate named normalization fusions and BLAS reductions.",
        "phases": phases,
    }
    args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({p:{k:round(v["kernel_time_share_pct"],3) for k,v in data["families"].items()} for p,data in phases.items()}))


if __name__ == "__main__":
    main()
