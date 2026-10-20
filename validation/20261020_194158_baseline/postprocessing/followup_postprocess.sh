#!/usr/bin/env bash
# CPU-only, after all 17 B1 and all 3 follow-up runs complete.
set -euo pipefail
experiment_root=${NVME_WORKDIR}/baseline/20261020_194158
audit_tools=${PERSONAL_WORKDIR}/baseline-audit-20261020
python_bin=${NVME_WORKDIR}/venvs/vllm-omni-cu129/bin/python
"$python_bin" "$audit_tools/audit_sweep.py" "$experiment_root/followup" \
  --levels '10 12 16' --repeats-at '' --num-prompts 1088 \
  --label a100-main-followup --out "$experiment_root/followup-independent-audit.json"
"$python_bin" "$audit_tools/summarize_hardware.py" "$experiment_root/followup" \
  --out "$experiment_root/followup-hardware-summary.json"
"$python_bin" "$experiment_root/postprocessing/compare_followup.py" \
  --root "$experiment_root" --out "$experiment_root/postprocessing/followup-comparison"
