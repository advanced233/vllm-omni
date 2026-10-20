#!/usr/bin/env bash
set -uo pipefail
source ${NVME_WORKDIR}/venvs/vllm-omni-cu129/bin/activate
source ${PERSONAL_WORKDIR}/activate_cuda129.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 TMPDIR=${NVME_WORKDIR}/tmp/unit-tests MAX_JOBS=4
OUT=${NVME_WORKDIR}/gdn-quality/20261020_184405/unit_tests
export PYTHONPATH="$OUT${PYTHONPATH:+:$PYTHONPATH}"
cd ${WORKSPACE}/vllm-omni
python -m pytest -v -s --run-level core_model --tb=long -p actual_triton_plugin --junitxml="$OUT/supplemental-junit.xml" --durations=10 @"$OUT/supplemental-nodeids.txt" > "$OUT/supplemental-pytest.log" 2>&1
status=$?
printf '%s\n' "$status" > "$OUT/supplemental-exit.txt"
nvidia-smi > "$OUT/gpu-final.txt"
git status --porcelain > "$OUT/git-final.txt"
exit "$status"
