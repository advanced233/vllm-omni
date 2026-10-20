#!/usr/bin/env bash
set -uo pipefail
source ${NVME_WORKDIR}/venvs/vllm-omni-cu129/bin/activate
source ${PERSONAL_WORKDIR}/activate_cuda129.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 TMPDIR=${NVME_WORKDIR}/tmp/unit-tests MAX_JOBS=4
OUT=${NVME_WORKDIR}/gdn-quality/20261020_184405/unit_tests
cd ${WORKSPACE}/vllm-omni
nvidia-smi > "$OUT/gpu-before.txt"
git rev-parse HEAD > "$OUT/commit.txt"
git status --porcelain > "$OUT/git-before.txt"
date -Is > "$OUT/start.txt"
python -c 'import torch,vllm;print(torch.__version__,torch.version.cuda,vllm.__version__);print(torch.cuda.get_device_name(),torch.cuda.get_device_capability())' > "$OUT/versions-device.txt" 2>&1
python -m pytest -v --run-level core_model --tb=long --junitxml="$OUT/junit.xml" --durations=20 @"$OUT/selected-nodeids.txt" 2>&1 | tee "$OUT/pytest.log"
status=${PIPESTATUS[0]}
printf '%s\n' "$status" > "$OUT/exit.txt"
date -Is > "$OUT/end.txt"
nvidia-smi > "$OUT/gpu-after.txt"
git status --porcelain > "$OUT/git-after.txt"
exit "$status"
