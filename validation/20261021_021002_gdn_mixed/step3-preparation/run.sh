#!/usr/bin/env bash
set -euo pipefail
# Remove probe activation BEFORE Python startup/sitecustomize can execute.
for key in ${!DRAMA3_GDN_@}; do unset "$key"; done
unset PYTHONPATH VLLM_CUSTOM_SCOPES_FOR_PROFILING VLLM_NVTX_SCOPES_FOR_PROFILING
source ${NVME_WORKDIR}/venvs/vllm-omni-cu129/bin/activate
source ${PERSONAL_WORKDIR}/activate_cuda129.sh
exec python ${NVME_WORKDIR}/gdn-mixed/20261021_021002/step3-preparation/run_service_controls.py \
 --experiment-root ${NVME_WORKDIR}/gdn-mixed/20261021_021002 \
 --tools-root ${NVME_WORKDIR}/gdn-mixed/20261021_021002/tool_snapshot/drama3_bench \
 --deploy-config ${NVME_WORKDIR}/baseline/20261020_194158/deploy_config.yaml "$@"
