# User-owned environment for the paired GDN-state quality experiment.
source ${NVME_WORKDIR}/venvs/vllm-omni-cu129/bin/activate
source ${PERSONAL_WORKDIR}/activate_cuda129.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
export EXPERIMENT_ROOT=${NVME_WORKDIR}/gdn-quality/20261020_184405
export NVME_WORKDIR=${NVME_WORKDIR}
export TOOL_REPO=${PERSONAL_WORKDIR}/github-smoke-audio-20261020
export OMNI_REPO=${WORKSPACE}/vllm-omni
export SEED_TTS_ROOT="$NVME_WORKDIR/datasets/seed-tts-eval"
export CODES="$NVME_WORKDIR/bench-validation/20261020_165001_49599d4/ref_codes_en.jsonl"
export TMPDIR="$NVME_WORKDIR/tmp/gdn-184405"
export HF_HOME=${WORKSPACE}/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub" HF_XET_CACHE="$HF_HOME/xet"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MAX_JOBS=4 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
mkdir -p "$TMPDIR"
