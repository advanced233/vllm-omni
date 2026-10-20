#!/usr/bin/env bash
set -euo pipefail
source ${NVME_WORKDIR}/gdn-quality/20261020_184405/env.sh
exec python "$EXPERIMENT_ROOT/run_one.py" "$@"
