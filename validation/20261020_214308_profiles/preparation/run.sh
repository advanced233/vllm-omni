#!/usr/bin/env bash
set -euo pipefail
source ${NVME_WORKDIR}/profiling/20261020_214308/preparation/env.sh
exec python "$PROFILE_ROOT/preparation/run_profiles.py" "$@"
