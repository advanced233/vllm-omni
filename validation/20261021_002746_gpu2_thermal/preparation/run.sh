#!/usr/bin/env bash
set -euo pipefail
source ${NVME_WORKDIR}/gpu2-thermal/20261021_002746/preparation/env.sh
export OUT="$B1_ROOT/sweep" CONCURRENCY='1' REPEATS=1 REPEATS_AT='' LABEL=a100-gpu2-cold-c1
export BENCH_ARGS='--num-prompts 1088' EXPECTED_REQUESTS=1088
[[ ! -e "$OUT/sweep.json" ]] || exit 1
mkdir -p "$OUT"
cd "$TOOL_REPO"
exec python "$B1_ROOT/preparation/runner.py"
