#!/usr/bin/env bash
# Prepared only; launch after root agent reviews all 17 B1 runs.
set -euo pipefail
source ${NVME_WORKDIR}/baseline/20261020_194158/preparation/env.sh
export OUT="$B1_ROOT/followup" CONCURRENCY='10 12 16' REPEATS=1 REPEATS_AT='' LABEL=a100-main-followup
export BENCH_ARGS='--num-prompts 1088' EXPECTED_REQUESTS=1088
[[ ! -e "$OUT/sweep.json" ]] || exit 1
[[ -e "$B1_ROOT/ALLOW_FOLLOWUP" ]] || exit 1
mkdir -p "$OUT"
cd "$TOOL_REPO"
set +e
bash tools/drama3_bench/run_sweep.sh > "$OUT/runner.log" 2>&1
status=$?
printf '%s\n' "$status" > "$OUT/runner.exit"
exit "$status"
