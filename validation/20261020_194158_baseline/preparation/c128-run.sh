#!/usr/bin/env bash
set -euo pipefail
source ${NVME_WORKDIR}/baseline/20261020_194158/preparation/env.sh
export OUT="$B1_ROOT/c128-check" CONCURRENCY=128 REPEATS=1 REPEATS_AT='' LABEL=a100-main-c128-check
export BENCH_ARGS='--num-prompts 256' EXPECTED_REQUESTS=256
[[ ! -e "$OUT/sweep.json" ]] || exit 1
mkdir -p "$OUT"
cd "$TOOL_REPO"
set +e
bash tools/drama3_bench/run_sweep.sh > "$OUT/runner.log" 2>&1
status=$?
printf '%s\n' "$status" > "$OUT/runner.exit"
exit "$status"
