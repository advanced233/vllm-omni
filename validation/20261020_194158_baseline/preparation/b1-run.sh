#!/usr/bin/env bash
set -euo pipefail
source ${NVME_WORKDIR}/baseline/20261020_194158/preparation/env.sh
export OUT="$B1_ROOT/b1" CONCURRENCY='1 4 8 12 16 24 32 48 64 96 128' REPEATS=1 REPEATS_AT='64:3 128:5' LABEL=a100-main
export BENCH_ARGS='--num-prompts 1088' EXPECTED_REQUESTS=1088
[[ ! -e "$OUT/sweep.json" ]] || exit 1
[[ -e "$B1_ROOT/c128-check/validation.json" ]] || exit 1
python -c 'import json,os; assert json.load(open(os.environ["B1_ROOT"]+"/c128-check/validation.json"))["valid"]'
mkdir -p "$OUT"
cd "$TOOL_REPO"
set +e
bash tools/drama3_bench/run_sweep.sh > "$OUT/runner.log" 2>&1
status=$?
printf '%s\n' "$status" > "$OUT/runner.exit"
exit "$status"
