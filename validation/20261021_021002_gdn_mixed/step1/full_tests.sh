#!/usr/bin/env bash
set -uo pipefail
source ${NVME_WORKDIR}/gdn-mixed/20261021_021002/env.sh
cd ${WORKSPACE}/vllm-omni
python -m pytest -v tools/drama3_bench/gdn_mixed/test_mixed_core_variants.py --junitxml="$GDN_ROOT/step1/full_tests.junit.xml" > "$GDN_ROOT/step1/full_tests.log" 2>&1
status=$?
printf '%s\n' "$status" > "$GDN_ROOT/step1/full_tests.exit"
exit "$status"
