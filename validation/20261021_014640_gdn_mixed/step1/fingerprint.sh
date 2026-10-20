#!/usr/bin/env bash
set -uo pipefail
source ${NVME_WORKDIR}/gdn-mixed/20261021_014640/env.sh
cd ${WORKSPACE}/vllm-omni
python -m pytest -v tools/drama3_bench/gdn_mixed/test_mixed_core_variants.py::test_copy_is_the_installed_version --junitxml="$GDN_ROOT/step1/fingerprint.junit.xml" > "$GDN_ROOT/step1/fingerprint.log" 2>&1
status=$?
printf '%s\n' "$status" > "$GDN_ROOT/step1/fingerprint.exit"
exit "$status"
