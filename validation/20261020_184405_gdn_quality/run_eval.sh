#!/usr/bin/env bash
set -euo pipefail
source ${NVME_WORKDIR}/gdn-quality/20261020_184405/env.sh
if [[ ! -s "$EXPERIMENT_ROOT/eval-models/READY.json" ]]; then
    echo 'Pinned evaluation model download/verification is not marked READY.' >&2
    exit 1
fi
source "$EXPERIMENT_ROOT/eval-models/activate_eval_cache.sh"
python - <<'PY'
import importlib.util,json,os
from pathlib import Path
root=Path(os.environ['EXPERIMENT_ROOT'])
s=importlib.util.spec_from_file_location('run_one',root/'run_one.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
assert not m.gpu_apps(), 'GPU has compute process'
assert int(m.gpu().split(',')[3].strip()) <100, 'GPU not idle'
assert m.port_free(), 'Service port still open'
for mode in ('default','fp32_state'):
    run=root/mode
    summary=json.loads((run/'summary.json').read_text())
    assert summary['requests']==1088, (mode,summary['requests'])
    assert not (run/'quality.json').exists(), 'Preserve existing quality result'
    print(mode, 'generation requests/succeeded/failed:',summary['requests'],summary['succeeded'],summary['failed'])
PY
set +e
python "$TOOL_REPO/tools/drama3_bench/eval_quality.py" \
    "$EXPERIMENT_ROOT/default" "$EXPERIMENT_ROOT/fp32_state" \
    --seed-tts-root "$SEED_TTS_ROOT" --device cuda:0 \
    > "$EXPERIMENT_ROOT/eval_quality.log" 2>&1
rc=$?
set -e
printf '%s\n' "$rc" > "$EXPERIMENT_ROOT/eval_quality.exit"
python - <<'PY'
import json,os,subprocess
from pathlib import Path
root=Path(os.environ['EXPERIMENT_ROOT']); report={}
for mode in ('default','fp32_state'):
    path=root/mode/'quality.json'
    if not path.exists():
        report[mode]={'complete':False,'error':'quality.json missing'}
        continue
    t=json.loads(path.read_text())['totals']
    report[mode]={'complete':t['utterances']==1088 and t['scored']==1088 and t['sim_scored']==1088 and not t['errors'],'totals':t}
report['gpu_after']=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
(root/'quality_completeness.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
if not all(report[m]['complete'] for m in ('default','fp32_state')):
    raise SystemExit(2)
PY
exit "$rc"
