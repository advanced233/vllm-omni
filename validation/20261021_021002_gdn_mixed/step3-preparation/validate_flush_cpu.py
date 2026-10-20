"""Pure-CPU verification of user recorder's periodic flush for every tail offset.
Does not import torch/vLLM, start a server, or alter the user probe.
"""
import importlib.util,json,tempfile
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent.parent
path=ROOT/'tool_snapshot/drama3_bench/gdn_mixed/drama3_gdn_probe.py'
spec=importlib.util.spec_from_file_location('recorder_cpu_test',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
md=SimpleNamespace(num_prefills=1,num_actual_tokens=191,num_decode_tokens=15)
checks=[]
with tempfile.TemporaryDirectory(prefix='flush-cpu-',dir=ROOT/'step3-preparation') as td:
 for offset in range(32):
  p=Path(td)/f'{offset}.jsonl';rec=m.StepRecorder(p,'baseline',True)
  # Start with offset+1 main steps, then33 post-main steps. Timestamps are labels
  # assigned only on synthetic test records; no tool or timing code is patched.
  main=offset+1
  for i in range(main+33):
   rec.enter(object(),md);rec.rec['t']=float(i)
   for layer in range(30):rec.leave(layer*10,layer*10+5,{'chunk':5})
  persisted=[json.loads(x) for x in p.read_text().splitlines()]
  assert max(x['t'] for x in persisted)>=main
  assert all(any(x['t']==i for x in persisted) for i in range(main))
  assert all(x['calls']==30 for x in persisted)
  checks.append({'initial_tail_offset':offset,'main_steps':main,'persisted_steps_before_close':len(persisted),'durable_post_main_record':True})
  # Explicit close only after the proof above, to make test teardown idempotent.
  rec.close()
assert 'torch' not in __import__('sys').modules
(ROOT/'step3-preparation/flush_cpu_validation.json').write_text(json.dumps({'ok':True,'offsets_verified':32,'drain_steps':33,'tool_path':str(path),'gpu_used':False,'checks':checks},indent=2)+'\n')
print('32 flush tail-offset cases passed; no torch import/GPU used')
