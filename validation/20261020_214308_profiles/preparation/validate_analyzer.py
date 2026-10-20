"""CPU-only known-trace test: phase attribution, async lifetime, and overlap."""
import importlib.util
import json
from pathlib import Path
import tempfile
ROOT=Path(__file__).parent
spec=importlib.util.spec_from_file_location('trace_analyzer',ROOT/'analyze_trace.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
E=[]
for i,(phase,ts) in enumerate([('execute_context_0(0)_generation_8(8)',0),('execute_context_2(30)_generation_6(6)',100),('execute_context_8(80)_generation_0(0)',200)]):
    E.append(dict(name=phase,cat='user_annotation',ph='X',pid=1,tid=2,ts=ts,dur=10))
    E.append(dict(name='cudaGraphLaunch',cat='cuda_runtime',ph='X',pid=1,tid=2,ts=ts+2,dur=1,args={'correlation':i}))
    # Actual GPU execution outside the CPU scope still belongs to its launch step.
    E.append(dict(name='marlin_kernel',cat='kernel',ph='X',pid=0,tid=7,ts=ts+20,dur=20,args={'correlation':i}))
# A CUDA graph launch correlates to multiple kernels; retain every graph node.
E.append(dict(name='graph_node_2',cat='kernel',ph='X',pid=0,tid=7,ts=42,dur=5,args={'correlation':0}))
E.append(dict(name='graph_node_3',cat='kernel',ph='X',pid=0,tid=7,ts=48,dur=7,args={'correlation':0}))
E.append(dict(name='outside',cat='kernel',ph='X',pid=0,tid=7,ts=300,dur=10,args={'correlation':99}))
with tempfile.TemporaryDirectory(prefix='trace-analysis-',dir=ROOT) as tmp:
    p=Path(tmp)/'trace.json'
    p.write_text(json.dumps({'baseTimeNanoseconds':1000000000,'traceEvents':E}))
    r,busy=m.analyze(p)
    assert r['steps_total']==3 and r['kernel_events']==6
    for phase in ('decode_only','mixed','prefill_only'):
        assert r['phase_summary'][phase]['steps']==1
        assert r['phase_summary'][phase]['gpu_kernel_sum_us']==(32 if phase=='decode_only' else 20)
    assert r['steps'][0]['kernel_count']==3
    assert sum(k['count'] for k in r['kernel_table'] if k['phase']=='decode_only')==3
    assert r['unattributed_kernel_reasons']=={'no_runtime_correlation':1}
    assert m.union([(0,3),(1,4),(7,9)])==[[0,4],[7,9]]
    assert m.overlap([[0,4],[7,9]],[[2,8]])==3
(ROOT/'cpu_validation.json').write_text(json.dumps({'ok':True,'checks':['native phase parsing','CUPTI correlation even when kernel outlives CPU scope','one CUDA graph launch maps to all three correlated node kernels','unattributed work preserved','interval union','cross-stage interval overlap']},indent=2)+'\n')
print('CPU analyzer validation passed')
