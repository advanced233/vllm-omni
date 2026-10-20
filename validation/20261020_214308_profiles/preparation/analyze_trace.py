"""Summarize native scheduler scopes and correlated CUDA activity, never guess phase.
CUDA kernels are assigned only through their CUPTI correlation -> CPU launch ->
containing native execute scope. Unattributed work stays explicitly separate.
Kernel categories are naming heuristics; raw names/count/time are always retained.
"""
import argparse
import bisect
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import re

STEP = re.compile(r'^execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)$')

def union(intervals):
    merged = []
    for a,b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a,b])
    return merged

def overlap(a,b):
    i=j=0
    total=0.0
    while i<len(a) and j<len(b):
        total += max(0, min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1] <= b[j][1]: i += 1
        else: j += 1
    return total

def quantile(vals, q):
    if not vals: return None
    vals=sorted(vals)
    pos=(len(vals)-1)*q
    lo=int(pos)
    return vals[lo] + (vals[min(lo+1,len(vals)-1)]-vals[lo])*(pos-lo)

def category(name):
    low=name.lower()
    for cat,pattern in [('marlin','marlin'),('gdn_or_causal_conv',r'gated_delta|gdn|causal_conv|fused_recurrent|chunk_gated|chunk_fwd|solve_tril|wy_fast'),('attention',r'flash.*attn|attention|fmha'),('sampling',r'sampl|top_k|topk|top_p|topp|multinomial|gumbel|draw'),('convolution',r'conv|cudnn'),('gemm',r'gemm|sgemm|hgemm|cutlass|matmul'),('elementwise_or_reduction',r'elementwise|reduce|reduction|triton_poi|triton_red')]:
        if re.search(pattern,low): return cat
    return 'other'

def analyze(path):
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:
        trace=json.load(f)
    events=trace['traceEvents']
    steps=[]
    for e in events:
        m=STEP.match(e.get('name',''))
        if m and e.get('ph')=='X' and e.get('cat')=='user_annotation':
            ctx,ct,gen,gt=map(int,m.groups())
            phase='mixed' if ctx and gen else 'prefill_only' if ctx else 'decode_only' if gen else 'empty'
            steps.append({'ts':e['ts'],'end':e['ts']+e.get('dur',0),'duration_us':e.get('dur',0),'phase':phase,'context_requests':ctx,'context_tokens':ct,'generation_requests':gen,'generation_tokens':gt,'pid':e['pid'],'tid':e['tid'],'kernel_count':0,'kernel_us':0.0,'kernels':[]})
    steps.sort(key=lambda x:x['ts'])
    bythread=defaultdict(list)
    for i,s in enumerate(steps): bythread[(s['pid'],s['tid'])].append((s['ts'],i))
    thread_starts={k:[t for t,i in pairs] for k,pairs in bythread.items()}
    def contains(e):
        key=(e.get('pid'),e.get('tid'))
        if key not in bythread: return None
        pos=bisect.bisect_right(thread_starts[key],e['ts'])-1
        if pos<0:return None
        idx=bythread[key][pos][1]
        return idx if e['ts']<=steps[idx]['end'] else None
    runtime={}
    graph_launches=[]
    for e in events:
        if e.get('cat') in ('cuda_runtime','cuda_driver') and e.get('ph')=='X':
            corr=e.get('args',{}).get('correlation')
            if corr is not None: runtime[corr]=(contains(e),e)
            if 'GraphLaunch' in e.get('name',''):graph_launches.append(e)
    totals=defaultdict(lambda: {'count':0,'duration_us':0.0})
    groups=defaultdict(lambda: defaultdict(float))
    intervals=[]
    unassigned=Counter()
    kernels=[]
    for e in events:
        if e.get('cat')!='kernel' or e.get('ph')!='X':continue
        kernels.append(e)
        dur=e.get('dur',0)
        intervals.append((e['ts'],e['ts']+dur))
        corr=e.get('args',{}).get('correlation')
        item=runtime.get(corr)
        idx=item[0] if item is not None else None
        phase=steps[idx]['phase'] if idx is not None else 'unattributed'
        name=e['name']
        key=(phase,name)
        totals[key]['count']+=1
        totals[key]['duration_us']+=dur
        groups[phase][category(name)]+=dur
        if idx is not None:
            steps[idx]['kernel_count']+=1
            steps[idx]['kernel_us']+=dur
            steps[idx]['kernels'].append((e['ts'],e['ts']+dur))
        else: unassigned['no_runtime_correlation' if item is None else 'launch_outside_execute_scope']+=1
    phase_summary={}
    for phase in ('decode_only','mixed','prefill_only','empty'):
        selected=[s for s in steps if s['phase']==phase]
        spans=[]
        for s in selected:
            if s['kernels']:
                spans.append(max(b for a,b in s['kernels'])-min(a for a,b in s['kernels']))
        phase_summary[phase]={'steps':len(selected),'steps_with_correlated_kernels':sum(bool(s['kernels']) for s in selected),'cpu_scope_us_p50':quantile([s['duration_us'] for s in selected],.5),'cpu_scope_us_p95':quantile([s['duration_us'] for s in selected],.95),'gpu_kernel_sum_us':sum(s['kernel_us'] for s in selected),'gpu_step_span_us_p50':quantile(spans,.5),'gpu_step_span_us_p95':quantile(spans,.95),'generation_requests_histogram':dict(Counter(s['generation_requests'] for s in selected))}
    for s in steps:
        s['gpu_busy_union_us']=sum(b-a for a,b in union(s.pop('kernels')))
    top=[dict(phase=phase,name=name,category_heuristic=category(name),**v) for (phase,name),v in sorted(totals.items(),key=lambda kv:-kv[1]['duration_us'])]
    cpu_scopes=defaultdict(lambda:{'count':0,'inclusive_us':0.0})
    for e in events:
        if e.get('cat')=='user_annotation' and e.get('ph')=='X' and not STEP.match(e.get('name','')):
            cpu_scopes[e['name']]['count']+=1
            cpu_scopes[e['name']]['inclusive_us']+=e.get('dur',0)
    base=trace.get('baseTimeNanoseconds')
    busy=union(intervals)
    result={'trace':str(path),'baseTimeNanoseconds':base,'displayTimeUnit':trace.get('displayTimeUnit'),'steps_total':len(steps),'phase_summary':phase_summary,'kernel_events':len(kernels),'graph_launch_events':len(graph_launches),'kernel_total_us':sum(e.get('dur',0) for e in kernels),'kernel_busy_union_us':sum(b-a for a,b in busy),'unattributed_kernel_reasons':dict(unassigned),'phase_kernel_categories_heuristic_us':dict(groups),'kernel_table':top,'cpu_user_scopes_inclusive':dict(cpu_scopes),'steps':steps,'limitations':['CPU execute scope duration is dispatch time, not GPU latency; async work can outlive scope.','GPU sum double counts concurrent kernels; union is separate.','Phase comes only from native SchedulerOutput annotation. CUDA correlation required for phase-specific attribution.','Kernel-name category is heuristic and cannot separate backbone from codebook tower when names coincide.','Sampling outside execute scopes remains unattributed.','Profiler timings and counters are not B1 throughput measurements.']}
    offset=base/1000 if isinstance(base,(float,int)) else None
    return result,[(a+offset,b+offset) for a,b in busy] if offset is not None else None

def main():
    p=argparse.ArgumentParser()
    p.add_argument('traces',nargs='+',type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    results=[]
    intervals=[]
    for i,path in enumerate(a.traces):
        result,busy=analyze(path)
        (a.out/f'trace_{i}_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
        results.append({k:v for k,v in result.items() if k not in ('kernel_table','steps','cpu_user_scopes_inclusive')})
        intervals.append(busy)
    comparison={'traces':results,'cross_trace_overlap':[], 'stage0_minimum_evidence': {'decode_steps_with_correlated_kernels': 5, 'mixed_steps_with_correlated_kernels': 3, 'instruction': 'Check these thresholds on the stage 0 trace; collect another bounded window if either class is missing or insufficient. Also verify graph_internal_kernels_present in the probe and actual full-graph replay kernels in this trace.'}}
    for i in range(len(results)):
        for j in range(i+1,len(results)):
            if intervals[i] is not None and intervals[j] is not None:
                comparison['cross_trace_overlap'].append({'trace_a':i,'trace_b':j,'kernel_busy_overlap_us':overlap(intervals[i],intervals[j]),'clock_basis':'trace baseTimeNanoseconds + event ts * 1000; verify both traces same machine and overlapping capture','interpretation':'Observed concurrent device activity; does not prove how much MPS slows either stage.'})
    (a.out/'summary.json').write_text(json.dumps(comparison,indent=2)+'\n')
    print(json.dumps(comparison,indent=2))

if __name__=='__main__':main()
