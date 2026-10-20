"""Read-only audit of completed runs; no service changes."""
import argparse,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--expected',type=int,required=True);a=p.parse_args()
root=Path(a.out);runs=[]
for summary in sorted(root.glob('c*/r*/summary.json')):
    d=summary.parent;s=json.loads(summary.read_text());rec=[json.loads(x) for x in (d/'requests.jsonl').read_text().splitlines()];log=(d/'server.log').read_text(errors='replace')
    key=[x for x in log.splitlines() if re.search('KV cache size|Maximum concurrency|preempt|OutOfMemory|out of memory|CUDA error|ERROR|Traceback',x,re.I)]
    (d/'key_lines.txt').write_text('\n'.join(key)+'\n')
    fatal=[x for x in log.splitlines() if re.search('out of memory|OutOfMemory|CUDA error|EngineDeadError|preempted|resuming preempted|total_num_cumulative_preemption[^\n]*[1-9]',x,re.I)]
    errors=[];events=[]
    for r in rec:
        if r.get('ok'):
            events.extend([(r['sent_s'],1),(r['sent_s']+r['e2e_s'],-1)])
    cur=mx=0;at_full=0.;prev=None;c=int(d.parent.name[1:])
    for t,delta in sorted(events):
        if prev is not None and cur==c:at_full+=t-prev
        cur+=delta;mx=max(mx,cur);prev=t
    if s['requests']!=a.expected or s['succeeded']!=a.expected or s['failed']!=0:errors.append('summary_counts')
    if sorted(r['index'] for r in rec)!=list(range(a.expected)):errors.append('row_indices')
    if len({r['utt_id'] for r in rec})!=a.expected:errors.append('utt_id_duplicates')
    if not all(r['ok'] and r['status']==200 for r in rec):errors.append('request_failure')
    if not all(r['seed'] is None for r in rec):errors.append('explicit_seed')
    if s['client']['loop_lag_ms']['p99']>=10:errors.append('loop_lag_p99')
    if (d/'audio').exists():errors.append('unexpected_audio')
    if fatal:errors.append('preemption_or_cuda_error')
    clean=json.loads((d/'supervisor_cleanup.json').read_text()) if (d/'supervisor_cleanup.json').exists() else {}
    if not clean.get('clean'):errors.append('cleanup')
    expected_exit=f'c={c} r={d.name[1:]}: client exit 0, stopping server'
    if expected_exit not in (root/'sweep.log').read_text():errors.append('client_exit')
    if mx!=c:errors.append('not_reached_requested_concurrency')
    result={'path':str(d),'concurrency':c,'repeat':int(d.name[1:]),'requests':s['requests'],'succeeded':s['succeeded'],'failed':s['failed'],'loop_lag_p99_ms':s['client']['loop_lag_ms']['p99'],'startup_s':float((d/'startup_seconds.txt').read_text()),'duration_s':s['duration_s'],'throughput_audio_s_per_s':s['throughput_audio_s_per_s'],'ttfa_ms':s['ttfa_ms'],'rtf':s['rtf'],'long_takes':s.get('long_takes'),'full_concurrency_window':s['full_concurrency_window'],'max_client_inflight_lower_bound':mx,'seconds_at_requested_client_inflight_lower_bound':round(at_full,3),'inflight_method':'Intervals sent_s to last audio arrival (e2e_s rounded to 0.1 ms). Client HTTP in-flight lower bound, not number scheduled on GPU.','key_lines':key,'preemption_oom_cuda_lines':fatal,'cleanup':clean,'errors':errors}
    runs.append(result)
result={'valid':bool(runs) and all(not r['errors'] for r in runs),'completed_runs':len(runs),'runs':runs}
(root/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));raise SystemExit(not result['valid'])
