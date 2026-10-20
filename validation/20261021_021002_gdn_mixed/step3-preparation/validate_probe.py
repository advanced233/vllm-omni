"""CPU-only validation of the actual stage0 probe, timed window and tail flush."""
import argparse, importlib.util, json, re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--tools',type=Path,required=True);a=p.parse_args()
spec=importlib.util.spec_from_file_location('user_summarize_probe',a.tools/'gdn_mixed/summarize_probe.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
settings=json.loads((a.run/'service_settings.json').read_text());mode=settings['mode'];text=(a.run/'server.log').read_text(errors='replace')
assert 'NOT patched' not in text and '[drama3-gdn-probe] not installed:' not in text,'Probe installation rejected'
pids=sorted(set(int(x) for x in re.findall(r'StageEngineCoreProc_stage0_replica0 pid=(\d+)',text)))
assert len(pids)==1,f'Need unique stage0 PID, got {pids}'
stage0=pids[0]
if mode=='installed':
 assert not re.search(r'\[drama3-gdn-probe pid=\d+\] patched',text),'Installed control unexpectedly patched'
 assert not list((a.run/'probe').glob('gdn_probe_*.jsonl')),'Installed control unexpectedly recorded probe data'
 out={'mode':mode,'stage0_pid':stage0,'unpatched_installed_control':True,'ok':True}
else:
 variant='direct_out' if mode=='direct_out_laps' else 'baseline';laps=mode!='baseline_no_laps'
 needle=f'[drama3-gdn-probe pid={stage0}] patched mixed_forward_core: variant={variant} laps={laps}'
 assert needle in text,'API patch success is insufficient; stage0 patch not confirmed'
 records=m.load(a.run/'probe');stage=[r for r in records if r['pid']==stage0]
 assert stage,'No stage0 records'
 assert all(r['variant']==variant and r['laps']==laps for r in stage),'Wrong variant/laps'
 lo,hi=m.bench_window(a.run/'client/summary.json');timed=[r for r in stage if lo<=r['t']<=hi]
 assert timed,'No timed-window stage0 records'
 assert all(r['calls']==30 for r in timed),'Incomplete timed per-step30-layer records'
 completed=json.loads((a.run/'bench_boundaries.json').read_text())['main_client_completed_wall_s']
 assert max(r['t'] for r in stage)>completed,'No durable post-main record; final timed buffer may be lost'
 out={'ok':True,'mode':mode,'stage0_pid':stage0,'expected_variant':variant,'expected_laps':laps,'window':[lo,hi],'all_stage0_steps':len(stage),'timed_steps':len(timed),'calls_per_step':sorted(set(r['calls'] for r in timed)),'outside_window_incomplete_records':sum(r['calls']!=30 for r in stage if not lo<=r['t']<=hi),'persisted_post_main':True,'main_client_completed_wall_s':completed,'last_persisted_t':max(r['t'] for r in stage),'timed_summary':m.summarize(timed),'flush_rationale':'Sequential append contains records later than the actual main client completion; therefore main-window buffered records were flushed before shutdown.'}
(a.run/'probe_validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
