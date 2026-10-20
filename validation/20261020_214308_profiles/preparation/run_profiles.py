"""Run isolated diagnostic services; never part of B1. GPU handover required."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT=Path(os.environ['PROFILE_ROOT']);PREP=ROOT/'preparation'
UUID='GPU_PHYSICAL_1'
STOP=False
FIELDS='timestamp,memory.used,memory.total,clocks.sm,temperature.gpu,utilization.gpu,power.draw,clocks_throttle_reasons.active,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_power_brake_slowdown'
def save(p,obj):p.write_text(json.dumps(obj,indent=2)+'\n')
def cmd(*args):return subprocess.check_output(args,text=True).strip()
def interrupted(*_):
 global STOP
 STOP=True
for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,interrupted)
def ensure_handover():
 assert (ROOT/'GPU_HANDOVER_AUTHORIZED').exists(),'Root GPU handover flag absent'
 assert (ROOT/'graph_probe/probe.json').exists(),'Graph replay visibility probe not completed'
 assert json.loads((ROOT/'graph_probe/probe.json').read_text())['graph_internal_kernels_present'],'Graph replay probe failed'
 assert not (ROOT/'BLOCK_NEXT').exists(),'GPU occupancy guard tripped'
 apps=cmd('nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader')
 assert UUID not in apps,'GPU 1 has active compute processes'
 assert int(cmd('nvidia-smi','-i',UUID,'--query-gpu=memory.used','--format=csv,noheader,nounits'))<128,'GPU 1 not idle'
 lock=json.loads((ROOT/'helper-lock.json').read_text())
 for name,sha in lock.items():assert hashlib.sha256((PREP/name).read_bytes()).hexdigest()==sha,name

def run_one(c,attempt):
 ensure_handover()
 run=ROOT/f'c{c}'/f'attempt{attempt}'
 run.mkdir(parents=True,exist_ok=False)
 (run/'helper-lock-executed.json').write_text((ROOT/'helper-lock.json').read_text())
 config=run/'deploy_config.yaml'
 subprocess.run([sys.executable,str(PREP/'make_config.py'),'--source',str(ROOT/'default.yaml'),'--out',str(config),'--trace-dir',str(run/'traces')],check=True)
 env=dict(os.environ,PROFILE_DEPLOY_CONFIG=str(config))
 supervisor=client=monitor=None
 failure=None
 try:
  with (run/'gpu_thermal.csv').open('w') as f,(run/'gpu-monitor.stderr').open('w') as e:
   monitor=subprocess.Popen(['nvidia-smi','-i',UUID,'--query-gpu='+FIELDS,'--format=csv','-l','1'],stdout=f,stderr=e)
  started=time.time();(run/'t_start').write_text(str(started)+'\n')
  with (run/'server.log').open('w') as f:
   supervisor=subprocess.Popen([sys.executable,str(PREP/'server_supervisor.py')],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  (run/'supervisor.pid').write_text(str(supervisor.pid)+'\n')
  deadline=time.monotonic()+2100
  while time.monotonic()<deadline:
   assert not STOP,'Interrupted before readiness'
   assert supervisor.poll() is None,f'Supervisor exited {supervisor.returncode}'
   try:
    with urllib.request.urlopen('http://127.0.0.1:8091/health',timeout=2) as r:
     if r.status==200:break
   except Exception:pass
   time.sleep(3)
  else:raise TimeoutError('Readiness exceeded 35 minutes')
  (run/'startup_seconds.txt').write_text(f'{time.time()-started:.3f}\n')
  serverlog=(run/'server.log').read_text(errors='replace')
  assert 'Profiler endpoints are enabled' in serverlog,'Profiler routes not enabled'
  argv=['taskset','-c','60-63',sys.executable,str(PREP/'capture_client.py'),'--codes',os.environ['CODES'],'--ref-mode','codes','--concurrency',str(c),'--num-prompts','1088','--num-warmups','8','--label',f'diagnostic-profile-c{c}','--out',str(run/'client')]
  save(run/'client-command.json',{'argv':argv,'server_affinity':os.environ['SERVER_CPUS'],'no_explicit_seed':True,'save_audio':False,'start_after_s':env['PROFILE_START_AFTER_S'],'capture_s':env['PROFILE_DURATION_S']})
  with (run/'client.log').open('w') as f:
   client=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  deadline=time.monotonic()+2700
  while client.poll() is None:
   assert not STOP,'Interrupted during workload'
   assert supervisor.poll() is None,f'Supervisor exited {supervisor.returncode}'
   assert time.monotonic()<deadline,'Workload exceeded 45 minutes'
   time.sleep(3)
  (run/'client.exit').write_text(str(client.returncode)+'\n')
  assert client.returncode==0,f'Client failed {client.returncode}'
  s=json.loads((run/'client/summary.json').read_text())
  assert s['succeeded']==1088 and s['failed']==0,s
  api=json.loads((run/'client/profile_api.json').read_text())
  assert [x['action'] for x in api]==['start_profile','stop_profile'] and all(x['status']==200 for x in api),api
 except BaseException as e:
  failure=f'{type(e).__name__}: {e}'
  save(run/'failure.json',{'error':failure})
 finally:
  if client is not None and client.poll() is None:
   client.terminate()
   try:client.wait(timeout=20)
   except subprocess.TimeoutExpired:client.kill();client.wait()
  if supervisor is not None and supervisor.poll() is None:
   # Supervisor forwards SIGINT only to its API, then validates own stage/MPS cleanup.
   supervisor.send_signal(signal.SIGINT)
   try:supervisor.wait(timeout=290)
   except subprocess.TimeoutExpired:
    failure=(failure or '')+' Supervisor cleanup timeout; do not kill arbitrary groups'
  if monitor is not None and monitor.poll() is None:monitor.terminate();monitor.wait(timeout=10)
  cleanup=run/'supervisor_cleanup.json'
  if not cleanup.exists() or not json.loads(cleanup.read_text())['clean']:
   failure=(failure or '')+' Missing or incomplete supervisor cleanup'
  if failure:
   save(run/'failure.json',{'error':failure})
   raise RuntimeError(failure)
 # Gzip is spawned in background by native profiler; wait for complete files.
 deadline=time.monotonic()+180
 while time.monotonic()<deadline:
  traces=sorted((run/'traces').rglob('trace_rank0.json.gz'))
  pending=list((run/'traces').rglob('trace_rank0.json'))
  if len(traces)>=2 and not pending:
   checks=[subprocess.run(['gzip','-t',str(t)],capture_output=True).returncode for t in traces]
   if all(x==0 for x in checks):break
  time.sleep(2)
 else:raise RuntimeError('Expected complete stage 0 and stage 1 traces not found')
 with (run/'analysis.log').open('w') as f:
  subprocess.run([sys.executable,str(PREP/'analyze_trace.py'),*[str(t) for t in traces],'--out',str(run/'analysis')],stdout=f,stderr=subprocess.STDOUT,check=True)
 summaries=json.loads((run/'analysis/summary.json').read_text())['traces']
 stage0=[x for x in summaries if 'stage0_' in x['trace'] or 'stage_0_' in x['trace']]
 assert len(stage0)==1,f'Cannot identify unique stage 0 trace: {[x["trace"] for x in summaries]}'
 s=stage0[0]
 adequate=s['phase_summary']['decode_only']['steps_with_correlated_kernels']>=5 and s['phase_summary']['mixed']['steps_with_correlated_kernels']>=3
 coverage={'phase_samples_adequate':adequate,'all_stages_have_cuda_kernels':all(x['kernel_events']>0 for x in summaries),'stage0_graph_launches':s['graph_launch_events'],'requires_manual_graph_node_and_drop_audit':True,'trace_window_note':'Each stage independently stops at max_iterations or HTTP stop, so stage windows may differ.'}
 save(run/'coverage_gate.json',coverage)
 print(json.dumps({'completed':str(run),'coverage':coverage}),flush=True)
 return adequate and coverage['all_stages_have_cuda_kernels']

p=argparse.ArgumentParser();p.add_argument('--concurrency',type=int,nargs='+',default=[8,16,64]);p.add_argument('--attempt',type=int,default=1);a=p.parse_args()
for c in a.concurrency:
 if STOP:break
 if not run_one(c,a.attempt):
  print(f'C{c}: insufficient capture coverage; preserving attempt and stopping for review',flush=True)
  raise SystemExit(2)
