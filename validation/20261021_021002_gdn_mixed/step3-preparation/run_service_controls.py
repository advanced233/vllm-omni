"""External four-service runner; never starts before explicit Step1/2 authorization.
Run from the existing activated user CUDA12.9 environment. User tools unchanged.
"""
import argparse, ast, hashlib, json, os, re, signal, socket, subprocess, sys, time
from pathlib import Path
import yaml

P=Path
PREP=P(__file__).resolve().parent
UUID='GPU_PHYSICAL_1'
REPO=P('${WORKSPACE}/vllm-omni')
MODEL='${NVME_WORKDIR}/models/fishaudio/drama-3'
CODES='${NVME_WORKDIR}/bench-validation/20261020_165001_49599d4/ref_codes_en.jsonl'
FIELDS='timestamp,memory.used,memory.total,clocks.sm,clocks.mem,temperature.gpu,temperature.memory,utilization.gpu,power.draw,clocks_throttle_reasons.active,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_power_brake_slowdown'
MODES=['baseline_laps','baseline_no_laps','installed','direct_out_laps']
STOP=False

def on_signal(*_):
 global STOP
 STOP=True
for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:signal.signal(sig,on_signal)
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def cmd(*a):return subprocess.check_output(a,text=True).strip()
def digest(p):return hashlib.sha256(P(p).read_bytes()).hexdigest()
def clean_env(source):
 env=dict(source)
 for key in list(env):
  if key.startswith('DRAMA3_GDN_') or key in ['VLLM_CUSTOM_SCOPES_FOR_PROFILING','VLLM_NVTX_SCOPES_FOR_PROFILING']:env.pop(key,None)
 paths=[x for x in env.get('PYTHONPATH','').split(os.pathsep) if x and not x.rstrip('/').endswith('/gdn_mixed/probe_site')]
 if paths:env['PYTHONPATH']=os.pathsep.join(paths)
 else:env.pop('PYTHONPATH',None)
 return env

def patch_status(run,mode,require=False):
 text=(run/'server.log').read_text(errors='replace')
 assert 'NOT patched' not in text and '[drama3-gdn-probe] not installed:' not in text,'Probe rejected installation'
 pids=sorted(set(map(int,re.findall(r'StageEngineCoreProc_stage0_replica0 pid=(\d+)',text))))
 if mode=='installed':
  assert not re.search(r'\[drama3-gdn-probe pid=\d+\] patched',text),'Installed control unexpectedly patched'
 elif require:
  assert len(pids)==1,f'No unique stage0 process: {pids}'
  variant='direct_out' if mode=='direct_out_laps' else 'baseline';laps=mode!='baseline_no_laps'
  assert f'[drama3-gdn-probe pid={pids[0]}] patched mixed_forward_core: variant={variant} laps={laps}' in text,'No stage0-specific successful patch'
 return pids

def assert_free():
 assert UUID not in cmd('nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'),'GPU1 busy'
 assert int(cmd('nvidia-smi','-i',UUID,'--query-gpu=memory.used','--format=csv,noheader,nounits'))<128,'GPU1 memory busy'
 with socket.socket() as s:assert s.connect_ex(('127.0.0.1',8091))!=0,'8091 occupied'

def fingerprint():
 source=(REPO/'vllm_omni/model_executor/models/fish_omni/gdn.py').read_text()
 node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='mixed_forward_core')
 segment=ast.get_source_segment(source,node)
 return hashlib.sha256('\n'.join(line.rstrip() for line in segment.strip().splitlines()).encode()).hexdigest()

def run_client(run,out,n,c,warmups,env,supervisor,mode,tools):
 argv=['taskset','-c','60-63',sys.executable,str(tools/'stream_bench.py'),'--codes',CODES,'--ref-mode','codes','--num-prompts',str(n),'--concurrency',str(c),'--num-warmups',str(warmups),'--label','gdn-'+mode,'--out',str(out)]
 out.mkdir();write(out/'command.json',{'argv':argv,'probe_environment_absent':not any(k.startswith('DRAMA3_GDN_') for k in env),'no_explicit_seed':True,'no_audio':True,'thermal_gate':False})
 client=None
 try:
  with (out/'client.log').open('w') as f:client=subprocess.Popen(argv,env=env,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  (out/'client.pid').write_text(str(client.pid)+'\n');deadline=time.monotonic()+1800
  while client.poll() is None:
   assert not STOP,'Interrupted'
   assert supervisor.poll() is None,'Server supervisor exited'
   assert not (P(env['GDN_EXPERIMENT_ROOT'])/'BLOCK_NEXT').exists(),'Foreign GPU process detected'
   assert time.monotonic()<deadline,'Client timeout'
   patch_status(run,mode)
   time.sleep(1)
  (out/'client.exit').write_text(str(client.returncode)+'\n');assert client.returncode==0,'Client failed'
  s=json.loads((out/'summary.json').read_text());rows=[json.loads(x) for x in (out/'requests.jsonl').read_text().splitlines()]
  assert s['succeeded']==n and s['failed']==0 and len(rows)==n
  assert sorted(x['index'] for x in rows)==list(range(n)) and all(x['ok'] and x['status']==200 for x in rows)
  assert not (out/'audio').exists()
  return s
 finally:
  if client is not None and client.poll() is None:
   client.terminate()
   try:client.wait(timeout=20)
   except subprocess.TimeoutExpired:client.kill();client.wait()

def run_one(root,tools,deploy,mode,base_env):
 assert not (root/'BLOCK_NEXT').exists(),'Untracked GPU process was observed; stop before next group'
 assert_free()
 run=root/'service_controls'/mode;run.mkdir(parents=True,exist_ok=False)
 env=dict(base_env,GDN_EXPERIMENT_ROOT=str(root),OMNI_REPO=str(REPO),TOOL_REPO=str(tools.parent.parent),SERVER_CPUS='0-27,32-59')
 env['TMPDIR']=f'${NVME_WORKDIR}/tmp/gdn-{root.name[-6:]}';P(env['TMPDIR']).mkdir(parents=True,exist_ok=True)
 injection={}
 if mode!='installed':
  injection={'DRAMA3_GDN_PROBE':'1','DRAMA3_GDN_VARIANT':'direct_out' if mode=='direct_out_laps' else 'baseline','DRAMA3_GDN_PROBE_DIR':str((run/'probe').resolve()),'PYTHONPATH':str(tools/'gdn_mixed/probe_site')+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')}
  if mode=='baseline_no_laps':injection['DRAMA3_GDN_PROBE_LAPS']='0'
 (run/'probe').mkdir()
 settings={'mode':mode,'service_only_environment':injection,'deploy_config':str(deploy),'static_installed_function_sha256':fingerprint(),'probe_enabled':mode!='installed','main_requests':256,'main_concurrency':16,'drain_requests_per_batch':33,'drain_concurrency':1,'no_torch_profiler':True,'no_thermal_gate':True}
 write(run/'service_settings.json',settings)
 supervisor=monitor=None;failure=None
 try:
  with (run/'gpu_thermal.csv').open('w') as f,(run/'gpu-monitor.stderr').open('w') as e:monitor=subprocess.Popen(['nvidia-smi','-i',UUID,'--query-gpu='+FIELDS,'--format=csv','-l','1'],stdout=f,stderr=e)
  (run/'gpu-monitor.pid').write_text(str(monitor.pid)+'\n')
  with (run/'server.log').open('w') as f:supervisor=subprocess.Popen([sys.executable,str(PREP/'server_supervisor.py')],env=env,cwd=REPO,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  (run/'supervisor.pid').write_text(str(supervisor.pid)+'\n');started=time.time();deadline=time.monotonic()+2100
  while time.monotonic()<deadline:
   assert not STOP and supervisor.poll() is None,'Startup interrupted/failed'
   assert not (root/'BLOCK_NEXT').exists(),'Foreign GPU process detected during startup'
   patch_status(run,mode)
   try:
    import urllib.request
    with urllib.request.urlopen('http://127.0.0.1:8091/health',timeout=2) as response:
     if response.status==200:break
   except Exception:pass
   time.sleep(2)
  else:raise TimeoutError('Readiness timeout')
  patch_status(run,mode,require=True)
  (run/'startup_seconds.txt').write_text(f'{time.time()-started:.3f}\n')
  assert monitor.poll() is None,'Hardware monitor failed'
  run_client(run,run/'client',256,16,8,env,supervisor,mode,tools)
  write(run/'bench_boundaries.json',{'main_client_completed_wall_s':time.time(),'note':'Actual completion used for durable post-window flush proof; official summarize_probe conservative window used for statistics.'})
  for n in range(3):
   run_client(run,run/f'flush_drain_{n+1}',33,1,0,env,supervisor,mode,tools)
   check=subprocess.run([sys.executable,str(PREP/'validate_probe.py'),'--run',str(run),'--tools',str(tools)],env=env,capture_output=True,text=True)
   (run/f'probe-check-before-stop-{n+1}.log').write_text(check.stdout+check.stderr)
   if check.returncode==0:break
   assert 'No durable post-main record' in check.stderr,'Probe semantic validation failed; additional drain cannot repair wrong variant/call count'
  else:raise RuntimeError('Probe flush/completeness validation failed after3 bounded drain batches')
 except BaseException as e:
  failure=f'{type(e).__name__}: {e}';write(run/'failure.json',{'error':failure})
 finally:
  if supervisor is not None and supervisor.poll() is None:
   supervisor.send_signal(signal.SIGINT)
   try:supervisor.wait(timeout=290)
   except subprocess.TimeoutExpired:failure=(failure or '')+' Supervisor cleanup timeout; no arbitrary process group kill'
  if monitor is not None and monitor.poll() is None:monitor.terminate();monitor.wait(timeout=10)
  cleanup=run/'supervisor_cleanup.json'
  if not cleanup.exists() or not json.loads(cleanup.read_text())['clean']:failure=(failure or '')+' Missing/failed cleanup'
  if failure:write(run/'failure.json',{'error':failure});raise RuntimeError(failure)
 subprocess.run([sys.executable,str(PREP/'validate_probe.py'),'--run',str(run),'--tools',str(tools)],env=env,check=True,stdout=(run/'probe-check-after-stop.log').open('w'))
 with (run/'official_probe_summary.txt').open('w') as f:
  subprocess.run([sys.executable,str(tools/'gdn_mixed/summarize_probe.py'),str(run/'probe'),'--bench-summary',str(run/'client/summary.json'),'--json',str(run/'official_probe_summary.json')],env=env,check=True,stdout=f,stderr=subprocess.STDOUT)
 print(f'COMPLETED {mode}',flush=True)

ap=argparse.ArgumentParser();ap.add_argument('--experiment-root',type=P,required=True);ap.add_argument('--tools-root',type=P,required=True);ap.add_argument('--deploy-config',type=P,required=True);ap.add_argument('--modes',nargs='+',choices=MODES,default=MODES);args=ap.parse_args()
root=args.experiment_root.resolve();tools=args.tools_root.resolve();deploy=args.deploy_config.resolve()
auth=json.loads((root/'STEP3_AUTHORIZED.json').read_text());assert auth['step1_all_37_passed'] and auth['step2_completed'] and auth['root_gpu_handover']
assert P(auth['step1_evidence']).exists() and P(auth['step2_evidence']).exists()
step1=json.loads(P(auth['step1_evidence']).read_text())
assert step1['step1_gate_passed'] and step1['collected']==37 and step1['counts']=={'passed':37}
assert step1['tool_commit']=='7aeaa5b46ec1cdbdb7e5d43b396b4d05af87fa37'
assert P(sys.prefix).resolve()==P('${NVME_WORKDIR}/venvs/vllm-omni-cu129').resolve(),'Use existing private venv'
base=clean_env(os.environ);base.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='1',MAX_JOBS='4',OMP_NUM_THREADS='8',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
for k in ['CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY']:base.pop(k,None)
cfg=yaml.safe_load(deploy.read_text());assert cfg['stages'][0]['mamba_ssm_cache_dtype']=='bfloat16';assert all('profiler_config' not in x for x in cfg['stages'])
expected=re.search(r'EXPECTED_SOURCE_SHA256 = "([a-f0-9]+)"',(tools/'gdn_mixed/drama3_gdn_variants.py').read_text()).group(1);assert fingerprint()==expected
paths=[deploy,P(CODES),tools/'stream_bench.py',*sorted((tools/'gdn_mixed').rglob('*.py')),*sorted(PREP.glob('*.py'))]
lock={'tool_snapshot_commit':'7aeaa5b46ec1cdbdb7e5d43b396b4d05af87fa37','server_commit':cmd('git','-C',str(REPO),'rev-parse','HEAD'),'files':{str(p):digest(p) for p in paths},'function_sha256':expected,'authorization':auth,'no_source_mutation':True}
if (root/'step3-lock.json').exists():assert json.loads((root/'step3-lock.json').read_text())==lock,'Source lock changed'
else:write(root/'step3-lock.json',lock)
for mode in args.modes:run_one(root,tools,deploy,mode,base)
