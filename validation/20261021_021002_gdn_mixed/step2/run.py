"""Run the unchanged user microbenchmark with external provenance/telemetry only."""
import hashlib,importlib.metadata,json,math,os,psutil,re,subprocess,sys,time
from pathlib import Path
ROOT=Path(os.environ['GDN_ROOT']);OUT=ROOT/'step2';REPO=Path('${WORKSPACE}/vllm-omni');UUID='GPU_PHYSICAL_1'
FIELDS='timestamp,uuid,temperature.gpu,temperature.memory,clocks.sm,clocks.mem,power.draw,power.limit,utilization.gpu,memory.used,clocks_event_reasons.active,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_power_brake_slowdown'
def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2)+'\n')
def query():return subprocess.check_output(['nvidia-smi','-i',UUID,'--query-gpu='+FIELDS,'--format=csv,noheader'],text=True).strip()
def apps():return subprocess.check_output(['nvidia-smi','-i',UUID,'--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
assert json.loads((ROOT/'step1/summary.json').read_text())['step1_gate_passed']
assert not apps(),'GPU1 busy'
assert not any(k.startswith('DRAMA3_') or k=='PROBE_FORCE' for k in os.environ),'unexpected probe variable'
assert not os.environ.get('PYTHONPATH'),'PYTHONPATH injection is not allowed'
lock=json.loads((ROOT/'source-lock.json').read_text())
for name,sha in lock['snapshot_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
argv=['taskset','-c','0-27',sys.executable,'tools/drama3_bench/gdn_mixed/bench_mixed_core.py','--cases','c8','c16','c64','--state-dtype','bf16','--iters','200','--kernel-time','--json',str(OUT/'gdn_bench_bf16.json')]
save('metadata.json',{'argv':argv,'cwd':str(REPO),'tool_commit':lock['tool_commit'],'server_commit':lock['server_commit'],'snapshot_sha256':lock['snapshot_sha256'],'gpu_before':query(),'environment':{k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','CUDA_HOME','LD_LIBRARY_PATH','PYTHONPATH','OMP_NUM_THREADS','MAX_JOBS','TMPDIR']},'probe_environment_present':False,'monitor_cpu_affinity':'60-63','requested_bench_cpu_affinity':'0-27','started_epoch_s':time.time(),'versions':{p:importlib.metadata.version(p) for p in ['torch','vllm','vllm-omni','triton']}})
monitor=None;bench=None;matched=False;rc=1;failure=None
try:
 with (OUT/'gpu_thermal.csv').open('w') as f,(OUT/'gpu-monitor.stderr').open('w') as e:
  monitor=subprocess.Popen(['taskset','-c','60-63','nvidia-smi','-i',UUID,'--query-gpu='+FIELDS,'--format=csv','-l','1'],stdout=f,stderr=e)
 save('monitor.json',{'pid':monitor.pid,'create_time':psutil.Process(monitor.pid).create_time()})
 with (OUT/'bench.log').open('w') as log:
  bench=subprocess.Popen(argv,cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
  save('process.json',{'pid':bench.pid,'create_time':psutil.Process(bench.pid).create_time()})
  for line in bench.stdout:
   log.write(line);log.flush()
   if line.startswith('installed mixed_forward_core sha256 '):
    matched=': matches the copy' in line
    save('fingerprint_line.json',{'line':line.strip(),'matches':matched,'observed_epoch_s':time.time(),'cpu_affinity':psutil.Process(bench.pid).cpu_affinity()})
    if not matched:bench.terminate();failure='fingerprint mismatch';break
  rc=bench.wait()
 (OUT/'bench.exit').write_text(str(rc)+'\n')
 if rc or not matched:failure=failure or f'exit={rc}, fingerprint_matches={matched}'
 if not failure:
  data=json.loads((OUT/'gdn_bench_bf16.json').read_text());rows=data['rows'];expected={(c,v) for c in ['c8','c16','c64'] for v in ['installed','baseline','direct_out']}
  assert len(rows)==9 and {(x['case'],x['variant']) for x in rows}==expected
  for x in rows:
   assert x['state_dtype']=='bfloat16'
   for k in ['host_us_p50','host_us_p90','gpu_span_us_p50','kernel_us']:assert isinstance(x[k],(int,float)) and math.isfinite(x[k]) and x[k]>0,(x['case'],x['variant'],k,x[k])
  save('validation.json',{'valid':True,'rows':9,'fingerprint_matches':matched,'runtime_exit':rc,'step3_started':False})
except BaseException as exc:
 failure=f'{type(exc).__name__}: {exc}';save('failure.json',{'error':failure});raise
finally:
 if bench is not None and bench.poll() is None:bench.terminate();bench.wait(timeout=30)
 if monitor is not None and monitor.poll() is None:monitor.terminate();monitor.wait(timeout=10)
 save('cleanup.json',{'bench_exit':bench.returncode if bench else None,'monitor_exit':monitor.returncode if monitor else None,'gpu_after':query(),'gpu_compute_apps_after':apps(),'failure':failure,'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
 if failure:save('failure.json',{'error':failure})
raise SystemExit(1 if failure else 0)
