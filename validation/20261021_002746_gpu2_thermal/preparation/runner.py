"""Continuous read-only thermal monitor starts BEFORE service startup."""
import json,os,signal,socket,subprocess,threading,time,urllib.request
from pathlib import Path
ROOT=Path(os.environ['B1_ROOT']);OUT=Path(os.environ['OUT']);UUID='GPU_PHYSICAL_2'
FIELDS='timestamp,temperature.gpu,temperature.memory,clocks.sm,clocks.mem,power.draw,power.limit,utilization.gpu,memory.used,clocks_event_reasons.active,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_power_brake_slowdown'
def query():return subprocess.check_output(['nvidia-smi','-i',UUID,'--query-gpu='+FIELDS,'--format=csv,noheader'],text=True).strip()
def save(name,obj):(ROOT/name).write_text(json.dumps(obj,indent=2)+'\n')
def anchor():
 before=time.time_ns(); perf=time.perf_counter_ns(); after=time.time_ns(); return {'wall_before_ns':before,'perf_counter_ns':perf,'wall_after_ns':after}
def phase_watch():
 seen=set()
 while not stop.is_set():
  log=OUT/'runner.log'
  if log.exists():
   text=log.read_text(errors='replace')
   for key,token in [('warmup_print_observed','warmup: 8 requests'),('timed_print_before_gc_observed','timed: 1088 requests'),('client_exit_observed','client exit 0')]:
    if key not in seen and token in text:
     seen.add(key)
     with (ROOT/'phase_observations.jsonl').open('a') as f:f.write(json.dumps({'event':key,'clock_anchor':anchor(),'poll_interval_s':0.1,'note':'Log observation, not exact run_phase boundary'})+'\n')
  stop.wait(.1)
apps=subprocess.check_output(['nvidia-smi','-i',UUID,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert not apps,'GPU2 occupied'
with socket.socket() as s:assert s.connect_ex(('127.0.0.1',8091))!=0,'port8091 busy'
(ROOT/'nvidia-smi-before.txt').write_text(subprocess.check_output(['nvidia-smi','-q','-i',UUID],text=True));save('before_service.json',{'epoch_s':time.time(),'gpu_thermal':query(),'fields':FIELDS})
stop=threading.Event();monitor=sweep=None;rc=1

def health():
 while not stop.is_set():
  try:
   with urllib.request.urlopen('http://127.0.0.1:8091/health',timeout=.5) as r:
    if r.status==200:
     save('health_ready.json',{'epoch_s':time.time(),'monotonic_s':time.monotonic(),'gpu_thermal':query(),'poll_interval_s':1});return
  except Exception:pass
  stop.wait(1)
try:
 with (ROOT/'thermal_continuous.csv').open('w') as f,(ROOT/'thermal-monitor.stderr').open('w') as e:
  monitor=subprocess.Popen(['nvidia-smi','-i',UUID,'--query-gpu='+FIELDS,'--format=csv','-l','1'],stdout=f,stderr=e)
 save('clock_anchor_before.json',anchor())
 save('monitor.json',{'pid':monitor.pid,'started_epoch_s':time.time(),'fields':FIELDS,'interval_s':1})
 deadline=time.monotonic()+10
 while len((ROOT/'thermal_continuous.csv').read_text().splitlines())<2:
  assert monitor.poll() is None,'thermal monitor exited'
  if time.monotonic()>deadline:raise RuntimeError('no first sample; refusing to launch')
  time.sleep(.1)
 t=threading.Thread(target=health,daemon=True);t.start()
 threading.Thread(target=phase_watch,daemon=True).start()
 with (OUT/'runner.log').open('w') as log:
  sweep=subprocess.Popen(['bash','tools/drama3_bench/run_sweep.sh'],stdout=log,stderr=subprocess.STDOUT)
 rc=sweep.wait();(OUT/'runner.exit').write_text(str(rc)+'\n')
finally:
 save('clock_anchor_after.json',anchor())
 stop.set()
 if monitor is not None and monitor.poll() is None:monitor.terminate();monitor.wait(timeout=10)
 save('monitor_cleanup.json',{'monitor_pid':monitor.pid if monitor else None,'monitor_exit':monitor.returncode if monitor else None,'epoch_s':time.time(),'gpu_thermal_after':query(),'sweep_exit':rc})
raise SystemExit(rc)
