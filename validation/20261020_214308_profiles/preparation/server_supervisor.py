"""Forward sweep shutdown to only the API; preserve normal stage/MPS cleanup."""
import hashlib, importlib.metadata, json, os, pathlib, re, signal, socket, subprocess, time
import psutil
P = pathlib.Path
ROOT = P(os.environ['PROFILE_ROOT'])
REPO = P(os.environ['OMNI_REPO'])
TOOL = P(os.environ['TOOL_REPO'])
RUN = P(os.readlink('/proc/self/fd/1')).parent
UUID = 'GPU_PHYSICAL_1'
SERVER = None
STOP = False
TRACKED = {}

def cmd(*args): return subprocess.check_output(args, text=True).strip()
def save(name, obj): (RUN/name).write_text(json.dumps(obj, indent=2)+'\n')
def gpu(): return cmd('nvidia-smi','-i',UUID,'--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits')
def apps(): return [s for s in cmd('nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits').splitlines() if s.startswith(UUID)]
def port_free():
    with socket.socket() as s: return s.connect_ex(('127.0.0.1',8091)) != 0

def remember():
    procs=[]
    try: procs=[psutil.Process(SERVER.pid)]+psutil.Process(SERVER.pid).children(recursive=True)
    except psutil.Error: pass
    for p in psutil.process_iter(['name','uids']):
        try:
            if 'nvidia-cuda-mps' in (p.info['name'] or '') and p.info['uids'].real==os.getuid():
                e=p.environ()
                if e.get('CUDA_MPS_PIPE_DIRECTORY','').startswith(os.environ['TMPDIR']+'/vllm-omni-mps-') and e.get('CUDA_VISIBLE_DEVICES')==UUID: procs.append(p)
        except psutil.Error: pass
    for p in procs:
        try:
            if p.uids().real==os.getuid(): TRACKED[p.pid]={'pid':p.pid,'create_time':p.create_time(),'name':p.name()}
        except psutil.Error: pass

def live():
    out=[]
    for pid,r in TRACKED.items():
        try:
            p=psutil.Process(pid)
            if p.uids().real==os.getuid() and p.create_time()==r['create_time'] and p.status()!=psutil.STATUS_ZOMBIE: out.append(p)
        except psutil.Error: pass
    return out

def stop(sig,frame):
    global STOP
    STOP=True

for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP): signal.signal(s,stop)
assert not (ROOT/'BLOCK_NEXT').exists(), 'External GPU occupancy detected; see BLOCK_NEXT'
lock=json.loads((ROOT/'source-lock.json').read_text())
assert cmd('git','-C',str(REPO),'rev-parse','HEAD')==lock['server_commit']
assert not cmd('git','-C',str(REPO),'status','--porcelain','--untracked-files=no')
for relative,sha in lock['tool_sha256'].items(): assert hashlib.sha256((TOOL/relative).read_bytes()).hexdigest()==sha,relative
assert hashlib.sha256((ROOT/'default.yaml').read_bytes()).hexdigest()==lock['deploy_sha256']
assert not apps() and int(gpu().split(',')[2])<128 and port_free(),'GPU or port busy'
argv=['taskset','-c',os.environ['SERVER_CPUS'],'vllm','serve','${NVME_WORKDIR}/models/fishaudio/drama-3','--omni','--host','127.0.0.1','--port','8091','--served-model-name','fishaudio/drama-3','--deploy-config',os.environ['PROFILE_DEPLOY_CONFIG'],'--stage-init-timeout','1800']
save('run_metadata.json',dict(lock,profile_deploy_sha256=hashlib.sha256(P(os.environ['PROFILE_DEPLOY_CONFIG']).read_bytes()).hexdigest(),diagnostic_only=True,server_argv=argv,client_affinity='60-63',server_affinity=os.environ['SERVER_CPUS'],client_smt_siblings_excluded='28-31',gpu_before=gpu(),packages={p:importlib.metadata.version(p) for p in ['torch','vllm','vllm-omni','triton','aiohttp','uvloop']},environment={k:os.environ.get(k) for k in ['BENCH_ARGS','CUDA_VISIBLE_DEVICES','CUDA_HOME','LD_LIBRARY_PATH','OMP_NUM_THREADS','MAX_JOBS','TMPDIR','TRITON_CACHE_DIR','VLLM_CACHE_ROOT','TORCHINDUCTOR_CACHE_DIR','CUDA_CACHE_PATH','HF_HUB_OFFLINE','VLLM_CUSTOM_SCOPES_FOR_PROFILING']},timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
SERVER=subprocess.Popen(argv,cwd=REPO,start_new_session=True)
(RUN/'api.pid').write_text(str(SERVER.pid)+'\n')
print(f'External supervisor {os.getpid()} owns API {SERVER.pid}',flush=True)
last_observed=0.
try:
    while not STOP and SERVER.poll() is None:
        remember()
        if time.monotonic()-last_observed>=15:
            snapshot=[]
            for row in apps():
                pid=int(row.split(',')[1].strip())
                try:
                    p=psutil.Process(pid); entry={'pid':pid,'uid':p.uids().real,'name':p.name(),'owned_by_this_run':pid in TRACKED}
                    snapshot.append(entry)
                    if not entry['owned_by_this_run']:
                        (ROOT/'BLOCK_NEXT').write_text(json.dumps({'time':time.time(),'reason':'untracked GPU compute process; do not signal it','process':entry})+'\n')
                except psutil.Error: pass
            with (RUN/'gpu-process-observations.jsonl').open('a') as f: f.write(json.dumps({'time':time.time(),'processes':snapshot})+'\n')
            last_observed=time.monotonic()
        time.sleep(1)
finally:
    remember(); signals=[]
    if SERVER.poll() is None:
        SERVER.send_signal(signal.SIGINT); signals.append({'pid':SERVER.pid,'signal':'SIGINT'})
        try: SERVER.wait(timeout=180)
        except subprocess.TimeoutExpired:
            SERVER.terminate(); signals.append({'pid':SERVER.pid,'signal':'SIGTERM'})
            try: SERVER.wait(timeout=30)
            except subprocess.TimeoutExpired: SERVER.kill(); SERVER.wait(); signals.append({'pid':SERVER.pid,'signal':'SIGKILL'})
    for sig,seconds in [(signal.SIGTERM,20),(signal.SIGKILL,10)]:
        remaining=live()
        if not remaining: break
        for p in remaining:
            try: p.send_signal(sig); signals.append({'pid':p.pid,'signal':signal.Signals(sig).name})
            except psutil.Error: pass
        psutil.wait_procs(remaining,timeout=seconds)
    for _ in range(20):
        if port_free() and not live() and not apps() and int(gpu().split(',')[2])<128: break
        time.sleep(1)
    result={'tracked_processes':list(TRACKED.values()),'signals':signals,'server_exit':SERVER.returncode,'gpu_after':gpu(),'apps_after':apps(),'port_free':port_free(),'own_processes_remaining':[p.pid for p in live()]}
    result['clean']=result['port_free'] and not result['apps_after'] and not result['own_processes_remaining'] and int(result['gpu_after'].split(',')[2])<128
    save('supervisor_cleanup.json',result)
    print('External supervisor cleanup: '+json.dumps(result),flush=True)
    if not result['clean']: raise SystemExit(1)
raise SystemExit(0 if STOP else SERVER.returncode)
