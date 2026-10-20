"""Launch exactly one quality run, preserve all outputs, stop only its own server."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import psutil
import yaml

ROOT = Path(os.environ['EXPERIMENT_ROOT'])
EXP = json.loads((ROOT / 'experiment.json').read_text())
GPU_UUID = EXP['gpu_uuid']
TOOLS = Path(os.environ['TOOL_REPO']) / 'tools/drama3_bench'
REPO = Path(os.environ['OMNI_REPO'])


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def command(args):
    return subprocess.check_output(args, text=True).strip()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gpu():
    return command(['nvidia-smi', '-i', GPU_UUID, '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'])


def gpu_apps():
    rows = command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'])
    return [line for line in rows.splitlines() if line.startswith(GPU_UUID)]


def port_free():
    sock = socket.socket()
    try:
        return sock.connect_ex(('127.0.0.1', 8091)) != 0
    finally:
        sock.close()


def check_yaml():
    a = (ROOT / 'default.yaml').read_text()
    b = (ROOT / 'fp32_state.yaml').read_text()
    assert a.count('mamba_ssm_cache_dtype: bfloat16') == 1
    assert a.replace('mamba_ssm_cache_dtype: bfloat16', 'mamba_ssm_cache_dtype: float32') == b
    original = yaml.safe_load(a)
    modified = yaml.safe_load(b)
    original['stages'][0]['mamba_ssm_cache_dtype'] = 'float32'
    assert original == modified


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['default', 'fp32_state'])
    args = ap.parse_args()
    check_yaml()
    assert command(['git', '-C', str(REPO), 'rev-parse', 'HEAD']) == EXP['server_commit']
    assert command(['git', '-C', str(TOOLS.parent.parent), 'rev-parse', 'HEAD']) == EXP['tool_commit']
    assert len(Path(EXP['codes']).read_text().splitlines()) == 1088
    assert not gpu_apps(), 'GPU 1 has a compute process; refusing to start'
    assert int(gpu().split(',')[3].strip()) < 100, 'GPU 1 memory is not at idle baseline'
    assert port_free(), 'Port 8091 is occupied; refusing to start'
    run = ROOT / args.mode
    run.mkdir(exist_ok=True)
    assert not any((run / p).exists() for p in ['server.pid', 'summary.json', 'server.log']), 'Run already attempted; preserve it'
    shutil.copy2(ROOT / f'{args.mode}.yaml', run / 'deploy_config.yaml')
    dtype = yaml.safe_load((run / 'deploy_config.yaml').read_text())['stages'][0]['mamba_ssm_cache_dtype']
    server_args = ['vllm', 'serve', EXP['model'], '--omni', '--served-model-name', 'fishaudio/drama-3', '--host', '127.0.0.1', '--port', '8091', '--deploy-config', str(run / 'deploy_config.yaml'), '--stage-init-timeout', '1800']
    client_args = [sys.executable, str(TOOLS / 'stream_bench.py'), '--codes', EXP['codes'], '--ref-mode', 'codes', '--concurrency', '16', '--num-prompts', '1088', '--seed', '0', '--num-warmups', '8', '--save-audio', '--label', f'quality-{args.mode}', '--meta', f'commit={EXP["server_commit"]}', '--meta', f'tool_commit={EXP["tool_commit"]}', '--meta', f'mamba_ssm_cache_dtype={dtype}', '--meta', f'deploy_sha256={digest(run / "deploy_config.yaml")}', '--out', str(run)]
    env_keys = ['CUDA_DEVICE_ORDER', 'CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'CUDA_PATH', 'PATH', 'LD_LIBRARY_PATH', 'CPATH', 'TMPDIR', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'VLLM_CACHE_ROOT', 'CUDA_CACHE_PATH', 'FLASHINFER_WORKSPACE_BASE', 'TORCH_EXTENSIONS_DIR', 'OMP_NUM_THREADS', 'MAX_JOBS', 'HF_HOME', 'HF_HUB_CACHE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']
    meta = dict(EXP, mode=args.mode, mamba_ssm_cache_dtype=dtype, deploy_sha256=digest(run / 'deploy_config.yaml'), codes_sha256=digest(EXP['codes']), server_argv=server_args, client_argv=client_args, client_cpu_affinity='inherited, not taskset', process_cpu_affinity=psutil.Process().cpu_affinity(), versions={k: importlib.metadata.version(k) for k in ['torch', 'vllm', 'vllm-omni', 'triton', 'transformers']}, environment={k: os.environ.get(k) for k in env_keys}, gpu_before=gpu(), timestamp_start=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    write(run / 'run_metadata.json', meta)
    server = monitor = client = None
    tracked = {}
    clean = {}
    client_rc = None
    failure = None

    def remember():
        if server is None:
            return
        try:
            parent = psutil.Process(server.pid)
            procs = [parent] + parent.children(recursive=True)
            for p in procs:
                try:
                    if p.uids().real == os.getuid():
                        tracked[p.pid] = {'pid': p.pid, 'create_time': p.create_time(), 'name': p.name()}
                except psutil.Error:
                    pass
        except psutil.Error:
            pass
        # MPS daemonizes and may leave the API descendant tree. Its private
        # socket is scoped to this experiment's TMPDIR, UID and physical UUID.
        for p in psutil.process_iter(['pid', 'name', 'uids']):
            try:
                if 'nvidia-cuda-mps' not in (p.info['name'] or '') or p.info['uids'].real != os.getuid():
                    continue
                env = p.environ()
                pipe = env.get('CUDA_MPS_PIPE_DIRECTORY', '')
                if pipe.startswith(os.environ['TMPDIR'] + '/vllm-omni-mps-') and env.get('CUDA_VISIBLE_DEVICES') == GPU_UUID:
                    tracked[p.pid] = {'pid': p.pid, 'create_time': p.create_time(), 'name': p.name(), 'private_mps_pipe': pipe}
            except psutil.Error:
                pass

    def live_owned():
        live = []
        for pid, info in tracked.items():
            try:
                p = psutil.Process(pid)
                if p.create_time() == info['create_time'] and p.uids().real == os.getuid() and p.status() != psutil.STATUS_ZOMBIE:
                    live.append(p)
            except psutil.Error:
                pass
        return live

    def interrupted(signum, _frame):
        raise RuntimeError(f'Interrupted by signal {signum}; cleaning own server')

    for sig in [signal.SIGTERM, signal.SIGHUP, signal.SIGINT]:
        signal.signal(sig, interrupted)
    try:
        with (run / 'gpu.csv').open('w') as g, (run / 'gpu-monitor.stderr').open('w') as ge:
            monitor = subprocess.Popen(['nvidia-smi', '-i', GPU_UUID, '--query-gpu=timestamp,memory.used,clocks.sm,power.draw,temperature.gpu,utilization.gpu', '--format=csv', '-l', '1'], stdout=g, stderr=ge)
        (run / 'gpu-monitor.pid').write_text(f'{monitor.pid}\n')
        started = time.time()
        (run / 't_start').write_text(f'{started}\n')
        with (run / 'server.log').open('w') as log:
            server = subprocess.Popen(server_args, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (run / 'server.pid').write_text(f'{server.pid}\n')
        print(f'{args.mode}: server PID {server.pid}, starting', flush=True)
        deadline = time.monotonic() + 2100
        while time.monotonic() < deadline:
            remember()
            if server.poll() is not None:
                raise RuntimeError(f'Server exited before readiness: {server.returncode}')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8091/health', timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:
                pass
            time.sleep(5)
        else:
            raise TimeoutError('Service did not become healthy within 35 minutes')
        elapsed = time.time() - started
        (run / 'startup_seconds.txt').write_text(f'{elapsed:.3f}\n')
        print(f'{args.mode}: healthy after {elapsed:.3f}s; running C16 × 1088', flush=True)
        with (run / 'client.log').open('w') as log:
            client = subprocess.Popen(client_args, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        (run / 'client.pid').write_text(f'{client.pid}\n')
        client_deadline = time.monotonic() + 2700
        while client.poll() is None:
            remember()
            if server.poll() is not None:
                raise RuntimeError(f'Server exited during client: {server.returncode}')
            if time.monotonic() >= client_deadline:
                raise TimeoutError('Client exceeded 45-minute outer limit')
            time.sleep(5)
        client_rc = client.returncode
        (run / 'client.exit').write_text(f'{client_rc}\n')
        print(f'{args.mode}: client exit {client_rc}', flush=True)
        if (run / 'summary.json').exists():
            summary = json.loads((run / 'summary.json').read_text())
            print(json.dumps({k: summary[k] for k in ['requests', 'succeeded', 'failed', 'duration_s', 'client']}, ensure_ascii=False), flush=True)
        else:
            raise RuntimeError('Client did not create summary.json')
    except BaseException as exc:
        failure = f'{type(exc).__name__}: {exc}'
        write(run / 'failure.json', {'error': failure, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
        print(failure, flush=True)
    finally:
        if client is not None and client.poll() is None:
            client.terminate()
            try:
                client.wait(timeout=30)
            except subprocess.TimeoutExpired:
                client.kill()
                client.wait()
        remember()
        clean['tracked_processes'] = list(tracked.values())
        clean['signals'] = []
        if server is not None and server.poll() is None:
            server.send_signal(signal.SIGINT)
            clean['signals'].append({'pid': server.pid, 'signal': 'SIGINT'})
            try:
                server.wait(timeout=180)
            except subprocess.TimeoutExpired:
                server.terminate()
                clean['signals'].append({'pid': server.pid, 'signal': 'SIGTERM'})
                try:
                    server.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    server.kill()
                    clean['signals'].append({'pid': server.pid, 'signal': 'SIGKILL'})
                    server.wait()
        if server is not None:
            (run / 'server.exit').write_text(f'{server.returncode}\n')
            clean['server_exit'] = server.returncode
        # After the API has had time to clean up, only signal recorded descendants
        # with matching creation times and UID. Never signal another user's job.
        for sig, wait_s in [(signal.SIGTERM, 30), (signal.SIGKILL, 10)]:
            remaining = live_owned()
            if not remaining:
                break
            for proc in remaining:
                try:
                    proc.send_signal(sig)
                    clean['signals'].append({'pid': proc.pid, 'signal': signal.Signals(sig).name})
                except psutil.Error:
                    pass
            psutil.wait_procs(remaining, timeout=wait_s)
        if monitor is not None and monitor.poll() is None:
            monitor.terminate()
            monitor.wait(timeout=10)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if port_free() and not live_owned() and not gpu_apps() and int(gpu().split(',')[3].strip()) < 100:
                break
            time.sleep(2)
        clean.update(gpu_after=gpu(), gpu_compute_apps_after=gpu_apps(), port_free=port_free(), own_processes_remaining=[p.pid for p in live_owned()], timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        clean['clean'] = clean['port_free'] and not clean['own_processes_remaining'] and not clean['gpu_compute_apps_after'] and int(clean['gpu_after'].split(',')[3].strip()) < 100
        write(run / 'cleanup.json', clean)
        print(f'{args.mode}: cleanup={clean["clean"]}, GPU {clean["gpu_after"]}', flush=True)
        lines = (run / 'server.log').read_text(errors='replace').splitlines() if (run / 'server.log').exists() else []
        needles = ['KV cache size', 'Maximum concurrency', 'Fp8 MoE backend', 'GDN prefill kernel', 'Started private MPS', 'mamba_ssm_cache_dtype', 'Traceback', 'ERROR', 'preempt']
        (run / 'key_lines.txt').write_text('\n'.join(line for line in lines if any(n.lower() in line.lower() for n in needles)) + '\n')
    if (run / 'summary.json').exists():
        analyze = [sys.executable, str(TOOLS / 'analyze_audio.py'), str(run), '--codes', EXP['codes'], '--seed-tts-root', EXP['dataset_root'], '--all', '--json', str(run / 'audio_analysis.json')]
        with (run / 'audio_analysis.log').open('w') as log:
            result = subprocess.run(analyze, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        (run / 'audio_analysis.exit').write_text(f'{result.returncode}\n')
        if result.returncode != 0:
            failure = f'analyze_audio failed {result.returncode}'
        else:
            rows = json.loads((run / 'audio_analysis.json').read_text())
            counts = {'rows': len(rows), 'noise_run': sum('noise-run' in r['flags'] for r in rows), 'at_frame_cap': sum('max-frames' in r['flags'] for r in rows)}
            counts['noise_run_pct'] = 100 * counts['noise_run'] / len(rows) if rows else None
            write(run / 'audio_analysis_counts.json', counts)
            print(json.dumps(counts), flush=True)
    return 1 if failure or not clean.get('clean') else (client_rc or 0)


if __name__ == '__main__':
    raise SystemExit(main())
