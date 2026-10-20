"""External profiler trigger; benchmark source and request construction stay unchanged.
Run under client taskset; service must already be warm/healthy and have bounded
per-stage torch profilers configured. Profile results are NOT baseline timings.
"""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

TOOL = Path(os.environ.get('TOOL_REPO', '${PERSONAL_WORKDIR}/github-smoke-audio-20261020')) / 'tools/drama3_bench/stream_bench.py'
spec = importlib.util.spec_from_file_location('profile_stream_bench', TOOL)
bench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bench
spec.loader.exec_module(bench)
original = bench.run_phase
calls = 0

async def phase(session, url, requests, concurrency, args, audio_dir):
    global calls
    calls += 1
    timed = calls == (2 if args.num_warmups else 1)
    if not timed:
        return await original(session, url, requests, concurrency, args, audio_dir)
    out = Path(args.out)
    events = []
    stop = asyncio.Event()
    async def api(action):
        record = {'action': action, 'wall_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns()}
        async with session.post(args.base_url.rstrip('/') + '/' + action, json={'stages': [0, 1]}) as response:
            record.update(status=response.status, body=await response.text())
            events.append(record)
            response.raise_for_status()
    async def capture():
        try:
            await asyncio.wait_for(stop.wait(), timeout=float(os.environ.get('PROFILE_START_AFTER_S', '20')))
            raise RuntimeError('Workload finished before profiler start; increase num-prompts')
        except asyncio.TimeoutError:
            pass
        await api('start_profile')
        try:
            await asyncio.wait_for(stop.wait(), timeout=float(os.environ.get('PROFILE_DURATION_S', '5')))
        except asyncio.TimeoutError:
            pass
        finally:
            await api('stop_profile')
    task = asyncio.create_task(capture())
    try:
        result = await original(session, url, requests, concurrency, args, audio_dir)
    finally:
        stop.set()
        try:
            await task
        finally:
            (out / 'profile_api.json').write_text(json.dumps(events, indent=2) + '\n')
    return result

bench.run_phase = phase
if __name__ == '__main__':
    raise SystemExit(bench.main())
