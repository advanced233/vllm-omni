"""Tiny CUDA graph replay trace coverage probe. Requires exclusive GPU authorization.
It captures before profiler starts, matching the real server's graph lifecycle.
"""
import argparse
import json
from pathlib import Path
import torch
p = argparse.ArgumentParser()
p.add_argument('--out', required=True)
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=False)
x = torch.randn(1024, 1024, device='cuda')
y = torch.randn_like(x)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        z = torch.sin(x @ y)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    z = torch.sin(x @ y)
torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
    for _ in range(3):
        with torch.profiler.record_function('execute_context_0(0)_generation_8(8)'):
            g.replay()
    torch.cuda.synchronize()
prof.export_chrome_trace(str(out / 'trace.json'))
trace = json.loads((out / 'trace.json').read_text())
kernels = [e for e in trace['traceEvents'] if e.get('cat') == 'kernel']
launches = [e for e in trace['traceEvents'] if 'cudaGraphLaunch' in e.get('name', '')]
result = {'kernel_events':len(kernels), 'graph_launch_events':len(launches),
          'kernel_names':sorted(set(e['name'] for e in kernels)),
          'kernel_args_example':[e.get('args') for e in kernels[:3]],
          'runtime_args_example':[e.get('args') for e in launches[:3]],
          'graph_internal_kernels_present':len(kernels) >= 6, 'torch':torch.__version__}
(out / 'probe.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
assert result['graph_internal_kernels_present'], 'Graph-internal CUDA kernel coverage absent'
