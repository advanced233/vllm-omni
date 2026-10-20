"""Add only bounded profiler settings to both existing stage dictionaries."""
import argparse
from pathlib import Path
import yaml
p = argparse.ArgumentParser()
p.add_argument('--source', required=True)
p.add_argument('--out', required=True)
p.add_argument('--trace-dir', required=True)
p.add_argument('--max-iterations', type=int, default=256)
p.add_argument('--backend', choices=['torch', 'cuda'], default='torch')
a = p.parse_args()
src = yaml.safe_load(Path(a.source).read_text())
for stage in src['stages']:
    iteration_limit = 4096 if stage['stage_id'] == 1 else a.max_iterations
    cfg = dict(profiler=a.backend, max_iterations=iteration_limit, delay_iterations=0)
    if a.backend == 'torch':
        cfg.update(torch_profiler_dir=str(Path(a.trace_dir).resolve()), torch_profiler_use_gzip=True,
                   torch_profiler_with_stack=False, torch_profiler_record_shapes=False,
                   torch_profiler_with_memory=False, torch_profiler_with_flops=False)
    stage['profiler_config'] = cfg
Path(a.out).write_text(yaml.safe_dump(src, sort_keys=False))
original = yaml.safe_load(Path(a.source).read_text())
check = yaml.safe_load(Path(a.out).read_text())
for stage in check['stages']:
    del stage['profiler_config']
assert original == check
