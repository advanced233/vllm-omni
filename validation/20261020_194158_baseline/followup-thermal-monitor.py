"""Read-only named clock-throttle reasons; no GPU setting changes."""
import csv,json,subprocess,time
from pathlib import Path
root=Path('${NVME_WORKDIR}/baseline/20261020_194158/followup')
fields='timestamp,clocks.sm,temperature.gpu,utilization.gpu,power.draw,clocks_throttle_reasons.active,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_power_brake_slowdown'
while not (root/'runner.exit').exists():
    runs=sorted((p.parent for p in root.glob('c*/r*/api.pid') if not (p.parent/'supervisor_cleanup.json').exists()),key=lambda p:(p/'api.pid').stat().st_mtime)
    if runs:
        run=runs[-1];out=run/'thermal_reasons.csv'
        result=subprocess.run(['nvidia-smi','-i','1','--query-gpu='+fields,'--format=csv,noheader'],capture_output=True,text=True,timeout=10)
        if result.returncode==0:
            with out.open('a') as f:
                if out.stat().st_size==0:f.write(fields+'\n')
                f.write(result.stdout)
        else:
            with (root/'thermal-monitor-errors.log').open('a') as f:f.write(result.stderr)
    time.sleep(10)
