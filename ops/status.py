import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'runs/php_4096_20000'
state = json.loads((root / 'status.json').read_text())
print('phase:', state['phase'], 'supervisor PID:', state['supervisor_pid'], 'updated:', state['updated_at'])
for name, job in state['jobs'].items():
    log = Path(job.get('log', ''))
    current = job.get('completed_updates') if job['state'] == 'complete' else None
    if log.is_file():
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size - 131072))
            tail = stream.read().decode(errors='replace')
        tail = re.sub(r'\x1b\[[0-9;]*m', '', tail)
        iterations = re.findall(r'Learning iteration\s+(\d+)\s*/\s*(\d+)', tail, re.I)
        if iterations:
            current = int(iterations[-1][0]) + 1
    env_count = str(job.get('num_envs', '?'))
    if job.get('gpu_count', 1) > 1:
        env_count += f'x{job["gpu_count"]}={job["total_envs"]}'
    print(f'{name:12} GPU={job["gpu"]} envs={env_count} PID={job.get("pid")} {job["state"]:9} '
          f'updates={current if current is not None else "initializing"}/{job["target"]} '
          f'saved={job.get("saved_updates", 0)} attempts={job["attempts"]}')
