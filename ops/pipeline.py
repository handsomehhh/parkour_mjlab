"""Detached, restartable five-teacher -> student supervisor (stdlib only)."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--run-root', type=Path, default=ROOT / 'runs/php_4096_20000')
parser.add_argument('--teacher-updates', type=int, default=20000)
parser.add_argument('--student-updates', type=int, default=30000)
parser.add_argument('--teacher-envs', type=int, default=4096)
parser.add_argument('--student-envs', type=int, default=4096)
args = parser.parse_args()
if min(args.teacher_updates, args.student_updates, args.teacher_envs, args.student_envs) < 1:
    parser.error('Update targets and environment counts must be positive')
RUNS = args.run_root.resolve()
SKILLS = ['Locomotion', 'Low-Step', 'High-Step', 'Low-Climb', 'High-Climb']
STUDENT_GPUS = '0,1,2,3,4'
PYTHON = str(ROOT / '.venv/bin/python')
WORKER = str(ROOT / 'ops/worker.py')
RUNS.mkdir(parents=True, exist_ok=True)
STATE = dict(supervisor_pid=os.getpid(), phase='teachers', jobs={})
PROCS = {}


def write_state():
    STATE['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    tmp = RUNS / 'status.json.tmp'
    tmp.write_text(json.dumps(STATE, indent=2) + '\n')
    tmp.replace(RUNS / 'status.json')


def checkpoint(folder):
    files = list(folder.glob('model_*.pt'))
    return max(files, key=lambda x: int(x.stem.split('_')[-1])) if files else None


def launch(name, gpu, target, num_envs, teachers=None):
    folder = RUNS / name
    folder.mkdir(exist_ok=True)
    job = STATE['jobs'].setdefault(name, dict(attempts=0, target=target, gpu=gpu))
    marker = folder / 'completed.json'
    if marker.exists() and json.loads(marker.read_text())['completed_updates'] == target:
        job.update(state='complete', completed_updates=target)
        return
    job['attempts'] += 1
    gpu_count = len(str(gpu).split(','))
    command = [PYTHON, '-u']
    if teachers:
        command += ['-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                    f'--nproc_per_node={gpu_count}', '--max_restarts=0']
    command += [WORKER, '--task', 'Unitree-G1-PHP-Student' if teachers
                else f'Unitree-G1-PHP-{name}-Teacher', '--log-dir', str(folder),
               '--target', str(target), '--num-envs', str(num_envs)]
    ckpt = checkpoint(folder)
    if ckpt:
        command += ['--resume', str(ckpt)]
    if teachers:
        command += ['--teachers', *teachers]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_GL='egl',
               MUJOCO_EGL_DEVICE_ID='0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               PYTHONUNBUFFERED='1', WANDB_MODE='disabled', PHP_ONLY='1')
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(key, None)
    # Never rely on the temporary installation proxy for running training.
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(key, None)
    log = open(folder / f'attempt_{job["attempts"]}.log', 'a', buffering=1)
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    log.close()
    PROCS[name] = process
    job.update(pid=process.pid, state='running', num_envs=num_envs,
               gpu_count=gpu_count, total_envs=num_envs * gpu_count, command=command,
               log=str(folder / f'attempt_{job["attempts"]}.log'), started_at=time.time())
    print(time.ctime(), 'START', name, process.pid, flush=True)
    write_state()


def wait_jobs(specs, teachers=None):
    for name, gpu, target, num_envs in specs:
        launch(name, gpu, target, num_envs, teachers)
    while True:
        complete = 0
        for name, gpu, target, num_envs in specs:
            job = STATE['jobs'][name]
            folder = RUNS / name
            ckpt = checkpoint(folder)
            job['latest_checkpoint'] = str(ckpt) if ckpt else None
            job['saved_updates'] = int(ckpt.stem.split('_')[-1]) + 1 if ckpt else 0
            process = PROCS.get(name)
            if process is not None and process.poll() is None:
                log = Path(job['log'])
                last_progress = max(job['started_at'], log.stat().st_mtime)
                if time.time() - last_progress > 1800:
                    print(time.ctime(), 'NO LOG PROGRESS FOR 30 MINUTES', name, flush=True)
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
            if process is None or process.poll() is not None:
                if (folder / 'completed.json').exists():
                    marker = json.loads((folder / 'completed.json').read_text())
                    if marker['completed_updates'] == target and (process is None or process.returncode == 0):
                        job.update(state='complete', completed_updates=target)
                        complete += 1
                        continue
                if job['attempts'] >= 5:
                    job.update(state='failed', exit_code=process.returncode if process else None)
                    STATE['phase'] = 'failed'
                    write_state()
                    raise RuntimeError(f'{name} exhausted retries; see {job["log"]}')
                print(time.ctime(), 'RESTART', name, flush=True)
                time.sleep(10)
                launch(name, gpu, target, num_envs, teachers)
        write_state()
        if complete == len(specs):
            return
        time.sleep(30)


def stop(sig, frame):
    STATE['phase'] = 'stopped'
    for process in PROCS.values():
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    write_state()
    sys.exit(128 + sig)


def main():
    lock = open(RUNS / 'supervisor.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Persist attempt counters over a supervisor restart.
    if (RUNS / 'status.json').exists():
        previous = json.loads((RUNS / 'status.json').read_text())
        for name, job in previous.get('jobs', {}).items():
            pid = job.get('pid')
            if job.get('state') == 'running' and pid and Path(f'/proc/{pid}').exists():
                raise RuntimeError(f'Worker {pid} is still running; stop it before restarting supervisor')
        STATE['jobs'] = previous.get('jobs', {})
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    wait_jobs([(s, gpu, args.teacher_updates, args.teacher_envs) for gpu, s in enumerate(SKILLS)])
    teachers = [str(RUNS / s / f'model_{args.teacher_updates - 1}.pt') for s in SKILLS]
    (RUNS / 'teacher_checkpoints.json').write_text(json.dumps(teachers, indent=2) + '\n')
    STATE['phase'] = 'student'
    write_state()
    wait_jobs([('Student', STUDENT_GPUS, args.student_updates, args.student_envs)], teachers)
    STATE['phase'] = 'complete'
    write_state()


if __name__ == '__main__':
    main()
