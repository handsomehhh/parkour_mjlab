# PHP teacher and student training

This directory contains the restartable training pipeline used for five Unitree G1
PHP teachers followed by one distributed student. It uses the repository's task
definitions and RSL-RL algorithm without modifying upstream source files.

The teachers are Locomotion, Low-Step, High-Step, Low-Climb, and High-Climb. They
run concurrently on GPUs 0–4, with 4096 environments and 20,000 updates each.
Each teacher produces `model_19999.pt` after its final update; RSL-RL checkpoint
indices start at zero. Checkpoints are saved every 200 updates.

After all five teachers finish and their final checkpoints pass the finite-parameter
check, the supervisor launches one student policy using PyTorch distributed training
on GPUs 0–4. Each rank has 4096 environments (20,480 across five GPUs). The student
runs 30,000 synchronized updates and saves checkpoints every 500 updates. Its
original PPO/DAgger settings remain in the task configuration. The simulation uses
MJLab's default 50 CCD iterations; 500 exceeded H100 memory at 4096 environments.

Create a Python 3.11 environment and install the pinned runtime packages:

```bash
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r ops/requirements.freeze.txt
uv pip install --python .venv/bin/python -e . --no-deps
```

The task's motion archives are separate data assets and are not committed to Git.
Place the extracted official PHP motion bundles in
`src/assets/motions/php_motion/<bundle-name>/` before starting training. See the
project `README.md` for task usage and dataset details.

Start or resume the pipeline from the repository root:

```bash
mkdir -p runs/php_4096_20000
nohup .venv/bin/python -u ops/pipeline.py >> runs/php_4096_20000/supervisor.log 2>&1 < /dev/null &
.venv/bin/python ops/status.py
```

The supervisor holds a lock to prevent duplicate starts. It restarts failed workers
from their newest complete checkpoint, up to five attempts. It starts the student
only after all five teachers have valid final checkpoints. Send SIGTERM to the
`supervisor_pid` in `runs/php_4096_20000/status.json` to stop it and its workers.
The project saves logs and checkpoints under `runs/php_4096_20000/`.

The distributed student path was checked with a five-GPU, three-update smoke run:
all five ranks loaded every teacher, synchronized parameters, and produced a finite
student checkpoint. Environment counts and GPU assignments appear in `status.py`.
