"""Run one PHP policy to an absolute number of completed PPO updates."""
import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['PHP_ONLY'] = '1'
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True)
    p.add_argument('--log-dir', type=Path, required=True)
    p.add_argument('--target', type=int, required=True)
    p.add_argument('--num-envs', type=int, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--teachers', nargs=5)
    args = p.parse_args()
    import torch
    import torch.distributed as dist
    import mjlab.tasks
    import src.tasks
    import scripts.train as train
    from mjlab.tasks.registry import load_runner_cls
    from mjlab.rl import MjlabOnPolicyRunner
    torch.set_num_threads(4)
    cfg = train.TrainConfig.from_task(args.task)
    cfg.env.scene.num_envs = args.num_envs
    # 4096 worlds need the standard MuJoCo CCD scratch size to fit on an 80 GiB H100.
    cfg.env.sim.mujoco.ccd_iterations = 50
    cfg.agent.max_iterations = args.target
    cfg.agent.save_interval = 200 if not args.teachers else 500
    cfg.agent.logger = 'tensorboard'
    cfg.agent.upload_model = False
    cfg.agent.run_name = args.log_dir.name
    if args.teachers:
        cfg.agent.teacher_checkpoint_files = tuple(args.teachers)
    if args.resume:
        cfg.agent.resume = True
        cfg.agent.load_run = re.escape(args.log_dir.name) + '$'
        cfg.agent.load_checkpoint = re.escape(args.resume.name) + '$'
    base = load_runner_cls(args.task) or MjlabOnPolicyRunner
    completion = {}

    class AbsoluteRunner(base):
        def save(self, path, infos=None):
            path = Path(path)
            temporary = path.with_suffix(path.suffix + '.partial')
            super().save(str(temporary), infos)
            temporary.replace(path)

        def load(self, *a, **kw):
            result = super().load(*a, **kw)
            if not args.teachers:
                # RSL-RL teacher stores the last completed zero-based index.
                self.current_learning_iteration += 1
            return result

        def learn(self, num_learning_iterations, init_at_random_ep_len=False):
            remaining = args.target - self.current_learning_iteration
            if remaining > 0:
                super().learn(remaining, init_at_random_ep_len)
            if dist.is_initialized():
                dist.barrier()
            if int(os.environ.get('RANK', '0')) != 0:
                return
            final = args.log_dir / f'model_{args.target - 1}.pt'
            if not final.is_file():
                raise RuntimeError(f'Missing final checkpoint: {final}')
            checkpoint = torch.load(final, map_location='cpu', weights_only=False)
            if checkpoint['iter'] != args.target - 1:
                raise RuntimeError('Final iteration mismatch')
            for name in ('actor_state_dict', 'critic_state_dict'):
                for key, value in checkpoint[name].items():
                    if torch.is_tensor(value) and not torch.isfinite(value).all():
                        raise RuntimeError(f'Nonfinite final parameter: {name}.{key}')
            gpu_count = dist.get_world_size() if dist.is_initialized() else 1
            completion.update(task=args.task, completed_updates=args.target,
                              checkpoint=str(final), num_envs=args.num_envs,
                              gpu_count=gpu_count,
                              total_envs=args.num_envs * gpu_count)

    train.load_runner_cls = lambda _: AbsoluteRunner
    args.log_dir.mkdir(parents=True, exist_ok=True)
    train.run_train(args.task, cfg, args.log_dir)
    if int(os.environ.get('RANK', '0')) != 0:
        return
    tmp = args.log_dir / 'completed.json.tmp'
    tmp.write_text(json.dumps(completion, indent=2) + '\n')
    tmp.replace(args.log_dir / 'completed.json')
    print('[PIPELINE] Validated completion:', completion, flush=True)


if __name__ == '__main__':
    main()
