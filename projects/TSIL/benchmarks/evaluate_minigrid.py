"""Fresh native-reward evaluation of a TSIL MiniGrid final checkpoint."""

import argparse
from dataclasses import fields
import json
from pathlib import Path

# Match the training stack's mandatory Isaac Gym-before-PyTorch import order.
try:
    import isaacgym  # noqa: F401
except ModuleNotFoundError:
    pass
import numpy as np
import torch

from projects.TSIL.training import TSILArgs
from projects.TSIL.benchmarks.minigrid import TSILMiniGridEnv, MiniGridAgent


def evaluate(run_dir, episodes_per_task=32, seed=1042, device="cuda", checkpoint='best_suc_tail', episodes_per_config=4, exact_episodes=None):
    config = json.loads((run_dir / "config.json").read_text())
    if config.get('minigrid_config_bank'):
        from projects.TSIL.benchmarks.evaluate_configurations import evaluate_configurations
        return evaluate_configurations(run_dir, checkpoint, episodes_per_config, seed, device, total_episodes=exact_episodes)
    if checkpoint != 'last':
        raise ValueError('Legacy runs lack checkpoint-specific target metadata; use last')
    # Configs written before full observations existed used the partial CNN.
    config.setdefault("minigrid_observation", "partial")
    config.setdefault("minigrid_reward", "native_bonus")
    declared = {field.name for field in fields(TSILArgs)}
    args = TSILArgs(**{key: value for key, value in config.items() if key in declared})
    args.task_counts = [1] * len(args.tasks)
    args.seed = seed
    args.deterministic = False
    env = TSILMiniGridEnv(args, device=device)
    state = json.loads((run_dir / "minigrid_state.json").read_text())
    for index, task in enumerate(args.tasks):
        env.obs_range["time2end"][1][index] = state["deadline_by_task"][str(task)]
    try:
        agent = MiniGridAgent(env, args, device=device)
        agent.load_checkpoint(ckpt_path=str(run_dir / "checkpoints/eps_last"), map_location=device, strict=True)
        agent.set_mode("eval")
        torch.manual_seed(seed)
        np.random.seed(seed)
        obs = env.reset()["obs"]
        records = {task: [] for task in args.tasks}
        returns = np.zeros(env.num_envs)
        lengths = np.zeros(env.num_envs, dtype=np.int64)
        with torch.no_grad():
            for _ in range(episodes_per_task * (int(env.episode_limits.max().item()) + 1) + 2):
                valid = ~env._needs_reset.copy()
                action, _ = agent.get_action_and_value(obs, action_only=True, update=False)
                observation, _, done, info = env.step(action)
                obs = observation["obs"]
                returns += info["native_reward"].cpu().numpy()
                lengths += valid.astype(np.int64)
                for index in done.nonzero(as_tuple=True)[0].cpu().tolist():
                    task = args.tasks[index]
                    if len(records[task]) < episodes_per_task:
                        records[task].append(dict(native_return=float(returns[index]),
                            success=bool(info["success"][index].item()), length=int(lengths[index])))
                    returns[index] = lengths[index] = 0
                if all(len(rows) == episodes_per_task for rows in records.values()):
                    break
        if not all(len(rows) == episodes_per_task for rows in records.values()):
            raise RuntimeError("Evaluation did not finish the requested episodes")
        per_task = {}
        for task, rows in records.items():
            successful_lengths = [row["length"] for row in rows if row["success"]]
            per_task[str(task)] = dict(
                success_rate=float(np.mean([row["success"] for row in rows])),
                mean_return=float(np.mean([row["native_return"] for row in rows])),
                mean_episode_length=float(np.mean([row["length"] for row in rows])),
                successful_episode_length=float(np.mean(successful_lengths)) if successful_lengths else None,
                episodes=rows,
            )
        return dict(seed=seed, stochastic=True, episodes_per_task=episodes_per_task,
            observation_mode=env.minigrid_observation, image_shape=env.image_shape,
            reward_mode=args.minigrid_reward,
            success_rate=float(np.mean([row["success_rate"] for row in per_task.values()])),
            mean_return=float(np.mean([row["mean_return"] for row in per_task.values()])),
            per_task=per_task, deadline_by_task=state["deadline_by_task"])
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1042)
    parser.add_argument("--device", default="cuda")
    parser.add_argument('--checkpoint', default='best_suc_tail')
    parser.add_argument('--exact-episodes', type=int, default=2000,
                        help='Total episodes per task for configuration-bank runs (default: 2000); overrides episodes-per-config')
    parser.add_argument('--episodes-per-config', type=int, default=4,
                        help='Compatibility option; exact-episodes takes precedence for configuration-bank runs')
    args = parser.parse_args()
    if args.episodes_per_task <= 0:
        parser.error("episodes-per-task must be positive")
    result = evaluate(args.run_dir, args.episodes_per_task, args.seed, args.device,
                      args.checkpoint, args.episodes_per_config, args.exact_episodes)
    path = args.run_dir / ('minigrid_evaluation.json' if args.checkpoint == 'last'
                           else f'minigrid_evaluation_{args.checkpoint}.json')
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Fresh success={result['success_rate']:.4f}; native return={result['mean_return']:.4f}; {path}")


if __name__ == "__main__":
    main()
