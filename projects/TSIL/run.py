"""Portable training and evaluation entrypoint for one TSIL task and policy seed."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def render(arguments):
    """Render the shared parser's explicit bool/list syntax without shell quoting."""
    result = []
    for key, value in arguments.items():
        if value is None:
            if key == "target_kl":
                result.append("--target_kl")
            continue
        result.append("--" + key)
        result.extend(str(item) for item in (value if isinstance(value, list) else [value]))
    return result


def training_arguments(benchmark, task, seed=None, minigrid_version="repair-v4", smoke=False):
    """Load task settings and select the requested environment definition."""
    config = ROOT / "configs" / benchmark / f"T{task:02d}.json"
    if not config.is_file():
        raise ValueError(f"No supported {benchmark} task T{task:02d}")
    arguments = json.loads(config.read_text())
    counts = arguments["task_counts"]
    if isinstance(counts, str):
        counts = counts.split()
    arguments["task_counts"] = [int(count) for count in counts]
    # Match the shared parser default when a compact config omits rollout length.
    arguments.setdefault("num_steps", 32)
    arguments["seed"] = seed if seed is not None else (48 if benchmark == "metaworld" else 42)
    if benchmark == "minigrid":
        repaired = minigrid_version == "repair-v4" and task in {10, 11}
        bank = "mt15_repair_v4_seed42.json" if repaired else "mt15_fixed_seed42.json"
        arguments["minigrid_config_bank"] = str(ROOT / "projects/TSIL/benchmarks/banks" / bank)
        arguments["minigrid_suite"] = "mt15_repair_v4" if repaired else "mt15_fixed"
        arguments["task_name"] = "MiniGridMT15_REPAIR_V4" if repaired else "MiniGridMT15_FIXED"
    if smoke:
        arguments["total_iters"] = 12
        if benchmark == "metaworld":
            arguments.update(task_counts=[640], minibatch_size=2048, sil_batch_size=2048)
    if arguments.get("total_iters") is not None:
        arguments["total_timesteps"] = (arguments["total_iters"] * arguments["num_steps"]
                                        * sum(arguments["task_counts"]))
    return arguments


def evaluation_command(run, episodes, checkpoint, device):
    config = json.loads((run / "config.json").read_text())
    checkpoint_path = run / "checkpoints" / ("eps_" + checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint {checkpoint_path}; short smoke runs may only have --checkpoint last")
    if config["benchmark_backend"] != "minigrid" and device == "cpu":
        raise ValueError("MetaWorld evaluation requires CUDA/Isaac Gym")
    if config["benchmark_backend"] == "minigrid":
        return [sys.executable, "-u", "-m", "projects.TSIL.benchmarks.evaluate_minigrid",
                "--run-dir", str(run), "--checkpoint", checkpoint,
                "--exact-episodes", str(episodes), "--device", device]
    task = int(config["tasks"][0])
    arguments = dict(
        task_name=f"MT15_T{task}", checkpoint=str(run), index_episode=checkpoint,
        num_envs=episodes, target_episodes=episodes, target_success_eps=episodes,
        tasks=[task], task_counts=[episodes], seed=20260921 + task, warmup_episodes=0,
        fixed_configs=True, global_configs=False, apply_noise=False,
        apply_noise_eval=False, deterministic=False, rendering=False,
        graphics_device_id=-1, eval_dir=str(run / "evaluation"),
    )
    return [sys.executable, "-u", "-m", "projects.TSIL.eval", "--saving", "--strict_eval", *render(arguments)]


def runtime_environment(gpu):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", WANDB_MODE="disabled", PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in [ROOT, ROOT / "MTBench", ROOT / "isaacgym/python"])
    env["PATH"] = str(Path(sys.prefix) / "bin") + os.pathsep + env.get("PATH", "")
    env["MAX_JOBS"] = "2"
    env["LD_LIBRARY_PATH"] = str(Path(sys.prefix) / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    train = sub.add_parser("train")
    train.add_argument("--benchmark", choices=["metaworld", "minigrid"], required=True, help="Benchmark to train")
    train.add_argument("--task", type=int, required=True, help="Task ID; trains one task")
    train.add_argument("--seed", type=int, help="Training seed (default: MetaWorld 48, MiniGrid 42)")
    train.add_argument("--minigrid-version", choices=["repair-v4", "historical"], default="repair-v4",
                       help="MiniGrid environment definition (default: repair-v4)")
    train.add_argument("--output", type=Path, default=Path("results"), help="Training output parent (default: results/)")
    train.add_argument("--smoke", action="store_true", help="12 updates; MetaWorld uses 640 environments")
    train.add_argument("--iterations", type=int, help="Override the training update budget")
    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--run-dir", type=Path, required=True, help="Saved run folder containing config.json and checkpoints/")
    evaluate.add_argument("--episodes", type=int, default=2000, help="Total evaluation episodes, at least 32 (default: 2000)")
    evaluate.add_argument("--checkpoint", default="best_suc_tail", choices=["last", "best_suc_tail"],
                          help="Checkpoint to evaluate (default: best_suc_tail)")
    for command in [train, evaluate]:
        command.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0); unused with --device cpu")
        command.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Compute device (default: cuda); CPU is MiniGrid only")
        command.add_argument("--dry-run", action="store_true", help="Print command/settings without launching; eval requires an existing checkpoint")
    options = parser.parse_args()
    if options.operation == "train":
        arguments = training_arguments(options.benchmark, options.task, options.seed, options.minigrid_version, options.smoke)
        if options.iterations is not None:
            if options.iterations <= 0:
                parser.error("--iterations must be positive")
            arguments["total_iters"] = options.iterations
            arguments["total_timesteps"] = (options.iterations * arguments["num_steps"]
                                            * sum(arguments["task_counts"]))
        if options.benchmark == "metaworld" and options.device == "cpu":
            parser.error("MetaWorld requires CUDA/Isaac Gym")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        output = options.output.resolve() / f"{options.benchmark}_T{options.task:02d}_s{arguments['seed']}_{stamp}"
        arguments.update(result_dir=str(output), script_time=stamp, cuda=options.device == "cuda")
        command = [sys.executable, "-u", "-m", "projects.TSIL.train", *render(arguments)]
        plan = dict(benchmark=options.benchmark, minigrid_version=options.minigrid_version,
                    arguments=arguments, command=command, output=str(output))
    else:
        if options.episodes < 32:
            parser.error("Evaluation must cover at least 32 configuration slots")
        command = evaluation_command(options.run_dir.resolve(), options.episodes, options.checkpoint, options.device)
        plan = dict(command=command)
    if options.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if options.operation == "train":
        output.mkdir(parents=True, exist_ok=False)
        (output / "launch.json").write_text(json.dumps(plan, indent=2) + "\n")
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=runtime_environment(options.gpu)))


if __name__ == "__main__":
    main()
