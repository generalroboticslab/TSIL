"""Round-trip shipped CLI configs and reject unsafe/unsupported launch options."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.training.args import Args
from projects.TSIL.run import evaluation_command, render, training_arguments
from projects.TSIL.training import parse_args


def parsed(arguments):
    # Parse real training flags, without creating run directories or using CUDA.
    arguments = dict(arguments, saving=False, cuda=False)
    with contextlib.redirect_stdout(io.StringIO()):
        return parse_args(render(arguments))


def main():
    count = 0
    for path in sorted((ROOT / "configs").glob("*/*.json")):
        benchmark, task = path.parent.name, int(path.stem[1:])
        for smoke in (False, True):
            arguments = training_arguments(benchmark, task, smoke=smoke)
            args = parsed(arguments)
            assert arguments["num_steps"] == Args.num_steps
            assert args.num_minibatches >= 2
            assert args.sil_freeze_replay_gap
            assert args.sil_replay_normalization_scope == "slice"
            assert args.gradient_clip_scope == "joint"
            assert args.update_epochs == args.sil_replay_epochs == 5
            if smoke:
                assert args.total_iters == 12
                assert args.total_timesteps == 12 * args.batch_size
            count += 1
    assert count == 60
    # Exercise the outer CLI override; parser-only smoke cases hide string counts.
    for benchmark in ("metaworld", "minigrid"):
        result = subprocess.run(
            [sys.executable, "-m", "projects.TSIL.run", "train", "--benchmark", benchmark,
             "--task", "1", "--iterations", "40", "--dry-run"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        plan = json.loads(result.stdout)["arguments"]
        assert plan["total_iters"] == 40
        assert plan["total_timesteps"] == 40 * plan["num_steps"] * sum(plan["task_counts"])
    one_batch = training_arguments("minigrid", 10, smoke=True)
    one_batch["minibatch_size"] = one_batch["num_steps"] * sum(one_batch["task_counts"])
    try:
        parsed(one_batch)
    except ValueError as exc:
        assert "at least two minibatches" in str(exc)
    else:
        raise AssertionError("A validation-only configuration must not train silently")
    with tempfile.TemporaryDirectory() as temp:
        run = Path(temp)
        (run / "config.json").write_text(json.dumps({"benchmark_backend": "metaworld", "tasks": [1]}))
        try:
            evaluation_command(run, 32, "best_suc_tail", "cuda")
        except FileNotFoundError as exc:
            assert "--checkpoint last" in str(exc)
        else:
            raise AssertionError("Missing checkpoint accepted")
        (run / "checkpoints").mkdir()
        (run / "checkpoints/eps_last").touch()
        try:
            evaluation_command(run, 32, "last", "cpu")
        except ValueError as exc:
            assert "requires CUDA" in str(exc)
        else:
            raise AssertionError("CPU MetaWorld evaluation accepted")
    print("PASS: 60 full/smoke configurations parsed; invalid minibatch/device/checkpoint rejected")


if __name__ == "__main__":
    main()
