"""Generate paper-style figures and tables from explicitly selected saved runs."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.plotting.train_data import _load_local_plot_metric_history
from projects.TSIL.ckpt_layout import _run_folder_complete
from projects.TSIL.figures import main_method_results as figures
from projects.TSIL.figures.benchmark_curves import plot_mt1_training_grid
from projects.TSIL.reports import paper_metrics


def load_runs(paths, labels=None):
    """Use explicit runs so reruns and different protocols are never silently selected."""
    if labels and len(labels) != len(paths):
        raise ValueError("--labels must contain one label per --run-dirs entry")
    records = []
    seen = set()
    for index, path in enumerate(paths):
        path = path.resolve()
        if path in seen:
            raise ValueError(f"Duplicate run: {path}")
        seen.add(path)
        config = json.loads((path / "config.json").read_text())
        if len(config["tasks"]) != 1:
            raise ValueError(f"Expected a single-task run: {path}")
        if not _run_folder_complete(str(path)):
            raise ValueError(f"Training is incomplete or its history is missing: {path}")
        label = labels[index] if labels else config["method_name"]
        records.append((path, config, label))
    backends = {config.get("benchmark_backend", "metaworld") for _, config, _ in records}
    if len(backends) != 1:
        raise ValueError("Generate MetaWorld and MiniGrid reports separately")
    # These switches need distinct labels when several variants share a method name.
    if not labels:
        protocols = {}
        keys = ("sil_freeze_replay_gap", "sil_replay_normalization_scope", "gradient_clip_scope",
                "policy_grad_noise_mode", "policy_grad_noise_scale", "dense_reward_dropout_p",
                "clip_coef", "learning_rate", "minigrid_suite")
        for _, config, label in records:
            signature = tuple(str(config.get(key)) for key in keys)
            if label in protocols and protocols[label] != signature:
                raise ValueError("Different protocols share a method name; supply distinct --labels")
            protocols[label] = signature
    seen_seeds = set()
    task_sets = {}
    for path, config, label in records:
        task, seed = int(config["tasks"][0]), int(config["seed"])
        key = (label, task, seed)
        if key in seen_seeds:
            raise ValueError(f"Duplicate task/seed for label {label}: T{task}, seed {seed}")
        seen_seeds.add(key)
        task_sets.setdefault(label, set()).add(task)
    if len({tuple(sorted(tasks)) for tasks in task_sets.values()}) > 1:
        raise ValueError("Methods must cover the same tasks; select a matched task set")
    return records, backends.pop()


def evaluation_rows(records, checkpoint):
    """Read the selected checkpoint's evaluation; never substitute another checkpoint."""
    rows = []
    for run, config, label in records:
        task = int(config["tasks"][0])
        if config.get("benchmark_backend") == "minigrid":
            filename = "minigrid_evaluation.json" if checkpoint == "last" else f"minigrid_evaluation_{checkpoint}.json"
            path = run / filename
            if not path.exists():
                continue
            evaluation = json.loads(path.read_text())
            if evaluation.get("checkpoint") != checkpoint:
                raise ValueError(f"Evaluation checkpoint mismatch: {path}")
            result = evaluation["per_task"][str(task)]
            rows.append(dict(method=label, task_id=task, seed=config["seed"],
                             checkpoint=checkpoint, success_rate=result["success_rate"],
                             episodes=len(result["episodes"]), source=str(path)))
        else:
            paths = sorted((run / "evaluation").glob(f"**/*_EVAL_{checkpoint}/data.csv"))
            if len(paths) > 1:
                raise ValueError(f"Multiple evaluations for {run}; select one before reporting")
            if not paths:
                continue
            with paths[0].open() as stream:
                result = list(csv.DictReader(stream))[-1]
            rows.append(dict(method=label, task_id=task, seed=config["seed"],
                             checkpoint=checkpoint, success_rate=float(result["success_rate"]),
                             episodes=int(float(result["num_episodes"])), source=str(paths[0])))
    return rows


def generate(args):
    records, backend = load_runs(args.run_dirs, args.labels)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Use a new output directory to preserve previous reports: {args.output}")
    methods = list(dict.fromkeys(label for _, _, label in records))
    tasks = sorted({int(config["tasks"][0]) for _, config, _ in records})
    runs = {method: {task: [] for task in tasks} for method in methods}
    for run, config, label in records:
        runs[label][int(config["tasks"][0])].append(str(run))
    args.output.mkdir(parents=True, exist_ok=True)
    settings = SimpleNamespace(methods=methods, legends=methods, display_labels=methods,
                               task_ids=tasks, x_key="misc/steps", tail_frac=0.1,
                               threshold=0.8, bar_error="sem", format="pdf",
                               time_unit="steps" if backend == "minigrid" else "s",
                               figure_dir=args.output / "figures")
    settings.figure_dir.mkdir()
    colors = figures._method_colors(methods)
    rows = figures._collect_main_rows(runs, tasks, methods, settings.x_key, settings.tail_frac)
    if not rows:
        raise ValueError("No success histories available in the selected runs")
    figures.plot_main_results(settings, runs, rows, colors, methods)
    # Reuse the existing table definitions and record the exact run selection.
    table_runs = {method: {task: {method: paths} for task, paths in grouped.items()}
                  for method, grouped in runs.items()}
    table_rows = paper_metrics._collect_run_rows(settings, table_runs)
    summary = paper_metrics._summary_rows(table_rows, methods, methods, settings.threshold)
    paper_metrics.write_tables(args.output / "tables", summary, table_rows)
    from projects.TSIL.plot import _aggregate_runs
    for metric, name, ylabel in [
        ("reward/success", "task_success", "Training success"),
        ("signal/success_eps_time", "task_completion_time", f"Successful episode time ({settings.time_unit})"),
        ("misc/success_episodes", "task_success_count", "Successful episodes"),
    ]:
        grid = {task: {} for task in tasks}
        for task in tasks:
            for method in methods:
                curves = [_load_local_plot_metric_history(run, settings.x_key, metric)
                          for run in runs[method][task]]
                grid[task][method] = _aggregate_runs([c for c in curves if c is not None], 500, 1, f"T{task}/{method}")
        plot_mt1_training_grid(grid, tasks, methods, method_colors=colors,
                               method_labels=methods, x_label="Environment steps", y_label=ylabel,
                               n_cols=min(5, len(tasks)), save_path=str(settings.figure_dir / f"{name}.pdf"), show=False)
    evaluated = evaluation_rows(records, args.checkpoint)
    if evaluated:
        paper_metrics._write_csv(args.output / "tables/evaluation_runs.csv", evaluated)
        paper_metrics._write_markdown(args.output / "tables/evaluation_runs.md", evaluated)
    # A comparative evaluation figure requires every selected checkpoint evaluation.
    # Raw available measurements are still exported when coverage is incomplete.
    if evaluated and len(evaluated) == len(records):
        fig, ax = plt.subplots(figsize=(max(5, len(methods)), 3.5))
        means, errors, counts = figures._bar_values(evaluated, methods, "success_rate", "sem")
        figures._plot_bars(ax, methods, means, errors, colors, "Success rate", "Evaluation success", ylim=(0, 1.05))
        fig.tight_layout()
        fig.savefig(settings.figure_dir / "evaluation_success.pdf", bbox_inches="tight")
        plt.close(fig)
    evaluated_keys = {(row["method"], row["task_id"], row["seed"]) for row in evaluated}
    missing_evaluations = [dict(path=str(p), label=l, task=int(c["tasks"][0]), seed=c["seed"])
                           for p, c, l in records
                           if (l, int(c["tasks"][0]), c["seed"]) not in evaluated_keys]
    manifest = dict(benchmark=backend, checkpoint=args.checkpoint,
                    runs=[dict(path=str(p), task=int(c["tasks"][0]), seed=c["seed"], label=l)
                          for p, c, l in records],
                    evaluations_found=len(evaluated), evaluations_missing=len(records)-len(evaluated),
                    missing_evaluations=missing_evaluations,
                    aggregation="mean across seeds within task, then equal mean across tasks",
                    x_axis="environment steps", time_unit=settings.time_unit,
                    evaluation_figure_complete=not missing_evaluations,
                    note="Reports summarize the selected runs; missing evaluations are not assigned zero success.")
    (args.output / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote figures and tables for {len(records)} runs to {args.output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", help="One label per run; reuse a label to aggregate seeds")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=["best_suc_tail", "last"], default="best_suc_tail")
    generate(parser.parse_args(argv))


if __name__ == "__main__":
    main()
