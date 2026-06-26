"""TSIL plotting entrypoint and benchmark data adapter."""

from __future__ import annotations

import argparse
import math
import os
import sys
from importlib import import_module
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from core.common.project_paths import project_train_res_root
from core.plotting.train_data import (
    _load_local_plot_metric_history,
    _success_eps_time_fill_value,
)
from projects.TSIL.ckpt_layout import (
    DEFAULT_EXPERIMENT,
    DEFAULT_TRAINING_STAGE,
    benchmark_task_path_label,
    discover_benchmark_task_methods,
)
from projects.TSIL.figures.benchmark_curves import (
    _selected_x_by_method,
    plot_mt1_success_summary,
    plot_mt1_training_grid,
)


def load_benchmark_curves_by_task_method(
    train_res_dir: str,
    benchmark: str,
    task_ids: List[int],
    train_stage: str = DEFAULT_TRAINING_STAGE,
    methods: Optional[List[str]] = None,
    method_experiments: Optional[List[str]] = None,
    experiment: str = DEFAULT_EXPERIMENT,
    project: str = "TSIL",
    entity: Optional[str] = None,
    metric_key: str = "reward/success",
    x_key: str = "misc/iterations",
    num_interp_points: int = 500,
    smoothing_window: int = 1,
    use_cache: bool = True,
) -> dict:
    """Load benchmark training curves by task and method."""
    from core.plotting.train_data import _load_run_curve

    api = None
    discovered_methods = methods
    experiment_by_method = {}
    if method_experiments is not None:
        if methods is None:
            raise ValueError("method_experiments requires explicit methods")
        if len(method_experiments) != len(methods):
            raise ValueError("method_experiments must match methods length")
        experiment_by_method = dict(zip(methods, method_experiments))
        task_methods = {tid: {} for tid in task_ids}
        for method, method_experiment in zip(methods, method_experiments):
            discovered = discover_benchmark_task_methods(
                train_res_dir,
                benchmark,
                task_ids,
                train_stage,
                [method],
                experiment=method_experiment,
            )
            for tid in task_ids:
                task_methods[tid][method] = discovered.get(tid, {}).get(method, [])
    else:
        task_methods = discover_benchmark_task_methods(
            train_res_dir,
            benchmark,
            task_ids,
            train_stage,
            methods,
            experiment=experiment,
        )
        discovered_methods = (
            list(next(iter(task_methods.values())).keys())
            if task_methods
            else (methods or [])
        )
    grid_data: dict = {"__methods__": discovered_methods}

    for tid in task_ids:
        grid_data[tid] = {}
        for method in discovered_methods:
            task_label = benchmark_task_path_label(
                benchmark,
                experiment_by_method.get(method, experiment),
                train_stage,
                tid,
            )
            run_folders = task_methods.get(tid, {}).get(method, [])
            if not run_folders:
                print(f"  {task_label}/{method}: no run folders found")
                grid_data[tid][method] = None
                continue

            runs_data = []
            for run_folder in run_folders:
                result, api = _load_run_curve(
                    run_folder,
                    project,
                    entity,
                    x_key,
                    metric_key,
                    api,
                    use_cache,
                )
                if result is not None:
                    runs_data.append(result)

            grid_data[tid][method] = _aggregate_runs(
                runs_data,
                num_interp_points,
                smoothing_window,
                f"{task_label}/{method}",
            )

    return grid_data


def load_benchmark_signal_curves_by_task_method(
    train_res_dir: str,
    benchmark: str,
    task_ids: List[int],
    train_stage: str = DEFAULT_TRAINING_STAGE,
    methods: Optional[List[str]] = None,
    method_experiments: Optional[List[str]] = None,
    experiment: str = DEFAULT_EXPERIMENT,
    metric_key: str = "top10pct_Rdense_sr",
    x_key: str = "iteration",
    num_interp_points: int = 500,
    smoothing_window: int = 1,
) -> dict:
    """Load benchmark signal curves by task and method."""
    from core.plotting.train_data import _load_local_signal_history

    discovered_methods = methods
    experiment_by_method = {}
    if method_experiments is not None:
        if methods is None:
            raise ValueError("method_experiments requires explicit methods")
        if len(method_experiments) != len(methods):
            raise ValueError("method_experiments must match methods length")
        experiment_by_method = dict(zip(methods, method_experiments))
        task_methods = {tid: {} for tid in task_ids}
        for method, method_experiment in zip(methods, method_experiments):
            discovered = discover_benchmark_task_methods(
                train_res_dir,
                benchmark,
                task_ids,
                train_stage,
                [method],
                experiment=method_experiment,
            )
            for tid in task_ids:
                task_methods[tid][method] = discovered.get(tid, {}).get(method, [])
    else:
        task_methods = discover_benchmark_task_methods(
            train_res_dir,
            benchmark,
            task_ids,
            train_stage,
            methods,
            experiment=experiment,
        )
        discovered_methods = (
            list(next(iter(task_methods.values())).keys())
            if task_methods
            else (methods or [])
        )
    grid_data: dict = {"__methods__": discovered_methods}

    for tid in task_ids:
        grid_data[tid] = {}
        for method in discovered_methods:
            task_label = benchmark_task_path_label(
                benchmark,
                experiment_by_method.get(method, experiment),
                train_stage,
                tid,
            )
            run_folders = task_methods.get(tid, {}).get(method, [])
            if not run_folders:
                print(f"  {task_label}/{method}: no run folders found")
                grid_data[tid][method] = None
                continue

            runs_data = [
                result
                for result in (
                    _load_local_signal_history(run_folder, metric_key, x_key)
                    for run_folder in run_folders
                )
                if result is not None
            ]
            if not runs_data:
                print(f"  {task_label}/{method}: no local history for {metric_key}")
            grid_data[tid][method] = _aggregate_runs(
                runs_data,
                num_interp_points,
                smoothing_window,
                f"{task_label}/{method}",
            )

    return grid_data


def _aggregate_runs(
    runs_data: list,
    num_interp_points: int,
    smoothing_window: int,
    label: str,
):
    import numpy as np
    from core.plotting.train_data import interpolate_to_common_x, smooth_curve

    if not runs_data:
        return None

    common_x, interp_ys = interpolate_to_common_x(runs_data, num_interp_points)
    if common_x is None:
        return None

    if smoothing_window > 1:
        interp_ys = np.array([smooth_curve(y, smoothing_window) for y in interp_ys])

    mean_y = np.mean(interp_ys, axis=0)
    std_y = (
        np.zeros_like(interp_ys[0])
        if len(interp_ys) <= 1
        else np.std(interp_ys, axis=0, ddof=1)
    )
    print(
        f"  {label}: {len(runs_data)} run(s), "
        f"final={mean_y[-1]:.4f}+/-{std_y[-1]:.4f}"
    )
    return common_x, mean_y, std_y, len(runs_data)

ACTIVE_TASKS = (
    "BenchmarkTrainCurve",
    "BenchmarkTrainCurveSummary",
    "BenchmarkSuccessSpeedPareto",
    "BenchmarkSignalMetricGrid",
    "BenchmarkSignalMetricSummary",
    "LearningSignalMap",
    "LearningSignalSummary",
    "LearningSignalCombined",
    "LearningSignalSurface",
    "PolicyDirectionLandscape",
    "PolicyDirectionLandscapeGroupGrid",
    "SilRevisitMechanism",
    "SilRevisitOutcome",
)


def parse_args():
    parser = argparse.ArgumentParser(description="TSIL public plotting")
    parser.add_argument("--task", type=str, choices=ACTIVE_TASKS, default="BenchmarkTrainCurve")
    parser.add_argument(
        "--task_ids",
        type=str,
        nargs="+",
        default=[],
        help="Task IDs for benchmark plots, e.g. 28 29 or T28 T29",
    )
    parser.add_argument("--methods", type=str, nargs="+", default=[])
    parser.add_argument("--method_experiments", type=str, nargs="+", default=None)
    parser.add_argument("--legends", type=str, nargs="+", default=[])
    parser.add_argument("--colors", type=str, nargs="+", default=[])
    parser.add_argument("--x_key", type=str, default=None)
    parser.add_argument("--y_key", type=str, default=None)
    parser.add_argument(
        "--project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "TSIL"),
        help="Weights & Biases project name for optional curve downloads",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY"),
        help="Optional Weights & Biases entity/team for curve downloads",
    )
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--experiment", type=str, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--training-stage", type=str, default=DEFAULT_TRAINING_STAGE)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--plot_title", type=str, default=None)
    parser.add_argument("--hide_legend", action="store_true")
    parser.add_argument("--refresh_wandb", action="store_true")
    parser.add_argument("--disable_pvalue_analysis", action="store_true")
    parser.add_argument("--signal_history", type=str, nargs="+", default=None)
    parser.add_argument("--signal_summary_history", type=str, nargs="+", default=None)
    parser.add_argument("--signal_summary_legends", type=str, nargs="+", default=None)
    parser.add_argument("--signal_num_bins", type=int, default=12)
    parser.add_argument("--signal_last_frac", type=float, default=1.0)
    parser.add_argument("--signal_summary_last_frac", type=float, default=None)
    parser.add_argument("--signal_combined_cache_path", type=str, default=None)
    parser.add_argument("--signal_combined_refresh_cache", action="store_true")
    parser.add_argument("--signal_min_count", type=int, default=1)
    parser.add_argument("--policy_landscape", type=str, nargs="+", default=None)
    parser.add_argument("--policy_landscape_group_size", type=int, default=4)
    return parser.parse_args()


def _train_res_dir(project_name: str):
    return project_train_res_root(project_name)


def _project_plot_exts(project_name: str):
    if project_name == "TSIL":
        return sys.modules[__name__]
    try:
        return import_module(f"projects.{project_name}.plot")
    except ModuleNotFoundError as exc:
        raise ValueError(
            f"Project '{project_name}' does not provide a benchmark plotting adapter."
        ) from exc


def _require_benchmark(benchmark):
    if not benchmark:
        raise ValueError("--benchmark is required for benchmark plotting.")
    return benchmark


def _benchmark_context(args, project_name: str):
    benchmark = _require_benchmark(args.benchmark)
    return benchmark, args.experiment, _project_plot_exts(project_name)


def _task_ids_from_args(args):
    if not args.task_ids:
        raise ValueError("--task_ids is required for benchmark plotting.")
    return [int(str(task_id).replace("T", "")) for task_id in args.task_ids]


def _method_style_from_args(args):
    return (
        args.methods if args.methods else None,
        args.legends if args.legends else None,
        args.colors if args.colors else None,
    )


def _benchmark_plot_prefix(task_ids):
    return f"BenchmarkMT{len(task_ids)}"


def _default_benchmark_plot_path(
    task_ids,
    y_name: str,
    summary: bool = False,
    project_name: str = "TSIL",
    benchmark: Optional[str] = None,
    experiment: Optional[str] = None,
):
    prefix = _benchmark_plot_prefix(task_ids)
    suite_dir = benchmark or prefix.lower().replace("benchmarkmt", "mt")
    path_parts = [
        _train_res_dir(project_name),
        "plot_res",
        "benchmark",
        suite_dir,
    ]
    if experiment:
        path_parts.append(experiment)
    path_parts.extend(["pdf", f"{prefix}_TrainCurve_{y_name}{'_summary' if summary else ''}.pdf"])
    return os.path.join(*path_parts)


def _signal_metric_label(metric_key: str) -> str:
    metric_label_map = {
        "top10pct_Rdense_sr": "Top 10% Rdense Success Rate",
        "top_10perc_Rdense_sr": "Top 10% Rdense Success Rate",
        "dense_tail_purity_10": "Top 10% Rdense Success Rate",
        "succ_posadv_ratio": "Success Positive-Advantage Ratio",
        "succ_posadv_step_frac": "Success Positive-Advantage Step Fraction",
        "fast_succ_posadv_ratio": "Fast Success Positive-Advantage Ratio",
        "adv_ep_used": "Advantage Episodes Used",
        "episode_signal_count": "Episode Signal Count",
        "success_eps_time": "Success Episode Time",
        "fast_success_rate": "Fast Success Rate",
        "fast_success_count": "Fast Success Count",
        "fast_revisit_gap_steps": "First Fast-Revisit Gap",
        "sil_fast_success_rate": "Fast Success Rate",
        "sil_fast_success_count": "Fast Success Count",
        "sil_revisit_gap_steps": "First Fast-Revisit Gap",
        "sil_revisit_nll_topk_mean": "Fast Top-k Revisit NLL",
        "sil_revisit_logp_topk_mean": "Fast Top-k Revisit Log Probability",
        "sil_supervised_nll_topk_mean": "Positive-Gap Fast-Memory NLL",
        "sil_supervised_logp_topk_mean": "Positive-Gap Fast-Memory Log Probability",
        "sil_supervised_weight_frac": "Positive-Gap Fast-Memory Fraction",
        "sil_archive_best_eps_time": "Best Archived Success Time",
    }
    return metric_label_map.get(metric_key, metric_key)


def _train_curve_metric_label(metric_key: str) -> str:
    metric_label_map = {
        "reward/success": "Success Rate",
        "reward/eps_G": "Discounted Episode Return",
        "reward/eps_sum_rew": "Episode Reward Sum",
        "reward/eps_dense_return": "Dense Episode Return",
        "reward/eps_r": "Episode Reward Sum",
        "signal/success_eps_time": "Success Episode Time",
        "misc/success_episodes": "Cumulative Successful Episodes",
        "train/replay_memory_logp": "Memory Action Log Probability",
        "train/replay_memory_weighted_logp": "Positive-Gap Weighted Memory Log Probability",
        "train/replay_memory_positive_gap_mean": "Memory Positive Gap",
        "train/sil_cos_ppo_memory": "cos(PPO, Memory)",
        "train/sil_cos_sil_memory": "cos(SIL, Memory)",
        "train/sil_cos_joint_memory": "cos(PPO+SIL, Memory)",
        "train/sil_alignment_gain": "SIL Alignment Gain",
        "train/sil_revisit_nll_topk_mean": "Fast Top-k Revisit NLL",
        "train/sil_revisit_logp_topk_mean": "Fast Top-k Revisit Log Probability",
        "train/sil_revisit_reference_count": "Fast Top-k Reference Count",
        "train/sil_supervised_nll_topk_mean": "Positive-Gap Fast-Memory NLL",
        "train/sil_supervised_logp_topk_mean": "Positive-Gap Fast-Memory Log Probability",
        "train/sil_supervised_weight_frac": "Positive-Gap Fast-Memory Fraction",
        "train/sil_archive_best_eps_time": "Best Archived Success Time",
    }
    y_name = metric_key.split("/")[-1]
    return metric_label_map.get(
        metric_key,
        _signal_metric_label(y_name) if metric_key.startswith("signal/") else y_name,
    )


def _train_curve_x_label(x_key: str) -> str:
    return {
        "misc/interaction_time": "Interaction Time (s)",
        "interaction_time": "Interaction Time (s)",
        "misc/iterations": "Iterations",
        "misc/steps": "Frames",
        "misc/episodes": "Episodes",
    }.get(x_key, x_key)


def _unit_interval_ylim(metric_key: str):
    normalized_key = metric_key.lower()
    unit_interval_keys = {
        "reward/success",
        "reward/success_rate",
        "success",
        "success_rate",
    }
    if normalized_key in unit_interval_keys or normalized_key.endswith("/success_rate"):
        return (0, 1.05)
    return None


def _load_success_selected_x(plot_exts, train_res_dir, task_ids, methods, x_key, args):
    grid_data = plot_exts.load_benchmark_curves_by_task_method(
        train_res_dir,
        args.benchmark,
        task_ids,
        train_stage=args.training_stage,
        methods=methods,
        method_experiments=args.method_experiments,
        experiment=args.experiment,
        project=args.project,
        entity=args.wandb_entity,
        metric_key="reward/success",
        x_key=x_key,
        num_interp_points=500,
        smoothing_window=1,
        use_cache=not args.refresh_wandb,
    )
    return _selected_x_by_method(grid_data, task_ids, methods or grid_data.get("__methods__", [])) or None


def _method_experiments_for_methods(args, methods):
    if not args.method_experiments:
        return [args.experiment] * len(methods)
    if len(args.method_experiments) != len(methods):
        raise ValueError("--method_experiments must match --methods length")
    return args.method_experiments


def _finite(values):
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def _mean_std(values):
    vals = _finite(values)
    if len(vals) == 0:
        return math.nan, math.nan
    std = 0.0 if len(vals) <= 1 else float(np.std(vals, ddof=1))
    return float(np.mean(vals)), std


def _tail_best_success(curve, tail_frac=0.1):
    if curve is None:
        return None
    x, y = curve
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    cutoff = x[-1] - float(tail_frac) * max(float(x[-1] - x[0]), 0.0)
    tail_mask = x >= cutoff
    if not tail_mask.any():
        tail_mask[-1] = True
    tail_x = x[tail_mask]
    tail_y = y[tail_mask]
    idx = int(np.argmax(tail_y))
    return float(tail_x[idx]), float(tail_y[idx])


def _interp_at(curve, x_value):
    if curve is None or not math.isfinite(float(x_value)):
        return math.nan
    x, y = curve
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() == 0:
        return math.nan
    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    return float(np.interp(float(x_value), x[order], y[order]))


def _plot_benchmark_success_speed_pareto(
    args,
    plot_exts,
    train_res_dir,
    task_ids,
    methods,
    method_labels,
    colors,
    save_path,
):
    from core.plotting.primitives import _save_and_show

    method_experiments = _method_experiments_for_methods(args, methods)
    summary = []
    for method, experiment in zip(methods, method_experiments):
        discovered = plot_exts.discover_benchmark_task_methods(
            train_res_dir,
            args.benchmark,
            task_ids,
            train_stage=args.training_stage,
            methods=[method],
            experiment=experiment,
        )
        success_values = []
        speed_values = []
        for task_id in task_ids:
            for run in discovered.get(task_id, {}).get(method, []):
                selected = _tail_best_success(
                    _load_local_plot_metric_history(
                        run,
                        args.x_key or "misc/iterations",
                        "reward/success",
                    )
                )
                if selected is None:
                    continue
                selected_x, selected_success = selected
                eps_time = _interp_at(
                    _load_local_plot_metric_history(
                        run,
                        args.x_key or "misc/iterations",
                        "signal/success_eps_time",
                    ),
                    selected_x,
                )
                max_eps_time = _success_eps_time_fill_value(run)
                if max_eps_time is None or max_eps_time <= 0 or not math.isfinite(eps_time):
                    speed = math.nan
                else:
                    speed = float(np.clip((max_eps_time - eps_time) / max_eps_time, 0.0, 1.0))
                success_values.append(selected_success)
                speed_values.append(speed)
        success_mean, success_std = _mean_std(success_values)
        speed_mean, speed_std = _mean_std(speed_values)
        summary.append((success_mean, success_std, speed_mean, speed_std))

    fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.8))
    plotted = False
    for idx, (success_mean, success_std, speed_mean, speed_std) in enumerate(summary):
        if not (math.isfinite(success_mean) and math.isfinite(speed_mean)):
            continue
        label = method_labels[idx] if method_labels and idx < len(method_labels) else methods[idx]
        color = colors[idx] if colors and idx < len(colors) else None
        ax.errorbar(
            success_mean,
            speed_mean,
            xerr=success_std if math.isfinite(success_std) else None,
            yerr=speed_std if math.isfinite(speed_std) else None,
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=7,
            linewidth=1.4,
            label=label,
        )
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "No valid runs found", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Last-10% Best Success Rate")
    ax.set_ylabel("Normalized Success Speed\n((Tmax - Tused) / Tmax)")
    ax.set_xlim(-0.1, 1.05)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Success-Speed Pareto")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    if plotted:
        ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    _save_and_show(fig, save_path, show=False, dpi=300, bbox_inches="tight")


def _run_benchmark_train_curve(args, project_name, is_summary: bool):
    train_res_dir = _train_res_dir(project_name)
    task_ids = _task_ids_from_args(args)
    methods, method_labels, colors = _method_style_from_args(args)
    x_key = args.x_key or "misc/iterations"
    y_key = args.y_key or "reward/success"
    y_name = y_key.split("/")[-1]
    y_label = _train_curve_metric_label(y_key)
    benchmark, experiment, plot_exts = _benchmark_context(args, project_name)
    label = f"{benchmark}/{experiment}"
    grid_data = plot_exts.load_benchmark_curves_by_task_method(
        train_res_dir=train_res_dir,
        benchmark=benchmark,
        task_ids=task_ids,
        train_stage=args.training_stage,
        methods=methods,
        method_experiments=args.method_experiments,
        experiment=experiment,
        project=args.project,
        entity=args.wandb_entity,
        metric_key=y_key,
        x_key=x_key,
        num_interp_points=500,
        smoothing_window=1,
        use_cache=not args.refresh_wandb,
    )

    bar_selected_x = None
    final_bar_metric = y_key == "misc/success_episodes"
    if is_summary and y_key != "reward/success" and not final_bar_metric:
        bar_selected_x = _load_success_selected_x(
            plot_exts,
            train_res_dir,
            task_ids,
            methods,
            x_key,
            args,
        )

    curve_kind = "summary curves" if is_summary else "training curves"
    print(f"Fetched {label} {curve_kind} for tasks {task_ids}.")
    save_path = args.save_path or _default_benchmark_plot_path(
        task_ids,
        y_name,
        summary=is_summary,
        project_name=project_name,
        benchmark=benchmark,
        experiment=experiment,
    )

    if is_summary:
        plot_mt1_success_summary(
            grid_data=grid_data,
            task_ids=task_ids,
            methods=methods,
            method_labels=method_labels,
            method_colors=colors,
            x_label=_train_curve_x_label(x_key),
            y_label=y_label,
            curve_title=f"Aggregate {y_label}",
            bar_title=f"{'Selected' if bar_selected_x is not None else ('Final' if final_bar_metric else 'Best')} {y_label}",
            save_path=save_path,
            show=True,
            ylim=_unit_interval_ylim(y_key),
            annotate_pvalues=not args.disable_pvalue_analysis,
            higher_is_better=not y_key.endswith("success_eps_time"),
            bar_value_mode="final" if final_bar_metric else "best",
            bar_selected_x=bar_selected_x,
            integer_bar_labels=y_key == "misc/success_episodes",
        )
        return

    plot_mt1_training_grid(
        grid_data=grid_data,
        task_ids=task_ids,
        methods=methods,
        method_colors=colors,
        method_labels=method_labels,
        x_label=_train_curve_x_label(x_key),
        y_label=y_label,
        n_cols=8,
        figsize_per_subplot=(4.5, 3),
        save_path=save_path,
        show=True,
        ylim=_unit_interval_ylim(y_key),
        suptitle=f"{label} Training Curves (per task)",
        plain_y_axis=y_key == "train/replay_dataset_trajectories",
    )


def _run_benchmark_signal_metric(args, project_name, is_summary: bool):
    train_res_dir = _train_res_dir(project_name)
    task_ids = _task_ids_from_args(args)
    methods, method_labels, colors = _method_style_from_args(args)
    x_key = args.x_key or "iteration"
    metric_key = args.y_key or "top10pct_Rdense_sr"
    y_label = _signal_metric_label(metric_key)
    benchmark, experiment, plot_exts = _benchmark_context(args, project_name)
    label = f"{benchmark}/{experiment}"
    grid_data = plot_exts.load_benchmark_signal_curves_by_task_method(
        train_res_dir=train_res_dir,
        benchmark=benchmark,
        task_ids=task_ids,
        train_stage=args.training_stage,
        methods=methods,
        method_experiments=args.method_experiments,
        experiment=experiment,
        metric_key=metric_key,
        x_key=x_key,
        num_interp_points=500,
        smoothing_window=1,
    )

    bar_selected_x = None
    if is_summary:
        bar_selected_x = _load_success_selected_x(
            plot_exts,
            train_res_dir,
            task_ids,
            methods,
            x_key,
            args,
        )

    plot_kind = "summary" if is_summary else "grid"
    print(f"Fetched {label} local signal metric {plot_kind} for tasks {task_ids}.")
    signal_ylim = None if metric_key == "success_eps_time" else (0, 1.05)
    save_path = args.save_path or _default_benchmark_plot_path(
        task_ids,
        metric_key,
        summary=is_summary,
        project_name=project_name,
        benchmark=benchmark,
        experiment=experiment,
    )

    if is_summary:
        plot_mt1_success_summary(
            grid_data=grid_data,
            task_ids=task_ids,
            methods=methods,
            method_labels=method_labels,
            method_colors=colors,
            x_label=x_key,
            y_label=y_label,
            curve_title=f"Aggregate {y_label}",
            bar_title=f"{'Selected' if bar_selected_x is not None else 'Best'} {y_label}",
            save_path=save_path,
            show=True,
            ylim=signal_ylim,
            annotate_pvalues=not args.disable_pvalue_analysis,
            higher_is_better=metric_key != "success_eps_time",
            bar_selected_x=bar_selected_x,
        )
        return

    plot_mt1_training_grid(
        grid_data=grid_data,
        task_ids=task_ids,
        methods=methods,
        method_colors=colors,
        method_labels=method_labels,
        x_label=x_key,
        y_label=y_label,
        n_cols=8,
        figsize_per_subplot=(4.5, 3),
        save_path=save_path,
        show=True,
        ylim=signal_ylim,
        suptitle=f"{label} {y_label} (per task)",
    )


def _run_learning_signal(args, kind: str):
    if args.signal_history is None:
        raise ValueError(f"--signal_history is required for {kind}")
    save_root = os.path.dirname(args.signal_history[0])

    if kind == "LearningSignalMap":
        from projects.TSIL.figures.signal_landscape import plot_learning_signal_map

        plot_learning_signal_map(
            signal_history_path=args.signal_history,
            labels=args.legends,
            save_path=args.save_path or os.path.join(save_root, "learning_signal_map.pdf"),
            show=True,
            num_bins=args.signal_num_bins,
            last_frac=args.signal_last_frac,
            min_count=args.signal_min_count,
        )
    elif kind == "LearningSignalSummary":
        from projects.TSIL.figures.signal_landscape import plot_learning_signal_summary

        plot_learning_signal_summary(
            signal_history_path=args.signal_history,
            labels=args.legends,
            save_path=args.save_path or os.path.join(save_root, "learning_signal_summary.pdf"),
            show=True,
            num_bins=args.signal_num_bins,
            last_frac=args.signal_last_frac,
        )
    elif kind == "LearningSignalCombined":
        from projects.TSIL.figures.signal_landscape import plot_learning_signal_combined

        plot_learning_signal_combined(
            signal_history_path=args.signal_history,
            labels=args.legends,
            save_path=args.save_path or os.path.join(save_root, "learning_signal_combined.pdf"),
            show=True,
            num_bins=args.signal_num_bins,
            last_frac=args.signal_last_frac,
            min_count=args.signal_min_count,
            summary_signal_history_path=args.signal_summary_history,
            summary_labels=args.signal_summary_legends,
            summary_last_frac=args.signal_summary_last_frac,
            data_cache_path=args.signal_combined_cache_path,
            refresh_cache=args.signal_combined_refresh_cache,
        )
    elif kind == "LearningSignalSurface":
        from projects.TSIL.figures.signal_landscape import plot_learning_signal_surface

        plot_learning_signal_surface(
            signal_history_path=args.signal_history,
            labels=args.legends,
            save_path=args.save_path or os.path.join(save_root, "learning_signal_surface.pdf"),
            show=True,
            num_bins=args.signal_num_bins,
            last_frac=args.signal_last_frac,
        )


def _run_policy_landscape(args):
    if args.policy_landscape is None:
        raise ValueError(f"--policy_landscape is required for {args.task}")
    save_root = os.path.dirname(args.policy_landscape[0])

    if args.task == "PolicyDirectionLandscapeGroupGrid":
        from projects.TSIL.figures.signal_landscape import plot_policy_direction_landscape_group_grid

        plot_policy_direction_landscape_group_grid(
            landscape_paths=args.policy_landscape,
            labels=args.legends,
            group_size=args.policy_landscape_group_size,
            save_path=args.save_path or os.path.join(save_root, "policy_direction_landscape_group_grid.pdf"),
            show=True,
            title=args.plot_title,
            show_legend=not args.hide_legend,
        )
        return

    from projects.TSIL.figures.signal_landscape import (
        plot_policy_direction_landscape,
        plot_policy_direction_landscape_grid,
    )

    save_path = args.save_path or os.path.join(save_root, "policy_direction_landscape.pdf")
    if len(args.policy_landscape) == 1:
        plot_policy_direction_landscape(
            landscape_path=args.policy_landscape[0],
            save_path=save_path,
            show=True,
            title=args.plot_title,
            show_legend=not args.hide_legend,
        )
    else:
        plot_policy_direction_landscape_grid(
            landscape_paths=args.policy_landscape,
            labels=args.legends,
            save_path=save_path,
            show=True,
            title=args.plot_title,
            show_legend=not args.hide_legend,
        )


def _run_sil_revisit(args):
    if args.signal_history is None:
        raise ValueError(f"--signal_history is required for {args.task}")
    save_root = os.path.dirname(args.signal_history[0])

    if args.task == "SilRevisitMechanism":
        from projects.TSIL.figures.sil_analysis import plot_sil_revisit_mechanism

        plot_sil_revisit_mechanism(
            signal_history_paths=args.signal_history,
            labels=args.legends,
            save_path=args.save_path or os.path.join(save_root, "sil_revisit_mechanism.pdf"),
            show=True,
        )
    elif args.task == "SilRevisitOutcome":
        from projects.TSIL.figures.sil_analysis import plot_sil_revisit_outcome

        plot_sil_revisit_outcome(
            signal_history_paths=args.signal_history,
            labels=args.legends,
            save_path=args.save_path or os.path.join(save_root, "sil_revisit_outcome.pdf"),
            show=True,
        )


def main(project_name: str = "TSIL"):
    args = parse_args()

    if args.task == "BenchmarkTrainCurve":
        _run_benchmark_train_curve(args, project_name, is_summary=False)
    elif args.task == "BenchmarkTrainCurveSummary":
        _run_benchmark_train_curve(args, project_name, is_summary=True)
    elif args.task == "BenchmarkSuccessSpeedPareto":
        train_res_dir = _train_res_dir(project_name)
        task_ids = _task_ids_from_args(args)
        methods, method_labels, colors = _method_style_from_args(args)
        if not methods:
            raise ValueError("BenchmarkSuccessSpeedPareto requires --methods")
        _require_benchmark(args.benchmark)
        plot_exts = _project_plot_exts(project_name)
        save_path = args.save_path or _default_benchmark_plot_path(
            task_ids,
            "summary_success_speed_pareto",
            project_name=project_name,
            benchmark=args.benchmark,
            experiment=args.experiment,
        )
        _plot_benchmark_success_speed_pareto(
            args,
            plot_exts,
            train_res_dir,
            task_ids,
            methods,
            method_labels,
            colors,
            save_path,
        )
    elif args.task == "BenchmarkSignalMetricGrid":
        _run_benchmark_signal_metric(args, project_name, is_summary=False)
    elif args.task == "BenchmarkSignalMetricSummary":
        _run_benchmark_signal_metric(args, project_name, is_summary=True)
    elif args.task.startswith("LearningSignal"):
        _run_learning_signal(args, args.task)
    elif args.task.startswith("PolicyDirectionLandscape"):
        _run_policy_landscape(args)
    elif args.task.startswith("SilRevisit"):
        _run_sil_revisit(args)
    else:
        raise ValueError(f"Unsupported TSIL plot task: {args.task}")


if __name__ == "__main__":
    main()
