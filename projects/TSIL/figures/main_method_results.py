"""Main-method paper figures for TSIL benchmark results."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker as mpl_ticker

from core.plotting.style import COLORS, NPG_PALETTE, style_axis
from core.plotting.train_data import (
    _load_local_plot_metric_history,
    _load_local_signal_history,
    interpolate_to_common_x,
)
from projects.TSIL.reports.curve_metrics import (
    _clean_curve,
    _interp_at,
    _last_value,
    _mean_error,
    _prepend_zero_success_curve,
    _success_metrics,
)
from projects.TSIL.ckpt_layout import discover_benchmark_task_methods


REWARD_TOKENS = ("1e1", "1e2", "1e3", "1e4", "1e5", "1e6")
REWARD_LABELS = ("1e1", "1e2", "1e3", "1e4", "1e5", "1e6")
REWARD_AXIS_LABELS = ("10", r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$", r"$10^6$")
REWARD_SWEEP_IH_TITLE = "Infinite-horizon PPO"
REWARD_SWEEP_ATTL_TITLE = "Temporal Self-Imitation Learning"
REWARD_SWEEP_ATTL_METHODS = ("PPO_ATTL_TSIL",)
DEFAULT_BAR_HEIGHT = 4.1
TITLE_FONT = 14
AXIS_FONT = 13
TICK_FONT = 11.5
BAR_TICK_FONT = 10.5
LEGEND_FONT = 11
MAIN_BAR_PANELS = [
    ("auc_success", None, "AUC success", (0.0, 1.05), False),
    ("success_episodes", None, "Successful episodes", None, True),
    ("tail_best_success", None, "Best success rate", (0.0, 1.05), False),
    ("success_eps_time", None, "Successful episode time (s)", None, False),
]


def _method_color(method, fallback_idx):
    if method == "PPO" or method.endswith(("_NOTRAIN", "_TSIL_NOTRAIN")):
        return NPG_PALETTE["navy"]
    if method.endswith(("_SIL_TRANS", "_SIL_RTOPK")):
        return NPG_PALETTE["vermillion"]
    if method.endswith(("_TSIL", "_TSILTOPK")):
        return NPG_PALETTE["teal"]
    return COLORS[fallback_idx % len(COLORS)]


def _method_colors(methods):
    return [_method_color(method, idx) for idx, method in enumerate(methods)]


def _math_sci_tick(value, _):
    if not np.isfinite(value):
        return ""
    if math.isclose(value, 0.0, abs_tol=1e-12):
        return "0"
    sign = "-" if value < 0 else ""
    value = abs(float(value))
    exponent = int(math.floor(math.log10(value)))
    coeff = value / (10 ** exponent)
    if math.isclose(coeff, 1.0, rel_tol=1e-8, abs_tol=1e-8):
        return rf"${sign}10^{{{exponent}}}$"
    return rf"${sign}{coeff:g}\times10^{{{exponent}}}$"


def _method_experiments(methods, experiments, default_experiment):
    if experiments is None:
        return [default_experiment] * len(methods)
    if len(experiments) != len(methods):
        raise ValueError("--method-experiments must match --methods length")
    return experiments


def _discover_runs(train_root, benchmark, train_stage, task_ids, methods, experiments):
    runs = {method: {task_id: [] for task_id in task_ids} for method in methods}
    for method, experiment in zip(methods, experiments):
        discovered = discover_benchmark_task_methods(
            str(train_root),
            benchmark,
            task_ids,
            train_stage=train_stage,
            methods=[method],
            experiment=experiment,
        )
        for task_id in task_ids:
            runs[method][task_id] = discovered.get(task_id, {}).get(method, [])
    return runs


def _load_curve(run_folder, x_key, metric_key):
    curve = _load_local_plot_metric_history(str(run_folder), x_key, metric_key)
    if curve is not None:
        return curve
    if metric_key.startswith("signal/"):
        signal_key = metric_key.split("/", 1)[1]
        for fallback_x in (x_key, x_key.split("/")[-1]):
            curve = _load_local_signal_history(str(run_folder), signal_key, fallback_x)
            if curve is not None:
                return curve
    return None


def _prepare_metric_curve(curve, metric_key):
    if metric_key == "reward/success":
        return _prepend_zero_success_curve(curve)
    return _clean_curve(curve)


def _aggregate_method_curve(runs_by_method, task_ids, method, x_key, metric_key, error_mode="sem"):
    task_curves = []
    for task_id in task_ids:
        run_curves = [
            _load_curve(run_folder, x_key, metric_key)
            for run_folder in runs_by_method.get(method, {}).get(task_id, [])
        ]
        run_curves = [_prepare_metric_curve(curve, metric_key) for curve in run_curves if curve is not None]
        run_curves = [curve for curve in run_curves if curve is not None and len(curve[0]) > 1]
        if not run_curves:
            continue
        common_x, interp_ys = interpolate_to_common_x(run_curves, num_points=500)
        if common_x is None:
            continue
        task_curves.append((common_x, np.mean(interp_ys, axis=0)))

    if not task_curves:
        return None

    common_x, task_means = interpolate_to_common_x(
        [(x, y) for x, y in task_curves],
        num_points=500,
    )
    if common_x is None:
        return None
    n_tasks = int(task_means.shape[0])
    mean_y = np.mean(task_means, axis=0)
    if n_tasks <= 1:
        std_y = np.zeros_like(mean_y)
    else:
        std_y = np.std(task_means, axis=0, ddof=1)
    err_y = std_y if error_mode == "std" else std_y / math.sqrt(n_tasks)
    return common_x, mean_y, err_y


def _collect_main_rows(runs_by_method, task_ids, methods, x_key, tail_frac):
    rows = []
    for method in methods:
        for task_id in task_ids:
            for run_folder in runs_by_method.get(method, {}).get(task_id, []):
                success = _success_metrics(
                    _load_curve(run_folder, x_key, "reward/success"),
                    tail_frac,
                )
                if success is None:
                    continue

                success_episode_curve = _load_curve(
                    run_folder,
                    "misc/interaction_time",
                    "misc/success_episodes",
                )
                rows.append({
                    "method": method,
                    "task_id": task_id,
                    "tail_best_success": success["tail_best_success"],
                    "auc_success": success["auc_success"],
                    "success_eps_time": _interp_at(
                        _load_curve(run_folder, x_key, "signal/success_eps_time"),
                        success["tail_best_x"],
                    ),
                    "success_episodes": _last_value(success_episode_curve),
                })
    return rows


def _collect_reward_rows(runs_by_method, task_ids, methods, x_key, tail_frac):
    rows = []
    for method in methods:
        for task_id in task_ids:
            for run_folder in runs_by_method.get(method, {}).get(task_id, []):
                metrics = _success_metrics(
                    _load_curve(run_folder, x_key, "reward/success"),
                    tail_frac,
                )
                if metrics is None:
                    continue
                rows.append({
                    "method": method,
                    "task_id": task_id,
                    "tail_best_success": metrics["tail_best_success"],
                })
    return rows


def _display_labels(legends, explicit_labels=None):
    return explicit_labels if explicit_labels else legends


def _x_label(x_key):
    return {
        "misc/steps": "Steps",
        "steps": "Steps",
        "misc/iterations": "Iterations",
        "iteration": "Iterations",
    }.get(x_key, x_key)


def _finish_axis(ax, ylim=None, sci_x=False, sci_y=False, tick_font=TICK_FONT):
    style_axis(ax)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=tick_font, pad=1.2)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if sci_x:
        ax.xaxis.set_major_formatter(mpl_ticker.FuncFormatter(_math_sci_tick))
        ax.xaxis.get_offset_text().set_visible(False)
    if sci_y:
        ax.yaxis.set_major_formatter(mpl_ticker.FuncFormatter(_math_sci_tick))
        ax.yaxis.get_offset_text().set_visible(False)


def _save(fig, path, pad_inches=0.015):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    print(f"Wrote {path}")


def _row_major_legend_items(handles, labels, ncol):
    nrows = int(math.ceil(len(labels) / float(ncol)))
    order = [
        row * ncol + col
        for col in range(ncol)
        for row in range(nrows)
        if row * ncol + col < len(labels)
    ]
    return [handles[idx] for idx in order], [labels[idx] for idx in order]


def _plot_curve_panel(
    ax,
    args,
    runs_by_method,
    colors,
    labels,
    x_key,
    metric_key,
    title,
    ylim=None,
    sci_y=False,
    collect_legend=False,
):
    handles = []
    legend_labels = []
    for idx, method in enumerate(args.methods):
        curve = _aggregate_method_curve(
            runs_by_method,
            args.task_ids,
            method,
            x_key,
            metric_key,
            args.bar_error,
        )
        if curve is None:
            continue
        x, y, err = curve
        color = colors[idx % len(colors)]
        line = ax.plot(x, y, color=color, linewidth=1.4, label=labels[idx])[0]
        ax.fill_between(x, y - err, y + err, color=color, alpha=0.18, linewidth=0)
        if collect_legend:
            handles.append(line)
            legend_labels.append(labels[idx])
    if not ax.lines:
        ax.text(0.5, 0.5, "No runs found", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title, fontsize=TITLE_FONT, pad=1)
    ax.set_xlabel(_x_label(x_key) if metric_key == "reward/success" else "Interaction time (s)", fontsize=AXIS_FONT, labelpad=1.0)
    ax.set_ylabel("")
    _finish_axis(
        ax,
        ylim=ylim,
        sci_x=x_key.endswith("steps") or x_key.endswith("interaction_time"),
        sci_y=sci_y,
    )
    if metric_key == "reward/success":
        ax.set_xlim(left=0.0)
    return handles, legend_labels


def _task_mean_values(rows, method, metric_key):
    values_by_task = {}
    for row in rows:
        if row["method"] != method:
            continue
        value = row.get(metric_key, math.nan)
        if value is None or not math.isfinite(float(value)):
            continue
        values_by_task.setdefault(row["task_id"], []).append(float(value))
    return [
        float(np.mean(values_by_task[task_id]))
        for task_id in sorted(values_by_task)
        if values_by_task[task_id]
    ]


def _bar_values(rows, methods, metric_key, error_mode):
    means = []
    errors = []
    counts = []
    for method in methods:
        mean, err, count = _mean_error(
            _task_mean_values(rows, method, metric_key),
            error_mode,
        )
        means.append(mean)
        errors.append(err)
        counts.append(count)
    return np.asarray(means), np.asarray(errors), counts


def _plot_bars(
    ax,
    labels,
    means,
    errors,
    colors,
    ylabel,
    title,
    ylim=None,
    sci_y=False,
    label_rotation=90,
):
    x = np.arange(len(labels))
    finite = np.isfinite(means)
    safe_errors = np.where(np.isfinite(errors), errors, 0.0)
    ax.bar(
        x[finite],
        means[finite],
        yerr=safe_errors[finite],
        width=0.72,
        color=[colors[idx % len(colors)] for idx in x[finite]],
        linewidth=0,
        error_kw={"ecolor": "#3A3A3A", "elinewidth": 0.7, "capsize": 0},
        zorder=3,
    )
    if not finite.any():
        ax.text(0.5, 0.5, "No runs found", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=label_rotation,
        ha="center",
        va="top" if label_rotation else "center",
        fontsize=BAR_TICK_FONT,
    )
    ax.set_ylabel("" if ylabel is None else ylabel, fontsize=AXIS_FONT, labelpad=1.0)
    ax.set_title(title, fontsize=TITLE_FONT, pad=1)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    _finish_axis(ax, ylim=ylim, sci_y=sci_y, tick_font=BAR_TICK_FONT)
    ax.set_xticklabels(
        labels,
        rotation=label_rotation,
        ha="center",
        va="top" if label_rotation else "center",
        fontsize=BAR_TICK_FONT,
    )


def plot_main_training_curves(args, runs_by_method, colors, labels):
    fig, axes = plt.subplots(2, 1, figsize=(4.8, 5.25))
    handles, legend_labels = _plot_curve_panel(
        axes[0],
        args,
        runs_by_method,
        colors,
        labels,
        args.x_key,
        "reward/success",
        "Success rate",
        ylim=(0.0, 1.05),
        collect_legend=True,
    )
    _plot_curve_panel(
        axes[1],
        args,
        runs_by_method,
        colors,
        labels,
        "misc/interaction_time",
        "misc/success_episodes",
        "Successful episodes",
        sci_y=True,
    )

    if handles:
        ncol = min(4, len(legend_labels))
        handles, legend_labels = _row_major_legend_items(handles, legend_labels, ncol)
        legend = fig.legend(
            handles,
            legend_labels,
            loc="upper center",
            ncol=ncol,
            frameon=False,
            fontsize=LEGEND_FONT,
            bbox_to_anchor=(0.5, 0.995),
            borderaxespad=0.0,
            handlelength=1.65,
            columnspacing=0.62,
            handletextpad=0.28,
            labelspacing=0.12,
        )
        for legend_line in legend.get_lines():
            legend_line.set_linewidth(2.6)
    fig.tight_layout(rect=(0.0, 0.015, 1.0, 0.925), pad=0.12, h_pad=0.50)
    _save(fig, args.figure_dir / f"main_method_training_curves.{args.format}")


def plot_main_bar_results(args, rows, colors, labels):
    fig, axes = plt.subplots(2, 2, figsize=(5.45, 5.35))
    for ax, (metric, ylabel, title, ylim, sci_y) in zip(axes.flat, MAIN_BAR_PANELS):
        means, errors, counts = _bar_values(rows, args.methods, metric, args.bar_error)
        _plot_bars(ax, labels, means, errors, colors, ylabel, title, ylim=ylim, sci_y=sci_y)
    fig.tight_layout(pad=0.15, h_pad=0.65, w_pad=0.45)
    _save(fig, args.figure_dir / f"main_method_bar_results.{args.format}")


def plot_main_results(args, runs_by_method, rows, colors, legends):
    labels = _display_labels(legends, args.display_labels)
    plot_main_training_curves(args, runs_by_method, colors, labels)
    plot_main_bar_results(args, rows, colors, labels)

def _reward_group_values(args, method_groups):
    methods = sorted({method for group in method_groups for method in group})
    runs = _discover_runs(
        args.train_root,
        args.benchmark,
        args.train_stage,
        args.task_ids,
        methods,
        ["sweep_suc_rew"] * len(methods),
    )
    rows = _collect_reward_rows(runs, args.task_ids, methods, args.x_key, args.tail_frac)
    means = []
    errors = []
    for group in method_groups:
        group_means, group_errors, _ = _bar_values(rows, group, "tail_best_success", args.bar_error)
        if np.isfinite(group_means).any():
            best_idx = int(np.nanargmax(group_means))
            means.append(group_means[best_idx])
            errors.append(group_errors[best_idx])
        else:
            means.append(math.nan)
            errors.append(math.nan)
    return np.asarray(means), np.asarray(errors)


def plot_reward_sweep(args):
    families = [
        (REWARD_SWEEP_IH_TITLE, [[f"PPO_MaxR_{token}"] for token in REWARD_TOKENS]),
        (
            REWARD_SWEEP_ATTL_TITLE,
            [
                [f"{method}_MaxR_{token}" for method in REWARD_SWEEP_ATTL_METHODS]
                for token in REWARD_TOKENS
            ],
        ),
    ]
    fig, axes = plt.subplots(len(families), 1, figsize=(5.15, DEFAULT_BAR_HEIGHT))
    reward_colors = COLORS[: len(REWARD_LABELS)]
    for ax, (title, method_groups) in zip(axes, families):
        means, errors = _reward_group_values(args, method_groups)
        _plot_bars(
            ax,
            REWARD_AXIS_LABELS,
            means,
            errors,
            reward_colors,
            "Success rate",
            title,
            ylim=(0.0, 1.05),
            label_rotation=0,
        )
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=True, pad=4.0)
    axes[-1].set_xlabel("Reward scale", fontsize=AXIS_FONT, labelpad=1.0)
    fig.tight_layout(pad=0.15, h_pad=0.55)
    _save(fig, args.figure_dir / f"reward_sweep_bars.{args.format}", pad_inches=0.04)

def parse_args():
    parser = argparse.ArgumentParser(description="Render paper group-1 composite figures")
    parser.add_argument("--train-root", type=Path, default=Path("results/TSIL/train_res"))
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--benchmark", default="mt01")
    parser.add_argument("--training-stage", dest="train_stage", default="scratch")
    parser.add_argument("--task-ids", type=int, nargs="+", required=True)
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    parser.add_argument("--x-key", default="misc/steps")
    parser.add_argument("--tail-frac", type=float, default=0.1)
    parser.add_argument("--bar-error", default="sem", choices=["sem", "std"])
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--method-experiments", nargs="+")
    parser.add_argument("--legends", nargs="+", required=True)
    parser.add_argument("--display-labels", nargs="+")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.display_labels and len(args.display_labels) != len(args.methods):
        raise SystemExit("--display-labels must match --methods length")
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    experiments = _method_experiments(args.methods, args.method_experiments, "compare_temporal")
    runs_by_method = _discover_runs(
        args.train_root,
        args.benchmark,
        args.train_stage,
        args.task_ids,
        args.methods,
        experiments,
    )
    colors = _method_colors(args.methods)
    rows = _collect_main_rows(runs_by_method, args.task_ids, args.methods, args.x_key, args.tail_frac)
    plot_main_results(args, runs_by_method, rows, colors, args.legends)
    plot_reward_sweep(args)


if __name__ == "__main__":
    main()
