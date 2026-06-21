"""Compact benchmark stability report plots for TSIL."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.6,
})
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from core.plotting.style import NPG_PALETTE, style_axis
from core.plotting.train_data import (
    _load_local_plot_metric_history,
    interpolate_to_common_x,
)
from projects.TSIL.ckpt_layout import discover_benchmark_task_methods


METHODS = [
    ("TSIL_NOTRAIN", "ATTL", NPG_PALETTE["navy"]),
    ("SIL_TRANS", "ATTL+SIL", NPG_PALETTE["vermillion"]),
    ("TSIL", "TSIL", NPG_PALETTE["teal"]),
]
PLOT_METHODS = [METHODS[2], METHODS[1], METHODS[0]]

STRESS_EXPERIMENTS = [
    ("policy_grad_noise", "scratch", "Policy-gradient noise", "noise scale",
     [(None, "0"), ("PGradN5", "5"), ("PGradN10", "10"), ("PGradN20", "20")]),
    ("dense_dropout", "scratch", "Dense reward dropout", "drop probability",
     [(None, "0"), ("Drop0.4", "0.4"), ("Drop0.6", "0.6"), ("Drop0.8", "0.8")]),
    ("sweep_clip", "scratch", "PPO clip", "clip range",
     [(None, "0.2"), ("Clip0.3", "0.3"), ("Clip0.5", "0.5"), ("Clip0.7", "0.7")]),
    ("sweep_lr", "scratch", "Learning rate", "learning rate",
     [(None, "5e-4"), ("LR1e-3", "1e-3"), ("LR5e-3", "5e-3"), ("LR1e-2", "1e-2")]),
]


def method_names(family, token):
    base = "PPO_ATTL"
    if token is None:
        return [base] if family == "off" else [f"{base}_{family}"]
    if family == "off":
        return [f"{base}_{token}", f"{base}_TSIL_NOTRAIN_{token}"]
    return [f"{base}_{family}_{token}"]


def method_name(family, token):
    return method_names(family, token)[0]


def write_table(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    if not rows:
        return
    columns = list(rows[0])
    with path.open("w") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row[col]) for col in columns) + " |\n")


def read_table(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def _matches_policy_noise_guard_filter(run, experiment, token):
    if experiment != "policy_grad_noise" or token is None:
        return True
    config_path = Path(run) / "config.json"
    if not config_path.exists():
        return False
    try:
        with config_path.open() as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    try:
        max_grad_norm = float(config.get("max_grad_norm", 0.0))
    except (TypeError, ValueError):
        return False
    return config.get("target_kl") is None and max_grad_norm >= 1e8


def discover_runs(train_root, benchmark, experiment, stage, task_ids, method, token=None):
    methods = [method] if isinstance(method, str) else list(method)
    task_methods = discover_benchmark_task_methods(
        str(train_root),
        benchmark,
        task_ids,
        train_stage=stage,
        methods=methods,
        experiment=experiment,
    )
    runs = []
    for tid in task_ids:
        for method_name_ in methods:
            for run in task_methods.get(tid, {}).get(method_name_, []):
                if _matches_policy_noise_guard_filter(run, experiment, token):
                    runs.append(run)
    return runs


def load_metric_curves(
    train_root, benchmark, experiment, stage, task_ids, method, metric_key, x_key,
    token=None, min_target_steps=0.0,
):
    runs = discover_runs(train_root, benchmark, experiment, stage, task_ids, method, token)
    curves = []
    for run in runs:
        curve = _load_local_plot_metric_history(run, x_key, metric_key)
        if curve is not None and len(curve[0]) > 1:
            config_path = Path(run) / "config.json"
            target_steps = 0.0
            if x_key in {"steps", "misc/steps"} and config_path.exists():
                try:
                    with config_path.open() as f:
                        config = json.load(f)
                    target_steps = float(config.get("total_timesteps") or 0.0)
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    target_steps = 0.0
                if min_target_steps and target_steps < min_target_steps:
                    continue
                if target_steps > 0 and float(curve[0][-1]) < 0.95 * target_steps:
                    continue
            elif min_target_steps:
                continue
            curves.append(curve)
    return curves, len(runs)


def average_curves(curves):
    if not curves:
        return None
    common_x, ys = interpolate_to_common_x(curves, num_points=500)
    if common_x is None:
        return None
    std_y = np.zeros_like(ys[0]) if ys.shape[0] <= 1 else np.std(ys, axis=0, ddof=1)
    return common_x, ys.mean(axis=0), std_y


def load_curve(
    train_root, benchmark, experiment, stage, task_ids, method, metric_key, x_key,
    token=None, min_target_steps=0.0,
):
    curves, run_count = load_metric_curves(
        train_root, benchmark, experiment, stage, task_ids, method, metric_key, x_key,
        token, min_target_steps,
    )
    return average_curves(curves), run_count, len(curves)


def load_curve_any(
    train_root, benchmark, experiment, stage, task_ids, method, metric_keys, x_key,
    token=None, min_target_steps=0.0,
):
    run_count = curve_count = 0
    for key in metric_keys:
        curve, run_count, curve_count = load_curve(
            train_root, benchmark, experiment, stage, task_ids, method, key, x_key,
            token, min_target_steps,
        )
        if curve is not None:
            return curve, run_count, curve_count
    return None, run_count, curve_count


def finite_round(value, digits=4):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        rounded = round(float(value), digits)
        return int(rounded) if digits == 0 else rounded
    return value


def finite_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if math.isfinite(result) else np.nan


def curve_metrics(curve, threshold=0.8):
    if curve is None:
        return "NA", "NA", "NA"
    best, auc, first = raw_curve_metrics(curve, threshold)
    return (
        finite_round(best, 4),
        finite_round(auc, 4),
        finite_round(first, 0),
    )


def raw_curve_metrics(curve, threshold=0.8):
    x, y = curve[0], curve[1]
    span = float(x[-1] - x[0])
    auc = float(np.trapz(y, x) / span) if span > 0 else float(y[-1])
    reached = np.flatnonzero(y >= threshold)
    first = float(x[reached[0]]) if len(reached) else np.nan
    tail = y[int(len(y) * 0.9):]
    return float(np.max(tail)), auc, first


def scalar_mean_std(values, digits=4):
    finite = np.array([finite_float(v) for v in values], dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return "NA", "NA"
    std = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    return finite_round(float(np.mean(finite)), digits), finite_round(std, digits)


def row_metric_sem(row, metric):
    std = finite_float(row.get(f"{metric}_std", "NA"))
    if not math.isfinite(std):
        return np.nan
    n = finite_float(row.get("run_count", "NA"))
    if math.isfinite(n) and n > 0:
        return std / math.sqrt(n)
    return std


def raw_curve_tail_mean(curve):
    y = curve[1]
    tail = y[int(len(y) * 0.9):]
    return float(np.mean(tail))


def curve_last(curve, digits=4):
    if curve is None:
        return "NA"
    return finite_round(raw_curve_last(curve), digits)


def raw_curve_last(curve):
    return float(curve[1][-1])


def format_stress_tick(label):
    if "e" not in str(label).lower():
        return label
    mantissa, exponent = str(label).lower().split("e", 1)
    try:
        mantissa = float(mantissa)
        exponent = int(exponent)
    except ValueError:
        return label
    if math.isclose(mantissa, 1.0):
        return rf"$10^{{{exponent}}}$"
    if math.isclose(mantissa, round(mantissa)):
        mantissa = int(round(mantissa))
    return rf"${mantissa}\times 10^{{{exponent}}}$"


def stress_tick_labels(experiment, tokens):
    labels = [label for _, label in tokens]
    if experiment == "sweep_lr":
        return labels
    return [format_stress_tick(label) for label in labels]


def collect_stress(args):
    rows = []
    curves = {}
    missing = []
    for experiment, stage, title, xlabel, tokens in STRESS_EXPERIMENTS:
        for token, label in tokens:
            for family, method_label, _ in METHODS:
                method = method_names(family, token)
                source_experiment = "compare_tsil" if token is None else experiment
                source_stage = "scratch" if token is None else stage
                min_target_steps = args.min_target_steps_by_experiment.get(source_experiment, 0.0)
                success_curves, run_count = load_metric_curves(
                    args.train_root, args.benchmark, source_experiment, source_stage,
                    args.task_ids, method, "reward/success", args.x_key, token,
                    min_target_steps,
                )
                curve = average_curves(success_curves)
                curve_count = len(success_curves)
                status = "ok" if curve is not None else ("no_metric" if run_count else "missing")
                if status != "ok":
                    missing.append(f"{source_experiment}/{source_stage}/{token or 'base'}/{method[0]}: {status}")
                best_values, auc_values, first_values = [], [], []
                for run_curve in success_curves:
                    run_best, run_auc, run_first = raw_curve_metrics(run_curve)
                    best_values.append(run_best)
                    auc_values.append(run_auc)
                    first_values.append(run_first)
                best, best_std = scalar_mean_std(best_values, 4)
                final_success, final_success_std = scalar_mean_std(
                    [raw_curve_last(run_curve) for run_curve in success_curves], 4,
                )
                auc, auc_std = scalar_mean_std(auc_values, 4)
                first, first_std = scalar_mean_std(first_values, 0)
                success_episode_curves, _ = load_metric_curves(
                    args.train_root, args.benchmark, source_experiment, source_stage,
                    args.task_ids, method, "misc/success_episodes", args.x_key, token,
                    min_target_steps,
                )
                success_episodes_value, success_episodes_std = scalar_mean_std(
                    [raw_curve_last(run_curve) for run_curve in success_episode_curves], 0,
                )
                episode_time_curves, _ = load_metric_curves(
                    args.train_root, args.benchmark, source_experiment, source_stage,
                    args.task_ids, method, "signal/success_eps_time", args.x_key, token,
                    min_target_steps,
                )
                episode_time_value, episode_time_std = scalar_mean_std(
                    [raw_curve_tail_mean(run_curve) for run_curve in episode_time_curves], 4,
                )
                first_revisit, _, _ = load_curve_any(
                    args.train_root, args.benchmark, source_experiment, source_stage,
                    args.task_ids, method,
                    ["signal/first_fast_revisit_steps", "signal/sil_first_revisit_steps"],
                    args.x_key,
                    token,
                    min_target_steps,
                )
                rows.append({
                    "experiment": experiment,
                    "stress_value": label,
                    "method": method_label,
                    "best_success": best,
                    "best_success_std": best_std,
                    "final_success": final_success,
                    "final_success_std": final_success_std,
                    "auc_success": auc,
                    "auc_success_std": auc_std,
                    "steps_to_80": first,
                    "steps_to_80_std": first_std,
                    "success_episodes": success_episodes_value,
                    "success_episodes_std": success_episodes_std,
                    "episode_time": episode_time_value,
                    "episode_time_std": episode_time_std,
                    "first_revisit_steps": curve_last(first_revisit, 0),
                    "status": status,
                    "run_count": curve_count,
                })
                curves[(experiment, token, family)] = curve
    return rows, curves, missing


def aggregate_stress_rows(rows):
    base_labels = {
        experiment: tokens[0][1]
        for experiment, _, _, _, tokens in STRESS_EXPERIMENTS
        if tokens
    }
    aggregate_rows = []
    for experiment, _, _, _, _ in STRESS_EXPERIMENTS:
        for _, method_label, _ in METHODS:
            base = next(
                (
                    row for row in rows
                    if row.get("experiment") == experiment
                    and row.get("method") == method_label
                    and row.get("stress_value") == base_labels.get(experiment)
                ),
                None,
            )
            stress_rows = [
                row for row in rows
                if row.get("experiment") == experiment
                and row.get("method") == method_label
                and row.get("stress_value") != base_labels.get(experiment)
            ]
            if not stress_rows:
                continue
            bests = [finite_float(row.get("best_success")) for row in stress_rows]
            aucs = [finite_float(row.get("auc_success")) for row in stress_rows]
            finals = [finite_float(row.get("final_success")) for row in stress_rows]
            success_episodes = [finite_float(row.get("success_episodes")) for row in stress_rows]
            episode_times = [finite_float(row.get("episode_time")) for row in stress_rows]
            steps_to_80 = [finite_float(row.get("steps_to_80")) for row in stress_rows]
            bests = [value for value in bests if math.isfinite(value)]
            aucs = [value for value in aucs if math.isfinite(value)]
            finals = [value for value in finals if math.isfinite(value)]
            success_episodes = [value for value in success_episodes if math.isfinite(value)]
            episode_times = [value for value in episode_times if math.isfinite(value)]
            steps_to_80 = [value for value in steps_to_80 if math.isfinite(value)]
            base_best = finite_float(base.get("best_success") if base else "NA")
            base_auc = finite_float(base.get("auc_success") if base else "NA")
            base_success_episodes = finite_float(base.get("success_episodes") if base else "NA")
            measured = sum(
                1 for row in stress_rows
                if str(row.get("status", "")).startswith(("ok", "measured"))
            )
            estimated = len(stress_rows) - measured
            aggregate_rows.append({
                "experiment": experiment,
                "method": method_label,
                "stress_values": ",".join(row.get("stress_value", "") for row in stress_rows),
                "mean_success_rate": finite_round(np.mean(bests) if bests else np.nan, 4),
                "success_rate_std": finite_round(np.std(bests, ddof=1) if len(bests) > 1 else (0.0 if bests else np.nan), 4),
                "mean_auc_success": finite_round(np.mean(aucs) if aucs else np.nan, 4),
                "auc_success_std": finite_round(np.std(aucs, ddof=1) if len(aucs) > 1 else (0.0 if aucs else np.nan), 4),
                "mean_final_success": finite_round(np.mean(finals) if finals else np.nan, 4),
                "mean_success_episodes": finite_round(np.mean(success_episodes) if success_episodes else np.nan, 0),
                "success_episodes_std": finite_round(np.std(success_episodes, ddof=1) if len(success_episodes) > 1 else (0.0 if success_episodes else np.nan), 0),
                "mean_success_eps_time": finite_round(np.mean(episode_times) if episode_times else np.nan, 4),
                "success_eps_time_std": finite_round(np.std(episode_times, ddof=1) if len(episode_times) > 1 else (0.0 if episode_times else np.nan), 4),
                "mean_steps_to_80_successful_only": finite_round(np.mean(steps_to_80) if steps_to_80 else np.nan, 0),
                "steps_to_80_successful_only_std": finite_round(np.std(steps_to_80, ddof=1) if len(steps_to_80) > 1 else (0.0 if steps_to_80 else np.nan), 0),
                "steps_to_80_n": len(steps_to_80),
                "collapse_count_lt_0.8": sum(1 for value in bests if value < 0.8),
                "delta_success_vs_base": finite_round(np.mean(bests) - base_best if bests and math.isfinite(base_best) else np.nan, 4),
                "delta_auc_vs_base": finite_round(np.mean(aucs) - base_auc if aucs and math.isfinite(base_auc) else np.nan, 4),
                "delta_success_episodes_vs_base": finite_round(
                    np.mean(success_episodes) - base_success_episodes
                    if success_episodes and math.isfinite(base_success_episodes)
                    else np.nan,
                    0,
                ),
                "measured_stress_rows": measured,
                "estimated_stress_rows": estimated,
            })
    return aggregate_rows


def compact_stress_rows(rows):
    columns = [
        "experiment", "stress_value", "method",
        "best_success", "best_success_std",
        "auc_success", "auc_success_std",
        "final_success", "final_success_std",
        "success_episodes", "success_episodes_std",
        "episode_time", "episode_time_std",
        "steps_to_80", "status", "run_count",
    ]
    return [{column: row.get(column, "") for column in columns} for row in rows]


def plot_stress(rows, out_dir, fmt):
    row_index = {(r["experiment"], r["stress_value"], r["method"]): r for r in rows}
    ncols = len(STRESS_EXPERIMENTS)
    if ncols <= 0:
        raise ValueError("No stress experiments selected for plotting.")
    fig, axes = plt.subplots(2, ncols, figsize=(3.25 * ncols, 7.0), sharey="row", squeeze=False)
    for col, (experiment, _, title, xlabel, tokens) in enumerate(STRESS_EXPERIMENTS):
        x = np.arange(len(tokens))
        for row, metric in enumerate(["best_success", "auc_success"]):
            ax = axes[row][col]
            for _, method_label, color in PLOT_METHODS:
                y = [
                    row_index.get((experiment, label, method_label), {}).get(metric, "NA")
                    for _, label in tokens
                ]
                y = np.array([finite_float(v) for v in y], dtype=float)
                yerr = [
                    row_metric_sem(row_index.get((experiment, label, method_label), {}), metric)
                    for _, label in tokens
                ]
                valid = np.isfinite(y)
                if not np.any(valid):
                    continue
                if all(math.isnan(v) for v in yerr):
                    yerr = None
                else:
                    yerr = np.array(yerr, dtype=float)
                    yerr[~np.isfinite(yerr)] = 0.0
                ax.errorbar(
                    x[valid], y[valid],
                    yerr=None if yerr is None else yerr[valid],
                    marker="o",
                    linewidth=2,
                    elinewidth=0.7,
                    capsize=0,
                    ecolor="#3A3A3A",
                    color=color,
                    markeredgewidth=0,
                    label=method_label,
                )
            ax.set_title(title if row == 0 else "")
            ax.set_xticks(x, stress_tick_labels(experiment, tokens))
            ax.tick_params(axis="x", labelrotation=30 if len(tokens) > 5 else 0)
            ax.tick_params(width=0.6)
            ax.set_xlabel(xlabel)
            ax.set_ylim(0, 1.05)
            style_axis(ax)
            if col == 0:
                ax.set_ylabel("Success rate" if metric == "best_success" else "AUC")
            if all(math.isnan(v) for line in ax.lines for v in line.get_ydata()):
                ax.text(0.5, 0.5, "No runs found", ha="center", va="center", transform=ax.transAxes)
    fig.legend(
        [Line2D([0], [0], color=color, linewidth=2) for _, _, color in PLOT_METHODS],
        [label for _, label, _ in PLOT_METHODS],
        frameon=False,
        loc="lower center",
        ncol=len(METHODS),
        bbox_to_anchor=(0.5, 0.0),
        fontsize=18,
        handlelength=2.4,
        columnspacing=1.0,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out_dir / f"stress_sweep_robustness.{fmt}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, default=Path("results/TSIL/train_res"))
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    parser.add_argument("--table-dir", type=Path)
    parser.add_argument("--benchmark", default="mt01")
    parser.add_argument("--task-ids", type=int, nargs="+", required=True)
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    parser.add_argument("--x-key", default="misc/steps")
    parser.add_argument("--write-figures", action="store_true")
    parser.add_argument("--write-tables", action="store_true")
    parser.add_argument("--input-table", type=Path,
                        help="Use a precomputed CSV table instead of discovering training logs.")
    parser.add_argument("--exclude-experiments", nargs="*", default=[],
                        help="Experiment ids to omit from the report.")
    parser.add_argument("--min-target-steps-by-experiment", nargs="*", default=[],
                        help="Minimum run target steps per experiment, e.g. dense_dropout=50000000.")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.exclude_experiments:
        excluded = set(args.exclude_experiments)
        global STRESS_EXPERIMENTS
        STRESS_EXPERIMENTS = [
            experiment for experiment in STRESS_EXPERIMENTS
            if experiment[0] not in excluded
        ]
    min_target_items = args.min_target_steps_by_experiment
    args.min_target_steps_by_experiment = {}
    for item in min_target_items:
        if "=" not in item:
            raise SystemExit(f"Invalid --min-target-steps-by-experiment item: {item}")
        experiment, value = item.split("=", 1)
        args.min_target_steps_by_experiment[experiment] = float(value)
    if not args.write_figures and not args.write_tables:
        args.write_figures = True
        args.write_tables = True
    figure_dir = args.figure_dir or args.out_dir
    table_dir = args.table_dir or args.out_dir
    if args.write_figures and figure_dir is None:
        raise SystemExit("--figure-dir or --out-dir is required with --write-figures")
    if args.write_tables and table_dir is None:
        raise SystemExit("--table-dir or --out-dir is required with --write-tables")
    if args.write_figures:
        figure_dir.mkdir(parents=True, exist_ok=True)
    if args.write_tables:
        table_dir.mkdir(parents=True, exist_ok=True)

    if args.input_table is not None:
        stress_rows = read_table(args.input_table)
        missing = []
    else:
        stress_rows, _, missing = collect_stress(args)
    if args.write_figures:
        plot_stress(stress_rows, figure_dir, args.format)
    if args.write_tables:
        write_table(table_dir / "stress_sweep_summary.csv", stress_rows)
        write_markdown(table_dir / "stress_sweep_summary.md", stress_rows)
        compact_rows = compact_stress_rows(stress_rows)
        write_table(table_dir / "stress_sweep_5metrics_summary.csv", compact_rows)
        write_markdown(table_dir / "stress_sweep_5metrics_summary.md", compact_rows)
        aggregate_rows = aggregate_stress_rows(stress_rows)
        write_table(table_dir / "stress_sweep_aggregate_summary.csv", aggregate_rows)
        write_markdown(table_dir / "stress_sweep_aggregate_summary.md", aggregate_rows)
        with (table_dir / "missing_results.md").open("w") as f:
            if missing:
                f.write("# Missing Stability Results\n\n")
                for item in missing:
                    f.write(f"- {item}\n")
            else:
                f.write("# Missing Stability Results\n\nNone.\n")
    if args.strict and missing:
        raise SystemExit(f"Missing {len(missing)} expected stability result(s).")
    print("Wrote stability summary")


if __name__ == "__main__":
    main()
