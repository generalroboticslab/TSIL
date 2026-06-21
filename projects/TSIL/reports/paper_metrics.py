"""Paper-level benchmark metric tables and compact summary figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from core.plotting.train_data import (
    _load_local_plot_metric_history,
)
from projects.TSIL.reports.curve_metrics import (
    _clean_curve,
    _curve_metrics,
    _interp_at,
    _last_value,
    _mean_std,
    _tail_start_x,
)
from projects.TSIL.ckpt_layout import discover_benchmark_task_methods


SIGNAL_METRICS = ("fast_success", "slow_failure", "success_focused", "reward_distracted")
SIL_METRICS = {
    "archive_best_eps_time": "train/sil_archive_best_eps_time",
}
POSITIVE_GAP_NLL_KEY = "train/sil_supervised_nll_topk_mean"
POSITIVE_GAP_ACTIVITY_KEY = "train/sil_supervised_weight_frac"


def _fmt(value, digits=4):
    if value is None or not math.isfinite(float(value)):
        return "NA"
    if digits == 0:
        return int(round(float(value)))
    return round(float(value), digits)


def _fmt_sci(value, digits=4):
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}e}"


def _success_curve(run, x_key):
    direct = _load_local_plot_metric_history(run, x_key, "reward/success")
    if direct is not None:
        return direct

    success_eps = _clean_curve(
        _load_local_plot_metric_history(run, x_key, "misc/success_episodes"),
        min_points=1,
    )
    total_eps = _clean_curve(
        _load_local_plot_metric_history(run, x_key, "misc/episodes"),
        min_points=1,
    )
    if success_eps is None or total_eps is None:
        return None
    x = np.union1d(success_eps[0], total_eps[0])
    successes = np.interp(x, success_eps[0], success_eps[1])
    totals = np.interp(x, total_eps[0], total_eps[1])
    mask = totals > 0
    if not mask.any():
        return None
    return x[mask], successes[mask] / totals[mask]


def _method_experiments(args):
    method_experiments = args.method_experiments or [args.experiment] * len(args.methods)
    if len(method_experiments) != len(args.methods):
        raise ValueError("--method-experiments must match --methods length")
    return method_experiments


def _runs_by_method(args):
    runs_by_method = {}
    for method, experiment in zip(args.methods, _method_experiments(args)):
        runs_by_method[method] = discover_benchmark_task_methods(
            str(args.train_root),
            args.benchmark,
            args.task_ids,
            train_stage=args.train_stage,
            methods=[method],
            experiment=experiment,
        )
    return runs_by_method


def _collect_run_rows(args):
    labels = args.legends or args.methods
    label_by_method = {
        method: labels[idx] if idx < len(labels) else method
        for idx, method in enumerate(args.methods)
    }
    runs_by_method = _runs_by_method(args)

    rows = []
    for task_id in args.task_ids:
        for method in args.methods:
            runs = runs_by_method.get(method, {}).get(task_id, {}).get(method, [])
            for run in runs:
                success = _success_curve(run, args.x_key)
                metrics = _curve_metrics(
                    success,
                    threshold=args.threshold,
                    tail_frac=args.tail_frac,
                    start_at_zero=True,
                )
                if metrics is None:
                    continue
                eps_time_curve = _load_local_plot_metric_history(
                    run,
                    args.x_key,
                    "signal/success_eps_time",
                )
                selected_eps_time = _interp_at(eps_time_curve, metrics["tail_best_x"])
                success_episodes = _last_value(
                    _load_local_plot_metric_history(
                        run,
                        args.x_key,
                        "misc/success_episodes",
                    )
                )
                rows.append({
                    "task_id": task_id,
                    "method": method,
                    "label": label_by_method[method],
                    "run": Path(run).name,
                    **metrics,
                    "selected_success_eps_time": selected_eps_time,
                    "success_episodes": success_episodes,
                })
    return rows


def _summary_rows(run_rows, methods, legends, threshold):
    labels = legends or methods
    rows = []
    for idx, method in enumerate(methods):
        label = labels[idx] if idx < len(labels) else method
        subset = [row for row in run_rows if row["method"] == method]
        success_mean, success_std, success_n = _mean_std(
            row["tail_best_success"] for row in subset
        )
        auc_mean, auc_std, auc_n = _mean_std(row["auc_success"] for row in subset)
        steps_mean, steps_std, steps_n = _mean_std(
            row["steps_to_threshold"] for row in subset
        )
        time_mean, time_std, time_n = _mean_std(
            row["selected_success_eps_time"] for row in subset
        )
        episodes_mean, episodes_std, episodes_n = _mean_std(
            row["success_episodes"] for row in subset
        )
        rows.append({
            "method": method,
            "label": label,
            "run_count": len(subset),
            "last10_best_success_mean": _fmt(success_mean),
            "last10_best_success_std": _fmt(success_std),
            "auc_success_mean": _fmt(auc_mean),
            "auc_success_std": _fmt(auc_std),
            f"steps_to_{int(threshold * 100)}_mean": _fmt(steps_mean, 0),
            f"steps_to_{int(threshold * 100)}_std": _fmt(steps_std, 0),
            f"steps_to_{int(threshold * 100)}_n": steps_n,
            "selected_success_eps_time_mean": _fmt(time_mean),
            "selected_success_eps_time_std": _fmt(time_std),
            "selected_success_eps_time_n": time_n,
            "success_episodes_mean": _fmt_sci(episodes_mean),
            "success_episodes_std": _fmt_sci(episodes_std),
            "success_episodes_n": episodes_n,
            "_success_mean": success_mean,
            "_success_std": success_std,
            "_success_n": success_n,
            "_auc_n": auc_n,
        })
    return rows


def _read_jsonl(path):
    records = []
    try:
        with open(path, "r") as file_obj:
            for line in file_obj:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        return []
    return records


def _last_frac_records(records, last_frac):
    if not records or float(last_frac) >= 1.0:
        return records
    iterations = np.asarray([float(record.get("iteration", 0.0)) for record in records], dtype=float)
    cutoff = float(np.quantile(iterations, 1.0 - float(last_frac)))
    return [record for record in records if float(record.get("iteration", 0.0)) >= cutoff]


def _signal_summary_metrics(records):
    masses = np.asarray([float(record.get("positive_adv_mass_update", 0.0)) for record in records], dtype=float)
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        return {name: 0.0 for name in SIGNAL_METRICS}
    successes = np.asarray([float(bool(record.get("success", False))) for record in records], dtype=float)
    times = np.asarray([
        float(record.get("eps_time", 0.0)) / max(float(record.get("max_eps_time", 1.0)), 1e-8)
        for record in records
    ], dtype=float)
    dense_returns = np.asarray([float(record.get("dense_return", 0.0)) for record in records], dtype=float)
    failures = 1.0 - successes
    dense_cut = np.quantile(dense_returns, 0.75)
    return {
        "fast_success": float(np.sum(masses * successes * np.clip(1.0 - times, 0.0, 1.0)) / total_mass),
        "slow_failure": float(np.sum(masses * failures * np.clip(times, 0.0, 1.0)) / total_mass),
        "success_focused": float(np.sum(masses * successes) / total_mass),
        "reward_distracted": float(np.sum(masses * failures * (dense_returns >= dense_cut)) / total_mass),
    }


def _task_balanced_metric_values(rows, metric):
    values = []
    for task_id in sorted({row.get("task_id") for row in rows}):
        mean, _, n = _mean_std(
            row.get(metric, math.nan) for row in rows if row.get("task_id") == task_id
        )
        if n > 0:
            values.append(mean)
    return values


def _numeric_summary_rows(run_rows, methods, legends, metric_names, task_balanced=False):
    labels = legends or methods
    rows = []
    for idx, method in enumerate(methods):
        subset = [row for row in run_rows if row["method"] == method]
        out = {
            "method": method,
            "label": labels[idx] if idx < len(labels) else method,
            "run_count": len(subset),
        }
        for metric in metric_names:
            values = (
                _task_balanced_metric_values(subset, metric)
                if task_balanced
                else (row.get(metric, math.nan) for row in subset)
            )
            mean, std, n = _mean_std(values)
            out[f"{metric}_mean"] = _fmt(mean)
            out[f"{metric}_std"] = _fmt(std)
            out[f"{metric}_n"] = n
        rows.append(out)
    return rows


def _collect_signal_rows(args):
    labels = args.legends or args.methods
    label_by_method = {
        method: labels[idx] if idx < len(labels) else method
        for idx, method in enumerate(args.methods)
    }
    runs_by_method = _runs_by_method(args)
    rows = []
    for task_id in args.task_ids:
        for method in args.methods:
            runs = runs_by_method.get(method, {}).get(task_id, {}).get(method, [])
            for run in runs:
                records = _read_jsonl(Path(run) / "trajectories" / "training_episode_signal_history.jsonl")
                records = _last_frac_records(records, args.signal_last_frac)
                if not records:
                    continue
                rows.append({
                    "task_id": task_id,
                    "method": method,
                    "label": label_by_method[method],
                    "run": Path(run).name,
                    **_signal_summary_metrics(records),
                })
    return rows


def _tail_mean(curve, tail_frac):
    cleaned = _clean_curve(curve, min_points=1)
    if cleaned is None:
        return math.nan
    x, y = cleaned
    tail_mask = x >= _tail_start_x(x, tail_frac)
    if not tail_mask.any():
        tail_mask[-1] = True
    return float(np.mean(y[tail_mask]))


def _masked_curve_value_at_frac(curve, frac, active_curve=None, positive_only=False):
    cleaned = _clean_curve(curve, min_points=1)
    if cleaned is None:
        return math.nan
    x, y = cleaned
    frac = min(max(float(frac), 0.0), 1.0)
    target_x = float(x[0]) + frac * float(x[-1] - x[0])
    mask = x <= target_x
    if not mask.any():
        mask[0] = True
    if positive_only:
        mask &= y > 0.0
    active_cleaned = _clean_curve(active_curve, min_points=1)
    if active_cleaned is not None:
        active_x, active_y = active_cleaned
        active_by_x = {float(x_val): float(y_val) for x_val, y_val in zip(active_x, active_y)}
        mask &= np.asarray([active_by_x.get(float(x_val), 0.0) > 0.0 for x_val in x], dtype=bool)
    valid_idx = np.flatnonzero(mask)
    if len(valid_idx) == 0:
        return math.nan
    return float(y[valid_idx[-1]])


def _collect_sil_rows(args):
    labels = args.legends or args.methods
    label_by_method = {
        method: labels[idx] if idx < len(labels) else method
        for idx, method in enumerate(args.methods)
    }
    runs_by_method = _runs_by_method(args)
    rows = []
    for task_id in args.task_ids:
        for method in args.methods:
            runs = runs_by_method.get(method, {}).get(task_id, {}).get(method, [])
            for run in runs:
                row = {
                    "task_id": task_id,
                    "method": method,
                    "label": label_by_method[method],
                    "run": Path(run).name,
                }
                for name, metric_key in SIL_METRICS.items():
                    row[name] = _tail_mean(
                        _load_local_plot_metric_history(run, args.x_key, metric_key),
                        args.tail_frac,
                    )
                positive_gap_curve = _load_local_plot_metric_history(run, args.x_key, POSITIVE_GAP_NLL_KEY)
                positive_gap_activity = _load_local_plot_metric_history(run, args.x_key, POSITIVE_GAP_ACTIVITY_KEY)
                row["positive_gap_nll_50pct"] = _masked_curve_value_at_frac(
                    positive_gap_curve,
                    0.5,
                    active_curve=positive_gap_activity,
                    positive_only=True,
                )
                rows.append(row)
    return rows


def _write_csv(path, rows, include_private=False):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [key for key in rows[0] if include_private or not key.startswith("_")]
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _write_markdown(path, rows):
    if not rows:
        return
    keys = [key for key in rows[0] if not key.startswith("_")]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file_obj:
        file_obj.write("| " + " | ".join(keys) + " |\n")
        file_obj.write("| " + " | ".join(["---"] * len(keys)) + " |\n")
        for row in rows:
            file_obj.write("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |\n")


def write_tables(table_dir, summary_rows, run_rows, prefix="main_metrics"):
    table_dir = Path(table_dir)
    _write_csv(table_dir / "summary_csv" / f"{prefix}_summary.csv", summary_rows)
    _write_markdown(table_dir / "summary_md" / f"{prefix}_summary.md", summary_rows)
    _write_csv(table_dir / "raw" / f"{prefix}_runs.csv", run_rows, include_private=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, default=Path("results/TSIL/train_res"))
    parser.add_argument("--benchmark", default="mt01")
    parser.add_argument("--experiment", default="compare_temporal")
    parser.add_argument("--training-stage", dest="train_stage", default="scratch")
    parser.add_argument("--task-ids", type=int, nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--method-experiments", nargs="+")
    parser.add_argument("--legends", nargs="+")
    parser.add_argument("--x-key", default="misc/steps")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--tail-frac", type=float, default=0.1)
    parser.add_argument("--table-dir", type=Path)
    parser.add_argument("--table-prefix", default="main_metrics")
    parser.add_argument("--write-table", action="store_true")
    parser.add_argument("--write-signal-table", action="store_true")
    parser.add_argument("--write-sil-table", action="store_true")
    parser.add_argument("--signal-last-frac", type=float, default=0.25)
    return parser.parse_args()


def main():
    args = parse_args()
    if not (args.write_table or args.write_signal_table or args.write_sil_table):
        args.write_table = True
    if (args.write_table or args.write_signal_table or args.write_sil_table) and args.table_dir is None:
        raise SystemExit("--table-dir is required with table-writing modes")

    if args.write_table:
        run_rows = _collect_run_rows(args)
        summary_rows = _summary_rows(run_rows, args.methods, args.legends, args.threshold)
        write_tables(args.table_dir, summary_rows, run_rows, prefix=args.table_prefix)
        print(f"Wrote paper metric tables to {args.table_dir}")
    if args.write_signal_table:
        run_rows = _collect_signal_rows(args)
        summary_rows = _numeric_summary_rows(
            run_rows,
            args.methods,
            args.legends,
            SIGNAL_METRICS,
            task_balanced=True,
        )
        write_tables(args.table_dir, summary_rows, run_rows, prefix=args.table_prefix)
        print(f"Wrote learning-signal tables to {args.table_dir}")
    if args.write_sil_table:
        run_rows = _collect_sil_rows(args)
        sil_metric_names = tuple(SIL_METRICS) + ("positive_gap_nll_50pct",)
        summary_rows = _numeric_summary_rows(run_rows, args.methods, args.legends, sil_metric_names)
        write_tables(args.table_dir, summary_rows, run_rows, prefix=args.table_prefix)
        print(f"Wrote TSIL/SIL diagnostic tables to {args.table_dir}")


if __name__ == "__main__":
    main()
