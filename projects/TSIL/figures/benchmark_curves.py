"""TSIL benchmark training-curve and summary plots."""
from __future__ import annotations

from math import ceil
from typing import List, Optional

import numpy as np
from matplotlib import pyplot as plt

from core.plotting.primitives import _save_and_show
from core.plotting.style import (
    COLORS,
    FontSize,
    LegendSize,
    TickSize,
    YBins,
    apply_axis_power_scale,
    style_axis,
)
from core.plotting.train_data import (
    load_curves_from_run_dirs,
    interpolate_to_common_x,
    smooth_curve,
)


def _curve_parts(curve):
    x_vals, mean_y, std_y = curve[:3]
    count = int(curve[3]) if len(curve) >= 4 else 1
    return x_vals, mean_y, std_y, max(count, 1)


def _sem_from_std(std_y, count):
    std_y = np.asarray(std_y, dtype=float)
    return std_y / np.sqrt(float(count)) if count > 1 else np.zeros_like(std_y)


def _weighted_mean_std(values, stds, counts):
    values = np.asarray(values, dtype=float)
    stds = np.asarray(stds, dtype=float)
    weights = np.asarray(counts, dtype=float)
    total = float(np.sum(weights))
    if total <= 0.0:
        weights = np.ones(len(values), dtype=float)
        total = float(len(values))
    mean = np.average(values, axis=0, weights=weights)
    if total <= 1.0:
        std = np.zeros_like(mean, dtype=float)
    else:
        weight_shape = (len(weights),) + (1,) * max(values.ndim - 1, 0)
        sample_weights = weights.reshape(weight_shape)
        dof_weights = np.maximum(weights - 1.0, 0.0).reshape(weight_shape)
        within = dof_weights * (stds ** 2)
        between = sample_weights * ((values - mean) ** 2)
        std = np.sqrt(np.maximum(0.0, np.sum(within + between, axis=0) / (total - 1.0)))
    return mean, std, total


# ═══════════════════════════════════════════════════════════════════════
#  Statistics helpers
# ═══════════════════════════════════════════════════════════════════════

def _format_p_value(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 1e-4:
        return "<1e-4"
    if p_value < 1e-3:
        return f"{p_value:.1e}"
    return f"{p_value:.3f}"


def _format_named_p_value(name: str, p_value: float) -> str:
    formatted = _format_p_value(p_value)
    if formatted == "n/a":
        return f"{name}=n/a"
    if formatted.startswith("<"):
        return f"{name}{formatted}"
    return f"{name}={formatted}"


def _minimum_attainable_pvalue(
    n_nonzero_pairs: int,
    exact_threshold: int,
    num_samples: int,
) -> float:
    """Smallest attainable p-value for the current permutation regime."""
    if n_nonzero_pairs <= 0:
        return 1.0
    if n_nonzero_pairs <= exact_threshold:
        return 2.0 / float(1 << n_nonzero_pairs)
    return 1.0 / float(num_samples + 1)


def _holm_bonferroni_adjust(
    records: List[dict],
    key_in: str = "p_value",
    key_out: str = "p_value_adj",
) -> List[dict]:
    """In-place Holm-Bonferroni adjustment for a list of test records."""
    if not records:
        return records

    ordered = sorted(
        enumerate(records),
        key=lambda item: (
            np.inf if not np.isfinite(item[1].get(key_in, np.nan))
            else float(item[1][key_in])
        ),
    )

    n_tests = len(ordered)
    running_max = 0.0
    for rank, (original_idx, record) in enumerate(ordered):
        raw_p = float(record.get(key_in, np.nan))
        if not np.isfinite(raw_p):
            adjusted = np.nan
        else:
            adjusted = min(1.0, (n_tests - rank) * raw_p)
            adjusted = max(running_max, adjusted)
            running_max = adjusted
        records[original_idx][key_out] = adjusted
    return records


def _paired_permutation_pvalue(
    values_a: np.ndarray,
    values_b: np.ndarray,
    num_samples: int = 20000,
    seed: int = 0,
    exact_threshold: int = 15,
) -> dict:
    """Two-sided paired sign-flip permutation test on matched observations."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    full_diff = a[mask] - b[mask]
    diff = full_diff[np.abs(full_diff) > 1e-12]

    n_pairs = int(len(full_diff))
    n_nonzero = int(len(diff))
    if n_pairs == 0:
        return {
            "p_value": 1.0,
            "n_pairs": 0,
            "n_nonzero_pairs": 0,
            "mean_diff": 0.0,
            "median_diff": 0.0,
            "test_name": "paired permutation (degenerate)",
        }

    mean_diff = float(np.mean(full_diff))
    median_diff = float(np.median(full_diff))

    if n_nonzero == 0:
        return {
            "p_value": 1.0,
            "n_pairs": n_pairs,
            "n_nonzero_pairs": 0,
            "mean_diff": mean_diff,
            "median_diff": median_diff,
            "test_name": "paired permutation (all paired differences are zero)",
        }

    observed = float(abs(np.mean(diff)))

    if n_nonzero <= exact_threshold:
        n_perm = 1 << n_nonzero
        signs = (
            2 * (((np.arange(n_perm)[:, None] >> np.arange(n_nonzero)) & 1).astype(float)) - 1.0
        )
        permuted = np.abs((signs * diff[None, :]).mean(axis=1))
        extreme = int(np.sum(permuted >= observed - 1e-12))
        p_value = extreme / float(n_perm)
        test_name = f"paired permutation exact (2^{n_nonzero})"
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=float), size=(num_samples, n_nonzero))
        permuted = np.abs((signs * diff[None, :]).mean(axis=1))
        extreme = int(np.sum(permuted >= observed - 1e-12))
        p_value = (extreme + 1.0) / (num_samples + 1.0)
        test_name = f"paired permutation MC ({num_samples:,})"

    return {
        "p_value": float(p_value),
        "n_pairs": n_pairs,
        "n_nonzero_pairs": n_nonzero,
        "mean_diff": mean_diff,
        "median_diff": median_diff,
        "test_name": test_name,
    }


def _draw_significance_bracket(
    ax: plt.Axes,
    x0: float,
    x1: float,
    y: float,
    text: str,
    line_height: float,
    text_pad: float,
):
    ax.plot(
        [x0, x0, x1, x1],
        [y, y + line_height, y + line_height, y],
        color="black",
        linewidth=1.2,
        clip_on=False,
    )
    ax.text(
        (x0 + x1) / 2.0,
        y + line_height + text_pad,
        text,
        ha="center",
        va="bottom",
        fontsize=TickSize * 0.78,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════

def plot_training_curves_from_data(
    plot_data: dict,
    colors: List[str] = None,
    x_label: str = "Training Steps",
    y_label: str = "Success Rate",
    title: str = None,
    save_path: str = None,
    figsize: tuple = (10, 6),
    xlim: tuple = None,
    ylim: tuple = None,
    draw_legend: bool = True,
    line_width: float = 2.5,
    ax: plt.Axes = None,
):
    """Plot training curves from pre-processed ``{label: (x, mean, std)}``."""
    if colors is None:
        colors = COLORS[: len(plot_data)]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    for i, (label, curve) in enumerate(plot_data.items()):
        x_vals, mean_y, std_y, count = _curve_parts(curve)
        err_y = _sem_from_std(std_y, count) if len(curve) >= 4 else std_y
        color = colors[i] if i < len(colors) else None
        line = ax.plot(x_vals, mean_y, linewidth=line_width, label=label, color=color)
        ax.fill_between(
            x_vals,
            mean_y - err_y,
            mean_y + err_y,
            color=line[0].get_color(),
            alpha=0.18,
            linewidth=0,
        )
        print(f"Plotted '{label}': final mean={mean_y[-1]:.4f} ± {err_y[-1]:.4f}")

    ax.set_xlabel(x_label, fontsize=FontSize)
    ax.set_ylabel(y_label, fontsize=FontSize)
    if title:
        ax.set_title(title, fontsize=FontSize)
    ax.tick_params(axis="both", labelsize=TickSize)
    ax.locator_params(axis="y", nbins=YBins)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    style_axis(ax)
    apply_axis_power_scale(ax, "x")
    apply_axis_power_scale(ax, "y")
    ax.set_axisbelow(True)
    if draw_legend:
        ax.legend(fontsize=LegendSize, loc="best")
    plt.tight_layout()

    if save_path:
        _save_and_show(fig, save_path, show=False, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_wandb_training_curves(
    train_res_dirs: List[str],
    labels: List[str] = None,
    colors: List[str] = None,
    project: str = "TSIL",
    entity: str = None,
    metric_key: str = "reward/success",
    x_key: str = "misc/s_steps",
    x_label: str = "Training Steps",
    y_label: str = "Success Rate",
    title: str = None,
    save_path: str = None,
    figsize: tuple = (10, 6),
    xlim: tuple = None,
    ylim: tuple = None,
    num_interp_points: int = 500,
    smoothing_window: int = 1,
    draw_legend: bool = True,
    line_width: float = 2.5,
    start_time: str = None,
    end_time: str = None,
    use_cache: bool = True,
):
    """Convenience wrapper: load + interpolate + plot training curves."""
    print("Loading training curves...")
    all_runs_data = load_curves_from_run_dirs(
        train_res_dirs,
        project,
        entity,
        metric_key,
        x_key,
        start_time=start_time,
        end_time=end_time,
        use_cache=use_cache,
    )

    if labels is None:
        labels = list(all_runs_data.keys())
    if colors is None:
        colors = COLORS[: len(labels)]

    plot_data = {}
    for i, (dir_key, runs_data) in enumerate(all_runs_data.items()):
        if not runs_data:
            print(f"Warning: No data for {dir_key}")
            continue
        common_x, interp_ys = interpolate_to_common_x(runs_data, num_interp_points)
        if common_x is None:
            continue
        if smoothing_window > 1:
            interp_ys = np.array([smooth_curve(y, smoothing_window) for y in interp_ys])
        label = labels[i] if i < len(labels) else dir_key
        plot_data[label] = (
            common_x,
            np.mean(interp_ys, axis=0),
            np.zeros_like(interp_ys[0]) if len(interp_ys) <= 1 else np.std(interp_ys, axis=0, ddof=1),
        )
        print(f"Processed {dir_key}: {len(runs_data)} runs")

    return plot_training_curves_from_data(
        plot_data=plot_data,
        colors=colors,
        x_label=x_label,
        y_label=y_label,
        title=title,
        save_path=save_path,
        figsize=figsize,
        xlim=xlim,
        ylim=ylim,
        draw_legend=draw_legend,
        line_width=line_width,
    )


def plot_mt1_training_grid(
    grid_data: dict,
    task_ids: List[int],
    methods: Optional[List[str]] = None,
    method_labels: Optional[List[str]] = None,
    method_colors: Optional[List[str]] = None,
    x_label: str = "Training Steps",
    y_label: str = "Success Rate",
    n_cols: int = 4,
    figsize_per_subplot: tuple = (4, 3),
    line_width: float = 2.5,
    save_path: Optional[str] = None,
    show: bool = True,
    ylim: tuple = None,
    suptitle: Optional[str] = None,
    plain_y_axis: bool = False,
) -> plt.Figure:
    """Draw a grid of training-curve subplots (one per task)."""
    if methods is None:
        methods = grid_data.get("__methods__", [])
    if method_labels is None:
        method_labels = methods
    if method_colors is None:
        method_colors = COLORS[: len(methods)]

    n_tasks = len(task_ids)
    n_rows = ceil(n_tasks / n_cols)
    fig_w = figsize_per_subplot[0] * n_cols
    fig_h = figsize_per_subplot[1] * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for idx, tid in enumerate(task_ids):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        task_data = grid_data.get(tid, {})
        for mi, method in enumerate(methods):
            curve = task_data.get(method)
            if curve is None:
                continue
            x_vals, mean_y, std_y, count = _curve_parts(curve)
            err_y = _sem_from_std(std_y, count)
            color = method_colors[mi] if mi < len(method_colors) else None
            label = method_labels[mi] if mi < len(method_labels) else method
            line = ax.plot(x_vals, mean_y, linewidth=line_width, label=label, color=color)
            ax.fill_between(
                x_vals,
                mean_y - err_y,
                mean_y + err_y,
                color=line[0].get_color(),
                alpha=0.18,
                linewidth=0,
            )

        ax.set_title(f"T{tid}", fontsize=FontSize * 0.9)
        ax.tick_params(axis="both", labelsize=TickSize * 0.85)
        ax.locator_params(axis="y", nbins=YBins)
        if plain_y_axis:
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.yaxis.get_offset_text().set_visible(False)
        style_axis(ax)
        ax.set_axisbelow(True)
        if ylim:
            ax.set_ylim(*ylim)
        if row == n_rows - 1:
            ax.set_xlabel(x_label, fontsize=FontSize * 0.8)
        if col == 0:
            ax.set_ylabel(y_label, fontsize=FontSize * 0.8)
        apply_axis_power_scale(ax, "x")
        apply_axis_power_scale(ax, "y")

    for idx in range(n_tasks, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    handles, labels = [], []
    for ax in fig.axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)

    if suptitle:
        fig.suptitle(suptitle, fontsize=FontSize, y=1.04)

    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=len(methods),
            fontsize=LegendSize,
            frameon=False,
        )

    _save_and_show(fig, save_path, show, dpi=300, bbox_inches="tight")
    return fig


def _selected_x_by_method(grid_data: dict, task_ids: List[int], methods: List[str],
                          higher_is_better: bool = True) -> dict:
    selected_x = {}
    for method in methods:
        curves = [
            (curve[0], curve[1])
            for tid in task_ids
            for curve in [grid_data.get(tid, {}).get(method)]
            if curve is not None
        ]
        common_x, interp_ys = interpolate_to_common_x(curves, num_points=500)
        if common_x is None:
            continue
        mean_y = np.mean(interp_ys, axis=0)
        start_idx = max(int(0.9 * len(mean_y)), 0)
        tail_y = mean_y[start_idx:]
        idx = start_idx + (np.argmax(tail_y) if higher_is_better else np.argmin(tail_y))
        selected_x[method] = float(common_x[int(idx)])
    return selected_x


def plot_mt1_success_summary(
    grid_data: dict,
    task_ids: List[int],
    methods: Optional[List[str]] = None,
    method_labels: Optional[List[str]] = None,
    method_colors: Optional[List[str]] = None,
    x_label: str = "Training Steps",
    y_label: str = "Success Rate",
    curve_title: str = "Aggregate Success Curves",
    bar_title: Optional[str] = None,
    figsize: tuple = (24, 7),
    line_width: float = 2.5,
    save_path: Optional[str] = None,
    show: bool = True,
    ylim: tuple = (0.0, 1.05),
    annotate_pvalues: bool = True,
    higher_is_better: bool = True,
    bar_value_mode: str = "best",
    bar_selected_x: Optional[dict] = None,
    pvalue_num_samples: int = 20000,
    pvalue_seed: int = 0,
    integer_bar_labels: bool = False,
) -> plt.Figure:
    """Draw an MT1 summary figure with aggregate curves and summary bars."""
    if methods is None:
        methods = grid_data.get("__methods__", [])
    if method_labels is None:
        method_labels = methods
    if method_colors is None:
        method_colors = COLORS[: len(methods)]
    bar_value_name = "Selected" if bar_selected_x is not None else ("Best" if bar_value_mode == "best" else "Final")
    if bar_selected_x is None and bar_value_mode == "best":
        bar_selected_x = _selected_x_by_method(grid_data, task_ids, methods, higher_is_better)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1.0, 1.6]},
    )
    curve_ax, bar_ax = axes

    final_success = {}
    final_success_std = {}
    final_success_count = {}
    for mi, method in enumerate(methods):
        task_curves = []
        task_stds = []
        task_counts = []
        task_final = np.full(len(task_ids), np.nan, dtype=float)
        task_final_std = np.full(len(task_ids), np.nan, dtype=float)
        task_final_count = np.zeros(len(task_ids), dtype=float)
        selected_x = None if bar_selected_x is None else bar_selected_x.get(method)

        for ti, tid in enumerate(task_ids):
            curve = grid_data.get(tid, {}).get(method)
            if curve is None:
                continue
            x_vals, mean_y, std_y, count = _curve_parts(curve)
            task_curves.append((x_vals, mean_y))
            task_stds.append((x_vals, std_y))
            task_counts.append(count)
            task_final_count[ti] = count
            if selected_x is None:
                task_final[ti] = mean_y[-1]
                task_final_std[ti] = std_y[-1]
            else:
                task_final[ti] = np.interp(selected_x, x_vals, mean_y)
                task_final_std[ti] = np.interp(selected_x, x_vals, std_y)

        final_success[method] = task_final
        final_success_std[method] = task_final_std
        final_success_count[method] = task_final_count
        if not task_curves:
            continue

        common_x, interp_ys = interpolate_to_common_x(task_curves, num_points=500)
        if common_x is None:
            continue

        _, interp_stds = interpolate_to_common_x(task_stds, num_points=500)
        if interp_stds is None:
            continue
        mean_y, std_y, total_count = _weighted_mean_std(interp_ys, interp_stds, task_counts)
        err_y = _sem_from_std(std_y, total_count)
        color = method_colors[mi] if mi < len(method_colors) else None
        label = method_labels[mi] if mi < len(method_labels) else method

        line = curve_ax.plot(
            common_x,
            mean_y,
            linewidth=line_width,
            label=label,
            color=color,
        )
        curve_ax.fill_between(
            common_x,
            mean_y - err_y,
            mean_y + err_y,
            color=line[0].get_color(),
            alpha=0.18,
            linewidth=0,
        )
        print(
            f"Summary '{method}': aggregate final="
            f"{mean_y[-1]:.4f}±{err_y[-1]:.4f} SEM "
            f"across {int(total_count)} run(s)"
        )

    curve_ax.set_xlabel(x_label, fontsize=FontSize)
    curve_ax.set_ylabel(y_label, fontsize=FontSize)
    curve_ax.set_title(curve_title, fontsize=FontSize)
    curve_ax.tick_params(axis="both", labelsize=TickSize)
    curve_ax.locator_params(axis="y", nbins=YBins)
    style_axis(curve_ax)
    curve_ax.set_axisbelow(True)
    if ylim is not None:
        curve_ax.set_ylim(*ylim)
    apply_axis_power_scale(curve_ax, "x")
    apply_axis_power_scale(curve_ax, "y")
    curve_ax.legend(fontsize=LegendSize, loc="best")

    bar_positions = []
    bar_heights = []
    bar_errors = []
    bar_labels = []
    bar_colors = []
    plotted_methods = []
    for mi, method in enumerate(methods):
        values = final_success.get(method)
        if values is None:
            continue
        valid_mask = ~np.isnan(values)
        valid_values = values[valid_mask]
        if len(valid_values) == 0:
            continue

        valid_counts = final_success_count[method][valid_mask]
        avg_final, std_final, total_count = _weighted_mean_std(
            valid_values,
            final_success_std[method][valid_mask],
            valid_counts,
        )
        avg_final = float(avg_final)
        sem_final = float(_sem_from_std(std_final, total_count))
        label = method_labels[mi] if mi < len(method_labels) else method
        color = method_colors[mi] if mi < len(method_colors) else None
        bar_positions.append(len(bar_positions))
        bar_heights.append(avg_final)
        bar_errors.append(sem_final)
        bar_labels.append(label)
        bar_colors.append(color)
        plotted_methods.append(method)
        print(
            f"Bar '{method}': avg {bar_value_name.lower()}={avg_final:.4f} ± {sem_final:.4f} SEM "
            f"across {int(total_count)} run(s)"
        )

    if bar_positions:
        bars = bar_ax.bar(
            bar_positions,
            bar_heights,
            yerr=bar_errors,
            color=bar_colors,
            linewidth=0,
            error_kw={"ecolor": "#3A3A3A", "elinewidth": 0.7, "capsize": 0},
            alpha=0.95,
        )
        label_pad = 0.015 * max(max(bar_heights) if bar_heights else 1.0, 1.0)
        for bar, value in zip(bars, bar_heights):
            bar_ax.text(
                bar.get_x() + bar.get_width() * 0.25,
                value + label_pad,
                f"{value:.0f}" if integer_bar_labels else f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=TickSize * 0.8,
            )

    if bar_title:
        bar_ax.set_title(bar_title, fontsize=FontSize)
    bar_ax.set_ylabel(f"Average {bar_value_name} {y_label}", fontsize=FontSize)
    bar_ax.set_xticks(bar_positions)
    bar_ax.set_xticklabels(
        bar_labels,
        rotation=20,
        ha="right",
        fontsize=TickSize * 0.85,
    )
    bar_ax.tick_params(axis="y", labelsize=TickSize)
    bar_ax.locator_params(axis="y", nbins=YBins)
    style_axis(bar_ax)
    bar_ax.set_axisbelow(True)
    if bar_positions:
        bar_ax.set_xlim(-0.6, len(bar_positions) - 0.4)
    if ylim is not None:
        bar_ax.set_ylim(*ylim)
    apply_axis_power_scale(bar_ax, "y")

    if annotate_pvalues and len(plotted_methods) >= 2:
        pairwise_stats = []
        for i, method_a in enumerate(plotted_methods):
            for j in range(i + 1, len(plotted_methods)):
                method_b = plotted_methods[j]
                label_a = bar_labels[i]
                label_b = bar_labels[j]
                stats = _paired_permutation_pvalue(
                    final_success[method_a],
                    final_success[method_b],
                    num_samples=pvalue_num_samples,
                    seed=pvalue_seed + i * 1009 + j,
                )
                pairwise_stats.append({
                    "method_a": method_a,
                    "method_b": method_b,
                    "label_a": label_a,
                    "label_b": label_b,
                    **stats,
                })

        print(
            f"\nPairwise {bar_value_name.lower()} {y_label} statistics "
            f"(paired permutation test on task-wise {bar_value_name.lower()} values):"
        )
        for record in sorted(pairwise_stats, key=lambda rec: rec["p_value"]):
            print(
                f"  {record['label_a']} vs {record['label_b']}: "
                f"delta={record['mean_diff']:+.4f}, "
                f"{_format_named_p_value('p_raw', record['p_value'])}, "
                f"n={record['n_pairs']}, nz={record['n_nonzero_pairs']} "
                f"[{record['test_name']}]"
            )

        bar_scores = np.asarray(bar_heights, dtype=float)
        best_idx = int(
            np.argmax(bar_scores) if higher_is_better else np.argmin(bar_scores)
        )
        best_method = plotted_methods[best_idx]
        best_label = bar_labels[best_idx]

        ref_stats = []
        for record in pairwise_stats:
            if record["method_a"] == best_method or record["method_b"] == best_method:
                other_label = (
                    record["label_b"]
                    if record["method_a"] == best_method
                    else record["label_a"]
                )
                signed_delta = (
                    record["mean_diff"]
                    if record["method_a"] == best_method
                    else -record["mean_diff"]
                )
                ref_stats.append({
                    **record,
                    "other_label": other_label,
                    "signed_delta": signed_delta,
                })

        if ref_stats:
            _holm_bonferroni_adjust(ref_stats)
            direction = "higher" if higher_is_better else "lower"
            print(
                f"\nBest-vs-other Holm-corrected statistics "
                f"({direction} is better; reference: {best_label}):"
            )
            for record in sorted(ref_stats, key=lambda rec: rec["p_value"]):
                print(
                    f"  {best_label} vs {record['other_label']}: "
                    f"d={record['signed_delta']:+.4f}, "
                    f"{_format_named_p_value('p_raw', record['p_value'])}, "
                    f"{_format_named_p_value('p_h', record['p_value_adj'])}"
                )

            idx_by_method = {
                method: idx for idx, method in enumerate(plotted_methods)
            }
            top_of_bar = {
                idx: bar_heights[idx] + bar_errors[idx]
                for idx in range(len(bar_heights))
            }
            y_lo, y_hi = bar_ax.get_ylim()
            y_span = y_hi - y_lo
            base_y = max(top_of_bar.values()) + 0.045 * y_span
            line_height = 0.014 * y_span
            text_pad = 0.008 * y_span
            level_gap = 0.055 * y_span

            left_records = []
            right_records = []
            for record in ref_stats:
                other_method = (
                    record["method_b"]
                    if record["method_a"] == best_method
                    else record["method_a"]
                )
                other_idx = idx_by_method[other_method]
                enriched = {
                    **record,
                    "other_idx": other_idx,
                    "span": abs(other_idx - best_idx),
                }
                if other_idx < best_idx:
                    left_records.append(enriched)
                else:
                    right_records.append(enriched)

            left_records.sort(key=lambda rec: rec["other_idx"], reverse=True)
            right_records.sort(key=lambda rec: rec["other_idx"])

            max_levels = max(len(left_records), len(right_records), 1)
            required_top = (
                base_y
                + (max_levels - 1) * level_gap
                + line_height
                + text_pad
                + 0.045 * y_span
            )
            if required_top > y_hi:
                bar_ax.set_ylim(y_lo, required_top)
                y_lo, y_hi = bar_ax.get_ylim()
                y_span = y_hi - y_lo
                base_y = max(top_of_bar.values()) + 0.045 * y_span
                line_height = 0.014 * y_span
                text_pad = 0.008 * y_span
                level_gap = 0.055 * y_span

            for level, record in enumerate(left_records):
                p_min = _minimum_attainable_pvalue(
                    record["n_nonzero_pairs"],
                    exact_threshold=15,
                    num_samples=pvalue_num_samples,
                )
                _draw_significance_bracket(
                    bar_ax,
                    bar_positions[record["other_idx"]],
                    bar_positions[best_idx],
                    base_y + level * level_gap,
                    (
                        f"{_format_named_p_value('p_raw', record['p_value'])} "
                        f"(min={_format_p_value(p_min)})"
                    ),
                    line_height=line_height,
                    text_pad=text_pad,
                )

            for level, record in enumerate(right_records):
                p_min = _minimum_attainable_pvalue(
                    record["n_nonzero_pairs"],
                    exact_threshold=15,
                    num_samples=pvalue_num_samples,
                )
                _draw_significance_bracket(
                    bar_ax,
                    bar_positions[best_idx],
                    bar_positions[record["other_idx"]],
                    base_y + level * level_gap,
                    (
                        f"{_format_named_p_value('p_raw', record['p_value'])} "
                        f"(min={_format_p_value(p_min)})"
                    ),
                    line_height=line_height,
                    text_pad=text_pad,
                )

    _save_and_show(fig, save_path, show, dpi=300, bbox_inches="tight")
    return fig
