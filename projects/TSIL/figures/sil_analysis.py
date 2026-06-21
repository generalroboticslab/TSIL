"""TSIL/SIL revisit diagnostic figures."""

from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from core.plotting.primitives import _save_and_show
from core.plotting.style import (
    COLORS,
    FontSize,
    LegendSize,
    apply_axis_power_scale,
    style_axis,
)
from core.plotting.train_data import interpolate_to_common_x


def _read_jsonl(path):
    records = []
    with open(path, "r") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _record_value(record, key):
    candidates = [key]
    if "/" in key:
        candidates.append(key.split("/", 1)[1])
        candidates.append(key.split("/")[-1])
    aliases = {
        "reward/success": ("success", "reward_success", "success_rate"),
        "success": ("reward/success", "reward_success", "success_rate"),
        "sil_revisit_nll_topk_mean": ("train/sil_revisit_nll_topk_mean",),
        "sil_revisit_logp_topk_mean": ("train/sil_revisit_logp_topk_mean",),
        "sil_supervised_nll_topk_mean": ("train/sil_supervised_nll_topk_mean",),
        "sil_supervised_logp_topk_mean": ("train/sil_supervised_logp_topk_mean",),
        "sil_supervised_weight_frac": ("train/sil_supervised_weight_frac",),
        "fast_success_rate": ("signal/fast_success_rate", "signal/sil_fast_success_rate", "train/sil_fast_success_rate"),
        "sil_archive_best_eps_time": ("train/sil_archive_best_eps_time",),
        "fast_revisit_gap_steps": ("signal/fast_revisit_gap_steps", "signal/sil_revisit_gap_steps", "train/sil_revisit_gap_steps"),
        "sil_fast_success_rate": ("signal/fast_success_rate", "signal/sil_fast_success_rate", "train/sil_fast_success_rate"),
        "sil_revisit_gap_steps": ("signal/fast_revisit_gap_steps", "signal/sil_revisit_gap_steps", "train/sil_revisit_gap_steps"),
    }
    candidates.extend(aliases.get(key, ()))
    for candidate in candidates:
        value = record.get(candidate)
        if value is not None:
            return float(value)
    return None


def _curve_from_records(records, y_key, x_key="steps", require_archive=False):
    xs, ys = [], []
    for record in records:
        x_val = record.get(x_key)
        if x_val is None and x_key == "steps":
            x_val = record.get("misc/steps")
        y_val = _record_value(record, y_key)
        if require_archive:
            ref_count = _record_value(record, "sil_revisit_reference_count")
            if ref_count is None or ref_count <= 0:
                continue
        if y_key.startswith("sil_supervised_"):
            weight_frac = _record_value(record, "sil_supervised_weight_frac")
            if weight_frac is None or weight_frac <= 0.0:
                continue
        if x_val is None or y_val is None:
            continue
        xs.append(float(x_val))
        ys.append(float(y_val))
    if not xs:
        return None
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if not mask.any():
        return None
    order = np.argsort(xs[mask])
    return xs[mask][order], ys[mask][order]


def _group_histories(signal_history_paths, labels=None):
    labels = list(labels or [])
    groups = OrderedDict()
    for idx, path in enumerate(signal_history_paths):
        label = labels[idx] if idx < len(labels) else os.path.basename(os.path.dirname(str(path)))
        groups.setdefault(label, []).append(_read_jsonl(path))
    return groups


def _aggregate_group_curves(histories, y_key, x_key="steps", require_archive=False):
    curves = [
        curve
        for curve in (_curve_from_records(records, y_key, x_key=x_key, require_archive=require_archive) for records in histories)
        if curve is not None
    ]
    if not curves:
        return None
    common_x, interp_ys = interpolate_to_common_x(curves, num_points=500)
    if common_x is None:
        return None
    std_y = np.zeros_like(interp_ys[0]) if len(curves) <= 1 else np.std(interp_ys, axis=0, ddof=1)
    err_y = std_y / math.sqrt(len(curves)) if len(curves) > 1 else np.zeros_like(std_y)
    return common_x, np.mean(interp_ys, axis=0), err_y


def _plot_group_curve(ax, groups, y_key, ylabel, x_key="steps", ylim=None, require_archive=False):
    for idx, (label, histories) in enumerate(groups.items()):
        curve = _aggregate_group_curves(histories, y_key, x_key=x_key, require_archive=require_archive)
        if curve is None:
            continue
        x_vals, mean_y, std_y = curve
        color = COLORS[idx % len(COLORS)]
        ax.plot(x_vals, mean_y, label=label, color=color, linewidth=2.5)
        ax.fill_between(x_vals, mean_y - std_y, mean_y + std_y, color=color, alpha=0.18, linewidth=0)
    style_axis(ax)
    ax.set_xlabel("Steps")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _steps_to_threshold(records, threshold=0.8):
    curve = _curve_from_records(records, "success", x_key="steps")
    if curve is None:
        return np.nan
    xs, ys = curve
    hits = np.flatnonzero(ys >= float(threshold))
    return float(xs[int(hits[0])]) if hits.size > 0 else np.nan


def _last_value(records, key):
    for record in reversed(records):
        value = _record_value(record, key)
        if value is not None and np.isfinite(value):
            return float(value)
    return np.nan


def _format_step_axis(ax, axis="x", exponent=7):
    if axis in ("x", "both"):
        apply_axis_power_scale(ax, "x", exponent=exponent)
    if axis in ("y", "both"):
        apply_axis_power_scale(ax, "y", exponent=exponent)


def plot_sil_revisit_mechanism(
    signal_history_paths,
    labels=None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize=(20.0, 5.0),
):
    groups = _group_histories(signal_history_paths, labels=labels)
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    _plot_group_curve(
        axes[0],
        groups,
        "sil_revisit_nll_topk_mean",
        "Absolute Fast-Memory NLL",
        require_archive=True,
    )
    _plot_group_curve(
        axes[1],
        groups,
        "sil_supervised_nll_topk_mean",
        "Positive-Gap Fast-Memory NLL",
        require_archive=True,
    )
    _plot_group_curve(
        axes[2],
        groups,
        "fast_success_rate",
        "Fast Success Rate",
        ylim=(0.0, 1.05),
        require_archive=True,
    )
    _plot_group_curve(
        axes[3],
        groups,
        "sil_archive_best_eps_time",
        "Success Time (lower is faster)",
        require_archive=True,
    )
    for ax in axes:
        _format_step_axis(ax, "x")
    axes[0].set_title("Absolute Memory Difficulty", fontsize=FontSize)
    axes[1].set_title("Positive-Gap Memory Difficulty", fontsize=FontSize)
    axes[2].set_title("Fast-Success Re-Collection", fontsize=FontSize)
    axes[3].set_title("Best Archived Success Time", fontsize=FontSize)
    handles, labels_out = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(
            handles,
            labels_out,
            loc="best",
            fontsize=LegendSize * 0.85,
            frameon=False,
        )
    _save_and_show(fig, save_path, show, dpi=300, bbox_inches="tight")
    return fig


def plot_sil_revisit_outcome(
    signal_history_paths,
    labels=None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize=(15.0, 5.0),
):
    groups = _group_histories(signal_history_paths, labels=labels)
    fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw={"width_ratios": [1.4, 1.0, 1.0]})
    _plot_group_curve(axes[0], groups, "success", "Success Rate", ylim=(0.0, 1.05))
    _format_step_axis(axes[0], "x")
    axes[0].set_title("Learning Curve", fontsize=FontSize)

    thresholds = (0.8, 0.9)
    x = np.arange(len(groups), dtype=float)
    width = 0.34
    for offset, threshold in zip((-0.5 * width, 0.5 * width), thresholds):
        values = []
        errors = []
        for histories in groups.values():
            steps = np.asarray([_steps_to_threshold(records, threshold) for records in histories], dtype=float)
            finite_steps = steps[np.isfinite(steps)]
            values.append(float(np.mean(finite_steps)) if finite_steps.size else np.nan)
            errors.append(float(np.std(finite_steps, ddof=1) / math.sqrt(finite_steps.size)) if finite_steps.size > 1 else 0.0)
        values = np.asarray(values, dtype=float)
        errors = np.asarray(errors, dtype=float)
        finite = np.isfinite(values)
        if finite.any():
            axes[1].bar(
                x[finite] + offset,
                values[finite],
                width=width,
                yerr=errors[finite],
                linewidth=0,
                error_kw={"ecolor": "#3A3A3A", "elinewidth": 0.7, "capsize": 0},
                label=f"{int(threshold * 100)}%",
            )
    style_axis(axes[1])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(list(groups.keys()), rotation=20, ha="right")
    axes[1].set_ylabel("Steps")
    _format_step_axis(axes[1], "y")
    axes[1].set_title("Steps To High Success", fontsize=FontSize)
    axes[1].legend(frameon=False, loc="upper right", fontsize=LegendSize * 0.9)

    for idx, (label, histories) in enumerate(groups.items()):
        color = COLORS[idx % len(COLORS)]
        xs, ys = [], []
        for records in histories:
            gap = _last_value(records, "fast_revisit_gap_steps")
            steps80 = _steps_to_threshold(records, 0.8)
            if np.isfinite(gap) and float(gap) >= 0.0 and np.isfinite(steps80):
                xs.append(gap)
                ys.append(steps80)
        if xs:
            xs_arr = np.asarray(xs, dtype=float)
            ys_arr = np.asarray(ys, dtype=float)
            axes[2].errorbar(
                float(np.mean(xs_arr)),
                float(np.mean(ys_arr)),
                xerr=float(np.std(xs_arr, ddof=1) / math.sqrt(len(xs_arr))) if len(xs_arr) > 1 else None,
                yerr=float(np.std(ys_arr, ddof=1) / math.sqrt(len(ys_arr))) if len(ys_arr) > 1 else None,
                fmt="o",
                color=color,
                ecolor="#3A3A3A",
                markersize=7,
                capsize=0,
                elinewidth=0.7,
                label=label,
            )
    style_axis(axes[2])
    axes[2].set_xlabel("First Revisit Gap (steps)")
    axes[2].set_ylabel("Steps To 80% Success")
    _format_step_axis(axes[2], "both")
    axes[2].set_title("Revisit Gap vs Sample Efficiency", fontsize=FontSize)

    handles, labels_out = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(
            handles,
            labels_out,
            loc="best",
            fontsize=LegendSize * 0.85,
            frameon=False,
        )
    _save_and_show(fig, save_path, show, dpi=300, bbox_inches="tight")
    return fig
