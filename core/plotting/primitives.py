"""Reusable matplotlib drawing primitives.

All functions accept raw arrays/lists; data loading belongs in
``core.plotting.train_data``.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np
from matplotlib import pyplot as plt

from .style import (
    PALETTE, COLORS,
    FontSize, MarkerSize, TickSize, AxisLabelSize, LegendSize, YBins,
    BaselineLineWidth, TimeawareLineWidth, ReferenceLineWidth, ThresholdLineWidth,
    LegendHandleLineWidth,
    ScheduleColor, ThresholdColor, VanillaColor,
    _normalize_style_key, LEGEND_STYLE_MAP,
    style_axis,
)


# ═══════════════════════════════════════════════════════════════════════
#  Save / show helper
# ═══════════════════════════════════════════════════════════════════════

def _save_and_show(fig, save_path: Optional[str], show: bool = True, **savefig_kw):
    """Tight-layout → save → print path → show."""
    fig.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, **savefig_kw)
        print(f"[Plot saved] {os.path.abspath(save_path)}")
        stem, ext = os.path.splitext(save_path)
        if ext.lower() != ".png":
            path_parts = save_path.split(os.sep)
            if "pdf" in path_parts:
                pdf_idx = path_parts.index("pdf")
                png_parts = list(path_parts)
                png_parts[pdf_idx] = "png"
                png_path = os.sep.join(png_parts)
                png_path = os.path.splitext(png_path)[0] + ".png"
            else:
                png_path = f"{stem}.png"
            os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
            fig.savefig(png_path, **savefig_kw)
            print(f"[Plot saved] {os.path.abspath(png_path)}")
    if show:
        plt.show()


def save_and_show_fig(
    fig: plt.Figure,
    save_path: Optional[str],
    show: bool = True,
    **savefig_kw,
):
    """Public wrapper around :func:`_save_and_show`."""
    _save_and_show(fig, save_path, show, **savefig_kw)


# ═══════════════════════════════════════════════════════════════════════
#  Bar plots
# ═══════════════════════════════════════════════════════════════════════

def draw_bar_plot(
    legends: List[str],
    colors: List[str],
    mean: "List[float] | List[List[float]]",
    std: "Optional[List[float] | List[List[float]]]" = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    group_labels: Optional[List[str]] = None,
    bar_width: float = 0.2,
    figsize: tuple = (6, 4),
    save_path: Optional[str] = None,
    show: bool = True,
    draw_number: bool = False,
    ylim: Optional[tuple] = None,
    ax: Optional[plt.Axes] = None,
    draw_legend: bool = True,
    annotations: Optional[List[dict]] = None,
    bar_colors: Optional[List[str]] = None,
    grid: bool = False,
    hlines: Optional[List[dict]] = None,
):
    """Draw a single-group or multi-group bar plot.

    Parameters
    ----------
    mean : 1-D → one group; 2-D (list of lists) → multiple groups.
    std  : same shape as *mean*, or None to omit error bars.
    bar_colors : list[str], optional
        Per-group bar colours (length == n_groups).  Overrides *colors*
        when ``n_items == 1``.
    """
    # normalise to 2-D
    if not isinstance(mean[0], (list, np.ndarray)):
        mean = [mean]
        std = [std] if std is not None else None

    n_groups = len(mean)
    n_items = len(legends)

    local_fig = ax is None
    if local_fig:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    group_centers = np.arange(n_groups)
    offsets = np.linspace(
        -(n_items - 1) / 2 * bar_width,
         (n_items - 1) / 2 * bar_width,
        n_items,
    )

    for i in range(n_items):
        vals = [mean[g][i] for g in range(n_groups)]
        errs = [std[g][i] for g in range(n_groups)] if std is not None else None
        x_pos = group_centers + offsets[i]

        if bar_colors is not None and n_items == 1:
            ax.bar(x_pos, vals, width=bar_width, color=bar_colors,
                   yerr=errs, capsize=0, ecolor="#3A3A3A",
                   linewidth=0,
                   error_kw={"elinewidth": 0.7},
                   label=legends[i], alpha=1.0)
        else:
            ax.bar(x_pos, vals, width=bar_width, color=colors[i],
                   yerr=errs, capsize=0, ecolor="#3A3A3A",
                   linewidth=0,
                   error_kw={"elinewidth": 0.7},
                   label=legends[i], alpha=1.0)

        if draw_number:
            for xi, vi in zip(x_pos, vals):
                ax.text(xi, vi + 0.01, f"{vi:.2f}",
                        ha="center", va="bottom", fontsize=TickSize * 0.9,
                        color="#333333")

    # ticks
    if group_labels is not None:
        ax.set_xticks(group_centers)
        ax.set_xticklabels(group_labels, fontsize=TickSize)
    elif n_groups == 1:
        ax.set_xticks([])
    else:
        ax.set_xticks(group_centers)
        ax.set_xticklabels([f"Group {g+1}" for g in range(n_groups)], fontsize=TickSize)

    ax.tick_params(axis="both", labelsize=TickSize)
    ax.locator_params(axis="y", nbins=YBins)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=AxisLabelSize)
    if title:
        ax.set_title(title, fontsize=FontSize)
    if ylim is not None:
        ax.set_ylim(ylim)
    style_axis(ax)
    if draw_legend:
        ax.legend(fontsize=LegendSize)
    if grid:
        ax.grid(axis="y", alpha=0.18, linewidth=0.6, zorder=0)

    # optional horizontal reference lines
    if hlines:
        for hl in hlines:
            kw = dict(hl)
            y = kw.pop("y")
            ax.axhline(y=y, **kw)

    # optional text annotations
    if annotations:
        for ann in annotations:
            kw = dict(ann)
            if "transform" not in kw or kw.get("transform") is None:
                kw["transform"] = ax.transAxes
            ax.text(kw.pop("x"), kw.pop("y"), kw.pop("s"), **kw)

    if local_fig:
        _save_and_show(plt.gcf(), save_path, show, dpi=300, bbox_inches="tight")


def draw_small_bar_plot(
    legends: List[str],
    colors: List[str],
    values: List[float],
    stds: Optional[List[float]] = None,
    ylabel: Optional[str] = None,
    draw_legend: bool = True,
    draw_number: bool = False,
    bar_width: float = 0.1,
    figsize: tuple = (2.5, 4),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Simple side-by-side bar plot (single group, no x-tick labels)."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    x = 0
    for i, (val, leg) in enumerate(zip(values, legends)):
        err = stds[i] if stds is not None else None
        ax.bar(x, val, width=bar_width, color=colors[i],
               yerr=err, capsize=0, ecolor="#3A3A3A",
               linewidth=0, error_kw={"elinewidth": 0.7},
               label=leg, alpha=1.0)
        if draw_number:
            ax.text(x, val + 0.01, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=TickSize * 0.9,
                    color="#333333")
        x += bar_width
    ax.set_xticks([])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=AxisLabelSize)
    ax.locator_params(axis="y", nbins=YBins)
    ax.tick_params(axis="both", labelsize=TickSize)
    style_axis(ax)
    if draw_legend:
        ax.legend(fontsize=LegendSize)
    _save_and_show(fig, save_path, show, dpi=300)


# ═══════════════════════════════════════════════════════════════════════
#  Line / curve plots
# ═══════════════════════════════════════════════════════════════════════

def draw_line_with_std(
    ax: plt.Axes,
    x: np.ndarray,
    y_mean: np.ndarray,
    y_std: Optional[np.ndarray] = None,
    *,
    color: str = "blue",
    label: str = "",
    linewidth: float = BaselineLineWidth,
    linestyle: str = "-",
    marker: Optional[str] = "o",
    markersize: float = MarkerSize,
    marker_edge: bool = False,
    fill_alpha: float = 0.18,
    zorder: int = 5,
):
    """Plot a line with optional std-dev band on an existing axes."""
    norm = _normalize_style_key(label)
    style = LEGEND_STYLE_MAP.get(norm, {})
    line_width = style.get("linewidth", linewidth)
    line_marker = style.get("marker", marker)
    line_style = style.get("linestyle", linestyle)
    marker_size = MarkerSize * 0.5 if line_marker else markersize

    kw = dict(linewidth=line_width, color=color, label=label,
              zorder=zorder, linestyle=line_style)
    if line_marker:
        kw.update(marker=line_marker, markersize=marker_size)
        if marker_edge:
            kw.update(markeredgewidth=marker_size / 5, markeredgecolor="white")
    ax.plot(x, y_mean, **kw)
    if y_std is not None:
        ax.fill_between(x, y_mean - y_std, y_mean + y_std,
                        alpha=fill_alpha, color=color, linewidth=0, zorder=1)


def draw_horizontal_line(
    ax: plt.Axes,
    x: np.ndarray,
    y_val: float,
    *,
    color: str = VanillaColor,
    label: str = "",
    linewidth: float = BaselineLineWidth * 1.1,
    linestyle: str = "--",
    alpha: float = 0.85,
    zorder: int = 5,
):
    """Draw a horizontal reference line spanning *x*."""
    ax.plot(x, np.full_like(x, y_val, dtype=float),
            linestyle=linestyle, linewidth=BaselineLineWidth,
            color=color, label=label, alpha=alpha, zorder=zorder)


def draw_errorbar_markers(
    ax: plt.Axes,
    x: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    *,
    color: str = "#E63946",
    label: str = "",
    markersize: float = MarkerSize * 4,
    zorder: int = 10,
):
    """Draw error-bar markers (horizontal dashes) for thresholds / budgets."""
    norm = _normalize_style_key(label)
    if "schedule" in norm:
        color = ScheduleColor
    elif "threshold" in norm:
        color = ThresholdColor
    ax.errorbar(x, y_mean, yerr=y_std, fmt="_",
                markersize=MarkerSize * 1.2, color=color, ecolor="#3A3A3A",
                elinewidth=0.7, capsize=0,
                zorder=zorder, alpha=0.9, label=label)


# ═══════════════════════════════════════════════════════════════════════
#  Axis finalisation
# ═══════════════════════════════════════════════════════════════════════

def finalize_axes(
    ax: plt.Axes,
    *,
    ylim: Optional[tuple] = None,
    draw_legend: bool = True,
    grid: bool = True,
    tick_size: float = TickSize,
    label_size: float = AxisLabelSize,
    legend_size: float = LegendSize,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend_order: Optional[Sequence[str]] = None,
    legend_top: bool = False,
    legend_ncol: Optional[int] = None,
):
    """Apply common cosmetics to axes (ticks, grid, legend, ylim)."""
    if ylim:
        ax.set_ylim(ylim)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_size)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_size)
    ax.locator_params(axis="y", nbins=YBins)
    ax.tick_params(axis="both", labelsize=tick_size)
    if grid:
        style_axis(ax)
        ax.set_axisbelow(True)
    if draw_legend:
        handles, labels = ax.get_legend_handles_labels()
        if legend_order:
            ordered_handles, ordered_labels, used = [], [], set()
            for target in legend_order:
                for idx, lbl in enumerate(labels):
                    if lbl == target and idx not in used:
                        ordered_handles.append(handles[idx])
                        ordered_labels.append(lbl)
                        used.add(idx)
            for idx, lbl in enumerate(labels):
                if idx not in used:
                    ordered_handles.append(handles[idx])
                    ordered_labels.append(lbl)
            handles, labels = ordered_handles, ordered_labels
        legend_kw = dict(fontsize=legend_size)
        if legend_top:
            legend_kw.update(
                loc="upper center",
                bbox_to_anchor=(0.5, 0.985),
                ncol=legend_ncol or max(1, len(labels)),
                columnspacing=1.2,
                handlelength=3.4,
                handletextpad=0.6,
            )
            fig = ax.figure
            for legend in list(fig.legends):
                legend.remove()
            legend = fig.legend(handles, labels, **legend_kw)
        else:
            legend = ax.legend(handles, labels, **legend_kw)
        for handle in getattr(legend, "legendHandles", []):
            if hasattr(handle, "set_linewidth"):
                handle.set_linewidth(LegendHandleLineWidth)
