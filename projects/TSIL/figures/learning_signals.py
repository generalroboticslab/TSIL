"""Temporal-target learning-signal maps, summaries, and surfaces."""


from __future__ import annotations


import json


import math


import os


from typing import Optional


import numpy as np


from matplotlib import pyplot as plt


from matplotlib import colors as mpl_colors


from matplotlib import ticker as mpl_ticker


from matplotlib.patches import Patch


from core.plotting.primitives import _save_and_show


from core.plotting.style import NPG_PALETTE, style_axis


WINDOW_FRACTIONS = (
    (0.00, 0.25, "0-25%"),
    (0.25, 0.50, "25-50%"),
    (0.50, 0.75, "50-75%"),
    (0.75, 1.00, "75-100%"),
)


SIGNAL_TITLE_FONT = 15


SIGNAL_AXIS_FONT = 14


SIGNAL_TICK_FONT = 12


SIGNAL_ROW_FONT = 15


SIGNAL_COLORBAR_FONT = 15


SIGNAL_COLORBAR_TICK_FONT = 12


def _read_jsonl(path):
    records = []
    with open(path, "r") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _episode_history_path(signal_history_path):
    default_path = os.path.join(os.path.dirname(str(signal_history_path)), "training_episode_signal_history.jsonl")
    if os.path.isfile(default_path):
        return default_path
    for record in reversed(_read_jsonl(signal_history_path)):
        relpath = record.get("episode_history_file")
        if relpath:
            path = os.path.join(os.path.dirname(str(signal_history_path)), relpath)
            if os.path.isfile(path):
                return path
    return None


def _compact_episode_record(record):
    return {
        "iteration": float(record.get("iteration", 0.0)),
        "episodes": float(record.get("episodes", 0.0)),
        "episode_id": int(record.get("episode_id", 0)),
        "eps_time": float(record.get("eps_time", 0.0)),
        "max_eps_time": float(record.get("max_eps_time", 1.0)),
        "dense_return": float(record.get("dense_return", 0.0)),
        "positive_adv_mass_update": float(record.get("positive_adv_mass_update", 0.0)),
        "success": bool(record.get("success", False)),
    }


def _episode_records(signal_history_path, last_frac=1.0):
    episode_path = _episode_history_path(signal_history_path)
    if episode_path is None:
        raise ValueError(f"No training_episode_signal_history.jsonl for {signal_history_path}")
    records = [
        _compact_episode_record(record) for record in _read_jsonl(episode_path)
        if record.get("episode_step_count", 0) > 0
    ]
    if not records:
        raise ValueError("No episode signal records found.")
    last_frac = float(last_frac)
    if 0.0 < last_frac < 1.0:
        iterations = [record["iteration"] for record in records]
        cutoff = np.quantile(iterations, 1.0 - last_frac)
        records = [record for record in records if record["iteration"] >= cutoff]
    return records


def _task_key_from_signal_history_path(signal_history_path):
    for part in os.path.normpath(str(signal_history_path)).split(os.sep):
        if len(part) > 1 and part[0] == "T" and part[1:].isdigit():
            return part
    return str(signal_history_path)


def _normalize_dense_return_records(path_records):
    dense_values_by_task = {}
    for _, path, records in path_records:
        task_key = _task_key_from_signal_history_path(path)
        dense_values_by_task.setdefault(task_key, []).extend(
            float(record.get("dense_return", 0.0)) for record in records
        )

    dense_bounds_by_task = {}
    for task_key, dense_values in dense_values_by_task.items():
        dense_values = np.asarray(dense_values, dtype=float)
        dense_min = float(np.nanmin(dense_values))
        dense_max = float(np.nanmax(dense_values))
        dense_bounds_by_task[task_key] = (dense_min, max(dense_max - dense_min, 0.0))

    normalized_path_records = []
    for label, path, records in path_records:
        task_key = _task_key_from_signal_history_path(path)
        dense_min, dense_span = dense_bounds_by_task[task_key]
        normalized_records = []
        for record in records:
            normalized_record = dict(record)
            dense_value = float(record.get("dense_return", 0.0))
            if dense_span > 1e-8:
                normalized_dense = (dense_value - dense_min) / dense_span
            else:
                normalized_dense = 0.0
            normalized_record["dense_return"] = float(np.clip(normalized_dense, 0.0, 1.0))
            normalized_records.append(normalized_record)
        normalized_path_records.append((label, path, normalized_records))
    return normalized_path_records


def _labeled_episode_records(signal_history_paths, labels=None, last_frac=1.0, normalize_dense_return=False):
    paths = _as_list(signal_history_paths)
    labels = list(labels or [])
    path_records = []
    for idx, path in enumerate(paths):
        label = labels[idx] if idx < len(labels) else os.path.basename(os.path.dirname(str(path)))
        path_records.append((label, path, _episode_records(path, last_frac=last_frac)))
    if normalize_dense_return:
        path_records = _normalize_dense_return_records(path_records)

    groups = {}
    ordered_labels = []
    for label, _, records in path_records:
        if label not in groups:
            groups[label] = []
            ordered_labels.append(label)
        groups[label].extend(records)
    return [(label, groups[label]) for label in ordered_labels]


def _sort_episode_records(records):
    return sorted(
        records,
        key=lambda record: (
            float(record.get("iteration", 0.0)),
            float(record.get("episodes", 0.0)),
            int(record.get("episode_id", 0)),
        ),
    )


def _records_fraction_from_ordered(ordered, start_frac, end_frac):
    if not ordered:
        return []
    iterations = np.asarray([
        float(record.get("iteration", 0.0)) for record in ordered
    ], dtype=float)
    finite = np.isfinite(iterations)
    if not finite.any():
        return []

    iter_min = float(np.min(iterations[finite]))
    iter_max = float(np.max(iterations[finite]))
    iter_span = iter_max - iter_min
    if iter_span <= 0.0:
        return list(ordered) if float(start_frac) <= 0.0 else []

    start_iter = iter_min + float(start_frac) * iter_span
    end_iter = iter_min + float(end_frac) * iter_span
    if float(end_frac) >= 1.0:
        mask = finite & (iterations >= start_iter) & (iterations <= end_iter)
    else:
        mask = finite & (iterations >= start_iter) & (iterations < end_iter)
    return [record for record, keep in zip(ordered, mask) if keep]


def _dense_edges_for_groups(groups, num_bins):
    dense_values = [
        float(record.get("dense_return", 0.0))
        for _, records in groups
        for record in records
    ]
    dense_min = float(np.min(dense_values))
    dense_max = float(np.max(dense_values))
    if abs(dense_max - dense_min) < 1e-8:
        dense_min -= 0.5
        dense_max += 0.5
    return np.linspace(dense_min, dense_max, int(num_bins) + 1)


def _build_learning_signal_map_from_records(records, time_edges, dense_edges):
    num_bins = len(time_edges) - 1
    times = np.asarray([
        float(record.get("eps_time", 0.0)) / max(float(record.get("max_eps_time", 1.0)), 1e-8)
        for record in records
    ], dtype=float)
    dense_returns = np.asarray([float(record.get("dense_return", 0.0)) for record in records], dtype=float)
    masses = np.asarray([float(record.get("positive_adv_mass_update", 0.0)) for record in records], dtype=float)
    successes = np.asarray([float(bool(record.get("success", False))) for record in records], dtype=float)

    mass = np.zeros((num_bins, num_bins), dtype=float)
    episode_count = np.zeros((num_bins, num_bins), dtype=float)
    success_count = np.zeros((num_bins, num_bins), dtype=float)

    time_idx = np.clip(np.searchsorted(time_edges, np.clip(times, 0.0, 1.0), side="right") - 1, 0, num_bins - 1)
    dense_idx = np.clip(np.searchsorted(dense_edges, dense_returns, side="right") - 1, 0, num_bins - 1)
    for i, j, mass_value, success in zip(time_idx, dense_idx, masses, successes):
        mass[i, j] += mass_value
        episode_count[i, j] += 1.0
        success_count[i, j] += success

    return {
        "time_edges": time_edges.tolist(),
        "dense_return_edges": dense_edges.tolist(),
        "positive_advantage_mass": mass.tolist(),
        "episode_count": episode_count.tolist(),
        "success_count": success_count.tolist(),
    }


def _learning_signal_groups(signal_history_paths, labels=None, last_frac=1.0, num_bins=12, normalize_dense_return=False):
    groups = _labeled_episode_records(
        signal_history_paths,
        labels=labels,
        last_frac=last_frac,
        normalize_dense_return=normalize_dense_return,
    )
    time_edges = np.linspace(0.0, 1.0, int(num_bins) + 1)
    if normalize_dense_return:
        dense_edges = np.linspace(0.0, 1.0, int(num_bins) + 1)
    else:
        dense_edges = _dense_edges_for_groups(groups, num_bins)
    return groups, time_edges, dense_edges


def _learning_signal_maps(signal_history_paths, labels=None, num_bins=12, last_frac=1.0, normalize_dense_return=False):
    groups, time_edges, dense_edges = _learning_signal_groups(
        signal_history_paths,
        labels=labels,
        last_frac=last_frac,
        num_bins=num_bins,
        normalize_dense_return=normalize_dense_return,
    )
    return [
        (label, _build_learning_signal_map_from_records(records, time_edges, dense_edges))
        for label, records in groups
    ]


def _map_arrays(signal_map):
    time_edges = np.asarray(signal_map["time_edges"], dtype=float)
    dense_edges = np.asarray(signal_map["dense_return_edges"], dtype=float)
    mass = np.asarray(signal_map["positive_advantage_mass"], dtype=float)
    episode_count = np.asarray(signal_map["episode_count"], dtype=float)
    success_count = np.asarray(signal_map["success_count"], dtype=float)
    total_mass = float(np.nansum(mass))
    mass_rate = mass / total_mass if total_mass > 0 else mass
    success_rate = np.divide(
        success_count,
        episode_count,
        out=np.full_like(success_count, np.nan, dtype=float),
        where=episode_count > 0,
    )
    return time_edges, dense_edges, mass_rate, success_rate, episode_count


def _success_rate_cmap():
    cmap = mpl_colors.LinearSegmentedColormap.from_list(
        "success_blue_red",
        ["#334fb8", "#c11131"],
    )
    cmap.set_bad("#e5e5e5")
    return cmap


def _success_rate_categories(success_rate, episode_count, min_count=1):
    success_rate = np.asarray(success_rate, dtype=float)
    episode_count = np.asarray(episode_count, dtype=float)
    categories = np.full(success_rate.shape, np.nan, dtype=float)
    valid = np.isfinite(success_rate) & (episode_count >= int(min_count))
    categories[valid] = 0.0
    categories[valid & (success_rate > 0.5)] = 1.0
    categories[valid & np.isclose(success_rate, 1.0)] = 2.0
    return np.ma.masked_invalid(categories)


def _success_mass_overlay_rgba(success_categories, mass_rate, episode_count=None, min_count=1):
    categories = np.ma.asarray(success_categories)
    category_values = categories.filled(-1).astype(int)
    valid = (~np.ma.getmaskarray(categories)) & (category_values >= 0)
    palette = np.asarray([
        [0x21 / 255.0, 0x66 / 255.0, 0xac / 255.0],
        [0x9e / 255.0, 0xca / 255.0, 0xe1 / 255.0],
        [0xf0 / 255.0, 0xf0 / 255.0, 0xf0 / 255.0],
    ])
    rgba = np.ones((*category_values.shape, 4), dtype=float)
    for category in range(len(palette)):
        rgba[valid & (category_values == category), :3] = palette[category]

    mass = np.asarray(mass_rate, dtype=float)
    mass_valid = valid & np.isfinite(mass) & (mass > 0.0)
    if episode_count is not None:
        mass_valid &= np.asarray(episode_count, dtype=float) >= int(min_count)
    strength = np.zeros(category_values.shape, dtype=float)
    if np.any(mass_valid):
        max_mass = max(float(np.nanmax(mass[mass_valid])), 1e-12)
        strength[mass_valid] = np.sqrt(np.clip(mass[mass_valid] / max_mass, 0.0, 1.0))
    red_filter = np.asarray([1.0, 0.0, 0.0])
    base_rgb = rgba[..., :3]
    rgba[..., :3] = base_rgb * (1.0 - strength[..., None]) + (base_rgb * red_filter) * strength[..., None]
    return rgba


def _draw_rgba_cell_mesh(ax, time_edges, dense_edges, rgba):
    rgba = np.asarray(rgba, dtype=float)
    flat_rgba = rgba.reshape(-1, 4)
    unique_rgba, inverse = np.unique(flat_rgba, axis=0, return_inverse=True)
    cmap = mpl_colors.ListedColormap(unique_rgba)
    norm = mpl_colors.BoundaryNorm(np.arange(len(unique_rgba) + 1) - 0.5, len(unique_rgba))
    ax.pcolormesh(
        time_edges,
        dense_edges,
        inverse.reshape(rgba.shape[:2]),
        cmap=cmap,
        norm=norm,
        shading="flat",
        antialiased=False,
        linewidth=0.0,
        edgecolors="none",
        rasterized=False,
    )


def _draw_success_category_legend(ax, font_size=15.0):
    ax.axis("off")
    handles = [
        Patch(facecolor="#f0f0f0", edgecolor="none", label="SR=1"),
        Patch(facecolor="#9ecae1", edgecolor="none", label="0.5<SR<1"),
        Patch(facecolor="#2166ac", edgecolor="none", label="SR<=0.5"),
    ]
    ax.legend(
        handles=handles,
        title="Cell: success rate (SR)",
        loc="center",
        bbox_to_anchor=(0.5, 0.54),
        ncol=3,
        frameon=False,
        fontsize=font_size,
        title_fontsize=18.0,
        handlelength=0.95,
        handletextpad=0.24,
        columnspacing=0.46,
        borderaxespad=0.0,
    )


def _plot_mass_markers(ax, time_edges, dense_edges, mass_rate, episode_count=None, min_count=1, color="#ffffff"):
    mass_values = np.asarray(mass_rate, dtype=float)
    positive_mask = np.isfinite(mass_values) & (mass_values > 0.0)
    if episode_count is not None:
        positive_mask &= np.asarray(episode_count, dtype=float) >= int(min_count)
    if not np.any(positive_mask):
        return

    time_centers = 0.5 * (time_edges[:-1] + time_edges[1:])
    dense_centers = 0.5 * (dense_edges[:-1] + dense_edges[1:])
    xx, yy = np.meshgrid(time_centers, dense_centers, indexing="ij")
    shown_mass = mass_values[positive_mask]
    max_mass = max(float(np.nanmax(shown_mass)), 1e-12)
    sizes = 18.0 + 220.0 * np.sqrt(shown_mass / max_mass)
    ax.scatter(
        xx[positive_mask],
        yy[positive_mask],
        s=sizes,
        color=color,
        alpha=0.65,
        linewidths=0,
        zorder=3,
    )


def _set_signal_axis_labels(ax, row, col, nrows, title):
    ax.grid(False)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.xaxis.set_major_locator(mpl_ticker.FixedLocator([0.0, 0.5, 1.0]))
    ax.yaxis.set_major_locator(mpl_ticker.FixedLocator([0.0, 0.5, 1.0]))
    ax.set_title(title if row == 0 else "", fontsize=SIGNAL_TITLE_FONT)
    if row == nrows - 1:
        ax.set_xlabel(r"Norm. $T^{used}$", fontsize=SIGNAL_AXIS_FONT)
    else:
        ax.set_xlabel("")
    if col == 0:
        ax.set_ylabel(r"Norm. $G^{task}$", fontsize=SIGNAL_AXIS_FONT)
    else:
        ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=SIGNAL_TICK_FONT)


def _set_signal_row_label(ax, label):
    ax.text(
        0.03,
        0.96,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=SIGNAL_ROW_FONT,
        color="black",
    )


def _set_compact_signal_axis_labels(ax, row, col, nrows, ncols, title):
    ax.grid(False)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(mpl_ticker.FixedLocator([0.0, 0.5, 1.0]))
    ax.yaxis.set_major_locator(mpl_ticker.FixedLocator([0.04, 0.5, 1.0]))
    ax.xaxis.set_major_formatter(mpl_ticker.FixedFormatter(["0", "", "1"]))
    ax.yaxis.set_major_formatter(mpl_ticker.FixedFormatter(["0", "", "1"]))
    ax.set_title(title if row == 0 else "", fontsize=18.0, pad=6.5)
    ax.set_xlabel(
        r"Normalized $T^{used}$" if row == nrows - 1 and col == ncols // 2 else "",
        fontsize=18,
        labelpad=3.5,
    )
    ax.set_ylabel(
        r"Normalized $G^{task}$" if col == 0 and row == nrows // 2 else "",
        fontsize=18,
        labelpad=3.5,
    )
    ax.tick_params(axis="both", labelsize=14, pad=2.0, length=3.0)
    if row != nrows - 1:
        ax.tick_params(axis="x", labelbottom=False)
    if col != 0:
        ax.tick_params(axis="y", labelleft=False)


def _set_compact_signal_row_label(ax, label):
    ax.text(
        0.03,
        0.95,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14.5,
        color="black",
        bbox={"facecolor": "white", "alpha": 0.76, "edgecolor": "none", "pad": 1.4},
    )


def plot_learning_signal_map(
    signal_history_path,
    labels=None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize=None,
    num_bins=12,
    last_frac=1.0,
    min_count=1,
):
    """Plot episode-aggregated positive advantage mass over success-rate maps."""
    groups, time_edges, dense_edges = _learning_signal_groups(
        signal_history_path,
        labels=labels,
        num_bins=num_bins,
        last_frac=last_frac,
        normalize_dense_return=True,
    )
    nrows = len(groups)
    ncols = len(WINDOW_FRACTIONS) + 1
    figsize = figsize or (18.0, max(3.4, 2.35 * nrows + 0.45))
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(
        nrows + 1,
        ncols,
        height_ratios=[0.26] + [1.0] * nrows,
        wspace=0.20,
        hspace=0.42,
    )
    axes = np.empty((nrows, ncols), dtype=object)
    for row in range(nrows):
        for col in range(ncols):
            axes[row, col] = fig.add_subplot(grid[row + 1, col])
    success_cax = fig.add_subplot(grid[0, :len(WINDOW_FRACTIONS)])
    count_cax = fig.add_subplot(grid[0, len(WINDOW_FRACTIONS):])

    cmap_success = _success_rate_cmap()
    cmap_count = plt.get_cmap("Blues").copy()
    cmap_count.set_bad("#e5e5e5")

    group_rows = []
    for label, records in groups:
        ordered_records = _sort_episode_records(records)
        count_map = _build_learning_signal_map_from_records(ordered_records, time_edges, dense_edges)
        group_rows.append((label, ordered_records, count_map))
    max_count = max(
        1.0,
        *(
            float(np.nanmax(np.asarray(signal_map["episode_count"], dtype=float)))
            for _, _, signal_map in group_rows
        ),
    )
    success_norm = mpl_colors.Normalize(vmin=0.0, vmax=1.0)
    count_norm = mpl_colors.Normalize(vmin=0.0, vmax=np.log10(1.0 + max_count))

    for row, (label, ordered_records, count_map) in enumerate(group_rows):
        for col, (start_frac, end_frac, window_label) in enumerate(WINDOW_FRACTIONS):
            window_records = _records_fraction_from_ordered(ordered_records, start_frac, end_frac)
            signal_map = _build_learning_signal_map_from_records(window_records, time_edges, dense_edges)
            _, _, mass_rate, success_rate, episode_count = _map_arrays(signal_map)
            success_plot = np.ma.masked_where(episode_count.T < int(min_count), success_rate.T)
            ax = axes[row, col]
            style_axis(ax)
            ax.grid(False)
            ax.pcolormesh(
                time_edges,
                dense_edges,
                success_plot,
                cmap=cmap_success,
                norm=success_norm,
                shading="auto",
            )
            _plot_mass_markers(ax, time_edges, dense_edges, mass_rate, episode_count=episode_count, min_count=min_count)
            _set_signal_axis_labels(ax, row, col, nrows, window_label)
            if col == 0:
                _set_signal_row_label(ax, label)

        count_col = ncols - 1
        _, _, _, _, episode_count = _map_arrays(count_map)
        count_plot = np.ma.masked_where(episode_count.T <= 0, np.log10(1.0 + episode_count.T))
        ax = axes[row, count_col]
        style_axis(ax)
        ax.grid(False)
        ax.pcolormesh(
            time_edges,
            dense_edges,
            count_plot,
            cmap=cmap_count,
            norm=count_norm,
            shading="auto",
        )
        _set_signal_axis_labels(ax, row, count_col, nrows, "Episode Count")

    success_sm = plt.cm.ScalarMappable(norm=success_norm, cmap=cmap_success)
    success_sm.set_array([])
    count_sm = plt.cm.ScalarMappable(norm=count_norm, cmap=cmap_count)
    count_sm.set_array([])
    success_cbar = fig.colorbar(success_sm, cax=success_cax, orientation="horizontal")
    success_cbar.set_label(
        "Cell color: success rate in the cell. White circle: "
        "positive advantage mass share.",
        fontsize=SIGNAL_COLORBAR_FONT,
    )
    success_cbar.ax.tick_params(labelsize=SIGNAL_COLORBAR_TICK_FONT, pad=1)
    success_cbar.ax.xaxis.set_ticks_position("top")
    success_cbar.ax.xaxis.set_label_position("top")
    count_cbar = fig.colorbar(count_sm, cax=count_cax, orientation="horizontal")
    count_cbar.set_label("Episode count log10(1+n)", fontsize=SIGNAL_COLORBAR_FONT)
    count_cbar.ax.xaxis.set_major_locator(mpl_ticker.MaxNLocator(nbins=4))
    count_cbar.ax.tick_params(labelsize=SIGNAL_COLORBAR_TICK_FONT, pad=1)
    count_cbar.ax.xaxis.set_ticks_position("top")
    count_cbar.ax.xaxis.set_label_position("top")
    _save_and_show(fig, save_path, show, dpi=300, bbox_inches="tight")
    return fig


def _summary_metrics(records):
    masses = np.asarray([float(record.get("positive_adv_mass_update", 0.0)) for record in records], dtype=float)
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        return {"fast-success": 0.0, "slow-failure": 0.0, "success-aligned": 0.0, "reward-distracted": 0.0}
    successes = np.asarray([float(bool(record.get("success", False))) for record in records], dtype=float)
    times = np.asarray([
        float(record.get("eps_time", 0.0)) / max(float(record.get("max_eps_time", 1.0)), 1e-8)
        for record in records
    ], dtype=float)
    dense_returns = np.asarray([float(record.get("dense_return", 0.0)) for record in records], dtype=float)
    failure_rate = 1.0 - successes
    dense_min = float(np.min(dense_returns))
    dense_span = max(float(np.max(dense_returns)) - dense_min, 1e-8)
    dense_rank = np.clip((dense_returns - dense_min) / dense_span, 0.0, 1.0)
    return {
        "fast-success": float(np.sum(masses * successes * np.clip(1.0 - times, 0.0, 1.0)) / total_mass),
        "slow-failure": float(np.sum(masses * failure_rate * np.clip(times, 0.0, 1.0)) / total_mass),
        "success-aligned": float(np.sum(masses * successes) / total_mass),
        "reward-distracted": float(np.sum(masses * dense_rank * np.clip(times, 0.0, 1.0)) / total_mass),
    }


def _labeled_summary_metric_stats(signal_history_paths, labels=None, last_frac=1.0):
    paths = _as_list(signal_history_paths)
    labels = list(labels or [])
    groups = {}
    for idx, path in enumerate(paths):
        label = labels[idx] if idx < len(labels) else os.path.basename(os.path.dirname(str(path)))
        groups.setdefault(label, []).append(_summary_metrics(_episode_records(path, last_frac=last_frac)))

    stats = []
    for label, samples in groups.items():
        means = {}
        errors = {}
        for name in samples[0]:
            values = np.asarray([sample[name] for sample in samples], dtype=float)
            means[name] = float(np.mean(values))
            std = 0.0 if len(values) <= 1 else float(np.std(values, ddof=1))
            errors[name] = std / math.sqrt(len(values)) if len(values) > 1 else 0.0
        stats.append((label, means, errors))
    return stats


def plot_learning_signal_summary(
    signal_history_path,
    labels=None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize=None,
    num_bins=12,
    last_frac=1.0,
):
    groups = _labeled_summary_metric_stats(signal_history_path, labels=labels, last_frac=last_frac)
    metric_groups = (
        ("Higher is better", ["success-aligned", "fast-success"]),
        ("Lower is better", ["reward-distracted", "slow-failure"]),
    )
    metric_colors = {
        "fast-success": NPG_PALETTE["teal"],
        "slow-failure": NPG_PALETTE["vermillion"],
        "success-aligned": NPG_PALETTE["navy"],
        "reward-distracted": NPG_PALETTE["peach"],
    }
    metric_labels = {
        "success-aligned": "Success aligned",
        "fast-success": "Fast success",
        "reward-distracted": "Reward distracted",
        "slow-failure": "Slow failure",
    }
    labels = [label for label, _, _ in groups]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=figsize or (14.0, max(4.2, 0.52 * len(labels) + 2.0)), sharey=True)
    for ax, (group_title, metric_names) in zip(axes, metric_groups):
        style_axis(ax)
        height = 0.72 / len(metric_names)
        values = np.asarray([[means[name] for name in metric_names] for _, means, _ in groups])
        errors = np.asarray([[err[name] for name in metric_names] for _, _, err in groups])
        for idx, name in enumerate(metric_names):
            ax.barh(
                y + (idx - 0.5) * height,
                values[:, idx],
                height=height,
                label=metric_labels[name],
                color=metric_colors[name],
                linewidth=0,
                xerr=errors[:, idx],
                error_kw={"ecolor": "#3A3A3A", "elinewidth": 0.7, "capsize": 0},
            )
        ax.set_title(group_title, fontsize=SIGNAL_TITLE_FONT)
        ax.set_xlim(0.0, 1.0)
        ax.xaxis.set_major_locator(mpl_ticker.MaxNLocator(nbins=5))
        ax.tick_params(axis="both", labelsize=SIGNAL_TICK_FONT)
        ax.grid(True, axis="y", alpha=0.18, linewidth=0.6)
        ax.set_xlabel("Positive advantage mass share", fontsize=SIGNAL_AXIS_FONT)
        ax.legend(loc="best", fontsize=SIGNAL_COLORBAR_TICK_FONT, frameon=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=SIGNAL_TICK_FONT)
    axes[1].tick_params(axis="y", labelleft=False)
    axes[0].invert_yaxis()
    fig.tight_layout()
    _save_and_show(fig, save_path, show, dpi=300, bbox_inches="tight")
    return fig


def _draw_compact_learning_signal_summary(axes, groups, show_direction_labels=False):
    metric_groups = (
        ["success-aligned", "fast-success"],
        ["reward-distracted", "slow-failure"],
    )
    metric_colors = {
        "fast-success": NPG_PALETTE["teal"],
        "slow-failure": NPG_PALETTE["vermillion"],
        "success-aligned": NPG_PALETTE["navy"],
        "reward-distracted": NPG_PALETTE["peach"],
    }
    metric_labels = {
        "success-aligned": "success mass",
        "fast-success": "fast-success mass",
        "reward-distracted": "reward-distracted mass",
        "slow-failure": "slow-failure mass",
    }
    method_labels = [label for label, _, _ in groups]
    y = np.arange(len(method_labels))
    for ax_idx, (ax, metric_names) in enumerate(zip(axes, metric_groups)):
        style_axis(ax)
        height = 0.82 / len(metric_names)
        values = np.asarray([[means[name] for name in metric_names] for _, means, _ in groups])
        errors = np.asarray([[err[name] for name in metric_names] for _, _, err in groups])
        for idx, name in enumerate(metric_names):
            legend_label = metric_labels[name]
            if show_direction_labels:
                arrow = "↑" if ax_idx == 0 else "↓"
                legend_label = f"{legend_label} {arrow}"
            ax.barh(
                y + (idx - 0.5) * height,
                values[:, idx],
                height=height,
                label=legend_label,
                color=metric_colors[name],
                linewidth=0,
                xerr=errors[:, idx],
                error_kw={"ecolor": "#3A3A3A", "elinewidth": 0.6, "capsize": 0},
            )
        row_max = np.nanmax(values, axis=1) if len(values) else np.asarray([])
        for label, ypos, xmax in zip(method_labels, y, row_max):
            ax.text(
                min(float(xmax) + 0.035, 1.06),
                ypos,
                label,
                ha="left",
                va="center",
                fontsize=14.0,
                color="#222222",
            )
        ax.set_xlim(0.0, 1.16)
        ax.xaxis.set_major_locator(mpl_ticker.FixedLocator([0.0, 0.5, 1.0]))
        ax.xaxis.set_major_formatter(mpl_ticker.FixedFormatter(["0", "", "1"]))
        ax.tick_params(axis="x", labelsize=13.5, pad=2.0)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.grid(True, axis="y", alpha=0.16, linewidth=0.5)
        ax.margins(y=0.16)
        ax.set_xlabel(
            "Weighted positive advantage\nmass fraction" if ax_idx == len(axes) - 1 else "",
            fontsize=18.0,
            labelpad=4.5,
        )
        ax.legend(
            loc="upper right" if ax_idx == len(axes) - 1 else "lower right",
            fontsize=16.0,
            frameon=False,
            handlelength=1.05,
            handletextpad=0.28,
            labelspacing=0.18,
            borderaxespad=0.10,
        )
    for ax in axes:
        ax.set_yticks(y)
        ax.invert_yaxis()


def _combined_signal_source_metadata(signal_history_paths):
    sources = []
    for path in _as_list(signal_history_paths):
        signal_path = os.path.abspath(str(path))
        episode_path = _episode_history_path(signal_path)
        sources.append({
            "signal_path": signal_path,
            "signal_mtime": os.path.getmtime(signal_path) if os.path.isfile(signal_path) else None,
            "episode_path": os.path.abspath(episode_path) if episode_path else None,
            "episode_mtime": (
                os.path.getmtime(episode_path)
                if episode_path and os.path.isfile(episode_path)
                else None
            ),
        })
    return sources


def _combined_cache_metadata(
    signal_history_path,
    labels,
    num_bins,
    last_frac,
    summary_signal_history_path,
    summary_labels,
    summary_last_frac,
):
    summary_paths = summary_signal_history_path if summary_signal_history_path is not None else signal_history_path
    summary_frac = last_frac if summary_last_frac is None else summary_last_frac
    return {
        "version": 1,
        "map_sources": _combined_signal_source_metadata(signal_history_path),
        "map_labels": list(labels or []),
        "summary_sources": _combined_signal_source_metadata(summary_paths),
        "summary_labels": list(summary_labels or labels or []),
        "num_bins": int(num_bins),
        "last_frac": float(last_frac),
        "summary_last_frac": float(summary_frac),
    }


def _build_learning_signal_combined_data(
    signal_history_path,
    labels=None,
    num_bins=12,
    last_frac=1.0,
    summary_signal_history_path=None,
    summary_labels=None,
    summary_last_frac=None,
):
    groups, time_edges, dense_edges = _learning_signal_groups(
        signal_history_path,
        labels=labels,
        num_bins=num_bins,
        last_frac=last_frac,
        normalize_dense_return=True,
    )
    rows = []
    for label, records in groups:
        ordered_records = _sort_episode_records(records)
        windows = []
        for start_frac, end_frac, window_label in WINDOW_FRACTIONS:
            window_records = _records_fraction_from_ordered(ordered_records, start_frac, end_frac)
            windows.append({
                "label": window_label,
                "map": _build_learning_signal_map_from_records(window_records, time_edges, dense_edges),
            })
        rows.append({
            "label": label,
            "windows": windows,
            "count_map": _build_learning_signal_map_from_records(ordered_records, time_edges, dense_edges),
        })

    summary_paths = summary_signal_history_path if summary_signal_history_path is not None else signal_history_path
    summary_plot_labels = summary_labels if summary_labels is not None else labels
    summary_frac = last_frac if summary_last_frac is None else summary_last_frac
    summary_groups = _labeled_summary_metric_stats(
        summary_paths,
        labels=summary_plot_labels,
        last_frac=summary_frac,
    )
    return {
        "time_edges": time_edges.tolist(),
        "dense_edges": dense_edges.tolist(),
        "rows": rows,
        "summary_groups": summary_groups,
    }


def _load_learning_signal_combined_cache(cache_path, metadata):
    if not cache_path or not os.path.isfile(cache_path):
        return None
    try:
        with open(cache_path, "r") as file_obj:
            payload = json.load(file_obj)
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("metadata") == metadata:
        print(f"[Data loaded] {os.path.abspath(cache_path)}")
        return payload.get("data")
    print(f"[Data stale] {os.path.abspath(cache_path)}")
    return None


def _write_learning_signal_combined_cache(cache_path, metadata, data):
    if not cache_path:
        return
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as file_obj:
        json.dump({"metadata": metadata, "data": data}, file_obj)
        file_obj.write("\n")
    print(f"[Data saved] {os.path.abspath(cache_path)}")


def _learning_signal_combined_data(
    signal_history_path,
    labels=None,
    num_bins=12,
    last_frac=1.0,
    summary_signal_history_path=None,
    summary_labels=None,
    summary_last_frac=None,
    data_cache_path=None,
    refresh_cache=False,
):
    metadata = _combined_cache_metadata(
        signal_history_path,
        labels,
        num_bins,
        last_frac,
        summary_signal_history_path,
        summary_labels,
        summary_last_frac,
    )
    if not refresh_cache:
        cached = _load_learning_signal_combined_cache(data_cache_path, metadata)
        if cached is not None:
            return cached

    data = _build_learning_signal_combined_data(
        signal_history_path,
        labels=labels,
        num_bins=num_bins,
        last_frac=last_frac,
        summary_signal_history_path=summary_signal_history_path,
        summary_labels=summary_labels,
        summary_last_frac=summary_last_frac,
    )
    _write_learning_signal_combined_cache(data_cache_path, metadata, data)
    return data


def plot_learning_signal_combined(
    signal_history_path,
    labels=None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize=None,
    num_bins=12,
    last_frac=1.0,
    min_count=1,
    summary_signal_history_path=None,
    summary_labels=None,
    summary_last_frac=None,
    data_cache_path=None,
    refresh_cache=False,
    column_wspace=0.004,
    summary_direction_labels=False,
):
    """Plot compact learning-signal maps with summary bars in one figure."""
    plot_data = _learning_signal_combined_data(
        signal_history_path,
        labels=labels,
        num_bins=num_bins,
        last_frac=last_frac,
        summary_signal_history_path=summary_signal_history_path,
        summary_labels=summary_labels,
        summary_last_frac=summary_last_frac,
        data_cache_path=data_cache_path,
        refresh_cache=refresh_cache,
    )
    time_edges = np.asarray(plot_data["time_edges"], dtype=float)
    dense_edges = np.asarray(plot_data["dense_edges"], dtype=float)
    group_rows = plot_data["rows"]
    summary_groups = plot_data["summary_groups"]

    nrows = len(group_rows)
    ncols = len(WINDOW_FRACTIONS) + 1
    figsize = figsize or (14.5, max(8.4, 1.94 * nrows + 1.60))
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(
        nrows + 1,
        ncols + 1,
        height_ratios=[0.72] + [1.0] * nrows,
        width_ratios=[0.80] * ncols + [1.79],
        wspace=column_wspace,
        hspace=0.075,
    )

    header_grid = grid[0, :].subgridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.24)
    legend_ax = fig.add_subplot(header_grid[0, 0])
    overlay_cax = fig.add_subplot(header_grid[0, 1])
    count_cax = fig.add_subplot(header_grid[0, 2])
    axes = np.empty((nrows, ncols), dtype=object)
    first_ax = None
    for row in range(nrows):
        for col in range(ncols):
            if first_ax is None:
                axes[row, col] = fig.add_subplot(grid[row + 1, col])
                first_ax = axes[row, col]
            else:
                axes[row, col] = fig.add_subplot(grid[row + 1, col], sharex=first_ax, sharey=first_ax)

    summary_grid = grid[1:, ncols].subgridspec(2, 1, hspace=0.11)
    summary_axes = np.empty(2, dtype=object)
    summary_axes[0] = fig.add_subplot(summary_grid[0, 0])
    summary_axes[1] = fig.add_subplot(summary_grid[1, 0], sharex=summary_axes[0])

    max_count = max(
        1.0,
        *(
            float(np.nanmax(np.asarray(row_data["count_map"]["episode_count"], dtype=float)))
            for row_data in group_rows
        ),
    )

    count_cmap = plt.get_cmap("Blues").copy()
    count_cmap.set_bad("#ffffff")
    count_norm = mpl_colors.Normalize(vmin=0.0, vmax=np.log10(1.0 + max_count))

    _draw_success_category_legend(legend_ax, font_size=15.0)
    overlay_cmap = mpl_colors.LinearSegmentedColormap.from_list("mass_overlay_red", ["#ffffff", "#d73027"])
    overlay_sm = plt.cm.ScalarMappable(norm=mpl_colors.Normalize(vmin=0.0, vmax=1.0), cmap=overlay_cmap)
    overlay_sm.set_array([])
    overlay_cbar = fig.colorbar(overlay_sm, cax=overlay_cax, orientation="horizontal")
    overlay_cbar.set_ticks([0.0, 1.0])
    overlay_cbar.ax.xaxis.set_major_formatter(mpl_ticker.FixedFormatter(["0", "1"]))
    overlay_cbar.ax.tick_params(labelsize=15.0, pad=1.0, length=2.4)
    overlay_cbar.ax.set_title("Overlay: positive advantage mass", fontsize=18.0, pad=5.0)

    count_sm = plt.cm.ScalarMappable(norm=count_norm, cmap=count_cmap)
    count_sm.set_array([])
    count_cbar = fig.colorbar(count_sm, cax=count_cax, orientation="horizontal")
    count_cbar.set_label("")
    count_cbar.ax.xaxis.set_major_locator(mpl_ticker.MaxNLocator(nbins=3))
    count_cbar.ax.xaxis.set_major_formatter(mpl_ticker.FormatStrFormatter("%g"))
    count_cbar.ax.tick_params(labelsize=15.0, pad=1.0, length=2.4)
    count_cbar.ax.set_title(r"# Episodes $\log_{10}(1+n)$", fontsize=18.0, pad=5.0)

    for row, row_data in enumerate(group_rows):
        for col, window_data in enumerate(row_data["windows"]):
            signal_map = window_data["map"]
            _, _, mass_rate, success_rate, episode_count = _map_arrays(signal_map)
            success_plot = _success_rate_categories(success_rate.T, episode_count.T, min_count=min_count)
            cell_rgba = _success_mass_overlay_rgba(
                success_plot,
                mass_rate.T,
                episode_count=episode_count.T,
                min_count=min_count,
            )
            ax = axes[row, col]
            style_axis(ax)
            ax.grid(False)
            _draw_rgba_cell_mesh(ax, time_edges, dense_edges, cell_rgba)
            panel_title = f"iter: {window_data['label']}" if col == 0 else window_data["label"]
            _set_compact_signal_axis_labels(ax, row, col, nrows, ncols, panel_title)
            if col == 0:
                _set_compact_signal_row_label(ax, row_data["label"])

        count_col = ncols - 1
        _, _, _, _, episode_count = _map_arrays(row_data["count_map"])
        count_plot = np.ma.masked_where(episode_count.T <= 0, np.log10(1.0 + episode_count.T))
        ax = axes[row, count_col]
        style_axis(ax)
        ax.grid(False)
        ax.pcolormesh(
            time_edges,
            dense_edges,
            count_plot,
            cmap=count_cmap,
            norm=count_norm,
            shading="auto",
        )
        _set_compact_signal_axis_labels(ax, row, count_col, nrows, ncols, "# Episodes")

    _draw_compact_learning_signal_summary(
        summary_axes,
        summary_groups,
        show_direction_labels=summary_direction_labels,
    )

    fig.subplots_adjust(
        left=0.050,
        right=0.988,
        bottom=0.072,
        top=0.990,
        wspace=column_wspace,
        hspace=0.075,
    )
    overlay_pos = overlay_cax.get_position()
    header_bar_width = 0.74
    header_bar_height = 0.12
    header_bar_y = 0.43
    overlay_cax.set_position([
        overlay_pos.x0 + 0.5 * (1.0 - header_bar_width) * overlay_pos.width,
        overlay_pos.y0 + header_bar_y * overlay_pos.height,
        overlay_pos.width * header_bar_width,
        overlay_pos.height * header_bar_height,
    ])
    count_pos = count_cax.get_position()
    count_cax.set_position([
        count_pos.x0 + 0.5 * (1.0 - header_bar_width) * count_pos.width,
        count_pos.y0 + header_bar_y * count_pos.height,
        count_pos.width * header_bar_width,
        count_pos.height * header_bar_height,
    ])
    _save_learning_signal_figure(fig, save_path, show, dpi=300, bbox_inches="tight", pad_inches=0.01)
    return fig


def plot_learning_signal_surface(
    signal_history_path,
    labels=None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize=None,
    num_bins=12,
    last_frac=1.0,
):
    maps = _learning_signal_maps(signal_history_path, labels=labels, num_bins=num_bins, last_frac=last_frac)
    ncols = min(3, len(maps))
    nrows = int(np.ceil(len(maps) / ncols))
    fig = plt.figure(figsize=figsize or (5 * ncols, 4.2 * nrows))
    success_cmap = _success_rate_cmap()
    success_norm = mpl_colors.Normalize(vmin=0.0, vmax=1.0)
    max_z = 0.0
    for idx, (label, signal_map) in enumerate(maps, start=1):
        time_edges, dense_edges, mass_rate, success_rate, episode_count = _map_arrays(signal_map)
        x = 0.5 * (time_edges[:-1] + time_edges[1:])
        y = 0.5 * (dense_edges[:-1] + dense_edges[1:])
        xx, yy = np.meshgrid(x, y, indexing="ij")
        zz = np.where(episode_count > 0, mass_rate, np.nan)
        if np.isfinite(zz).any():
            max_z = max(max_z, float(np.nanmax(zz)))
        facecolors = success_cmap(success_norm(np.nan_to_num(success_rate, nan=0.0)))
        ax = fig.add_subplot(nrows, ncols, idx, projection="3d")
        ax.plot_surface(xx, yy, zz, facecolors=facecolors, linewidth=0, antialiased=False, shade=False)
        ax.contourf(
            xx,
            yy,
            np.ma.masked_where(episode_count <= 0, success_rate),
            zdir="z",
            offset=0.0,
            levels=np.linspace(0.0, 1.0, 11),
            cmap=success_cmap,
            norm=success_norm,
            alpha=0.35,
        )
        ax.set_title(label)
        ax.set_xlabel("Episode Time / Max Time")
        ax.set_ylabel("Dense Return")
        ax.set_zlabel("Positive Advantage Mass")
        ax.view_init(elev=28, azim=-135)
    for ax in fig.axes:
        if hasattr(ax, "set_zlim"):
            ax.set_zlim(0.0, max(max_z * 1.05, 1e-6))
    success_sm = plt.cm.ScalarMappable(norm=success_norm, cmap=success_cmap)
    success_sm.set_array([])
    cbar = fig.colorbar(success_sm, ax=fig.axes, fraction=0.02, pad=0.02)
    cbar.set_label("Empirical Success Rate")
    fig.tight_layout()
    _save_and_show(fig, save_path, show, dpi=300, bbox_inches="tight")
    return fig


POLICY_LANDSCAPE_ZLABEL = "Fast-success memory log-prob"


POLICY_LANDSCAPE_TITLE_FONT = 26


POLICY_LANDSCAPE_PANEL_TITLE_FONT = 19


POLICY_LANDSCAPE_AXIS_FONT = 20


POLICY_LANDSCAPE_TICK_FONT = 16


POLICY_LANDSCAPE_LEGEND_FONT = 19


POLICY_LANDSCAPE_COLORBAR_FONT = 19


POLICY_LANDSCAPE_COLORBAR_TICK_FONT = 16


def _save_learning_signal_figure(fig, save_path: Optional[str], show: bool = True, **savefig_kw):
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, **savefig_kw)
        print(f"[Plot saved] {os.path.abspath(save_path)}")
    if show:
        plt.show()


__all__ = [
    "plot_learning_signal_map",
    "plot_learning_signal_summary",
    "plot_learning_signal_combined",
    "plot_learning_signal_surface",
]
