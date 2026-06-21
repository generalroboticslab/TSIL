"""Shared curve and scalar metrics for TSIL paper analysis."""

from __future__ import annotations

import math

import numpy as np


def _finite(values):
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def _sample_std(vals):
    return 0.0 if len(vals) <= 1 else float(np.std(vals, ddof=1))


def _mean_std(values):
    vals = _finite(values)
    if len(vals) == 0:
        return math.nan, math.nan, 0
    return float(np.mean(vals)), _sample_std(vals), int(len(vals))


def _clean_curve(curve, min_points=1):
    if curve is None:
        return None
    x, y = curve
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < min_points:
        return None
    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique_x = np.unique(x)
    if len(unique_x) != len(x):
        y = np.asarray([y[x == value].mean() for value in unique_x], dtype=float)
        x = unique_x
    if len(x) < min_points:
        return None
    return x, y


def _prepend_zero_success_curve(curve):
    cleaned = _clean_curve(curve, min_points=1)
    if cleaned is None:
        return None
    x, y = cleaned
    if float(x[0]) <= 0.0:
        return x, y
    zero_hold_x = np.nextafter(float(x[0]), -np.inf)
    return (
        np.concatenate(([0.0, zero_hold_x], x)),
        np.concatenate(([0.0, 0.0], y)),
    )


def _tail_start_x(x, tail_frac):
    if len(x) == 0:
        return math.nan
    span = float(x[-1] - x[0])
    if span <= 0.0:
        return float(x[max(0, int((1.0 - tail_frac) * len(x)))])
    return float(x[-1] - tail_frac * span)


def _curve_metrics(curve, threshold=0.8, tail_frac=0.1, start_at_zero=False):
    cleaned = (
        _prepend_zero_success_curve(curve)
        if start_at_zero
        else _clean_curve(curve, min_points=2)
    )
    if cleaned is None:
        return None
    x, y = cleaned
    if len(x) < 2:
        return None

    span = float(x[-1] - x[0])
    auc = float(np.trapz(y, x) / span) if span > 0 else float(y[-1])
    reached = np.flatnonzero(y >= float(threshold))
    first_threshold_x = float(x[int(reached[0])]) if len(reached) else math.nan

    tail_mask = x >= _tail_start_x(x, tail_frac)
    if not tail_mask.any():
        tail_mask[-1] = True
    tail_x = x[tail_mask]
    tail_y = y[tail_mask]
    best_idx = int(np.argmax(tail_y))
    return {
        "tail_best_success": float(tail_y[best_idx]),
        "tail_best_x": float(tail_x[best_idx]),
        "tail_mean_success": float(np.mean(tail_y)),
        "tail_success_std": _sample_std(tail_y),
        "auc_success": auc,
        "steps_to_threshold": first_threshold_x,
    }


def _success_metrics(curve, tail_frac):
    metrics = _curve_metrics(curve, tail_frac=tail_frac, start_at_zero=True)
    if metrics is None:
        return None
    return {
        "tail_best_success": metrics["tail_best_success"],
        "tail_best_x": metrics["tail_best_x"],
        "auc_success": metrics["auc_success"],
    }


def _interp_at(curve, x_value):
    cleaned = _clean_curve(curve, min_points=1)
    if cleaned is None or not math.isfinite(float(x_value)):
        return math.nan
    x, y = cleaned
    return float(np.interp(float(x_value), x, y))


def _last_value(curve):
    cleaned = _clean_curve(curve, min_points=1)
    return math.nan if cleaned is None else float(cleaned[1][-1])


def _mean_error(values, mode):
    vals = _finite(values)
    if len(vals) == 0:
        return math.nan, math.nan, 0
    std = _sample_std(vals)
    if mode == "sem":
        err = std / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
    else:
        err = std
    return float(np.mean(vals)), err, int(len(vals))
