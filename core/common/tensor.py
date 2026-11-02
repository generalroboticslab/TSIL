"""Tensor and array conversion helpers."""

import numpy as np
import torch


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    else:
        x = np.asarray(x)

    if len(x.shape) == 0:
        x = np.expand_dims(x, axis=0)
    return x


def replace_nan(x, value=0.0):
    """Replace NaN values in a numpy array with a specified value."""
    x = np.asarray(x)
    x[np.isnan(x)] = value
    return x


def get_args_attr(args, attr_name, default_v=None):
    if hasattr(args, attr_name):
        return getattr(args, attr_name)
    return default_v


def weighted_average(entries, key):
    """Weighted average over a list of (weight, dict) pairs."""
    if not entries:
        return 0.0
    total_weight = sum(weight for weight, _ in entries)
    if total_weight <= 0:
        return 0.0
    return sum(weight * float(item.get(key, 0.0)) for weight, item in entries) / total_weight


__all__ = ["to_numpy", "replace_nan", "get_args_attr", "weighted_average"]
