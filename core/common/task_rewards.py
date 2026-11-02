"""Explicit task reward controls, separate from adaptive reward normalization."""
import json
import math
import torch

def parse_dense_scales(spec):
    values = json.loads(spec) if isinstance(spec, str) else spec
    if not isinstance(values, dict):
        raise ValueError("task_dense_reward_scales must be a JSON object")
    out = {}
    for key, value in values.items():
        if str(int(key)) != str(key) or int(key) < 0:
            raise ValueError("Dense scale keys must be nonnegative integer task IDs")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("Dense scales must be finite nonnegative values")
        out[int(key)] = value
    return out

def dense_scales_for_tasks(task_ids, default, spec):
    if not math.isfinite(float(default)) or default < 0:
        raise ValueError("reward_scale must be finite nonnegative")
    scales = torch.full_like(task_ids, float(default), dtype=torch.float32)
    for task, value in parse_dense_scales(spec).items():
        scales[task_ids == task] = value
    return scales

def replace_native_success_reward(reward, success, value):
    if value == -1:
        return reward
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError("Invalid native success reward")
    return torch.where(success.bool(), torch.full_like(reward, float(value)), reward)
