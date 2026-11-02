"""Checkpoint-paired Meta-World configuration deadlines.

Training keeps one fixed object/goal configuration per vector environment and
learns its deadline independently.  This module serializes that state beside a
checkpoint and transfers learned deadlines to fresh, in-distribution evaluation
configurations with task-local KNN-5.  Unsolved training configurations never
provide target labels.
"""

import hashlib
import json
from pathlib import Path

import torch


FORMAT_VERSION = 1
PROTOCOL = "metaworld_checkpoint_targets_knn5"


def sidecar_path(checkpoint_path):
    return Path(str(checkpoint_path) + ".metaworld.json")


def _canonical_payload(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    temporary.replace(path)


def build_target_bank(env, checkpoint_name):
    """Capture the environment configuration/deadline state for one checkpoint."""
    features = torch.as_tensor(env.last_rand_vecs).detach().cpu().float()
    task_ids = torch.as_tensor(env.task_indices).detach().cpu().long()
    deadlines = torch.as_tensor(env.obs_range["time2end"][1]).detach().cpu().float()
    learned = getattr(env, "deadline_was_updated", None)
    if learned is None:
        initial = float(getattr(env, "initial_time2end_upper", env.max_eps_time))
        learned = deadlines < initial - 1e-7
    else:
        learned = torch.as_tensor(learned).detach().cpu().bool()
    if features.ndim != 2 or features.shape[0] != deadlines.numel():
        raise ValueError("Meta-World target-bank features and deadlines are not aligned")
    if task_ids.numel() != deadlines.numel() or learned.numel() != deadlines.numel():
        raise ValueError("Meta-World target-bank task IDs and learned mask are not aligned")
    payload = {
        "format_version": FORMAT_VERSION,
        "protocol": PROTOCOL,
        "checkpoint": str(checkpoint_name),
        "feature_semantics": "task reset last_rand_vecs (object/goal configuration)",
        "task_ids": task_ids.tolist(),
        "features": features.tolist(),
        "deadline_seconds": deadlines.tolist(),
        "learned": learned.tolist(),
        "max_episode_seconds": float(env.max_eps_time),
        "knn_k": 5,
    }
    payload["sha256"] = hashlib.sha256(_canonical_payload(payload)).hexdigest()
    return payload


def validate_target_bank(bank):
    bank = dict(bank)
    digest = bank.pop("sha256", None)
    if bank.get("format_version") != FORMAT_VERSION or bank.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported Meta-World target-bank format")
    expected = hashlib.sha256(_canonical_payload(bank)).hexdigest()
    if digest != expected:
        raise ValueError("Meta-World target-bank checksum mismatch")
    sizes = [len(bank[name]) for name in ("task_ids", "features", "deadline_seconds", "learned")]
    if not sizes[0] or len(set(sizes)) != 1:
        raise ValueError("Meta-World target-bank arrays are empty or misaligned")
    return {**bank, "sha256": digest}


def load_target_bank(checkpoint_path):
    path = sidecar_path(checkpoint_path)
    return validate_target_bank(json.loads(path.read_text()))


def infer_targets(bank, query_features, query_task_ids, full_horizon, k=5):
    """Infer held-out deadlines from successful training configurations only."""
    bank = validate_target_bank(bank)
    queries = torch.as_tensor(query_features).detach().cpu().float()
    query_tasks = torch.as_tensor(query_task_ids).detach().cpu().long().flatten()
    features = torch.tensor(bank["features"], dtype=torch.float32)
    tasks = torch.tensor(bank["task_ids"], dtype=torch.long)
    deadlines = torch.tensor(bank["deadline_seconds"], dtype=torch.float32)
    learned = torch.tensor(bank["learned"], dtype=torch.bool)
    if queries.ndim != 2 or queries.shape[0] != query_tasks.numel():
        raise ValueError("Meta-World evaluation features and task IDs are not aligned")
    if features.shape[1] != queries.shape[1]:
        raise ValueError("Meta-World training/evaluation feature widths differ")
    if int(k) < 1:
        raise ValueError("KNN neighbor count must be positive")

    inferred = torch.full((queries.shape[0],), float(full_horizon), dtype=torch.float32)
    source_counts = torch.zeros(queries.shape[0], dtype=torch.long)
    neighbor_counts = torch.zeros(queries.shape[0], dtype=torch.long)
    for task_id in torch.unique(query_tasks):
        query_mask = query_tasks == task_id
        task_mask = tasks == task_id
        source_mask = task_mask & learned
        source_count = int(source_mask.sum().item())
        source_counts[query_mask] = source_count
        if source_count == 0:
            continue
        source = features[source_mask]
        source_deadlines = deadlines[source_mask]
        # Normalize within each task so dimensions with wider physical units do
        # not dominate Euclidean distance. Constant dimensions contribute zero.
        task_features = features[task_mask]
        lower, upper = task_features.min(dim=0).values, task_features.max(dim=0).values
        scale = upper - lower
        scale = torch.where(scale > 1e-8, scale, torch.ones_like(scale))
        distances = torch.cdist((queries[query_mask] - lower) / scale,
                                (source - lower) / scale)
        count = min(int(k), source_count)
        nearest = distances.topk(count, largest=False, dim=1).indices
        inferred[query_mask] = source_deadlines[nearest].mean(dim=1).clamp(
            min=0.0, max=float(full_horizon))
        neighbor_counts[query_mask] = count
    return inferred, {
        "source_counts": source_counts,
        "neighbor_counts": neighbor_counts,
        "fallback_count": int((neighbor_counts == 0).sum().item()),
    }


def apply_targets_to_fixed_env(env, bank):
    """Assign bank targets and reset once so the first policy input sees them."""
    if not getattr(env, "fixed", False):
        raise ValueError("Checkpoint KNN targets require fixed evaluation configurations")
    targets, diagnostics = infer_targets(
        bank,
        env.last_rand_vecs,
        env.task_indices,
        full_horizon=env.max_eps_time,
        k=bank.get("knn_k", 5),
    )
    targets = targets.to(device=env.device, dtype=env.obs_range["time2end"][1].dtype)
    env.obs_range["time2end"][0].copy_(-targets)
    env.obs_range["time2end"][1].copy_(targets)
    # A second reset preserves fixed object/goal configurations and propagates
    # the assigned bounds into time2end_init, live time buffers, and obs_buf.
    env.reset_all()
    return targets, diagnostics


def apply_single_training_configuration(env, bank, source_offset=0):
    """Repeat one checkpoint-paired training reset and deadline per task."""
    if not getattr(env, "fixed", False):
        raise ValueError("A repeated training configuration requires fixed evaluation configurations")
    bank = validate_target_bank(bank)
    source_tasks = torch.tensor(bank["task_ids"], dtype=torch.long)
    source_features = torch.tensor(bank["features"], dtype=torch.float32)
    source_deadlines = torch.tensor(bank["deadline_seconds"], dtype=torch.float32)
    destination_tasks = torch.as_tensor(env.task_indices).detach().cpu().long()
    assigned = torch.empty(destination_tasks.numel(), dtype=torch.float32)
    selected = {}
    for task_id in torch.unique(destination_tasks):
        source = torch.nonzero(source_tasks == task_id, as_tuple=False).flatten()
        if source.numel() == 0:
            raise ValueError(f"Target bank has no configuration for task {int(task_id)}")
        source_index = int(source[int(source_offset) % source.numel()].item())
        destination = torch.nonzero(destination_tasks == task_id, as_tuple=False).flatten()
        feature = source_features[source_index]
        env.last_rand_vecs[destination.to(env.device)] = feature.to(env.device)
        assigned[destination] = source_deadlines[source_index]
        selected[int(task_id)] = {
            "source_index": source_index,
            "configuration": feature.tolist(),
            "deadline_seconds": float(source_deadlines[source_index]),
        }
    assigned = assigned.to(device=env.device, dtype=env.obs_range["time2end"][1].dtype)
    env.obs_range["time2end"][0].copy_(-assigned)
    env.obs_range["time2end"][1].copy_(assigned)
    env.reset_all()
    return assigned, selected
