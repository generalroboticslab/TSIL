"""Task-layout helpers shared by training and evaluation."""

import torch


def resolve_task_embedding_dim(envs, requested_dim=0):
    """Resolve the task suffix width from the environment observation contract.

    ``requested_dim=0`` means auto-detect from the environment.  It must never
    be inferred from the number of tasks in a run: MTBench uses an absolute
    50-way one-hot even for a single selected task.
    """
    requested_dim = int(requested_dim or 0)
    if requested_dim < 0:
        raise ValueError(f"task_embedding_dim must be non-negative, got {requested_dim}")

    env_dim = getattr(envs, "task_observation_embedding_dim", None)
    if env_dim is None:
        env_dim = getattr(envs, "task_embedding_dim", None)
    if env_dim is None:
        return requested_dim

    env_dim = int(env_dim)
    if env_dim < 0:
        raise ValueError(
            f"Environment task observation embedding width must be non-negative, got {env_dim}"
        )
    if requested_dim not in (0, env_dim):
        raise ValueError(
            "task_embedding_dim must match the task suffix actually appended by "
            f"the environment: requested {requested_dim}, environment {env_dim}"
        )
    return env_dim


def build_task_id_lookup(unique_task_ids, device):
    """Map sparse task ids to contiguous indices."""
    if unique_task_ids.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    max_task_id = int(unique_task_ids.max().item()) + 1
    lookup = torch.zeros(max_task_id, dtype=torch.long, device=device)
    for idx, tid in enumerate(unique_task_ids):
        lookup[int(tid.item())] = idx
    return lookup


__all__ = ["build_task_id_lookup", "resolve_task_embedding_dim"]
