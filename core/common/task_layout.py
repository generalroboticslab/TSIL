"""Task-layout helpers shared by training and evaluation."""

import torch


def build_task_id_lookup(unique_task_ids, device):
    """Map sparse task ids to contiguous indices."""
    if unique_task_ids.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    max_task_id = int(unique_task_ids.max().item()) + 1
    lookup = torch.zeros(max_task_id, dtype=torch.long, device=device)
    for idx, tid in enumerate(unique_task_ids):
        lookup[int(tid.item())] = idx
    return lookup


__all__ = ["build_task_id_lookup"]

