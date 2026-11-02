"""Diagnostic reference selection, independent of replay training membership."""

import hashlib
import torch


def top1_gap_ranked_reference(loss, agent, context, budget=32000, per_trajectory=5):
    """Take up to five distinct signed-gap-ranked steps from each top-1 success.

    The origin's critic ranks candidates once; negative gaps remain eligible.
    Small environment chunks bound candidate-state memory. No training RNG,
    archive membership, normalizer statistics, or optimizer state is modified.
    """
    memory = context.get("tsil_memory")
    if memory is None or not getattr(memory, "_archive_allocated", False):
        return None
    environment_count = memory._archive_num_envs
    if environment_count * per_trajectory > budget:
        raise ValueError("Reference budget cannot cover five steps per environment")

    fields = ("obs", "state", "action", "returns", "task_ids", "traj_ids")
    selected = {name: [] for name in fields}
    trajectory_offset = 0
    provenance = []
    with torch.no_grad():
        for start in range(0, environment_count, 64):
            reference = loss._fastest_reference_batch(
                context,
                best_per_env_only=True,
                env_subset=slice(start, start + 64),
            )
            if reference is None:
                continue
            _, _, _, signed_gap = loss._positive_gap_for_batch(
                agent,
                reference["state"],
                reference["returns"],
                reference["task_ids"],
                return_value=True,
            )
            indices = []
            for trajectory in range(reference["ref_count"]):
                candidates = torch.nonzero(
                    reference["traj_ids"] == trajectory, as_tuple=True
                )[0]
                # Stable ordering breaks equal-gap ties by the original time step.
                order = torch.argsort(
                    signed_gap[candidates], descending=True, stable=True
                )
                chosen = candidates[order[:per_trajectory]]
                indices.append(chosen)
                provenance.append(
                    [
                        int(reference["trajectory_env_ids"][trajectory]),
                        int(reference["trajectory_slot_ids_at_capture"][trajectory]),
                        reference["step_indices"][chosen].cpu().tolist(),
                    ]
                )
            indices = torch.cat(indices)
            for name in fields:
                values = reference[name][indices].clone()
                if name == "traj_ids":
                    values += trajectory_offset
                selected[name].append(values)
            trajectory_offset += reference["ref_count"]
    if not trajectory_offset:
        return None
    result = {name: torch.cat(parts) for name, parts in selected.items()}
    result.update(
        ref_count=trajectory_offset,
        reference_kind="Top-1 fast success, signed-gap-ranked steps",
        reference_indices_sha256=hashlib.sha256(repr(provenance).encode()).hexdigest(),
    )
    return result
