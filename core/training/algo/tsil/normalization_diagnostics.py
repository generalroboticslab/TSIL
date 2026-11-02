"""Read-only paired normalization probes at scheduled pre-update snapshots."""

import json
from pathlib import Path
import torch
from core.training.algo.tsil.replay_normalization import (
    normalization_audit,
    normalize_actor_weights,
)


def record_normalization_audit(loss, agent, context, iteration, slice_direction):
    """Audit all first-epoch slices and the first slice's counterfactual gradient."""
    dataset = context.get("_sil_aligned_replay_dataset")
    batch = context.get("diagnostic_replay_batch")
    order = context.get("_sil_replay_order")
    folder = getattr(loss.args, "trajectory_dir", None)
    if dataset is None or batch is None or order is None or not folder:
        return
    if "sil_frozen_positive_gap" not in dataset:
        raise ValueError("Paired normalization diagnostics require frozen gaps")
    if getattr(loss.args, "sil_replay_normalization_scope", "slice") != "slice":
        raise ValueError(
            "Paired normalization diagnostics use slice training as the baseline"
        )
    with torch.no_grad():
        raw = dataset["sil_frozen_positive_gap"].detach().clone()
        if "sil_weights" in dataset:
            raw *= dataset["sil_weights"].to(raw)
        summary = normalization_audit(
            raw, order, context["_sil_replay_num_slices"], loss.args
        )
        selected = order[: len(order) // context["_sil_replay_num_slices"]]
        cached = normalize_actor_weights(raw, loss.args)[selected]
    if len(cached) != len(batch["action"]):
        raise ValueError(
            "Counterfactual gradient must use the exact first replay slice"
        )
    optional = {
        "time_target_batch": "time_targets",
        "episode_time_batch": "episode_times",
        "sil_weight_batch": "sil_weights",
        "importance_weight_batch": "sil_importance_weights",
        "frozen_positive_gap_batch": "sil_frozen_positive_gap",
        "sil_success_mask": "sil_success_mask",
        "sil_return_fill_mask": "sil_return_fill_mask",
        "bootstrap_value_batch": "sil_bootstrap_values",
        "bootstrap_mask_batch": "sil_bootstrap_mask",
        "observed_return_batch": "observed_returns",
        "env_id_batch": "env_ids",
    }
    counterfactual, _, _, _ = loss._compute_batch_losses(
        agent,
        batch["obs"],
        batch["state"],
        batch["action"],
        batch["returns"],
        batch["task_ids"],
        frozen_actor_weight_batch=cached,
        return_stats=True,
        **{argument: batch.get(field) for argument, field in optional.items()},
    )
    params = loss._trainable_params(agent)
    direction = (
        -loss._grad_vector(counterfactual, params).detach()
        if counterfactual.requires_grad
        else torch.zeros_like(slice_direction)
    )
    left, right = float(slice_direction.norm()), float(direction.norm())
    summary.update(
        iteration=int(iteration),
        epoch=0,
        gradient_scope="first_logical_replay_slice",
        slice_gradient_norm=left,
        dataset_gradient_norm=right,
        gradient_cosine=(
            float(torch.dot(slice_direction, direction) / (left * right))
            if left and right
            else None
        ),
    )
    target = Path(folder) / f"normalization_audit_iter{iteration:06d}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2) + "\n")
