"""Actor replay-weight scaling and paired, read-only normalization summaries."""

import torch


def normalize_actor_weights(weights, args):
    """Scale positive weights without centering; retain historical cap semantics."""
    if not args.sil_normalize_gap:
        return weights
    active = weights[weights > 0]
    if active.numel() == 0:
        return weights
    mode = getattr(args, "sil_gap_normalization", "std")
    if mode == "max":
        return weights / active.max()
    if mode == "clip":
        return weights.clamp(max=1.0)
    normalized = weights / (active.std(unbiased=False) + 1e-8)
    cap = float(getattr(args, "sil_normalized_gap_clip", 1.0))
    if cap > 0:
        normalized = normalized.clamp(max=cap)
    return torch.where(weights > 0, normalized, weights)


def normalization_audit(weights, order, num_slices, args):
    """Compare both scopes on identical weights/order, without altering training.

    Store standard deviations themselves, leaving ratios to offline analysis.
    Quartiles are defined globally among positive raw weights for both arms.
    """
    raw = weights.detach().flatten()
    positive = raw[raw > 0]
    if not positive.numel():
        return {"positive_count": 0, "global_std": None, "slices": []}
    boundaries = torch.quantile(
        positive.float(), torch.tensor([0.25, 0.5, 0.75], device=raw.device)
    )
    quartiles = torch.bucketize(raw.float(), boundaries)
    global_weights = normalize_actor_weights(raw, args)
    cap = float(getattr(args, "sil_normalized_gap_clip", 1.0))
    records = []
    for index in range(num_slices):
        indices = order[
            index * len(order) // num_slices : (index + 1) * len(order) // num_slices
        ]
        values = raw[indices]
        mask = values > 0
        active = values[mask]
        local_weights = normalize_actor_weights(values, args)
        arms = {}
        for name, normalized in (
            ("slice", local_weights),
            ("dataset", global_weights[indices]),
        ):
            arms[name] = {
                "weight_sum": float(normalized.sum()),
                "capped_count": (
                    int(((normalized >= cap) & mask).sum()) if cap > 0 else 0
                ),
                "quartile_weight_sums": [
                    float(normalized[(quartiles[indices] == q) & mask].sum())
                    for q in range(4)
                ],
            }
        records.append(
            {
                "count": len(indices),
                "positive_count": int(mask.sum()),
                "local_std": (
                    float(active.std(unbiased=False)) if active.numel() else None
                ),
                **arms,
            }
        )
    return {
        "positive_count": len(positive),
        "global_std": float(positive.std(unbiased=False)),
        "quartile_boundaries": boundaries.cpu().tolist(),
        "slices": records,
    }
