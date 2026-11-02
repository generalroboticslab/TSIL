"""Small, simulator-independent helpers for evaluation warm-up accounting."""


def update_per_env_warmup_counts(counts, terminal_mask, required):
    """Count completed warm-up episodes and report when every lane is ready."""
    counts.add_(terminal_mask.to(dtype=counts.dtype))
    return bool((counts >= required).all().item())
