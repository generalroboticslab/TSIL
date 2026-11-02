"""Bounded success-only actor weighting; critic targets and sampling stay fixed."""
import torch


def success_actor_weights(weights, success, env_ids, state, args):
    """Anneal with fresh on-policy successes, shared by duplicate config workers.

    Balance the extra term across represented successful configurations by their
    sample counts. Cap its mean over the whole replay batch. A separate floor
    ablation can reopen critic-gated successes; the default cannot.
    """
    mode = args.sil_success_actor_schedule
    if mode == 'off':
        return weights, {}
    if state is None or env_ids is None or success is None:
        raise ValueError('Success weighting needs configuration metadata and success labels')
    mask = success.bool()
    extra = torch.zeros_like(weights)
    boost_mean = 1.0
    if mask.any():
        ids = env_ids.long()[mask]
        counts = state['success_counts'][ids].to(weights.dtype).clamp(min=1)
        boost = torch.full_like(counts, args.sil_success_weight_boost)
        if mode == 'automatic':
            boost = 1 + (boost - 1) / (1 + (counts - 1) / args.sil_success_weight_decay)
        base = weights[mask]
        raised = torch.maximum(base, torch.full_like(base, args.sil_success_weight_floor)) * boost
        _, inverse, samples = torch.unique(state['config_ids'][ids], return_inverse=True, return_counts=True)
        balance = float(ids.numel()) / (samples.numel() * samples[inverse].to(weights.dtype))
        extra[mask] = (raised - base) * balance
        boost_mean = float(boost.mean().item())
    uncapped = extra.mean()
    scale = (args.sil_success_extra_mean_cap / uncapped.clamp(min=1e-12)).clamp(max=1)
    extra = extra * scale
    return weights + extra, {
        'replay_success_auto_boost_mean': boost_mean,
        'replay_success_extra_weight_mean': float(extra.mean().item()),
        'replay_success_extra_cap_hit': float(scale.item() < 1),
        'replay_success_reopened_frac': float(((weights <= 0) & (extra > 0)).float().mean().item()),
    }
