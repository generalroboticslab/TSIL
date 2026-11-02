"""Shared policy-gradient math helpers."""

from typing import Tuple

import torch


@torch.jit.script
def compute_gae_jit(
    rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, timeouts: torch.Tensor,
    next_value: torch.Tensor, next_done: torch.Tensor, next_timeout: torch.Tensor,
    gamma: float, gae_lambda: float, value_bootstrap: bool,
    costs: torch.Tensor, values_c: torch.Tensor,
    next_value_c: torch.Tensor, c_gamma: torch.Tensor,
    use_cost: bool,
    values_t: torch.Tensor, next_value_t: torch.Tensor,
    ctrl_dt: float, use_time: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """JIT-compiled GAE computation for rewards, costs, and time."""
    num_steps = rewards.shape[0]

    advantages = torch.zeros_like(rewards)
    advantages_c = torch.zeros_like(costs)
    advantages_t = torch.zeros_like(rewards)

    lastgaelam = torch.zeros_like(rewards[0])
    lastgaelam_c = torch.zeros_like(costs[0])

    dones = torch.cat((dones, next_done.unsqueeze(0)), dim=0) == 1
    timeouts = torch.cat((timeouts, next_timeout.unsqueeze(0)), dim=0) == 1
    terminates = (dones == True) & (timeouts == False) if value_bootstrap else dones

    values = torch.cat((values, next_value.unsqueeze(0)), dim=0)
    values_c = torch.cat((values_c, next_value_c.unsqueeze(0)), dim=0)
    values_t = torch.cat((values_t, next_value_t.unsqueeze(0)), dim=0)

    for t in range(num_steps - 1, -1, -1):
        nextnonterminal = 1.0 - terminates[t + 1].float()

        nextvalues = values[t + 1]
        nextvalues_c = values_c[t + 1]
        nextvalues_t = values_t[t + 1]

        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * lastgaelam * nextnonterminal
        advantages[t] = lastgaelam = torch.where(dones[t], 0., lastgaelam)

        if use_cost:
            nextnonterminal_vert = nextnonterminal.view(-1, 1)
            delta_c = costs[t] + c_gamma * nextvalues_c * nextnonterminal_vert - values_c[t]
            lastgaelam_c = delta_c + c_gamma * gae_lambda * lastgaelam_c * nextnonterminal_vert
            advantages_c[t] = lastgaelam_c = torch.where(
                dones[t].view(-1, 1),
                torch.zeros_like(lastgaelam_c),
                lastgaelam_c
            )

        if use_time:
            delta_t = ctrl_dt + nextvalues_t * nextnonterminal - values_t[t]
            delta_t = torch.where(timeouts[t], 0., delta_t)
            advantages_t[t] = torch.where(terminates[t], -values_t[t], delta_t)

    returns = advantages + values[:-1]
    returns_c = advantages_c + values_c[:-1]
    returns_t = advantages_t + values_t[:-1]

    return returns, advantages, returns_c, advantages_c, returns_t, advantages_t


def weighted_mean(values, sample_weights=None, dim=None):
    """Mean with optional per-sample weights."""
    if sample_weights is None:
        return values.mean() if dim is None else values.mean(dim=dim)
    weights = sample_weights.to(device=values.device, dtype=values.dtype)
    while weights.dim() < values.dim():
        weights = weights.unsqueeze(-1)
    if dim is None:
        denom = weights.expand_as(values).sum().clamp_min(1e-8)
        return (values * weights).sum() / denom
    denom = weights.expand_as(values).sum(dim=dim).clamp_min(1e-8)
    return (values * weights).sum(dim=dim) / denom


def combine_sample_weights(sample_weights, mask=None, indices=None, device=None, dtype=None):
    """Combine optional sample weights with an optional boolean/numeric mask."""
    if mask is None:
        return sample_weights
    if indices is not None:
        mask = mask[indices]
    if sample_weights is not None:
        device = sample_weights.device if device is None else device
        dtype = sample_weights.dtype if dtype is None else dtype
    weights = mask.to(device=device or mask.device, dtype=dtype or torch.float32)
    if sample_weights is None:
        return weights
    return sample_weights.to(device=weights.device, dtype=weights.dtype) * weights


def compute_value_loss(
    newvalue,
    old_values,
    returns,
    clip_vloss,
    clip_coef,
    loss_weight=0.5,
    reduce_dims=None,
    sample_weights=None,
):
    """Compute value loss with optional PPO-style clipping."""
    if sample_weights is not None:
        valid = sample_weights > 0
        while valid.dim() < newvalue.dim():
            valid = valid.unsqueeze(-1)
        # Remove invalid values before squaring, not after forming inf/nan losses.
        newvalue = torch.where(valid, newvalue, torch.zeros_like(newvalue))
        old_values = torch.where(valid, old_values, torch.zeros_like(old_values))
        returns = torch.where(valid, returns, torch.zeros_like(returns))
    v_loss_unclipped = (newvalue - returns) ** 2

    if clip_vloss:
        v_clipped = old_values + torch.clamp(
            newvalue - old_values,
            -clip_coef,
            clip_coef,
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss = torch.max(v_loss_unclipped, v_loss_clipped)
    else:
        v_loss = v_loss_unclipped

    if sample_weights is None:
        if reduce_dims is not None:
            return loss_weight * v_loss.mean(dim=reduce_dims).sum()
        return loss_weight * v_loss.mean()

    weights = sample_weights.to(device=v_loss.device, dtype=v_loss.dtype)
    while weights.dim() < v_loss.dim():
        weights = weights.unsqueeze(-1)
    if reduce_dims is not None:
        denom = weights.expand_as(v_loss).sum(dim=reduce_dims).clamp_min(1e-8)
        return loss_weight * (v_loss * weights).sum(dim=reduce_dims).div(denom).sum()
    denom = weights.expand_as(v_loss).sum().clamp_min(1e-8)
    return loss_weight * (v_loss * weights).sum() / denom
