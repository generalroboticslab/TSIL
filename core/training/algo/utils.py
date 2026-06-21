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


def compute_value_loss(newvalue, old_values, returns, clip_vloss, clip_coef, loss_weight=0.5, reduce_dims=None):
    """Compute value loss with optional PPO-style clipping."""
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

    if reduce_dims is not None:
        return loss_weight * v_loss.mean(dim=reduce_dims).sum()
    return loss_weight * v_loss.mean()
