"""PPO advantage and loss helpers."""

import torch

from core.training.algo.utils import compute_gae_jit, compute_value_loss


def compute_ppo_advantages(args, device, dtype, storage, agent, envs):
    """Compute GAE advantages and returns for the PPO path."""
    with torch.no_grad():
        if args.use_lstm:
            next_value, _, next_value_c, next_value_t = agent.get_value(
                storage.next_state, storage.next_lstm_state, storage.next_done,
            )
        else:
            next_value, next_value_c, next_value_t = agent.get_value(storage.next_state)

        c_gamma = torch.tensor(args.c_gamma, dtype=dtype, device=device).view(1, -1)

        returns, advantages, returns_c, advantages_c, returns_t, advantages_t = compute_gae_jit(
            storage.rewards, storage.values, storage.dones, storage.timeouts,
            next_value, storage.next_done, storage.next_timeout,
            args.gamma, args.gae_lambda, args.value_bootstrap,
            storage.costs, storage.values_c, next_value_c, c_gamma, args.use_cost,
            storage.values_t, next_value_t, envs.ctrl_dt, args.use_timeawareness,
        )

        returns_t = torch.clamp(returns_t, min=0.0, max=envs.max_eps_time)
        return returns, advantages, returns_c, advantages_c, returns_t, advantages_t


def compute_ppo_losses(args, batch, mb_inds, ratio, newvalue, newvalue_c, newvalue_t, c_scale, mb_advantages):
    """Compute PPO policy/value losses for one minibatch."""
    clipped_ratio = torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * clipped_ratio
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    v_loss_all = 0.0
    newvalue = newvalue.view(-1)
    v_loss = compute_value_loss(
        newvalue, batch["values"][mb_inds], batch["returns"][mb_inds],
        args.clip_vloss, args.clip_coef, 0.5,
    )
    v_loss_all += v_loss

    v_loss_t = None
    if args.use_timeawareness:
        newvalue_t = newvalue_t.view(-1)
        v_loss_t = compute_value_loss(
            newvalue_t, batch["values_t"][mb_inds], batch["returns_t"][mb_inds],
            args.clip_vloss, args.clip_coef, 0.25,
        )
        v_loss_all += v_loss_t

    v_loss_c = None
    L_viol = None
    if args.use_cost:
        mb_advantages_c = batch["advantages_c"][mb_inds]
        cost_loss1 = mb_advantages_c * ratio.view(-1, 1)
        cost_loss2 = mb_advantages_c * clipped_ratio.view(-1, 1)
        L_clip_c = torch.max(cost_loss1, cost_loss2).mean(dim=0)

        L_viol = L_clip_c + batch["norm_return_c"][mb_inds].mean(dim=0)
        L_viol = (c_scale * torch.clamp(L_viol, min=0.0)).sum()
        pg_loss += L_viol

        newvalue_c = newvalue_c.view(-1, args.num_cost)
        v_loss_c = compute_value_loss(
            newvalue_c, batch["values_c"][mb_inds], batch["returns_c"][mb_inds],
            args.clip_vloss, args.clip_coef, 0.5, reduce_dims=0,
        )
        v_loss_all += v_loss_c

    return pg_loss, v_loss, v_loss_all, v_loss_t, v_loss_c, L_viol, mb_advantages
