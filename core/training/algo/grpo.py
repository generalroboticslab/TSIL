"""GRPO return and loss helpers."""

import torch

from core.training.algo.utils import compute_gae_jit, compute_value_loss


def compute_grpo_returns(args, device, dtype, storage, agent, envs):
    """Compute reward-to-go returns for GRPO."""
    with torch.no_grad():
        rewards_to_go = torch.zeros_like(storage.rewards)
        g_next = torch.zeros(args.num_envs, device=device, dtype=dtype)

        dones_ext = torch.cat((storage.dones, storage.next_done.unsqueeze(0)), dim=0)

        for t in reversed(range(args.num_steps)):
            nextnonterminal = 1.0 - dones_ext[t + 1].float()
            rewards_to_go[t] = storage.rewards[t] + args.gamma * g_next * nextnonterminal
            g_next = rewards_to_go[t]

        advantages = returns = rewards_to_go

        returns_c = torch.zeros_like(storage.costs)
        advantages_c = torch.zeros_like(storage.costs)
        returns_t = torch.zeros_like(storage.rewards)
        advantages_t = torch.zeros_like(storage.rewards)

        if args.use_cost or args.use_timeawareness:
            if args.use_lstm:
                next_value, _, next_value_c, next_value_t = agent.get_value(
                    storage.next_state, storage.next_lstm_state, storage.next_done,
                )
            else:
                next_value, next_value_c, next_value_t = agent.get_value(storage.next_state)

            c_gamma = torch.tensor(args.c_gamma, dtype=dtype, device=device).view(1, -1)

            _, _, returns_c, advantages_c, returns_t, advantages_t = compute_gae_jit(
                storage.rewards, storage.values, storage.dones, storage.timeouts,
                next_value, storage.next_done, storage.next_timeout,
                args.gamma, args.gae_lambda, args.value_bootstrap,
                storage.costs, storage.values_c, next_value_c, c_gamma, args.use_cost,
                storage.values_t, next_value_t, envs.ctrl_dt, args.use_timeawareness,
            )
            returns_t = torch.clamp(returns_t, min=0.0, max=envs.max_eps_time)

        return returns, advantages, returns_c, advantages_c, returns_t, advantages_t


def compute_grpo_losses(
    args,
    device,
    batch,
    mb_inds,
    ratio,
    newlogprob,
    newvalue_c,
    newvalue_t,
    c_scale,
    mb_advantages,
):
    """Compute GRPO policy/value losses for one minibatch."""
    if args.grpo_use_clipping:
        clipped_ratio = torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * clipped_ratio
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    else:
        pg_loss = (-newlogprob * mb_advantages).mean()

    v_loss = torch.tensor(0.0, device=device)
    v_loss_all = 0.0

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
        clipped_ratio_c = torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        mb_advantages_c = batch["advantages_c"][mb_inds]
        cost_loss1 = mb_advantages_c * ratio.view(-1, 1)
        cost_loss2 = mb_advantages_c * clipped_ratio_c.view(-1, 1)
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
