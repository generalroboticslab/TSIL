"""PPO advantage and loss helpers."""

import torch

from core.training.algo.utils import (
    combine_sample_weights,
    compute_gae_jit,
    compute_value_loss,
    weighted_mean,
)


def reward_values_for_timeout_bootstrap(storage, agent, next_value):
    """Select an optional timeout-only continuation value without changing V(s)."""
    timeout_value_fn = getattr(agent, "get_timeout_bootstrap_value", None)
    if not callable(timeout_value_fn):
        return storage.values, next_value

    values = storage.values.clone()
    timeout_boundaries = storage.dones.bool() & storage.timeouts.bool()
    if bool(timeout_boundaries.any().item()):
        values[timeout_boundaries] = timeout_value_fn(
            storage.states[timeout_boundaries], denorm=True, update=False,
        )

    selected_next_value = next_value
    final_timeout = storage.next_done.bool() & storage.next_timeout.bool()
    if bool(final_timeout.any().item()):
        selected_next_value = next_value.clone()
        selected_next_value[final_timeout] = timeout_value_fn(
            storage.next_state[final_timeout], denorm=True, update=False,
        )
    return values, selected_next_value


def compute_ppo_advantages(args, device, dtype, storage, agent, envs, rewards=None):
    """Compute GAE advantages and returns for the PPO path."""
    with torch.no_grad():
        if args.use_lstm:
            next_value, _, next_value_c, next_value_t = agent.get_value(
                storage.next_state, storage.next_lstm_state, storage.next_done,
            )
        else:
            next_value, next_value_c, next_value_t = agent.get_value(storage.next_state)
        reward_values, next_value = reward_values_for_timeout_bootstrap(
            storage, agent, next_value,
        )

        c_gamma = torch.tensor(args.c_gamma, dtype=dtype, device=device).view(1, -1)

        rewards = storage.rewards if rewards is None else rewards
        returns, advantages, returns_c, advantages_c, returns_t, advantages_t = compute_gae_jit(
            rewards, reward_values, storage.dones, storage.timeouts,
            next_value, storage.next_done, storage.next_timeout,
            args.gamma, args.gae_lambda, args.value_bootstrap,
            storage.costs, storage.values_c, next_value_c, c_gamma, args.use_cost,
            storage.values_t, next_value_t, envs.ctrl_dt, args.use_timeawareness,
        )

        returns_t = torch.clamp(returns_t, min=0.0, max=envs.max_eps_time)
        return returns, advantages, returns_c, advantages_c, returns_t, advantages_t


def compute_ppo_losses(
    args,
    batch,
    mb_inds,
    ratio,
    newvalue,
    newvalue_c,
    newvalue_t,
    c_scale,
    mb_advantages,
    sample_weights=None,
):
    """Compute PPO policy/value losses for one minibatch."""
    valid_only = getattr(args, 'ppo_boundary_mode', 'research') == 'valid_only'
    masked_critic = getattr(args, 'ppo_boundary_mode', 'research') in {'valid_only', 'exclude_value'}
    if masked_critic:
        valid = batch['policy_sample_valid'][mb_inds]
        ratio = torch.where(valid, ratio, torch.ones_like(ratio))
        mb_advantages = torch.where(valid, mb_advantages, torch.zeros_like(mb_advantages))
    clipped_ratio = torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * clipped_ratio
    actor_terms = torch.max(pg_loss1, pg_loss2)
    if valid_only:
        actor_terms = actor_terms[valid]
        actor_weights = sample_weights[valid] if sample_weights is not None else None
        pg_loss = weighted_mean(actor_terms, actor_weights) if actor_terms.numel() else actor_terms.sum()
    else:
        pg_loss = weighted_mean(actor_terms, sample_weights)
    value_weights = combine_sample_weights(
        sample_weights,
        batch.get("value_target_valid"),
        mb_inds,
        device=batch["returns"].device,
        dtype=batch["returns"].dtype,
    )

    v_loss_all = 0.0
    newvalue = newvalue.view(-1)
    if masked_critic:
        value_valid = batch['value_target_valid'][mb_inds]
        v_loss = (compute_value_loss(
            newvalue[value_valid], batch['values'][mb_inds][value_valid], batch['returns'][mb_inds][value_valid],
            args.clip_vloss, args.clip_coef, 0.5,
            sample_weights=value_weights[value_valid] if value_weights is not None else None,
        ) if bool(value_valid.any()) else newvalue[value_valid].sum())
        # Corrected modes support the ordinary single-critic PPO/TSIL objective.
        return pg_loss, v_loss, v_loss, None, None, None, mb_advantages
    v_loss = compute_value_loss(
        newvalue, batch["values"][mb_inds], batch["returns"][mb_inds],
        args.clip_vloss, args.clip_coef, 0.5, sample_weights=value_weights,
    )
    v_loss_all += v_loss

    v_loss_t = None
    if args.use_timeawareness:
        newvalue_t = newvalue_t.view(-1)
        v_loss_t = compute_value_loss(
            newvalue_t, batch["values_t"][mb_inds], batch["returns_t"][mb_inds],
            args.clip_vloss, args.clip_coef, 0.25, sample_weights=value_weights,
        )
        v_loss_all += v_loss_t

    v_loss_c = None
    L_viol = None
    if args.use_cost:
        mb_advantages_c = batch["advantages_c"][mb_inds]
        cost_loss1 = mb_advantages_c * ratio.view(-1, 1)
        cost_loss2 = mb_advantages_c * clipped_ratio.view(-1, 1)
        L_clip_c = weighted_mean(torch.max(cost_loss1, cost_loss2), sample_weights, dim=0)

        L_viol = L_clip_c + weighted_mean(batch["norm_return_c"][mb_inds], sample_weights, dim=0)
        L_viol = (c_scale * torch.clamp(L_viol, min=0.0)).sum()
        pg_loss += L_viol

        newvalue_c = newvalue_c.view(-1, args.num_cost)
        v_loss_c = compute_value_loss(
            newvalue_c, batch["values_c"][mb_inds], batch["returns_c"][mb_inds],
            args.clip_vloss, args.clip_coef, 0.5, reduce_dims=0, sample_weights=value_weights,
        )
        v_loss_all += v_loss_c

    return pg_loss, v_loss, v_loss_all, v_loss_t, v_loss_c, L_viol, mb_advantages
