"""PPO / GRPO policy-update component.

PolicyUpdater provides the core computation for policy gradient updates,
including GAE advantage estimation, batch preparation, and the minibatch
optimisation loop.  All state is received through explicit method parameters.


The high-level orchestration of a policy update (project hooks and logging)
stays in the trainer; this module only handles the math.
"""

import json
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from core.agents.utils import bound_loss
from core.training.algo.grpo import compute_grpo_losses, compute_grpo_returns
from core.training.algo.kl_regularization import SuccessActivatedPenalty, rollout_kl_penalty
from core.training.algo.ppo import (
    compute_ppo_advantages,
    compute_ppo_losses,
    reward_values_for_timeout_bootstrap,
)
from core.training.algo.tsil.loss import TsilReplayLoss
from core.training.algo.utils import compute_gae_jit


class PolicyUpdater:
    """PPO / GRPO policy-update computation.

    Constructor args
    ----------------
    args : training Args dataclass
    device, dtype : torch device and floating dtype
    """

    def __init__(self, args, device, dtype):
        self.args = args
        self.device = device
        self.dtype = dtype
        self.kl_penalty_activation = SuccessActivatedPenalty(
            getattr(args, "ppo_kl_activation_success", 0.0),
        )
        self.last_effective_kl_penalty = 0.0

    def update_state_dict(self):
        return {"kl_penalty_activation": self.kl_penalty_activation.state_dict()}

    def load_update_state_dict(self, state):
        self.kl_penalty_activation.load_state_dict(state["kl_penalty_activation"])

    # ------------------------------------------------------------------
    # Advantage / return computation
    # ------------------------------------------------------------------

    def compute_advantages(self, storage, agent, envs):
        """Compute GAE advantages and returns (PPO path)."""
        return compute_ppo_advantages(self.args, self.device, self.dtype, storage, agent, envs)

    def compute_grpo_returns(self, storage, agent, envs):
        """Compute reward-to-go returns for GRPO."""
        return compute_grpo_returns(self.args, self.device, self.dtype, storage, agent, envs)

    @staticmethod
    def _rollout_transition_masks(storage):
        next_done = torch.empty_like(storage.dones)
        next_timeout = torch.empty_like(storage.timeouts)
        if storage.dones.shape[0] > 1:
            next_done[:-1].copy_(storage.dones[1:])
            next_timeout[:-1].copy_(storage.timeouts[1:])
        next_done[-1].copy_(storage.next_done)
        next_timeout[-1].copy_(storage.next_timeout)

        valid = ~storage.dones.bool()
        timeout = valid & next_done.bool() & next_timeout.bool()
        terminal = valid & next_done.bool() & (~next_timeout.bool())
        return valid, timeout, terminal

    def compute_rollout_bootstrap_diagnostics(self, storage, agent, envs):
        """Compare PPO GAE targets with and without timeout value bootstrap."""
        if bool(getattr(self.args, "use_grpo", False)):
            return {}

        with torch.no_grad():
            if self.args.use_lstm:
                next_value, _, next_value_c, next_value_t = agent.get_value(
                    storage.next_state, storage.next_lstm_state, storage.next_done,
                )
            else:
                next_value, next_value_c, next_value_t = agent.get_value(storage.next_state)
            reward_values, next_value = reward_values_for_timeout_bootstrap(
                storage, agent, next_value,
            )

            c_gamma = torch.tensor(self.args.c_gamma, dtype=self.dtype, device=self.device).view(1, -1)
            returns_vb, adv_vb, _, _, _, _ = compute_gae_jit(
                storage.rewards, reward_values, storage.dones, storage.timeouts,
                next_value, storage.next_done, storage.next_timeout,
                self.args.gamma, self.args.gae_lambda, True,
                storage.costs, storage.values_c, next_value_c, c_gamma, self.args.use_cost,
                storage.values_t, next_value_t, envs.ctrl_dt, self.args.use_timeawareness,
            )
            returns_novb, adv_novb, _, _, _, _ = compute_gae_jit(
                storage.rewards, reward_values, storage.dones, storage.timeouts,
                next_value, storage.next_done, storage.next_timeout,
                self.args.gamma, self.args.gae_lambda, False,
                storage.costs, storage.values_c, next_value_c, c_gamma, self.args.use_cost,
                storage.values_t, next_value_t, envs.ctrl_dt, self.args.use_timeawareness,
            )

            num_steps = storage.rewards.shape[0]
            next_values = torch.empty_like(reward_values)
            if num_steps > 1:
                next_values[:-1].copy_(reward_values[1:])
            next_values[-1].copy_(next_value)

            valid_transition, timeout_transition, terminal_transition = (
                self._rollout_transition_masks(storage)
            )
            valid_count = valid_transition.to(dtype=torch.float32).sum().clamp_min(1.0)

            returns_delta = returns_vb - returns_novb
            adv_delta = adv_vb - adv_novb
            value_mse_vb = (storage.values - returns_vb).pow(2)
            value_mse_novb = (storage.values - returns_novb).pow(2)

            def masked_mean(values, mask):
                if not bool(mask.any().item()):
                    return 0.0
                return float(values[mask].detach().float().mean().item())

            def masked_positive_frac(values, mask):
                if not bool(mask.any().item()):
                    return 0.0
                return float((values[mask] > 0).to(dtype=torch.float32).mean().item())

            stats = {
                "rollout_timeout_transition_count": int(timeout_transition.sum().item()),
                "rollout_terminal_transition_count": int(terminal_transition.sum().item()),
                "rollout_timeout_transition_frac": float(
                    timeout_transition.to(dtype=torch.float32).sum().item() / float(valid_count.item())
                ),
                "rollout_terminal_transition_frac": float(
                    terminal_transition.to(dtype=torch.float32).sum().item() / float(valid_count.item())
                ),
                "rollout_timeout_next_value_mean": masked_mean(next_values, timeout_transition),
                "rollout_timeout_state_value_mean": masked_mean(storage.values, timeout_transition),
                "rollout_timeout_reward_mean": masked_mean(storage.rewards, timeout_transition),
                "rollout_timeout_return_vb_mean": masked_mean(returns_vb, timeout_transition),
                "rollout_timeout_return_novb_mean": masked_mean(returns_novb, timeout_transition),
                "rollout_timeout_return_delta_mean": masked_mean(returns_delta, timeout_transition),
                "rollout_timeout_adv_vb_mean": masked_mean(adv_vb, timeout_transition),
                "rollout_timeout_adv_novb_mean": masked_mean(adv_novb, timeout_transition),
                "rollout_timeout_adv_delta_mean": masked_mean(adv_delta, timeout_transition),
                "rollout_timeout_posadv_vb_frac": masked_positive_frac(adv_vb, timeout_transition),
                "rollout_timeout_posadv_novb_frac": masked_positive_frac(adv_novb, timeout_transition),
                "rollout_timeout_value_mse_vb": masked_mean(value_mse_vb, timeout_transition),
                "rollout_timeout_value_mse_novb": masked_mean(value_mse_novb, timeout_transition),
                "rollout_return_delta_mean": masked_mean(returns_delta, valid_transition),
                "rollout_adv_delta_mean": masked_mean(adv_delta, valid_transition),
                "rollout_value_mse_vb": masked_mean(value_mse_vb, valid_transition),
                "rollout_value_mse_novb": masked_mean(value_mse_novb, valid_transition),
            }
            ensemble_fn = getattr(agent, "get_reward_value_ensemble", None)
            if callable(ensemble_fn):
                flat_states = storage.states.reshape((-1,) + storage.states.shape[2:])
                all_values = ensemble_fn(flat_states, denorm=True, update=False).view(
                    *storage.values.shape, -1,
                )
                final_values = ensemble_fn(storage.next_state, denorm=True, update=False)
                next_ensemble = torch.empty((*storage.values.shape, all_values.shape[-1]), device=self.device)
                if num_steps > 1:
                    next_ensemble[:-1].copy_(all_values[1:])
                next_ensemble[-1].copy_(final_values)
                disagreement = (next_ensemble[..., 0] - next_ensemble[..., 1]).abs()
                stats.update({
                    "rollout_timeout_next_value_1_mean": masked_mean(next_ensemble[..., 0], timeout_transition),
                    "rollout_timeout_next_value_2_mean": masked_mean(next_ensemble[..., 1], timeout_transition),
                    "rollout_timeout_next_value_min_mean": masked_mean(
                        next_ensemble.min(dim=-1).values, timeout_transition,
                    ),
                    "rollout_timeout_critic_disagreement_mean": masked_mean(
                        disagreement, timeout_transition,
                    ),
                })
            return stats

    # ------------------------------------------------------------------
    # Batch preparation
    # ------------------------------------------------------------------

    def prepare_batch(self, storage, envs, agent, task_indices, unique_task_ids,
                      returns, advantages, returns_c, advantages_c, returns_t):
        """Flatten rollout data and normalise values / advantages."""
        args = self.args
        obs_shape = envs.obs_space.shape
        state_shape = envs.state_space.shape
        act_shape = envs.act_space.shape

        boundary_mode = getattr(args, "ppo_boundary_mode", "research")
        legacy_boundary = boundary_mode == "legacy"
        exclude_boundary_value = boundary_mode in {"exclude_value", "valid_only"}
        boundary = storage.dones.bool()
        _, timeout_transition, _ = self._rollout_transition_masks(storage)
        timeout_boundary = boundary & storage.timeouts.bool()
        terminal_boundary = boundary & (
            (~storage.timeouts.bool()) | (not bool(args.value_bootstrap))
        )
        returns = returns.clone()
        if not legacy_boundary and not exclude_boundary_value:
            returns[terminal_boundary] = 0.0
        advantages = advantages.clone()
        if not legacy_boundary:
            advantages[boundary] = 0.0
        if args.use_cost:
            returns_c = returns_c.clone()
            returns_c[terminal_boundary] = 0.0
            advantages_c = advantages_c.clone()
            advantages_c[boundary] = 0.0

        c_gamma = torch.tensor(args.c_gamma, dtype=self.dtype, device=self.device).view(1, -1)
        c_scale = torch.tensor(args.c_scale, dtype=self.dtype, device=self.device).view(1, -1)

        batch = {
            "obs": storage.obs.reshape((-1,) + obs_shape),
            "states": storage.states.reshape((-1,) + state_shape),
            "actions": storage.actions.reshape((-1,) + act_shape),
            "dones": storage.dones.reshape(-1),
            "logprobs": storage.logprobs.reshape(-1),
            "values": storage.values.reshape(-1),
            "advantages_raw": advantages.reshape(-1).clone(),
            "advantages": advantages.reshape(-1),
            "returns": returns.reshape(-1),
            "policy_sample_valid": (torch.ones_like(boundary) if legacy_boundary else ~boundary).reshape(-1),
            "timeout_transition": timeout_transition.reshape(-1),
            "value_target_valid": (
                torch.ones_like(boundary) if legacy_boundary else
                ~boundary if exclude_boundary_value else
                ~(timeout_boundary & bool(args.value_bootstrap))
            ).reshape(-1),
            "task_indices": task_indices.repeat(args.num_steps),
            "dense_rewards": getattr(
                storage, "dense_rewards", torch.zeros_like(advantages),
            ).reshape(-1),
            "dense_rewards_raw": getattr(
                storage, "dense_rewards_raw", torch.zeros_like(advantages),
            ).reshape(-1),
            "peg_target_distance": getattr(
                storage,
                "peg_target_distance",
                torch.full_like(advantages, float("nan")),
            ).reshape(-1),
        }
        if bool(getattr(args, "causal_shadow_diagnostics", False)) and not bool(args.use_grpo):
            _, advantages_no_dense, _, _, _, _ = compute_ppo_advantages(
                args,
                self.device,
                self.dtype,
                storage,
                agent,
                envs,
                rewards=storage.rewards - getattr(
                    storage, "dense_rewards", torch.zeros_like(storage.rewards),
                ),
            )
            advantages_no_dense = advantages_no_dense.reshape(-1)
            advantages_no_dense[boundary.reshape(-1)] = 0.0
            batch["advantages_no_dense_raw"] = advantages_no_dense.clone()
        if bool(getattr(args, "outcome_diagnostics", False)):
            episode_ids = storage.step_episode_ids.reshape(-1)
            unique_episode_ids, inverse = torch.unique(episode_ids, return_inverse=True)
            outcome_by_unique = torch.full(
                (int(unique_episode_ids.numel()),),
                -1,
                dtype=torch.int8,
                device=self.device,
            )
            dense_by_unique = torch.full(
                (int(unique_episode_ids.numel()),),
                torch.nan,
                dtype=torch.float32,
                device=self.device,
            )
            for index, episode_id in enumerate(unique_episode_ids.detach().cpu().tolist()):
                episode_id = int(episode_id)
                if episode_id in storage.rollout_completed_episode_success:
                    outcome_by_unique[index] = int(
                        bool(storage.rollout_completed_episode_success[episode_id])
                    )
                    dense_by_unique[index] = float(
                        storage.rollout_completed_episode_dense_return.get(episode_id, 0.0)
                    )
            batch["episode_ids"] = episode_ids
            batch["completed_episode_outcome"] = outcome_by_unique[inverse]
            batch["completed_episode_dense_return"] = dense_by_unique[inverse]

        if self._uses_agent_temporal_critic(agent):
            train_success_ids, eval_success_ids = self._temporal_success_episode_split(
                storage,
            )
            eval_targets = self._successful_temporal_mc_targets(
                storage,
                envs,
                episode_ids=eval_success_ids,
            )
            batch["temporal_model_targets"] = self._temporal_model_targets_from_rollout(
                storage,
                envs,
                agent,
                success_episode_ids=train_success_ids,
                eval_targets=eval_targets,
            )
            batch["temporal_model_eval_targets"] = eval_targets.reshape(-1)
        if bool(getattr(args, "track_signal_metrics", False)) and not bool(getattr(args, "use_grpo", False)):
            batch["_rollout_bootstrap_diagnostics"] = self.compute_rollout_bootstrap_diagnostics(
                storage, agent, envs,
            )

        if args.use_cost:
            batch["values_c"] = storage.values_c.reshape(-1, args.num_cost)
            batch["advantages_c"] = advantages_c.reshape(-1, args.num_cost)
            batch["returns_c"] = returns_c.reshape(-1, args.num_cost)
            batch["norm_return_c"] = batch["returns_c"].clone()

        if args.use_timeawareness:
            batch["values_t"] = storage.values_t.reshape(-1)
            batch["returns_t"] = returns_t.reshape(-1)

        if args.use_lstm:
            batch["lstm_states"] = tuple(
                s.permute(0, 2, 1, 3).reshape(-1, s.shape[1], s.shape[3]).clone()
                for s in storage.lstm_state_storage
            )

        if args.norm_value:
            task_ids = batch["task_indices"] if args.pertask_norm else None
            if not args.use_grpo:
                if legacy_boundary:
                    batch["returns"] = agent.normalize_value(batch["returns"], update=True, task_ids=task_ids)
                else:
                    valid = batch["value_target_valid"]
                    valid_task_ids = task_ids[valid] if task_ids is not None else None
                    agent.normalize_value(
                        batch["returns"][valid], update=True, task_ids=valid_task_ids,
                    )
                    batch["returns"] = agent.normalize_value(batch["returns"], task_ids=task_ids)
                batch["values"] = agent.normalize_value(batch["values"], task_ids=task_ids)
            if args.use_timeawareness:
                valid = batch["value_target_valid"]
                valid_task_ids = task_ids[valid] if task_ids is not None else None
                agent.normalize_value_t(
                    batch["returns_t"][valid], update=True, task_ids=valid_task_ids,
                )
                batch["returns_t"] = agent.normalize_value_t(batch["returns_t"], task_ids=task_ids)
                batch["values_t"] = agent.normalize_value_t(batch["values_t"], task_ids=task_ids)

        policy_valid = batch["policy_sample_valid"]
        if args.pertask_norm_adv:
            for task_id in unique_task_ids:
                task_mask = batch["task_indices"] == task_id
                mask = task_mask & policy_valid
                task_adv = batch["advantages"][mask]
                if task_adv.numel() > 0:
                    if "advantages_no_dense_raw" in batch:
                        no_dense = batch["advantages_no_dense_raw"][mask]
                        batch.setdefault(
                            "advantages_no_dense",
                            torch.zeros_like(batch["advantages_no_dense_raw"]),
                        )[mask] = (
                            no_dense - no_dense.mean()
                        ) / (no_dense.std(unbiased=False) + 1e-8)
                    batch["advantages"][mask] = (
                        task_adv - task_adv.mean()
                    ) / (task_adv.std(unbiased=legacy_boundary) + 1e-8)
                if args.use_cost:
                    task_adv_c = batch["advantages_c"][mask]
                    if task_adv_c.numel() > 0:
                        task_adv_c_mu = task_adv_c.mean(dim=0)
                        task_adv_c_std = task_adv_c.std(dim=0, unbiased=False)
                        batch["advantages_c"][mask] = (
                            task_adv_c - task_adv_c_mu
                        ) / (task_adv_c_std + 1e-8)
                        batch["norm_return_c"][task_mask] = (
                            (1.0 - c_gamma) * batch["returns_c"][task_mask] + task_adv_c_mu
                        ) / (task_adv_c_std + 1e-8)
        else:
            adv = batch["advantages"][policy_valid]
            if adv.numel() > 0:
                if "advantages_no_dense_raw" in batch:
                    no_dense = batch["advantages_no_dense_raw"][policy_valid]
                    batch["advantages_no_dense"] = torch.zeros_like(
                        batch["advantages_no_dense_raw"],
                    )
                    batch["advantages_no_dense"][policy_valid] = (
                        no_dense - no_dense.mean()
                    ) / (no_dense.std(unbiased=False) + 1e-8)
                batch["advantages"][policy_valid] = (
                    adv - adv.mean()
                ) / (adv.std(unbiased=legacy_boundary) + 1e-8)
            if args.use_cost:
                adv_c = batch["advantages_c"][policy_valid]
                if adv_c.numel() > 0:
                    adv_c_mu = adv_c.mean(dim=0)
                    adv_c_std = adv_c.std(dim=0, unbiased=False)
                    batch["advantages_c"][policy_valid] = (
                        adv_c - adv_c_mu
                    ) / (adv_c_std + 1e-8)
                    batch["norm_return_c"] = (
                        (1.0 - c_gamma) * batch["returns_c"] + adv_c_mu
                    ) / (adv_c_std + 1e-8)
        batch["advantages"][~policy_valid] = 0.0
        if "advantages_no_dense" in batch:
            batch["advantages_no_dense"][~policy_valid] = 0.0
        if args.use_cost:
            batch["advantages_c"][~policy_valid] = 0.0

        if bool(getattr(args, "track_signal_metrics", False)) and not bool(args.use_grpo):
            timeout_valid = batch["timeout_transition"] & policy_valid
            positive_rate = 0.0
            if bool(timeout_valid.any().item()):
                positive_rate = float(
                    (batch["advantages"][timeout_valid] > 0).to(dtype=torch.float32).mean().item()
                )
            diagnostics = batch.setdefault("_rollout_bootstrap_diagnostics", {})
            diagnostics["rollout_timeout_posadv_normalized_frac"] = positive_rate
            diagnostics["rollout_timeout_policy_grad_measured"] = 0

        # Stash constants used by the minibatch loop
        batch["_c_scale"] = c_scale

        return batch

    def _uses_agent_temporal_critic(self, agent):
        return (
            getattr(self.args, "use_sil", False)
            and str(getattr(self.args, "sil_temporal_model", "standalone")).lower() == "standalone"
            and str(getattr(self.args, "sil_filter_mode", "return")).lower() in {"temporal", "both"}
            and getattr(agent, "temporal_critic", None) is not None
        )

    @staticmethod
    def _temporal_success_episode_split(storage):
        successful_ids = sorted(
            int(episode_id)
            for episode_id, success in storage.rollout_completed_episode_success.items()
            if success
        )
        if len(successful_ids) < 2:
            return successful_ids, []
        eval_ids = successful_ids[::5]
        if len(eval_ids) == len(successful_ids):
            eval_ids = eval_ids[:1]
        eval_set = set(eval_ids)
        return [episode_id for episode_id in successful_ids if episode_id not in eval_set], eval_ids

    def _successful_temporal_mc_targets(self, storage, envs, episode_ids=None):
        num_steps, num_envs = storage.rewards.shape
        ctrl_dt = float(getattr(envs, "ctrl_dt", getattr(envs, "dt", 1.0)))
        if episode_ids is None:
            episode_ids = [
                int(episode_id)
                for episode_id, success in storage.rollout_completed_episode_success.items()
                if success
            ]
        targets = torch.full(
            (num_steps, num_envs), float("nan"), dtype=self.dtype, device=self.device,
        )
        step_grid = torch.arange(num_steps, dtype=self.dtype, device=self.device).unsqueeze(1)
        boundary = storage.dones.bool()
        successful_terminal = torch.zeros_like(boundary)
        for episode_id in episode_ids:
            episode_mask = (storage.step_episode_ids == episode_id) & ~boundary
            if not bool(episode_mask.any().item()):
                continue
            positions = episode_mask.nonzero(as_tuple=False)
            final_step_index = int(positions[:, 0].max().item())
            final_step = torch.tensor(final_step_index, dtype=self.dtype, device=self.device)
            remaining = (final_step - step_grid + 1.0) * ctrl_dt
            targets = torch.where(episode_mask, remaining.expand_as(targets), targets)
            terminal_step = final_step_index + 1
            if terminal_step < num_steps:
                terminal_envs = positions[positions[:, 0] == final_step_index, 1]
                successful_terminal[terminal_step, terminal_envs] = boundary[terminal_step, terminal_envs]

        targets = torch.where(successful_terminal, torch.zeros_like(targets), targets)

        max_eps_time = float(getattr(envs, "max_eps_time", 1e9))
        return torch.clamp(targets, min=0.0, max=max_eps_time)

    def _failed_terminal_temporal_mask(self, storage):
        _, _, terminal_transition = self._rollout_transition_masks(storage)
        failed_mask = torch.zeros_like(storage.dones, dtype=torch.bool)
        num_steps = int(storage.dones.shape[0])
        for episode_id, success in storage.rollout_completed_episode_success.items():
            if success:
                continue
            episode_mask = (storage.step_episode_ids == int(episode_id)) & ~storage.dones.bool()
            terminal_steps = (episode_mask & terminal_transition).nonzero(as_tuple=False)
            if not bool(terminal_steps.numel()):
                continue
            failed_mask |= episode_mask
            for step, env_id in terminal_steps.tolist():
                if step + 1 < num_steps:
                    failed_mask[step + 1, env_id] = True
        return failed_mask

    def _temporal_model_targets_from_rollout(
        self,
        storage,
        envs,
        agent,
        success_episode_ids=None,
        eval_targets=None,
    ):
        num_steps, num_envs = storage.rewards.shape
        ctrl_dt = float(getattr(envs, "ctrl_dt", getattr(envs, "dt", 1.0)))
        mc_targets = self._successful_temporal_mc_targets(
            storage,
            envs,
            episode_ids=success_episode_ids,
        )

        target_mode = str(getattr(self.args, "sil_temporal_target_mode", "mc")).lower()
        if target_mode == "mc":
            targets = mc_targets
        elif target_mode in {"td", "td_lambda"}:
            state_shape = storage.states.shape[2:]
            next_states = torch.empty_like(storage.states)
            if num_steps > 1:
                next_states[:-1].copy_(storage.states[1:])
            next_states[-1].copy_(storage.next_state)

            next_done = torch.empty_like(storage.dones)
            next_timeout = torch.empty_like(storage.timeouts)
            if num_steps > 1:
                next_done[:-1].copy_(storage.dones[1:])
                next_timeout[:-1].copy_(storage.timeouts[1:])
            next_done[-1].copy_(storage.next_done)
            next_timeout[-1].copy_(storage.next_timeout)
            with torch.no_grad():
                next_value_t = agent.get_temporal_filter_value(
                    next_states.reshape((-1,) + state_shape), update=False,
                ).view(num_steps, num_envs)
            bootstrap = (~next_done.bool()) | next_timeout.bool()
            td_targets = ctrl_dt + next_value_t * bootstrap.to(dtype=self.dtype)
            targets = td_targets
            if target_mode == "td_lambda":
                trace_lambda = float(getattr(self.args, "sil_temporal_lambda", 0.95))
                targets = td_targets.clone()
                for step in range(num_steps - 2, -1, -1):
                    can_trace = ~next_done[step].bool()
                    traced = ctrl_dt + (
                        (1.0 - trace_lambda) * next_value_t[step]
                        + trace_lambda * targets[step + 1]
                    )
                    targets[step] = torch.where(can_trace, traced, targets[step])
            targets = torch.where(
                storage.dones.bool(), torch.full_like(targets, float("nan")), targets,
            )
            true_terminal = storage.dones.bool() & (~storage.timeouts.bool())
            targets = torch.where(true_terminal, torch.zeros_like(targets), targets)
            targets = torch.where(
                self._failed_terminal_temporal_mask(storage),
                torch.full_like(targets, float("nan")),
                targets,
            )
        else:
            raise ValueError(f"Unknown temporal target mode: {target_mode}")

        if eval_targets is not None:
            targets = torch.where(
                torch.isfinite(eval_targets),
                torch.full_like(targets, float("nan")),
                targets,
            )
        max_eps_time = float(getattr(envs, "max_eps_time", 1e9))
        targets = torch.clamp(targets.reshape(-1), min=0.0, max=max_eps_time)
        return targets

    # ------------------------------------------------------------------
    # Minibatch update loop
    # ------------------------------------------------------------------

    def _flat_indices_for_epoch(self, b_inds, batch):
        """Return shuffled flat indices for one update epoch."""
        np.random.shuffle(b_inds)
        return b_inds

    def _compute_temporal_model_loss(self, agent, batch, mb_inds):
        if "temporal_model_targets" not in batch or not self._uses_agent_temporal_critic(agent):
            return torch.tensor(0.0, device=self.device), {}

        mb_inds = torch.as_tensor(mb_inds, dtype=torch.long, device=self.device)
        states = batch["states"][mb_inds].to(device=self.device, dtype=self.dtype)
        targets = batch["temporal_model_targets"][mb_inds].to(device=self.device, dtype=self.dtype)
        pred = agent.get_temporal_filter_value(states, update=False).view(-1)
        valid = torch.isfinite(targets)
        seen_samples = int(
            getattr(agent, "temporal_critic_seen_samples", torch.zeros(())).detach().item()
        )
        ready_samples = int(getattr(self.args, "sil_temporal_ready_samples", 0))
        if not bool(valid.any().item()):
            return torch.tensor(0.0, device=self.device), {
                "replay_temporal_model_updates": 0,
                "replay_temporal_model_samples": 0,
                "replay_temporal_model_ready": int(seen_samples >= ready_samples),
                "replay_temporal_model_seen_samples": seen_samples,
            }

        pred_valid = pred[valid]
        targets_valid = targets[valid]
        loss = 0.5 * (pred_valid - targets_valid).pow(2).mean()
        samples = int(valid.sum().item())

        stats = {
            "replay_temporal_model_loss": float(loss.detach().item()),
            "replay_temporal_model_updates": 1,
            "replay_temporal_model_samples": samples,
            "replay_temporal_model_ready": int(seen_samples >= ready_samples),
            "replay_temporal_model_seen_samples": seen_samples,
            "replay_temporal_model_target_mean": float(targets_valid.detach().mean().item()),
        }
        return loss, stats

    def temporal_model_evaluation(self, agent, batch):
        targets = batch.get("temporal_model_eval_targets")
        if targets is None or not self._uses_agent_temporal_critic(agent):
            return {}
        valid = torch.isfinite(targets)
        samples = int(valid.sum().item())
        stats = {
            "replay_temporal_model_eval_samples": samples,
            "replay_temporal_model_pred_mean": 0.0,
            "replay_temporal_model_mae": 0.0,
            "replay_temporal_model_rmse": 0.0,
            "replay_temporal_model_corr": 0.0,
            "replay_temporal_countdown_mae": 0.0,
            "replay_temporal_countdown_corr": 0.0,
        }
        if samples == 0:
            return stats

        states = batch["states"][valid].to(device=self.device, dtype=self.dtype)
        targets = targets[valid].to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            predictions = agent.get_temporal_filter_value(states, update=False).view(-1)
        errors = predictions - targets
        countdown = torch.clamp(states[:, 2] - states[:, 1], min=0.0)

        def correlation(first, second):
            if int(first.numel()) < 2:
                return 0.0
            first = first.float() - first.float().mean()
            second = second.float() - second.float().mean()
            denominator = first.square().mean().sqrt() * second.square().mean().sqrt()
            if float(denominator.item()) <= 1e-8:
                return 0.0
            return float((first * second).mean().div(denominator).item())

        stats.update({
            "replay_temporal_model_pred_mean": float(predictions.mean().item()),
            "replay_temporal_model_mae": float(errors.abs().mean().item()),
            "replay_temporal_model_rmse": float(errors.square().mean().sqrt().item()),
            "replay_temporal_model_corr": correlation(predictions, targets),
            "replay_temporal_countdown_mae": float((countdown - targets).abs().mean().item()),
            "replay_temporal_countdown_corr": correlation(countdown, targets),
        })
        return stats

    def _loss_grad_vector(self, loss, agent):
        params = self._policy_parameters(agent)
        if not params or not isinstance(loss, torch.Tensor) or not loss.requires_grad:
            return None
        grads = torch.autograd.grad(
            loss,
            params,
            allow_unused=True,
            retain_graph=True,
        )
        return self._pack_grads(params, grads).detach()

    @staticmethod
    def _gradient_cosine(first, second):
        denom = first.norm() * second.norm()
        if float(denom.item()) <= 1e-12:
            return 0.0
        return float(torch.dot(first, second).div(denom).item())

    def _timeout_policy_gradient_diagnostics(
        self,
        batch,
        mb_inds,
        ratio,
        mb_advantages,
        agent,
    ):
        valid = batch["policy_sample_valid"][mb_inds]
        timeout = batch["timeout_transition"][mb_inds] & valid
        if not bool(timeout.any().item()):
            return None

        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.args.clip_coef,
            1.0 + self.args.clip_coef,
        )
        per_sample_loss = torch.maximum(
            -mb_advantages * ratio,
            -mb_advantages * clipped_ratio,
        )
        denominator = valid.to(dtype=per_sample_loss.dtype).sum().clamp_min(1.0)
        all_loss = (per_sample_loss * valid.to(dtype=per_sample_loss.dtype)).sum() / denominator
        timeout_loss = (
            per_sample_loss * timeout.to(dtype=per_sample_loss.dtype)
        ).sum() / denominator

        all_grad = self._loss_grad_vector(all_loss, agent)
        timeout_grad = self._loss_grad_vector(timeout_loss, agent)
        if all_grad is None or timeout_grad is None:
            return None
        all_norm = all_grad.norm()
        timeout_norm = timeout_grad.norm()
        ratio_value = 0.0
        if float(all_norm.item()) > 1e-12:
            ratio_value = float(timeout_norm.div(all_norm).item())
        return {
            "rollout_timeout_policy_grad_measured": 1,
            "rollout_timeout_policy_grad_norm": float(timeout_norm.item()),
            "rollout_policy_grad_norm": float(all_norm.item()),
            "rollout_timeout_policy_grad_norm_ratio": ratio_value,
            "rollout_timeout_policy_grad_cosine": self._gradient_cosine(
                timeout_grad,
                all_grad,
            ),
        }

    def _replay_policy_gradient_diagnostics(
        self,
        agent,
        ppo_policy_loss,
        replay_policy_loss,
        replay_value_loss=None,
        ppo_value_loss=None,
        components=None,
    ):
        ppo_grad = self._loss_grad_vector(ppo_policy_loss, agent)
        replay_grad = self._loss_grad_vector(
            float(self.args.sil_coef) * replay_policy_loss,
            agent,
        )
        if ppo_grad is None or replay_grad is None:
            return None
        ppo_norm = ppo_grad.norm()
        replay_norm = replay_grad.norm()
        ratio_value = 0.0
        if float(ppo_norm.item()) > 1e-12:
            ratio_value = float(replay_norm.div(ppo_norm).item())
        result = {
            "replay_gradient_measured": 1,
            "replay_ppo_policy_grad_norm": float(ppo_norm.item()),
            "replay_policy_grad_norm": float(replay_norm.item()),
            "replay_to_ppo_grad_norm_ratio": ratio_value,
            "replay_to_ppo_grad_cosine": self._gradient_cosine(
                replay_grad,
                ppo_grad,
            ),
        }

        if components is not None:
            for label, component in zip(("success", "failure"), components):
                g = self._loss_grad_vector(float(self.args.sil_coef) * component, agent)
                result["replay_" + label + "_policy_grad_norm"] = float(g.norm().item())
                result["replay_" + label + "_to_ppo_grad_cosine"] = self._gradient_cosine(g, ppo_grad)
        if replay_value_loss is not None and ppo_value_loss is not None:
            params = tuple(p for p in agent.critic.parameters() if p.requires_grad)
            def norm(loss):
                gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                return float(torch.stack([g.square().sum() for g in gs if g is not None]).sum().sqrt().item())
            result["replay_value_grad_norm"] = norm(float(self.args.sil_coef) * float(self.args.sil_vf_coef) * replay_value_loss)
            result["replay_ppo_value_grad_norm"] = norm(float(self.args.vf_coef) * ppo_value_loss)
        return result

    def _outcome_policy_gradient_diagnostics(self, batch, agent):
        outcomes = batch.get("completed_episode_outcome")
        dense_returns = batch.get("completed_episode_dense_return")
        if outcomes is None or dense_returns is None:
            return {}

        valid = batch["policy_sample_valid"]
        success = valid & (outcomes == 1)
        failure = valid & (outcomes == 0)
        if not bool(success.any().item()) or not bool(failure.any().item()):
            return {
                "outcome_gradient_measured": 0,
                "outcome_gradient_success_sample_count": int(success.sum().item()),
                "outcome_gradient_failure_sample_count": int(failure.sum().item()),
            }

        failure_dense = dense_returns[failure]
        threshold = torch.quantile(failure_dense.float(), 0.9)
        high_dense_failure = failure & (dense_returns >= threshold)
        max_items = max(int(getattr(self.args, "outcome_diagnostic_batch_size", 4096)), 1)

        def selected_indices(mask):
            indices = torch.nonzero(mask, as_tuple=False).view(-1)
            if int(indices.numel()) <= max_items:
                return indices
            positions = torch.linspace(
                0,
                int(indices.numel()) - 1,
                max_items,
                device=indices.device,
            ).round().to(torch.long)
            return indices[positions]

        directions = {}
        norms = {}
        counts = {}
        for name, mask in (
            ("success", success),
            ("failure", failure),
            ("high_dense_failure", high_dense_failure),
            ("all", valid),
        ):
            indices = selected_indices(mask)
            counts[name] = int(mask.sum().item())
            if int(indices.numel()) == 0:
                continue
            loss = self.policy_loss_for_indices(agent, batch, indices)
            gradient = self._loss_grad_vector(loss, agent)
            if gradient is None:
                continue
            directions[name] = -gradient
            norms[name] = float(gradient.norm().item())

        def cosine(first, second):
            if first not in directions or second not in directions:
                return 0.0
            return self._gradient_cosine(directions[first], directions[second])

        return {
            "outcome_gradient_measured": int("high_dense_failure" in directions),
            "outcome_gradient_success_sample_count": counts.get("success", 0),
            "outcome_gradient_failure_sample_count": counts.get("failure", 0),
            "outcome_gradient_high_dense_failure_sample_count": counts.get("high_dense_failure", 0),
            "outcome_gradient_success_norm": norms.get("success", 0.0),
            "outcome_gradient_failure_norm": norms.get("failure", 0.0),
            "outcome_gradient_high_dense_failure_norm": norms.get("high_dense_failure", 0.0),
            "outcome_gradient_all_norm": norms.get("all", 0.0),
            "outcome_gradient_failure_success_cosine": cosine("failure", "success"),
            "outcome_gradient_high_dense_failure_success_cosine": cosine(
                "high_dense_failure", "success"
            ),
            "outcome_gradient_high_dense_failure_all_cosine": cosine(
                "high_dense_failure", "all"
            ),
        }

    def run_minibatch_updates(self, batch, agent, optimizer, initial_lstm_state,
                              lr_scheduler, cur_ent, global_update_iter, global_step=0,
                              replay_module=None, replay_context=None, current_success_rate=None):
        """Run the PPO / GRPO minibatch update loop.

        Returns
        -------
        policy_diverged, valid_met, num_agent_updates,
        agent_params_store, optim_params_store, illed_met, replay_stats
        """
        args = self.args

        effective_kl_penalty = self.kl_penalty_activation.coefficient(
            getattr(args, "ppo_kl_penalty_coef", 0.0), current_success_rate,
        )
        self.last_effective_kl_penalty = effective_kl_penalty

        if args.use_lstm:
            envsperbatch = args.num_envs // args.num_minibatches
            envinds = np.arange(args.num_envs)
            flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
            end_idx = args.num_envs
            step_num = envsperbatch
        else:
            b_inds = np.arange(args.batch_size)
            end_idx = args.batch_size
            step_num = args.minibatch_size

        self.last_update_trace = []
        trace_enabled = bool(getattr(args, "update_diagnostic_trace", False))
        num_agent_updates = 0
        policy_diverged = False
        is_warmup = global_update_iter < args.warmup_iters
        update_norm = args.norm_obs and not args.tw_train
        agent_params_store = deepcopy(agent.state_dict())
        optim_params_store = deepcopy(optimizer.state_dict())
        separate_replay_updates = int(getattr(args, "sil_separate_updates", 0))
        use_replay = (
            replay_module is not None
            and replay_context is not None
            and separate_replay_updates <= 0
        )

        c_scale = batch["_c_scale"]

        valid_met = self._init_valid_metrics()
        illed_met = self._init_diverged_metrics()
        replay_helper = replay_module if replay_module is not None else TsilReplayLoss
        if use_replay and not getattr(args, "sil_actual_direction_diagnostics", False):
            replay_helper.update_diagnostics(
                agent, replay_context, batch, self, global_update_iter,
            )
        base_replay_stats = replay_helper.init_iteration_stats(replay_context)
        replay_totals = replay_helper.init_total_stats()
        (
            replay_policy_loss,
            replay_value_loss,
            replay_positive_frac,
            replay_sampled_positive_frac,
            replay_step_count,
        ) = (
            self._zero_replay_batch_losses()
        )
        replay_total_loss = torch.tensor(0.0, device=self.device)
        policy_grad_norm_pre_noise = torch.tensor(0.0, device=self.device)
        policy_grad_norm_post_noise = torch.tensor(0.0, device=self.device)
        policy_grad_norm_post_clip = torch.tensor(0.0, device=self.device)
        policy_grad_noise_diagnostics = self._zero_policy_grad_noise_diagnostics()
        gradient_interval = int(getattr(args, "signal_gradient_interval", 0))
        collect_gradient_diagnostics = (
            gradient_interval > 0
            and int(global_update_iter) % gradient_interval == 0
        )
        if collect_gradient_diagnostics and bool(getattr(args, "outcome_diagnostics", False)):
            batch.setdefault("_rollout_bootstrap_diagnostics", {}).update(
                self._outcome_policy_gradient_diagnostics(batch, agent)
            )
        timeout_gradient_measured = False
        replay_gradient_measured = False
        replay_batches_per_epoch = max(len(range(0, end_idx, step_num)) - 1, 1)
        replay_update_interval = max(int(getattr(args, "sil_update_interval", 1)), 1)
        replay_batches_per_epoch = max((replay_batches_per_epoch + replay_update_interval - 1) // replay_update_interval, 1)
        event_recorder = (replay_context or {}).get("_update_event_recorder")
        if event_recorder is not None:
            event_recorder.prepared(replay_helper, replay_context)
        previous_update_pending = False
        for epoch in range(args.update_epochs):
            if args.use_lstm:
                np.random.shuffle(envinds)
            else:
                b_inds = self._flat_indices_for_epoch(b_inds, batch)
            if use_replay:
                replay_helper.reset_epoch_sampler(replay_context, replay_batches_per_epoch, agent=agent)
                base_replay_stats.update({key: value for key, value in replay_context.get("stats", {}).items()
                                          if key.startswith("replay_screen_")})
                if event_recorder is not None:
                    event_recorder.replay_epoch(epoch, replay_context)

            for iter_idx, start in enumerate(range(0, end_idx, step_num)):
                end = start + step_num
                last_batch = False
                if end >= end_idx:
                    last_batch = True
                    start = max(0, end_idx - step_num)

                if args.use_lstm:
                    mbenvinds = envinds[start:end]
                    mb_inds = flatinds[:, mbenvinds].ravel()
                    _, mu, newlogprob, entropy, newvalue, _, newvalue_c, newvalue_t = agent.get_action_and_value(
                        batch["obs"][mb_inds],
                        (
                            initial_lstm_state[0][:, mbenvinds],
                            initial_lstm_state[1][:, mbenvinds],
                            initial_lstm_state[2][:, mbenvinds],
                            initial_lstm_state[3][:, mbenvinds],
                        ),
                        batch["dones"][mb_inds],
                        batch["states"][mb_inds],
                        batch["actions"][mb_inds],
                        denorm=False,
                        update=update_norm,
                    )
                else:
                    mb_inds = b_inds[start:end]
                    if not bool(batch["policy_sample_valid"][mb_inds].any().item()):
                        if trace_enabled:
                            self.last_update_trace.append({"iteration": int(global_update_iter),
                                "epoch": epoch, "minibatch": iter_idx, "skipped_all_invalid": True})
                        continue
                    _, mu, newlogprob, entropy, newvalue, newvalue_c, newvalue_t = agent.get_action_and_value(
                        batch["obs"][mb_inds],
                        batch["states"][mb_inds],
                        batch["actions"][mb_inds],
                        denorm=False,
                        update=update_norm,
                    )

                policy_valid_mb = batch["policy_sample_valid"][mb_inds]
                logratio = newlogprob - batch["logprobs"][mb_inds]
                # Mask before exp: exp(inf) * zero can still poison backward.
                logratio = torch.where(policy_valid_mb, logratio, torch.zeros_like(logratio))
                ratio = logratio.exp()
                entropy_mean = entropy[policy_valid_mb].mean()

                with torch.no_grad():
                    approx_kl = ((ratio[policy_valid_mb] - 1) - logratio[policy_valid_mb]).mean()
                    if trace_enabled:
                        self.last_update_trace.append({
                            "iteration": int(global_update_iter), "step": int(global_step),
                            "epoch": epoch, "minibatch": iter_idx, "validation_only": bool(last_batch),
                            "approx_kl": float(approx_kl.item()),
                            "clip_fraction": float(((ratio[policy_valid_mb] - 1).abs() > args.clip_coef).float().mean().item()),
                            "ratio_max": float(ratio[policy_valid_mb].max().item()),
                            "entropy": float(entropy_mean.item()),
                            "value_mean": float(newvalue.mean().item()),
                            "value_abs_max": float(newvalue.abs().max().item()),
                            "target_mean": float(batch["returns"][mb_inds].mean().item()),
                            "lr": float(optimizer.param_groups[0]["lr"]),
                            "hard_threshold_exceeded": bool(self._check_kl_divergence(approx_kl)),
                        })

                    if iter_idx > 0 and previous_update_pending:
                        policy_diverged = self._check_kl_divergence(approx_kl)
                        if policy_diverged:
                            self._record_diverged_update(
                                illed_met,
                                approx_kl,
                                ratio,
                                entropy_mean,
                                mb_advantages,
                            )
                            continue

                        num_agent_updates += 1
                        agent_params_store, optim_params_store = self._record_valid_update(
                            valid_met,
                            approx_kl,
                            entropy_mean,
                            mb_advantages,
                            pg_loss,
                            v_loss,
                            v_loss_t,
                            v_loss_c,
                            L_viol,
                            bd_loss,
                            policy_grad_norm_pre_noise,
                            policy_grad_norm_post_noise,
                            policy_grad_norm_post_clip,
                            policy_grad_noise_diagnostics,
                            agent,
                            optimizer,
                        )
                        if use_replay:
                            replay_helper.accumulate_total_stats(
                                replay_totals,
                                replay_policy_loss,
                                replay_value_loss,
                                replay_total_loss,
                                replay_positive_frac,
                                replay_sampled_positive_frac,
                                replay_step_count,
                                replay_context.get("_last_sil_batch_stats", {}) if replay_context is not None else {},
                            )

                        previous_update_pending = False
                        if last_batch:
                            break

                if last_batch:
                    break
                if args.use_grpo:
                    pg_loss, v_loss, v_loss_all, v_loss_t, v_loss_c, L_viol, mb_advantages = (
                        self._compute_grpo_losses(batch, mb_inds, ratio, newlogprob, newvalue_c, newvalue_t, c_scale)
                    )
                else:
                    pg_loss, v_loss, v_loss_all, v_loss_t, v_loss_c, L_viol, mb_advantages = (
                        self._compute_ppo_losses(batch, mb_inds, ratio, newvalue, newvalue_c, newvalue_t, c_scale)
                    )

                if collect_gradient_diagnostics and not timeout_gradient_measured:
                    timeout_gradient_stats = self._timeout_policy_gradient_diagnostics(
                        batch,
                        mb_inds,
                        ratio,
                        mb_advantages,
                        agent,
                    )
                    if timeout_gradient_stats is not None:
                        batch.setdefault("_rollout_bootstrap_diagnostics", {}).update(
                            timeout_gradient_stats
                        )
                        timeout_gradient_measured = True

                run_replay_update = use_replay and (iter_idx % replay_update_interval == 0)
                temporal_model_loss, temporal_model_stats = self._compute_temporal_model_loss(agent, batch, mb_inds)
                auxiliary_value_loss = torch.zeros((), device=self.device, dtype=self.dtype)
                auxiliary_value_fn = getattr(agent, "auxiliary_value_loss", None)
                if callable(auxiliary_value_fn):
                    auxiliary_value_loss = auxiliary_value_fn(
                        batch["states"][mb_inds],
                        batch["returns"][mb_inds],
                        batch["value_target_valid"][mb_inds],
                    )
                if run_replay_update and replay_context is not None and temporal_model_stats:
                    replay_context["_temporal_model_update_stats"] = temporal_model_stats
                (
                    replay_policy_loss,
                    replay_value_loss,
                    replay_positive_frac,
                    replay_sampled_positive_frac,
                    replay_step_count,
                ) = (
                    replay_module.compute_joint_losses(agent, replay_context)
                    if run_replay_update else self._zero_replay_batch_losses()
                )
                if (
                    collect_gradient_diagnostics
                    and not replay_gradient_measured
                    and run_replay_update
                    and int(replay_step_count) > 0
                    and replay_policy_loss.requires_grad
                ):
                    replay_gradient_stats = self._replay_policy_gradient_diagnostics(
                        agent,
                        pg_loss,
                        replay_policy_loss,
                        replay_value_loss=replay_value_loss,
                        ppo_value_loss=v_loss_all,
                        components=getattr(replay_module, "_last_actor_components", None),
                    )
                    if replay_gradient_stats is not None:
                        base_replay_stats.update(replay_gradient_stats)
                        replay_gradient_measured = True

                if use_replay and hasattr(replay_module, 'capture_pre_step_landscape'):
                    replay_module.capture_pre_step_landscape(
                        agent, replay_context, float(not is_warmup) * pg_loss,
                        replay_policy_loss if run_replay_update else None,
                        global_update_iter, mb_inds, policy_valid_mb, epoch, iter_idx)

                pg_coef, ent_coef = float(not is_warmup), float(not is_warmup)
                loss = pg_coef * pg_loss + args.vf_coef * v_loss_all - ent_coef * cur_ent * entropy_mean
                loss += pg_coef * rollout_kl_penalty(
                    logratio, policy_valid_mb, effective_kl_penalty,
                )
                loss += args.vf_coef * auxiliary_value_loss
                loss += float(getattr(args, "sil_temporal_vf_coef", 0.25)) * temporal_model_loss
                replay_total_loss = float(args.sil_coef) * (
                    replay_policy_loss + float(args.sil_vf_coef) * replay_value_loss
                )
                loss += replay_total_loss

                bd_loss = 0.0
                if not args.beta:
                    bd_loss = args.bounds_loss_coef * bound_loss(mu[policy_valid_mb], soft_bound=1.0)
                    loss += bd_loss

                policy_grad_noise = []
                clean_ppo_policy_grads = []
                corrupted_ppo_policy_grads = []
                policy_grad_noise_diagnostics = self._zero_policy_grad_noise_diagnostics()
                use_policy_grad_noise = (
                    float(
                        getattr(
                            args,
                            "effective_policy_grad_noise_scale",
                            getattr(args, "policy_grad_noise_scale", 0.0),
                        )
                    ) > 0.0
                )
                if use_policy_grad_noise:
                    ppo_policy_loss = pg_coef * pg_loss - ent_coef * cur_ent * entropy_mean
                    optimizer.zero_grad()
                    ppo_policy_loss.backward(retain_graph=True)
                    policy_parameters = self._policy_parameters(agent)
                    clean_ppo_policy_grads = self._capture_parameter_grads(policy_parameters)
                    policy_grad_norm_pre_noise = self._captured_grad_norm(clean_ppo_policy_grads)
                    policy_grad_noise = self._sample_policy_grad_noise(agent)
                    if self._policy_grad_noise_mode() == "norm_preserved_direction":
                        (
                            corrupted_ppo_policy_grads,
                            policy_grad_noise_diagnostics,
                        ) = self._norm_preserved_corrupted_grads(
                            clean_ppo_policy_grads,
                            policy_grad_noise,
                        )
                        policy_grad_norm_post_noise = policy_grad_noise_diagnostics[
                            "policy_grad_norm_restored_noise"
                        ]
                    else:
                        policy_grad_norm_post_noise = self._grad_norm_with_added_noise(
                            policy_parameters, policy_grad_noise,
                        )
                        policy_grad_noise_diagnostics.update({
                            "policy_grad_norm_raw_noise": policy_grad_norm_post_noise,
                            "policy_grad_norm_restored_noise": policy_grad_norm_post_noise,
                        })

                optimizer.zero_grad()
                loss.backward()
                self.after_policy_backward(
                    agent,
                    batch,
                    mb_inds,
                    policy_loss_scale=pg_coef,
                )
                if corrupted_ppo_policy_grads:
                    replacement_stats = self._replace_policy_grad_component(
                        clean_ppo_policy_grads,
                        corrupted_ppo_policy_grads,
                    )
                    policy_grad_noise_diagnostics.update(replacement_stats)
                elif policy_grad_noise:
                    self._add_grad_noise(policy_grad_noise)
                elif not use_policy_grad_noise:
                    policy_grad_norm_pre_noise = self._grad_norm(self._policy_parameters(agent))
                    policy_grad_norm_post_noise = policy_grad_norm_pre_noise
                self._clip_gradients(agent)
                policy_grad_norm_post_clip = self._grad_norm(self._policy_parameters(agent))
                optimizer.step()
                if replay_module is not None:
                    replay_module._last_actor_components = None
                previous_update_pending = True

                self._update_learning_rate(optimizer, lr_scheduler, approx_kl, global_step)

            if policy_diverged:
                break

        if trace_enabled and getattr(args, "instance_dir", None):
            with open(os.path.join(args.instance_dir, "minibatch_trace.jsonl"), "a") as stream:
                for row in self.last_update_trace:
                    stream.write(json.dumps(row) + "\n")
        replay_stats = replay_helper.finalize_iteration_stats(
            base_replay_stats,
            replay_totals,
            num_agent_updates,
        )

        return (
            policy_diverged,
            valid_met,
            num_agent_updates,
            agent_params_store,
            optim_params_store,
            illed_met,
            replay_stats,
        )

    def run_separate_replay_updates(
        self,
        batch,
        agent,
        optimizer,
        replay_module,
        replay_context,
        global_update_iter,
    ):
        """Run SIL replay as a small optimizer phase after PPO."""
        args = self.args
        num_updates = int(getattr(args, "sil_separate_updates", 0))
        if num_updates <= 0 or replay_module is None or replay_context is None:
            return TsilReplayLoss.init_iteration_stats(replay_context)

        replay_module.update_diagnostics(
            agent, replay_context, batch, self, global_update_iter,
        )
        base_replay_stats = replay_module.init_iteration_stats(replay_context)
        replay_totals = replay_module.init_total_stats()
        replay_module.reset_epoch_sampler(replay_context, num_updates, agent=agent)

        for _ in range(num_updates):
            (
                replay_policy_loss,
                replay_value_loss,
                replay_positive_frac,
                replay_sampled_positive_frac,
                replay_step_count,
            ) = replay_module.compute_joint_losses(agent, replay_context)

            replay_total_loss = float(args.sil_coef) * (
                replay_policy_loss + float(args.sil_vf_coef) * replay_value_loss
            )
            if (
                int(replay_step_count) <= 0
                or not replay_total_loss.requires_grad
                or not torch.isfinite(replay_total_loss)
            ):
                break

            optimizer.zero_grad()
            replay_total_loss.backward()
            self._clip_gradients(agent)
            optimizer.step()

            replay_module.accumulate_total_stats(
                replay_totals,
                replay_policy_loss,
                replay_value_loss,
                replay_total_loss,
                replay_positive_frac,
                replay_sampled_positive_frac,
                replay_step_count,
                replay_context.get("_last_sil_batch_stats", {}),
            )

        return replay_module.finalize_iteration_stats(
            base_replay_stats,
            replay_totals,
            num_updates,
        )

    def policy_loss_for_indices(self, agent, batch, indices, sample_weights=None):
        if getattr(self.args, "ppo_boundary_mode", "research") == "valid_only":
            keep = batch["policy_sample_valid"][indices]
            indices = torch.as_tensor(indices, device=keep.device)[keep]
            if sample_weights is not None:
                sample_weights = sample_weights[keep]
            if indices.numel() == 0:
                return next(agent.parameters()).reshape(-1)[:0].sum()
        newlogprob = agent.get_logprob(batch["obs"][indices], batch["actions"][indices], update=False)
        ratio = (newlogprob - batch["logprobs"][indices]).exp()
        advantages = batch["advantages"][indices]
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
        losses = torch.max(pg_loss1, pg_loss2)
        if sample_weights is None:
            return losses.mean()
        weights = sample_weights.to(device=losses.device, dtype=losses.dtype)
        return (losses * weights).sum() / weights.sum().clamp_min(1e-8)

    def policy_logprob_delta(self, agent, batch):
        """Measure the realized joint update on every sampled rollout action."""
        if not bool(getattr(self.args, "outcome_diagnostics", False)):
            return None
        count = int(batch["actions"].shape[0])
        chunk_size = max(int(getattr(self.args, "outcome_diagnostic_batch_size", 4096)), 1)
        post_logprob = torch.empty_like(batch["logprobs"])
        with torch.no_grad():
            for start in range(0, count, chunk_size):
                end = min(start + chunk_size, count)
                post_logprob[start:end] = agent.get_logprob(
                    batch["obs"][start:end],
                    batch["actions"][start:end],
                    update=False,
                ).detach().view(-1)
        return post_logprob - batch["logprobs"]

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def _compute_ppo_losses(self, batch, mb_inds, ratio, newvalue, newvalue_c, newvalue_t, c_scale):
        mb_advantages = batch["advantages"][mb_inds]
        sample_weights = self._minibatch_sample_weights(batch, mb_inds)
        return compute_ppo_losses(
            self.args, batch, mb_inds, ratio, newvalue, newvalue_c, newvalue_t,
            c_scale, mb_advantages, sample_weights=sample_weights,
        )

    def _compute_grpo_losses(self, batch, mb_inds, ratio, newlogprob, newvalue_c, newvalue_t, c_scale):
        mb_advantages = batch["advantages"][mb_inds]
        sample_weights = self._minibatch_sample_weights(batch, mb_inds)
        return compute_grpo_losses(
            self.args, self.device, batch, mb_inds, ratio, newlogprob,
            newvalue_c, newvalue_t, c_scale, mb_advantages, sample_weights=sample_weights,
        )

    def _minibatch_sample_weights(self, batch, mb_inds):
        return None

    def after_policy_backward(self, agent, batch, mb_inds, policy_loss_scale=1.0):
        """Project hook after the full minibatch loss is backpropagated."""

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _init_valid_metrics(self):
        return {
            "pg_loss": 0.0,
            "v_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "mb_advantages": 0.0,
            "v_loss_t": 0.0,
            "v_loss_c": 0.0,
            "cost_loss": 0.0,
            "bound_loss": 0.0,
            "policy_grad_norm_pre_noise": 0.0,
            "policy_grad_norm_post_noise": 0.0,
            "policy_grad_norm_post_clip": 0.0,
            **{
                key: 0.0
                for key in self._zero_policy_grad_noise_diagnostics()
            },
        }

    def _zero_policy_grad_noise_diagnostics(self):
        return {
            "policy_grad_noise_norm": torch.tensor(0.0, device=self.device),
            "policy_grad_norm_raw_noise": torch.tensor(0.0, device=self.device),
            "policy_grad_norm_restored_noise": torch.tensor(0.0, device=self.device),
            "policy_grad_clean_corrupted_cosine": torch.tensor(0.0, device=self.device),
            "policy_grad_nonppo_norm": torch.tensor(0.0, device=self.device),
            "policy_grad_final_pre_clip_norm": torch.tensor(0.0, device=self.device),
            "policy_grad_corrupted_nonppo_cosine": torch.tensor(0.0, device=self.device),
            "policy_grad_final_nonppo_cosine": torch.tensor(0.0, device=self.device),
            "policy_grad_final_clean_ppo_cosine": torch.tensor(0.0, device=self.device),
        }

    def _init_diverged_metrics(self):
        return {
            "approx_kl": 0.0,
            "mb_advantages": 0.0,
            "ratio": 0.0,
            "entropy_mean": 0.0,
        }

    def _zero_replay_batch_losses(self):
        return (
            torch.tensor(0.0, device=self.device),
            torch.tensor(0.0, device=self.device),
            0.0,
            0.0,
            0,
        )

    def _check_kl_divergence(self, approx_kl):
        return self.args.target_kl is not None and approx_kl > self.args.target_kl

    def _record_diverged_update(self, illed_met, approx_kl, ratio, entropy_mean, mb_advantages):
        illed_met.update({
            "approx_kl": approx_kl,
            "ratio": ratio,
            "entropy_mean": entropy_mean,
            "mb_advantages": mb_advantages,
        })

    def _record_valid_update(
        self,
        valid_met,
        approx_kl,
        entropy_mean,
        mb_advantages,
        pg_loss,
        v_loss,
        v_loss_t,
        v_loss_c,
        L_viol,
        bd_loss,
        policy_grad_norm_pre_noise,
        policy_grad_norm_post_noise,
        policy_grad_norm_post_clip,
        policy_grad_noise_diagnostics,
        agent,
        optimizer,
    ):
        args = self.args
        valid_met.update({
            "pg_loss": pg_loss + valid_met["pg_loss"],
            "v_loss": v_loss + valid_met["v_loss"],
            "entropy": entropy_mean + valid_met["entropy"],
            "approx_kl": approx_kl + valid_met["approx_kl"],
            "mb_advantages": mb_advantages + valid_met["mb_advantages"],
            "v_loss_t": (v_loss_t + valid_met["v_loss_t"]) if args.use_timeawareness else None,
            "v_loss_c": (v_loss_c + valid_met["v_loss_c"]) if args.use_cost else None,
            "cost_loss": (L_viol + valid_met["cost_loss"]) if args.use_cost else None,
            "bound_loss": bd_loss + valid_met["bound_loss"],
            "policy_grad_norm_pre_noise": policy_grad_norm_pre_noise + valid_met["policy_grad_norm_pre_noise"],
            "policy_grad_norm_post_noise": policy_grad_norm_post_noise + valid_met["policy_grad_norm_post_noise"],
            "policy_grad_norm_post_clip": policy_grad_norm_post_clip + valid_met["policy_grad_norm_post_clip"],
            **{
                key: value + valid_met[key]
                for key, value in policy_grad_noise_diagnostics.items()
            },
        })
        return deepcopy(agent.state_dict()), deepcopy(optimizer.state_dict())

    def _clip_gradients(self, agent):
        # Preserve historical joint clipping unless the ablation is explicit.
        scope = getattr(self.args, "gradient_clip_scope", "joint")
        if scope == "joint":
            nn.utils.clip_grad_norm_(self._ppo_parameters(agent), self.args.max_grad_norm)
        elif scope == "actor_critic":
            from core.training.algo.gradient_clipping import clip_actor_critic_gradients

            clip_actor_critic_gradients(agent, self.args.max_grad_norm)
        else:
            raise ValueError(f"Unknown gradient clipping scope: {scope}")

    def _grad_norm(self, parameters):
        grads = [p.grad.detach().norm(2) for p in parameters if p.grad is not None]
        if not grads:
            return torch.tensor(0.0, device=self.device)
        return torch.linalg.vector_norm(torch.stack(grads), ord=2)

    def _policy_grad_noise_mode(self):
        return str(getattr(self.args, "policy_grad_noise_mode", "additive")).lower()

    @staticmethod
    def _capture_parameter_grads(parameters):
        return [
            (param, param.grad.detach().clone())
            for param in parameters
            if param.grad is not None
        ]

    def _captured_grad_norm(self, captured):
        if not captured:
            return torch.tensor(0.0, device=self.device)
        return torch.sqrt(sum(grad.pow(2).sum() for _, grad in captured))

    def _captured_grad_cosine(self, left, right):
        right_by_param = {id(param): grad for param, grad in right}
        pairs = [
            (left_grad, right_by_param[id(param)])
            for param, left_grad in left
            if id(param) in right_by_param
        ]
        if not pairs:
            return torch.tensor(0.0, device=self.device)
        dot = sum(left_grad.mul(right_grad).sum() for left_grad, right_grad in pairs)
        left_norm = torch.sqrt(sum(left_grad.pow(2).sum() for left_grad, _ in pairs))
        right_norm = torch.sqrt(sum(right_grad.pow(2).sum() for _, right_grad in pairs))
        denominator = left_norm * right_norm
        if not bool(torch.isfinite(denominator).item()) or float(denominator.item()) <= 0.0:
            return torch.tensor(0.0, device=self.device)
        return dot / denominator

    def _norm_preserved_corrupted_grads(self, clean_grads, noise):
        """Corrupt PPO direction while preserving its global actor-gradient norm."""
        noise_by_param = {id(param): delta.detach() for param, delta in noise}
        raw_grads = []
        noise_grads = []
        for param, clean_grad in clean_grads:
            delta = noise_by_param.get(id(param))
            if delta is None:
                delta = torch.zeros_like(clean_grad)
            raw_grads.append((param, clean_grad + delta))
            noise_grads.append((param, delta))

        clean_norm = self._captured_grad_norm(clean_grads)
        raw_norm = self._captured_grad_norm(raw_grads)
        if (
            bool(torch.isfinite(clean_norm).item())
            and bool(torch.isfinite(raw_norm).item())
            and float(clean_norm.item()) > 0.0
            and float(raw_norm.item()) > 0.0
        ):
            factor = clean_norm / raw_norm
            corrupted = [(param, grad * factor) for param, grad in raw_grads]
        else:
            corrupted = [(param, grad.clone()) for param, grad in clean_grads]

        diagnostics = self._zero_policy_grad_noise_diagnostics()
        diagnostics.update({
            "policy_grad_noise_norm": self._captured_grad_norm(noise_grads),
            "policy_grad_norm_raw_noise": raw_norm,
            "policy_grad_norm_restored_noise": self._captured_grad_norm(corrupted),
            "policy_grad_clean_corrupted_cosine": self._captured_grad_cosine(
                clean_grads, corrupted,
            ),
        })
        return corrupted, diagnostics

    def _replace_policy_grad_component(self, clean_grads, corrupted_grads):
        """Replace clean PPO actor gradients inside the already-backpropagated full loss."""
        corrupted_by_param = {id(param): grad for param, grad in corrupted_grads}
        residual_grads = []
        final_grads = []
        clean_present = []
        corrupted_present = []
        for param, clean_grad in clean_grads:
            if param.grad is None:
                raise RuntimeError(
                    "Full backward omitted a parameter present in the PPO actor gradient"
                )
            corrupted_grad = corrupted_by_param[id(param)]
            residual_grad = param.grad.detach().clone() - clean_grad
            param.grad.add_(corrupted_grad - clean_grad)
            residual_grads.append((param, residual_grad))
            final_grads.append((param, param.grad.detach().clone()))
            clean_present.append((param, clean_grad))
            corrupted_present.append((param, corrupted_grad))

        return {
            "policy_grad_nonppo_norm": self._captured_grad_norm(residual_grads),
            "policy_grad_final_pre_clip_norm": self._captured_grad_norm(final_grads),
            "policy_grad_corrupted_nonppo_cosine": self._captured_grad_cosine(
                corrupted_present, residual_grads,
            ),
            "policy_grad_final_nonppo_cosine": self._captured_grad_cosine(
                final_grads, residual_grads,
            ),
            "policy_grad_final_clean_ppo_cosine": self._captured_grad_cosine(
                final_grads, clean_present,
            ),
        }

    def _policy_parameters(self, agent):
        if hasattr(agent, "named_policy_parameters"):
            return [param for _, param in agent.named_policy_parameters()]
        return [param for param in agent.parameters() if param.requires_grad]

    def _pack_grads(self, params, grads):
        packed = []
        for param, grad in zip(params, grads):
            if grad is None:
                packed.append(torch.zeros(param.numel(), device=self.device, dtype=self.dtype))
            else:
                packed.append(grad.reshape(-1).to(device=self.device, dtype=self.dtype))
        return torch.cat(packed) if packed else torch.tensor([], device=self.device, dtype=self.dtype)

    def _assign_policy_grad_vector(self, params, grad_vector, mix):
        pointer = 0
        for param in params:
            numel = param.numel()
            grad = grad_vector[pointer:pointer + numel].view_as(param)
            pointer += numel
            if param.grad is None:
                param.grad = grad.clone()
            elif mix >= 1.0:
                param.grad.copy_(grad)
            elif mix > 0.0:
                param.grad.mul_(1.0 - mix).add_(grad, alpha=mix)

    def _ppo_parameters(self, agent):
        if hasattr(agent, "ppo_parameters"):
            return list(agent.ppo_parameters())
        return [param for param in agent.parameters() if param.requires_grad]

    def _sample_policy_grad_noise(self, agent):
        scale = float(
            getattr(
                self.args,
                "effective_policy_grad_noise_scale",
                getattr(self.args, "policy_grad_noise_scale", 0.0),
            )
        )
        if scale <= 0.0:
            return []

        noise = []
        for param in self._policy_parameters(agent):
            grad = param.grad
            if grad is None:
                continue
            grad_rms = grad.detach().pow(2).mean().sqrt()
            if bool(torch.isfinite(grad_rms).item()) and float(grad_rms.item()) > 0.0:
                noise.append((param, torch.randn_like(grad) * grad_rms * scale))
        return noise

    def _add_grad_noise(self, noise):
        for param, delta in noise:
            if param.grad is None:
                param.grad = delta.clone()
            else:
                param.grad.add_(delta)

    def _grad_norm_with_added_noise(self, parameters, noise):
        noise_by_param = {id(param): delta for param, delta in noise}
        grads = []
        for param in parameters:
            grad = param.grad
            delta = noise_by_param.get(id(param))
            if grad is None and delta is None:
                continue
            if grad is None:
                value = delta
            elif delta is None:
                value = grad
            else:
                value = grad + delta
            grads.append(value.detach().norm(2))
        if not grads:
            return torch.tensor(0.0, device=self.device)
        return torch.linalg.vector_norm(torch.stack(grads), ord=2)

    def _update_learning_rate(self, optimizer, lr_scheduler, approx_kl, global_step=0):
        if not self.args.anneal_lr:
            return
        if self.args.scheduler == "adapt":
            new_lr = lr_scheduler.update(optimizer.param_groups[0]["lr"], approx_kl)
        else:
            new_lr, _ = lr_scheduler.update(global_step)
        optimizer.param_groups[0]["lr"] = new_lr
