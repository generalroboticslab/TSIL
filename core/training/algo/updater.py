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
from core.training.algo.ppo import compute_ppo_advantages, compute_ppo_losses


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

    # ------------------------------------------------------------------
    # Advantage / return computation
    # ------------------------------------------------------------------

    def compute_advantages(self, storage, agent, envs):
        """Compute GAE advantages and returns (PPO path)."""
        return compute_ppo_advantages(self.args, self.device, self.dtype, storage, agent, envs)

    def compute_grpo_returns(self, storage, agent, envs):
        """Compute reward-to-go returns for GRPO."""
        return compute_grpo_returns(self.args, self.device, self.dtype, storage, agent, envs)

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
            "task_indices": task_indices.repeat(args.num_steps),
        }

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
                batch["returns"] = agent.normalize_value(batch["returns"], update=True, task_ids=task_ids)
                batch["values"] = agent.normalize_value(batch["values"], task_ids=task_ids)
            if args.use_timeawareness:
                batch["returns_t"] = agent.normalize_value_t(batch["returns_t"], update=True, task_ids=task_ids)
                batch["values_t"] = agent.normalize_value_t(batch["values_t"], task_ids=task_ids)

        if args.pertask_norm_adv:
            for task_id in unique_task_ids:
                mask = batch["task_indices"] == task_id
                task_adv = batch["advantages"][mask]
                batch["advantages"][mask] = (task_adv - task_adv.mean()) / (task_adv.std() + 1e-8)
                if args.use_cost:
                    task_adv_c = batch["advantages_c"][mask]
                    task_adv_c_mu = task_adv_c.mean(dim=0)
                    task_adv_c_std = task_adv_c.std(dim=0)
                    batch["advantages_c"][mask] = (task_adv_c - task_adv_c_mu) / (task_adv_c_std + 1e-8)
                    batch["norm_return_c"][mask] = (1.0 - c_gamma) * batch["returns_c"][mask]
                    batch["norm_return_c"][mask] = (batch["norm_return_c"][mask] + task_adv_c_mu) / (task_adv_c_std + 1e-8)
        else:
            adv = batch["advantages"]
            batch["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-8)
            if args.use_cost:
                adv_c = batch["advantages_c"]
                adv_c_mu = adv_c.mean(dim=0)
                adv_c_std = adv_c.std(dim=0)
                batch["advantages_c"] = (adv_c - adv_c_mu) / (adv_c_std + 1e-8)
                batch["norm_return_c"] = (1.0 - c_gamma) * batch["returns_c"]
                batch["norm_return_c"] = (batch["norm_return_c"] + adv_c_mu) / (adv_c_std + 1e-8)

        # Stash constants used by the minibatch loop
        batch["_c_scale"] = c_scale

        return batch

    # ------------------------------------------------------------------
    # Minibatch update loop
    # ------------------------------------------------------------------

    def run_minibatch_updates(self, batch, agent, optimizer, initial_lstm_state,
                              lr_scheduler, cur_ent, global_update_iter, global_step=0,
                              replay_module=None, replay_context=None):
        """Run the PPO / GRPO minibatch update loop.

        Returns
        -------
        policy_diverged, valid_met, num_agent_updates,
        agent_params_store, optim_params_store, illed_met, replay_stats
        """
        args = self.args

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
        if use_replay:
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
        replay_batches_per_epoch = max(len(range(0, end_idx, step_num)) - 1, 1)
        replay_update_interval = max(int(getattr(args, "sil_update_interval", 1)), 1)
        replay_batches_per_epoch = max((replay_batches_per_epoch + replay_update_interval - 1) // replay_update_interval, 1)
        for _ in range(args.update_epochs):
            if args.use_lstm:
                np.random.shuffle(envinds)
            else:
                np.random.shuffle(b_inds)
            if use_replay:
                replay_helper.reset_epoch_sampler(replay_context, replay_batches_per_epoch, agent=agent)

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
                    _, mu, newlogprob, entropy, newvalue, newvalue_c, newvalue_t = agent.get_action_and_value(
                        batch["obs"][mb_inds],
                        batch["states"][mb_inds],
                        batch["actions"][mb_inds],
                        denorm=False,
                        update=update_norm,
                    )

                logratio = newlogprob - batch["logprobs"][mb_inds]
                ratio = logratio.exp()
                entropy_mean = entropy.mean()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                    if iter_idx > 0:
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

                run_replay_update = use_replay and (iter_idx % replay_update_interval == 0)
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

                pg_coef, ent_coef = float(not is_warmup), float(not is_warmup)
                loss = pg_coef * pg_loss + args.vf_coef * v_loss_all - ent_coef * cur_ent * entropy_mean
                replay_total_loss = float(args.sil_coef) * (
                    replay_policy_loss + float(args.sil_vf_coef) * replay_value_loss
                )
                loss += replay_total_loss

                bd_loss = 0.0
                if not args.beta:
                    bd_loss = args.bounds_loss_coef * bound_loss(mu, soft_bound=1.0)
                    loss += bd_loss

                policy_grad_noise = []
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
                    policy_grad_norm_pre_noise = self._grad_norm(self._policy_parameters(agent))
                    policy_grad_noise = self._sample_policy_grad_noise(agent)
                    policy_grad_norm_post_noise = self._grad_norm_with_added_noise(
                        self._policy_parameters(agent), policy_grad_noise,
                    )

                optimizer.zero_grad()
                loss.backward()
                if policy_grad_noise:
                    self._add_grad_noise(policy_grad_noise)
                elif not use_policy_grad_noise:
                    policy_grad_norm_pre_noise = self._grad_norm(self._policy_parameters(agent))
                    policy_grad_norm_post_noise = policy_grad_norm_pre_noise
                self._clip_gradients(agent)
                policy_grad_norm_post_clip = self._grad_norm(self._policy_parameters(agent))
                optimizer.step()

                self._update_learning_rate(optimizer, lr_scheduler, approx_kl, global_step)

            if policy_diverged:
                break

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

    def policy_loss_for_indices(self, agent, batch, indices):
        newlogprob = agent.get_logprob(batch["obs"][indices], batch["actions"][indices], update=False)
        ratio = (newlogprob - batch["logprobs"][indices]).exp()
        advantages = batch["advantages"][indices]
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
        return torch.max(pg_loss1, pg_loss2).mean()

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def _compute_ppo_losses(self, batch, mb_inds, ratio, newvalue, newvalue_c, newvalue_t, c_scale):
        mb_advantages = batch["advantages"][mb_inds]
        return compute_ppo_losses(
            self.args, batch, mb_inds, ratio, newvalue, newvalue_c, newvalue_t,
            c_scale, mb_advantages,
        )

    def _compute_grpo_losses(self, batch, mb_inds, ratio, newlogprob, newvalue_c, newvalue_t, c_scale):
        mb_advantages = batch["advantages"][mb_inds]
        return compute_grpo_losses(
            self.args, self.device, batch, mb_inds, ratio, newlogprob,
            newvalue_c, newvalue_t, c_scale, mb_advantages,
        )

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
        })
        return deepcopy(agent.state_dict()), deepcopy(optimizer.state_dict())

    def _clip_gradients(self, agent):
        nn.utils.clip_grad_norm_(agent.parameters(), self.args.max_grad_norm)

    def _grad_norm(self, parameters):
        grads = [p.grad.detach().norm(2) for p in parameters if p.grad is not None]
        if not grads:
            return torch.tensor(0.0, device=self.device)
        return torch.linalg.vector_norm(torch.stack(grads), ord=2)

    def _policy_parameters(self, agent):
        if hasattr(agent, "named_policy_parameters"):
            return [param for _, param in agent.named_policy_parameters()]
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
