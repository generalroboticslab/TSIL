"""TSIL / BC replay loss from archived trajectory memory."""

import json
import os

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from core.training.algo.utils import compute_value_loss


class TsilReplayLoss:
    """Computes SIL / BC auxiliary losses from TSIL trajectory memory samples."""

    def __init__(self, args, device, dtype):
        self.args = args
        self.device = device
        self.dtype = dtype

    @staticmethod
    def _base_stats(replay_mode="disabled"):
        return {
            "replay_mode": replay_mode,
            "replay_active": 0,
            "replay_has_demo": 0,
            "replay_steps": 0,
            "replay_policy_loss": 0.0,
            "replay_value_loss": 0.0,
            "replay_total_loss": 0.0,
            "replay_positive_frac": 0.0,
            "replay_positive_available_frac": 0.0,
            "replay_sampled_positive_frac": 0.0,
            "replay_dataset_steps": 0,
            "replay_dataset_trajectories": 0,
            "replay_unique_trajectories": 0,
            "replay_fastest_tasks": 0,
            "replay_return_fallback_tasks": 0,
            "replay_success_pool_trajectories": 0,
            "replay_fallback_pool_trajectories": 0,
            "replay_sample_success_frac": 0.0,
            "replay_sample_fallback_frac": 0.0,
            "replay_positive_success_frac": 0.0,
            "replay_positive_fallback_frac": 0.0,
            "replay_gap_success_mean": 0.0,
            "replay_gap_fallback_mean": 0.0,
            "replay_policy_loss_success": 0.0,
            "replay_policy_loss_fallback": 0.0,
            "replay_delayed": 0,
            "replay_start_iter": 0,
            "replay_memory_logp": 0.0,
            "replay_memory_weighted_logp": 0.0,
            "replay_memory_positive_gap_mean": 0.0,
            "sil_cos_ppo_memory": 0.0,
            "sil_cos_sil_memory": 0.0,
            "sil_cos_joint_memory": 0.0,
            "sil_alignment_gain": 0.0,
            "sil_landscape_file": "",
            "sil_revisit_reference_count": 0,
            "sil_revisit_logp_topk_mean": 0.0,
            "sil_revisit_nll_topk_mean": 0.0,
            "sil_revisit_nll_topk_std": 0.0,
            "sil_revisit_reference_count_fastest1": 0,
            "sil_revisit_logp_fastest1_mean": 0.0,
            "sil_revisit_nll_fastest1_mean": 0.0,
            "sil_revisit_nll_fastest1_std": 0.0,
            "sil_supervised_logp_topk_mean": 0.0,
            "sil_supervised_nll_topk_mean": 0.0,
            "sil_supervised_weight_frac": 0.0,
            "sil_supervised_logp_fastest1_mean": 0.0,
            "sil_supervised_nll_fastest1_mean": 0.0,
            "sil_supervised_weight_fastest1_frac": 0.0,
            "sil_archive_best_eps_time": 0.0,
            "sil_archive_best_steps": 0,
        }

    def default_stats(self):
        replay_mode = self.args.sil_mode if self.args.use_sil else "disabled"
        return self._base_stats(replay_mode)

    @staticmethod
    def init_iteration_stats(replay_context):
        if replay_context is None:
            return TsilReplayLoss._base_stats()

        stats = TsilReplayLoss._base_stats()
        stats.update(replay_context.get("stats", {}))
        return stats

    @staticmethod
    def init_total_stats():
        return {
            "replay_policy_loss": 0.0,
            "replay_value_loss": 0.0,
            "replay_total_loss": 0.0,
            "replay_positive_frac": 0.0,
            "replay_positive_available_frac": 0.0,
            "replay_sampled_positive_frac": 0.0,
            "replay_memory_logp": 0.0,
            "replay_memory_weighted_logp": 0.0,
            "replay_memory_positive_gap_mean": 0.0,
            "replay_sample_success_frac": 0.0,
            "replay_sample_fallback_frac": 0.0,
            "replay_positive_success_frac": 0.0,
            "replay_positive_fallback_frac": 0.0,
            "replay_gap_success_mean": 0.0,
            "replay_gap_fallback_mean": 0.0,
            "replay_policy_loss_success": 0.0,
            "replay_policy_loss_fallback": 0.0,
            "replay_steps": 0,
            "replay_updates": 0,
        }

    @staticmethod
    def accumulate_total_stats(
        replay_totals,
        replay_policy_loss,
        replay_value_loss,
        replay_total_loss,
        replay_positive_frac,
        replay_sampled_positive_frac,
        replay_step_count,
        replay_batch_stats=None,
    ):
        replay_totals["replay_policy_loss"] += float(replay_policy_loss.detach().item())
        replay_totals["replay_value_loss"] += float(replay_value_loss.detach().item())
        replay_totals["replay_total_loss"] += float(replay_total_loss.detach().item())
        replay_totals["replay_positive_available_frac"] += float(replay_positive_frac)
        replay_totals["replay_positive_frac"] += float(replay_positive_frac)
        replay_totals["replay_sampled_positive_frac"] += float(replay_sampled_positive_frac)
        replay_batch_stats = replay_batch_stats or {}
        replay_totals["replay_memory_logp"] += float(replay_batch_stats.get("memory_logp", 0.0))
        replay_totals["replay_memory_weighted_logp"] += float(replay_batch_stats.get("memory_weighted_logp", 0.0))
        replay_totals["replay_memory_positive_gap_mean"] += float(replay_batch_stats.get("memory_positive_gap_mean", 0.0))
        for key in (
            "replay_sample_success_frac",
            "replay_sample_fallback_frac",
            "replay_positive_success_frac",
            "replay_positive_fallback_frac",
            "replay_gap_success_mean",
            "replay_gap_fallback_mean",
            "replay_policy_loss_success",
            "replay_policy_loss_fallback",
        ):
            replay_totals[key] += float(replay_batch_stats.get(key, 0.0))
        replay_totals["replay_steps"] += int(replay_step_count)
        replay_totals["replay_updates"] += 1

    @staticmethod
    def finalize_iteration_stats(base_replay_stats, replay_totals, num_agent_updates):
        replay_stats = dict(base_replay_stats)
        replay_updates = replay_totals["replay_updates"]
        if not replay_stats["replay_has_demo"] or replay_updates == 0:
            return replay_stats

        divisor = max(int(replay_updates), int(num_agent_updates), 1)
        replay_stats.update({
            "replay_active": 1,
            "replay_steps": replay_totals["replay_steps"],
            "replay_policy_loss": replay_totals["replay_policy_loss"] / divisor,
            "replay_value_loss": replay_totals["replay_value_loss"] / divisor,
            "replay_total_loss": replay_totals["replay_total_loss"] / divisor,
            "replay_positive_frac": replay_totals["replay_positive_frac"] / divisor,
            "replay_positive_available_frac": replay_totals["replay_positive_available_frac"] / divisor,
            "replay_sampled_positive_frac": replay_totals["replay_sampled_positive_frac"] / divisor,
            "replay_memory_logp": replay_totals["replay_memory_logp"] / divisor,
            "replay_memory_weighted_logp": replay_totals["replay_memory_weighted_logp"] / divisor,
            "replay_memory_positive_gap_mean": replay_totals["replay_memory_positive_gap_mean"] / divisor,
            "replay_sample_success_frac": replay_totals["replay_sample_success_frac"] / divisor,
            "replay_sample_fallback_frac": replay_totals["replay_sample_fallback_frac"] / divisor,
            "replay_positive_success_frac": replay_totals["replay_positive_success_frac"] / divisor,
            "replay_positive_fallback_frac": replay_totals["replay_positive_fallback_frac"] / divisor,
            "replay_gap_success_mean": replay_totals["replay_gap_success_mean"] / divisor,
            "replay_gap_fallback_mean": replay_totals["replay_gap_fallback_mean"] / divisor,
            "replay_policy_loss_success": replay_totals["replay_policy_loss_success"] / divisor,
            "replay_policy_loss_fallback": replay_totals["replay_policy_loss_fallback"] / divisor,
        })
        return replay_stats

    def reset_epoch_sampler(self, replay_context, num_ppo_minibatches, agent=None):
        if replay_context is not None and replay_context.get("has_tsil_memory"):
            replay_context["_sil_batch_size"] = max(int(getattr(self.args, "sil_batch_size", 1)), 1)
            replay_context["_sil_num_minibatches"] = max(int(num_ppo_minibatches), 1)

    def prepare_iteration(
        self,
        tsil_memory,
        unique_task_ids,
        reward_normalizer=None,
        global_update_iter=0,
        schedule_context=None,
    ):
        stats = self.default_stats()
        if not self.args.use_sil:
            return {"demos": [], "dataset": None, "stats": stats}

        sil_start_iter = int(getattr(self.args, "sil_start_iter", 0))
        stats["replay_start_iter"] = sil_start_iter
        if int(global_update_iter) < sil_start_iter:
            stats["replay_delayed"] = 1
            return {"demos": [], "dataset": None, "stats": stats}

        if (
            tsil_memory is not None
            and hasattr(tsil_memory, "has_sil_demo")
            and tsil_memory.has_sil_demo()
        ):
            return tsil_memory.prepare_sil_context(
                reward_normalizer,
                schedule_context,
                stats,
            )

        return {"demos": [], "dataset": None, "stats": stats}

    def _positive_gap_for_batch(self, agent, state_batch, return_batch, task_id_batch):
        value, _, _ = agent.get_value(state_batch, denorm=False, update=False)
        if self.args.norm_value:
            normalize_task_ids = task_id_batch if self.args.pertask_norm else None
            return_batch = agent.normalize_value(
                return_batch,
                denorm=False,
                update=False,
                task_ids=normalize_task_ids,
            )
        return torch.clamp(return_batch - value.view(-1), min=0.0)

    def _compute_batch_losses(
        self,
        agent,
        obs_batch,
        state_batch,
        action_batch,
        return_batch,
        task_id_batch,
        sil_weight_batch=None,
        sil_success_mask=None,
        loss_scale=1.0,
        return_stats=False,
    ):
        logprob = agent.get_logprob(obs_batch, action_batch, update=False)
        if self.args.sil_mode == "bc":
            weights = None
            if (
                str(getattr(self.args, "sil_sample_unit", "transition")).lower() == "trajectory"
                and sil_weight_batch is not None
            ):
                weights = sil_weight_batch.to(device=logprob.device, dtype=logprob.dtype)
            loss = -logprob.mean() if weights is None else -(weights * logprob).mean()
            weighted_logp = float(logprob.detach().mean().item())
            if weights is not None:
                weight_sum = float(weights.detach().sum().item())
                if weight_sum > 0:
                    weighted_logp = float((weights.detach() * logprob.detach()).sum().item() / weight_sum)
            sample_success_frac = (
                float(sil_success_mask.to(dtype=torch.float32).mean().item())
                if sil_success_mask is not None and int(sil_success_mask.numel()) > 0 else 0.0
            )
            stats = {
                "memory_logp": float(logprob.detach().mean().item()),
                "memory_weighted_logp": weighted_logp,
                "memory_positive_gap_mean": 0.0,
                "replay_sample_success_frac": sample_success_frac,
                "replay_sample_fallback_frac": 1.0 - sample_success_frac,
            }
            result = (loss, torch.tensor(0.0, device=self.device), 1.0)
            return (*result, stats) if return_stats else result

        positive_gap = self._positive_gap_for_batch(agent, state_batch, return_batch, task_id_batch)
        positive_mask = positive_gap > 0
        weights = positive_gap.detach()
        if sil_weight_batch is not None:
            weights = weights * sil_weight_batch.to(device=weights.device, dtype=weights.dtype)
        if self.args.sil_normalize_gap and positive_mask.any():
            active_weights = weights[weights > 0]
            if active_weights.numel() > 0:
                pos_std = active_weights.std(unbiased=False)
                normalized = torch.clamp(weights / (pos_std + 1e-8), max=1.0)
                weights = torch.where(weights > 0, normalized, weights)

        policy_loss = float(loss_scale) * (-(weights * logprob).mean())
        value_loss = float(loss_scale) * (0.5 * (positive_gap ** 2).mean())
        positive_frac = float(positive_mask.float().mean().item())
        weight_sum = float(weights.detach().sum().item())
        weighted_logp = (
            float((weights.detach() * logprob.detach()).sum().item() / weight_sum)
            if weight_sum > 0 else float(logprob.detach().mean().item())
        )
        stats = {
            "memory_logp": float(logprob.detach().mean().item()),
            "memory_weighted_logp": weighted_logp,
            "memory_positive_gap_mean": float(positive_gap.detach().mean().item()),
        }
        if sil_success_mask is not None and int(sil_success_mask.numel()) > 0:
            success_mask = sil_success_mask.to(device=positive_gap.device, dtype=torch.bool)
            fallback_mask = ~success_mask
            policy_terms = -(weights.detach() * logprob.detach())

            def masked_mean(values, mask):
                return float(values[mask].mean().item()) if bool(mask.any().item()) else 0.0

            stats.update({
                "replay_sample_success_frac": float(success_mask.to(dtype=torch.float32).mean().item()),
                "replay_sample_fallback_frac": float(fallback_mask.to(dtype=torch.float32).mean().item()),
                "replay_positive_success_frac": masked_mean(positive_mask.to(dtype=torch.float32), success_mask),
                "replay_positive_fallback_frac": masked_mean(positive_mask.to(dtype=torch.float32), fallback_mask),
                "replay_gap_success_mean": masked_mean(positive_gap.detach(), success_mask),
                "replay_gap_fallback_mean": masked_mean(positive_gap.detach(), fallback_mask),
                "replay_policy_loss_success": masked_mean(policy_terms, success_mask),
                "replay_policy_loss_fallback": masked_mean(policy_terms, fallback_mask),
            })
        result = (policy_loss, value_loss, positive_frac)
        return (*result, stats) if return_stats else result

    def compute_joint_losses(self, agent, replay_context):
        if (
            not self.args.use_sil
            or not bool(getattr(self.args, "sil_train", True))
            or replay_context is None
        ):
            if replay_context is not None:
                replay_context["_last_sil_batch_stats"] = {}
            return (
                torch.tensor(0.0, device=self.device),
                torch.tensor(0.0, device=self.device),
                0.0,
                0.0,
                0,
            )

        sample_batch_fn = replay_context.get("sample_batch_fn")
        if replay_context.get("has_tsil_memory") and sample_batch_fn is not None:
            replay_context["_sil_value_agent"] = agent
            batch = sample_batch_fn(replay_context)
            if batch is None or int(batch["action"].shape[0]) == 0:
                replay_context["_last_sil_batch_stats"] = {}
                return (
                    torch.tensor(0.0, device=self.device),
                    torch.tensor(0.0, device=self.device),
                    0.0,
                    0.0,
                    0,
                )

            policy_loss, value_loss, positive_frac, batch_stats = self._compute_batch_losses(
                agent,
                batch["obs"],
                batch["state"],
                batch["action"],
                batch["returns"],
                batch["task_ids"],
                sil_weight_batch=batch.get("sil_weights"),
                sil_success_mask=batch.get("sil_success_mask"),
                return_stats=True,
            )
            replay_context["_last_sil_batch_stats"] = batch_stats
            return (
                policy_loss,
                value_loss,
                positive_frac,
                positive_frac,
                int(batch["action"].shape[0]),
            )

        replay_context["_last_sil_batch_stats"] = {}
        return (
            torch.tensor(0.0, device=self.device),
            torch.tensor(0.0, device=self.device),
            0.0,
            0.0,
            0,
        )

    def update_diagnostics(self, agent, replay_context, rollout_batch, updater, global_update_iter):
        if not self.args.use_sil or replay_context is None or not replay_context.get("has_tsil_memory"):
            return

        stats = replay_context.setdefault("stats", self.default_stats())
        analysis_batch_size = max(int(getattr(self.args, "sil_analysis_batch_size", 1024)), 1)
        ref_batch = self._fastest_reference_batch(replay_context, max_total_steps=analysis_batch_size)
        if ref_batch is None:
            return

        logp_stats = self._reference_logp_stats(agent, ref_batch)
        stats.update(logp_stats)
        fastest1_batch = self._fastest_reference_batch(
            replay_context,
            max_total_steps=analysis_batch_size,
            best_per_env_only=True,
        )
        if fastest1_batch is not None:
            stats.update(self._reference_logp_stats(agent, fastest1_batch, key_suffix="fastest1"))

        interval = int(getattr(self.args, "sil_analysis_interval", 0))
        if interval <= 0 or int(global_update_iter) <= 0 or int(global_update_iter) % interval != 0:
            return

        analysis_batch = self._fastest_reference_batch(
            replay_context,
            max_total_steps=analysis_batch_size,
        )
        if analysis_batch is None:
            return
        self._update_direction_diagnostics(
            agent,
            replay_context,
            rollout_batch,
            updater,
            int(global_update_iter),
            analysis_batch,
            stats,
        )

    def _fastest_reference_batch(self, replay_context, max_total_steps=None, best_per_env_only=False):
        memory = replay_context.get("tsil_memory")
        if memory is None or not getattr(memory, "_archive_allocated", False):
            return None

        fastest_idx = memory._archive_fastest_idx
        if best_per_env_only:
            fastest_idx = fastest_idx[:, :1]
        valid = fastest_idx >= 0
        if not bool(valid.any().item()):
            return None

        env_grid = torch.arange(memory._archive_num_envs, device=self.device).unsqueeze(1).expand_as(fastest_idx)
        env_ids = env_grid[valid]
        slot_ids = fastest_idx[valid]
        lengths = memory._archive_length[env_ids, slot_ids]
        positive = lengths > 0
        if not bool(positive.any().item()):
            return None

        env_ids = env_ids[positive]
        slot_ids = slot_ids[positive]
        lengths = lengths[positive]
        step_budget = None if max_total_steps is None else max(int(max_total_steps), 1)
        if step_budget is not None and int(lengths.numel()) > step_budget:
            select_idx = torch.linspace(0, int(lengths.numel()) - 1, step_budget, device=self.device).round().to(torch.long)
            env_ids = env_ids[select_idx]
            slot_ids = slot_ids[select_idx]
            lengths = lengths[select_idx]
        ref_count = int(lengths.numel())
        per_traj_limit = None
        if step_budget is not None:
            per_traj_limit = max(step_budget // max(ref_count, 1), 1)

        schedule_context = replay_context.get("schedule_context")
        reward_normalizer = replay_context.get("reward_normalizer")
        obs_parts, state_parts, action_parts, return_parts, task_parts, traj_parts = [], [], [], [], [], []
        for traj_idx, (env_id, slot_id, length) in enumerate(zip(env_ids.tolist(), slot_ids.tolist(), lengths.tolist())):
            length = int(length)
            if per_traj_limit is not None and length > per_traj_limit:
                step_idx = torch.linspace(0, length - 1, per_traj_limit, device=self.device).round().to(torch.long)
            else:
                step_idx = torch.arange(length, device=self.device, dtype=torch.long)
            task_id = int(memory._archive_task_id[env_id, slot_id].item())
            env_tensor = torch.tensor([env_id], dtype=torch.long, device=self.device)
            slot_tensor = torch.tensor([slot_id], dtype=torch.long, device=self.device)
            task_tensor = torch.tensor([task_id], dtype=torch.long, device=self.device)
            length_tensor = torch.tensor([length], dtype=torch.long, device=self.device)

            obs = memory._archive_obs[env_id, slot_id, step_idx].to(dtype=self.dtype).clone()
            state = memory._archive_state[env_id, slot_id, step_idx].to(dtype=self.dtype).clone()
            if schedule_context is not None:
                deadline = memory._deadline_for_batch(env_tensor, task_tensor, schedule_context)[0]
                # Public time obs is [time_ratio, time_cur, ddl]. Replaying
                # under a new deadline leaves elapsed time unchanged.
                if obs.ndim >= 2 and obs.shape[-1] > 2:
                    obs[:, 2] = deadline.to(dtype=obs.dtype)
                if state.ndim >= 2 and state.shape[-1] > 2:
                    state[:, 2] = deadline.to(dtype=state.dtype)

            reward_seq = memory._archive_base_reward[env_id, slot_id].clone().unsqueeze(0)
            if schedule_context is not None and float(schedule_context.get("r_epstime_scale", 0.0)) > 0.0:
                reward_seq[0, length - 1] += memory._timeaware_bonus_for_batch(
                    env_tensor,
                    slot_tensor,
                    task_tensor,
                    schedule_context,
                )[0]
                reward_seq = torch.clamp(reward_seq, min=0.0)
            if self.args.norm_rew and reward_normalizer is not None:
                reward_seq = reward_normalizer(
                    reward_seq.reshape(-1),
                    task_tensor.expand(memory._archive_max_len),
                ).view(1, memory._archive_max_len).to(dtype=torch.float32, device=self.device)
            returns = memory._discounted_returns_from_rewards(reward_seq, length_tensor)[0, step_idx]

            obs_parts.append(obs)
            state_parts.append(state)
            action_parts.append(memory._archive_action[env_id, slot_id, step_idx].to(dtype=self.dtype))
            return_parts.append(returns.to(dtype=self.dtype))
            task_parts.append(torch.full((int(step_idx.numel()),), task_id, dtype=torch.long, device=self.device))
            traj_parts.append(torch.full((int(step_idx.numel()),), traj_idx, dtype=torch.long, device=self.device))

        if not obs_parts:
            return None

        summary = memory.fastest_reference_summary()
        return {
            "obs": torch.cat(obs_parts, dim=0),
            "state": torch.cat(state_parts, dim=0),
            "action": torch.cat(action_parts, dim=0),
            "returns": torch.cat(return_parts, dim=0),
            "task_ids": torch.cat(task_parts, dim=0),
            "traj_ids": torch.cat(traj_parts, dim=0),
            "ref_count": ref_count,
            "best_eps_time": summary.get("best_eps_time"),
            "best_steps": summary.get("best_steps"),
        }

    def _reference_objective(self, agent, ref_batch):
        logprob = agent.get_logprob(ref_batch["obs"], ref_batch["action"], update=False).view(-1)
        traj_ids = ref_batch["traj_ids"]
        ref_count = int(ref_batch["ref_count"])
        sums = torch.zeros((ref_count,), dtype=logprob.dtype, device=logprob.device).scatter_add(0, traj_ids, logprob)
        counts = torch.zeros((ref_count,), dtype=logprob.dtype, device=logprob.device).scatter_add(
            0,
            traj_ids,
            torch.ones_like(logprob),
        )
        return (sums / torch.clamp(counts, min=1.0)).mean()

    def _reference_logp_stats(self, agent, ref_batch, key_suffix="topk"):
        batch_size = max(int(getattr(self.args, "sil_analysis_batch_size", 1024)), 1)
        logps = []
        with torch.no_grad():
            for start in range(0, int(ref_batch["obs"].shape[0]), batch_size):
                end = start + batch_size
                logps.append(
                    agent.get_logprob(
                        ref_batch["obs"][start:end],
                        ref_batch["action"][start:end],
                        update=False,
                    ).detach().float().view(-1)
                )
        logprob = torch.cat(logps, dim=0)
        traj_ids = ref_batch["traj_ids"]
        ref_count = int(ref_batch["ref_count"])
        sums = torch.zeros((ref_count,), dtype=logprob.dtype, device=logprob.device).scatter_add(0, traj_ids, logprob)
        counts = torch.zeros((ref_count,), dtype=logprob.dtype, device=logprob.device).scatter_add(
            0,
            traj_ids,
            torch.ones_like(logprob),
        )
        traj_logp = sums / torch.clamp(counts, min=1.0)
        traj_nll = -traj_logp
        supervised_logp = torch.tensor(0.0, dtype=logprob.dtype, device=logprob.device)
        supervised_weight_frac = 0.0
        if {"state", "returns", "task_ids"}.issubset(ref_batch):
            with torch.no_grad():
                value, _, _ = agent.get_value(ref_batch["state"], denorm=False, update=False)
                returns = ref_batch["returns"]
                if self.args.norm_value:
                    normalize_task_ids = ref_batch["task_ids"] if self.args.pertask_norm else None
                    returns = agent.normalize_value(
                        returns,
                        denorm=False,
                        update=False,
                        task_ids=normalize_task_ids,
                    )
                weights = torch.clamp(returns.to(dtype=logprob.dtype) - value.detach().float().view(-1), min=0.0)
                weight_sum = weights.sum()
                if float(weight_sum.item()) > 0.0:
                    supervised_logp = (weights * logprob).sum() / weight_sum
                    supervised_weight_frac = float((weights > 0).float().mean().item())
        best_eps_time = ref_batch.get("best_eps_time")
        best_steps = ref_batch.get("best_steps")
        if key_suffix == "topk":
            return {
                "sil_revisit_reference_count": ref_count,
                "sil_revisit_logp_topk_mean": float(traj_logp.mean().item()),
                "sil_revisit_nll_topk_mean": float(traj_nll.mean().item()),
                "sil_revisit_nll_topk_std": float(traj_nll.std(unbiased=False).item()) if ref_count > 1 else 0.0,
                "sil_supervised_logp_topk_mean": float(supervised_logp.item()),
                "sil_supervised_nll_topk_mean": float((-supervised_logp).item()),
                "sil_supervised_weight_frac": supervised_weight_frac,
                "sil_archive_best_eps_time": 0.0 if best_eps_time is None else float(best_eps_time),
                "sil_archive_best_steps": 0 if best_steps is None else int(best_steps),
            }
        return {
            f"sil_revisit_reference_count_{key_suffix}": ref_count,
            f"sil_revisit_logp_{key_suffix}_mean": float(traj_logp.mean().item()),
            f"sil_revisit_nll_{key_suffix}_mean": float(traj_nll.mean().item()),
            f"sil_revisit_nll_{key_suffix}_std": float(traj_nll.std(unbiased=False).item()) if ref_count > 1 else 0.0,
            f"sil_supervised_logp_{key_suffix}_mean": float(supervised_logp.item()),
            f"sil_supervised_nll_{key_suffix}_mean": float((-supervised_logp).item()),
            f"sil_supervised_weight_{key_suffix}_frac": supervised_weight_frac,
        }

    @staticmethod
    def _trainable_params(agent):
        return [param for param in agent.parameters() if param.requires_grad]

    @staticmethod
    def _grad_vector(objective, params):
        grads = torch.autograd.grad(objective, params, allow_unused=True, retain_graph=False)
        flat = [
            (torch.zeros_like(param) if grad is None else grad).reshape(-1)
            for param, grad in zip(params, grads)
        ]
        return torch.cat(flat) if flat else None

    @staticmethod
    def _cosine(a, b):
        denom = torch.linalg.norm(a) * torch.linalg.norm(b)
        if float(denom.detach().item()) <= 0.0:
            return 0.0
        return float((torch.dot(a, b) / denom).detach().item())

    @staticmethod
    def _unit_vector(vec):
        norm = torch.linalg.norm(vec)
        if float(norm.detach().item()) <= 0.0:
            return None
        return vec / norm

    def _policy_update_direction(self, agent, rollout_batch, updater):
        params = self._trainable_params(agent)
        if not params:
            return None
        n_items = int(rollout_batch["actions"].shape[0])
        if n_items <= 0:
            return None
        batch_size = min(max(int(getattr(self.args, "sil_analysis_batch_size", 1024)), 1), n_items)
        indices = torch.arange(batch_size, dtype=torch.long, device=rollout_batch["actions"].device)
        loss = updater.policy_loss_for_indices(agent, rollout_batch, indices)
        grad = self._grad_vector(loss, params)
        return None if grad is None else -grad.detach()

    def _memory_update_direction(self, agent, ref_batch):
        params = self._trainable_params(agent)
        if not params:
            return None
        objective = self._reference_objective(agent, ref_batch)
        grad = self._grad_vector(objective, params)
        return None if grad is None else grad.detach()

    def _update_direction_diagnostics(
        self,
        agent,
        replay_context,
        rollout_batch,
        updater,
        global_update_iter,
        ref_batch,
        stats,
    ):
        params = self._trainable_params(agent)
        if not params:
            return

        ppo_dir = self._policy_update_direction(agent, rollout_batch, updater)
        memory_dir = self._memory_update_direction(agent, ref_batch)
        if ppo_dir is None or memory_dir is None:
            return

        joint_dir = ppo_dir + float(getattr(self.args, "sil_coef", 0.0)) * memory_dir
        stats["sil_cos_ppo_memory"] = self._cosine(ppo_dir, memory_dir)
        stats["sil_cos_sil_memory"] = 1.0 if float(torch.linalg.norm(memory_dir).item()) > 0.0 else 0.0
        stats["sil_cos_joint_memory"] = self._cosine(joint_dir, memory_dir)
        stats["sil_alignment_gain"] = stats["sil_cos_joint_memory"] - stats["sil_cos_ppo_memory"]

        bx = self._unit_vector(ppo_dir)
        if bx is None:
            return
        by_raw = memory_dir - torch.dot(memory_dir, bx) * bx
        by = self._unit_vector(by_raw)
        if by is None:
            by = self._unit_vector(memory_dir)
        if by is None:
            return

        span = float(getattr(self.args, "sil_landscape_span", 0.05))
        grid = max(int(getattr(self.args, "sil_landscape_grid", 9)), 3)
        xs = torch.linspace(-span, span, grid, device=self.device)
        ys = torch.linspace(-span, span, grid, device=self.device)
        base_vector = parameters_to_vector(params).detach().clone()

        def project_arrow(vec):
            unit = self._unit_vector(vec)
            if unit is None:
                return [0.0, 0.0]
            arrow_len = 0.8 * span
            return [
                float((torch.dot(unit, bx) * arrow_len).detach().item()),
                float((torch.dot(unit, by) * arrow_len).detach().item()),
            ]

        surface = []
        try:
            with torch.no_grad():
                for y_val in ys:
                    row = []
                    for x_val in xs:
                        vector_to_parameters(base_vector + x_val * bx + y_val * by, params)
                        row.append(float(self._reference_objective(agent, ref_batch).detach().item()))
                    surface.append(row)
        finally:
            vector_to_parameters(base_vector, params)

        trajectory_dir = getattr(self.args, "trajectory_dir", None)
        if not trajectory_dir:
            return
        os.makedirs(trajectory_dir, exist_ok=True)
        path = os.path.join(trajectory_dir, f"sil_direction_landscape_iter{global_update_iter:06d}.json")
        with open(path, "w") as file_obj:
            json.dump(
                {
                    "iteration": int(global_update_iter),
                    "x": [float(v.item()) for v in xs],
                    "y": [float(v.item()) for v in ys],
                    "surface": surface,
                    "origin": [0.0, 0.0],
                    "ppo_update": project_arrow(ppo_dir),
                    "ppo_sil_update": project_arrow(joint_dir),
                    "xlabel": "PPO update axis",
                    "ylabel": "SIL-only axis orthogonal to PPO",
                    "zlabel": "Fast-success memory log-prob",
                    "title": f"SIL fast-success memory landscape @ iter {global_update_iter}",
                },
                file_obj,
            )
        stats["sil_landscape_file"] = path


__all__ = ["TsilReplayLoss"]
