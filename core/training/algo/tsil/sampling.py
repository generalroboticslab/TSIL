"""Private TSIL trajectory-memory sampling helpers."""

import numpy as np
import torch


class _TsilMemorySamplingMixin:
    """Sampling and replay-context methods for TsilTrajectoryMemory."""

    def _build_sil_source_indices(self):
        if not self._archive_allocated:
            empty = torch.empty((0, self._replay_topk()), dtype=torch.long, device=self.device)
            return empty, torch.empty((0,), dtype=torch.long, device=self.device)

        topk = self._replay_topk()
        if self._replay_source() == "return":
            source_idx = self._archive_return_idx.clone()
        else:
            source_idx = torch.full((self._archive_num_envs, topk), -1, dtype=torch.long, device=self.device)
            for env_id in range(self._archive_num_envs):
                write_idx = 0
                seen = set()
                for slot in self._archive_fastest_idx[env_id].tolist():
                    if slot < 0 or slot in seen:
                        continue
                    source_idx[env_id, write_idx] = int(slot)
                    seen.add(int(slot))
                    write_idx += 1
                    if write_idx >= topk:
                        break
                if write_idx >= topk:
                    continue
                for slot in self._archive_return_idx[env_id].tolist():
                    if slot < 0 or slot in seen:
                        continue
                    source_idx[env_id, write_idx] = int(slot)
                    seen.add(int(slot))
                    write_idx += 1
                    if write_idx >= topk:
                        break

        return source_idx, (source_idx >= 0).sum(dim=1)

    def _build_sil_pools(self):
        empty = torch.empty((0,), dtype=torch.long, device=self.device)
        if not self._archive_allocated:
            empty_idx = torch.empty((0, self._replay_topk()), dtype=torch.long, device=self.device)
            return {
                "success_idx": empty_idx,
                "success_counts": empty,
                "success_env_ids": empty,
                "success_slot_ids": empty,
                "fallback_idx": empty_idx,
                "fallback_counts": empty,
                "fallback_env_ids": empty,
                "fallback_slot_ids": empty,
            }

        success_idx = self._archive_fastest_idx.clone()
        if self._replay_source() == "fastest":
            fallback_idx = torch.full_like(self._archive_return_idx, -1)
            for env_id in range(self._archive_num_envs):
                write_idx = 0
                fastest_slots = {
                    int(slot)
                    for slot in self._archive_fastest_idx[env_id].tolist()
                    if int(slot) >= 0
                }
                for slot in self._archive_return_idx[env_id].tolist():
                    if int(slot) < 0 or int(slot) in fastest_slots:
                        continue
                    fallback_idx[env_id, write_idx] = int(slot)
                    write_idx += 1
                    if write_idx >= fallback_idx.shape[1]:
                        break
        else:
            fallback_idx = self._archive_return_idx.clone()

        env_grid = torch.arange(self._archive_num_envs, device=self.device).unsqueeze(1)
        env_grid = env_grid.expand_as(success_idx)
        success_mask = success_idx >= 0
        fallback_mask = fallback_idx >= 0
        success_env_ids = env_grid[success_mask]
        fallback_env_ids = env_grid[fallback_mask]

        return {
            "success_idx": success_idx,
            "success_counts": success_mask.sum(dim=1),
            "success_env_ids": success_env_ids,
            "success_slot_ids": success_idx[success_mask],
            "fallback_idx": fallback_idx,
            "fallback_counts": fallback_mask.sum(dim=1),
            "fallback_env_ids": fallback_env_ids,
            "fallback_slot_ids": fallback_idx[fallback_mask],
        }

    def prepare_sil_context(self, reward_normalizer, schedule_context, stats):
        source_idx, source_counts = self._build_sil_source_indices()
        pools = self._build_sil_pools()
        has_source = source_counts > 0
        selected_slots = source_idx[source_idx >= 0]
        has_source_any = bool(has_source.any().item())
        if self._use_global_fastest_mix():
            success_slots = pools["success_slot_ids"]
            fallback_slots = pools["fallback_slot_ids"]
            selected_slots = torch.cat([success_slots, fallback_slots], dim=0)
            has_source_any = int(selected_slots.numel()) > 0
        stats["replay_has_demo"] = int(has_source_any)
        stats["replay_dataset_trajectories"] = int(selected_slots.numel())
        stats["replay_unique_trajectories"] = int(selected_slots.numel())
        if self._use_global_fastest_mix():
            selected_envs = torch.cat([pools["success_env_ids"], pools["fallback_env_ids"]], dim=0)
            stats["replay_dataset_steps"] = int(
                self._archive_length[selected_envs, selected_slots].sum().item()
            ) if int(selected_slots.numel()) > 0 else 0
        else:
            stats["replay_dataset_steps"] = int(
                self._archive_length[
                    torch.arange(self._archive_num_envs, device=self.device).unsqueeze(1).expand_as(source_idx)[source_idx >= 0],
                    selected_slots,
                ].sum().item()
            ) if int(selected_slots.numel()) > 0 else 0
        stats["replay_fastest_tasks"] = int((self._archive_fastest_idx >= 0).any(dim=1).sum().item())
        stats["replay_success_pool_trajectories"] = int(pools["success_slot_ids"].numel())
        stats["replay_fallback_pool_trajectories"] = int(pools["fallback_slot_ids"].numel())
        if self._replay_source() == "fastest":
            fastest_counts = (self._archive_fastest_idx >= 0).sum(dim=1)
            if self._use_global_fastest_mix():
                fallback_envs = pools["fallback_env_ids"]
                stats["replay_return_fallback_tasks"] = int(torch.unique(fallback_envs).numel())
            else:
                stats["replay_return_fallback_tasks"] = int(((source_counts > fastest_counts) & has_source).sum().item())

        return {
            "demos": [],
            "dataset": None,
            "has_tsil_memory": True,
            "tsil_memory": self,
            "source_idx": source_idx,
            "source_counts": source_counts,
            **pools,
            "reward_normalizer": reward_normalizer,
            "schedule_context": schedule_context,
            "sample_batch_fn": self.sample_sil_batch,
            "positive_filter_enabled": False,
            "stats": stats,
        }

    def _deadline_for_batch(self, env_ids, task_ids, schedule_context):
        max_eps_time = float((schedule_context or {}).get("max_eps_time", 1.0))
        deadline = torch.full((int(env_ids.numel()),), max_eps_time, dtype=torch.float32, device=self.device)
        if schedule_context is None:
            return deadline

        by_env = schedule_context.get("time2end_by_env")
        if by_env is not None:
            by_env = torch.as_tensor(by_env, dtype=torch.float32, device=self.device)
            valid = env_ids < int(by_env.numel())
            if bool(valid.any().item()):
                deadline[valid] = by_env[env_ids[valid]]
            return deadline

        by_task = schedule_context.get("time2end_by_task", {})
        if by_task:
            for task_id in torch.unique(task_ids).tolist():
                mask = task_ids == int(task_id)
                deadline[mask] = float(by_task.get(int(task_id), max_eps_time))
        return deadline

    def _timeaware_bonus_for_batch(self, env_ids, slot_ids, task_ids, schedule_context):
        bonus = torch.zeros((int(env_ids.numel()),), dtype=torch.float32, device=self.device)
        if schedule_context is None:
            return bonus
        scale = float(schedule_context.get("r_epstime_scale", 0.0))
        if scale <= 0.0:
            return bonus

        success = self._archive_success[env_ids, slot_ids]
        if not bool(success.any().item()):
            return bonus

        eps_time = self._archive_eps_time[env_ids, slot_ids]
        max_eps_time = max(float(schedule_context.get("max_eps_time", 1.0)), 1e-8)
        deadline = self._deadline_for_batch(env_ids, task_ids, schedule_context)
        remaining = deadline - eps_time

        if schedule_context.get("ratio_range", None) is None:
            eps_time_reward = torch.ones_like(eps_time)
            if bool(schedule_context.get("update_ddl", False)):
                safe_eps_time = torch.clamp(eps_time, min=float(schedule_context.get("ctrl_dt", 1e-8)))
                eps_time_reward = eps_time_reward + torch.clamp(deadline / safe_eps_time, min=0.0, max=1.0)
            # eps_time_reward = 1.0 + (max_eps_time - eps_time) / max_eps_time
            # if bool(schedule_context.get("update_ddl", False)):
            #     eps_time_reward = eps_time_reward + (eps_time <= deadline + 0.1 * deadline).to(dtype=eps_time_reward.dtype)
        else:
            eps_time_reward = -torch.clamp(torch.abs(remaining), max=5.0)

        bonus = scale * eps_time_reward
        return torch.where(success, bonus, torch.zeros_like(bonus))

    def _discounted_returns_from_rewards(self, rewards, lengths, bootstrap_values=None):
        returns = torch.zeros_like(rewards)
        if bootstrap_values is None:
            running = torch.zeros((int(rewards.shape[0]),), dtype=rewards.dtype, device=rewards.device)
        else:
            running = bootstrap_values.to(dtype=rewards.dtype, device=rewards.device).view(-1)
        gamma = float(self.args.gamma)
        for idx in range(int(rewards.shape[1]) - 1, -1, -1):
            active = idx < lengths
            running = torch.where(active, rewards[:, idx] + gamma * running, running)
            returns[:, idx] = torch.where(active, running, torch.zeros_like(running))
        return returns

    def _fallback_bootstrap_values(self, env_ids, slot_ids, task_ids, replay_context, schedule_context):
        values = torch.zeros((int(env_ids.numel()),), dtype=torch.float32, device=self.device)
        if not getattr(self.args, "value_bootstrap", False):
            return values
        agent = (replay_context or {}).get("_sil_value_agent")
        if agent is None or self._archive_timeout is None or self._archive_terminal_next_state is None:
            return values
        mask = self._archive_timeout[env_ids, slot_ids] & (~self._archive_success[env_ids, slot_ids])
        if not bool(mask.any().item()):
            return values

        state = self._archive_terminal_next_state[env_ids[mask], slot_ids[mask]].to(dtype=self.dtype).clone()
        if schedule_context is not None and state.ndim >= 2 and state.shape[-1] > 2:
            deadline = self._deadline_for_batch(env_ids, task_ids, schedule_context)
            state[:, 2] = deadline[mask].to(dtype=state.dtype)
        with torch.no_grad():
            value, _, _ = agent.get_value(state, denorm=True, update=False)
        values[mask] = value.detach().to(dtype=torch.float32, device=self.device).view(-1)
        return values

    def _sample_pool(self, pool_env_ids, pool_slot_ids, sample_count):
        sample_count = int(sample_count)
        if sample_count <= 0 or int(pool_slot_ids.numel()) == 0:
            empty = torch.empty((0,), dtype=torch.long, device=self.device)
            return empty, empty
        choice = torch.randint(
            0,
            int(pool_slot_ids.numel()),
            (sample_count,),
            dtype=torch.long,
            device=self.device,
        )
        return pool_env_ids[choice], pool_slot_ids[choice]

    def _sample_source_indices(self, source_idx, source_counts, batch_size):
        return self._sample_source_indices_with_cursor(source_idx, source_counts, batch_size, "_round_robin_cursor")

    def _sample_source_indices_with_cursor(self, source_idx, source_counts, batch_size, cursor_name):
        available_envs = torch.nonzero(source_counts > 0, as_tuple=False).view(-1)
        if int(available_envs.numel()) == 0:
            empty = torch.empty((0,), dtype=torch.long, device=self.device)
            return empty, empty

        cursor = int(getattr(self, cursor_name, 0))
        positions = (
            torch.arange(batch_size, dtype=torch.long, device=self.device) + cursor
        ) % int(available_envs.numel())
        setattr(self, cursor_name, int((cursor + batch_size) % int(available_envs.numel())))
        env_ids = available_envs[positions]
        counts = source_counts[env_ids]
        ranks = torch.floor(torch.rand((batch_size,), device=self.device) * counts.to(torch.float32)).to(torch.long)
        return env_ids, source_idx[env_ids, ranks]

    def _sample_global_fastest_mix(
        self,
        replay_context,
        batch_size,
        sample_unit="transition",
        transition_budget=None,
    ):
        success_idx = replay_context.get("success_idx")
        success_counts = replay_context.get("success_counts")
        success_envs = replay_context.get("success_env_ids")
        success_slots = replay_context.get("success_slot_ids")
        fallback_idx = replay_context.get("fallback_idx")
        fallback_counts = replay_context.get("fallback_counts")
        fallback_envs = replay_context.get("fallback_env_ids")
        fallback_slots = replay_context.get("fallback_slot_ids")
        if (
            success_idx is None or success_counts is None or success_envs is None or success_slots is None
            or fallback_idx is None or fallback_counts is None or fallback_envs is None or fallback_slots is None
        ):
            pools = self._build_sil_pools()
            success_idx = pools["success_idx"]
            success_counts = pools["success_counts"]
            success_envs = pools["success_env_ids"]
            success_slots = pools["success_slot_ids"]
            fallback_idx = pools["fallback_idx"]
            fallback_counts = pools["fallback_counts"]
            fallback_envs = pools["fallback_env_ids"]
            fallback_slots = pools["fallback_slot_ids"]

        has_success = int(success_slots.numel()) > 0
        has_fallback = int(fallback_slots.numel()) > 0
        if not has_success and not has_fallback:
            empty = torch.empty((0,), dtype=torch.long, device=self.device)
            return empty, empty
        if not has_success:
            return self._sample_source_indices_with_cursor(
                fallback_idx, fallback_counts, batch_size, "_fallback_round_robin_cursor"
            )

        frac = min(max(self._success_sample_frac(), 0.0), 1.0)
        success_count = int(round(float(batch_size) * frac))
        if not has_fallback:
            success_count = int(batch_size)
        elif success_count > 0:
            if str(sample_unit).lower() == "trajectory":
                transition_budget = max(int(transition_budget or batch_size), 1)
                target_success_steps = int(round(float(transition_budget) * frac))
                min_success_steps = max(1, int(round(float(transition_budget) * min(frac, 0.05))))
                success_lengths = self._archive_length[success_envs, success_slots].to(dtype=torch.float32)
                pool_cap_steps = int(success_lengths.sum().item())
                success_step_budget = min(target_success_steps, max(min_success_steps, pool_cap_steps))
                avg_success_len = max(float(success_lengths.mean().item()), 1.0)
                success_count = min(
                    success_count,
                    max(1, int(np.ceil(float(success_step_budget) / avg_success_len))),
                )
            else:
                min_success_count = max(1, int(round(float(batch_size) * min(frac, 0.05))))
                pool_cap = int(self._archive_length[success_envs, success_slots].sum().item())
                success_count = min(success_count, max(min_success_count, pool_cap))
        fallback_count = int(batch_size) - success_count
        if fallback_count > 0 and not has_fallback:
            success_count = int(batch_size)
            fallback_count = 0

        success_env_ids, success_slot_ids = self._sample_source_indices_with_cursor(
            success_idx, success_counts, success_count, "_success_round_robin_cursor"
        )
        fallback_env_ids, fallback_slot_ids = self._sample_source_indices_with_cursor(
            fallback_idx, fallback_counts, fallback_count, "_fallback_round_robin_cursor"
        )
        env_ids = torch.cat([success_env_ids, fallback_env_ids], dim=0)
        slot_ids = torch.cat([success_slot_ids, fallback_slot_ids], dim=0)
        if int(env_ids.numel()) > 1:
            perm = torch.randperm(int(env_ids.numel()), device=self.device)
            env_ids = env_ids[perm]
            slot_ids = slot_ids[perm]
        return env_ids, slot_ids

    def _estimate_trajectory_sample_count(self, replay_context, transition_budget):
        transition_budget = max(int(transition_budget), 1)
        source_idx = replay_context.get("source_idx")
        source_counts = replay_context.get("source_counts")
        if source_idx is None or source_counts is None:
            source_idx, source_counts = self._build_sil_source_indices()

        if self._use_global_fastest_mix():
            success_envs = replay_context.get("success_env_ids")
            success_slots = replay_context.get("success_slot_ids")
            fallback_envs = replay_context.get("fallback_env_ids")
            fallback_slots = replay_context.get("fallback_slot_ids")
            if success_envs is None or success_slots is None or fallback_envs is None or fallback_slots is None:
                pools = self._build_sil_pools()
                success_envs = pools["success_env_ids"]
                success_slots = pools["success_slot_ids"]
                fallback_envs = pools["fallback_env_ids"]
                fallback_slots = pools["fallback_slot_ids"]
            env_ids = torch.cat([success_envs, fallback_envs], dim=0)
            slot_ids = torch.cat([success_slots, fallback_slots], dim=0)
        else:
            valid = source_idx >= 0
            env_grid = torch.arange(self._archive_num_envs, device=self.device).unsqueeze(1).expand_as(source_idx)
            env_ids = env_grid[valid]
            slot_ids = source_idx[valid]

        if int(slot_ids.numel()) == 0:
            return 1
        avg_len = float(self._archive_length[env_ids, slot_ids].to(dtype=torch.float32).mean().item())
        avg_len = max(avg_len, 1.0)
        return max(min(int((transition_budget + avg_len - 1) // avg_len), transition_budget), 1)

    def _trajectory_sil_weights(self, env_ids, slot_ids, lengths, flat_count, schedule_context):
        traj_weights = torch.ones((int(env_ids.numel()),), dtype=self.dtype, device=self.device)
        success = self._archive_success[env_ids, slot_ids]
        if self._use_global_fastest_mix():
            task_ids = self._archive_task_id[env_ids, slot_ids]
            success_weight = self._temporal_efficiency_weights(env_ids, slot_ids, task_ids, schedule_context)
            traj_weights = torch.where(
                success,
                success_weight.to(dtype=self.dtype),
                traj_weights,
            )
        elif self._replay_source() == "fastest":
            min_priority = float(getattr(self.args, "sil_speed_priority_min", 0.25))
            max_priority = float(getattr(self.args, "sil_speed_priority_max", 2.0))
            max_eps_time = max(float((schedule_context or {}).get("max_eps_time", 1.0)), 1e-8)
            speed = torch.clamp(1.0 - self._archive_eps_time[env_ids, slot_ids] / max_eps_time, min=0.0, max=1.0)
            traj_weights = torch.where(
                success,
                1.0 + speed,
                torch.full_like(speed, min_priority),
            ).clamp(min=min_priority, max=max_priority).to(dtype=self.dtype)

        # Convert a per-trajectory mean into a flat per-step mean.
        scale = float(flat_count) / torch.clamp(
            lengths.to(dtype=self.dtype) * max(int(env_ids.numel()), 1),
            min=1.0,
        )
        return traj_weights * scale

    def _temporal_efficiency_weights(self, env_ids, slot_ids, task_ids, schedule_context):
        ctrl_dt = max(float((schedule_context or {}).get("ctrl_dt", 1e-8)), 1e-8)
        eps_time = torch.clamp(self._archive_eps_time[env_ids, slot_ids], min=ctrl_dt)
        deadline = self._deadline_for_batch(env_ids, task_ids, schedule_context).to(dtype=eps_time.dtype)
        return 1.0 + torch.clamp(deadline / eps_time, min=0.0, max=1.0)

    def sample_sil_trajectory_batch(self, replay_context):
        transition_budget = max(int(replay_context.get("_sil_sample_batch_size", getattr(self.args, "sil_batch_size", 1))), 1)
        traj_count = self._estimate_trajectory_sample_count(replay_context, transition_budget)
        source_idx = replay_context.get("source_idx")
        source_counts = replay_context.get("source_counts")
        if source_idx is None or source_counts is None:
            source_idx, source_counts = self._build_sil_source_indices()

        if self._use_global_fastest_mix():
            env_ids, slot_ids = self._sample_global_fastest_mix(
                replay_context,
                traj_count,
                sample_unit="trajectory",
                transition_budget=transition_budget,
            )
        else:
            env_ids, slot_ids = self._sample_source_indices(source_idx, source_counts, traj_count)
        if int(env_ids.numel()) == 0:
            return None

        lengths = self._archive_length[env_ids, slot_ids]
        task_ids = self._archive_task_id[env_ids, slot_ids]
        time_grid = torch.arange(self._archive_max_len, dtype=torch.long, device=self.device).unsqueeze(0)
        valid = time_grid < lengths.unsqueeze(1)
        flat_count = int(valid.sum().item())
        if flat_count <= 0:
            return None

        obs_seq = self._archive_obs[env_ids, slot_ids].to(dtype=self.dtype).clone()
        state_seq = self._archive_state[env_ids, slot_ids].to(dtype=self.dtype).clone()
        action_seq = self._archive_action[env_ids, slot_ids].to(dtype=self.dtype)

        schedule_context = replay_context.get("schedule_context")
        if schedule_context is not None:
            deadline = self._deadline_for_batch(env_ids, task_ids, schedule_context)
            if obs_seq.ndim >= 3 and obs_seq.shape[-1] > 2:
                obs_seq[:, :, 2] = deadline.to(dtype=obs_seq.dtype).unsqueeze(1)
            if state_seq.ndim >= 3 and state_seq.shape[-1] > 2:
                state_seq[:, :, 2] = deadline.to(dtype=state_seq.dtype).unsqueeze(1)

        reward_seq = self._archive_base_reward[env_ids, slot_ids].clone()
        if schedule_context is not None and float(schedule_context.get("r_epstime_scale", 0.0)) > 0.0:
            terminal_idx = lengths - 1
            reward_seq[torch.arange(int(env_ids.numel()), dtype=torch.long, device=self.device), terminal_idx] += (
                self._timeaware_bonus_for_batch(env_ids, slot_ids, task_ids, schedule_context)
            )
            reward_seq = torch.clamp(reward_seq, min=0.0)

        if self.args.norm_rew:
            reward_normalizer = replay_context.get("reward_normalizer")
            if reward_normalizer is None:
                raise ValueError("SIL replay requires a reward_normalizer when norm_rew=True.")
            flat_task_ids = task_ids.unsqueeze(1).expand(-1, self._archive_max_len).reshape(-1)
            reward_seq = reward_normalizer(
                reward_seq.reshape(-1),
                flat_task_ids,
            ).view(int(env_ids.numel()), self._archive_max_len).to(dtype=torch.float32, device=self.device)

        bootstrap_values = self._fallback_bootstrap_values(
            env_ids, slot_ids, task_ids, replay_context, schedule_context,
        )
        returns = self._discounted_returns_from_rewards(reward_seq, lengths, bootstrap_values)
        traj_weights = self._trajectory_sil_weights(env_ids, slot_ids, lengths, flat_count, schedule_context)
        step_weights = traj_weights.unsqueeze(1).expand_as(reward_seq)[valid].to(dtype=self.dtype)
        success = self._archive_success[env_ids, slot_ids]

        return {
            "obs": obs_seq[valid],
            "state": state_seq[valid],
            "action": action_seq[valid],
            "returns": returns[valid].to(dtype=self.dtype),
            "task_ids": task_ids.unsqueeze(1).expand(-1, self._archive_max_len)[valid],
            "sil_weights": step_weights,
            "sil_success_mask": success.unsqueeze(1).expand(-1, self._archive_max_len)[valid],
            "sil_fallback_mask": (~success).unsqueeze(1).expand(-1, self._archive_max_len)[valid],
            "env_ids": env_ids.unsqueeze(1).expand(-1, self._archive_max_len)[valid],
            "slot_ids": slot_ids.unsqueeze(1).expand(-1, self._archive_max_len)[valid],
            "time_idx": time_grid.expand(int(env_ids.numel()), -1)[valid],
            "trajectory_count": int(env_ids.numel()),
        }

    def sample_sil_batch(self, replay_context):
        if self._sil_sample_unit() == "trajectory":
            return self.sample_sil_trajectory_batch(replay_context)

        source_idx = replay_context.get("source_idx")
        source_counts = replay_context.get("source_counts")
        if source_idx is None or source_counts is None:
            source_idx, source_counts = self._build_sil_source_indices()

        batch_size = max(int(replay_context.get("_sil_sample_batch_size", getattr(self.args, "sil_batch_size", 1))), 1)
        if self._use_global_fastest_mix():
            env_ids, slot_ids = self._sample_global_fastest_mix(replay_context, batch_size)
        else:
            env_ids, slot_ids = self._sample_source_indices(source_idx, source_counts, batch_size)
        if int(env_ids.numel()) == 0:
            return None
        lengths = self._archive_length[env_ids, slot_ids]
        time_idx = torch.floor(torch.rand((batch_size,), device=self.device) * lengths.to(torch.float32)).to(torch.long)
        task_ids = self._archive_task_id[env_ids, slot_ids]

        obs = self._archive_obs[env_ids, slot_ids, time_idx].to(dtype=self.dtype).clone()
        state = self._archive_state[env_ids, slot_ids, time_idx].to(dtype=self.dtype).clone()
        action = self._archive_action[env_ids, slot_ids, time_idx].to(dtype=self.dtype)

        schedule_context = replay_context.get("schedule_context")
        if schedule_context is not None:
            deadline = self._deadline_for_batch(env_ids, task_ids, schedule_context)
            # Public time obs is [time_ratio, time_cur, ddl]. SIL keeps the
            # archived elapsed time and only relabels the episode deadline.
            if obs.ndim >= 2 and obs.shape[-1] > 2:
                obs[:, 2] = deadline.to(dtype=obs.dtype)
            if state.ndim >= 2 and state.shape[-1] > 2:
                state[:, 2] = deadline.to(dtype=state.dtype)

        reward_seq = self._archive_base_reward[env_ids, slot_ids].clone()
        if schedule_context is not None and float(schedule_context.get("r_epstime_scale", 0.0)) > 0.0:
            terminal_idx = lengths - 1
            reward_seq[torch.arange(batch_size, dtype=torch.long, device=self.device), terminal_idx] += (
                self._timeaware_bonus_for_batch(env_ids, slot_ids, task_ids, schedule_context)
            )
            reward_seq = torch.clamp(reward_seq, min=0.0)

        if self.args.norm_rew:
            reward_normalizer = replay_context.get("reward_normalizer")
            if reward_normalizer is None:
                raise ValueError("SIL replay requires a reward_normalizer when norm_rew=True.")
            flat_task_ids = task_ids.unsqueeze(1).expand(-1, self._archive_max_len).reshape(-1)
            reward_seq = reward_normalizer(
                reward_seq.reshape(-1),
                flat_task_ids,
            ).view(batch_size, self._archive_max_len).to(dtype=torch.float32, device=self.device)

        bootstrap_values = self._fallback_bootstrap_values(
            env_ids, slot_ids, task_ids, replay_context, schedule_context,
        )
        returns = self._discounted_returns_from_rewards(reward_seq, lengths, bootstrap_values)
        returns_batch = returns[torch.arange(batch_size, dtype=torch.long, device=self.device), time_idx].to(dtype=self.dtype)

        sil_weights = torch.ones((batch_size,), dtype=self.dtype, device=self.device)
        success = self._archive_success[env_ids, slot_ids]
        if self._use_global_fastest_mix():
            success_weight = self._temporal_efficiency_weights(env_ids, slot_ids, task_ids, schedule_context)
            sil_weights = torch.where(
                success,
                success_weight.to(dtype=self.dtype),
                sil_weights,
            )
        elif self._replay_source() == "fastest":
            min_priority = float(getattr(self.args, "sil_speed_priority_min", 0.25))
            max_priority = float(getattr(self.args, "sil_speed_priority_max", 2.0))
            max_eps_time = max(float((schedule_context or {}).get("max_eps_time", 1.0)), 1e-8)
            speed = torch.clamp(1.0 - self._archive_eps_time[env_ids, slot_ids] / max_eps_time, min=0.0, max=1.0)
            sil_weights = torch.where(
                success,
                1.0 + speed,
                torch.full_like(speed, min_priority),
            ).clamp(min=min_priority, max=max_priority).to(dtype=self.dtype)

        return {
            "obs": obs,
            "state": state,
            "action": action,
            "returns": returns_batch,
            "task_ids": task_ids,
            "sil_weights": sil_weights,
            "sil_success_mask": success,
            "sil_fallback_mask": ~success,
            "env_ids": env_ids,
            "slot_ids": slot_ids,
            "time_idx": time_idx,
        }
