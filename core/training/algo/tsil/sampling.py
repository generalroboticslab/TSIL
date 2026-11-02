"""Private TSIL trajectory-memory sampling helpers."""

import torch


class _TsilMemorySamplingMixin:
    """Sampling and replay-context methods for TsilTrajectoryMemory."""

    def _relabel_public_deadline(self, tensor, deadline):
        """Relabel deadline-derived public fields while preserving elapsed time."""
        if tensor.ndim < 2 or tensor.shape[-1] <= 2:
            return
        # Keep replay inputs consistent with the environment masking policy.
        # In particular, PPO-SIL must not unmask the target during relabeling.
        if getattr(self.args, "fix_priv", False) or not getattr(self.args, "time2end", True):
            masked = 0.0 if getattr(self.args, "benchmark_backend", None) == "minigrid" else 1.0
            tensor[..., :3] = masked
            if getattr(self.args, "explicit_remaining_time_obs", False):
                tensor[..., 3] = masked
            return
        deadline = deadline.to(dtype=tensor.dtype, device=tensor.device)
        tensor[..., 2] = deadline
        if bool(getattr(self.args, "explicit_remaining_time_obs", False)):
            if tensor.shape[-1] <= 3:
                raise ValueError("Explicit remaining-time replay state is missing public feature index 3")
            tensor[..., 3] = deadline - tensor[..., 1]

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
        if getattr(self, "_global_per", None) is not None:
            return self._global_per.context(self, reward_normalizer, schedule_context, stats)
        source_idx, source_counts = self._build_sil_source_indices()
        pools = self._build_sil_pools()
        if bool(getattr(self.args, "sil_aligned_success_only", False)):
            source_idx = pools["success_idx"]
            source_counts = pools["success_counts"]
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

    def _remaining_time_targets_for_batch(self, lengths, time_idx, schedule_context):
        ctrl_dt = float((schedule_context or {}).get("ctrl_dt", 1.0))
        remaining_steps = torch.clamp(
            lengths.to(dtype=torch.float32, device=self.device)
            - time_idx.to(dtype=torch.float32, device=self.device),
            min=0.0,
        )
        return (remaining_steps * ctrl_dt).to(dtype=self.dtype, device=self.device)

    def _timeaware_bonus_for_batch(self, env_ids, slot_ids, task_ids, schedule_context):
        bonus = torch.zeros((int(env_ids.numel()),), dtype=torch.float32, device=self.device)
        if schedule_context is None:
            return bonus
        scale = float(schedule_context.get("r_epstime_scale", 0.0))
        replay_scale = float(getattr(getattr(self, "args", None), "sil_relabel_reward_scale", -1.0))
        if replay_scale >= 0.0:
            scale = replay_scale
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
            efficiency_bonus = schedule_context.get("temporal_efficiency_bonus")
            if efficiency_bonus is None:
                efficiency_bonus = schedule_context.get("update_ddl", False)
            if efficiency_bonus:
                safe_eps_time = torch.clamp(eps_time, min=float(schedule_context.get("ctrl_dt", 1e-8)))
                eps_time_reward = eps_time_reward + torch.clamp(deadline / safe_eps_time, min=0.0, max=1.0)
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

    def _fallback_bootstrap_values(
        self,
        env_ids,
        slot_ids,
        task_ids,
        replay_context,
        schedule_context,
        return_mask=False,
    ):
        values = torch.zeros((int(env_ids.numel()),), dtype=torch.float32, device=self.device)
        active = torch.zeros_like(values, dtype=torch.bool)
        if not getattr(self.args, "value_bootstrap", False):
            return (values, active) if return_mask else values
        agent = (replay_context or {}).get("_sil_value_agent")
        if agent is None or self._archive_timeout is None or self._archive_terminal_next_state is None:
            if getattr(self.args, "sil_success_bootstrap", False):
                raise ValueError("Success bootstrap requires critic and archived terminal next states")
            return (values, active) if return_mask else values
        success = self._archive_success[env_ids, slot_ids]
        success_bootstrap = bool(getattr(self.args, "sil_success_bootstrap", False))
        mask = self._archive_timeout[env_ids, slot_ids] & (~success)
        if success_bootstrap:
            mask = mask | success
        if not bool(mask.any().item()):
            return (values, active) if return_mask else values

        state = self._archive_terminal_next_state[env_ids[mask], slot_ids[mask]].to(dtype=self.dtype).clone()
        if schedule_context is not None and state.ndim >= 2 and state.shape[-1] > 2:
            deadline = self._deadline_for_batch(env_ids, task_ids, schedule_context)
            self._relabel_public_deadline(state, deadline[mask])
        with torch.no_grad():
            value, _, _ = agent.get_value(state, denorm=True, update=False)
        value = value.detach().to(dtype=torch.float32, device=self.device).view(-1)
        if success_bootstrap:
            selected_success = success[mask]
            raw_success = value[selected_success]
            if raw_success.numel():
                replay_context["_success_target_stats"] = {
                    "success_terminal_value_mean": float(raw_success.mean().item()),
                    "success_terminal_value_max": float(raw_success.max().item()),
                    "success_terminal_negative_frac": float((raw_success < 0).float().mean().item()),
                    "success_terminal_added_value_mean": float(raw_success.clamp_min(0).mean().item()),
                }
            # Only the added successful continuation is clamped. Base rewards and
            # timeout continuation retain their original signs. Values are already
            # in reward-normalized return units (denorm undoes value normalization).
            value = torch.where(selected_success, value.clamp_min(0), value)
        values[mask] = value
        active[mask] = True
        return (values, active) if return_mask else values

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

    def _rank_env_source_order(self, source_idx):
        """Return candidates rank-major, then environment-minor."""
        if source_idx is None or int(source_idx.numel()) == 0:
            empty = torch.empty((0,), dtype=torch.long, device=self.device)
            return empty, empty
        valid = source_idx >= 0
        if not bool(valid.any().item()):
            empty = torch.empty((0,), dtype=torch.long, device=self.device)
            return empty, empty
        env_grid = torch.arange(int(source_idx.shape[0]), device=self.device).unsqueeze(1).expand_as(source_idx)
        rank_first_valid = valid.transpose(0, 1).contiguous()
        return (
            env_grid.transpose(0, 1).contiguous()[rank_first_valid],
            source_idx.transpose(0, 1).contiguous()[rank_first_valid],
        )

    def _sample_source_indices_with_cursor(self, source_idx, source_counts, batch_size, cursor_name):
        del source_counts  # source_idx already encodes valid candidates
        batch_size = int(batch_size)
        order_env_ids, order_slot_ids = self._rank_env_source_order(source_idx)
        order_count = int(order_slot_ids.numel())
        if batch_size <= 0 or order_count == 0:
            empty = torch.empty((0,), dtype=torch.long, device=self.device)
            return empty, empty

        cursor = int(getattr(self, cursor_name, 0)) % order_count
        positions = (
            torch.arange(batch_size, dtype=torch.long, device=self.device) + cursor
        ) % order_count
        setattr(self, cursor_name, int((cursor + batch_size) % order_count))
        return order_env_ids[positions], order_slot_ids[positions]

    def _sample_source_trajectories_until_budget(
        self,
        source_idx,
        source_counts,
        transition_budget,
        cursor_name,
        *,
        allow_repeat=True,
        selected_steps_by_env=None,
        env_step_quota=0,
    ):
        """Select complete trajectories until a transition budget is reached."""
        del source_counts
        transition_budget = int(transition_budget)
        order_env_ids, order_slot_ids = self._rank_env_source_order(source_idx)
        order_count = int(order_slot_ids.numel())
        if transition_budget <= 0 or order_count == 0:
            empty = torch.empty((0,), dtype=torch.long, device=self.device)
            return empty, empty, 0

        if selected_steps_by_env is None:
            selected_steps_by_env = torch.zeros(
                self._archive_num_envs, dtype=torch.long, device=self.device,
            )
        cursor = int(getattr(self, cursor_name, 0)) % order_count
        selected_positions = []
        selected_steps = 0
        offset = 0
        rejected_since_selection = 0
        max_visits = None if allow_repeat else order_count

        while selected_steps < transition_budget and (max_visits is None or offset < max_visits):
            position = int((cursor + offset) % order_count)
            offset += 1
            env_id = int(order_env_ids[position].item())
            slot_id = int(order_slot_ids[position].item())
            length = int(self._archive_length[env_id, slot_id].item())
            current_env_steps = int(selected_steps_by_env[env_id].item())
            quota_allows = (
                env_step_quota <= 0
                or current_env_steps == 0
                or current_env_steps + length <= env_step_quota
            )
            if length <= 0 or not quota_allows:
                rejected_since_selection += 1
                if allow_repeat and rejected_since_selection >= order_count:
                    break
                continue

            selected_positions.append(position)
            selected_steps += length
            selected_steps_by_env[env_id] += length
            rejected_since_selection = 0

        setattr(self, cursor_name, int((cursor + offset) % order_count))
        if not selected_positions:
            empty = torch.empty((0,), dtype=torch.long, device=self.device)
            return empty, empty, 0
        positions = torch.as_tensor(selected_positions, dtype=torch.long, device=self.device)
        return order_env_ids[positions], order_slot_ids[positions], int(selected_steps)

    def _success_repeat_mode(self):
        return str(getattr(self.args, "sil_success_repeat", "to_fraction")).strip().lower()

    def _trajectory_pool_steps(self, env_ids, slot_ids):
        if env_ids is None or slot_ids is None or int(slot_ids.numel()) == 0:
            return 0
        return int(self._archive_length[env_ids, slot_ids].sum().item())

    def _success_transition_budget(self, success_envs, success_slots, transition_budget, frac):
        target = max(int(round(float(transition_budget) * frac)), 0)
        pool_steps = self._trajectory_pool_steps(success_envs, success_slots)
        mode = self._success_repeat_mode()
        if mode == "none":
            return min(target, pool_steps), False
        if mode == "to_floor":
            floor = min(max(float(getattr(self.args, "sil_success_floor", 0.0)), 0.0), frac)
            budget = min(target, max(pool_steps, int(round(float(transition_budget) * floor))))
            return budget, budget > pool_steps
        if mode == "to_fraction":
            return target, True
        raise ValueError(f"Unknown sil_success_repeat mode: {mode}")

    def _resolved_env_step_quota(self, success_envs, fallback_envs, transition_budget):
        configured = int(getattr(self.args, "sil_env_step_quota", 0))
        if configured == 0:
            return 0
        if configured > 0:
            return configured
        eligible = torch.cat([success_envs, fallback_envs], dim=0)
        eligible_count = int(torch.unique(eligible).numel()) if int(eligible.numel()) > 0 else 0
        if eligible_count <= 0:
            return 0
        uniform_share = (int(transition_budget) + eligible_count - 1) // eligible_count
        return max(int(self._archive_max_len), uniform_share)

    def _flat_batch_count(self, batch):
        if not batch or "action" not in batch:
            return 0
        return int(batch["action"].shape[0])

    def _shuffle_flat_sil_batch(self, batch):
        count = self._flat_batch_count(batch)
        if count <= 1 or not bool(getattr(self.args, "sil_shuffle_replay_transitions", False)):
            return batch
        perm = torch.randperm(count, device=batch["action"].device)
        return {
            key: value[perm]
            if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == count
            else value
            for key, value in batch.items()
        }

    def _slice_flat_sil_batch(self, batch, start, end):
        count = self._flat_batch_count(batch)
        if count <= 0 or int(start) >= count:
            return None
        start = max(int(start), 0)
        end = min(max(int(end), start), count)
        sliced = {
            key: value[start:end]
            if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == count
            else value
            for key, value in batch.items()
        }
        if "env_ids" in sliced and "slot_ids" in sliced and int(sliced["env_ids"].numel()) > 0:
            pairs = torch.stack([sliced["env_ids"].long(), sliced["slot_ids"].long()], dim=1)
            sliced["trajectory_count"] = int(torch.unique(pairs, dim=0).shape[0])
        return sliced

    def _index_flat_sil_batch(self, batch, indices):
        count = self._flat_batch_count(batch)
        if count <= 0 or indices is None or int(indices.numel()) == 0:
            return None
        indexed = {
            key: value[indices]
            if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == count
            else value
            for key, value in batch.items()
        }
        if "env_ids" in indexed and "slot_ids" in indexed:
            pairs = torch.stack([indexed["env_ids"].long(), indexed["slot_ids"].long()], dim=1)
            indexed["trajectory_count"] = int(torch.unique(pairs, dim=0).shape[0])
        return indexed

    def _use_update_scope_replay(self, replay_context):
        return (
            str(getattr(self.args, "sil_replay_build_scope", "minibatch")).strip().lower() == "update"
            and not bool((replay_context or {}).get("_sil_building_replay_batch", False))
        )

    def _next_update_scope_replay_slice(self, replay_context):
        batch = replay_context.get("_sil_replay_batch")
        if batch is None:
            return None
        order = replay_context.get("_sil_replay_order")
        if order is not None:
            slice_index = int(replay_context.get("_sil_replay_slice_index", 0))
            num_slices = max(int(replay_context.get("_sil_replay_num_slices", 1)), 1)
            count = int(order.numel())
            if slice_index >= num_slices or count <= 0:
                return None
            start = (slice_index * count) // num_slices
            end = ((slice_index + 1) * count) // num_slices
            replay_context["_sil_replay_slice_index"] = slice_index + 1
            return self._index_flat_sil_batch(batch, order[start:end])
        batch_size = max(int(replay_context.get("_sil_batch_size", getattr(self.args, "sil_batch_size", 1))), 1)
        cursor = int(replay_context.get("_sil_replay_cursor", 0))
        sliced = self._slice_flat_sil_batch(batch, cursor, cursor + batch_size)
        replay_context["_sil_replay_cursor"] = cursor + (
            0 if sliced is None else self._flat_batch_count(sliced)
        )
        return sliced

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
            if str(sample_unit).lower() == "trajectory":
                selected_steps_by_env = torch.zeros(
                    self._archive_num_envs, dtype=torch.long, device=self.device,
                )
                quota = self._resolved_env_step_quota(success_envs, fallback_envs, transition_budget or batch_size)
                env_ids, slot_ids, _ = self._sample_source_trajectories_until_budget(
                    fallback_idx,
                    fallback_counts,
                    max(int(transition_budget or batch_size), 1),
                    "_fallback_round_robin_cursor",
                    selected_steps_by_env=selected_steps_by_env,
                    env_step_quota=quota,
                )
                return env_ids, slot_ids
            return self._sample_source_indices_with_cursor(
                fallback_idx, fallback_counts, batch_size, "_fallback_round_robin_cursor"
            )

        frac = min(max(self._success_sample_frac(), 0.0), 1.0)
        if str(sample_unit).lower() == "trajectory":
            transition_budget = max(int(transition_budget or batch_size), 1)
            selected_steps_by_env = torch.zeros(
                self._archive_num_envs, dtype=torch.long, device=self.device,
            )
            quota = self._resolved_env_step_quota(success_envs, fallback_envs, transition_budget)
            success_budget, allow_success_repeat = self._success_transition_budget(
                success_envs, success_slots, transition_budget, frac,
            )
            success_env_ids, success_slot_ids, success_steps = self._sample_source_trajectories_until_budget(
                success_idx,
                success_counts,
                success_budget,
                "_success_round_robin_cursor",
                allow_repeat=allow_success_repeat,
                selected_steps_by_env=selected_steps_by_env,
                env_step_quota=quota,
            )
            fallback_env_ids, fallback_slot_ids, _ = self._sample_source_trajectories_until_budget(
                fallback_idx,
                fallback_counts,
                max(transition_budget - int(success_steps), 0),
                "_fallback_round_robin_cursor",
                allow_repeat=True,
                selected_steps_by_env=selected_steps_by_env,
                env_step_quota=quota,
            )
            return (
                torch.cat([success_env_ids, fallback_env_ids], dim=0),
                torch.cat([success_slot_ids, fallback_slot_ids], dim=0),
            )

        success_count = int(round(float(batch_size) * frac))
        if frac >= 1.0 - 1e-8 or not has_fallback:
            success_count = int(batch_size)
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
        return env_ids, slot_ids

    def _trajectory_replay_priorities(self, env_ids, slot_ids, schedule_context):
        traj_weights = torch.ones((int(env_ids.numel()),), dtype=self.dtype, device=self.device)
        success = self._archive_success[env_ids, slot_ids]
        if self._use_global_fastest_mix():
            task_ids = self._archive_task_id[env_ids, slot_ids]
            success_weight = self._success_replay_weights(env_ids, slot_ids, task_ids, schedule_context)
            traj_weights = torch.where(
                success,
                success_weight.to(dtype=self.dtype),
                traj_weights,
            )
        elif self._replay_source() == "fastest":
            task_ids = (
                self._archive_task_id[env_ids, slot_ids]
                if hasattr(self, "_archive_task_id")
                else torch.zeros_like(env_ids)
            )
            traj_weights = self._fastest_trajectory_weights(
                env_ids, slot_ids, task_ids, schedule_context,
            ).to(dtype=self.dtype)

        return traj_weights

    def _trajectory_sil_weights(self, env_ids, slot_ids, lengths, flat_count, schedule_context):
        traj_weights = self._trajectory_replay_priorities(
            env_ids, slot_ids, schedule_context,
        )

        # Convert a per-trajectory mean into a flat per-step mean.
        scale = float(flat_count) / torch.clamp(
            lengths.to(dtype=self.dtype) * max(int(env_ids.numel()), 1),
            min=1.0,
        )
        return traj_weights * scale

    def _episode_limits_for_batch(self, env_ids, schedule_context):
        limits = (schedule_context or {}).get("episode_limits_by_env")
        if limits is not None:
            return torch.as_tensor(limits, dtype=torch.float32, device=self.device)[env_ids].clamp(min=1e-8)
        return max(float((schedule_context or {}).get("max_eps_time", 1.0)), 1e-8)

    def _temporal_efficiency_weights(self, env_ids, slot_ids, task_ids, schedule_context):
        ctrl_dt = max(float((schedule_context or {}).get("ctrl_dt", 1e-8)), 1e-8)
        eps_time = torch.clamp(self._archive_eps_time[env_ids, slot_ids], min=ctrl_dt)
        deadline = self._deadline_for_batch(env_ids, task_ids, schedule_context).to(dtype=eps_time.dtype)
        return 1.0 + torch.clamp(deadline / eps_time, min=0.0, max=1.0)

    def _fastest_trajectory_weights(self, env_ids, slot_ids, task_ids, schedule_context):
        """Return success-only trajectory priorities for ordinary fastest replay."""
        success = self._archive_success[env_ids, slot_ids]
        mode = str(getattr(self.args, "sil_trajectory_weight_mode", "episode_limit")).lower()
        if mode == "none":
            return torch.ones((int(env_ids.numel()),), dtype=self.dtype, device=self.device)
        if mode == "adaptive_deadline":
            # Manuscript Sec. 3.3: w_i(tau) = 1 + 1_success min(D_i/T_success, 1).
            success_weights = self._temporal_efficiency_weights(
                env_ids, slot_ids, task_ids, schedule_context,
            )
            return torch.where(success, success_weights, torch.ones_like(success_weights))
        if mode == "episode_limit":
            min_priority = float(getattr(self.args, "sil_speed_priority_min", 0.25))
            max_priority = float(getattr(self.args, "sil_speed_priority_max", 2.0))
            max_eps_time = self._episode_limits_for_batch(env_ids, schedule_context)
            speed = torch.clamp(
                1.0 - self._archive_eps_time[env_ids, slot_ids] / max_eps_time,
                min=0.0,
                max=1.0,
            )
            return torch.where(
                success,
                1.0 + speed,
                torch.full_like(speed, min_priority),
            ).clamp(min=min_priority, max=max_priority)
        raise ValueError(
            "sil_trajectory_weight_mode must be 'none', 'episode_limit' or 'adaptive_deadline', "
            f"got {mode}"
        )

    def _success_replay_weights(self, env_ids, slot_ids, task_ids, schedule_context):
        mode = str(getattr(self.args, "sil_success_weight_mode", "adaptive")).lower()
        if mode == "adaptive":
            return self._temporal_efficiency_weights(env_ids, slot_ids, task_ids, schedule_context)
        if mode == "fixed":
            weight = float(getattr(self.args, "sil_fixed_success_weight", 2.0))
            return torch.full((int(env_ids.numel()),), weight, dtype=torch.float32, device=self.device)
        raise ValueError(
            "sil_success_weight_mode must be 'adaptive' or 'fixed', "
            f"got {mode}"
        )

    def _aligned_source_trajectories(self, source_idx):
        if source_idx is None or int(source_idx.numel()) == 0:
            empty = torch.empty((0,), dtype=torch.long, device=self.device)
            return empty, empty
        per_env = min(
            max(int(getattr(self.args, "sil_replay_trajectories_per_env", 1)), 1),
            int(source_idx.shape[1]),
        )
        rotate = bool(getattr(self.args, "sil_replay_rotate_trajectories", False))
        if not rotate or per_env >= int(source_idx.shape[1]):
            selected = source_idx[:, :per_env]
        else:
            counts = (source_idx >= 0).sum(dim=1)
            safe_counts = counts.clamp_min(1)
            phase = int(getattr(self, "_aligned_source_rank_cursor", 0))
            offsets = torch.arange(per_env, dtype=torch.long, device=self.device)
            ranks = (phase + offsets.unsqueeze(0)) % safe_counts.unsqueeze(1)
            selected = torch.gather(source_idx, 1, ranks)
            valid_ranks = offsets.unsqueeze(0) < torch.clamp(counts, max=per_env).unsqueeze(1)
            selected = torch.where(valid_ranks, selected, torch.full_like(selected, -1))
            self._aligned_source_rank_cursor = phase + 1
        valid = selected >= 0
        if not bool(valid.any().item()):
            empty = torch.empty((0,), dtype=torch.long, device=self.device)
            return empty, empty
        env_grid = torch.arange(self._archive_num_envs, device=self.device).unsqueeze(1).expand_as(selected)
        return env_grid[valid], selected[valid]

    def _aligned_replay_source_indices(self, replay_context, source_idx):
        """Choose the per-environment view used by PPO-aligned replay.

        The default preserves the historical combined fastest-success plus
        high-return ranking.  The success-first option instead rotates only
        through an environment's successful trajectories when any exist and
        falls back to its high-return trajectories only while it is unsolved.
        """
        if bool(getattr(self.args, "sil_replay_no_fallback", False)):
            return self._archive_fastest_idx
        if bool(getattr(self.args, "sil_aligned_success_only", False)):
            success_idx = replay_context.get("success_idx")
            return self._build_sil_pools()["success_idx"] if success_idx is None else success_idx
        if not bool(getattr(self.args, "sil_aligned_success_first", False)):
            return source_idx

        success_idx = replay_context.get("success_idx")
        fallback_idx = replay_context.get("fallback_idx")
        if success_idx is None or fallback_idx is None:
            pools = self._build_sil_pools()
            success_idx = pools["success_idx"]
            fallback_idx = pools["fallback_idx"]

        preferred = fallback_idx.clone()
        has_success = (success_idx >= 0).any(dim=1)
        preferred[has_success] = success_idx[has_success]
        return preferred

    def _aligned_transition_indices(self, env_ids, slot_ids):
        lengths = self._archive_length[env_ids, slot_ids]
        per_trajectory = max(
            int(getattr(self.args, "sil_replay_transitions_per_trajectory", 1)),
            1,
        )
        counts = torch.clamp(lengths, min=0, max=per_trajectory)
        max_count = int(counts.max().item()) if int(counts.numel()) > 0 else 0
        if max_count <= 0:
            empty = torch.empty((0,), dtype=torch.long, device=self.device)
            return empty, empty, counts, torch.zeros_like(counts, dtype=torch.bool)

        positions = torch.arange(max_count, dtype=torch.long, device=self.device).unsqueeze(0)
        valid = positions < counts.unsqueeze(1)
        safe_lengths = lengths.clamp_min(1)
        cursors = self._archive_transition_cursor[env_ids, slot_ids]
        offsets = self._archive_transition_offset[env_ids, slot_ids]
        starts = (offsets + cursors) % safe_lengths
        time_indices = (starts.unsqueeze(1) + positions) % safe_lengths.unsqueeze(1)
        trajectory_rows = torch.arange(
            int(env_ids.numel()), dtype=torch.long, device=self.device,
        ).unsqueeze(1).expand_as(time_indices)[valid]
        time_indices = time_indices[valid]
        wrapped = cursors + counts >= safe_lengths
        self._archive_transition_cursor[env_ids, slot_ids] = (
            cursors + counts
        ) % safe_lengths
        return trajectory_rows, time_indices, counts, wrapped

    def sample_sil_aligned_batch(self, replay_context):
        source_idx = replay_context.get("source_idx")
        source_counts = replay_context.get("source_counts")
        if source_idx is None or source_counts is None:
            source_idx, source_counts = self._build_sil_source_indices()
        del source_counts

        source_idx = self._aligned_replay_source_indices(replay_context, source_idx)
        env_ids, slot_ids = self._aligned_source_trajectories(source_idx)
        if int(slot_ids.numel()) == 0:
            return None
        transition_selection = getattr(self.args, "sil_transition_selection", "window")
        positive_selection = transition_selection in {
            "least_selected_positive", "gap_proportional_positive",
            "gap_proportional_replacement", "global_gap_proportional_positive",
        }
        global_transition_selection = transition_selection == "global_gap_proportional_positive"
        if not positive_selection:
            trajectory_rows, time_idx, sample_counts, wrapped = self._aligned_transition_indices(env_ids, slot_ids)
            if int(time_idx.numel()) == 0:
                return None

        lengths = self._archive_length[env_ids, slot_ids]
        task_ids = self._archive_task_id[env_ids, slot_ids]
        schedule_context = replay_context.get("schedule_context")
        reward_seq = self._archive_base_reward[env_ids, slot_ids].clone()
        if schedule_context is not None and float(schedule_context.get("r_epstime_scale", 0.0)) > 0.0:
            terminal_idx = lengths - 1
            reward_seq[
                torch.arange(int(env_ids.numel()), dtype=torch.long, device=self.device),
                terminal_idx,
            ] += self._timeaware_bonus_for_batch(env_ids, slot_ids, task_ids, schedule_context)

        if self.args.norm_rew:
            reward_normalizer = replay_context.get("reward_normalizer")
            if reward_normalizer is None:
                raise ValueError("SIL replay requires a reward_normalizer when norm_rew=True.")
            flat_task_ids = task_ids.unsqueeze(1).expand(-1, self._archive_max_len).reshape(-1)
            reward_seq = reward_normalizer(
                reward_seq.reshape(-1), flat_task_ids,
            ).view(int(env_ids.numel()), self._archive_max_len).to(
                dtype=torch.float32, device=self.device,
            )

        bootstrap_values, bootstrap_active = self._fallback_bootstrap_values(
            env_ids, slot_ids, task_ids, replay_context, schedule_context, return_mask=True,
        )
        observed_returns = self._discounted_returns_from_rewards(reward_seq, lengths)
        returns = self._discounted_returns_from_rewards(reward_seq, lengths, bootstrap_values)

        if positive_selection:
            from core.training.algo.tsil.positive_sampling import screen_and_select
            trajectory_rows, time_idx, sample_counts, wrapped = screen_and_select(
                self, env_ids, slot_ids, returns, task_ids, schedule_context, replay_context)

        flat_env_ids = env_ids[trajectory_rows]
        flat_slot_ids = slot_ids[trajectory_rows]
        flat_task_ids = task_ids[trajectory_rows]
        obs = self._archive_obs[flat_env_ids, flat_slot_ids, time_idx].to(dtype=self.dtype).clone()
        state = self._archive_state[flat_env_ids, flat_slot_ids, time_idx].to(dtype=self.dtype).clone()
        action = self._archive_action[flat_env_ids, flat_slot_ids, time_idx].to(dtype=self.dtype)
        if schedule_context is not None:
            deadline = self._deadline_for_batch(env_ids, task_ids, schedule_context)
            self._relabel_public_deadline(obs, deadline[trajectory_rows])
            self._relabel_public_deadline(state, deadline[trajectory_rows])

        flat_count = int(time_idx.numel())
        trajectory_weights = self._trajectory_sil_weights(
            env_ids, slot_ids, sample_counts, flat_count, schedule_context,
        )
        if global_transition_selection:
            if bool(getattr(self.args, "sil_global_trajectory_weighting", False)):
                # Preserve TSIL's success-efficiency priority after selecting
                # transitions globally. Do not apply the inverse-count scale:
                # that scale would restore equal per-trajectory loss mass and
                # partly undo the global-transition ablation.
                global_priorities = self._trajectory_replay_priorities(
                    env_ids, slot_ids, schedule_context,
                )
                step_weights = global_priorities[trajectory_rows]
            else:
                step_weights = torch.ones(flat_count, dtype=self.dtype, device=self.device)
        else:
            step_weights = trajectory_weights[trajectory_rows]
        success = self._archive_success[env_ids, slot_ids]
        fastest_membership = torch.zeros_like(success)
        if self._replay_source() == "fastest":
            fastest_membership = (
                slot_ids.unsqueeze(1) == self._archive_fastest_idx[env_ids]
            ).any(dim=1)
        unique_env_count = max(int(torch.unique(env_ids).numel()), 1)

        return {
            "obs": obs,
            "state": state,
            "action": action,
            "returns": returns[trajectory_rows, time_idx].to(dtype=self.dtype),
            "observed_returns": observed_returns[trajectory_rows, time_idx].to(dtype=self.dtype),
            "step_indices": time_idx,
            "trajectory_lengths": lengths[trajectory_rows],
            "time_targets": self._remaining_time_targets_for_batch(
                lengths[trajectory_rows], time_idx, schedule_context,
            ),
            "episode_times": self._archive_eps_time[env_ids, slot_ids][trajectory_rows].to(dtype=self.dtype),
            "task_ids": flat_task_ids,
            "sil_weights": step_weights.to(dtype=self.dtype),
            **replay_context.pop("_replacement_per_fields", {}),
            "sil_bootstrap_values": bootstrap_values[trajectory_rows].to(dtype=self.dtype),
            "sil_bootstrap_mask": bootstrap_active[trajectory_rows],
            "sil_success_mask": success[trajectory_rows],
            "sil_fallback_mask": (~success)[trajectory_rows],
            "sil_return_fill_mask": (~fastest_membership)[trajectory_rows],
            "env_ids": flat_env_ids,
            "slot_ids": flat_slot_ids,
            "time_idx": time_idx,
            "trajectory_count": int(env_ids.numel()),
            "selection_env_step_quota": 0,
            **replay_context.get("_success_target_stats", {}),
            "aligned_candidate_envs": float(unique_env_count),
            "aligned_selected_trajectories": float(int(env_ids.numel())),
            "aligned_dataset_steps": float(flat_count),
            "aligned_mean_trajectories_per_env": float(int(env_ids.numel()) / unique_env_count),
            "aligned_fastest_source_frac": float(fastest_membership.float().mean().item()),
            "aligned_return_fill_frac": float((~fastest_membership).float().mean().item()),
            "aligned_short_trajectory_frac": float((lengths < int(getattr(
                self.args, "sil_replay_transitions_per_trajectory", 1,
            ))).float().mean().item()),
            "aligned_transition_wrap_frac": float(wrapped.float().mean().item()),
            "aligned_replay_epochs": float(int(getattr(self.args, "sil_replay_epochs", 1))),
        }

    def sample_sil_trajectory_batch(self, replay_context):
        transition_budget = max(int(replay_context.get("_sil_sample_batch_size", getattr(self.args, "sil_batch_size", 1))), 1)
        source_idx = replay_context.get("source_idx")
        source_counts = replay_context.get("source_counts")
        if source_idx is None or source_counts is None:
            source_idx, source_counts = self._build_sil_source_indices()

        if self._use_global_fastest_mix():
            env_ids, slot_ids = self._sample_global_fastest_mix(
                replay_context,
                transition_budget,
                sample_unit="trajectory",
                transition_budget=transition_budget,
            )
        else:
            env_ids, slot_ids, _ = self._sample_source_trajectories_until_budget(
                source_idx,
                source_counts,
                transition_budget,
                "_round_robin_cursor",
            )
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
            self._relabel_public_deadline(obs_seq, deadline.unsqueeze(1))
            self._relabel_public_deadline(state_seq, deadline.unsqueeze(1))
        else:
            deadline = None

        reward_seq = self._archive_base_reward[env_ids, slot_ids].clone()
        if schedule_context is not None and float(schedule_context.get("r_epstime_scale", 0.0)) > 0.0:
            terminal_idx = lengths - 1
            reward_seq[torch.arange(int(env_ids.numel()), dtype=torch.long, device=self.device), terminal_idx] += (
                self._timeaware_bonus_for_batch(env_ids, slot_ids, task_ids, schedule_context)
            )

        if self.args.norm_rew:
            reward_normalizer = replay_context.get("reward_normalizer")
            if reward_normalizer is None:
                raise ValueError("SIL replay requires a reward_normalizer when norm_rew=True.")
            flat_task_ids = task_ids.unsqueeze(1).expand(-1, self._archive_max_len).reshape(-1)
            reward_seq = reward_normalizer(
                reward_seq.reshape(-1),
                flat_task_ids,
            ).view(int(env_ids.numel()), self._archive_max_len).to(dtype=torch.float32, device=self.device)

        bootstrap_values, bootstrap_active = self._fallback_bootstrap_values(
            env_ids, slot_ids, task_ids, replay_context, schedule_context, return_mask=True,
        )
        observed_returns = self._discounted_returns_from_rewards(reward_seq, lengths)
        returns = self._discounted_returns_from_rewards(reward_seq, lengths, bootstrap_values)
        traj_weights = self._trajectory_sil_weights(env_ids, slot_ids, lengths, flat_count, schedule_context)
        step_weights = traj_weights.unsqueeze(1).expand_as(reward_seq)[valid].to(dtype=self.dtype)
        success = self._archive_success[env_ids, slot_ids]
        selection_env_step_quota = 0
        if self._use_global_fastest_mix():
            selection_env_step_quota = self._resolved_env_step_quota(
                replay_context.get("success_env_ids", torch.empty(0, dtype=torch.long, device=self.device)),
                replay_context.get("fallback_env_ids", torch.empty(0, dtype=torch.long, device=self.device)),
                transition_budget,
            )
        time_targets = self._remaining_time_targets_for_batch(
            lengths.unsqueeze(1).expand(-1, self._archive_max_len)[valid],
            time_grid.expand(int(env_ids.numel()), -1)[valid],
            schedule_context,
        )

        return {
            "obs": obs_seq[valid],
            "state": state_seq[valid],
            "action": action_seq[valid],
            "returns": returns[valid].to(dtype=self.dtype),
            "observed_returns": observed_returns[valid].to(dtype=self.dtype),
            "time_targets": time_targets,
            "episode_times": self._archive_eps_time[env_ids, slot_ids].unsqueeze(1).expand_as(reward_seq)[valid].to(
                dtype=self.dtype,
            ),
            "task_ids": task_ids.unsqueeze(1).expand(-1, self._archive_max_len)[valid],
            "sil_weights": step_weights,
            "sil_bootstrap_values": bootstrap_values.unsqueeze(1).expand_as(reward_seq)[valid].to(dtype=self.dtype),
            "sil_bootstrap_mask": bootstrap_active.unsqueeze(1).expand_as(reward_seq)[valid],
            "sil_success_mask": success.unsqueeze(1).expand(-1, self._archive_max_len)[valid],
            "sil_fallback_mask": (~success).unsqueeze(1).expand(-1, self._archive_max_len)[valid],
            "env_ids": env_ids.unsqueeze(1).expand(-1, self._archive_max_len)[valid],
            "slot_ids": slot_ids.unsqueeze(1).expand(-1, self._archive_max_len)[valid],
            "time_idx": time_grid.expand(int(env_ids.numel()), -1)[valid],
            "trajectory_count": int(env_ids.numel()),
            "selection_env_step_quota": int(selection_env_step_quota),
        }

    def sample_sil_batch(self, replay_context):
        if self._use_update_scope_replay(replay_context):
            return self._next_update_scope_replay_slice(replay_context)
        if getattr(self, "_global_per", None) is not None:
            return self._global_per.sample(self, replay_context)

        if bool(getattr(self.args, "sil_aligned_replay", False)):
            return self.sample_sil_aligned_batch(replay_context)

        if self._sil_sample_unit() == "trajectory":
            return self._shuffle_flat_sil_batch(self.sample_sil_trajectory_batch(replay_context))

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
            # SIL keeps archived elapsed time and relabels deadline-derived fields.
            self._relabel_public_deadline(obs, deadline)
            self._relabel_public_deadline(state, deadline)
        else:
            deadline = None

        reward_seq = self._archive_base_reward[env_ids, slot_ids].clone()
        if schedule_context is not None and float(schedule_context.get("r_epstime_scale", 0.0)) > 0.0:
            terminal_idx = lengths - 1
            reward_seq[torch.arange(batch_size, dtype=torch.long, device=self.device), terminal_idx] += (
                self._timeaware_bonus_for_batch(env_ids, slot_ids, task_ids, schedule_context)
            )

        if self.args.norm_rew:
            reward_normalizer = replay_context.get("reward_normalizer")
            if reward_normalizer is None:
                raise ValueError("SIL replay requires a reward_normalizer when norm_rew=True.")
            flat_task_ids = task_ids.unsqueeze(1).expand(-1, self._archive_max_len).reshape(-1)
            reward_seq = reward_normalizer(
                reward_seq.reshape(-1),
                flat_task_ids,
            ).view(batch_size, self._archive_max_len).to(dtype=torch.float32, device=self.device)

        bootstrap_values, bootstrap_active = self._fallback_bootstrap_values(
            env_ids, slot_ids, task_ids, replay_context, schedule_context, return_mask=True,
        )
        observed_returns = self._discounted_returns_from_rewards(reward_seq, lengths)
        returns = self._discounted_returns_from_rewards(reward_seq, lengths, bootstrap_values)
        returns_batch = returns[torch.arange(batch_size, dtype=torch.long, device=self.device), time_idx].to(dtype=self.dtype)
        observed_returns_batch = observed_returns[
            torch.arange(batch_size, dtype=torch.long, device=self.device),
            time_idx,
        ].to(dtype=self.dtype)
        time_targets = self._remaining_time_targets_for_batch(lengths, time_idx, schedule_context)

        sil_weights = torch.ones((batch_size,), dtype=self.dtype, device=self.device)
        success = self._archive_success[env_ids, slot_ids]
        if self._use_global_fastest_mix():
            success_weight = self._success_replay_weights(env_ids, slot_ids, task_ids, schedule_context)
            sil_weights = torch.where(
                success,
                success_weight.to(dtype=self.dtype),
                sil_weights,
            )
        elif self._replay_source() == "fastest":
            sil_weights = self._fastest_trajectory_weights(
                env_ids, slot_ids, task_ids, schedule_context,
            ).to(dtype=self.dtype)

        return self._shuffle_flat_sil_batch({
            "obs": obs,
            "state": state,
            "action": action,
            "returns": returns_batch,
            "observed_returns": observed_returns_batch,
            "time_targets": time_targets,
            "episode_times": self._archive_eps_time[env_ids, slot_ids].to(dtype=self.dtype),
            "task_ids": task_ids,
            "sil_weights": sil_weights,
            "sil_bootstrap_values": bootstrap_values.to(dtype=self.dtype),
            "sil_bootstrap_mask": bootstrap_active,
            "sil_success_mask": success,
            "sil_fallback_mask": ~success,
            "env_ids": env_ids,
            "slot_ids": slot_ids,
            "time_idx": time_idx,
            "selection_env_step_quota": 0,
        })
