"""Rollout collection component.

RolloutCollector is a standalone class that steps the environment and agent,
writes observations/actions/rewards into a RolloutStorage, and notifies the
trainer through explicit callbacks when episodes complete.

Dependencies are passed through the constructor and method parameters — there
are no hidden ``self.*`` fields inherited from a parent class.
"""

import time

import torch

from core.agents.utils import linear_amplifier


class RolloutCollector:
    """Collects rollout data from the environment.

    Constructor args
    ----------------
    envs : IsaacGym VecEnv
    args : training Args dataclass
    device, dtype : torch device and floating dtype
    task_indices : Tensor[num_envs] mapping each env to its task id
    """

    def __init__(self, envs, args, device, dtype, task_indices):
        self.envs = envs
        self.args = args
        self.device = device
        self.dtype = dtype
        self.task_indices = task_indices

        # Per-env step accumulators (reset when episodes end)
        self.episode_accm_metrics = ["eps_G", "eps_sum_rew", "eps_dense_return", "eps_sum_cost", "eps_scenevel_p", "eps_sceneacc_p", "eps_act_p"]
        self.episode_accm = {
            m: torch.zeros(args.num_envs, dtype=dtype, device=device)
            for m in self.episode_accm_metrics
        }
        self.step_gamma = torch.ones(args.num_envs, dtype=dtype, device=device)

        # Mutable counter synced by the trainer before/after collect()
        self.global_step = 0
        self._dense_reward_episode_keep_mask = None

        # Timing (readable after collect returns)
        self.env_step_time = 0.0
        self.rollout_time = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(self, agent, storage, reward_normalizer=None, cost_normalizer=None,
                record_step_fn=None, on_episodes_done_fn=None,
                update_reward_stats=True):
        """Collect ``args.num_steps`` of rollout data.

        Parameters
        ----------
        agent : rollout agent
        storage : RolloutStorage  — data is written here
        reward_normalizer : optional per-task reward normalizer
        cost_normalizer : optional cost normalizer
        record_step_fn : ``fn(step_idx, step_action, org_reward, reward, org_cost, infos)``
            Called every step when trajectory archiving is enabled.
        on_episodes_done_fn : ``fn(terminal_ids, success_buf, terminal_metrics, infos)``
            Called when one or more episodes finish.  The trainer implements
            this to handle metric tracking, DDL updates, and archive
            finalization.

        Returns
        -------
        initial_lstm_state or None
        """
        args = self.args

        with torch.no_grad():
            rollout_start_time = time.perf_counter()
            env_step_time = 0.0
            storage.rollout_completed_episode_success = {}
            storage.rollout_completed_episode_time = {}
            storage.rollout_completed_episode_dense_return = {}
            storage.rollout_completed_episode_task_id = {}
            storage.rollout_completed_episode_id_by_env = {}

            if args.use_lstm:
                initial_lstm_state = [s.clone() for s in storage.next_lstm_state]

            for step in range(args.num_steps):
                self.global_step += args.num_envs

                # ---- snapshot current env state into storage ----
                storage.obs[step] = storage.next_obs
                storage.states[step] = storage.next_state
                storage.dones[step] = storage.next_done
                storage.timeouts[step] = storage.next_timeout
                storage.step_episode_ids[step] = storage.current_episode_ids

                # ---- get action ----
                if args.random_policy:
                    step_action = torch.rand(
                        (args.num_envs, self.envs.num_actions), device=self.device,
                    )
                else:
                    rollout_obs = storage.next_obs
                    rollout_state = storage.next_state

                    if args.use_lstm:
                        for lstm_stor, lstm_st in zip(storage.lstm_state_storage, storage.next_lstm_state):
                            lstm_stor[step] = lstm_st
                        step_action, _, logprob, _, value, storage.next_lstm_state, value_c, value_t = (
                            agent.get_action_and_value(
                                rollout_obs, storage.next_lstm_state, storage.next_done, rollout_state,
                            )
                        )
                    else:
                        step_action, _, logprob, _, value, value_c, value_t = (
                            agent.get_action_and_value(rollout_obs, rollout_state)
                        )

                    storage.actions[step] = step_action
                    storage.logprobs[step] = logprob
                    storage.values[step] = value
                    if args.use_cost:
                        storage.values_c[step] = value_c
                    if args.use_timeawareness:
                        storage.values_t[step] = value_t

                # ---- step environment ----
                env_step_start = time.perf_counter()
                next_obs_dict, reward, done, infos = self.envs.step(step_action)
                env_step_time += time.perf_counter() - env_step_start

                storage.next_obs = next_obs_dict["obs"].to(self.device)
                storage.next_state = next_obs_dict["states"].to(self.device)
                storage.next_done = done.to(self.device)
                if "time_outs" in infos:
                    storage.next_timeout = infos["time_outs"].to(self.device).float()

                # ---- track episode completions ----
                done_mask = storage.next_done.bool()
                if done_mask.any():
                    done_indices = done_mask.nonzero(as_tuple=True)[0]
                    success_flags = infos["success"][done_mask].bool().detach().cpu().tolist()
                    finished_episode_ids = storage.current_episode_ids[done_indices].detach().cpu().tolist()
                    eps_times = (
                        infos["eps_time"][done_mask].float().detach().cpu().tolist()
                        if "eps_time" in infos else [0.0] * len(done_indices)
                    )
                    env_ids = done_indices.detach().cpu().tolist()
                    task_ids = self.task_indices[done_indices].detach().cpu().tolist()
                    for env_id, task_id, episode_id, success, eps_time in zip(env_ids, task_ids, finished_episode_ids, success_flags, eps_times):
                        storage.rollout_completed_episode_success[int(episode_id)] = bool(success)
                        storage.rollout_completed_episode_time[int(episode_id)] = float(eps_time)
                        storage.rollout_completed_episode_task_id[int(episode_id)] = int(task_id)
                        storage.rollout_completed_episode_id_by_env[int(env_id)] = int(episode_id)

                    new_episode_ids = torch.arange(
                        storage.next_episode_uid,
                        storage.next_episode_uid + len(done_indices),
                        dtype=torch.long,
                        device=self.device,
                    )
                    storage.current_episode_ids[done_indices] = new_episode_ids
                    storage.next_episode_uid += len(done_indices)

                # ---- normalize rewards ----
                org_reward = self._shape_reward(reward.to(self.device), infos, done_mask=done_mask)
                if args.norm_rew and reward_normalizer is not None:
                    if update_reward_stats:
                        reward_normalizer.update_stats(org_reward, storage.next_done, self.task_indices)
                    reward = reward_normalizer(org_reward, self.task_indices)
                else:
                    reward = org_reward
                storage.rewards[step] = reward
                dense_reward = self._dense_signal_reward(org_reward, infos)

                # ---- costs ----
                org_cost = None
                if args.use_cost:
                    org_cost = infos["cost"].to(self.device)
                    cost = (
                        cost_normalizer.normalize(org_cost, storage.next_done)
                        if args.norm_cost and cost_normalizer is not None
                        else org_cost
                    )
                    storage.costs[step] = cost

                # ---- archive recording callback ----
                if record_step_fn is not None:
                    record_step_fn(
                        step_idx=step,
                        step_action=step_action,
                        org_reward=org_reward,
                        reward=reward,
                        org_cost=org_cost,
                        infos=infos,
                    )

                # ---- per-step accumulation + episode completion callback ----
                self._process_step_stats(
                    org_reward, dense_reward, org_cost, infos, done_mask, on_episodes_done_fn,
                )

            self.env_step_time = env_step_time
            self.rollout_time = time.perf_counter() - rollout_start_time

            return initial_lstm_state if args.use_lstm else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _shape_reward(self, reward, infos, done_mask=None):
        args = self.args
        shaped = reward
        dense_reward_scale = getattr(args, "dense_reward_scale", [1.0, 1.0])
        dense_reward_mode = str(getattr(args, "dense_reward_mode", "none")).lower()

        dense_scale = 1.0
        if dense_reward_mode == "soft":
            dense_scale = linear_amplifier(
                *dense_reward_scale,
                self.global_step,
                args.total_timesteps,
                args.curri_rate,
            )
        elif dense_reward_mode == "hard":
            hard_clip = float(getattr(args, "hard_clip", 0.5))
            progress = self.global_step / max(float(args.total_timesteps), 1.0)
            dense_scale = dense_reward_scale[0] if progress < hard_clip else dense_reward_scale[1]

        if dense_scale != 1.0:
            success = infos["success"].to(self.device).float()
            sparse_reward = success * float(args.successRewardScale)
            shaped = sparse_reward + float(dense_scale) * (shaped - sparse_reward)

        if "dense_reward" in infos:
            dense_reward = infos["dense_reward"].to(self.device).float()
            dense_keep_mask = None
            dense_dropout_p = float(
                getattr(
                    args,
                    "effective_dense_reward_dropout_p",
                    getattr(args, "dense_reward_dropout_p", 0.0),
                )
            )
            dropout_mode = str(getattr(args, "dense_reward_dropout_mode", "step")).lower()
            if dense_dropout_p > 0.0:
                if dropout_mode == "episode":
                    dense_keep_mask = self._dense_reward_episode_keep(dense_reward, dense_dropout_p)
                else:
                    dense_keep_mask = self._sample_dense_reward_keep(
                        dense_reward.shape, dense_reward.dtype, dense_reward.device, dense_dropout_p,
                    )

            if dense_keep_mask is not None:
                shaped = shaped - dense_reward + dense_keep_mask * dense_reward
                if "reward_without_timeaware" in infos:
                    infos["reward_without_timeaware"] = (
                        infos["reward_without_timeaware"].to(self.device).float()
                        - dense_reward
                        + dense_keep_mask * dense_reward
                    )
            if dropout_mode == "episode" and done_mask is not None and bool(done_mask.any().item()):
                self._resample_dense_reward_episode_keep(done_mask, dense_reward, dense_dropout_p)

        step_cost = float(getattr(args, "step_cost", 0.0))
        if step_cost > 0:
            shaped = shaped - step_cost

        return shaped

    def _sample_dense_reward_keep(self, shape, dtype, device, dropout_p):
        if dropout_p >= 1.0:
            return torch.zeros(shape, dtype=dtype, device=device)
        return (torch.rand(shape, device=device) >= dropout_p).to(dtype)

    def _dense_reward_episode_mask_shape(self, dense_reward):
        return (dense_reward.shape[0],) + (1,) * max(dense_reward.dim() - 1, 0)

    def _dense_reward_episode_keep(self, dense_reward, dropout_p):
        mask_shape = self._dense_reward_episode_mask_shape(dense_reward)
        if (
            self._dense_reward_episode_keep_mask is None
            or tuple(self._dense_reward_episode_keep_mask.shape) != tuple(mask_shape)
        ):
            self._dense_reward_episode_keep_mask = self._sample_dense_reward_keep(
                mask_shape, dense_reward.dtype, dense_reward.device, dropout_p,
            )
        return self._dense_reward_episode_keep_mask.to(dtype=dense_reward.dtype, device=dense_reward.device)

    def _resample_dense_reward_episode_keep(self, done_mask, dense_reward, dropout_p):
        mask = self._dense_reward_episode_keep(dense_reward, dropout_p)
        done_ids = done_mask.nonzero(as_tuple=True)[0]
        mask[done_ids] = self._sample_dense_reward_keep(
            mask[done_ids].shape, dense_reward.dtype, dense_reward.device, dropout_p,
        )

    def _dense_signal_reward(self, org_reward, infos):
        if "dense_reward" in infos:
            return infos["dense_reward"].to(self.device).float()
        success = infos.get("success", torch.zeros_like(org_reward)).to(self.device).float()
        time_bonus = infos.get("timeaware_reward_bonus", torch.zeros_like(org_reward)).to(self.device).float()
        return org_reward - success * float(self.args.successRewardScale) - time_bonus

    def _process_step_stats(self, org_reward, dense_reward, org_cost, infos, done_mask, on_episodes_done_fn):
        """Accumulate per-step stats and fire the episode-done callback."""
        self.episode_accm["eps_G"] += self.step_gamma * org_reward
        self.episode_accm["eps_sum_rew"] += org_reward
        self.episode_accm["eps_dense_return"] += self.step_gamma * dense_reward
        self.step_gamma *= self.args.gamma
        self.episode_accm["eps_sum_cost"] += (
            org_cost.sum(dim=-1) if self.args.use_cost and org_cost is not None else 0
        )
        self.episode_accm["eps_scenevel_p"] += infos.get("scene_linvel_penalty", 0)
        self.episode_accm["eps_sceneacc_p"] += infos.get("scene_linacc_penalty", 0)
        self.episode_accm["eps_act_p"] += infos.get("rob_qvel_norm", 0)

        terminal_nums = done_mask.sum()
        if terminal_nums > 0 and on_episodes_done_fn is not None:
            terminal_ids = done_mask.nonzero().flatten()
            success_buf = infos["success"][done_mask].bool()

            terminal_metrics = self._extract_terminal_metrics(done_mask, terminal_nums, infos)

            on_episodes_done_fn(
                terminal_ids=terminal_ids,
                success_buf=success_buf,
                terminal_metrics=terminal_metrics,
                infos=infos,
            )

            # Reset accumulators for completed episodes
            for key in self.episode_accm:
                self.episode_accm[key][done_mask] = 0.0
            self.step_gamma[done_mask] = 1.0

    def _extract_terminal_metrics(self, done_mask, terminal_nums, infos):
        """Extract metric values for completed episodes from accumulators and infos."""
        n = terminal_nums.item()
        metrics = {}

        # From step accumulators
        for m in self.episode_accm_metrics:
            metrics[m] = self.episode_accm[m][done_mask]

        # From env infos
        for m in ["success", "eps_time_p", "eps_time"]:
            if m in infos:
                metrics[m] = infos[m][done_mask].float().to(self.device)
            else:
                metrics[m] = torch.zeros(n, dtype=self.dtype, device=self.device)

        for m in ["eps_max_scevel", "eps_sum_inst"]:
            if m in infos:
                metrics[m] = infos[m][done_mask].float().to(self.device)
            else:
                metrics[m] = torch.zeros(n, dtype=self.dtype, device=self.device)

        return metrics

    @staticmethod
    def compute_time2end_upper_update(envs, args, env_ids, success_times):
        """Compute (optionally smoothed) deadline upper-bound updates."""
        device = env_ids.device
        success_times = success_times.to(device)
        current_upper = envs.obs_range["time2end"][1][env_ids]
        target_upper = torch.minimum(current_upper, success_times)
        if not args.anneal_ddl:
            return target_upper
        return torch.lerp(current_upper, target_upper, args.ddl_anneal_alpha)
__all__ = ["RolloutCollector"]
