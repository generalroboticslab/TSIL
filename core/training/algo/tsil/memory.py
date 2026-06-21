"""TSIL trajectory memory for rollout capture and top-k archive storage."""

import json
import os
import time
from collections import deque

import numpy as np
import torch

from core.agents.utils import update_tensor_buffer, save_checkpoint
from core.common.io import save_json
from core.training.algo.tsil.sampling import _TsilMemorySamplingMixin


class TsilTrajectoryMemory(_TsilMemorySamplingMixin):
    """Manages per-env TSIL trajectory memory and optional env0 debug recording."""

    def __init__(self, args, unique_task_ids, device, envs, storage):
        self.args = args
        self.unique_task_ids = unique_task_ids
        self.device = device

        self.trajectory_archive = None
        self.sil_recording_enabled = bool(getattr(args, "use_sil", False))
        self.debug_recording_enabled = bool(
            getattr(args, "record_env0_trajectory", False)
            and getattr(args, "saving", False)
        )
        self.archive_enabled = False
        self.episode_start_snapshots = None
        self.episode_initial_obs = None
        self.episode_initial_state = None
        self._step_chunks = []
        self._recorded_step_count = 0
        self._reward_raw_cumsum = None
        self._reward_norm_cumsum = None
        self._episode_start_chunk_idx = None
        self._episode_start_step_count = None
        self._episode_start_reward_raw = None
        self._episode_start_reward_norm = None
        self._candidate_counter = 0
        self._debug_env0_dir = os.path.join(args.trajectory_dir, "episode_archives", "env0")
        self._debug_env0_first_path = None
        self._debug_env0_last_path = None
        self._debug_env0_count = 0
        self._deferred_debug_env_ids = set()
        self.dtype = getattr(storage, "dtype", storage.next_obs.dtype)
        self._archive_allocated = False
        self._archive_num_envs = 0
        self._archive_capacity = 0
        self._archive_max_len = 0
        self._archive_storage_dtype = torch.float16
        self._round_robin_cursor = 0
        self._success_round_robin_cursor = 0
        self._fallback_round_robin_cursor = 0
        self._archive_obs = None
        self._archive_state = None
        self._archive_terminal_next_state = None
        self._archive_action = None
        self._archive_base_reward = None
        self._archive_task_id = None
        self._archive_length = None
        self._archive_success = None
        self._archive_timeout = None
        self._archive_eps_time = None
        self._archive_global_step = None
        self._archive_return_score = None
        self._archive_valid = None
        self._archive_fastest_idx = None
        self._archive_return_idx = None
        self._fastest_anchor_eps_time = None
        self._fastest_anchor_steps = None

        if self.debug_recording_enabled and not hasattr(envs, "export_replay_snapshot"):
            print("[trajectory_archive] Env0 debug export disabled because the environment does not support replay snapshots.")
            self.debug_recording_enabled = False

        if self.debug_recording_enabled:
            from core.common.trajectory_archive import PerTaskTrajectoryArchive

            self.trajectory_archive = PerTaskTrajectoryArchive(
                args.trajectory_dir,
            )
            if hasattr(envs, "enable_replay_reset_snapshot_capture"):
                envs.enable_replay_reset_snapshot_capture([0])

        if not self.sil_recording_enabled and not self.debug_recording_enabled:
            return

        self.archive_enabled = True
        self.episode_start_snapshots = [None for _ in range(args.num_envs)]
        self.episode_initial_obs = torch.empty(
            (args.num_envs,) + tuple(storage.next_obs.shape[1:]),
            dtype=torch.float16,
            device=device,
        )
        self.episode_initial_state = torch.empty(
            (args.num_envs,) + tuple(storage.next_state.shape[1:]),
            dtype=torch.float16,
            device=device,
        )
        self._reward_raw_cumsum = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
        self._reward_norm_cumsum = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
        self._episode_start_chunk_idx = torch.zeros(args.num_envs, dtype=torch.long, device=device)
        self._episode_start_step_count = torch.zeros(args.num_envs, dtype=torch.long, device=device)
        self._episode_start_reward_raw = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
        self._episode_start_reward_norm = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
        self.initialize_episode_recorders(
            torch.arange(args.num_envs, device=device), storage, envs,
        )

    # ------------------------------------------------------------------
    # Episode start tracking
    # ------------------------------------------------------------------

    @staticmethod
    def _snapshot_to_numpy(snapshot):
        result = {}
        for key, value in snapshot.items():
            if isinstance(value, np.ndarray):
                result[key] = value.astype(np.float32, copy=False)
            else:
                result[key] = value
        return result

    def initialize_episode_recorders(self, env_ids, storage, envs, snapshot_overrides=None):
        if not self.archive_enabled or env_ids.numel() == 0:
            return

        snapshot_overrides = snapshot_overrides or {}
        env_ids_cpu = env_ids.detach().cpu().tolist()
        env_ids_device = env_ids.to(device=self.device, dtype=torch.long)
        self.episode_initial_obs[env_ids_device] = (
            storage.next_obs[env_ids_device].detach().to(dtype=torch.float16)
        )
        self.episode_initial_state[env_ids_device] = (
            storage.next_state[env_ids_device].detach().to(dtype=torch.float16)
        )
        self._episode_start_chunk_idx[env_ids_device] = len(self._step_chunks)
        self._episode_start_step_count[env_ids_device] = int(self._recorded_step_count)
        self._episode_start_reward_raw[env_ids_device] = self._reward_raw_cumsum[env_ids_device]
        self._episode_start_reward_norm[env_ids_device] = self._reward_norm_cumsum[env_ids_device]

        if self.debug_recording_enabled:
            for env_id in env_ids_cpu:
                if int(env_id) == 0:
                    snapshot = snapshot_overrides.get(int(env_id))
                    if snapshot is None:
                        snapshot = envs.export_replay_snapshot(env_id)
                    self.episode_start_snapshots[env_id] = self._snapshot_to_numpy(snapshot)
                else:
                    self.episode_start_snapshots[env_id] = None

        self._compact_step_chunks()

    def initialize_completed_episode_recorders(self, env_ids, storage, envs):
        if not self.archive_enabled or env_ids.numel() == 0:
            return
        if not self.debug_recording_enabled or not hasattr(envs, "pop_replay_reset_snapshots"):
            self.initialize_episode_recorders(env_ids, storage, envs)
            return

        env_ids_cpu = [int(env_id) for env_id in env_ids.detach().cpu().tolist()]
        deferred = {env_id for env_id in env_ids_cpu if env_id == 0}
        self._deferred_debug_env_ids.update(deferred)
        immediate = [env_id for env_id in env_ids_cpu if env_id not in deferred]
        if immediate:
            self.initialize_episode_recorders(
                torch.as_tensor(immediate, dtype=torch.long, device=self.device), storage, envs,
            )

    def _initialize_deferred_debug_recorders(self, storage, envs):
        if not self._deferred_debug_env_ids or envs is None or not hasattr(envs, "pop_replay_reset_snapshots"):
            return

        reset_snapshots = envs.pop_replay_reset_snapshots()
        ready = [env_id for env_id in sorted(self._deferred_debug_env_ids) if env_id in reset_snapshots]
        if not ready:
            return

        self._deferred_debug_env_ids.difference_update(ready)
        self.initialize_episode_recorders(
            torch.as_tensor(ready, dtype=torch.long, device=self.device),
            storage,
            envs,
            snapshot_overrides={env_id: reset_snapshots[env_id] for env_id in ready},
        )

    def _info_tensor(self, infos, key, like_tensor):
        value = infos.get(key)
        if value is None:
            return torch.zeros_like(like_tensor, dtype=torch.float32, device=self.device)
        if isinstance(value, torch.Tensor):
            return value.to(device=self.device, dtype=torch.float32)
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def record_rollout_step(self, step_idx, step_action, org_reward, reward,
                            org_cost, infos, storage, envs=None):
        if not self.archive_enabled:
            return

        reward_raw = org_reward.detach().to(device=self.device, dtype=torch.float32).clone()
        reward_norm = reward.detach().to(device=self.device, dtype=torch.float32).clone()
        timeaware_reward_bonus = self._info_tensor(
            infos, "timeaware_reward_bonus", org_reward,
        ).detach().clone()
        reward_without_timeaware = self._info_tensor(
            infos, "reward_without_timeaware", org_reward,
        ).detach().clone()
        if infos.get("reward_without_timeaware") is None:
            reward_without_timeaware = reward_raw - timeaware_reward_bonus
        next_timeout = getattr(storage, "next_timeout", torch.zeros_like(storage.next_done))
        chunk = {
            "obs": storage.obs[step_idx].detach().to(dtype=torch.float16).clone(),
            "state": storage.states[step_idx].detach().to(dtype=torch.float16).clone(),
            "next_state": storage.next_state.detach().to(dtype=torch.float16).clone(),
            "action": step_action.detach().to(device=self.device, dtype=torch.float16).clone(),
            "reward_raw": reward_raw,
            "reward_norm": reward_norm,
            "reward_without_timeaware": reward_without_timeaware,
            "timeaware_reward_bonus": timeaware_reward_bonus,
            "done": storage.next_done.detach().to(device=self.device, dtype=torch.bool).clone(),
            "timeout": next_timeout.detach().to(device=self.device, dtype=torch.bool).clone(),
            "scene_linvel": self._info_tensor(infos, "scene_linvel", org_reward).detach().clone(),
            "scene_linvel_penalty": self._info_tensor(infos, "scene_linvel_penalty", org_reward).detach().clone(),
            "scene_linacc_penalty": self._info_tensor(infos, "scene_linacc_penalty", org_reward).detach().clone(),
            "rob_qvel_norm": self._info_tensor(infos, "rob_qvel_norm", org_reward).detach().clone(),
        }
        if self.args.use_cost:
            chunk["cost"] = org_cost.detach().to(device=self.device, dtype=torch.float32).clone()

        self._step_chunks.append(chunk)
        self._recorded_step_count += 1
        self._reward_raw_cumsum += reward_raw
        self._reward_norm_cumsum += reward_norm
        self._initialize_deferred_debug_recorders(storage, envs)

    def _next_candidate_id(self):
        candidate_id = self._candidate_counter
        self._candidate_counter += 1
        return candidate_id

    @staticmethod
    def _metric_values(terminal_metrics, name, count, default=0.0):
        value = terminal_metrics.get(name)
        if value is None:
            return [default for _ in range(count)]
        return value.detach().cpu().tolist()

    def _stack_episode_field(self, candidate, field):
        env_id = int(candidate["env_id"])
        start = int(candidate["start_step"])
        end = int(candidate["end_step"])
        values = [chunk[field][env_id] for chunk in self._step_chunks[start:end]]
        if not values:
            return torch.empty((0,), dtype=torch.float32, device=self.device)
        return torch.stack(values, dim=0)

    @staticmethod
    def _tensor_to_numpy(tensor, dtype=None):
        array = tensor.detach().cpu().numpy()
        if dtype is None:
            return array
        return array.astype(dtype, copy=False)

    def _training_progress(self, global_step, global_update_iter):
        total_timesteps = getattr(self.args, "total_timesteps", None)
        if total_timesteps is not None and float(total_timesteps) > 0:
            return min(max(float(global_step) / float(total_timesteps), 0.0), 1.0)
        total_iters = getattr(self.args, "total_iters", None)
        if total_iters is not None and float(total_iters) > 0:
            return min(max(float(global_update_iter) / float(total_iters), 0.0), 1.0)
        return 1.0

    def _materialize_candidate(self, candidate):
        env_id = int(candidate["env_id"])
        reward_raw = self._stack_episode_field(candidate, "reward_raw").to(torch.float32)
        reward_norm = self._stack_episode_field(candidate, "reward_norm").to(torch.float32)

        obs = self._stack_episode_field(candidate, "obs")
        state = self._stack_episode_field(candidate, "state")
        action = self._stack_episode_field(candidate, "action")
        done = self._stack_episode_field(candidate, "done").to(torch.int8)
        initial_obs = candidate.get("initial_obs", self.episode_initial_obs[env_id])
        initial_state = candidate.get("initial_state", self.episode_initial_state[env_id])

        trajectory = {
            "initial_obs": self._tensor_to_numpy(initial_obs, np.float16),
            "initial_state": self._tensor_to_numpy(initial_state, np.float16),
            "obs": self._tensor_to_numpy(obs, np.float16),
            "state": self._tensor_to_numpy(state, np.float16),
            "action": self._tensor_to_numpy(action, np.float16),
            "reward": self._tensor_to_numpy(reward_raw, np.float32),
            "reward_raw": self._tensor_to_numpy(reward_raw, np.float32),
            "reward_norm": self._tensor_to_numpy(reward_norm, np.float32),
            "reward_without_timeaware": self._tensor_to_numpy(
                self._stack_episode_field(candidate, "reward_without_timeaware").to(torch.float32),
                np.float32,
            ),
            "timeaware_reward_bonus": self._tensor_to_numpy(
                self._stack_episode_field(candidate, "timeaware_reward_bonus").to(torch.float32),
                np.float32,
            ),
            "done": self._tensor_to_numpy(done, np.int8),
            "scene_linvel": self._tensor_to_numpy(self._stack_episode_field(candidate, "scene_linvel"), np.float32),
            "scene_linvel_penalty": self._tensor_to_numpy(self._stack_episode_field(candidate, "scene_linvel_penalty"), np.float32),
            "scene_linacc_penalty": self._tensor_to_numpy(self._stack_episode_field(candidate, "scene_linacc_penalty"), np.float32),
            "rob_qvel_norm": self._tensor_to_numpy(self._stack_episode_field(candidate, "rob_qvel_norm"), np.float32),
        }
        if self.args.use_cost:
            trajectory["cost"] = self._tensor_to_numpy(self._stack_episode_field(candidate, "cost"), np.float32)

        disk_record = {
            "summary": dict(candidate["summary"]),
            "snapshot": candidate["snapshot"],
            "trajectory": trajectory,
        }
        memory_record = {
            "obs": obs.to(dtype=torch.float32).detach(),
            "state": state.to(dtype=torch.float32).detach(),
            "action": action.to(dtype=torch.float32).detach(),
            "reward": reward_raw.to(dtype=torch.float32).detach(),
            "reward_without_timeaware": self._stack_episode_field(
                candidate, "reward_without_timeaware",
            ).to(dtype=torch.float32).detach(),
            "timeaware_reward_bonus": self._stack_episode_field(
                candidate, "timeaware_reward_bonus",
            ).to(dtype=torch.float32).detach(),
            "task_ids": torch.full(
                (int(action.shape[0]),),
                int(candidate["task_id"]),
                dtype=torch.long,
                device=self.device,
            ),
        }
        return disk_record, memory_record

    def _memory_record_for_sil(self, candidate):
        env_id = int(candidate["env_id"])
        action = self._stack_episode_field(candidate, "action")
        reward_raw = self._stack_episode_field(candidate, "reward_raw").to(torch.float32)
        next_state = self._stack_episode_field(candidate, "next_state").to(dtype=torch.float32)
        timeout = self._stack_episode_field(candidate, "timeout").to(dtype=torch.bool)
        return {
            "env_id": env_id,
            "task_id": int(candidate["task_id"]),
            "summary": dict(candidate["summary"]),
            "obs": self._stack_episode_field(candidate, "obs").to(dtype=torch.float32).detach(),
            "state": self._stack_episode_field(candidate, "state").to(dtype=torch.float32).detach(),
            "terminal_next_state": next_state[-1].detach(),
            "terminal_timeout": bool(timeout[-1].item()),
            "action": action.to(dtype=torch.float32).detach(),
            "reward": reward_raw.detach(),
            "reward_without_timeaware": self._stack_episode_field(
                candidate, "reward_without_timeaware",
            ).to(dtype=torch.float32).detach(),
            "timeaware_reward_bonus": self._stack_episode_field(
                candidate, "timeaware_reward_bonus",
            ).to(dtype=torch.float32).detach(),
            "task_ids": torch.full(
                (int(action.shape[0]),),
                int(candidate["task_id"]),
                dtype=torch.long,
                device=self.device,
            ),
        }

    # ------------------------------------------------------------------
    # SIL per-env padded archive
    # ------------------------------------------------------------------

    def _replay_source(self):
        return str(getattr(self.args, "sil_source", "fastest"))

    def _replay_topk(self):
        return max(int(getattr(self.args, "sil_topk", 1)), 1)

    def _sil_sample_unit(self):
        return str(getattr(self.args, "sil_sample_unit", "transition")).lower()

    def _success_sample_frac(self):
        return float(getattr(self.args, "sil_success_sample_frac", -1.0))

    def _use_global_fastest_mix(self):
        return self._replay_source() == "fastest" and self._success_sample_frac() >= 0.0

    @staticmethod
    def _archive_shapes_from_record(record):
        return (
            tuple(record["obs"].shape[1:]),
            tuple(record["state"].shape[1:]),
            tuple(record["action"].shape[1:]),
        )

    def _allocate_archive_tensors(self, num_envs, capacity, max_len, obs_shape, state_shape, action_shape):
        self._archive_num_envs = int(num_envs)
        self._archive_capacity = int(capacity)
        self._archive_max_len = int(max_len)
        shape_prefix = (self._archive_num_envs, self._archive_capacity, self._archive_max_len)
        self._archive_obs = torch.zeros(
            shape_prefix + tuple(obs_shape),
            dtype=self._archive_storage_dtype,
            device=self.device,
        )
        self._archive_state = torch.zeros(
            shape_prefix + tuple(state_shape),
            dtype=self._archive_storage_dtype,
            device=self.device,
        )
        slot_shape = (self._archive_num_envs, self._archive_capacity)
        self._archive_terminal_next_state = torch.zeros(
            slot_shape + tuple(state_shape),
            dtype=self._archive_storage_dtype,
            device=self.device,
        )
        self._archive_action = torch.zeros(
            shape_prefix + tuple(action_shape),
            dtype=self._archive_storage_dtype,
            device=self.device,
        )
        self._archive_base_reward = torch.zeros(shape_prefix, dtype=torch.float32, device=self.device)
        self._archive_task_id = torch.full(slot_shape, -1, dtype=torch.long, device=self.device)
        self._archive_length = torch.zeros(slot_shape, dtype=torch.long, device=self.device)
        self._archive_success = torch.zeros(slot_shape, dtype=torch.bool, device=self.device)
        self._archive_timeout = torch.zeros(slot_shape, dtype=torch.bool, device=self.device)
        self._archive_eps_time = torch.full(slot_shape, float("inf"), dtype=torch.float32, device=self.device)
        self._archive_global_step = torch.zeros(slot_shape, dtype=torch.long, device=self.device)
        self._archive_return_score = torch.full(slot_shape, -float("inf"), dtype=torch.float32, device=self.device)
        self._archive_valid = torch.zeros(slot_shape, dtype=torch.bool, device=self.device)
        topk_shape = (self._archive_num_envs, self._replay_topk())
        self._archive_fastest_idx = torch.full(topk_shape, -1, dtype=torch.long, device=self.device)
        self._archive_return_idx = torch.full(topk_shape, -1, dtype=torch.long, device=self.device)
        self._archive_allocated = True

    def _ensure_archive(self, env_id, record_length, record):
        obs_shape, state_shape, action_shape = self._archive_shapes_from_record(record)
        desired_num_envs = max(int(getattr(self.args, "num_envs", 0)), int(env_id) + 1, 1)
        desired_max_len = max(
            int(getattr(self.args, "episodeLength", 0)),
            int(getattr(self.args, "max_episode_length", 0)),
            int(getattr(self.args, "num_steps", 0)),
            int(record_length),
            1,
        )
        desired_capacity = 2 * self._replay_topk() + 1

        if not self._archive_allocated:
            self._allocate_archive_tensors(
                desired_num_envs,
                desired_capacity,
                desired_max_len,
                obs_shape,
                state_shape,
                action_shape,
            )
            return

        if int(env_id) < self._archive_num_envs and int(record_length) <= self._archive_max_len:
            return

        old_tensors = {
            "obs": self._archive_obs,
            "state": self._archive_state,
            "terminal_next_state": self._archive_terminal_next_state,
            "action": self._archive_action,
            "base_reward": self._archive_base_reward,
            "task_id": self._archive_task_id,
            "length": self._archive_length,
            "success": self._archive_success,
            "timeout": self._archive_timeout,
            "eps_time": self._archive_eps_time,
            "global_step": self._archive_global_step,
            "return_score": self._archive_return_score,
            "valid": self._archive_valid,
            "fastest_idx": self._archive_fastest_idx,
            "return_idx": self._archive_return_idx,
        }
        old_num_envs = self._archive_num_envs
        old_max_len = self._archive_max_len
        new_num_envs = max(self._archive_num_envs, desired_num_envs)
        new_max_len = max(self._archive_max_len, desired_max_len)

        self._allocate_archive_tensors(
            new_num_envs,
            self._archive_capacity,
            new_max_len,
            tuple(old_tensors["obs"].shape[3:]),
            tuple(old_tensors["state"].shape[3:]),
            tuple(old_tensors["action"].shape[3:]),
        )
        self._archive_obs[:old_num_envs, :, :old_max_len].copy_(old_tensors["obs"])
        self._archive_state[:old_num_envs, :, :old_max_len].copy_(old_tensors["state"])
        self._archive_terminal_next_state[:old_num_envs].copy_(old_tensors["terminal_next_state"])
        self._archive_action[:old_num_envs, :, :old_max_len].copy_(old_tensors["action"])
        self._archive_base_reward[:old_num_envs, :, :old_max_len].copy_(old_tensors["base_reward"])
        self._archive_task_id[:old_num_envs].copy_(old_tensors["task_id"])
        self._archive_length[:old_num_envs].copy_(old_tensors["length"])
        self._archive_success[:old_num_envs].copy_(old_tensors["success"])
        self._archive_timeout[:old_num_envs].copy_(old_tensors["timeout"])
        self._archive_eps_time[:old_num_envs].copy_(old_tensors["eps_time"])
        self._archive_global_step[:old_num_envs].copy_(old_tensors["global_step"])
        self._archive_return_score[:old_num_envs].copy_(old_tensors["return_score"])
        self._archive_valid[:old_num_envs].copy_(old_tensors["valid"])
        self._archive_fastest_idx[:old_num_envs].copy_(old_tensors["fastest_idx"])
        self._archive_return_idx[:old_num_envs].copy_(old_tensors["return_idx"])

    def _base_return_score(self, rewards):
        running = torch.zeros((), dtype=torch.float32, device=self.device)
        gamma = float(self.args.gamma)
        for idx in range(int(rewards.shape[0]) - 1, -1, -1):
            running = rewards[idx].to(device=self.device, dtype=torch.float32) + gamma * running
        return running

    def _archive_should_admit(self, env_id, success, eps_time, return_score):
        topk = self._replay_topk()
        return_slots = self._archive_return_idx[env_id]
        valid_return = return_slots[return_slots >= 0]
        if int(valid_return.numel()) < topk:
            return True
        worst_return = self._archive_return_score[env_id, valid_return].min()
        if float(return_score) > float(worst_return.item()):
            return True

        if not bool(success):
            return False

        fastest_slots = self._archive_fastest_idx[env_id]
        valid_fastest = fastest_slots[fastest_slots >= 0]
        if int(valid_fastest.numel()) < topk:
            return True
        slowest_fastest = self._archive_eps_time[env_id, valid_fastest].max()
        return float(eps_time) < float(slowest_fastest.item())

    def _clear_archive_slots(self, env_id, slots):
        if int(slots.numel()) == 0:
            return
        self._archive_obs[env_id, slots].zero_()
        self._archive_state[env_id, slots].zero_()
        self._archive_terminal_next_state[env_id, slots].zero_()
        self._archive_action[env_id, slots].zero_()
        self._archive_base_reward[env_id, slots].zero_()
        self._archive_task_id[env_id, slots] = -1
        self._archive_length[env_id, slots] = 0
        self._archive_success[env_id, slots] = False
        self._archive_timeout[env_id, slots] = False
        self._archive_eps_time[env_id, slots] = float("inf")
        self._archive_global_step[env_id, slots] = 0
        self._archive_return_score[env_id, slots] = -float("inf")
        self._archive_valid[env_id, slots] = False

    def _refresh_env_rankings(self, env_id):
        topk = self._replay_topk()
        valid_slots = torch.nonzero(self._archive_valid[env_id], as_tuple=False).view(-1)
        self._archive_fastest_idx[env_id].fill_(-1)
        self._archive_return_idx[env_id].fill_(-1)
        if int(valid_slots.numel()) == 0:
            return

        return_scores = self._archive_return_score[env_id, valid_slots]
        return_order = torch.argsort(return_scores, descending=True)
        return_selected = valid_slots[return_order[:topk]]
        self._archive_return_idx[env_id, :int(return_selected.numel())] = return_selected

        success_slots = valid_slots[self._archive_success[env_id, valid_slots]]
        if int(success_slots.numel()) > 0:
            fastest_scores = self._archive_eps_time[env_id, success_slots]
            fastest_order = torch.argsort(fastest_scores, descending=False)
            fastest_selected = success_slots[fastest_order[:topk]]
            self._archive_fastest_idx[env_id, :int(fastest_selected.numel())] = fastest_selected

        referenced = torch.zeros((self._archive_capacity,), dtype=torch.bool, device=self.device)
        return_valid = self._archive_return_idx[env_id]
        return_valid = return_valid[return_valid >= 0]
        fastest_valid = self._archive_fastest_idx[env_id]
        fastest_valid = fastest_valid[fastest_valid >= 0]
        if int(return_valid.numel()) > 0:
            referenced[return_valid] = True
        if int(fastest_valid.numel()) > 0:
            referenced[fastest_valid] = True
        stale_slots = torch.nonzero(self._archive_valid[env_id] & ~referenced, as_tuple=False).view(-1)
        self._clear_archive_slots(env_id, stale_slots)

    def insert_sil_trajectory(self, record):
        if not getattr(self.args, "use_sil", False):
            return False

        action = record.get("action")
        if action is None or int(action.shape[0]) == 0:
            return False

        env_id = int(record.get("env_id", 0))
        base_reward = record.get("reward_without_timeaware")
        if base_reward is None:
            base_reward = record.get("reward")
        if base_reward is None or int(base_reward.shape[0]) == 0:
            return False
        training_reward = record.get("reward")
        if training_reward is None:
            training_reward = base_reward
        length = min(
            int(action.shape[0]),
            int(base_reward.shape[0]),
            int(training_reward.shape[0]),
            int(record["obs"].shape[0]),
            int(record["state"].shape[0]),
        )
        if length <= 0:
            return False

        self._ensure_archive(env_id, length, record)
        base_reward = base_reward[:length].to(device=self.device, dtype=torch.float32)
        training_reward = training_reward[:length].to(device=self.device, dtype=torch.float32)
        summary = dict(record.get("summary", {}))
        success = bool(summary.get("success", False))
        eps_time = float(summary.get("eps_time", length))
        return_score = self._base_return_score(training_reward)
        if not self._archive_should_admit(env_id, success, eps_time, return_score):
            return False

        free_slots = torch.nonzero(~self._archive_valid[env_id], as_tuple=False).view(-1)
        if int(free_slots.numel()) == 0:
            self._refresh_env_rankings(env_id)
            free_slots = torch.nonzero(~self._archive_valid[env_id], as_tuple=False).view(-1)
        if int(free_slots.numel()) == 0:
            return False

        slot = int(free_slots[0].item())
        self._clear_archive_slots(env_id, torch.tensor([slot], dtype=torch.long, device=self.device))
        self._archive_obs[env_id, slot, :length].copy_(
            record["obs"][:length].to(device=self.device, dtype=self._archive_storage_dtype)
        )
        self._archive_state[env_id, slot, :length].copy_(
            record["state"][:length].to(device=self.device, dtype=self._archive_storage_dtype)
        )
        self._archive_action[env_id, slot, :length].copy_(
            action[:length].to(device=self.device, dtype=self._archive_storage_dtype)
        )
        terminal_next_state = record.get("terminal_next_state")
        if terminal_next_state is not None:
            self._archive_terminal_next_state[env_id, slot].copy_(
                terminal_next_state.to(device=self.device, dtype=self._archive_storage_dtype)
            )
        terminal_timeout = record.get("terminal_timeout", False)
        if isinstance(terminal_timeout, torch.Tensor):
            terminal_timeout = bool(terminal_timeout.item())
        self._archive_timeout[env_id, slot] = bool(terminal_timeout)
        self._archive_base_reward[env_id, slot, :length].copy_(base_reward)
        self._archive_task_id[env_id, slot] = int(record.get("task_id", summary.get("task_id", 0)))
        self._archive_length[env_id, slot] = length
        self._archive_success[env_id, slot] = success
        self._archive_eps_time[env_id, slot] = eps_time
        self._archive_global_step[env_id, slot] = int(summary.get("global_step", 0))
        self._archive_return_score[env_id, slot] = return_score
        self._archive_valid[env_id, slot] = True
        if success and self._fastest_anchor_steps is None:
            self._fastest_anchor_eps_time = eps_time
            self._fastest_anchor_steps = int(summary.get("global_step", 0))
        self._refresh_env_rankings(env_id)
        return True

    def has_sil_demo(self):
        if not self._archive_allocated:
            return False
        return bool((self._archive_return_idx >= 0).any().item() or (self._archive_fastest_idx >= 0).any().item())

    def fastest_reference_summary(self):
        if not self._archive_allocated:
            return {"count": 0, "best_eps_time": None, "best_steps": None}

        valid = self._archive_fastest_idx >= 0
        if not bool(valid.any().item()):
            return {"count": 0, "best_eps_time": None, "best_steps": None}

        env_ids = torch.arange(self._archive_num_envs, device=self.device).unsqueeze(1).expand_as(self._archive_fastest_idx)[valid]
        slot_ids = self._archive_fastest_idx[valid]
        eps_times = self._archive_eps_time[env_ids, slot_ids]
        best_idx = int(torch.argmin(eps_times).item())
        best_env = env_ids[best_idx]
        best_slot = slot_ids[best_idx]
        return {
            "count": int(slot_ids.numel()),
            "best_eps_time": float(eps_times[best_idx].item()),
            "best_steps": int(self._archive_global_step[best_env, best_slot].item()),
        }

    def fastest_anchor_summary(self):
        return {
            "anchor_eps_time": self._fastest_anchor_eps_time,
            "anchor_steps": self._fastest_anchor_steps,
        }

    def _compact_step_chunks(self):
        if not self._step_chunks:
            return

        min_active_start = int(self._episode_start_chunk_idx.min().item())
        min_needed = min_active_start
        if min_needed <= 0:
            return

        del self._step_chunks[:min_needed]
        self._episode_start_chunk_idx -= int(min_needed)

    def _write_debug_env0_candidate(self, candidate):
        if not self.debug_recording_enabled or int(candidate["env_id"]) != 0:
            return False
        if candidate.get("snapshot") is None:
            return False

        if self.trajectory_archive is None:
            return False

        updated = self.trajectory_archive.consider_env0_debug_episode(candidate, self._materialize_candidate)
        if updated or self.trajectory_archive.dirty:
            self.trajectory_archive.save_index()

        self._debug_env0_count += 1
        return True

    # ------------------------------------------------------------------
    # Finalize & archive
    # ------------------------------------------------------------------

    def finalize_archived_episodes(self, terminal_ids, success_buf, terminal_metrics,
                                   task_indices, global_step, global_episodes, global_update_iter):
        if not self.archive_enabled or len(terminal_ids) == 0:
            return

        terminal_count = len(terminal_ids)
        terminal_ids_device = terminal_ids.to(device=self.device, dtype=torch.long)
        terminal_ids_cpu = terminal_ids_device.detach().cpu().tolist()
        task_ids_cpu = task_indices[terminal_ids_device].detach().cpu().tolist()
        success_cpu = success_buf.detach().cpu().tolist()
        episode_lengths = (
            int(self._recorded_step_count) - self._episode_start_step_count[terminal_ids_device]
        ).detach().cpu().tolist()
        reward_raw_sums = (
            self._reward_raw_cumsum[terminal_ids_device] - self._episode_start_reward_raw[terminal_ids_device]
        ).detach().cpu().tolist()
        reward_norm_sums = (
            self._reward_norm_cumsum[terminal_ids_device] - self._episode_start_reward_norm[terminal_ids_device]
        ).detach().cpu().tolist()
        eps_time = self._metric_values(terminal_metrics, "eps_time", terminal_count)
        eps_time_p = self._metric_values(terminal_metrics, "eps_time_p", terminal_count)
        eps_max_scevel = self._metric_values(terminal_metrics, "eps_max_scevel", terminal_count)
        eps_sum_inst = self._metric_values(terminal_metrics, "eps_sum_inst", terminal_count)

        base_global_episodes = int(global_episodes) - terminal_count
        candidates = []
        for local_idx, env_id in enumerate(terminal_ids_cpu):
            task_id = int(task_ids_cpu[local_idx])
            snapshot = self.episode_start_snapshots[env_id]
            episode_length = int(episode_lengths[local_idx])
            if episode_length <= 0:
                continue

            summary = {
                "task_id": task_id,
                "global_step": int(global_step),
                "global_episodes": int(base_global_episodes + local_idx + 1),
                "iteration": int(global_update_iter),
                "episode_length": int(episode_length),
                "success": bool(success_cpu[local_idx]),
                "episode_return_raw": float(reward_raw_sums[local_idx]),
                "episode_return_norm": float(reward_norm_sums[local_idx]),
                "eps_time": float(eps_time[local_idx]),
                "eps_time_p": float(eps_time_p[local_idx]),
                "eps_max_scevel": float(eps_max_scevel[local_idx]),
                "eps_sum_inst": float(eps_sum_inst[local_idx]),
                "training_progress": self._training_progress(global_step, global_update_iter),
            }
            candidates.append({
                "candidate_id": self._next_candidate_id(),
                "order": local_idx,
                "task_id": task_id,
                "env_id": env_id,
                "start_step": int(self._episode_start_chunk_idx[env_id].item()),
                "end_step": len(self._step_chunks),
                "initial_obs": self.episode_initial_obs[env_id].detach().clone(),
                "initial_state": self.episode_initial_state[env_id].detach().clone(),
                "summary": summary,
                "snapshot": snapshot,
            })

        if not candidates:
            return

        if self.args.use_sil:
            for candidate in candidates:
                self.insert_sil_trajectory(self._memory_record_for_sil(candidate))

        if self.debug_recording_enabled:
            for candidate in candidates:
                self._write_debug_env0_candidate(candidate)

    def flush_pending_records(self):
        if not self.archive_enabled:
            return False

        self._compact_step_chunks()
        return False

    def refresh_archive_meta(self, meta_data=None):
        if meta_data is None:
            return
        sil_trajectories = int(self._archive_valid.sum().item()) if self._archive_allocated else 0
        sil_success_envs = (
            int((self._archive_valid & self._archive_success).any(dim=1).sum().item())
            if self._archive_allocated else 0
        )
        meta_data["trajectory_archive"] = {
            "sil_enabled": int(self.sil_recording_enabled),
            "sil_trajectories": sil_trajectories,
            "sil_success_envs": sil_success_envs,
            "env0_debug_enabled": int(self.debug_recording_enabled),
            "env0_debug_episodes": int(self._debug_env0_count),
            "env0_debug_dir": (
                os.path.relpath(self.trajectory_archive.archive_dir, self.args.trajectory_dir)
                if self.debug_recording_enabled else None
            ),
            "env0_debug_index": "archive_index.json" if self.debug_recording_enabled else None,
        }

    def num_tasks_with_success_archive(self):
        if not self._archive_allocated:
            return 0
        success_slots = self._archive_valid & self._archive_success
        if not bool(success_slots.any().item()):
            return 0
        archived_task_ids = {
            int(task_id)
            for task_id in self._archive_task_id[success_slots].detach().cpu().tolist()
            if int(task_id) >= 0
        }
        tracked_task_ids = {int(tid.item()) for tid in self.unique_task_ids}
        return len(archived_task_ids & tracked_task_ids)


__all__ = ["TsilTrajectoryMemory"]
