"""Training metric state and checkpoint bookkeeping."""

import json
import os
import time
from collections import deque

import numpy as np
import torch

from core.agents.utils import update_tensor_buffer, linear_amplifier, save_checkpoint
from core.common.io import save_json


class MetricTracker:
    """Centralized training metric state and per-task performance tracking.

    Owns all global counters, per-task sliding-window buffers, best-metric
    trackers, signal/noise metrics, curriculum state, timing, and metadata.
    """

    def __init__(self, args, num_tasks, unique_task_ids, device, dtype,
                 max_eps_time, avg_time2end_upper, reward_settings=None):
        self.args = args
        self.num_tasks = num_tasks
        self.unique_task_ids = unique_task_ids
        self.device = device
        self.dtype = dtype

        # Global counters
        self.global_update_iter = 0
        self.skipped_update_iter = 0
        self.global_step = 0
        self.global_episodes = 0
        self.reward_update_iters = 0
        self.reward_steps = 0
        self.reward_episodes = 0

        # Per-task sliding window
        self.per_task_buf_len = max(args.running_len // num_tasks, 50)
        self.per_task_episode_count = torch.zeros(num_tasks, dtype=torch.long, device=device)
        self.per_task_success_count = torch.zeros(num_tasks, dtype=torch.long, device=device)

        self.per_task_metrics = ["eps_G", "eps_sum_rew", "eps_dense_return", "success", "eps_sum_cost", "eps_time_p", "eps_time"]
        self.success_only_metrics = ["eps_time_p", "eps_time"]

        self.per_task_buf = {}
        for metric in self.per_task_metrics:
            self.per_task_buf[metric] = torch.zeros((num_tasks, self.per_task_buf_len), dtype=dtype, device=device)

        self.per_task_avg = {m: torch.zeros(num_tasks, dtype=dtype, device=device) for m in self.per_task_metrics}
        self.avg_task_metrics = {m: 0.0 for m in self.per_task_metrics}

        # Best metrics
        self.cur_eps_G = -torch.inf
        self.cur_eps_G_score = -torch.inf
        self.cur_eps_sum_rew = 0.0
        self.cur_eps_sum_cost = 0.0
        self.cur_success_rate = 0.
        self.cur_eps_time = 0.
        self.best_eps_G_score = -torch.inf
        self.best_success_rate = 0.
        self.best_tail_success_rate = 0.
        self.max_eps_time = max_eps_time
        self.avg_time2end_upper = avg_time2end_upper

        # Timing
        self.iter_time_buf = []
        self.iter_time_window = 100
        self.training_start_time = None
        self.rollout_time = 0.0
        self.rollout_env_step_time = 0.0
        self.update_time = 0.0
        self.performance_metrics = {}

        # Curriculum
        self.curri_episodes = 0
        self.curri_steps = 0
        self.success_episodes = 0
        self.curri_update_iters = 0
        self.curri_ratio = args.init_curri_ratio
        task_id_ints = [int(tid.item()) for tid in unique_task_ids]
        self.curriculum_above_per_task = {tid: 0 for tid in task_id_ints}
        self.curriculum_below_per_task = {tid: 0 for tid in task_id_ints}
        self.curri_ratio_per_task = {tid: self.curri_ratio for tid in task_id_ints}
        self.ready_to_record = False
        self.avg_buffer_reset = True

        # Signal/noise metrics
        reward_settings = reward_settings or {}
        self.signal_metrics_window = int(args.signal_metrics_window)
        self.dense_tail_frac = float(args.dense_tail_frac)
        self.success_reward_bonus = float(reward_settings.get("r_success", 0.0))
        self.signal_episode_windows = [deque(maxlen=self.signal_metrics_window) for _ in range(num_tasks)]
        self.episode_adv_mass = {}
        self.adv_episode_window = deque(maxlen=self.signal_metrics_window)
        self.per_task_dense_tail_purity = torch.full((num_tasks,), torch.nan, dtype=dtype, device=device)
        self.per_task_dense_tail_window_size = torch.zeros(num_tasks, dtype=torch.long, device=device)
        self.per_task_dense_tail_k = torch.zeros(num_tasks, dtype=torch.long, device=device)
        self.current_signal_metrics = {
            "top10pct_Rdense_sr": 0.0,
            "dense_tail_window_size": 0,
            "dense_tail_k": 0,
            "succ_posadv_ratio": 0.0,
            "fast_succ_posadv_ratio": 0.0,
            "succ_posadv_step_frac": 0.0,
            "adv_ep_used": 0,
            "episode_signal_count": 0,
            "success_eps_time": None,
            "adv_episodes_used": 0,
            "adv_raw_mean": 0.0,
            "adv_raw_std": 0.0,
            "adv_raw_abs_mean": 0.0,
            "adv_raw_abs_max": 0.0,
            "adv_raw_abs_median": 0.0,
            "sil_fast_success_rate": 0.0,
            "sil_fast_success_count": 0,
            "fast_success_rate": 0.0,
            "fast_success_count": 0,
            "sil_revisit_window_episode_count": 0,
            "sil_first_revisit_steps": None,
            "sil_revisit_gap_steps": None,
            "first_fast_revisit_steps": None,
            "fast_revisit_gap_steps": None,
            "sil_archive_best_eps_time": None,
            "sil_revisit_anchor_eps_time": None,
            "sil_revisit_anchor_steps": None,
        }
        self.latest_signal_episodes = []
        self.latest_signal_episodes_iter = None
        self.last_episode_signal_history_iter = None
        self.sil_first_revisit_steps = None
        initial_replay_mode = args.sil_mode if args.use_sil else "disabled"
        self.last_replay_stats = self.default_replay_stats(mode=initial_replay_mode)

        # Curriculum values
        self.cur_ent = args.ent_coef[0]

        # Metadata
        self.meta_data = {"milestone": {}, "training_info": {}, "trajectory_archive": {}, "training_signal_metrics": {}}
        self.milestone = self.meta_data["milestone"]
        self.training_info = self.meta_data["training_info"]
        self.training_signal_info = self.meta_data["training_signal_metrics"]
        self.update_training_signal_summary()

        self.start_time = time.time()

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def default_replay_stats(mode="disabled"):
        return {
            "replay_mode": mode,
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

    # ------------------------------------------------------------------
    # Episode-done metric tracking
    # ------------------------------------------------------------------

    def on_episodes_done(self, terminal_ids, success_buf, terminal_metrics,
                         task_indices, tid_to_tidx):
        """Update per-task metric tracking when episodes complete.

        The DDL update, archive finalization, and signal recording are
        handled by the trainer callback — this method only manages the
        counters and sliding-window buffers.
        """
        success_ids = terminal_ids[success_buf]
        terminal_nums_cpu = len(terminal_ids)
        self.global_episodes += terminal_nums_cpu
        self.curri_episodes += terminal_nums_cpu
        self.success_episodes += len(success_ids)

        # Per-task metric tracking
        task_ids = task_indices[terminal_ids]
        task_idxs = tid_to_tidx[task_ids]

        for task_idx in task_idxs.unique():
            task_mask = task_idxs == task_idx
            num_finished = task_mask.sum()
            if num_finished == 0:
                continue

            task_success = success_buf & task_mask
            self.per_task_episode_count[task_idx] += num_finished.item()
            self.per_task_success_count[task_idx] += task_success.sum().long()

            for metric in self.per_task_metrics:
                if metric not in terminal_metrics:
                    continue

                if metric in self.success_only_metrics:
                    metric_mask = task_success
                    valid_len = self.per_task_success_count[task_idx].item()
                else:
                    metric_mask = task_mask
                    valid_len = self.per_task_episode_count[task_idx].item()

                metric_values = terminal_metrics[metric][metric_mask]
                if len(metric_values) == 0:
                    continue

                buf = self.per_task_buf[metric][task_idx]
                update_tensor_buffer(buf, metric_values)
                valid_len = min(valid_len, self.per_task_buf_len)
                self.per_task_avg[metric][task_idx] = buf[-valid_len:].mean()

        # Global averages
        for metric in self.per_task_metrics:
            self.avg_task_metrics[metric] = self.per_task_avg[metric].mean().item()

        self.cur_eps_G = self.avg_task_metrics["eps_G"]
        self.cur_eps_sum_rew = self.avg_task_metrics["eps_sum_rew"]
        self.cur_eps_sum_cost = self.avg_task_metrics["eps_sum_cost"]
        cost_penalty = self.args.successRewardScale * self.cur_eps_sum_cost if self.args.use_cost else 0.0
        self.cur_eps_G_score = self.cur_eps_G - cost_penalty
        self.cur_success_rate = self.avg_task_metrics["success"]
        self.cur_eps_time = self.avg_task_metrics.get("eps_time", 0.0)
        self.ready_to_record = self.curri_episodes > self.args.running_len

        self.training_info["last_episode"] = {
            "iterations": self.global_update_iter,
            "episodes": self.global_episodes,
            "steps": self.global_step,
            "avg_task_success_rate": self.avg_task_metrics["success"],
            "avg_task_eps_G": self.avg_task_metrics["eps_G"],
            "avg_task_eps_G_score": self.cur_eps_G_score,
            "avg_task_eps_sum_rew": self.avg_task_metrics["eps_sum_rew"],
            "avg_task_eps_dense_return": self.avg_task_metrics["eps_dense_return"],
            "avg_task_eps_sum_cost": self.avg_task_metrics["eps_sum_cost"],
            "avg_task_eps_time": self.avg_task_metrics["eps_time"],
            "avg_time2end_upper": self.avg_time2end_upper,
            "per_task_metrics": self.snapshot_per_task_metrics(),
        }

    # ------------------------------------------------------------------
    # Performance / timing
    # ------------------------------------------------------------------

    def update_performance_metrics(self, num_steps, num_envs):
        """Compute throughput and timing metrics for the latest update."""
        curr_frames = float(num_steps * num_envs)
        play_time = float(self.rollout_time)
        step_time = float(self.rollout_env_step_time)
        update_time = float(self.update_time)
        total_time = play_time + update_time

        def _safe_div(num, den):
            return float(num / den) if den > 0 else 0.0

        self.performance_metrics = {
            'performance/step_inference_rl_update_fps': _safe_div(curr_frames, total_time),
            'performance/step_inference_fps': _safe_div(curr_frames, play_time),
            'performance/step_fps': _safe_div(curr_frames, step_time),
            'performance/rl_update_time': update_time,
            'performance/step_inference_time': play_time,
            'performance/step_time': step_time,
        }

    def snapshot_per_task_metrics(self):
        """Capture the current per-task metric snapshot for metadata."""
        return {
            tid.item(): {metric: self.per_task_avg[metric][i].item() for metric in self.per_task_metrics}
            for i, tid in enumerate(self.unique_task_ids)
        }

    def track_iter_time(self, start_time):
        """Track time taken for each update to compute ETA."""
        iter_duration = time.perf_counter() - start_time
        self.iter_time_buf.append(iter_duration)
        if len(self.iter_time_buf) > self.iter_time_window:
            self.iter_time_buf.pop(0)

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    def update_curriculum(self, envs):
        """Update curriculum learning parameters (for vanilla policy training)."""
        self.cur_ent = linear_amplifier(*self.args.ent_coef, self.global_step, self.args.total_timesteps, self.args.curri_rate)
        envs.cfg['r_epstime_scale'] = linear_amplifier(*self.args.epstimeRewardScale, self.global_step, self.args.total_timesteps, self.args.curri_rate)
        envs.cfg['r_scene_vel_scale'] = linear_amplifier(*self.args.scevelRewardScale, self.global_step, self.args.total_timesteps, self.args.curri_rate)

        if self.args.scratch and self.ready_to_record and self.args.success_threshold > 0:
            for task_idx, tid in enumerate(self.unique_task_ids):
                task_id = int(tid.item())
                task_success = float(self.per_task_avg["success"][task_idx].item())
                if task_success >= self.args.success_threshold:
                    self.curriculum_above_per_task[task_id] += 1
                    self.curriculum_below_per_task[task_id] = 0
                    if self.curriculum_above_per_task[task_id] >= self.args.curri_hold_iters:
                        self.curri_ratio_per_task[task_id] = min(self.curri_ratio_per_task[task_id] + self.args.curriculum_step, 1.0)
                        envs.update_dr_params(self.curri_ratio_per_task[task_id], task_id=task_id)
                        self.curriculum_above_per_task[task_id] = 0
                else:
                    self.curriculum_below_per_task[task_id] += 1
                    self.curriculum_above_per_task[task_id] = 0
                    if self.curriculum_below_per_task[task_id] >= self.args.curri_hold_iters:
                        self.curri_ratio_per_task[task_id] = max(self.curri_ratio_per_task[task_id] - self.args.curriculum_step, 0.0)
                        envs.update_dr_params(self.curri_ratio_per_task[task_id], task_id=task_id)
                        self.curriculum_below_per_task[task_id] = 0

            self.curri_ratio = sum(self.curri_ratio_per_task.values()) / len(self.curri_ratio_per_task)

    # ------------------------------------------------------------------
    # Signal / noise metrics
    # ------------------------------------------------------------------

    def refresh_dense_tail_metrics(self):
        if not self.args.track_signal_metrics:
            return

        valid_purities = []
        total_window_size = 0
        total_k = 0
        for task_idx, episode_window in enumerate(self.signal_episode_windows):
            window_size = len(episode_window)
            self.per_task_dense_tail_window_size[task_idx] = window_size
            if window_size == 0:
                self.per_task_dense_tail_purity[task_idx] = torch.nan
                self.per_task_dense_tail_k[task_idx] = 0
                continue

            dense_returns = np.asarray([entry["dense_return"] for entry in episode_window], dtype=np.float32)
            success_flags = np.asarray([entry["success"] for entry in episode_window], dtype=np.float32)
            tail_k = max(1, int(np.ceil(window_size * self.dense_tail_frac)))
            top_indices = np.argsort(-dense_returns)[:tail_k]
            purity = float(success_flags[top_indices].mean()) if len(top_indices) > 0 else 0.0

            self.per_task_dense_tail_purity[task_idx] = purity
            self.per_task_dense_tail_k[task_idx] = tail_k
            valid_purities.append(purity)
            total_window_size += window_size
            total_k += tail_k

        self.current_signal_metrics["top10pct_Rdense_sr"] = float(np.mean(valid_purities)) if valid_purities else 0.0
        self.current_signal_metrics["dense_tail_window_size"] = int(total_window_size)
        self.current_signal_metrics["dense_tail_k"] = int(total_k)

    def refresh_success_eps_time_metric(self):
        if not self.args.track_signal_metrics:
            return

        success_eps_times = []
        for episode_window in self.signal_episode_windows:
            for entry in episode_window:
                if entry["success"]:
                    success_eps_times.append(entry["eps_time"])

        self.current_signal_metrics["success_eps_time"] = (
            float(np.mean(success_eps_times)) if success_eps_times else None
        )

    def refresh_sil_revisit_episode_metrics(self, tsil_memory, threshold_ratio=1.1):
        if not self.args.track_signal_metrics:
            return

        best_summary = (
            tsil_memory.fastest_reference_summary()
            if tsil_memory is not None and hasattr(tsil_memory, "fastest_reference_summary")
            else {"count": 0, "best_eps_time": None, "best_steps": None}
        )
        anchor_summary = (
            tsil_memory.fastest_anchor_summary()
            if tsil_memory is not None and hasattr(tsil_memory, "fastest_anchor_summary")
            else {"anchor_eps_time": None, "anchor_steps": None}
        )
        best_eps_time = best_summary.get("best_eps_time")
        anchor_eps_time = anchor_summary.get("anchor_eps_time")
        anchor_steps = anchor_summary.get("anchor_steps")
        window_records = [
            entry
            for episode_window in self.signal_episode_windows
            for entry in episode_window
        ]
        self.current_signal_metrics["sil_revisit_window_episode_count"] = int(len(window_records))
        self.current_signal_metrics["sil_archive_best_eps_time"] = best_eps_time
        self.current_signal_metrics["sil_revisit_anchor_eps_time"] = anchor_eps_time
        self.current_signal_metrics["sil_revisit_anchor_steps"] = anchor_steps

        if anchor_eps_time is None:
            self.current_signal_metrics["sil_fast_success_rate"] = 0.0
            self.current_signal_metrics["sil_fast_success_count"] = 0
            self.current_signal_metrics["sil_first_revisit_steps"] = self.sil_first_revisit_steps
            self.current_signal_metrics["sil_revisit_gap_steps"] = None
            self.current_signal_metrics["fast_success_rate"] = 0.0
            self.current_signal_metrics["fast_success_count"] = 0
            self.current_signal_metrics["first_fast_revisit_steps"] = self.sil_first_revisit_steps
            self.current_signal_metrics["fast_revisit_gap_steps"] = None
            return

        threshold = float(threshold_ratio) * float(anchor_eps_time)
        fast_successes = [
            entry
            for entry in window_records
            if bool(entry.get("success", False))
            and float(entry.get("eps_time", float("inf"))) <= threshold
            and (
                anchor_steps is None
                or entry.get("steps") is None
                or int(entry["steps"]) > int(anchor_steps)
            )
        ]
        first_steps = [
            int(entry["steps"])
            for entry in fast_successes
            if entry.get("steps") is not None
        ]
        if first_steps and self.sil_first_revisit_steps is None:
            self.sil_first_revisit_steps = min(first_steps)

        count = int(len(fast_successes))
        denom = max(int(len(window_records)), 1)
        self.current_signal_metrics["sil_fast_success_count"] = count
        self.current_signal_metrics["sil_fast_success_rate"] = float(count / denom)
        self.current_signal_metrics["sil_first_revisit_steps"] = self.sil_first_revisit_steps
        self.current_signal_metrics["sil_revisit_gap_steps"] = (
            int(self.sil_first_revisit_steps) - int(anchor_steps)
            if self.sil_first_revisit_steps is not None and anchor_steps is not None
            else None
        )
        self.current_signal_metrics["fast_success_count"] = self.current_signal_metrics["sil_fast_success_count"]
        self.current_signal_metrics["fast_success_rate"] = self.current_signal_metrics["sil_fast_success_rate"]
        self.current_signal_metrics["first_fast_revisit_steps"] = self.current_signal_metrics["sil_first_revisit_steps"]
        self.current_signal_metrics["fast_revisit_gap_steps"] = self.current_signal_metrics["sil_revisit_gap_steps"]

    def record_completed_episode_signal(
        self,
        task_idx,
        dense_return,
        success,
        eps_time,
        steps=None,
        iteration=None,
        episode_id=None,
    ):
        if not self.args.track_signal_metrics:
            return
        self.signal_episode_windows[int(task_idx)].append({
            "dense_return": float(dense_return),
            "success": bool(success),
            "eps_time": float(eps_time),
            "steps": None if steps is None else int(steps),
            "iteration": None if iteration is None else int(iteration),
            "episode_id": None if episode_id is None else int(episode_id),
        })

    def update_training_signal_summary(self):
        summary = {
            "iteration": int(self.global_update_iter),
            "steps": int(self.global_step),
            "episodes": int(self.global_episodes),
            "success": float(self.avg_task_metrics.get("success", 0.0)),
            "history_file": "training_signal_history.jsonl",
            "episode_history_file": "training_episode_signal_history.jsonl",
            "dense_tail_frac": self.dense_tail_frac,
            "signal_metrics_window": self.signal_metrics_window,
            **self.current_signal_metrics,
            **self.last_replay_stats,
        }
        per_task = {}
        for task_idx, tid in enumerate(self.unique_task_ids):
            purity = self.per_task_dense_tail_purity[task_idx]
            purity_value = None if bool(torch.isnan(purity).item()) else float(purity.item())
            per_task[int(tid.item())] = {
                "top10pct_Rdense_sr": purity_value,
                "dense_tail_window_size": int(self.per_task_dense_tail_window_size[task_idx].item()),
                "dense_tail_k": int(self.per_task_dense_tail_k[task_idx].item()),
            }
        summary["per_task"] = per_task
        self.training_signal_info.clear()
        self.training_signal_info.update(summary)

    def compute_success_positive_advantage_metrics(
        self,
        advantages,
        step_episode_ids,
        completed_episode_success,
        completed_episode_time=None,
        completed_episode_dense_return=None,
        completed_episode_task_id=None,
        advantages_raw=None,
    ):
        episode_stats = self._episode_positive_advantage_stats(
            advantages,
            step_episode_ids,
            completed_episode_success,
            completed_episode_time,
            completed_episode_dense_return,
            completed_episode_task_id,
            advantages_raw,
        )

        self.current_signal_metrics["succ_posadv_ratio"] = episode_stats["success_ratio"]
        self.current_signal_metrics["fast_succ_posadv_ratio"] = episode_stats["fast_success_ratio"]
        self.current_signal_metrics["succ_posadv_step_frac"] = episode_stats["success_step_fraction"]
        self.current_signal_metrics["adv_ep_used"] = episode_stats["episodes_used"]
        self.current_signal_metrics["episode_signal_count"] = len(self.latest_signal_episodes)
        self.current_signal_metrics["adv_episodes_used"] = int(len(completed_episode_success))

    def _episode_positive_advantage_stats(self, advantages, step_episode_ids, completed_episode_success,
                                          completed_episode_time=None, completed_episode_dense_return=None,
                                          completed_episode_task_id=None, advantages_raw=None):
        episode_ids = step_episode_ids.detach().cpu().numpy().reshape(-1)
        advantages_np = advantages.detach().float().cpu().numpy().reshape(-1)
        positive_adv = np.maximum(advantages_np, 0.0)
        raw_positive_adv = (
            np.maximum(advantages_raw.detach().float().cpu().numpy().reshape(-1), 0.0)
            if advantages_raw is not None else positive_adv
        )
        unique_ids, inverse = np.unique(episode_ids, return_inverse=True)
        masses = np.bincount(inverse, weights=positive_adv, minlength=len(unique_ids))
        raw_masses = np.bincount(inverse, weights=raw_positive_adv, minlength=len(unique_ids))
        positive_steps = np.bincount(inverse, weights=(advantages_np > 0.0).astype(np.float32), minlength=len(unique_ids))
        step_counts = np.bincount(inverse, minlength=len(unique_ids))
        for ep_id, mass, raw_mass, pos_steps, steps in zip(unique_ids, masses, raw_masses, positive_steps, step_counts):
            ep_id = int(ep_id)
            stats = self.episode_adv_mass.setdefault(
                ep_id,
                {"update": 0.0, "raw": 0.0, "positive_steps": 0, "steps": 0},
            )
            stats["update"] += float(mass)
            stats["raw"] += float(raw_mass)
            stats["positive_steps"] += int(pos_steps)
            stats["steps"] += int(steps)

        max_eps_time = max(float(self.max_eps_time), 1e-8)
        self.latest_signal_episodes = []
        self.latest_signal_episodes_iter = int(self.global_update_iter)
        for ep_id, success in completed_episode_success.items():
            ep_id = int(ep_id)
            stats = self.episode_adv_mass.pop(
                ep_id,
                {"update": 0.0, "raw": 0.0, "positive_steps": 0, "steps": 0},
            )
            mass = float(stats["update"])
            eps_time = float(completed_episode_time.get(ep_id, max_eps_time)) if completed_episode_time else max_eps_time
            fast_weight = max(0.0, 1.0 - eps_time / max_eps_time) if bool(success) else 0.0
            self.adv_episode_window.append((mass, bool(success), fast_weight, int(stats["positive_steps"])))
            self.latest_signal_episodes.append({
                "episode_id": ep_id,
                "task_id": int(completed_episode_task_id.get(ep_id, -1)) if completed_episode_task_id else -1,
                "success": bool(success),
                "eps_time": eps_time,
                "max_eps_time": max_eps_time,
                "dense_return": float(completed_episode_dense_return.get(ep_id, 0.0)) if completed_episode_dense_return else 0.0,
                "positive_adv_mass_update": mass,
                "positive_adv_mass_raw": float(stats["raw"]),
                "positive_adv_step_count_update": int(stats["positive_steps"]),
                "episode_step_count": int(stats["steps"]),
                "fast_success_weight": float(fast_weight),
            })

        total_mass = sum(entry[0] for entry in self.adv_episode_window)
        success_mass = sum(entry[0] for entry in self.adv_episode_window if entry[1])
        fast_success_mass = sum(entry[0] * entry[2] for entry in self.adv_episode_window)
        total_positive_steps = sum(entry[3] for entry in self.adv_episode_window)
        success_positive_steps = sum(entry[3] for entry in self.adv_episode_window if entry[1])
        return {
            "success_ratio": float(success_mass / total_mass) if total_mass > 0 else 0.0,
            "fast_success_ratio": float(fast_success_mass / total_mass) if total_mass > 0 else 0.0,
            "success_step_fraction": float(success_positive_steps / total_positive_steps) if total_positive_steps > 0 else 0.0,
            "episodes_used": int(len(self.adv_episode_window)),
        }

    def compute_advantage_magnitude_metrics(self, advantages_raw):
        """Track raw (unnormalized) advantage magnitude statistics."""
        adv = advantages_raw.detach().float()
        abs_adv = adv.abs()
        self.current_signal_metrics["adv_raw_mean"] = float(adv.mean().item())
        self.current_signal_metrics["adv_raw_std"] = float(adv.std().item())
        self.current_signal_metrics["adv_raw_abs_mean"] = float(abs_adv.mean().item())
        self.current_signal_metrics["adv_raw_abs_max"] = float(abs_adv.max().item())
        self.current_signal_metrics["adv_raw_abs_median"] = float(abs_adv.median().item())

    # ------------------------------------------------------------------
    # Metadata persistence
    # ------------------------------------------------------------------

    def save_meta_data_snapshot(self, tsil_memory=None):
        if not self.args.saving:
            return
        if tsil_memory is not None:
            tsil_memory.refresh_archive_meta(self.meta_data)
        self.update_training_signal_summary()
        save_json(self.meta_data, os.path.join(self.args.trajectory_dir, "meta_data.json"))
        self._append_training_signal_history()

    def _append_training_signal_history(self):
        history_path = os.path.join(self.args.trajectory_dir, "training_signal_history.jsonl")
        os.makedirs(self.args.trajectory_dir, exist_ok=True)
        with open(history_path, "a") as file_obj:
            file_obj.write(json.dumps(self.training_signal_info) + "\n")
        if (
            self.latest_signal_episodes
            and self.latest_signal_episodes_iter == int(self.global_update_iter)
            and self.last_episode_signal_history_iter != self.latest_signal_episodes_iter
        ):
            episode_path = os.path.join(self.args.trajectory_dir, "training_episode_signal_history.jsonl")
            base = {
                "iteration": int(self.global_update_iter),
                "steps": int(self.global_step),
                "episodes": int(self.global_episodes),
            }
            with open(episode_path, "a") as file_obj:
                for record in self.latest_signal_episodes:
                    file_obj.write(json.dumps({**base, **record}) + "\n")
            self.last_episode_signal_history_iter = self.latest_signal_episodes_iter
            self.latest_signal_episodes = []

    # ------------------------------------------------------------------
    # Status printing
    # ------------------------------------------------------------------

    def print_status(self, iter, num_iters, is_multi_task, num_tasks_with_success_archive=0):
        """Print training status with ANSI escape codes for multi-line overwriting."""
        if is_multi_task:
            task_success = [(tid.item(), self.per_task_avg['success'][i].item())
                            for i, tid in enumerate(self.unique_task_ids)]
            worst_5 = sorted(task_success, key=lambda x: x[1])[:5]
            worst_str = " ".join([f"T{tid}:{suc:.2f}" for tid, suc in worst_5])
            print_msg = (f"Iter: {iter}/{num_iters} | Eps: {self.global_episodes} | "
                         f"GScore: {self.cur_eps_G_score:.2f}/{self.best_eps_G_score:.2f} | "
                         f"Suc: {self.avg_task_metrics['success']:.2f}/{self.best_success_rate:.2f} | "
                         f"HasSuc: {num_tasks_with_success_archive}/{self.num_tasks} | "
                         f"Worst5: [{worst_str}]")
            self.training_info["worst_5_tasks"] = worst_5
        else:
            print_msg = (f"Iter: {iter}/{num_iters} | Eps: {self.global_episodes} | "
                         f"GScore: {self.cur_eps_G_score:.3f}/{self.best_eps_G_score:.3f} | "
                         f"Suc: {self.cur_success_rate:.3f}/{self.best_success_rate:.3f}")

        if self.args.scratch:
            print_msg += f" | EpsT:{self.cur_eps_time:.1f}/{self.avg_time2end_upper:.1f}/{self.max_eps_time:.1f}"
        if self.args.use_cost:
            print_msg += f" | C:{self.avg_task_metrics.get('eps_sum_cost', 0):.1f}"

        if len(self.iter_time_buf) > 0:
            avg_iter_time = sum(self.iter_time_buf) / len(self.iter_time_buf)
            remaining_iters = max(num_iters - iter - 1, 0)
            eta_hours = remaining_iters * avg_iter_time / 3600
            print_msg += f" | ETA: {eta_hours:.2f}h ({avg_iter_time:.2f}s/iter)"

        print(print_msg + '\r', end='')

    def save_training_checkpoints(self, agent, args, reward_normalizer, before_checkpoint_save_fn, save_project_meta_fn):
        """Save model checkpoints based on performance.

        Parameters
        ----------
        agent : the trainable agent with ``save_checkpoint()``
        args : training Args dataclass
        reward_normalizer : reward normalizer (or None)
        before_checkpoint_save_fn : callable, project hook
        save_project_meta_fn : callable, project hook
        """
        if not (args.saving and not args.random_policy):
            return
        before_checkpoint_save_fn()

        if self.ready_to_record and (self.curri_ratio == 1 or args.success_threshold == 0):
            def checkpoint_info(score_metric, score):
                return {
                    'iterations': self.global_update_iter,
                    'score_metric': score_metric,
                    'score': score,
                    'cost_penalty_scale': self.args.successRewardScale if self.args.use_cost else 0.0,
                    'success_rate': self.cur_success_rate,
                    'eps_G': self.avg_task_metrics['eps_G'],
                    'eps_G_score': self.cur_eps_G_score,
                    'eps_sum_rew': self.avg_task_metrics['eps_sum_rew'],
                    'eps_dense_return': self.avg_task_metrics['eps_dense_return'],
                    'eps_sum_cost': self.avg_task_metrics['eps_sum_cost'],
                    'per_task_metrics': self.snapshot_per_task_metrics(),
                }

            score_metric = 'eps_G_minus_scaled_cost' if self.args.use_cost else 'eps_G'
            if self.cur_eps_G_score >= self.best_eps_G_score:
                self.best_eps_G_score = self.cur_eps_G_score
                self.training_info['best_G'] = checkpoint_info(score_metric, self.best_eps_G_score)
                agent.save_checkpoint(folder_path=args.checkpoint_dir, suffix='best_G', reward_normalizer=reward_normalizer)

            if self.cur_success_rate >= self.best_success_rate:
                self.best_success_rate = self.cur_success_rate
                self.training_info['best_suc'] = checkpoint_info('success', self.best_success_rate)
                agent.save_checkpoint(folder_path=args.checkpoint_dir, suffix='best_suc', reward_normalizer=reward_normalizer)

            tail_start_iter = int((1.0 - args.best_suc_tail_frac) * args.total_iters)
            if (
                self.global_update_iter >= tail_start_iter and
                self.cur_success_rate >= self.best_tail_success_rate
            ):
                self.best_tail_success_rate = self.cur_success_rate
                self.training_info['best_suc_tail'] = checkpoint_info('success_tail', self.best_tail_success_rate)
                agent.save_checkpoint(folder_path=args.checkpoint_dir, suffix='best_suc_tail', reward_normalizer=reward_normalizer)

            cur_local_success = self.success_episodes / self.curri_episodes if self.curri_episodes > 0 else 0
            if (self.cur_success_rate >= args.init_success and
                cur_local_success >= args.init_success and
                self.cur_eps_time >= self.max_eps_time and
                args.scratch):
                self.max_eps_time = self.cur_eps_time
                self.training_info['max_eps_time'] = {
                    'iterations': self.global_update_iter,
                    'eps_time': self.max_eps_time
                }
                agent.save_checkpoint(folder_path=args.checkpoint_dir, suffix='init', reward_normalizer=reward_normalizer)

        if self.global_update_iter % args.save_iter == 0 and self.global_update_iter > 0:
            self.training_info['last_ckpt_iter'] = self.global_update_iter
            if args.last_only:
                agent.save_checkpoint(folder_path=args.checkpoint_dir, suffix='last', reward_normalizer=reward_normalizer)
            elif not args.best_only:
                agent.save_checkpoint(folder_path=args.checkpoint_dir, suffix=str(self.global_update_iter), reward_normalizer=reward_normalizer)

        # Periodic percentage-based checkpoint saving
        if args.save_periodic_pct > 0:
            step_interval = max(int(round(args.save_periodic_pct * args.total_iters)), 1)
            if self.global_update_iter > 0 and self.global_update_iter % step_interval == 0:
                pct_index = self.global_update_iter // step_interval
                pct_label = int(round(pct_index * args.save_periodic_pct * 100))
                suffix = f"pct{pct_label}"
                agent.save_checkpoint(folder_path=args.checkpoint_dir, suffix=suffix, reward_normalizer=reward_normalizer)
                save_checkpoint(reward_normalizer, args.checkpoint_dir, ckpt_name="rew_norm_eps", suffix=suffix)

        save_project_meta_fn()
