"""Trajectory archive inspection, replay, and video/frame export."""

import json
import os
import tempfile

import cv2
import numpy as np
import torch

from core.common.trajectory_archive import load_archive_index, load_episode_record_h5, save_episode_record_h5
from core.evaluation.replay.video import (
    encode_video_frames,
    linspace_indices,
    normalize_video_frames,
    sample_frame_indices,
    transcode_video_ffmpeg,
)
from core.plotting import plot_compare_results, plot_replay_results

__all__ = ["TrajectoryReplayer"]


class TrajectoryReplayer:
    """Self-contained trajectory archive inspection and replay.

    Handles loading archived episodes, replaying them through the environment,
    video encoding/transcoding, and comparison plots.

    Parameters
    ----------
    envs : IsaacGym vec-env
    args : evaluation Args namespace
    device : torch.device
    task_indices : Tensor of per-env task ids
    """

    def __init__(self, envs, args, device, task_indices):
        self.envs = envs
        self.args = args
        self.device = device
        self.task_indices = task_indices

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def inspect(self):
        """Inspect and replay archived trajectories (top-level entry)."""
        source_dir = self._resolve_trajectory_source_dir()
        archive_index = self._load_archive_index(source_dir)
        for task_id in self._resolve_archive_tasks(archive_index):
            self._execute_replay_plan(source_dir, archive_index, task_id)

    # ------------------------------------------------------------------
    # Source resolution
    # ------------------------------------------------------------------

    def _resolve_trajectory_source_dir(self):
        if self.args.trajectory_source_dir is not None:
            return self.args.trajectory_source_dir
        if hasattr(self.args, "meta_path") and self.args.meta_path is not None:
            return os.path.dirname(self.args.meta_path)
        raise ValueError("trajectory_source_dir is required when archive source cannot be inferred from the checkpoint")

    @staticmethod
    def _normalize_archive_index(archive_index):
        """Accept legacy replay archives while keeping stage bucket semantics."""
        for task_data in archive_index.get("tasks", {}).values():
            if "reward_topk" in task_data and "stage_high_returns" not in task_data:
                task_data["stage_high_returns"] = task_data["reward_topk"]
            if "return_topk" in task_data and "stage_high_returns" not in task_data:
                task_data["stage_high_returns"] = task_data["return_topk"]
            if "fast_success_topk" in task_data and "stage_fast_successes" not in task_data:
                task_data["stage_fast_successes"] = task_data["fast_success_topk"]
        return archive_index

    def _load_env0_debug_index(self, source_dir):
        """Build a tiny replay index for older env0-only debug archives."""
        debug_files = [("first", "first.h5"), ("last", "last.h5")]
        archive_index = {"topk": 1, "tasks": {}}
        for bucket_name, filename in debug_files:
            rel_path = os.path.join("episode_archives", "env0", filename)
            abs_path = os.path.join(source_dir, rel_path)
            if not os.path.exists(abs_path):
                continue
            record = load_episode_record_h5(abs_path)
            summary = self._json_safe(record.get("summary", {}))
            task_id = int(summary.get("task_id", getattr(self.args, "trajectory_task_id", 0) or 0))
            task_entry = archive_index["tasks"].setdefault(str(task_id), {})
            score = float(summary.get("global_episodes", summary.get("episode_return_raw", 0.0)))
            task_entry.setdefault(bucket_name, []).append(
                {"path": rel_path, "score": score, "summary": summary}
            )
        if not archive_index["tasks"]:
            return None
        return archive_index

    def _load_archive_index(self, source_dir):
        try:
            return self._normalize_archive_index(load_archive_index(source_dir))
        except FileNotFoundError as exc:
            debug_index = self._load_env0_debug_index(source_dir)
            if debug_index is not None:
                return debug_index
            raise exc
        except ValueError as exc:
            if "reward_topk" not in str(exc):
                raise
            index_path = os.path.join(source_dir, "archive_index.json")
            with open(index_path, "r") as file_obj:
                return self._normalize_archive_index(json.load(file_obj))

    def _load_archive_record(self, source_dir, task_id, bucket, rank):
        archive_index = self._load_archive_index(source_dir)
        tasks_index = archive_index.get("tasks", {})
        task_entry = tasks_index.get(str(int(task_id)), {})
        entry = task_entry.get(bucket, [])
        if rank < 1 or rank > len(entry):
            available_tasks = sorted(int(tid) for tid in tasks_index.keys())
            available_buckets = sorted(task_entry.keys()) if task_entry else []
            raise IndexError(
                f"Rank {rank} is out of range for task {task_id} bucket '{bucket}'. "
                f"Available entries: {len(entry)}. Available tasks: {available_tasks}. "
                f"Available buckets for task {task_id}: {available_buckets}"
            )
        rel_path = entry[rank - 1]["path"]
        abs_path = os.path.join(source_dir, rel_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(
                f"Trajectory archive record not found: {abs_path}. "
                "The new trajectory logger persists only admitted trajectories at archive flush/finalization; "
                f"check {os.path.join(source_dir, 'archive_index.json')} and confirm the run flushed its archive."
            )
        return entry[rank - 1], load_episode_record_h5(abs_path)

    # ------------------------------------------------------------------
    # Environment helpers
    # ------------------------------------------------------------------

    def _get_replay_env_id(self, task_id):
        task_matches = torch.nonzero(self.task_indices == int(task_id), as_tuple=False).flatten()
        if task_matches.numel() == 0:
            raise ValueError(f"Task {task_id} is not available in the current evaluation env.")
        return int(task_matches[0].item())

    def _clear_replay_termination_state(self, env_id):
        """
        To avoid replay gets interrupted by the env done signal.
        """
        reset_buf = getattr(self.envs, "reset_buf", None)
        if reset_buf is None:
            return
        try:
            reset_buf[env_id] = 0
        except Exception:
            pass

    def _prepare_replay_env(self, snapshot, env_id=0):
        self.envs.reset_all()
        self.envs.load_replay_snapshot(env_id, snapshot)
        if self.args.trajectory_record_video and hasattr(self.envs, "begin_manual_camera_capture"):
            self.envs.begin_manual_camera_capture(env_id)
        self.envs.update_observations_dict()

    @staticmethod
    def _extract_info_scalar(infos, key, env_id, default=0.0, as_bool=False):
        value = infos.get(key, default)
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                scalar = value.item()
            else:
                scalar = value[env_id].item()
        elif isinstance(value, np.ndarray):
            if value.ndim == 0:
                scalar = value.item()
            else:
                scalar = value[env_id].item()
        elif isinstance(value, (list, tuple)):
            scalar = value[env_id]
        else:
            scalar = value
        return bool(scalar) if as_bool else float(scalar)

    # ------------------------------------------------------------------
    # Video helpers
    # ------------------------------------------------------------------

    def _finalize_replay_video_frames(self):
        if not self.args.trajectory_record_video or not hasattr(self.envs, "end_manual_camera_capture"):
            return []
        frames = self.envs.end_manual_camera_capture()
        return normalize_video_frames(frames)

    def _capture_current_replay_frame(self):
        if not self.args.trajectory_record_video or not hasattr(self.envs, "render_headless"):
            return
        if hasattr(self.envs, "capture_manual_camera_frame") and self.envs.capture_manual_camera_frame():
            return
        self.envs.render_headless()

    @staticmethod
    def _linspace_indices(start, end, count):
        return linspace_indices(start, end, count)

    def _sample_frame_indices(self, num_frames):
        sample_count = max(int(getattr(self.args, "trajectory_frame_count", 6)), 0)
        include_initial = bool(getattr(self.args, "trajectory_include_initial", True))
        return sample_frame_indices(num_frames, sample_count=sample_count, include_initial=include_initial)

    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {key: TrajectoryReplayer._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [TrajectoryReplayer._json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _save_replay_frames(self, task_id, label, bucket_name, rank, record, replay, output_dir=None):
        if not getattr(self.args, "trajectory_export_frames", False):
            return
        frames = replay.get("video_frames", [])
        frame_indices = self._sample_frame_indices(len(frames))
        if not frame_indices:
            print(f"No frames captured for frame export task {int(task_id):02d} {label}")
            return

        frame_root = output_dir or self.args.trajectory_dir
        frame_dir = os.path.join(frame_root, "frames", f"task_{int(task_id):02d}_{label}")
        os.makedirs(frame_dir, exist_ok=True)
        frame_files = []
        for export_idx, frame_idx in enumerate(frame_indices):
            frame = np.asarray(frames[frame_idx])
            frame = np.ascontiguousarray(frame[..., :3]).astype(np.uint8, copy=False)
            frame_path = os.path.join(frame_dir, f"frame_{export_idx:03d}.png")
            if not cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"Failed to save replay frame: {frame_path}")
            frame_files.append(os.path.basename(frame_path))

        manifest = {
            "task_id": int(task_id),
            "label": label,
            "bucket": bucket_name,
            "rank": int(rank),
            "source_frame_count": int(len(frames)),
            "sampled_frame_indices": [int(index) for index in frame_indices],
            "frame_files": frame_files,
            "summary": self._json_safe(record.get("summary", {})),
        }
        manifest_path = os.path.join(frame_dir, "manifest.json")
        with open(manifest_path, "w") as file_obj:
            json.dump(manifest, file_obj, indent=2)
        print(f"Saved trajectory replay frames to {frame_dir}")

    def _video_export_fps(self):
        fps = self.args.trajectory_video_fps
        if fps is None:
            capture_step_interval = float(getattr(self.envs, "camera_capture_step_interval", 1))
            action_dt = float(self.envs.ctrl_dt) * float(getattr(self.envs, "num_inter_steps", 1))
            fps = max(int(round(1.0 / (action_dt * capture_step_interval))), 1)
        play_speed = float(getattr(self.args, "trajectory_play_speed", 1.0))
        if play_speed <= 0:
            raise ValueError(f"trajectory_play_speed must be positive, got {play_speed}")
        fps = max(int(round(float(fps) * play_speed)), 1)
        return fps

    def _encode_video_frames(self, frames, video_path):
        return encode_video_frames(frames, video_path, self._video_export_fps())

    def _trajectory_video_stem(self, task_id, label, output_dir=None):
        video_dir = self.args.trajectory_dir if output_dir is None else os.path.join(output_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        return os.path.join(video_dir, f"task_{int(task_id):02d}_{label}")

    def _transcode_video_ffmpeg(self, input_path, output_stem):
        return transcode_video_ffmpeg(input_path, output_stem)

    def _save_replay_video(self, task_id, label, frames, output_dir=None):
        video_stem = self._trajectory_video_stem(task_id, label, output_dir=output_dir)
        temp_dir = os.path.dirname(video_stem)
        raw_fd, raw_path = tempfile.mkstemp(
            prefix=f"task_{int(task_id):02d}_{label}_raw_",
            suffix=".mp4",
            dir=temp_dir,
        )
        os.close(raw_fd)
        try:
            if not self._encode_video_frames(frames, raw_path):
                return

            final_path, ffmpeg_error = self._transcode_video_ffmpeg(raw_path, video_stem)
            if final_path is not None:
                print(f"Saved trajectory replay video to {final_path}")
                return

            fallback_path = video_stem + ".mp4"
            os.replace(raw_path, fallback_path)
            raw_path = None
            if ffmpeg_error is not None:
                print(f"ffmpeg post-processing unavailable; kept raw OpenCV video at {fallback_path}")
                print(ffmpeg_error)
            else:
                print(f"Saved raw trajectory replay video to {fallback_path}")
        finally:
            try:
                if raw_path is not None and os.path.exists(raw_path):
                    os.remove(raw_path)
            except OSError:
                pass

    @staticmethod
    def _append_cycles_suffix(label, cycles):
        if int(cycles) > 1:
            return f"{label}_cycles{int(cycles)}"
        return label

    def _single_replay_video_label(self, replay_plan, bucket_name, rank):
        label = replay_plan.get("save_label")
        if not label:
            if rank == 1 and bucket_name in {"last", "first", "first_success", "stage_fast_successes"}:
                label = bucket_name
            elif rank == 1 and bucket_name == "success_topk":
                label = "best"
            elif rank == 1 and bucket_name == "stage_high_returns":
                label = "return"
            else:
                label = f"{bucket_name}_rank{rank}"
        return self._append_cycles_suffix(label, replay_plan["cycles"])

    @staticmethod
    def _format_done_reason(replay):
        if replay.get("first_done_step") is None:
            return "none"
        first_done_success = replay.get("first_done_success")
        first_done_timeout = replay.get("first_done_timeout")
        return f"success={first_done_success} timeout={first_done_timeout}"

    def _print_replay_cycle_status(self, task_id, bucket_name, rank, cycle_idx, cycles, replay):
        replayed_steps = int(replay.get("episode_length", len(replay.get("reward", []))))
        expected_steps = int(replay.get("expected_steps", replayed_steps))
        frame_count = len(replay.get("video_frames", []))
        terminal_step = replay.get("first_done_step")
        print(f"  Cycle {cycle_idx}/{cycles}")
        print(f"    Replayed steps      : {replayed_steps}/{expected_steps}")
        print(f"    Captured frames     : {frame_count}")
        if terminal_step is not None:
            print(f"    First done step     : {terminal_step}")
            print(f"    First done reason   : {self._format_done_reason(replay)}")

    def _print_replay_summary(self, task_id, bucket_name, rank, cycles, replays):
        replayed_steps = sum(int(replay.get("episode_length", len(replay.get("reward", [])))) for replay in replays)
        expected_steps = sum(int(replay.get("expected_steps", len(replay.get("reward", [])))) for replay in replays)
        frame_count = sum(len(replay.get("video_frames", [])) for replay in replays)
        expected_frame_count = expected_steps
        print()
        print(f"Replay Summary | task {int(task_id):02d} | {bucket_name} | rank {rank}")
        print(f"  Cycles                 : {cycles}")
        print(f"  Total replayed steps   : {replayed_steps}/{expected_steps}")
        print(f"  Total captured frames  : {frame_count}")
        print(f"  Expected frame target  : {expected_frame_count}")
        print()

    def _run_replay_cycles(self, record, env_id, task_id, bucket_name, rank, cycles):
        replays = []
        print()
        print(f"=== Replay Block | task {int(task_id):02d} | {bucket_name} | rank {rank} ===")
        for cycle_idx in range(cycles):
            replay = self._replay_archive_record(record, env_id=env_id)
            replays.append(replay)
            self._print_replay_cycle_status(task_id, bucket_name, rank, cycle_idx + 1, cycles, replay)
        self._print_replay_summary(task_id, bucket_name, rank, cycles, replays)
        return replays

    # ------------------------------------------------------------------
    # Eval-time recording
    # ------------------------------------------------------------------

    def _resolve_eval_task_id(self):
        if self.args.trajectory_task_id is not None:
            return int(self.args.trajectory_task_id)
        if self.task_indices is not None:
            return int(self.task_indices[0].item())
        tasks = getattr(self.args, "tasks", [])
        return int(tasks[0]) if tasks else 0

    def _zero_eval_lstm_state(self, agent):
        if not getattr(self.args, "use_lstm", False):
            return None
        return (
            torch.zeros(agent.crt_lstm.num_layers, self.args.num_envs, agent.crt_lstm.hidden_size, device=self.device),
            torch.zeros(agent.crt_lstm.num_layers, self.args.num_envs, agent.crt_lstm.hidden_size, device=self.device),
            torch.zeros(agent.act_lstm.num_layers, self.args.num_envs, agent.act_lstm.hidden_size, device=self.device),
            torch.zeros(agent.act_lstm.num_layers, self.args.num_envs, agent.act_lstm.hidden_size, device=self.device),
        )

    def _eval_action(self, agent, obs, done, lstm_state):
        if agent is None or getattr(self.args, "random_policy", False) or getattr(self.args, "heuristic_policy", False):
            action = torch.rand((self.args.num_envs, self.envs.num_actions), device=self.device)
            return action, lstm_state
        if getattr(self.args, "use_lstm", False):
            action, _, lstm_state = agent.get_action_and_value(obs, lstm_state, done, action_only=True)
            return action, lstm_state
        action, _ = agent.get_action_and_value(obs, action_only=True)
        return action, lstm_state

    @staticmethod
    def _np_stack(values, dtype):
        return np.asarray(values, dtype=dtype)

    def _reset_eval_sample_env(self, env_id, replay_snapshot=None):
        if hasattr(self.envs, "last_rand_vecs"):
            self.envs.last_rand_vecs[env_id].zero_()
        if hasattr(self.envs, "last_rand_vecs_test"):
            self.envs.last_rand_vecs_test[env_id].zero_()
        self.envs.reset_all()
        next_obs_dict = self.envs.reset()
        if replay_snapshot is not None:
            self.envs.load_replay_snapshot(env_id, replay_snapshot)
            next_obs_dict = self.envs.obs_dict
        next_obs = next_obs_dict["obs"].to(self.device)
        next_state = next_obs_dict["states"].to(self.device)
        next_done = torch.zeros(self.args.num_envs, device=self.device)
        snapshot = self.envs.export_replay_snapshot(env_id)
        return next_obs, next_state, next_done, snapshot

    def _record_eval_sample(self, agent, env_id, task_id, sample_idx, replay_snapshot=None):
        next_obs, next_state, next_done, snapshot = self._reset_eval_sample_env(env_id, replay_snapshot)
        lstm_state = self._zero_eval_lstm_state(agent)
        initial_obs = next_obs[env_id].detach().cpu().numpy().astype(np.float16)
        initial_state = next_state[env_id].detach().cpu().numpy().astype(np.float16)

        manual_capture = (
            self.args.trajectory_record_video
            and hasattr(self.envs, "begin_manual_camera_capture")
            and self.envs.begin_manual_camera_capture(env_id)
        )
        if manual_capture and getattr(self.args, "trajectory_include_initial", True):
            self._capture_current_replay_frame()

        obs_values, state_values, action_values = [], [], []
        reward_values, done_values, success_values = [], [], []
        scene_linvel_values, eps_sum_inst_values, eps_max_scevel_values = [], [], []
        cumulative_reward = 0.0
        final_info = {}
        post_success_steps = max(int(getattr(self.args, "eval_trajectory_post_success_steps", 0)), 0)
        stop_arm_after_success = bool(getattr(self.args, "eval_trajectory_stop_arm_after_success", False))
        post_success_active = False
        post_success_recorded_steps = 0
        first_success_step = None
        hold_action = None
        max_steps = int(getattr(self.envs, "max_episode_length", getattr(self.args, "episodeLength", 1000)))

        with torch.no_grad():
            for _ in range(max_steps + post_success_steps):
                was_post_success_active = post_success_active
                obs_values.append(next_obs[env_id].detach().cpu().numpy())
                state_values.append(next_state[env_id].detach().cpu().numpy())
                if post_success_active and hold_action is not None:
                    action = hold_action.clone()
                else:
                    action, lstm_state = self._eval_action(agent, next_obs, next_done, lstm_state)
                next_obs_dict, reward, done, infos = self.envs.step(action)

                reward_value = float(reward[env_id].item())
                done_value = bool(done[env_id].item())
                success_value = self._extract_info_scalar(infos, "success", env_id, default=False, as_bool=True)
                cumulative_reward += reward_value
                action_values.append(action[env_id].detach().cpu().numpy())
                reward_values.append(reward_value)
                done_values.append(done_value)
                success_values.append(success_value)
                scene_linvel_values.append(self._extract_info_scalar(infos, "scene_linvel", env_id, default=0.0))
                eps_sum_inst_values.append(self._extract_info_scalar(infos, "eps_sum_inst", env_id, default=0.0))
                eps_max_scevel_values.append(self._extract_info_scalar(infos, "eps_max_scevel", env_id, default=0.0))
                if was_post_success_active:
                    post_success_recorded_steps += 1

                final_info = infos
                next_obs = next_obs_dict["obs"].to(self.device)
                next_state = next_obs_dict["states"].to(self.device)
                next_done = done.to(self.device)
                if done_value:
                    if success_value and post_success_steps > 0 and not post_success_active:
                        post_success_active = True
                        first_success_step = len(reward_values)
                        hold_action = action.detach().clone()
                        if stop_arm_after_success:
                            hold_action[env_id, :-1] = 0.0
                    if post_success_active and post_success_recorded_steps < post_success_steps:
                        self._clear_replay_termination_state(env_id)
                        next_done[env_id] = False
                        continue
                    break
                if post_success_active and post_success_recorded_steps >= post_success_steps:
                    break
                if not post_success_active and len(reward_values) >= max_steps:
                    break

        replay = {
            "reward": reward_values,
            "cumulative_reward": np.cumsum(reward_values).tolist(),
            "done": done_values,
            "success": success_values,
            "scene_linvel": scene_linvel_values,
            "eps_sum_inst": eps_sum_inst_values,
            "eps_max_scevel": eps_max_scevel_values,
            "episode_length": len(reward_values),
            "expected_steps": len(reward_values),
            "episode_return_raw": cumulative_reward,
            "final_success": bool(any(success_values)) if post_success_steps > 0 else (bool(success_values[-1]) if success_values else False),
            "video_frames": self._finalize_replay_video_frames() if manual_capture else [],
            "first_success_step": first_success_step,
            "post_success_recorded_steps": int(post_success_recorded_steps),
        }
        summary = {
            "task_id": int(task_id),
            "eval_sample_index": int(sample_idx),
            "episode_length": int(len(reward_values)),
            "success": bool(replay["final_success"]),
            "first_success_step": -1 if first_success_step is None else int(first_success_step),
            "post_success_recorded_steps": int(post_success_recorded_steps),
            "episode_return_raw": float(cumulative_reward),
            "eps_time": self._extract_info_scalar(final_info, "eps_time", env_id, default=0.0),
            "eps_time_p": self._extract_info_scalar(final_info, "eps_time_p", env_id, default=0.0),
            "eps_max_scevel": self._extract_info_scalar(final_info, "eps_max_scevel", env_id, default=0.0),
            "eps_sum_inst": self._extract_info_scalar(final_info, "eps_sum_inst", env_id, default=0.0),
        }
        record = {
            "summary": summary,
            "snapshot": snapshot,
            "trajectory": {
                "initial_obs": initial_obs,
                "initial_state": initial_state,
                "obs": self._np_stack(obs_values, np.float16),
                "state": self._np_stack(state_values, np.float16),
                "action": self._np_stack(action_values, np.float16),
                "reward": self._np_stack(reward_values, np.float32),
                "reward_raw": self._np_stack(reward_values, np.float32),
                "reward_norm": self._np_stack(reward_values, np.float32),
                "done": self._np_stack(done_values, np.int8),
                "success": self._np_stack(success_values, np.int8),
                "scene_linvel": self._np_stack(scene_linvel_values, np.float32),
                "eps_sum_inst": self._np_stack(eps_sum_inst_values, np.float32),
                "eps_max_scevel": self._np_stack(eps_max_scevel_values, np.float32),
            },
        }
        return record, replay

    def record_eval_samples(self, agent):
        task_id = self._resolve_eval_task_id()
        env_id = self._get_replay_env_id(task_id)
        sample_count = max(int(getattr(self.args, "eval_trajectory_samples", 5)), 1)
        success_only = bool(getattr(self.args, "eval_trajectory_success_only", False))
        same_config = bool(getattr(self.args, "eval_trajectory_same_config", False))
        start_snapshot_path = getattr(self.args, "eval_trajectory_start_snapshot", None)
        max_attempts = int(getattr(self.args, "eval_trajectory_max_attempts", 0) or sample_count)
        max_attempts = max(max_attempts, sample_count)
        output_dir = os.path.join(self.args.trajectory_dir, "eval_samples")
        os.makedirs(output_dir, exist_ok=True)

        manifest = {"task_id": int(task_id), "samples": []}
        if agent is not None:
            agent.deterministic = self.args.deterministic

        suffix = " successful" if success_only else ""
        print(f"Recording {sample_count}{suffix} eval trajectory sample(s) for task {int(task_id):02d}.")
        attempt_idx = 0
        shared_snapshot = None
        if start_snapshot_path:
            shared_snapshot = load_episode_record_h5(start_snapshot_path)["snapshot"]
            same_config = True
        while len(manifest["samples"]) < sample_count and attempt_idx < max_attempts:
            sample_idx = len(manifest["samples"])
            label = f"eval_sample_{sample_idx:02d}"
            record, replay = self._record_eval_sample(agent, env_id, task_id, attempt_idx, shared_snapshot)
            if success_only and not record["summary"]["success"]:
                print(
                    f"  attempt_{attempt_idx:02d}: skipped failed episode "
                    f"steps={record['summary']['episode_length']} "
                    f"return={record['summary']['episode_return_raw']:.3f}"
                )
                attempt_idx += 1
                continue
            if same_config and shared_snapshot is None:
                shared_snapshot = record["snapshot"]
                record, replay = self._record_eval_sample(agent, env_id, task_id, attempt_idx, shared_snapshot)
                if success_only and not record["summary"]["success"]:
                    print(
                        f"  attempt_{attempt_idx:02d}: skipped failed fixed-config episode "
                        f"steps={record['summary']['episode_length']} "
                        f"return={record['summary']['episode_return_raw']:.3f}"
                    )
                    shared_snapshot = None
                    attempt_idx += 1
                    continue
            h5_path = os.path.join(output_dir, f"task_{int(task_id):02d}_{label}.h5")
            save_episode_record_h5(record, h5_path)
            self._save_replay_frames(
                task_id, label, "eval_samples", sample_idx + 1, record, replay, output_dir=output_dir,
            )
            if getattr(self.args, "trajectory_record_video", False):
                self._save_replay_video(
                    task_id, label, replay.get("video_frames", []), output_dir=output_dir,
                )
            manifest["samples"].append({
                "path": os.path.basename(h5_path),
                "label": label,
                "attempt_index": int(attempt_idx),
                "summary": self._json_safe(record["summary"]),
            })
            print(
                f"  {label}: success={record['summary']['success']} "
                f"steps={record['summary']['episode_length']} return={record['summary']['episode_return_raw']:.3f}"
            )
            attempt_idx += 1

        if len(manifest["samples"]) < sample_count:
            raise RuntimeError(
                f"Recorded {len(manifest['samples'])}/{sample_count} requested eval samples for task {int(task_id):02d} "
                f"after {attempt_idx} attempt(s)."
            )

        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, "w") as file_obj:
            json.dump(manifest, file_obj, indent=2)
        print(f"Saved eval trajectory samples to {output_dir}")

    # ------------------------------------------------------------------
    # First-step replay debug
    # ------------------------------------------------------------------

    @staticmethod
    def _format_debug_value(value, precision=6):
        if value is None:
            return "n/a"
        if isinstance(value, (bool, np.bool_)):
            return str(bool(value))
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{precision}f}"
        array = np.asarray(value)
        if array.ndim == 0:
            scalar = array.item()
            if isinstance(scalar, (bool, np.bool_)):
                return str(bool(scalar))
            if isinstance(scalar, (int, np.integer)):
                return str(int(scalar))
            return f"{float(scalar):.{precision}f}"
        return np.array2string(array.astype(np.float64), precision=4, floatmode="fixed", separator=", ")

    @staticmethod
    def _success_distance_threshold(task_name):
        thresholds = {
            "pick_place": 0.05,
            "pick_place_wall": 0.05,
        }
        return thresholds.get(task_name)

    @staticmethod
    def _vector_diff_stats(lhs, rhs):
        if lhs is None or rhs is None:
            return {"l2": None, "max_abs": None}
        lhs_np = np.asarray(lhs, dtype=np.float64)
        rhs_np = np.asarray(rhs, dtype=np.float64)
        if lhs_np.shape != rhs_np.shape:
            return {"l2": None, "max_abs": None}
        diff = lhs_np - rhs_np
        return {
            "l2": float(np.linalg.norm(diff.reshape(-1))),
            "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        }

    @staticmethod
    def _maybe_distance(lhs, rhs):
        if lhs is None or rhs is None:
            return None
        lhs_np = np.asarray(lhs, dtype=np.float64).reshape(-1)
        rhs_np = np.asarray(rhs, dtype=np.float64).reshape(-1)
        if lhs_np.shape != rhs_np.shape:
            return None
        return float(np.linalg.norm(lhs_np - rhs_np))

    @staticmethod
    def _first_step_reward(record):
        trajectory = record.get("trajectory", {})
        reward_raw = trajectory.get("reward_raw")
        if reward_raw is None:
            reward_raw = trajectory.get("reward")
        if reward_raw is None or len(reward_raw) == 0:
            return None
        return float(np.asarray(reward_raw)[0])

    def _tensor_row_to_numpy(self, tensor, env_id):
        if tensor is None:
            return None
        try:
            return tensor[env_id].detach().cpu().numpy().copy()
        except Exception:
            return None

    def _current_obs_to_numpy(self, env_id):
        obs_dict = getattr(self.envs, "obs_dict", None)
        if not isinstance(obs_dict, dict) or "obs" not in obs_dict:
            return None
        obs_tensor = obs_dict["obs"]
        try:
            return obs_tensor[env_id].detach().cpu().numpy().copy()
        except Exception:
            return None

    def _task_debug_observation(self, task_name, env_id):
        try:
            from isaacgymenvs.tasks.franka.vec_task import task_fns
        except Exception:
            return None

        env_ids = torch.tensor([env_id], device=self.device, dtype=torch.long)
        obs_fn = getattr(task_fns, task_name).compute_observations
        with torch.no_grad():
            task_obs = obs_fn(self.envs, env_ids)
        if isinstance(task_obs, torch.Tensor):
            task_obs = task_obs[0].detach().cpu().numpy()
        return np.asarray(task_obs)

    def _collect_first_step_state(self, record, env_id=0):
        snapshot = record["snapshot"]
        task_id = int(snapshot["task_id"])
        task_name = self.envs.task_idx2name[task_id]
        success_threshold = self._success_distance_threshold(task_name)
        archived_initial_obs = np.asarray(record.get("trajectory", {}).get("initial_obs")) if "initial_obs" in record.get("trajectory", {}) else None

        self._prepare_replay_env(snapshot, env_id=env_id)

        pre_task_obs = self._task_debug_observation(task_name, env_id)
        pre_obj_pos = pre_task_obs[:3].copy() if pre_task_obs is not None and pre_task_obs.shape[0] >= 3 else None
        pre_target_pos = self._tensor_row_to_numpy(getattr(self.envs, "target_pos", None), env_id)
        pre_obj_init_pos = self._tensor_row_to_numpy(getattr(self.envs, "obj_init_pos", None), env_id)
        pre_last_rand_vec = self._tensor_row_to_numpy(getattr(self.envs, "last_rand_vecs", None), env_id)
        pre_obs = self._current_obs_to_numpy(env_id)
        pre_obj_to_target = self._maybe_distance(pre_obj_pos, pre_target_pos)

        actions = np.asarray(record["trajectory"]["action"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] == 0:
            raise ValueError("Trajectory record does not contain any archived actions to debug.")

        with torch.no_grad():
            action = torch.zeros((self.args.num_envs, actions.shape[1]), dtype=torch.float32, device=self.device)
            action[env_id] = torch.tensor(actions[0], dtype=torch.float32, device=self.device)
            next_obs_dict, reward, done, infos = self.envs.step(action)

        post_task_obs = self._task_debug_observation(task_name, env_id)
        post_obj_pos = post_task_obs[:3].copy() if post_task_obs is not None and post_task_obs.shape[0] >= 3 else None
        post_target_pos = self._tensor_row_to_numpy(getattr(self.envs, "target_pos", None), env_id)
        post_obj_to_target = self._maybe_distance(post_obj_pos, post_target_pos)

        archived_done = record.get("trajectory", {}).get("done")
        archived_first_done = None
        if archived_done is not None and len(archived_done) > 0:
            archived_first_done = bool(np.asarray(archived_done)[0])

        return {
            "task_id": task_id,
            "task_name": task_name,
            "success_threshold": success_threshold,
            "archived_episode_length": int(record.get("summary", {}).get("episode_length", actions.shape[0])),
            "archived_first_done": archived_first_done,
            "archived_first_reward": self._first_step_reward(record),
            "snapshot_progress_buf": snapshot.get("progress_buf"),
            "snapshot_last_rand_vec": snapshot.get("last_rand_vec"),
            "snapshot_target_pos": snapshot.get("target_pos"),
            "snapshot_obj_init_pos": snapshot.get("obj_init_pos"),
            "restored_pre": {
                "last_rand_vec": pre_last_rand_vec,
                "target_pos": pre_target_pos,
                "obj_init_pos": pre_obj_init_pos,
                "primary_obj_pos": pre_obj_pos,
                "obj_to_target": pre_obj_to_target,
                "geometric_success": (
                    (pre_obj_to_target is not None) and (success_threshold is not None) and (pre_obj_to_target < success_threshold)
                ),
                "progress_buf": int(self.envs.progress_buf[env_id].item()),
                "reset_buf": int(self.envs.reset_buf[env_id].item()),
                "success_buf": bool(self.envs.success_buf[env_id].item()),
                "timeout_buf": bool(self.envs.timeout_buf[env_id].item()),
                "obs_diff": self._vector_diff_stats(pre_obs, archived_initial_obs),
            },
            "after_first_action": {
                "action": actions[0].copy(),
                "reward": float(reward[env_id].item()),
                "done": bool(done[env_id].item()),
                "info_success": self._extract_info_scalar(infos, "success", env_id, default=False, as_bool=True),
                "info_timeout": self._extract_info_scalar(infos, "time_outs", env_id, default=False, as_bool=True),
                "primary_obj_pos": post_obj_pos,
                "target_pos": post_target_pos,
                "obj_to_target": post_obj_to_target,
                "geometric_success": (
                    (post_obj_to_target is not None) and (success_threshold is not None) and (post_obj_to_target < success_threshold)
                ),
                "progress_buf": int(self.envs.progress_buf[env_id].item()),
                "reset_buf": int(self.envs.reset_buf[env_id].item()),
                "success_buf": bool(self.envs.success_buf[env_id].item()),
                "timeout_buf": bool(self.envs.timeout_buf[env_id].item()),
                "obs": (
                    next_obs_dict["obs"][env_id].detach().cpu().numpy().copy()
                    if isinstance(next_obs_dict, dict) and "obs" in next_obs_dict
                    else None
                ),
            },
        }

    def _print_first_step_debug_report(self, bucket_name, rank, entry_path, debug_state):
        pre = debug_state["restored_pre"]
        post = debug_state["after_first_action"]
        target_delta = self._vector_diff_stats(debug_state.get("snapshot_target_pos"), pre.get("target_pos"))
        obj_init_delta = self._vector_diff_stats(debug_state.get("snapshot_obj_init_pos"), pre.get("obj_init_pos"))
        last_rand_delta = self._vector_diff_stats(debug_state.get("snapshot_last_rand_vec"), pre.get("last_rand_vec"))

        print()
        print(f"=== First-Step Replay Debug | task {int(debug_state['task_id']):02d} | {bucket_name} | rank {rank} ===")
        print(f"  Archive entry            : {entry_path}")
        print(f"  Task name                : {debug_state['task_name']}")
        print(f"  Archived episode length  : {debug_state['archived_episode_length']}")
        print(f"  Snapshot progress_buf    : {self._format_debug_value(debug_state.get('snapshot_progress_buf'))}")
        print(f"  Archived step-1 done     : {self._format_debug_value(debug_state.get('archived_first_done'))}")
        print(f"  Archived step-1 reward   : {self._format_debug_value(debug_state.get('archived_first_reward'))}")
        print()
        print("  Snapshot buffers")
        print(f"    last_rand_vec          : {self._format_debug_value(debug_state.get('snapshot_last_rand_vec'))}")
        print(f"    target_pos             : {self._format_debug_value(debug_state.get('snapshot_target_pos'))}")
        print(f"    obj_init_pos           : {self._format_debug_value(debug_state.get('snapshot_obj_init_pos'))}")
        print()
        print("  Restored pre-step state")
        print(f"    last_rand_vec          : {self._format_debug_value(pre.get('last_rand_vec'))}")
        print(f"    target_pos             : {self._format_debug_value(pre.get('target_pos'))}")
        print(f"    obj_init_pos           : {self._format_debug_value(pre.get('obj_init_pos'))}")
        print(f"    primary_obj_pos        : {self._format_debug_value(pre.get('primary_obj_pos'))}")
        print(f"    obj_to_target          : {self._format_debug_value(pre.get('obj_to_target'))}")
        if debug_state.get("success_threshold") is not None:
            print(
                f"    geometric_success      : {self._format_debug_value(pre.get('geometric_success'))} "
                f"(threshold {self._format_debug_value(debug_state['success_threshold'])})"
            )
        print(f"    target_pos delta (L2)  : {self._format_debug_value(target_delta.get('l2'))}")
        print(f"    obj_init delta (L2)    : {self._format_debug_value(obj_init_delta.get('l2'))}")
        print(f"    last_rand delta (L2)   : {self._format_debug_value(last_rand_delta.get('l2'))}")
        print(f"    initial_obs diff (L2)  : {self._format_debug_value(pre.get('obs_diff', {}).get('l2'))}")
        print(f"    initial_obs max diff   : {self._format_debug_value(pre.get('obs_diff', {}).get('max_abs'))}")
        print(f"    progress_buf           : {self._format_debug_value(pre.get('progress_buf'))}")
        print(
            "    reset/success/timeout  : "
            f"{self._format_debug_value(pre.get('reset_buf'))} / "
            f"{self._format_debug_value(pre.get('success_buf'))} / "
            f"{self._format_debug_value(pre.get('timeout_buf'))}"
        )
        print()
        print("  After first archived action")
        print(f"    action                 : {self._format_debug_value(post.get('action'))}")
        print(f"    reward                 : {self._format_debug_value(post.get('reward'))}")
        print(f"    done                   : {self._format_debug_value(post.get('done'))}")
        print(
            "    info success/timeout   : "
            f"{self._format_debug_value(post.get('info_success'))} / "
            f"{self._format_debug_value(post.get('info_timeout'))}"
        )
        print(f"    primary_obj_pos        : {self._format_debug_value(post.get('primary_obj_pos'))}")
        print(f"    target_pos             : {self._format_debug_value(post.get('target_pos'))}")
        print(f"    obj_to_target          : {self._format_debug_value(post.get('obj_to_target'))}")
        if debug_state.get("success_threshold") is not None:
            print(
                f"    geometric_success      : {self._format_debug_value(post.get('geometric_success'))} "
                f"(threshold {self._format_debug_value(debug_state['success_threshold'])})"
            )
        print(f"    progress_buf           : {self._format_debug_value(post.get('progress_buf'))}")
        print(
            "    reset/success/timeout  : "
            f"{self._format_debug_value(post.get('reset_buf'))} / "
            f"{self._format_debug_value(post.get('success_buf'))} / "
            f"{self._format_debug_value(post.get('timeout_buf'))}"
        )
        print()

    def _resolve_first_step_debug_targets(self, task_id, task_entry, archive_index):
        replay_plan = self._resolve_replay_plan(task_id, task_entry, archive_index)
        if replay_plan is None:
            return None
        if replay_plan["mode"] == "compare":
            rank = int(self.args.trajectory_rank)
            return [{"bucket": bucket_name, "rank": rank} for bucket_name in replay_plan["buckets"]]
        if replay_plan["mode"] == "single":
            return [{"bucket": replay_plan["bucket"], "rank": replay_plan["rank"]}]
        return [{"bucket": replay_plan["bucket"], "rank": rank} for rank in replay_plan["ranks"]]

    def debug_replay_first_step(self):
        """Print a focused diagnostic for the first archived replay transition."""
        source_dir = self._resolve_trajectory_source_dir()
        archive_index = self._load_archive_index(source_dir)
        for task_id in self._resolve_archive_tasks(archive_index):
            task_entry = archive_index.get("tasks", {}).get(str(int(task_id)), {})
            if not task_entry:
                print(f"Skipping task {task_id}: no archived trajectories found.")
                continue

            debug_targets = self._resolve_first_step_debug_targets(task_id, task_entry, archive_index)
            if not debug_targets:
                print(f"Skipping task {task_id}: no trajectories available for mode '{self.args.trajectory_bucket}'.")
                continue

            env_id = self._get_replay_env_id(task_id)
            for target in debug_targets:
                entry, record = self._load_archive_record(source_dir, task_id, target["bucket"], target["rank"])
                debug_state = self._collect_first_step_state(record, env_id=env_id)
                self._print_first_step_debug_report(
                    bucket_name=target["bucket"],
                    rank=target["rank"],
                    entry_path=entry["path"],
                    debug_state=debug_state,
                )

    # ------------------------------------------------------------------
    # Archive replay
    # ------------------------------------------------------------------

    def _replay_archive_record(self, record, env_id=0):
        snapshot = record["snapshot"]
        actions = np.asarray(record["trajectory"]["action"], dtype=np.float32)
        self._prepare_replay_env(snapshot, env_id=env_id)
        if getattr(self.args, "trajectory_export_frames", False) and getattr(self.args, "trajectory_include_initial", True):
            self._capture_current_replay_frame()

        replay = {
            "reward": [],
            "cumulative_reward": [],
            "done": [],
            "success": [],
            "scene_linvel": [],
            "scene_linvel_lim": [],
            "eps_max_scevel": [],
            "eps_sum_inst": [],
            "eps_time_goal": [],
            "real_cur_time": [],
            "observed_time2end": [],
            "real_time2end": [],
        }

        cumulative_reward = 0.0
        final_info = {}
        first_done_step = None
        first_done_success = None
        first_done_timeout = None
        with torch.no_grad():
            for step_idx, step_action in enumerate(actions):
                action = torch.zeros((self.args.num_envs, actions.shape[1]), dtype=torch.float32, device=self.device)
                action[env_id] = torch.tensor(step_action, dtype=torch.float32, device=self.device)
                next_obs_dict, reward, done, infos = self.envs.step(action)

                reward_value = float(reward[env_id].item())
                cumulative_reward += reward_value
                replay["reward"].append(reward_value)
                replay["cumulative_reward"].append(cumulative_reward)
                replay["done"].append(bool(done[env_id].item()))
                replay["success"].append(self._extract_info_scalar(infos, "success", env_id, default=False, as_bool=True))
                replay["scene_linvel"].append(self._extract_info_scalar(infos, "scene_linvel", env_id, default=0.0))
                replay["scene_linvel_lim"].append(self._extract_info_scalar(infos, "scene_linvel_lim", env_id, default=0.0))
                replay["eps_max_scevel"].append(self._extract_info_scalar(infos, "eps_max_scevel", env_id, default=0.0))
                replay["eps_sum_inst"].append(self._extract_info_scalar(infos, "eps_sum_inst", env_id, default=0.0))
                replay["eps_time_goal"].append(self._extract_info_scalar(infos, "eps_time_goal", env_id, default=0.0))
                replay["real_cur_time"].append(self._extract_info_scalar(infos, "real_cur_time", env_id, default=0.0))
                replay["observed_time2end"].append(self._extract_info_scalar(infos, "observed_time2end", env_id, default=0.0))
                replay["real_time2end"].append(self._extract_info_scalar(infos, "real_time2end", env_id, default=0.0))
                final_info = infos

                if done[env_id].item() == 1:
                    if first_done_step is None:
                        first_done_step = step_idx + 1
                        first_done_success = self._extract_info_scalar(infos, "success", env_id, default=False, as_bool=True)
                        first_done_timeout = self._extract_info_scalar(infos, "time_outs", env_id, default=False, as_bool=True)
                    if step_idx + 1 < actions.shape[0]:
                        self._clear_replay_termination_state(env_id)

        replay["episode_length"] = len(replay["reward"])
        replay["expected_steps"] = int(actions.shape[0])
        replay["first_done_step"] = first_done_step
        replay["first_done_success"] = first_done_success
        replay["first_done_timeout"] = first_done_timeout
        replay["episode_return_raw"] = cumulative_reward
        replay["final_success"] = bool(replay["success"][-1]) if replay["success"] else False
        replay["final_info"] = final_info
        replay["video_frames"] = self._finalize_replay_video_frames()
        return replay

    # ------------------------------------------------------------------
    # Task / plan resolution
    # ------------------------------------------------------------------

    def _resolve_archive_tasks(self, archive_index):
        archive_task_ids = sorted(int(task_id) for task_id in archive_index.get("tasks", {}).keys())
        if self.args.trajectory_task_id is not None:
            task_id = int(self.args.trajectory_task_id)
            if task_id not in archive_task_ids:
                raise ValueError(
                    f"Task {task_id} is not present in the trajectory archive. Available tasks: {archive_task_ids}"
                )
            return [task_id]

        config_task_ids = [int(task_id) for task_id in getattr(self.args, "tasks", [])]
        if config_task_ids:
            matched_task_ids = [task_id for task_id in config_task_ids if task_id in archive_task_ids]
            if matched_task_ids:
                return matched_task_ids
        return archive_task_ids

    def _resolve_replay_plan(self, task_id, task_entry, archive_index):
        bucket_mode = self.args.trajectory_bucket
        cycles = max(int(getattr(self.args, "trajectory_cycles", 2)), 1)

        if bucket_mode == "compare":
            success_bucket = "stage_fast_successes" if task_entry.get("stage_fast_successes", []) else "success_topk"
            return {"mode": "compare", "buckets": [success_bucket, "stage_high_returns"], "cycles": 1}

        if bucket_mode == "topk":
            success_entries = task_entry.get("success_topk", []) or task_entry.get("stage_fast_successes", [])
            if success_entries:
                max_rank = min(len(success_entries), int(archive_index.get("topk", len(success_entries))))
                return {
                    "mode": "loop",
                    "bucket": "success_topk" if task_entry.get("success_topk", []) else "stage_fast_successes",
                    "ranks": list(range(max_rank, 0, -1)),
                    "cycles": cycles,
                }
            if task_entry.get("stage_high_returns", []):
                print(f"Task {task_id}: no successful trajectories found, replaying 'stage_high_returns' instead.")
                return {"mode": "single", "bucket": "stage_high_returns", "rank": 1, "cycles": cycles, "save_label": "topk_fallback_return"}
            if task_entry.get("last", []):
                print(f"Task {task_id}: no successful trajectories found, replaying 'last' instead.")
                return {"mode": "single", "bucket": "last", "rank": 1, "cycles": cycles, "save_label": "topk_fallback_last"}
            if task_entry.get("first", []):
                print(f"Task {task_id}: no successful or last trajectory found, replaying 'first' instead.")
                return {"mode": "single", "bucket": "first", "rank": 1, "cycles": cycles, "save_label": "topk_fallback_first"}
            return None

        bucket_aliases = {
            "best": ("stage_fast_successes", 1),
            "fastest": ("stage_fast_successes", 1),
            "fast_success": ("stage_fast_successes", 1),
            "fast-success": ("stage_fast_successes", 1),
            "stage_fast_success": ("stage_fast_successes", 1),
            "stage_fast_successes": ("stage_fast_successes", 1),
            "first_success": ("first_success", 1),
            "first": ("first", 1),
            "last": ("last", 1),
            "return": ("stage_high_returns", 1),
            "reward": ("stage_high_returns", 1),
            "high_return": ("stage_high_returns", 1),
            "stage_high_return": ("stage_high_returns", 1),
            "stage_high_returns": ("stage_high_returns", 1),
        }
        target_bucket, target_rank = bucket_aliases.get(bucket_mode, (bucket_mode, int(self.args.trajectory_rank)))
        if task_entry.get(target_bucket, []):
            return {"mode": "single", "bucket": target_bucket, "rank": target_rank, "cycles": cycles, "save_label": bucket_mode}

        if target_bucket == "first" and task_entry.get("first_success", []):
            return {"mode": "single", "bucket": "first_success", "rank": 1, "cycles": cycles, "save_label": bucket_mode}
        if target_bucket in {"success_topk", "stage_fast_successes"} and task_entry.get("stage_high_returns", []):
            print(f"Task {task_id}: no trajectory for '{target_bucket}', falling back to 'stage_high_returns'.")
            return {"mode": "single", "bucket": "stage_high_returns", "rank": 1, "cycles": cycles, "save_label": f"{bucket_mode}_fallback_return"}
        if target_bucket in {"success_topk", "stage_fast_successes"} and task_entry.get("last", []):
            print(f"Task {task_id}: no trajectory for '{target_bucket}', falling back to 'last'.")
            return {"mode": "single", "bucket": "last", "rank": 1, "cycles": cycles, "save_label": f"{bucket_mode}_fallback_last"}
        return None

    # ------------------------------------------------------------------
    # Plan execution
    # ------------------------------------------------------------------

    def _execute_replay_plan(self, source_dir, archive_index, task_id):
        task_entry = archive_index.get("tasks", {}).get(str(int(task_id)), {})
        if not task_entry:
            print(f"Skipping task {task_id}: no archived trajectories found.")
            return

        replay_plan = self._resolve_replay_plan(task_id, task_entry, archive_index)
        if replay_plan is None:
            print(f"Skipping task {task_id}: no trajectories available for mode '{self.args.trajectory_bucket}'.")
            return

        env_id = self._get_replay_env_id(task_id)
        os.makedirs(self.args.trajectory_dir, exist_ok=True)

        if replay_plan["mode"] == "compare":
            rank = int(self.args.trajectory_rank)
            compare_payloads = {}
            for bucket_name in replay_plan["buckets"]:
                _, record = self._load_archive_record(source_dir, task_id, bucket_name, rank)
                replay = self._replay_archive_record(record, env_id=env_id)
                compare_payloads[bucket_name] = {
                    "record": record,
                    "replay": replay,
                }
                if self.args.trajectory_record_video:
                    self._save_replay_video(task_id, f"compare_{bucket_name}_rank{rank}", replay.get("video_frames", []))
            save_path = os.path.join(self.args.trajectory_dir, f"task_{task_id:02d}_compare_rank{rank}.png")
            plot_compare_results(task_id, rank, compare_payloads, save_path)
            print(f"Saved trajectory comparison to {save_path}")
            return

        if replay_plan["mode"] == "single":
            bucket_name = replay_plan["bucket"]
            rank = replay_plan["rank"]
            entry, record = self._load_archive_record(source_dir, task_id, bucket_name, rank)
            replays = self._run_replay_cycles(
                record,
                env_id=env_id,
                task_id=task_id,
                bucket_name=bucket_name,
                rank=rank,
                cycles=replay_plan["cycles"],
            )
            replay = replays[0]
            save_path = os.path.join(
                self.args.trajectory_dir,
                f"task_{task_id:02d}_{bucket_name}_rank{rank}.png",
            )
            plot_replay_results(task_id, bucket_name, rank, record, replay, save_path)
            print(f"Loaded archive entry: {entry['path']}")
            print(f"Saved trajectory replay plot to {save_path}")
            if self.args.trajectory_record_video:
                video_frames = []
                for replay in replays:
                    video_frames.extend(replay.get("video_frames", []))
                label = self._single_replay_video_label(replay_plan, bucket_name, rank)
                self._save_replay_video(task_id, label, video_frames)
            frame_label = f"{replay_plan.get('save_label') or bucket_name}_rank{rank}"
            self._save_replay_frames(task_id, frame_label, bucket_name, rank, record, replay)
            return

        for rank in replay_plan["ranks"]:
            entry, record = self._load_archive_record(source_dir, task_id, replay_plan["bucket"], rank)
            replays = self._run_replay_cycles(
                record,
                env_id=env_id,
                task_id=task_id,
                bucket_name=replay_plan["bucket"],
                rank=rank,
                cycles=replay_plan["cycles"],
            )
            replay = replays[0]
            save_path = os.path.join(
                self.args.trajectory_dir,
                f"task_{task_id:02d}_{replay_plan['bucket']}_rank{rank}.png",
            )
            plot_replay_results(task_id, replay_plan["bucket"], rank, record, replay, save_path)
            print(f"Loaded archive entry: {entry['path']}")
            print(f"Saved trajectory replay plot to {save_path}")
            if self.args.trajectory_record_video:
                video_frames = []
                for replay in replays:
                    video_frames.extend(replay.get("video_frames", []))
                label = self._append_cycles_suffix(
                    f"{replay_plan['bucket']}_rank{rank}",
                    replay_plan["cycles"],
                )
                self._save_replay_video(task_id, label, video_frames)
            frame_label = f"{replay_plan['bucket']}_rank{rank}"
            self._save_replay_frames(task_id, frame_label, replay_plan["bucket"], rank, record, replay)
