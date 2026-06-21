"""Evaluation metric tracking, aggregation, and result persistence."""

import os

import torch
from tabulate import tabulate

from core.agents.utils import update_tensor_buffer
from core.common.io import write_csv_line, save_json

__all__ = ["EvalMetrics"]


class EvalMetrics:
    """Metric buffers and per-task tracking for evaluation.

    Parameters
    ----------
    args : evaluation Args namespace (may be mutated: ``target_episodes``)
    device : torch.device
    num_tasks : int
    unique_task_ids : Tensor of unique task ids
    tid_to_tidx : Tensor mapping task id -> contiguous index
    is_multi_task : bool
    tensor_dtype : torch.dtype
    """

    def __init__(self, args, device, num_tasks, unique_task_ids, tid_to_tidx, is_multi_task, tensor_dtype):
        self.args = args
        self.device = device
        self.num_tasks = num_tasks
        self.unique_task_ids = unique_task_ids
        self.tid_to_tidx = tid_to_tidx
        self.is_multi_task = is_multi_task
        self.tensor_dtype = tensor_dtype

        self._init_buffers()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_buffers(self):
        args = self.args
        args.requested_target_episodes = getattr(args, "requested_target_episodes", args.target_episodes)
        buffer_len = args.target_episodes
        if args.target_success_eps is not None:
            args.target_episodes = int(3e6)
            buffer_len = args.target_success_eps + args.num_envs

        # Per-env step accumulators (reset on episode end)
        self.step_metrics = {
            "eps_r": torch.zeros((args.num_envs,), dtype=self.tensor_dtype).to(self.device),
        }

        # Global episode metrics buffer
        self.eps_metrics = {
            key: torch.zeros((buffer_len,), dtype=self.tensor_dtype).to(self.device)
            for key in self.step_metrics.keys()
        }
        self.eps_metrics.update({
            "eps_time_p": torch.zeros((buffer_len,), dtype=self.tensor_dtype).to(self.device),
            "eps_time": torch.zeros((buffer_len,), dtype=self.tensor_dtype).to(self.device),
            "eps_time_goal": torch.zeros((buffer_len,), dtype=self.tensor_dtype).to(self.device),
            "eps_max_inst": torch.zeros((buffer_len,), dtype=self.tensor_dtype).to(self.device),
            "eps_lim_inst": torch.zeros((buffer_len,), dtype=self.tensor_dtype).to(self.device),
            "eps_sum_inst": torch.zeros((buffer_len,), dtype=self.tensor_dtype).to(self.device),
            "eps_success": torch.zeros((buffer_len,), dtype=self.tensor_dtype).to(self.device),
            "interaction_time": torch.zeros((buffer_len,), dtype=self.tensor_dtype).to(self.device),
        })

        self.eps_metrics_avg = {key: 0. for key in self.eps_metrics.keys()}
        self.eps_metrics_std = {key: 0. for key in self.eps_metrics.keys()}

        # === Per-task metric tracking (for multi-task evaluation) ===
        self.per_task_buf_len = max(buffer_len // self.num_tasks, 100)

        self.per_task_episode_count = torch.zeros(self.num_tasks, dtype=torch.long, device=self.device)
        self.per_task_success_count = torch.zeros(self.num_tasks, dtype=torch.long, device=self.device)

        self.per_task_metrics = ["eps_r", "eps_success", "eps_time", "eps_time_p", "eps_max_inst", "eps_sum_inst"]
        self.success_only_metrics = ["eps_time", "eps_time_p", "eps_max_inst", "eps_sum_inst"]

        self.per_task_buf = {}
        for metric in self.per_task_metrics:
            self.per_task_buf[metric] = torch.zeros(
                (self.num_tasks, self.per_task_buf_len), dtype=self.tensor_dtype, device=self.device,
            )

        self.per_task_buf_ptr = torch.zeros(self.num_tasks, dtype=torch.long, device=self.device)

        self.per_task_avg = {
            metric: torch.zeros(self.num_tasks, dtype=self.tensor_dtype, device=self.device)
            for metric in self.per_task_metrics
        }
        self.avg_task_metrics = {metric: 0.0 for metric in self.per_task_metrics}

        self.speed_and_time_dict = {
            "time_ratio": [],
            "time_used": [],
            "time_goal": [],
            "time_mismatch": [],
            "max_inst": [],
            "sum_inst": [],
            "thred_inst": [],
            "success_rate": [],
            "interaction_time": [],
        }

    # ------------------------------------------------------------------
    # Episode metric updates
    # ------------------------------------------------------------------

    def update_episode_metrics(self, terminal_index, infos, task_indices):
        """Update metrics when episodes complete (supports multi-task).

        Returns
        -------
        int : number of successful episodes in this batch
        """
        terminal_ids = terminal_index.nonzero().flatten()
        terminal_nums = len(terminal_ids)
        success_buf = infos["success"][terminal_index]
        success_ids = terminal_ids[success_buf.to(torch.bool)]

        # Update global buffers
        update_tensor_buffer(self.eps_metrics["eps_r"], self.step_metrics['eps_r'][terminal_index])
        update_tensor_buffer(self.eps_metrics["eps_success"], infos["success"][terminal_index])

        if len(success_ids) > 0:
            update_tensor_buffer(self.eps_metrics["eps_time"], infos["eps_time"][success_ids])
            update_tensor_buffer(self.eps_metrics["eps_time_goal"], infos["eps_time_goal"][success_ids])
            update_tensor_buffer(self.eps_metrics["eps_time_p"], infos["eps_time_p"][success_ids])
            update_tensor_buffer(self.eps_metrics["eps_max_inst"], infos["eps_max_scevel"][success_ids])
            update_tensor_buffer(self.eps_metrics["eps_lim_inst"], infos["eps_lim_scevel"][success_ids])
            update_tensor_buffer(self.eps_metrics["eps_sum_inst"], infos["eps_sum_inst"][success_ids])
            update_tensor_buffer(self.eps_metrics["interaction_time"], infos["interaction_time"][success_ids])

        # === Per-task tracking (for multi-task evaluation) ===
        if self.is_multi_task and terminal_nums > 0:
            task_ids = task_indices[terminal_ids]
            task_idxs = self.tid_to_tidx[task_ids]

            terminal_metrics = {
                "eps_r": self.step_metrics['eps_r'][terminal_index],
                "eps_success": infos["success"][terminal_index].float(),
                "eps_time": infos["eps_time"][terminal_index].float(),
                "eps_time_p": infos["eps_time_p"][terminal_index].float(),
                "eps_max_inst": infos["eps_max_scevel"][terminal_index].float(),
                "eps_sum_inst": infos["eps_sum_inst"][terminal_index].float(),
            }

            for task_idx in task_idxs.unique():
                task_mask = (task_idxs == task_idx)
                num_finished = task_mask.sum().item()
                if num_finished == 0:
                    continue

                self.per_task_episode_count[task_idx] += num_finished

                task_success_mask = success_buf[task_mask].bool()
                num_success = task_success_mask.sum().item()
                self.per_task_success_count[task_idx] += num_success

                for metric in self.per_task_metrics:
                    if metric in self.success_only_metrics:
                        if num_success > 0:
                            vals = terminal_metrics[metric][task_mask][task_success_mask]
                            for v in vals:
                                ptr = self.per_task_buf_ptr[task_idx] % self.per_task_buf_len
                                self.per_task_buf[metric][task_idx, ptr] = v
                                self.per_task_buf_ptr[task_idx] += 1
                    else:
                        vals = terminal_metrics[metric][task_mask]
                        for v in vals:
                            ptr = self.per_task_buf_ptr[task_idx] % self.per_task_buf_len
                            self.per_task_buf[metric][task_idx, ptr] = v

        # Reset step accumulators for terminal envs
        for key in self.step_metrics.keys():
            self.step_metrics[key][terminal_index] = 0.

        return len(success_ids)

    # ------------------------------------------------------------------
    # Averaging / aggregation
    # ------------------------------------------------------------------

    def compute_average_metrics(self, num_episodes, num_success_eps):
        """Compute average metrics (global and per-task)."""
        for key in self.eps_metrics_avg.keys():
            eps_index = num_episodes if key in ["eps_r", "eps_success"] else num_success_eps
            if eps_index > 0:
                self.eps_metrics_avg[key] = torch.mean(self.eps_metrics[key][-eps_index:]).item()
                std_v = torch.std(self.eps_metrics[key][-eps_index:])
                self.eps_metrics_std[key] = std_v.item() if not torch.isnan(std_v) else 0.
            else:
                self.eps_metrics_avg[key] = 0.
                self.eps_metrics_std[key] = 0.

        if self.is_multi_task:
            for task_idx in range(self.num_tasks):
                num_eps = min(self.per_task_episode_count[task_idx].item(), self.per_task_buf_len)
                if num_eps > 0:
                    for metric in self.per_task_metrics:
                        self.per_task_avg[metric][task_idx] = self.per_task_buf[metric][task_idx, :num_eps].mean()

            for metric in self.per_task_metrics:
                valid_tasks = self.per_task_episode_count > 0
                if valid_tasks.any():
                    self.avg_task_metrics[metric] = self.per_task_avg[metric][valid_tasks].mean().item()

    def update_speed_time_dict(self, cur_goal_speed):
        """Update speed and time dictionary."""
        self.speed_and_time_dict["time_ratio"].append(cur_goal_speed)
        self.speed_and_time_dict["time_used"].append([self.eps_metrics_avg["eps_time"], self.eps_metrics_std["eps_time"]])
        self.speed_and_time_dict["time_goal"].append([self.eps_metrics_avg["eps_time_goal"], self.eps_metrics_std["eps_time_goal"]])
        self.speed_and_time_dict["time_mismatch"].append([self.eps_metrics_avg["eps_time_p"], self.eps_metrics_std["eps_time_p"]])
        self.speed_and_time_dict["max_inst"].append([self.eps_metrics_avg["eps_max_inst"], self.eps_metrics_std["eps_max_inst"]])
        self.speed_and_time_dict["thred_inst"].append([self.eps_metrics_avg["eps_lim_inst"], self.eps_metrics_std["eps_lim_inst"]])
        self.speed_and_time_dict["sum_inst"].append([self.eps_metrics_avg["eps_sum_inst"], self.eps_metrics_std["eps_sum_inst"]])
        self.speed_and_time_dict["interaction_time"].append([self.eps_metrics_avg["interaction_time"], self.eps_metrics_std["interaction_time"]])
        self.speed_and_time_dict["success_rate"].append(self.eps_metrics_avg["eps_success"])

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print_per_task_results(self):
        """Print per-task evaluation results in a table format."""
        if not self.is_multi_task:
            return

        try:
            from MTBench.isaacgymenvs.tasks.franka.vec_task.franka_base import TASK_IDX_TO_NAME
        except ImportError:
            TASK_IDX_TO_NAME = {}

        print("\n" + "=" * 80)
        print("PER-TASK EVALUATION RESULTS")
        print("=" * 80)

        headers = ["Task ID", "Task Name", "Episodes", "Success", "Success Rate", "Avg Reward", "Avg Time"]
        rows = []

        for i, tid in enumerate(self.unique_task_ids):
            tid_int = tid.item()
            task_name = TASK_IDX_TO_NAME.get(tid_int, f"Task_{tid_int}")[:20]
            num_eps = self.per_task_episode_count[i].item()
            num_success = self.per_task_success_count[i].item()
            success_rate = num_success / max(num_eps, 1) * 100
            avg_reward = self.per_task_avg["eps_r"][i].item()
            avg_time = self.per_task_avg["eps_time"][i].item() if num_success > 0 else 0.

            rows.append([
                tid_int,
                task_name,
                num_eps,
                num_success,
                f"{success_rate:.1f}%",
                f"{avg_reward:.2f}",
                f"{avg_time:.3f}s"
            ])

        print(tabulate(rows, headers=headers, tablefmt="grid"))

        total_eps = self.per_task_episode_count.sum().item()
        total_success = self.per_task_success_count.sum().item()
        avg_success_rate = total_success / max(total_eps, 1) * 100
        print(f"\nTotal: {total_eps} episodes, {total_success} successes ({avg_success_rate:.1f}%)")
        print(f"Average across tasks - Success: {self.avg_task_metrics.get('eps_success', 0) * 100:.1f}%, "
              f"Reward: {self.avg_task_metrics.get('eps_r', 0):.2f}")
        print("=" * 80 + "\n")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(self, num_episodes, machine_time, infos):
        """Save evaluation results."""
        if not self.args.saving:
            return

        csv_result = {
            "target_episodes": self.args.requested_target_episodes,
            "num_episodes": num_episodes,
            "max_trials": self.args.target_episodes,
            "success_rate": self.eps_metrics_avg["eps_success"],
            "avg_reward": self.eps_metrics_avg["eps_r"],
            "avg_eps_time": self.eps_metrics_avg["eps_time"],
            "std_eps_time": self.eps_metrics_std["eps_time"],
            "avg_eps_time_goal": self.eps_metrics_avg["eps_time_goal"],
            "std_eps_time_goal": self.eps_metrics_std["eps_time_goal"],
            "avg_eps_time_mismatch": self.eps_metrics_avg["eps_time_p"],
            "std_eps_time_mismatch": self.eps_metrics_std["eps_time_p"],
            "avg_sum_eps_inst": self.eps_metrics_avg["eps_sum_inst"],
            "machine_time": machine_time
        }

        if self.is_multi_task:
            csv_result["avg_task_success_rate"] = self.avg_task_metrics.get("eps_success", 0)
            csv_result["avg_task_reward"] = self.avg_task_metrics.get("eps_r", 0)

        write_csv_line(self.args.csv_file_path, csv_result)
        print(f"Saved evaluation CSV to {self.args.csv_file_path}")

        meta_data = {
            "episode": num_episodes,
            "episode_success": self.eps_metrics["eps_success"][-num_episodes:].cpu().tolist(),
            "episode_time": self.eps_metrics["eps_time"][-num_episodes:].cpu().tolist(),
            "episode_time_goal": self.eps_metrics["eps_time_goal"][-num_episodes:].cpu().tolist(),
            "speed_and_time": self.speed_and_time_dict
        }

        if self.is_multi_task:
            meta_data["per_task_results"] = {
                "task_ids": self.unique_task_ids.cpu().tolist(),
                "episode_counts": self.per_task_episode_count.cpu().tolist(),
                "success_counts": self.per_task_success_count.cpu().tolist(),
                "avg_success_rate": [self.per_task_avg["eps_success"][i].item() for i in range(self.num_tasks)],
                "avg_reward": [self.per_task_avg["eps_r"][i].item() for i in range(self.num_tasks)],
                "avg_time": [self.per_task_avg["eps_time"][i].item() for i in range(self.num_tasks)],
            }

        save_json(meta_data, os.path.join(self.args.trajectory_dir, f"meta_data.json"))

    # ------------------------------------------------------------------
    # Resets
    # ------------------------------------------------------------------

    def reset_evaluation_metrics(self):
        """Reset all metrics for a new evaluation run."""
        for key in self.step_metrics:
            self.step_metrics[key][:] = 0.
        for key in self.eps_metrics:
            self.eps_metrics[key][:] = 0.
        if self.is_multi_task:
            self.reset_per_task_metrics()

    def reset_step_metrics(self):
        """Reset step-level metrics (used during warmup)."""
        for key in self.step_metrics:
            self.step_metrics[key][:] = 0.

    def reset_per_task_metrics(self):
        """Reset per-task metrics for a new evaluation run."""
        self.per_task_episode_count.zero_()
        self.per_task_success_count.zero_()
        self.per_task_buf_ptr.zero_()
        for metric in self.per_task_metrics:
            self.per_task_buf[metric].zero_()
            self.per_task_avg[metric].zero_()
        for metric in self.avg_task_metrics:
            self.avg_task_metrics[metric] = 0.0
