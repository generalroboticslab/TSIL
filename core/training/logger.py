"""Training metric logging to local JSONL and optional Weights & Biases."""

from __future__ import annotations

import json
import os

import wandb


class TrainingLogger:
    """Build metric dictionaries once, then write them locally and to wandb."""

    _wandb_active = False

    def __init__(self, args):
        self.args = args
        self.local_history_path = os.path.join(args.instance_dir, "plot_metrics_history.jsonl")
        self._latest_plot_metrics = {}
        self._setup_wandb()

    def _setup_wandb(self):
        config = dict(
            Name=self.args.env_name,
            algorithm="PPO Continuous",
            num_envs=self.args.num_envs,
            lr=self.args.lr,
            gamma=self.args.gamma,
            alpha=self.args.ent_coef,
            deterministic=self.args.deterministic,
            seq_length=self.args.seq_length,
            random_policy=self.args.random_policy,
        )

        if not (self.args.saving and self.args.wandb):
            self.args.wandb_run_id = None
            self.args.wandb_run_path = None
            return

        wandb_kwargs = dict(
            project=self.args.wandb_project or self.args.env_name,
            config=config,
            name=self.args.run_name,
        )
        if self.args.wandb_entity:
            wandb_kwargs["entity"] = self.args.wandb_entity
        run = wandb.init(**wandb_kwargs)
        TrainingLogger._wandb_active = True
        self.args.wandb_run_id = run.id
        self.args.wandb_run_path = run.path

    @staticmethod
    def _scalar(value):
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _base_metrics(self, metrics):
        ctrl_dt = float(getattr(metrics.args, "dt", 1.0)) * float(
            getattr(metrics.args, "control_freq_inv", 1)
        )
        return {
            "iteration": int(metrics.global_update_iter),
            "steps": int(metrics.global_step),
            "episodes": int(metrics.global_episodes),
            "misc/iterations": int(metrics.global_update_iter),
            "misc/steps": int(metrics.global_step),
            "misc/episodes": int(metrics.global_episodes),
            "misc/success_episodes": int(getattr(metrics, "success_episodes", 0)),
            "misc/interaction_time": float(metrics.global_step) * ctrl_dt,
        }

    def _log_wandb(self, values, commit=True):
        if self.args.saving and self.args.wandb:
            wandb.log(values, commit=commit)

    def _append_local(self, values):
        if not self.args.saving:
            return
        os.makedirs(self.args.instance_dir, exist_ok=True)
        record = {key: self._scalar(value) for key, value in values.items()}
        with open(self.local_history_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_episode_metrics(self, metrics, is_multi_task):
        if not self.args.saving:
            return

        values = {
            **self._base_metrics(metrics),
            "reward/curriculum_ratio": metrics.curri_ratio,
            "reward/avg_time2end_upper": metrics.avg_time2end_upper,
        }

        if metrics.ready_to_record:
            if metrics.avg_buffer_reset:
                metrics.reward_episodes += metrics.curri_episodes
                metrics.reward_update_iters += metrics.curri_update_iters
                metrics.reward_steps += metrics.curri_steps
                metrics.avg_buffer_reset = False

            values.update({
                "misc/r_iterations": metrics.global_update_iter - metrics.reward_update_iters,
                "misc/r_episodes": metrics.global_episodes - metrics.reward_episodes,
                "misc/r_steps": metrics.global_step - metrics.reward_steps,
                "reward/eps_G_score": metrics.cur_eps_G_score,
                "reward/best_eps_G_score": metrics.best_eps_G_score,
            })
            for metric in metrics.per_task_metrics:
                values[f"reward/{metric}"] = metrics.avg_task_metrics[metric]

            if is_multi_task:
                for i, tid in enumerate(metrics.unique_task_ids):
                    tid_int = tid.item()
                    group_name = f"tasks_{(tid_int // 10) * 10:02d}-{(tid_int // 10) * 10 + 9:02d}"
                    for metric in metrics.per_task_metrics:
                        values[f"{group_name}/T{tid_int:02d}_{metric}"] = metrics.per_task_avg[metric][i].item()

        self._latest_plot_metrics.update(values)
        self._log_wandb(values, commit=self.args.random_policy)
        if self.args.random_policy:
            self._append_local(values)

    def log_training_metrics(self, metrics, agent, optimizer, envs,
                             pg_loss, v_loss, entropy_mean, approx_kl,
                             mb_advantages, explained_var,
                             v_loss_c=None, v_loss_t=None, L_viol=None, bd_loss=0.,
                             extra_metrics=None):
        if not self.args.saving:
            return

        values = {
            **metrics.performance_metrics,
            **self._base_metrics(metrics),
            "train/learning_rate": optimizer.param_groups[0]["lr"],
            "train/dense_reward_dropout_p": float(getattr(self.args, "dense_reward_dropout_p", 0.0)),
            "train/effective_dense_reward_dropout_p": float(
                getattr(
                    self.args,
                    "effective_dense_reward_dropout_p",
                    getattr(self.args, "dense_reward_dropout_p", 0.0),
                )
            ),
            "train/dense_reward_dropout_mode": str(getattr(self.args, "dense_reward_dropout_mode", "step")),
            "train/policy_grad_noise_scale": float(getattr(self.args, "policy_grad_noise_scale", 0.0)),
            "train/effective_policy_grad_noise_scale": float(
                getattr(
                    self.args,
                    "effective_policy_grad_noise_scale",
                    getattr(self.args, "policy_grad_noise_scale", 0.0),
                )
            ),
            "train/delayed_stress": int(bool(getattr(self.args, "delayed_stress", False))),
            "train/stress_active": int(bool(getattr(self.args, "stress_active", True))),
            "train/critic_loss": v_loss.item(),
            "train/policy_loss": pg_loss.item(),
            "train/bound_loss": bd_loss.item() if hasattr(bd_loss, "item") else bd_loss,
            "train/approx_kl": approx_kl.item(),
            "train/advantages": mb_advantages.mean().item(),
            "train/explained_variance": explained_var,
            "train/entropy_coef": metrics.cur_ent,
            "train/epstimeRewardScale": envs.cfg["r_epstime_scale"],
            "train/scevelRewardScale": envs.cfg["r_scene_vel_scale"],
            "signal/top10pct_Rdense_sr": metrics.current_signal_metrics["top10pct_Rdense_sr"],
            "signal/succ_posadv_ratio": metrics.current_signal_metrics["succ_posadv_ratio"],
            "signal/fast_succ_posadv_ratio": metrics.current_signal_metrics["fast_succ_posadv_ratio"],
            "signal/succ_posadv_step_frac": metrics.current_signal_metrics["succ_posadv_step_frac"],
            "signal/adv_ep_used": metrics.current_signal_metrics["adv_ep_used"],
            "signal/episode_signal_count": metrics.current_signal_metrics["episode_signal_count"],
            "signal/success_eps_time": metrics.current_signal_metrics["success_eps_time"],
            "signal/adv_episodes_used": metrics.current_signal_metrics["adv_episodes_used"],
            "signal/adv_raw_abs_mean": metrics.current_signal_metrics["adv_raw_abs_mean"],
            "signal/adv_raw_abs_max": metrics.current_signal_metrics["adv_raw_abs_max"],
            "signal/adv_raw_std": metrics.current_signal_metrics["adv_raw_std"],
            "signal/sil_fast_success_rate": metrics.current_signal_metrics.get("sil_fast_success_rate", 0.0),
            "signal/sil_fast_success_count": metrics.current_signal_metrics.get("sil_fast_success_count", 0),
            "signal/fast_success_rate": metrics.current_signal_metrics.get("sil_fast_success_rate", 0.0),
            "signal/fast_success_count": metrics.current_signal_metrics.get("sil_fast_success_count", 0),
            "signal/first_fast_revisit_steps": metrics.current_signal_metrics.get("sil_first_revisit_steps"),
            "signal/fast_revisit_gap_steps": metrics.current_signal_metrics.get("sil_revisit_gap_steps"),
            "signal/sil_first_revisit_steps": metrics.current_signal_metrics.get("sil_first_revisit_steps"),
            "signal/sil_revisit_gap_steps": metrics.current_signal_metrics.get("sil_revisit_gap_steps"),
            "signal/sil_archive_best_eps_time": metrics.current_signal_metrics.get("sil_archive_best_eps_time"),
            "signal/sil_revisit_anchor_eps_time": metrics.current_signal_metrics.get("sil_revisit_anchor_eps_time"),
            "signal/sil_revisit_anchor_steps": metrics.current_signal_metrics.get("sil_revisit_anchor_steps"),
            "train/replay_mode": metrics.last_replay_stats["replay_mode"],
            "train/replay_active": metrics.last_replay_stats["replay_active"],
            "train/replay_has_demo": metrics.last_replay_stats["replay_has_demo"],
            "train/replay_delayed": metrics.last_replay_stats.get("replay_delayed", 0),
            "train/replay_start_iter": metrics.last_replay_stats.get("replay_start_iter", 0),
            "train/replay_steps": metrics.last_replay_stats["replay_steps"],
            "train/replay_policy_loss": metrics.last_replay_stats["replay_policy_loss"],
            "train/replay_value_loss": metrics.last_replay_stats["replay_value_loss"],
            "train/replay_total_loss": metrics.last_replay_stats["replay_total_loss"],
            "train/replay_positive_frac": metrics.last_replay_stats["replay_positive_frac"],
            "train/replay_positive_available_frac": metrics.last_replay_stats["replay_positive_available_frac"],
            "train/replay_sampled_positive_frac": metrics.last_replay_stats["replay_sampled_positive_frac"],
            "train/replay_dataset_steps": metrics.last_replay_stats["replay_dataset_steps"],
            "train/replay_dataset_trajectories": metrics.last_replay_stats["replay_dataset_trajectories"],
            "train/replay_unique_trajectories": metrics.last_replay_stats["replay_unique_trajectories"],
            "train/replay_fastest_tasks": metrics.last_replay_stats["replay_fastest_tasks"],
            "train/replay_return_fallback_tasks": metrics.last_replay_stats["replay_return_fallback_tasks"],
            "train/replay_success_pool_trajectories": metrics.last_replay_stats.get("replay_success_pool_trajectories", 0),
            "train/replay_fallback_pool_trajectories": metrics.last_replay_stats.get("replay_fallback_pool_trajectories", 0),
            "train/replay_sample_success_frac": metrics.last_replay_stats.get("replay_sample_success_frac", 0.0),
            "train/replay_sample_fallback_frac": metrics.last_replay_stats.get("replay_sample_fallback_frac", 0.0),
            "train/replay_positive_success_frac": metrics.last_replay_stats.get("replay_positive_success_frac", 0.0),
            "train/replay_positive_fallback_frac": metrics.last_replay_stats.get("replay_positive_fallback_frac", 0.0),
            "train/replay_gap_success_mean": metrics.last_replay_stats.get("replay_gap_success_mean", 0.0),
            "train/replay_gap_fallback_mean": metrics.last_replay_stats.get("replay_gap_fallback_mean", 0.0),
            "train/replay_policy_loss_success": metrics.last_replay_stats.get("replay_policy_loss_success", 0.0),
            "train/replay_policy_loss_fallback": metrics.last_replay_stats.get("replay_policy_loss_fallback", 0.0),
            "train/replay_memory_logp": metrics.last_replay_stats["replay_memory_logp"],
            "train/replay_memory_weighted_logp": metrics.last_replay_stats["replay_memory_weighted_logp"],
            "train/replay_memory_positive_gap_mean": metrics.last_replay_stats["replay_memory_positive_gap_mean"],
            "train/sil_cos_ppo_memory": metrics.last_replay_stats["sil_cos_ppo_memory"],
            "train/sil_cos_sil_memory": metrics.last_replay_stats["sil_cos_sil_memory"],
            "train/sil_cos_joint_memory": metrics.last_replay_stats["sil_cos_joint_memory"],
            "train/sil_alignment_gain": metrics.last_replay_stats["sil_alignment_gain"],
            "train/sil_landscape_file": metrics.last_replay_stats["sil_landscape_file"],
            "train/sil_train": int(bool(getattr(self.args, "sil_train", True))),
            "train/sil_revisit_reference_count": metrics.last_replay_stats.get("sil_revisit_reference_count", 0),
            "train/sil_revisit_logp_topk_mean": metrics.last_replay_stats.get("sil_revisit_logp_topk_mean", 0.0),
            "train/sil_revisit_nll_topk_mean": metrics.last_replay_stats.get("sil_revisit_nll_topk_mean", 0.0),
            "train/sil_revisit_nll_topk_std": metrics.last_replay_stats.get("sil_revisit_nll_topk_std", 0.0),
            "train/sil_revisit_reference_count_fastest1": metrics.last_replay_stats.get("sil_revisit_reference_count_fastest1", 0),
            "train/sil_revisit_logp_fastest1_mean": metrics.last_replay_stats.get("sil_revisit_logp_fastest1_mean", 0.0),
            "train/sil_revisit_nll_fastest1_mean": metrics.last_replay_stats.get("sil_revisit_nll_fastest1_mean", 0.0),
            "train/sil_revisit_nll_fastest1_std": metrics.last_replay_stats.get("sil_revisit_nll_fastest1_std", 0.0),
            "train/sil_supervised_logp_topk_mean": metrics.last_replay_stats.get("sil_supervised_logp_topk_mean", 0.0),
            "train/sil_supervised_nll_topk_mean": metrics.last_replay_stats.get("sil_supervised_nll_topk_mean", 0.0),
            "train/sil_supervised_weight_frac": metrics.last_replay_stats.get("sil_supervised_weight_frac", 0.0),
            "train/sil_supervised_logp_fastest1_mean": metrics.last_replay_stats.get("sil_supervised_logp_fastest1_mean", 0.0),
            "train/sil_supervised_nll_fastest1_mean": metrics.last_replay_stats.get("sil_supervised_nll_fastest1_mean", 0.0),
            "train/sil_supervised_weight_fastest1_frac": metrics.last_replay_stats.get("sil_supervised_weight_fastest1_frac", 0.0),
            "train/sil_archive_best_eps_time": metrics.last_replay_stats.get("sil_archive_best_eps_time", 0.0),
            "train/sil_archive_best_steps": metrics.last_replay_stats.get("sil_archive_best_steps", 0),
        }
        if extra_metrics:
            values.update(extra_metrics)

        entropy_log = agent.prob_entropy.mean(dim=0)
        values.update({
            "entropy/entropy": entropy_mean.item(),
            "entropy/entropy_x": entropy_log[0].item(),
            "entropy/entropy_y": entropy_log[1].item(),
            "entropy/entropy_z": entropy_log[2].item(),
            "entropy/entropy_Rz": entropy_log[3].item(),
        })
        if self.args.beta:
            alpha = agent.probs.concentration0.mean(dim=0)
            beta = agent.probs.concentration1.mean(dim=0)
            values.update({
                "concentration_a/alpha_x": alpha[0].item(),
                "concentration_a/alpha_y": alpha[1].item(),
                "concentration_a/alpha_z": alpha[2].item(),
                "concentration_a/alpha_Rz": alpha[3].item(),
                "concentration_b/beta_x": beta[0].item(),
                "concentration_b/beta_y": beta[1].item(),
                "concentration_b/beta_z": beta[2].item(),
                "concentration_b/beta_Rz": beta[3].item(),
            })
        else:
            act_mu_log = agent.probs.mean
            values.update({
                "action/max_mu_x": act_mu_log.max().item(),
                "action/min_mu_x": act_mu_log.min().item(),
            })
        if self.args.use_cost:
            values.update({
                "train/critic_cost_loss": v_loss_c.item(),
                "train/actor_cost_loss": L_viol.item(),
            })
        if self.args.use_timeawareness:
            values["train/critic_time_loss"] = v_loss_t.item()

        plot_values = {**self._latest_plot_metrics, **values}
        self._latest_plot_metrics.update(values)
        self._append_local(plot_values)
        metrics.save_meta_data_snapshot()
        self._log_wandb(values)

    @staticmethod
    def finish():
        if TrainingLogger._wandb_active:
            wandb.finish()
            TrainingLogger._wandb_active = False


WandbLogger = TrainingLogger

__all__ = ["TrainingLogger", "WandbLogger"]
