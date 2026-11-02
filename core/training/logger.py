"""Training metric logging to local JSONL and optional Weights & Biases."""

from __future__ import annotations

import json
import os


class TrainingLogger:
    """Build metric dictionaries once, then write them locally and to wandb."""

    _wandb_active = False
    _wandb = None

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
        wandb = self._wandb_module()

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

    @classmethod
    def _wandb_module(cls):
        if cls._wandb is None:
            import wandb

            cls._wandb = wandb
        return cls._wandb

    @staticmethod
    def _scalar(value):
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _distribution_metrics(self, agent, entropy_mean):
        """Return metrics that match the agent's action distribution."""
        values = {"entropy/entropy": entropy_mean.item()}
        if getattr(agent, "is_categorical_policy", False):
            probabilities = agent.probs.probs
            values.update({
                "action/max_probability": probabilities.max().item(),
                "action/min_probability": probabilities.min().item(),
            })
            return values

        entropy_log = agent.prob_entropy.mean(dim=0).reshape(-1)
        component_names = ("x", "y", "z", "Rz")
        for index, component_entropy in enumerate(entropy_log):
            name = (
                component_names[index]
                if index < len(component_names)
                else f"dim_{index}"
            )
            values[f"entropy/entropy_{name}"] = component_entropy.item()

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
            action_mean = agent.probs.mean
            values.update({
                "action/max_mu_x": action_mean.max().item(),
                "action/min_mu_x": action_mean.min().item(),
            })
        return values

    def _base_metrics(self, metrics):
        ctrl_dt = float(getattr(metrics.args, "dt", 1.0)) * float(
            getattr(metrics.args, "control_freq_inv", 1)
        )
        if getattr(metrics.args, "benchmark_backend", None) == "minigrid":
            ctrl_dt = 1.0
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
            self._wandb_module().log(values, commit=commit)

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
            "train/policy_grad_noise_mode": str(
                getattr(self.args, "policy_grad_noise_mode", "additive")
            ),
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
            "signal/outcome_failure_episode_frac": metrics.current_signal_metrics.get(
                "outcome_failure_episode_frac", 0.0,
            ),
            "signal/outcome_failure_transition_share": metrics.current_signal_metrics.get(
                "outcome_failure_transition_share", 0.0,
            ),
            "signal/outcome_failure_posadv_mass_share": metrics.current_signal_metrics.get(
                "outcome_failure_posadv_mass_share", 0.0,
            ),
            "signal/outcome_failure_raw_posadv_mass_share": metrics.current_signal_metrics.get(
                "outcome_failure_raw_posadv_mass_share", 0.0,
            ),
            "signal/outcome_failure_positive_step_frac": metrics.current_signal_metrics.get(
                "outcome_failure_positive_step_frac", 0.0,
            ),
            "signal/outcome_failure_joint_logp_delta_mean": metrics.current_signal_metrics.get(
                "outcome_failure_joint_logp_delta_mean", 0.0,
            ),
            "signal/outcome_failure_joint_logp_increase_frac": metrics.current_signal_metrics.get(
                "outcome_failure_joint_logp_increase_frac", 0.0,
            ),
            "signal/outcome_high_dense_failure_episode_count": metrics.current_signal_metrics.get(
                "outcome_high_dense_failure_episode_count", 0,
            ),
            "signal/outcome_high_dense_failure_transition_share": metrics.current_signal_metrics.get(
                "outcome_high_dense_failure_transition_share", 0.0,
            ),
            "signal/outcome_high_dense_failure_posadv_mass_share": metrics.current_signal_metrics.get(
                "outcome_high_dense_failure_posadv_mass_share", 0.0,
            ),
            "signal/outcome_high_dense_failure_raw_posadv_mass_share": metrics.current_signal_metrics.get(
                "outcome_high_dense_failure_raw_posadv_mass_share", 0.0,
            ),
            "signal/outcome_high_dense_failure_positive_step_frac": metrics.current_signal_metrics.get(
                "outcome_high_dense_failure_positive_step_frac", 0.0,
            ),
            "signal/outcome_high_dense_failure_joint_logp_delta_mean": metrics.current_signal_metrics.get(
                "outcome_high_dense_failure_joint_logp_delta_mean", 0.0,
            ),
            "signal/outcome_high_dense_failure_joint_logp_increase_frac": metrics.current_signal_metrics.get(
                "outcome_high_dense_failure_joint_logp_increase_frac", 0.0,
            ),
            "signal/outcome_high_dense_failure_dense_per_step_mean": metrics.current_signal_metrics.get(
                "outcome_high_dense_failure_dense_per_step_mean", 0.0,
            ),
            "signal/outcome_success_joint_logp_delta_mean": metrics.current_signal_metrics.get(
                "outcome_success_joint_logp_delta_mean", 0.0,
            ),
            "signal/outcome_success_joint_logp_increase_frac": metrics.current_signal_metrics.get(
                "outcome_success_joint_logp_increase_frac", 0.0,
            ),
            "signal/outcome_gradient_measured": metrics.current_signal_metrics.get(
                "outcome_gradient_measured", 0,
            ),
            "signal/outcome_gradient_success_sample_count": metrics.current_signal_metrics.get(
                "outcome_gradient_success_sample_count", 0,
            ),
            "signal/outcome_gradient_failure_sample_count": metrics.current_signal_metrics.get(
                "outcome_gradient_failure_sample_count", 0,
            ),
            "signal/outcome_gradient_high_dense_failure_sample_count": metrics.current_signal_metrics.get(
                "outcome_gradient_high_dense_failure_sample_count", 0,
            ),
            "signal/outcome_gradient_success_norm": metrics.current_signal_metrics.get(
                "outcome_gradient_success_norm", 0.0,
            ),
            "signal/outcome_gradient_failure_norm": metrics.current_signal_metrics.get(
                "outcome_gradient_failure_norm", 0.0,
            ),
            "signal/outcome_gradient_high_dense_failure_norm": metrics.current_signal_metrics.get(
                "outcome_gradient_high_dense_failure_norm", 0.0,
            ),
            "signal/outcome_gradient_all_norm": metrics.current_signal_metrics.get(
                "outcome_gradient_all_norm", 0.0,
            ),
            "signal/outcome_gradient_failure_success_cosine": metrics.current_signal_metrics.get(
                "outcome_gradient_failure_success_cosine", 0.0,
            ),
            "signal/outcome_gradient_high_dense_failure_success_cosine": metrics.current_signal_metrics.get(
                "outcome_gradient_high_dense_failure_success_cosine", 0.0,
            ),
            "signal/outcome_gradient_high_dense_failure_all_cosine": metrics.current_signal_metrics.get(
                "outcome_gradient_high_dense_failure_all_cosine", 0.0,
            ),
            "signal/rollout_timeout_transition_count": metrics.current_signal_metrics.get(
                "rollout_timeout_transition_count", 0,
            ),
            "signal/rollout_terminal_transition_count": metrics.current_signal_metrics.get(
                "rollout_terminal_transition_count", 0,
            ),
            "signal/rollout_timeout_transition_frac": metrics.current_signal_metrics.get(
                "rollout_timeout_transition_frac", 0.0,
            ),
            "signal/rollout_terminal_transition_frac": metrics.current_signal_metrics.get(
                "rollout_terminal_transition_frac", 0.0,
            ),
            "signal/rollout_timeout_next_value_mean": metrics.current_signal_metrics.get(
                "rollout_timeout_next_value_mean", 0.0,
            ),
            "signal/rollout_timeout_next_value_1_mean": metrics.current_signal_metrics.get(
                "rollout_timeout_next_value_1_mean", 0.0,
            ),
            "signal/rollout_timeout_next_value_2_mean": metrics.current_signal_metrics.get(
                "rollout_timeout_next_value_2_mean", 0.0,
            ),
            "signal/rollout_timeout_next_value_min_mean": metrics.current_signal_metrics.get(
                "rollout_timeout_next_value_min_mean", 0.0,
            ),
            "signal/rollout_timeout_critic_disagreement_mean": metrics.current_signal_metrics.get(
                "rollout_timeout_critic_disagreement_mean", 0.0,
            ),
            "signal/rollout_timeout_state_value_mean": metrics.current_signal_metrics.get(
                "rollout_timeout_state_value_mean", 0.0,
            ),
            "signal/rollout_timeout_reward_mean": metrics.current_signal_metrics.get(
                "rollout_timeout_reward_mean", 0.0,
            ),
            "signal/rollout_timeout_return_vb_mean": metrics.current_signal_metrics.get(
                "rollout_timeout_return_vb_mean", 0.0,
            ),
            "signal/rollout_timeout_return_novb_mean": metrics.current_signal_metrics.get(
                "rollout_timeout_return_novb_mean", 0.0,
            ),
            "signal/rollout_timeout_return_delta_mean": metrics.current_signal_metrics.get(
                "rollout_timeout_return_delta_mean", 0.0,
            ),
            "signal/rollout_timeout_adv_vb_mean": metrics.current_signal_metrics.get(
                "rollout_timeout_adv_vb_mean", 0.0,
            ),
            "signal/rollout_timeout_adv_novb_mean": metrics.current_signal_metrics.get(
                "rollout_timeout_adv_novb_mean", 0.0,
            ),
            "signal/rollout_timeout_adv_delta_mean": metrics.current_signal_metrics.get(
                "rollout_timeout_adv_delta_mean", 0.0,
            ),
            "signal/rollout_timeout_posadv_vb_frac": metrics.current_signal_metrics.get(
                "rollout_timeout_posadv_vb_frac", 0.0,
            ),
            "signal/rollout_timeout_posadv_novb_frac": metrics.current_signal_metrics.get(
                "rollout_timeout_posadv_novb_frac", 0.0,
            ),
            "signal/rollout_timeout_posadv_normalized_frac": metrics.current_signal_metrics.get(
                "rollout_timeout_posadv_normalized_frac", 0.0,
            ),
            "signal/rollout_timeout_policy_grad_measured": metrics.current_signal_metrics.get(
                "rollout_timeout_policy_grad_measured", 0,
            ),
            "signal/rollout_timeout_policy_grad_norm": metrics.current_signal_metrics.get(
                "rollout_timeout_policy_grad_norm", 0.0,
            ),
            "signal/rollout_policy_grad_norm": metrics.current_signal_metrics.get(
                "rollout_policy_grad_norm", 0.0,
            ),
            "signal/rollout_timeout_policy_grad_norm_ratio": metrics.current_signal_metrics.get(
                "rollout_timeout_policy_grad_norm_ratio", 0.0,
            ),
            "signal/rollout_timeout_policy_grad_cosine": metrics.current_signal_metrics.get(
                "rollout_timeout_policy_grad_cosine", 0.0,
            ),
            "signal/rollout_timeout_value_mse_vb": metrics.current_signal_metrics.get(
                "rollout_timeout_value_mse_vb", 0.0,
            ),
            "signal/rollout_timeout_value_mse_novb": metrics.current_signal_metrics.get(
                "rollout_timeout_value_mse_novb", 0.0,
            ),
            "signal/rollout_return_delta_mean": metrics.current_signal_metrics.get(
                "rollout_return_delta_mean", 0.0,
            ),
            "signal/rollout_adv_delta_mean": metrics.current_signal_metrics.get(
                "rollout_adv_delta_mean", 0.0,
            ),
            "signal/rollout_value_mse_vb": metrics.current_signal_metrics.get(
                "rollout_value_mse_vb", 0.0,
            ),
            "signal/rollout_value_mse_novb": metrics.current_signal_metrics.get(
                "rollout_value_mse_novb", 0.0,
            ),
            "signal/sil_fast_success_rate": metrics.current_signal_metrics.get("sil_fast_success_rate", 0.0),
            "signal/sil_fast_success_count": metrics.current_signal_metrics.get("sil_fast_success_count", 0),
            "signal/normalized_completion_time": metrics.current_signal_metrics.get("normalized_completion_time"),
            "signal/fast_success_rate": metrics.current_signal_metrics.get("sil_fast_success_rate", 0.0),
            "signal/fast_success_count": metrics.current_signal_metrics.get("sil_fast_success_count", 0),
            "signal/first_fast_revisit_steps": metrics.current_signal_metrics.get("sil_first_revisit_steps"),
            "signal/fast_revisit_gap_steps": metrics.current_signal_metrics.get("sil_revisit_gap_steps"),
            "signal/sil_first_revisit_steps": metrics.current_signal_metrics.get("sil_first_revisit_steps"),
            "signal/sil_revisit_gap_steps": metrics.current_signal_metrics.get("sil_revisit_gap_steps"),
            "signal/sil_archive_best_eps_time": metrics.current_signal_metrics.get("sil_archive_best_eps_time"),
            "signal/sil_revisit_anchor_eps_time": metrics.current_signal_metrics.get("sil_revisit_anchor_eps_time"),
            "signal/sil_revisit_anchor_steps": metrics.current_signal_metrics.get("sil_revisit_anchor_steps"),
            "signal/ppo_kl_penalty_effective_coef": metrics.current_signal_metrics.get("ppo_kl_penalty_effective_coef", 0.0),
            "signal/ppo_kl_penalty_active": metrics.current_signal_metrics.get("ppo_kl_penalty_active", 0),
            "signal/ppo_kl_activation_success": metrics.current_signal_metrics.get("ppo_kl_activation_success", 0.0),
            "train/replay_mode": metrics.last_replay_stats["replay_mode"],
            "train/replay_filter_mode": metrics.last_replay_stats.get("replay_filter_mode", "return"),
            "train/replay_temporal_model_source": metrics.last_replay_stats.get("replay_temporal_model_source", "standalone"),
            "train/replay_temporal_model_loss": metrics.last_replay_stats.get("replay_temporal_model_loss", 0.0),
            "train/replay_temporal_model_updates": metrics.last_replay_stats.get("replay_temporal_model_updates", 0),
            "train/replay_temporal_model_samples": metrics.last_replay_stats.get("replay_temporal_model_samples", 0),
            "train/replay_temporal_model_eval_samples": metrics.last_replay_stats.get(
                "replay_temporal_model_eval_samples", 0,
            ),
            "train/replay_temporal_model_ready": metrics.last_replay_stats.get("replay_temporal_model_ready", 0),
            "train/replay_temporal_model_seen_samples": metrics.last_replay_stats.get(
                "replay_temporal_model_seen_samples", 0
            ),
            "train/replay_temporal_model_pred_mean": metrics.last_replay_stats.get("replay_temporal_model_pred_mean", 0.0),
            "train/replay_temporal_model_target_mean": metrics.last_replay_stats.get("replay_temporal_model_target_mean", 0.0),
            "train/replay_temporal_model_mae": metrics.last_replay_stats.get("replay_temporal_model_mae", 0.0),
            "train/replay_temporal_model_rmse": metrics.last_replay_stats.get("replay_temporal_model_rmse", 0.0),
            "train/replay_temporal_model_corr": metrics.last_replay_stats.get("replay_temporal_model_corr", 0.0),
            "train/replay_temporal_countdown_mae": metrics.last_replay_stats.get("replay_temporal_countdown_mae", 0.0),
            "train/replay_temporal_countdown_corr": metrics.last_replay_stats.get("replay_temporal_countdown_corr", 0.0),
            "train/replay_active": metrics.last_replay_stats["replay_active"],
            "train/replay_has_demo": metrics.last_replay_stats["replay_has_demo"],
            "train/replay_delayed": metrics.last_replay_stats.get("replay_delayed", 0),
            "train/replay_start_iter": metrics.last_replay_stats.get("replay_start_iter", 0),
            "train/replay_steps": metrics.last_replay_stats["replay_steps"],
            "train/replay_policy_loss": metrics.last_replay_stats["replay_policy_loss"],
            "train/replay_value_loss": metrics.last_replay_stats["replay_value_loss"],
            "train/replay_total_loss": metrics.last_replay_stats["replay_total_loss"],
            "train/replay_gradient_measured": metrics.last_replay_stats.get("replay_gradient_measured", 0),
            "train/replay_ppo_policy_grad_norm": metrics.last_replay_stats.get(
                "replay_ppo_policy_grad_norm", 0.0,
            ),
            "train/replay_policy_grad_norm": metrics.last_replay_stats.get(
                "replay_policy_grad_norm", 0.0,
            ),
            "train/replay_to_ppo_grad_norm_ratio": metrics.last_replay_stats.get(
                "replay_to_ppo_grad_norm_ratio", 0.0,
            ),
            "train/replay_to_ppo_grad_cosine": metrics.last_replay_stats.get(
                "replay_to_ppo_grad_cosine", 0.0,
            ),
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
            "train/replay_sample_unique_envs": metrics.last_replay_stats.get("replay_sample_unique_envs", 0.0),
            "train/replay_sample_unique_env_frac": metrics.last_replay_stats.get("replay_sample_unique_env_frac", 0.0),
            "train/replay_success_sample_unique_envs": metrics.last_replay_stats.get("replay_success_sample_unique_envs", 0.0),
            "train/replay_fallback_sample_unique_envs": metrics.last_replay_stats.get("replay_fallback_sample_unique_envs", 0.0),
            "train/replay_sample_unique_trajectories": metrics.last_replay_stats.get("replay_sample_unique_trajectories", 0.0),
            "train/replay_sample_effective_envs": metrics.last_replay_stats.get("replay_sample_effective_envs", 0.0),
            "train/replay_sample_top1_env_frac": metrics.last_replay_stats.get("replay_sample_top1_env_frac", 0.0),
            "train/replay_sample_top5_env_frac": metrics.last_replay_stats.get("replay_sample_top5_env_frac", 0.0),
            "train/replay_success_duplicate_transition_frac": metrics.last_replay_stats.get(
                "replay_success_duplicate_transition_frac", 0.0,
            ),
            "train/replay_env_step_quota": metrics.last_replay_stats.get("replay_env_step_quota", 0.0),
            "train/replay_positive_sample_unique_envs": metrics.last_replay_stats.get(
                "replay_positive_sample_unique_envs", 0.0,
            ),
            "train/replay_positive_sample_unique_env_frac": metrics.last_replay_stats.get(
                "replay_positive_sample_unique_env_frac", 0.0,
            ),
            "train/replay_effective_sample_unique_envs": metrics.last_replay_stats.get(
                "replay_effective_sample_unique_envs", 0.0,
            ),
            "train/replay_effective_sample_unique_env_frac": metrics.last_replay_stats.get(
                "replay_effective_sample_unique_env_frac", 0.0,
            ),
            "train/replay_aligned_candidate_envs": metrics.last_replay_stats.get(
                "replay_aligned_candidate_envs", 0.0,
            ),
            "train/replay_aligned_selected_trajectories": metrics.last_replay_stats.get(
                "replay_aligned_selected_trajectories", 0.0,
            ),
            "train/replay_aligned_dataset_steps": metrics.last_replay_stats.get(
                "replay_aligned_dataset_steps", 0.0,
            ),
            "train/replay_aligned_mean_trajectories_per_env": metrics.last_replay_stats.get(
                "replay_aligned_mean_trajectories_per_env", 0.0,
            ),
            "train/replay_aligned_fastest_source_frac": metrics.last_replay_stats.get(
                "replay_aligned_fastest_source_frac", 0.0,
            ),
            "train/replay_aligned_return_fill_frac": metrics.last_replay_stats.get(
                "replay_aligned_return_fill_frac", 0.0,
            ),
            "train/replay_aligned_short_trajectory_frac": metrics.last_replay_stats.get(
                "replay_aligned_short_trajectory_frac", 0.0,
            ),
            "train/replay_aligned_transition_wrap_frac": metrics.last_replay_stats.get(
                "replay_aligned_transition_wrap_frac", 0.0,
            ),
            "train/replay_aligned_replay_epochs": metrics.last_replay_stats.get(
                "replay_aligned_replay_epochs", 0.0,
            ),
            "train/replay_sample_success_frac": metrics.last_replay_stats.get("replay_sample_success_frac", 0.0),
            "train/replay_sample_fallback_frac": metrics.last_replay_stats.get("replay_sample_fallback_frac", 0.0),
            "train/replay_positive_success_frac": metrics.last_replay_stats.get("replay_positive_success_frac", 0.0),
            "train/replay_positive_fallback_frac": metrics.last_replay_stats.get("replay_positive_fallback_frac", 0.0),
            "train/replay_positive_success_precision": metrics.last_replay_stats.get(
                "replay_positive_success_precision", 0.0
            ),
            "train/replay_gap_success_mean": metrics.last_replay_stats.get("replay_gap_success_mean", 0.0),
            "train/replay_gap_fallback_mean": metrics.last_replay_stats.get("replay_gap_fallback_mean", 0.0),
            "train/replay_policy_loss_success": metrics.last_replay_stats.get("replay_policy_loss_success", 0.0),
            "train/replay_policy_loss_fallback": metrics.last_replay_stats.get("replay_policy_loss_fallback", 0.0),
            "train/replay_sample_return_mean": metrics.last_replay_stats.get("replay_sample_return_mean", 0.0),
            "train/replay_sample_return_success_mean": metrics.last_replay_stats.get(
                "replay_sample_return_success_mean", 0.0,
            ),
            "train/replay_sample_return_fallback_mean": metrics.last_replay_stats.get(
                "replay_sample_return_fallback_mean", 0.0,
            ),
            "train/replay_observed_return_mean": metrics.last_replay_stats.get("replay_observed_return_mean", 0.0),
            "train/replay_observed_return_fallback_mean": metrics.last_replay_stats.get(
                "replay_observed_return_fallback_mean", 0.0,
            ),
            "train/replay_bootstrap_return_delta_mean": metrics.last_replay_stats.get(
                "replay_bootstrap_return_delta_mean", 0.0,
            ),
            "train/replay_bootstrap_return_delta_fallback_mean": metrics.last_replay_stats.get(
                "replay_bootstrap_return_delta_fallback_mean", 0.0,
            ),
            "train/replay_state_value_mean": metrics.last_replay_stats.get("replay_state_value_mean", 0.0),
            "train/replay_state_value_fallback_mean": metrics.last_replay_stats.get(
                "replay_state_value_fallback_mean", 0.0,
            ),
            "train/replay_bootstrap_minus_state_value_fallback_mean": metrics.last_replay_stats.get(
                "replay_bootstrap_minus_state_value_fallback_mean", 0.0,
            ),
            "train/replay_bootstrap_active_frac": metrics.last_replay_stats.get("replay_bootstrap_active_frac", 0.0),
            "train/replay_bootstrap_value_mean": metrics.last_replay_stats.get("replay_bootstrap_value_mean", 0.0),
            "train/replay_bootstrap_active_value_mean": metrics.last_replay_stats.get(
                "replay_bootstrap_active_value_mean", 0.0,
            ),
            "train/replay_return_gap_mean": metrics.last_replay_stats.get("replay_return_gap_mean", 0.0),
            "train/replay_advantage_gap_mean": metrics.last_replay_stats.get(
                "replay_advantage_gap_mean",
                metrics.last_replay_stats.get("replay_return_gap_mean", 0.0),
            ),
            "train/replay_temporal_positive_frac": metrics.last_replay_stats.get("replay_temporal_positive_frac", 0.0),
            "train/replay_temporal_gap_mean": metrics.last_replay_stats.get("replay_temporal_gap_mean", 0.0),
            "train/replay_return_gap_fast_success_corr": metrics.last_replay_stats.get(
                "replay_return_gap_fast_success_corr", 0.0,
            ),
            "train/replay_temporal_gap_fast_success_corr": metrics.last_replay_stats.get(
                "replay_temporal_gap_fast_success_corr", 0.0,
            ),
            "train/replay_temporal_value_loss": metrics.last_replay_stats.get("replay_temporal_value_loss", 0.0),
            "train/replay_actor_nll_mean": metrics.last_replay_stats.get("replay_actor_nll_mean", 0.0),
            "train/replay_actor_need_mean": metrics.last_replay_stats.get("replay_actor_need_mean", 0.0),
            "train/replay_actor_weight_mean": metrics.last_replay_stats.get("replay_actor_weight_mean", 0.0),
            "train/replay_actor_bonus_mean": metrics.last_replay_stats.get("replay_actor_bonus_mean", 0.0),
            "train/replay_actor_add_gate_frac": metrics.last_replay_stats.get("replay_actor_add_gate_frac", 0.0),
            "train/replay_actor_abs_gap_mean": metrics.last_replay_stats.get("replay_actor_abs_gap_mean", 0.0),
            "train/replay_actor_closed_gap_frac": metrics.last_replay_stats.get("replay_actor_closed_gap_frac", 0.0),
            "train/replay_actor_high_nll_closed_gap_frac": metrics.last_replay_stats.get(
                "replay_actor_high_nll_closed_gap_frac", 0.0,
            ),
            "train/replay_actor_closed_gap_nll_mean": metrics.last_replay_stats.get(
                "replay_actor_closed_gap_nll_mean", 0.0,
            ),
            "train/replay_sil_weight_mean": metrics.last_replay_stats.get("replay_sil_weight_mean", 0.0),
            "train/replay_sil_weight_raw_mean": metrics.last_replay_stats.get("replay_sil_weight_raw_mean", 0.0),
            "train/replay_sil_weight_before_success_floor_mean": metrics.last_replay_stats.get(
                "replay_sil_weight_before_success_floor_mean", 0.0,
            ),
            "train/replay_success_actor_floor": metrics.last_replay_stats.get(
                "replay_success_actor_floor", 0.0,
            ),
            "train/replay_success_actor_floor_applied_frac": metrics.last_replay_stats.get(
                "replay_success_actor_floor_applied_frac", 0.0,
            ),
            "train/replay_effective_success_frac": metrics.last_replay_stats.get(
                "replay_effective_success_frac", 0.0,
            ),
            "train/replay_fast_success_retention_applied_frac": metrics.last_replay_stats.get(
                "replay_fast_success_retention_applied_frac", 0.0,
            ),
            "train/replay_fast_success_retention_weight_mean": metrics.last_replay_stats.get(
                "replay_fast_success_retention_weight_mean", 0.0,
            ),
            "train/replay_effective_fast_success_frac": metrics.last_replay_stats.get(
                "replay_effective_fast_success_frac", 0.0,
            ),
            **{'train/' + key: metrics.last_replay_stats.get(key, 0.0) for key in (
                'replay_success_auto_boost_mean', 'replay_success_extra_weight_mean',
                'replay_success_extra_cap_hit', 'replay_success_reopened_frac')},
            "train/replay_success_actor_weight_share_before": metrics.last_replay_stats.get(
                "replay_success_actor_weight_share_before", 0.0,
            ),
            "train/replay_success_actor_weight_share": metrics.last_replay_stats.get(
                "replay_success_actor_weight_share", 0.0,
            ),
            "train/replay_success_actor_boost": metrics.last_replay_stats.get(
                "replay_success_actor_boost", 0.0,
            ),
            "train/replay_success_actor_boost_cap_hit": metrics.last_replay_stats.get(
                "replay_success_actor_boost_cap_hit", 0.0,
            ),
            "train/replay_temporal_fallback_active": metrics.last_replay_stats.get(
                "replay_temporal_fallback_active", 0.0,
            ),
            "train/replay_actor_rescued_frac": metrics.last_replay_stats.get("replay_actor_rescued_frac", 0.0),
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
            "train/sil_fixed_anchor_active": metrics.last_replay_stats.get("sil_fixed_anchor_active", 0),
            "train/sil_fixed_anchor_capture_step": metrics.last_replay_stats.get(
                "sil_fixed_anchor_capture_step", 0,
            ),
            "train/sil_fixed_anchor_capture_success": metrics.last_replay_stats.get(
                "sil_fixed_anchor_capture_success", 0.0,
            ),
            "train/sil_fixed_anchor_transition_count": metrics.last_replay_stats.get(
                "sil_fixed_anchor_transition_count", 0,
            ),
            "train/sil_fixed_anchor_return_mean": metrics.last_replay_stats.get(
                "sil_fixed_anchor_return_mean", 0.0,
            ),
            "train/sil_fixed_anchor_value_mean": metrics.last_replay_stats.get(
                "sil_fixed_anchor_value_mean", 0.0,
            ),
            "train/sil_fixed_anchor_raw_gap_mean": metrics.last_replay_stats.get(
                "sil_fixed_anchor_raw_gap_mean", 0.0,
            ),
            "train/sil_fixed_anchor_positive_frac": metrics.last_replay_stats.get(
                "sil_fixed_anchor_positive_frac", 0.0,
            ),
            "train/sil_fixed_anchor_weight_mean": metrics.last_replay_stats.get(
                "sil_fixed_anchor_weight_mean", 0.0,
            ),
            "train/sil_fixed_anchor_effective_weight_mean": metrics.last_replay_stats.get(
                "sil_fixed_anchor_effective_weight_mean", 0.0,
            ),
            "train/sil_fixed_anchor_nll_mean": metrics.last_replay_stats.get(
                "sil_fixed_anchor_nll_mean", 0.0,
            ),
            "train/sil_fixed_anchor_nll_delta": metrics.last_replay_stats.get(
                "sil_fixed_anchor_nll_delta", 0.0,
            ),
            "train/sil_fixed_anchor_gated_grad_norm": metrics.last_replay_stats.get(
                "sil_fixed_anchor_gated_grad_norm", 0.0,
            ),
            "train/sil_fixed_anchor_effective_grad_norm": metrics.last_replay_stats.get(
                "sil_fixed_anchor_effective_grad_norm", 0.0,
            ),
        }
        # Preserve new opt-in replay diagnostics in the local history without
        # treating unmeasured actor statistics as zero.
        values.update({"train/" + key: value for key, value in metrics.last_replay_stats.items()
                       if key.startswith(("sil_fixed_anchor_", "sil_relabelled_anchor_", "replay_screen_", "replay_success_terminal_"))
                       or key in {"replay_full_actor_diagnostics_measured", "replay_actor_forward_frac",
                                  "replay_microbatch_size", "replay_loss_denominator", "replay_positive_count",
                                  "replay_candidate_count", "replay_denominator_multiplier", "replay_empty_positive_slice",
                                  "replay_success_policy_grad_norm", "replay_failure_policy_grad_norm",
                                  "replay_success_to_ppo_grad_cosine", "replay_failure_to_ppo_grad_cosine",
                                  "replay_value_grad_norm", "replay_ppo_value_grad_norm"}})
        if bool(getattr(self.args, "outcome_diagnostics", False)):
            values["diagnostic/current_success"] = metrics.cur_success_rate
            values["diagnostic/full_episode_window_ready"] = int(metrics.ready_to_record)
        if extra_metrics:
            values.update(extra_metrics)

        values.update(self._distribution_metrics(agent, entropy_mean))
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

    def log_skipped_training_update(self, metrics):
        """Keep observations and diagnostics when no optimizer step is accepted."""
        if not self.args.saving:
            return
        values = {
            **{k: v for k, v in self._latest_plot_metrics.items() if not k.startswith("train/")},
            **metrics.performance_metrics,
            **self._base_metrics(metrics),
            **{"signal/" + k: v for k, v in metrics.current_signal_metrics.items()},
            **{"train/" + k: v for k, v in metrics.last_replay_stats.items()},
            "train/optimizer_updates": 0,
            "debug/skipped_update_iter": int(metrics.skipped_update_iter),
        }
        self._append_local(values)
        metrics.save_meta_data_snapshot()
        self._log_wandb(values)

    @staticmethod
    def finish():
        if TrainingLogger._wandb_active:
            TrainingLogger._wandb_module().finish()
            TrainingLogger._wandb_active = False


WandbLogger = TrainingLogger

__all__ = ["TrainingLogger", "WandbLogger"]
