"""TSIL / BC replay loss from archived trajectory memory."""

import json
import math
import hashlib
import os

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from core.training.algo.utils import compute_value_loss
from core.training.algo.tsil.chunked_autograd import chunked_value, chunked_logprob


_REPLAY_COVERAGE_STAT_KEYS = (
    "replay_success_terminal_value_mean",
    "replay_success_terminal_value_max",
    "replay_success_terminal_negative_frac",
    "replay_success_terminal_added_value_mean",
    "replay_loss_denominator",
    "replay_positive_count",
    "replay_candidate_count",
    "replay_denominator_multiplier",
    "replay_empty_positive_slice",
    "replay_full_actor_diagnostics_measured",
    "replay_actor_forward_frac",
    "replay_microbatch_size",
    "replay_sample_unique_envs",
    "replay_sample_unique_env_frac",
    "replay_success_sample_unique_envs",
    "replay_fallback_sample_unique_envs",
    "replay_sample_unique_trajectories",
    "replay_sample_effective_envs",
    "replay_sample_top1_env_frac",
    "replay_sample_top5_env_frac",
    "replay_success_duplicate_transition_frac",
    "replay_env_step_quota",
    "replay_positive_sample_unique_envs",
    "replay_positive_sample_unique_env_frac",
    "replay_effective_sample_unique_envs",
    "replay_effective_sample_unique_env_frac",
    "replay_aligned_candidate_envs",
    "replay_aligned_selected_trajectories",
    "replay_aligned_dataset_steps",
    "replay_aligned_mean_trajectories_per_env",
    "replay_aligned_fastest_source_frac",
    "replay_aligned_return_fill_frac",
    "replay_aligned_short_trajectory_frac",
    "replay_aligned_transition_wrap_frac",
    "replay_aligned_replay_epochs",
)

_REPLAY_RETENTION_STAT_KEYS = (
    "replay_fast_success_retention_applied_frac",
    "replay_fast_success_retention_weight_mean",
    "replay_effective_fast_success_frac",
)

_REPLAY_SUCCESS_BALANCE_STAT_KEYS = (
    "replay_success_auto_boost_mean",
    "replay_success_extra_weight_mean",
    "replay_success_extra_cap_hit",
    "replay_success_reopened_frac",
    "replay_success_actor_weight_share_before",
    "replay_success_actor_weight_share",
    "replay_success_actor_boost",
    "replay_success_actor_boost_cap_hit",
)


class TsilReplayLoss:
    """Computes SIL / BC auxiliary losses from TSIL trajectory memory samples."""

    def __init__(self, args, device, dtype):
        self.args = args
        self.device = device
        self.dtype = dtype
        self._fixed_success_reference = None
        self._fixed_success_reference_step = None
        self._fixed_success_reference_success = None
        self._fixed_success_reference_initial_nll = None
        self._fixed_success_reference_saved = False
        self._full_actor_diagnostics = (
            int(getattr(args, "sil_analysis_interval", 0)) > 0
        )

    def _temporal_model_source(self):
        return str(getattr(self.args, "sil_temporal_model", "standalone")).lower()

    def _uses_standalone_temporal_model(self):
        return self._temporal_model_source() == "standalone"

    def _standalone_temporal_ready(self, agent=None):
        if agent is None or getattr(agent, "temporal_critic", None) is None:
            return False
        required = int(getattr(self.args, "sil_temporal_ready_samples", 0))
        seen = getattr(agent, "temporal_critic_seen_samples", None)
        return required <= 0 or (
            seen is not None and int(seen.detach().item()) >= required
        )

    @staticmethod
    def _base_stats(replay_mode="disabled"):
        return {
            "replay_mode": replay_mode,
            "replay_filter_mode": "return",
            "replay_temporal_model_source": "standalone",
            "replay_temporal_model_loss": 0.0,
            "replay_temporal_model_updates": 0,
            "replay_temporal_model_samples": 0,
            "replay_temporal_model_eval_samples": 0,
            "replay_temporal_model_ready": 0,
            "replay_temporal_model_seen_samples": 0,
            "replay_temporal_model_pred_mean": 0.0,
            "replay_temporal_model_target_mean": 0.0,
            "replay_temporal_model_mae": 0.0,
            "replay_temporal_model_rmse": 0.0,
            "replay_temporal_model_corr": 0.0,
            "replay_temporal_countdown_mae": 0.0,
            "replay_temporal_countdown_corr": 0.0,
            "replay_active": 0,
            "replay_has_demo": 0,
            "replay_steps": 0,
            "replay_policy_loss": 0.0,
            "replay_value_loss": 0.0,
            "replay_total_loss": 0.0,
            "replay_gradient_measured": 0,
            "replay_ppo_policy_grad_norm": 0.0,
            "replay_policy_grad_norm": 0.0,
            "replay_to_ppo_grad_norm_ratio": 0.0,
            "replay_to_ppo_grad_cosine": 0.0,
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
            **{key: 0.0 for key in _REPLAY_COVERAGE_STAT_KEYS},
            "replay_sample_success_frac": 0.0,
            "replay_sample_fallback_frac": 0.0,
            "replay_positive_success_frac": 0.0,
            "replay_positive_fallback_frac": 0.0,
            "replay_positive_success_precision": 0.0,
            "replay_gap_success_mean": 0.0,
            "replay_gap_fallback_mean": 0.0,
            "replay_policy_loss_success": 0.0,
            "replay_policy_loss_fallback": 0.0,
            "replay_sample_return_mean": 0.0,
            "replay_sample_return_success_mean": 0.0,
            "replay_sample_return_fallback_mean": 0.0,
            "replay_observed_return_mean": 0.0,
            "replay_observed_return_fallback_mean": 0.0,
            "replay_bootstrap_return_delta_mean": 0.0,
            "replay_bootstrap_return_delta_fallback_mean": 0.0,
            "replay_state_value_mean": 0.0,
            "replay_state_value_fallback_mean": 0.0,
            "replay_bootstrap_minus_state_value_fallback_mean": 0.0,
            "replay_bootstrap_active_frac": 0.0,
            "replay_bootstrap_value_mean": 0.0,
            "replay_bootstrap_active_value_mean": 0.0,
            "replay_return_gap_mean": 0.0,
            "replay_advantage_gap_mean": 0.0,
            "replay_temporal_positive_frac": 0.0,
            "replay_temporal_gap_mean": 0.0,
            "replay_return_gap_fast_success_corr": 0.0,
            "replay_temporal_gap_fast_success_corr": 0.0,
            "replay_temporal_value_loss": 0.0,
            "replay_actor_nll_mean": 0.0,
            "replay_actor_need_mean": 0.0,
            "replay_actor_weight_mean": 0.0,
            "replay_actor_bonus_mean": 0.0,
            "replay_actor_add_gate_frac": 0.0,
            "replay_actor_abs_gap_mean": 0.0,
            "replay_actor_closed_gap_frac": 0.0,
            "replay_actor_high_nll_closed_gap_frac": 0.0,
            "replay_actor_closed_gap_nll_mean": 0.0,
            "replay_sil_weight_mean": 0.0,
            "replay_sil_weight_raw_mean": 0.0,
            "replay_sil_weight_before_success_floor_mean": 0.0,
            "replay_success_actor_floor": 0.0,
            "replay_success_actor_floor_applied_frac": 0.0,
            "replay_effective_success_frac": 0.0,
            **{key: 0.0 for key in _REPLAY_RETENTION_STAT_KEYS},
            **{key: 0.0 for key in _REPLAY_SUCCESS_BALANCE_STAT_KEYS},
            "replay_temporal_fallback_active": 0.0,
            "replay_actor_rescued_frac": 0.0,
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
            "sil_fixed_anchor_active": 0,
            "sil_fixed_anchor_capture_step": 0,
            "sil_fixed_anchor_capture_success": 0.0,
            "sil_fixed_anchor_transition_count": 0,
            "sil_fixed_anchor_return_mean": 0.0,
            "sil_fixed_anchor_value_mean": 0.0,
            "sil_fixed_anchor_raw_gap_mean": 0.0,
            "sil_fixed_anchor_positive_frac": 0.0,
            "sil_fixed_anchor_weight_mean": 0.0,
            "sil_fixed_anchor_effective_weight_mean": 0.0,
            "sil_fixed_anchor_nll_mean": 0.0,
            "sil_fixed_anchor_nll_delta": 0.0,
            "sil_fixed_anchor_gated_grad_norm": 0.0,
            "sil_fixed_anchor_effective_grad_norm": 0.0,
        }

    def default_stats(self):
        replay_mode = self.args.sil_mode if self.args.use_sil else "disabled"
        stats = self._base_stats(replay_mode)
        stats["replay_filter_mode"] = str(
            getattr(self.args, "sil_filter_mode", "return")
        )
        stats["replay_temporal_model_source"] = self._temporal_model_source()
        return stats

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
            "replay_temporal_model_loss": 0.0,
            "replay_temporal_model_updates": 0,
            "replay_temporal_model_samples": 0,
            "replay_temporal_model_ready": 0,
            "replay_temporal_model_seen_samples": 0,
            "replay_temporal_model_target_mean": 0.0,
            "replay_sample_success_frac": 0.0,
            "replay_sample_fallback_frac": 0.0,
            "replay_positive_success_frac": 0.0,
            "replay_positive_fallback_frac": 0.0,
            "replay_positive_success_precision": 0.0,
            "replay_gap_success_mean": 0.0,
            "replay_gap_fallback_mean": 0.0,
            "replay_policy_loss_success": 0.0,
            "replay_policy_loss_fallback": 0.0,
            "replay_sample_return_mean": 0.0,
            "replay_sample_return_success_mean": 0.0,
            "replay_sample_return_fallback_mean": 0.0,
            "replay_observed_return_mean": 0.0,
            "replay_observed_return_fallback_mean": 0.0,
            "replay_bootstrap_return_delta_mean": 0.0,
            "replay_bootstrap_return_delta_fallback_mean": 0.0,
            "replay_state_value_mean": 0.0,
            "replay_state_value_fallback_mean": 0.0,
            "replay_bootstrap_minus_state_value_fallback_mean": 0.0,
            "replay_bootstrap_active_frac": 0.0,
            "replay_bootstrap_value_mean": 0.0,
            "replay_bootstrap_active_value_mean": 0.0,
            "replay_return_gap_mean": 0.0,
            "replay_advantage_gap_mean": 0.0,
            "replay_temporal_positive_frac": 0.0,
            "replay_temporal_gap_mean": 0.0,
            "replay_return_gap_fast_success_corr": 0.0,
            "replay_temporal_gap_fast_success_corr": 0.0,
            "replay_temporal_value_loss": 0.0,
            "replay_actor_nll_mean": 0.0,
            "replay_actor_need_mean": 0.0,
            "replay_actor_weight_mean": 0.0,
            "replay_actor_bonus_mean": 0.0,
            "replay_actor_add_gate_frac": 0.0,
            "replay_actor_abs_gap_mean": 0.0,
            "replay_actor_closed_gap_frac": 0.0,
            "replay_actor_high_nll_closed_gap_frac": 0.0,
            "replay_actor_closed_gap_nll_mean": 0.0,
            "replay_sil_weight_mean": 0.0,
            "replay_sil_weight_raw_mean": 0.0,
            "replay_sil_weight_before_success_floor_mean": 0.0,
            "replay_success_actor_floor": 0.0,
            "replay_success_actor_floor_applied_frac": 0.0,
            "replay_effective_success_frac": 0.0,
            **{key: 0.0 for key in _REPLAY_RETENTION_STAT_KEYS},
            **{key: 0.0 for key in _REPLAY_SUCCESS_BALANCE_STAT_KEYS},
            "replay_temporal_fallback_active": 0.0,
            "replay_actor_rescued_frac": 0.0,
            **{key: 0.0 for key in _REPLAY_COVERAGE_STAT_KEYS},
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
        replay_totals["replay_sampled_positive_frac"] += float(
            replay_sampled_positive_frac
        )
        replay_batch_stats = replay_batch_stats or {}
        replay_totals["replay_memory_logp"] += float(
            replay_batch_stats.get("memory_logp", 0.0)
        )
        replay_totals["replay_memory_weighted_logp"] += float(
            replay_batch_stats.get("memory_weighted_logp", 0.0)
        )
        replay_totals["replay_memory_positive_gap_mean"] += float(
            replay_batch_stats.get("memory_positive_gap_mean", 0.0)
        )
        for key in (
            "replay_sample_success_frac",
            "replay_sample_fallback_frac",
            *_REPLAY_COVERAGE_STAT_KEYS,
            "replay_positive_success_frac",
            "replay_positive_fallback_frac",
            "replay_positive_success_precision",
            "replay_gap_success_mean",
            "replay_gap_fallback_mean",
            "replay_policy_loss_success",
            "replay_policy_loss_fallback",
            "replay_sample_return_mean",
            "replay_sample_return_success_mean",
            "replay_sample_return_fallback_mean",
            "replay_observed_return_mean",
            "replay_observed_return_fallback_mean",
            "replay_bootstrap_return_delta_mean",
            "replay_bootstrap_return_delta_fallback_mean",
            "replay_state_value_mean",
            "replay_state_value_fallback_mean",
            "replay_bootstrap_minus_state_value_fallback_mean",
            "replay_bootstrap_active_frac",
            "replay_bootstrap_value_mean",
            "replay_bootstrap_active_value_mean",
            "replay_return_gap_mean",
            "replay_advantage_gap_mean",
            "replay_temporal_positive_frac",
            "replay_temporal_gap_mean",
            "replay_return_gap_fast_success_corr",
            "replay_temporal_gap_fast_success_corr",
            "replay_temporal_value_loss",
            "replay_actor_nll_mean",
            "replay_actor_need_mean",
            "replay_actor_weight_mean",
            "replay_actor_bonus_mean",
            "replay_actor_add_gate_frac",
            "replay_actor_abs_gap_mean",
            "replay_actor_closed_gap_frac",
            "replay_actor_high_nll_closed_gap_frac",
            "replay_actor_closed_gap_nll_mean",
            "replay_sil_weight_mean",
            "replay_sil_weight_raw_mean",
            "replay_sil_weight_before_success_floor_mean",
            "replay_success_actor_floor",
            "replay_success_actor_floor_applied_frac",
            "replay_effective_success_frac",
            *_REPLAY_RETENTION_STAT_KEYS,
            *_REPLAY_SUCCESS_BALANCE_STAT_KEYS,
            "replay_temporal_fallback_active",
            "replay_actor_rescued_frac",
        ):
            replay_totals[key] += float(replay_batch_stats.get(key, 0.0))
        temporal_model_updates = int(
            replay_batch_stats.get("replay_temporal_model_updates", 0)
        )
        replay_totals["replay_temporal_model_updates"] += temporal_model_updates
        replay_totals["replay_temporal_model_samples"] += int(
            replay_batch_stats.get("replay_temporal_model_samples", 0)
        )
        replay_totals["replay_temporal_model_ready"] = max(
            int(replay_totals.get("replay_temporal_model_ready", 0)),
            int(replay_batch_stats.get("replay_temporal_model_ready", 0)),
        )
        replay_totals["replay_temporal_model_seen_samples"] = max(
            int(replay_totals.get("replay_temporal_model_seen_samples", 0)),
            int(replay_batch_stats.get("replay_temporal_model_seen_samples", 0)),
        )
        if temporal_model_updates > 0:
            replay_totals["replay_temporal_model_loss"] += float(
                replay_batch_stats.get("replay_temporal_model_loss", 0.0)
            )
            replay_totals["replay_temporal_model_target_mean"] += float(
                replay_batch_stats.get("replay_temporal_model_target_mean", 0.0)
            )
        replay_totals["replay_steps"] += int(replay_step_count)
        replay_totals["replay_updates"] += 1

    @staticmethod
    def finalize_iteration_stats(base_replay_stats, replay_totals, num_agent_updates):
        replay_stats = dict(base_replay_stats)
        replay_updates = replay_totals["replay_updates"]
        if replay_updates == 0:
            return replay_stats

        divisor = max(int(replay_updates), int(num_agent_updates), 1)
        temporal_model_updates = max(
            int(replay_totals.get("replay_temporal_model_updates", 0)), 0
        )
        replay_stats.update(
            {
                "replay_active": int(replay_stats["replay_has_demo"]),
                "replay_steps": replay_totals["replay_steps"],
                "replay_policy_loss": replay_totals["replay_policy_loss"] / divisor,
                "replay_value_loss": replay_totals["replay_value_loss"] / divisor,
                "replay_total_loss": replay_totals["replay_total_loss"] / divisor,
                "replay_positive_frac": replay_totals["replay_positive_frac"] / divisor,
                "replay_positive_available_frac": replay_totals[
                    "replay_positive_available_frac"
                ]
                / divisor,
                "replay_sampled_positive_frac": replay_totals[
                    "replay_sampled_positive_frac"
                ]
                / divisor,
                "replay_memory_logp": replay_totals["replay_memory_logp"] / divisor,
                "replay_memory_weighted_logp": replay_totals[
                    "replay_memory_weighted_logp"
                ]
                / divisor,
                "replay_memory_positive_gap_mean": replay_totals[
                    "replay_memory_positive_gap_mean"
                ]
                / divisor,
                "replay_sample_success_frac": replay_totals[
                    "replay_sample_success_frac"
                ]
                / divisor,
                "replay_sample_fallback_frac": replay_totals[
                    "replay_sample_fallback_frac"
                ]
                / divisor,
                **{
                    key: replay_totals[key] / divisor
                    for key in _REPLAY_COVERAGE_STAT_KEYS
                },
                "replay_positive_success_frac": replay_totals[
                    "replay_positive_success_frac"
                ]
                / divisor,
                "replay_positive_fallback_frac": replay_totals[
                    "replay_positive_fallback_frac"
                ]
                / divisor,
                "replay_positive_success_precision": replay_totals[
                    "replay_positive_success_precision"
                ]
                / divisor,
                "replay_gap_success_mean": replay_totals["replay_gap_success_mean"]
                / divisor,
                "replay_gap_fallback_mean": replay_totals["replay_gap_fallback_mean"]
                / divisor,
                "replay_policy_loss_success": replay_totals[
                    "replay_policy_loss_success"
                ]
                / divisor,
                "replay_policy_loss_fallback": replay_totals[
                    "replay_policy_loss_fallback"
                ]
                / divisor,
                "replay_sample_return_mean": replay_totals["replay_sample_return_mean"]
                / divisor,
                "replay_sample_return_success_mean": replay_totals[
                    "replay_sample_return_success_mean"
                ]
                / divisor,
                "replay_sample_return_fallback_mean": replay_totals[
                    "replay_sample_return_fallback_mean"
                ]
                / divisor,
                "replay_observed_return_mean": replay_totals[
                    "replay_observed_return_mean"
                ]
                / divisor,
                "replay_observed_return_fallback_mean": replay_totals[
                    "replay_observed_return_fallback_mean"
                ]
                / divisor,
                "replay_bootstrap_return_delta_mean": replay_totals[
                    "replay_bootstrap_return_delta_mean"
                ]
                / divisor,
                "replay_bootstrap_return_delta_fallback_mean": (
                    replay_totals["replay_bootstrap_return_delta_fallback_mean"]
                    / divisor
                ),
                "replay_state_value_mean": replay_totals["replay_state_value_mean"]
                / divisor,
                "replay_state_value_fallback_mean": replay_totals[
                    "replay_state_value_fallback_mean"
                ]
                / divisor,
                "replay_bootstrap_minus_state_value_fallback_mean": (
                    replay_totals["replay_bootstrap_minus_state_value_fallback_mean"]
                    / divisor
                ),
                "replay_bootstrap_active_frac": replay_totals[
                    "replay_bootstrap_active_frac"
                ]
                / divisor,
                "replay_bootstrap_value_mean": replay_totals[
                    "replay_bootstrap_value_mean"
                ]
                / divisor,
                "replay_bootstrap_active_value_mean": replay_totals[
                    "replay_bootstrap_active_value_mean"
                ]
                / divisor,
                "replay_return_gap_mean": replay_totals["replay_return_gap_mean"]
                / divisor,
                "replay_advantage_gap_mean": replay_totals["replay_advantage_gap_mean"]
                / divisor,
                "replay_temporal_positive_frac": replay_totals[
                    "replay_temporal_positive_frac"
                ]
                / divisor,
                "replay_temporal_gap_mean": replay_totals["replay_temporal_gap_mean"]
                / divisor,
                "replay_return_gap_fast_success_corr": (
                    replay_totals["replay_return_gap_fast_success_corr"] / divisor
                ),
                "replay_temporal_gap_fast_success_corr": (
                    replay_totals["replay_temporal_gap_fast_success_corr"] / divisor
                ),
                "replay_temporal_value_loss": replay_totals[
                    "replay_temporal_value_loss"
                ]
                / divisor,
                "replay_actor_nll_mean": replay_totals["replay_actor_nll_mean"]
                / divisor,
                "replay_actor_need_mean": replay_totals["replay_actor_need_mean"]
                / divisor,
                "replay_actor_weight_mean": replay_totals["replay_actor_weight_mean"]
                / divisor,
                "replay_actor_bonus_mean": replay_totals["replay_actor_bonus_mean"]
                / divisor,
                "replay_actor_add_gate_frac": replay_totals[
                    "replay_actor_add_gate_frac"
                ]
                / divisor,
                "replay_actor_abs_gap_mean": replay_totals["replay_actor_abs_gap_mean"]
                / divisor,
                "replay_actor_closed_gap_frac": replay_totals[
                    "replay_actor_closed_gap_frac"
                ]
                / divisor,
                "replay_actor_high_nll_closed_gap_frac": (
                    replay_totals["replay_actor_high_nll_closed_gap_frac"] / divisor
                ),
                "replay_actor_closed_gap_nll_mean": replay_totals[
                    "replay_actor_closed_gap_nll_mean"
                ]
                / divisor,
                "replay_sil_weight_mean": replay_totals["replay_sil_weight_mean"]
                / divisor,
                "replay_sil_weight_raw_mean": replay_totals[
                    "replay_sil_weight_raw_mean"
                ]
                / divisor,
                "replay_sil_weight_before_success_floor_mean": (
                    replay_totals["replay_sil_weight_before_success_floor_mean"]
                    / divisor
                ),
                "replay_success_actor_floor": replay_totals[
                    "replay_success_actor_floor"
                ]
                / divisor,
                "replay_success_actor_floor_applied_frac": (
                    replay_totals["replay_success_actor_floor_applied_frac"] / divisor
                ),
                "replay_effective_success_frac": replay_totals[
                    "replay_effective_success_frac"
                ]
                / divisor,
                **{
                    key: replay_totals[key] / divisor
                    for key in _REPLAY_RETENTION_STAT_KEYS
                },
                **{
                    key: replay_totals[key] / divisor
                    for key in _REPLAY_SUCCESS_BALANCE_STAT_KEYS
                },
                "replay_temporal_fallback_active": replay_totals[
                    "replay_temporal_fallback_active"
                ]
                / divisor,
                "replay_actor_rescued_frac": replay_totals["replay_actor_rescued_frac"]
                / divisor,
                "replay_temporal_model_loss": (
                    replay_totals["replay_temporal_model_loss"]
                    / max(temporal_model_updates, 1)
                ),
                "replay_temporal_model_updates": temporal_model_updates,
                "replay_temporal_model_samples": int(
                    replay_totals.get("replay_temporal_model_samples", 0)
                ),
                "replay_temporal_model_ready": int(
                    replay_totals.get("replay_temporal_model_ready", 0)
                ),
                "replay_temporal_model_seen_samples": int(
                    replay_totals.get("replay_temporal_model_seen_samples", 0)
                ),
                "replay_temporal_model_target_mean": (
                    replay_totals["replay_temporal_model_target_mean"]
                    / max(temporal_model_updates, 1)
                ),
            }
        )
        return replay_stats

    # Dataset preparation: build once per update; reshuffle between replay epochs.
    def reset_epoch_sampler(self, replay_context, num_ppo_minibatches, agent=None):
        if replay_context is not None and replay_context.get("has_tsil_memory"):
            batch_size = max(int(getattr(self.args, "sil_batch_size", 1)), 1)
            num_minibatches = max(int(num_ppo_minibatches), 1)
            replay_context["_sil_batch_size"] = batch_size
            replay_context["_sil_num_minibatches"] = num_minibatches
            if bool(getattr(self.args, "sil_aligned_replay", False)):
                epoch_index = int(replay_context.get("_sil_replay_epoch_index", 0))
                replay_context["_sil_replay_epoch_index"] = epoch_index + 1
                replay_context["_sil_replay_slice_index"] = 0
                replay_context["_sil_replay_num_slices"] = num_minibatches
                if epoch_index >= int(getattr(self.args, "sil_replay_epochs", 1)):
                    replay_context.pop("_sil_replay_batch", None)
                    replay_context.pop("_sil_replay_order", None)
                    return

                dataset = replay_context.get("_sil_aligned_replay_dataset")
                if dataset is None:
                    sample_batch_fn = replay_context.get("sample_batch_fn")
                    if sample_batch_fn is None:
                        return
                    missing = object()
                    saved_agent = replay_context.get("_sil_value_agent", missing)
                    saved_building = replay_context.get(
                        "_sil_building_replay_batch", missing
                    )
                    try:
                        replay_context["_sil_value_agent"] = agent
                        replay_context["_sil_building_replay_batch"] = True
                        dataset = sample_batch_fn(replay_context)
                    finally:
                        if saved_agent is missing:
                            replay_context.pop("_sil_value_agent", None)
                        else:
                            replay_context["_sil_value_agent"] = saved_agent
                        if saved_building is missing:
                            replay_context.pop("_sil_building_replay_batch", None)
                        else:
                            replay_context["_sil_building_replay_batch"] = (
                                saved_building
                            )
                    if dataset is not None:
                        if getattr(self.args, "sil_freeze_replay_gap", False):
                            with torch.no_grad():
                                dataset["sil_frozen_positive_gap"] = (
                                    self._positive_gap_for_batch(
                                        agent,
                                        dataset["state"],
                                        dataset["returns"],
                                        dataset["task_ids"],
                                    )
                                    .detach()
                                    .clone()
                                )
                                if (
                                    getattr(
                                        self.args,
                                        "sil_replay_normalization_scope",
                                        "slice",
                                    )
                                    == "dataset"
                                ):
                                    weights = dataset["sil_frozen_positive_gap"].clone()
                                    if "sil_weights" in dataset:
                                        weights = weights * dataset["sil_weights"].to(
                                            weights
                                        )
                                    dataset["sil_frozen_actor_weights"] = (
                                        self._normalize_actor_gap_weights(weights)
                                        .detach()
                                        .clone()
                                    )
                        replay_context["_sil_aligned_replay_dataset"] = dataset

                if dataset is None or int(dataset["action"].shape[0]) == 0:
                    replay_context.pop("_sil_replay_batch", None)
                    replay_context.pop("_sil_replay_order", None)
                    return
                count = int(dataset["action"].shape[0])
                replay_context["_sil_replay_batch"] = dataset
                replay_context["_sil_replay_order"] = (
                    torch.randperm(count, device=dataset["action"].device)
                    if bool(getattr(self.args, "sil_shuffle_replay_transitions", False))
                    else torch.arange(
                        count, dtype=torch.long, device=dataset["action"].device
                    )
                )
                stats = replay_context.setdefault("stats", self.default_stats())
                for batch_key, stat_key in (
                    (
                        "success_terminal_value_mean",
                        "replay_success_terminal_value_mean",
                    ),
                    ("success_terminal_value_max", "replay_success_terminal_value_max"),
                    (
                        "success_terminal_negative_frac",
                        "replay_success_terminal_negative_frac",
                    ),
                    (
                        "success_terminal_added_value_mean",
                        "replay_success_terminal_added_value_mean",
                    ),
                    ("aligned_candidate_envs", "replay_aligned_candidate_envs"),
                    (
                        "aligned_selected_trajectories",
                        "replay_aligned_selected_trajectories",
                    ),
                    ("aligned_dataset_steps", "replay_aligned_dataset_steps"),
                    (
                        "aligned_mean_trajectories_per_env",
                        "replay_aligned_mean_trajectories_per_env",
                    ),
                    (
                        "aligned_fastest_source_frac",
                        "replay_aligned_fastest_source_frac",
                    ),
                    ("aligned_return_fill_frac", "replay_aligned_return_fill_frac"),
                    (
                        "aligned_short_trajectory_frac",
                        "replay_aligned_short_trajectory_frac",
                    ),
                    (
                        "aligned_transition_wrap_frac",
                        "replay_aligned_transition_wrap_frac",
                    ),
                    ("aligned_replay_epochs", "replay_aligned_replay_epochs"),
                ):
                    stats[stat_key] = float(dataset.get(batch_key, 0.0))
                return

            replay_context.pop("_sil_replay_batch", None)
            replay_context["_sil_replay_cursor"] = 0

            if (
                str(getattr(self.args, "sil_replay_build_scope", "minibatch"))
                .strip()
                .lower()
                != "update"
            ):
                return
            sample_batch_fn = replay_context.get("sample_batch_fn")
            if sample_batch_fn is None:
                return

            missing = object()
            saved_agent = replay_context.get("_sil_value_agent", missing)
            saved_batch_size = replay_context.get("_sil_sample_batch_size", missing)
            saved_building = replay_context.get("_sil_building_replay_batch", missing)
            try:
                replay_context["_sil_value_agent"] = agent
                replay_context["_sil_sample_batch_size"] = batch_size * num_minibatches
                replay_context["_sil_building_replay_batch"] = True
                batch = sample_batch_fn(replay_context)
            finally:
                if saved_agent is missing:
                    replay_context.pop("_sil_value_agent", None)
                else:
                    replay_context["_sil_value_agent"] = saved_agent
                if saved_batch_size is missing:
                    replay_context.pop("_sil_sample_batch_size", None)
                else:
                    replay_context["_sil_sample_batch_size"] = saved_batch_size
                if saved_building is missing:
                    replay_context.pop("_sil_building_replay_batch", None)
                else:
                    replay_context["_sil_building_replay_batch"] = saved_building
            if batch is not None and int(batch["action"].shape[0]) > 0:
                replay_context["_sil_replay_batch"] = batch

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

    def _positive_gap_for_batch(
        self, agent, state_batch, return_batch, task_id_batch, return_value=False
    ):
        microbatch = int(getattr(self.args, "sil_microbatch_size", 0))
        if microbatch:
            value = chunked_value(agent, state_batch, microbatch)
        else:
            value, _, _ = agent.get_value(state_batch, denorm=False, update=False)
        if self.args.norm_value:
            normalize_task_ids = task_id_batch if self.args.pertask_norm else None
            return_batch = agent.normalize_value(
                return_batch,
                denorm=False,
                update=False,
                task_ids=normalize_task_ids,
            )
        value = value.view(-1)
        raw_gap = return_batch - value
        positive_gap = torch.clamp(raw_gap, min=0.0)
        if return_value:
            return positive_gap, value, return_batch, raw_gap
        return positive_gap

    def _normalize_actor_gap_weights(self, weights):
        """Scale positive actor weights by their std, with an optional upper cap.

        The reduction spans the logical replay slice, including when network
        forwards use microbatches. No mean subtraction or critic rescaling.
        """
        from core.training.algo.tsil.replay_normalization import normalize_actor_weights

        return normalize_actor_weights(weights, self.args)

    def _actor_mismatch_multiplier(self, logprob):
        nll = (-logprob.detach()).clamp_min(0.0)
        scale = nll.std(unbiased=False).clamp_min(1e-8)
        need = torch.clamp((nll - nll.mean()) / scale, min=0.0)
        coef = float(getattr(self.args, "sil_actor_weight_coef", 0.0))
        if coef <= 0:
            return torch.ones_like(logprob), need, nll
        max_weight = float(getattr(self.args, "sil_actor_weight_max", 3.0))
        multiplier = torch.clamp(1.0 + coef * need, max=max_weight)
        return multiplier, need, nll

    def _apply_fast_success_retention(
        self,
        weights,
        raw_return_gap,
        sil_success_mask,
        sil_return_fill_mask,
    ):
        """Add a decaying actor-only weight for fastest-success replay samples."""
        zeros = torch.zeros_like(weights)
        empty_mask = torch.zeros_like(weights, dtype=torch.bool)
        coef = float(getattr(self.args, "sil_fast_success_retention_coef", 0.0))
        if coef <= 0.0 or sil_success_mask is None or sil_return_fill_mask is None:
            return weights, zeros, empty_mask, empty_mask

        success_mask = sil_success_mask.to(
            device=weights.device, dtype=torch.bool
        ).view(-1)
        return_fill_mask = sil_return_fill_mask.to(
            device=weights.device,
            dtype=torch.bool,
        ).view(-1)
        fast_success_mask = success_mask & ~return_fill_mask
        if not bool(fast_success_mask.any().item()):
            return weights, zeros, empty_mask, fast_success_mask

        temperature = float(
            getattr(
                self.args,
                "sil_fast_success_retention_temperature",
                0.05,
            )
        )
        nonpositive_gap = torch.minimum(raw_return_gap.detach().view(-1), zeros)
        candidate = float(coef) * torch.exp(nonpositive_gap / temperature)
        retention_weights = torch.where(fast_success_mask, candidate, zeros)
        updated = torch.where(
            fast_success_mask,
            torch.maximum(weights, retention_weights),
            weights,
        )
        applied_mask = fast_success_mask & (updated.detach() > weights.detach())
        return updated, retention_weights, applied_mask, fast_success_mask

    def _balance_success_actor_share(self, weights, sil_success_mask):
        """Give positive successful samples a bounded minimum actor-weight share.

        The replay dataset is unchanged.  This only rescales already-positive
        SIL actor weights, so it preserves both the hard ``G - V > 0`` gate and
        the environment coverage of aligned replay.
        """
        zero = weights.new_zeros(())
        one = weights.new_ones(())
        success_mask = (
            sil_success_mask.to(device=weights.device, dtype=torch.bool).view(-1)
            if sil_success_mask is not None
            else torch.zeros_like(weights, dtype=torch.bool)
        )
        success_mass = weights[success_mask].sum()
        fallback_mass = weights[~success_mask].sum()
        total_mass = success_mass + fallback_mass
        share_before = (
            success_mass / total_mass if float(total_mass.item()) > 0.0 else zero
        )

        min_share = float(getattr(self.args, "sil_success_actor_min_share", 0.0))
        boost_cap = float(getattr(self.args, "sil_success_actor_boost_cap", 16.0))
        boost = one
        cap_hit = zero
        if (
            min_share > 0.0
            and float(success_mass.item()) > 0.0
            and float(fallback_mass.item()) > 0.0
            and float(share_before.item()) < min_share
        ):
            required_boost = (
                min_share * fallback_mass / ((1.0 - min_share) * success_mass)
            )
            boost = torch.clamp(required_boost, min=1.0, max=boost_cap)
            cap_hit = (required_boost > boost_cap).to(dtype=weights.dtype)
            weights = torch.where(success_mask, weights * boost, weights)

        success_mass_after = weights[success_mask].sum()
        total_mass_after = success_mass_after + weights[~success_mask].sum()
        share_after = (
            success_mass_after / total_mass_after
            if float(total_mass_after.item()) > 0.0
            else zero
        )
        return weights, share_before, share_after, boost, cap_hit

    def _temporal_gap_for_batch(
        self,
        agent,
        state_batch,
        time_target_batch,
        task_id_batch,
    ):
        if time_target_batch is None:
            raise ValueError("Temporal SIL filtering requires sampled time_targets.")
        if self._uses_standalone_temporal_model():
            value_t = agent.get_temporal_filter_value(
                state_batch.detach().to(device=self.device, dtype=self.dtype),
                update=False,
            ).view(-1)
            target_t = time_target_batch.to(
                device=value_t.device, dtype=value_t.dtype
            ).view(-1)
            raw_gap = value_t - target_t
            if not self._standalone_temporal_ready(agent):
                return torch.zeros_like(raw_gap), value_t, target_t, raw_gap
            return torch.clamp(raw_gap, min=0.0), value_t, target_t, raw_gap

        _, _, value_t = agent.get_value(state_batch, denorm=False, update=False)
        if value_t is None:
            raise ValueError(
                "Temporal SIL filtering requires an agent time-value head."
            )
        target_t = time_target_batch.to(
            device=state_batch.device, dtype=value_t.dtype
        ).view(-1)
        if self.args.norm_value:
            normalize_task_ids = task_id_batch if self.args.pertask_norm else None
            target_t = agent.normalize_value_t(
                target_t,
                denorm=False,
                update=False,
                task_ids=normalize_task_ids,
            )
        value_t = value_t.view(-1)
        raw_gap = value_t - target_t
        return torch.clamp(raw_gap, min=0.0), value_t, target_t, raw_gap

    # Training objective: gates, actor weights, critic targets, and loss reductions.
    def _compute_batch_losses(
        self,
        agent,
        obs_batch,
        state_batch,
        action_batch,
        return_batch,
        task_id_batch,
        time_target_batch=None,
        episode_time_batch=None,
        sil_weight_batch=None,
        sil_success_mask=None,
        sil_return_fill_mask=None,
        bootstrap_value_batch=None,
        bootstrap_mask_batch=None,
        observed_return_batch=None,
        env_id_batch=None,
        loss_scale=1.0,
        return_stats=False,
        success_weight_state=None,
        audit_context=None,
        importance_weight_batch=None,
        frozen_positive_gap_batch=None,
        frozen_actor_weight_batch=None,
    ):
        microbatch = int(getattr(self.args, "sil_microbatch_size", 0))
        logprob = (
            None
            if microbatch
            else agent.get_logprob(obs_batch, action_batch, update=False)
        )
        if self.args.sil_mode == "bc":
            weights = None
            if (
                str(getattr(self.args, "sil_sample_unit", "transition")).lower()
                == "trajectory"
                and sil_weight_batch is not None
            ):
                weights = sil_weight_batch.to(
                    device=logprob.device, dtype=logprob.dtype
                )
            loss = -logprob.mean() if weights is None else -(weights * logprob).mean()
            weighted_logp = float(logprob.detach().mean().item())
            if weights is not None:
                weight_sum = float(weights.detach().sum().item())
                if weight_sum > 0:
                    weighted_logp = float(
                        (weights.detach() * logprob.detach()).sum().item() / weight_sum
                    )
            sample_success_frac = (
                float(sil_success_mask.to(dtype=torch.float32).mean().item())
                if sil_success_mask is not None and int(sil_success_mask.numel()) > 0
                else 0.0
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

        filter_mode = str(getattr(self.args, "sil_filter_mode", "return")).lower()
        return_gap, state_value, _, raw_return_gap = self._positive_gap_for_batch(
            agent,
            state_batch,
            return_batch,
            task_id_batch,
            return_value=True,
        )
        temporal_gap = torch.zeros_like(return_gap)
        raw_temporal_gap = torch.zeros_like(return_gap)
        temporal_value_loss = torch.tensor(
            0.0, device=return_gap.device, dtype=return_gap.dtype
        )
        no_success_samples = (
            sil_success_mask is not None
            and int(sil_success_mask.numel()) > 0
            and not bool(sil_success_mask.to(dtype=torch.bool).any().item())
        )
        temporal_unavailable = (
            filter_mode in {"temporal", "both"}
            and self._uses_standalone_temporal_model()
            and not self._standalone_temporal_ready(agent)
        )
        if filter_mode in {"temporal", "both"}:
            temporal_gap, value_t, target_t, raw_temporal_gap = (
                self._temporal_gap_for_batch(
                    agent,
                    state_batch,
                    time_target_batch,
                    task_id_batch,
                )
            )
            temporal_value_terms = 0.5 * (value_t - target_t) ** 2
            if (
                bool(getattr(self.args, "sil_temporal_success_only", True))
                and sil_success_mask is not None
            ):
                success_mask = sil_success_mask.to(
                    device=temporal_gap.device, dtype=torch.bool
                )
                temporal_gap = torch.where(
                    success_mask, temporal_gap, torch.zeros_like(temporal_gap)
                )
                temporal_value_loss = (
                    temporal_value_terms[success_mask].mean()
                    if bool(success_mask.any().item())
                    else torch.zeros(
                        (), device=temporal_gap.device, dtype=temporal_gap.dtype
                    )
                )
            else:
                temporal_value_loss = temporal_value_terms.mean()

        if filter_mode == "return":
            positive_gap = return_gap
            actor_gap_residual = raw_return_gap
            value_loss_gap = 0.5 * (return_gap**2).mean()
        elif filter_mode == "temporal":
            use_return_fallback = no_success_samples or temporal_unavailable
            positive_gap = return_gap if use_return_fallback else temporal_gap
            actor_gap_residual = (
                raw_return_gap if use_return_fallback else raw_temporal_gap
            )
            value_loss_gap = (
                0.5 * (return_gap**2).mean()
                if use_return_fallback
                else torch.zeros_like(return_gap).mean()
            )
            if not self._uses_standalone_temporal_model():
                value_loss_gap = value_loss_gap + temporal_value_loss
        elif filter_mode == "both":
            positive_gap = torch.where(
                temporal_gap > 0, return_gap, torch.zeros_like(return_gap)
            )
            actor_gap_residual = raw_return_gap
            value_loss_gap = 0.5 * (return_gap**2).mean()
            if not self._uses_standalone_temporal_model():
                value_loss_gap = value_loss_gap + temporal_value_loss
        else:
            raise ValueError(
                "sil_filter_mode must be 'return', 'temporal', or 'both', "
                f"got {filter_mode}"
            )

        # Freeze the actor's raw positive advantage for the whole update. The
        # default per-slice normalization is retained unless dataset-normalized
        # actor weights are cached below. Critic loss stays current-gap SIL.
        if frozen_positive_gap_batch is not None:
            if filter_mode != "return":
                raise ValueError("Frozen actor gap requires return filtering")
            positive_gap = frozen_positive_gap_batch.detach().to(return_gap)

        if microbatch:
            actor_mask = positive_gap.detach() > 0
            if sil_weight_batch is not None:
                actor_mask = actor_mask & (sil_weight_batch > 0)
            if self._full_actor_diagnostics:
                actor_mask = torch.ones_like(actor_mask)
            logprob = chunked_logprob(
                agent, obs_batch, action_batch, actor_mask, microbatch
            )
        actor_multiplier, actor_need, actor_nll = self._actor_mismatch_multiplier(
            logprob
        )
        pre_actor_gap = positive_gap
        actor_rescued_mask = torch.zeros_like(positive_gap, dtype=torch.bool)
        actor_margin = float(getattr(self.args, "sil_actor_temporal_margin", 0.0))
        if (
            float(getattr(self.args, "sil_actor_weight_coef", 0.0)) > 0
            and actor_margin > 0
            and filter_mode == "temporal"
            and not no_success_samples
        ):
            margin_gap = torch.clamp(
                raw_temporal_gap + actor_margin, min=0.0, max=actor_margin
            )
            margin_gap = margin_gap * torch.clamp(actor_need, min=0.0, max=1.0)
            positive_gap = torch.maximum(positive_gap, margin_gap)
            actor_rescued_mask = (pre_actor_gap <= 0) & (positive_gap > 0)

        positive_mask = positive_gap > 0
        closed_gap_mask = actor_gap_residual.detach().abs() <= 1e-3
        high_nll_closed_gap_mask = closed_gap_mask & (actor_need.detach() > 0)
        weights = positive_gap.detach()
        if sil_weight_batch is not None:
            weights = weights * sil_weight_batch.to(
                device=weights.device, dtype=weights.dtype
            )
        if frozen_actor_weight_batch is None:
            weights = self._normalize_actor_gap_weights(weights)
        else:
            if frozen_positive_gap_batch is None:
                raise ValueError("Cached actor weights require frozen gaps")
            weights = frozen_actor_weight_batch.detach().to(weights)
        if importance_weight_batch is not None:
            # Apply AFTER actor normalization/capping. Do not renormalize away
            # the probability correction. Critic receives the same IS factors.
            weights = weights * importance_weight_batch.detach().to(weights)
        actor_bonus = torch.zeros_like(weights)
        actor_add_gate = torch.zeros_like(weights, dtype=torch.bool)
        if float(getattr(self.args, "sil_actor_weight_coef", 0.0)) > 0:
            actor_multiplier = actor_multiplier.to(
                device=weights.device, dtype=weights.dtype
            )
            actor_mode = str(
                getattr(self.args, "sil_actor_weight_mode", "multiply")
            ).lower()
            if actor_mode == "multiply":
                weights = weights * actor_multiplier
            elif actor_mode in {"add", "add_norm", "rescue_norm"}:
                base_weights = weights
                base_active = base_weights > 0
                base_mean = (
                    base_weights[base_active].mean()
                    if bool(base_active.any().item())
                    else torch.tensor(
                        float(getattr(self.args, "sil_actor_gap_alpha", 0.0)),
                        device=weights.device,
                        dtype=weights.dtype,
                    )
                )
                actor_bonus = torch.clamp(actor_multiplier - 1.0, min=0.0)
                alpha = float(getattr(self.args, "sil_actor_gap_alpha", 0.0))
                if actor_mode == "rescue_norm" and sil_success_mask is not None:
                    # Preserve positive critic evidence and reopen only a small negative margin.
                    actor_add_gate = sil_success_mask.to(
                        device=weights.device, dtype=torch.bool
                    ) & (actor_gap_residual.detach() >= -alpha)
                else:
                    actor_add_gate = base_weights > alpha
                weights = base_weights + torch.where(
                    actor_add_gate, actor_bonus, torch.zeros_like(actor_bonus)
                )
                if actor_mode in {"add_norm", "rescue_norm"}:
                    active = weights > 0
                    if (
                        bool(active.any().item())
                        and float(base_mean.detach().item()) > 0
                    ):
                        weights = weights * (
                            base_mean / (weights[active].mean() + 1e-8)
                        )
                actor_rescued_mask = actor_rescued_mask | (
                    actor_add_gate
                    & (base_weights.detach() <= 0)
                    & (weights.detach() > 0)
                )
            else:
                raise ValueError(
                    "sil_actor_weight_mode must be 'multiply', 'add', 'add_norm', or 'rescue_norm', "
                    f"got {actor_mode}"
                )

        weights, retention_weights, retention_applied, fast_success_mask = (
            self._apply_fast_success_retention(
                weights,
                raw_return_gap,
                sil_success_mask,
                sil_return_fill_mask,
            )
        )
        actor_rescued_mask = actor_rescued_mask | (
            retention_applied & (pre_actor_gap.detach() <= 0)
        )

        weights_before_success_floor = weights
        success_floor = float(getattr(self.args, "sil_success_actor_floor", 0.0))
        success_floor_applied = torch.zeros_like(weights, dtype=torch.bool)
        if success_floor > 0.0 and sil_success_mask is not None:
            floor_success_mask = sil_success_mask.to(
                device=weights.device, dtype=torch.bool
            )
            success_floor_applied = floor_success_mask & (
                weights.detach() < success_floor
            )
            weights = torch.where(
                floor_success_mask,
                torch.maximum(weights, torch.full_like(weights, success_floor)),
                weights,
            )

        (
            weights,
            success_weight_share_before,
            success_weight_share,
            success_weight_boost,
            success_weight_boost_cap_hit,
        ) = self._balance_success_actor_share(weights, sil_success_mask)

        success_weight_stats = {}
        if getattr(self.args, "sil_success_actor_schedule", "off") != "off":
            from projects.TSIL.success_weighting import success_actor_weights

            weights, success_weight_stats = success_actor_weights(
                weights, sil_success_mask, env_id_batch, success_weight_state, self.args
            )
        raw_weight_mean = weights.detach().mean()
        target_weight_mean = float(
            getattr(self.args, "sil_replay_weight_target_mean", 0.0)
        )
        if target_weight_mean > 0 and float(raw_weight_mean.item()) > 0:
            weights = weights * (target_weight_mean / raw_weight_mean)

        if audit_context is not None and getattr(self.args, "sil_audit_interval", 0):
            from projects.TSIL.replay_audit import capture

            capture(
                self,
                agent,
                audit_context[0],
                audit_context[1],
                state_value,
                raw_return_gap,
                weights,
            )

        # Keep the default reduction byte-for-byte equivalent. Both new losses
        # use the positive return-gap count over this entire optimizer slice.
        positive_count = (return_gap.detach() > 0).sum()
        positive_reduction = (
            getattr(self.args, "sil_loss_denominator", "all") == "positive"
        )
        if positive_reduction:
            denominator = positive_count.clamp_min(1).to(dtype=weights.dtype)
            if getattr(self.args, "sil_global_per", False):
                denominator = positive_count.clamp_min(64).to(dtype=weights.dtype)
            active = return_gap.detach() > 0
            actor_active = (
                active if frozen_positive_gap_batch is None else positive_gap > 0
            )
            actor_denominator = (
                denominator
                if frozen_positive_gap_batch is None
                else actor_active.sum().clamp_min(1).to(weights)
            )
            policy_loss = float(loss_scale) * (
                -(weights[actor_active] * logprob[actor_active]).sum()
                / actor_denominator
            )
            value_loss_gap = 0.5 * return_gap[active].square().sum() / denominator
            if importance_weight_batch is not None:
                critic_terms = 0.5 * return_gap[active].square()
                if getattr(self.args, "sil_global_per", False):
                    # Released SIL's clipped positive-gap critic gradient.
                    critic_terms = (
                        -0.5
                        * state_value[active]
                        * return_gap[active].detach().clamp(max=1.0)
                    )
                value_loss_gap = (
                    importance_weight_batch[active].detach() * critic_terms
                ).sum() / denominator
        else:
            denominator = weights.new_tensor(weights.numel())
            policy_loss = float(loss_scale) * (-(weights * logprob).mean())
        value_loss = float(loss_scale) * value_loss_gap
        self._last_actor_components = None
        if audit_context is not None and sil_success_mask is not None:
            interval = int(getattr(self.args, "signal_gradient_interval", 0))
            iteration = int(audit_context[1].get("audit_iteration", 0))
            if interval > 0 and iteration % interval == 0:
                terms = (
                    -(weights * logprob)
                    * float(loss_scale)
                    / (actor_denominator if positive_reduction else denominator)
                )
                success_rows = sil_success_mask.bool()
                self._last_actor_components = (
                    terms[success_rows].sum(),
                    terms[~success_rows].sum(),
                )
        positive_frac = float(positive_mask.float().mean().item())
        weight_sum = float(weights.detach().sum().item())
        weighted_logp = (
            float((weights.detach() * logprob.detach()).sum().item() / weight_sum)
            if weight_sum > 0
            else float(logprob.detach().mean().item())
        )

        def masked_mean(values, mask):
            return float(values[mask].mean().item()) if bool(mask.any().item()) else 0.0

        def masked_correlation(values, targets, mask):
            if int(mask.sum().item()) < 2:
                return 0.0
            values = values[mask].float()
            targets = targets[mask].float()
            values = values - values.mean()
            targets = targets - targets.mean()
            denom = values.square().mean().sqrt() * targets.square().mean().sqrt()
            if float(denom.item()) <= 1e-8:
                return 0.0
            return float(((values * targets).mean() / denom).item())

        success_mask = (
            sil_success_mask.to(device=positive_gap.device, dtype=torch.bool)
            if sil_success_mask is not None
            else torch.zeros_like(positive_gap, dtype=torch.bool)
        )
        fast_success = (
            -episode_time_batch.to(
                device=positive_gap.device, dtype=positive_gap.dtype
            ).view(-1)
            if episode_time_batch is not None
            else torch.zeros_like(positive_gap)
        )

        stats = {
            "memory_logp": float(logprob.detach().mean().item()),
            "memory_weighted_logp": weighted_logp,
            "memory_positive_gap_mean": float(positive_gap.detach().mean().item()),
            "replay_filter_mode": filter_mode,
            "replay_temporal_model_source": self._temporal_model_source(),
            "replay_temporal_model_ready": int(self._standalone_temporal_ready(agent)),
            "replay_temporal_model_seen_samples": 0,
            "replay_return_gap_mean": float(return_gap.detach().mean().item()),
            "replay_advantage_gap_mean": float(return_gap.detach().mean().item()),
            "replay_temporal_positive_frac": float(
                (temporal_gap > 0).float().mean().item()
            ),
            "replay_temporal_gap_mean": float(temporal_gap.detach().mean().item()),
            "replay_return_gap_fast_success_corr": masked_correlation(
                return_gap.detach(),
                fast_success,
                success_mask,
            ),
            "replay_temporal_gap_fast_success_corr": masked_correlation(
                temporal_gap.detach(),
                fast_success,
                success_mask,
            ),
            "replay_temporal_value_loss": float(temporal_value_loss.detach().item()),
            "replay_actor_nll_mean": float(actor_nll.detach().mean().item()),
            "replay_actor_need_mean": float(actor_need.detach().mean().item()),
            "replay_actor_weight_mean": float(actor_multiplier.detach().mean().item()),
            "replay_actor_bonus_mean": float(actor_bonus.detach().mean().item()),
            "replay_actor_add_gate_frac": masked_mean(
                actor_add_gate.to(dtype=torch.float32),
                success_mask,
            ),
            "replay_actor_abs_gap_mean": float(
                actor_gap_residual.detach().abs().mean().item()
            ),
            "replay_actor_closed_gap_frac": float(
                closed_gap_mask.to(dtype=torch.float32).mean().item()
            ),
            "replay_actor_high_nll_closed_gap_frac": float(
                high_nll_closed_gap_mask.to(dtype=torch.float32).mean().item()
            ),
            "replay_actor_closed_gap_nll_mean": masked_mean(
                actor_nll.detach(), closed_gap_mask
            ),
            "replay_sil_weight_mean": float(weights.detach().mean().item()),
            "replay_sil_weight_raw_mean": float(raw_weight_mean.item()),
            "replay_sil_weight_before_success_floor_mean": float(
                weights_before_success_floor.detach().mean().item()
            ),
            **success_weight_stats,
            "replay_success_actor_floor": success_floor,
            "replay_success_actor_floor_applied_frac": masked_mean(
                success_floor_applied.to(dtype=torch.float32),
                success_mask,
            ),
            "replay_effective_success_frac": masked_mean(
                (weights.detach() > 0).to(dtype=torch.float32),
                success_mask,
            ),
            "replay_success_actor_weight_share_before": float(
                success_weight_share_before.detach().item()
            ),
            "replay_success_actor_weight_share": float(
                success_weight_share.detach().item()
            ),
            "replay_success_actor_boost": float(success_weight_boost.detach().item()),
            "replay_success_actor_boost_cap_hit": float(
                success_weight_boost_cap_hit.detach().item()
            ),
            "replay_fast_success_retention_applied_frac": masked_mean(
                retention_applied.to(dtype=torch.float32),
                fast_success_mask,
            ),
            "replay_fast_success_retention_weight_mean": masked_mean(
                retention_weights.detach(),
                fast_success_mask,
            ),
            "replay_effective_fast_success_frac": masked_mean(
                (weights.detach() > 0).to(dtype=torch.float32),
                fast_success_mask,
            ),
            "replay_temporal_fallback_active": int(
                temporal_unavailable or no_success_samples
            ),
            "replay_actor_rescued_frac": masked_mean(
                actor_rescued_mask.to(dtype=torch.float32),
                success_mask,
            ),
            "replay_sample_return_mean": float(return_batch.detach().mean().item()),
            "replay_state_value_mean": float(state_value.detach().mean().item()),
        }
        if env_id_batch is not None and int(env_id_batch.numel()) == int(
            positive_mask.numel()
        ):
            sample_env_ids = env_id_batch.to(
                device=positive_mask.device, dtype=torch.long
            ).view(-1)
            sampled_env_count = max(int(torch.unique(sample_env_ids).numel()), 1)
            positive_env_count = (
                int(torch.unique(sample_env_ids[positive_mask]).numel())
                if bool(positive_mask.any().item())
                else 0
            )
            effective_mask = weights.detach() > 0
            effective_env_count = (
                int(torch.unique(sample_env_ids[effective_mask]).numel())
                if bool(effective_mask.any().item())
                else 0
            )
            stats.update(
                {
                    "replay_positive_sample_unique_envs": float(positive_env_count),
                    "replay_positive_sample_unique_env_frac": float(
                        positive_env_count / sampled_env_count
                    ),
                    "replay_effective_sample_unique_envs": float(effective_env_count),
                    "replay_effective_sample_unique_env_frac": float(
                        effective_env_count / sampled_env_count
                    ),
                }
            )
        if observed_return_batch is not None:
            observed_returns = observed_return_batch.to(
                device=return_gap.device, dtype=return_gap.dtype
            ).view(-1)
            bootstrap_delta = return_batch.detach().view(-1) - observed_returns.detach()
            stats["replay_observed_return_mean"] = float(
                observed_returns.detach().mean().item()
            )
            stats["replay_bootstrap_return_delta_mean"] = float(
                bootstrap_delta.detach().mean().item()
            )
        if bootstrap_value_batch is not None:
            bootstrap_values = bootstrap_value_batch.to(
                device=return_gap.device, dtype=return_gap.dtype
            ).view(-1)
            stats["replay_bootstrap_value_mean"] = float(
                bootstrap_values.detach().mean().item()
            )
            if bootstrap_mask_batch is not None:
                bootstrap_mask = bootstrap_mask_batch.to(
                    device=return_gap.device, dtype=torch.bool
                ).view(-1)
            else:
                bootstrap_mask = bootstrap_values != 0
            stats["replay_bootstrap_active_frac"] = float(
                bootstrap_mask.to(dtype=torch.float32).mean().item()
            )
            stats["replay_bootstrap_active_value_mean"] = masked_mean(
                bootstrap_values.detach(), bootstrap_mask
            )
        if sil_success_mask is not None and int(sil_success_mask.numel()) > 0:
            fallback_mask = ~success_mask
            policy_terms = -(weights.detach() * logprob.detach())

            stats.update(
                {
                    "replay_sample_success_frac": float(
                        success_mask.to(dtype=torch.float32).mean().item()
                    ),
                    "replay_sample_fallback_frac": float(
                        fallback_mask.to(dtype=torch.float32).mean().item()
                    ),
                    "replay_positive_success_frac": masked_mean(
                        positive_mask.to(dtype=torch.float32), success_mask
                    ),
                    "replay_positive_fallback_frac": masked_mean(
                        positive_mask.to(dtype=torch.float32), fallback_mask
                    ),
                    "replay_positive_success_precision": masked_mean(
                        success_mask.to(dtype=torch.float32), positive_mask
                    ),
                    "replay_gap_success_mean": masked_mean(
                        positive_gap.detach(), success_mask
                    ),
                    "replay_gap_fallback_mean": masked_mean(
                        positive_gap.detach(), fallback_mask
                    ),
                    "replay_policy_loss_success": masked_mean(
                        policy_terms, success_mask
                    ),
                    "replay_policy_loss_fallback": masked_mean(
                        policy_terms, fallback_mask
                    ),
                    "replay_sample_return_success_mean": masked_mean(
                        return_batch.detach(), success_mask
                    ),
                    "replay_sample_return_fallback_mean": masked_mean(
                        return_batch.detach(), fallback_mask
                    ),
                    "replay_state_value_fallback_mean": masked_mean(
                        state_value.detach(), fallback_mask
                    ),
                }
            )
            if observed_return_batch is not None:
                observed_returns = observed_return_batch.to(
                    device=return_gap.device, dtype=return_gap.dtype
                ).view(-1)
                bootstrap_delta = (
                    return_batch.detach().view(-1) - observed_returns.detach()
                )
                stats["replay_observed_return_fallback_mean"] = masked_mean(
                    observed_returns.detach(),
                    fallback_mask,
                )
                stats["replay_bootstrap_return_delta_fallback_mean"] = masked_mean(
                    bootstrap_delta.detach(),
                    fallback_mask,
                )
            if bootstrap_value_batch is not None:
                bootstrap_values = bootstrap_value_batch.to(
                    device=return_gap.device, dtype=return_gap.dtype
                ).view(-1)
                stats["replay_bootstrap_minus_state_value_fallback_mean"] = masked_mean(
                    bootstrap_values.detach() - state_value.detach(),
                    fallback_mask,
                )
        stats.update(
            {
                "replay_full_actor_diagnostics_measured": int(
                    not microbatch or self._full_actor_diagnostics
                ),
                "replay_actor_forward_frac": (
                    float(actor_mask.float().mean().item()) if microbatch else 1.0
                ),
                "replay_microbatch_size": microbatch,
                "replay_loss_denominator": (
                    float(denominator.item())
                    if positive_count.item() or not positive_reduction
                    else 0.0
                ),
                "replay_positive_count": float(positive_count.item()),
                "replay_candidate_count": float(weights.numel()),
                "replay_denominator_multiplier": (
                    float(weights.numel() / max(int(positive_count.item()), 1))
                    if positive_count.item()
                    else 0.0
                ),
                "replay_empty_positive_slice": float(positive_count.item() == 0),
            }
        )
        if microbatch and not self._full_actor_diagnostics:
            # Inactive log-probabilities are placeholders used only in zero
            # loss terms. Never report them as measured population statistics.
            for key in (
                "memory_logp",
                "replay_actor_nll_mean",
                "replay_actor_need_mean",
                "replay_actor_high_nll_closed_gap_frac",
                "replay_actor_closed_gap_nll_mean",
            ):
                stats[key] = float("nan")
            if weight_sum <= 0:
                stats["memory_weighted_logp"] = float("nan")
        if audit_context is not None and getattr(self.args, "sil_global_per", False):
            batch, context = audit_context
            context["global_per"].update_priorities(
                batch["per_indices"], return_gap.detach().clamp(max=1.0)
            )
        result = (policy_loss, value_loss, positive_frac)
        return (*result, stats) if return_stats else result

    def _coverage_stats_for_batch(self, batch, replay_context):
        if batch is None or "env_ids" not in batch or "slot_ids" not in batch:
            return {}
        env_ids = batch["env_ids"].long()
        slot_ids = batch["slot_ids"].long()
        count = int(env_ids.numel())
        if count <= 0:
            return {}

        unique_envs, env_counts = torch.unique(env_ids, return_counts=True)
        probs = env_counts.float() / float(count)
        effective_envs = torch.exp(-(probs * torch.log(probs.clamp_min(1e-12))).sum())
        top_counts = torch.topk(env_counts, k=min(5, int(env_counts.numel()))).values
        pairs = torch.stack([env_ids, slot_ids], dim=1)

        success_mask = batch.get("sil_success_mask")
        if success_mask is None or int(success_mask.numel()) != count:
            success_mask = torch.zeros(count, dtype=torch.bool, device=env_ids.device)
        else:
            success_mask = success_mask.to(device=env_ids.device, dtype=torch.bool)
        fallback_mask = ~success_mask

        candidate_envs = []
        for key in ("success_env_ids", "fallback_env_ids"):
            values = (replay_context or {}).get(key)
            if values is not None and int(values.numel()) > 0:
                candidate_envs.append(values.long())
        candidate_count = (
            int(torch.unique(torch.cat(candidate_envs)).numel())
            if candidate_envs
            else 0
        )
        candidate_count = max(candidate_count, int(unique_envs.numel()), 1)

        success_unique_envs = (
            int(torch.unique(env_ids[success_mask]).numel())
            if bool(success_mask.any().item())
            else 0
        )
        fallback_unique_envs = (
            int(torch.unique(env_ids[fallback_mask]).numel())
            if bool(fallback_mask.any().item())
            else 0
        )
        success_duplicate_frac = 0.0
        if bool(success_mask.any().item()):
            success_triples = torch.stack(
                [
                    env_ids[success_mask],
                    slot_ids[success_mask],
                    batch["time_idx"][success_mask].long(),
                ],
                dim=1,
            )
            success_count = int(success_triples.shape[0])
            unique_success_count = int(torch.unique(success_triples, dim=0).shape[0])
            success_duplicate_frac = float(
                (success_count - unique_success_count) / max(success_count, 1)
            )

        stats = {
            "replay_sample_unique_envs": float(int(unique_envs.numel())),
            "replay_sample_unique_env_frac": float(
                int(unique_envs.numel()) / candidate_count
            ),
            "replay_success_sample_unique_envs": float(success_unique_envs),
            "replay_fallback_sample_unique_envs": float(fallback_unique_envs),
            "replay_sample_unique_trajectories": float(
                int(torch.unique(pairs, dim=0).shape[0])
            ),
            "replay_sample_effective_envs": float(effective_envs.detach().item()),
            "replay_sample_top1_env_frac": float(
                env_counts.max().float().item() / count
            ),
            "replay_sample_top5_env_frac": float(
                top_counts.sum().float().item() / count
            ),
            "replay_success_duplicate_transition_frac": success_duplicate_frac,
            "replay_env_step_quota": float(batch.get("selection_env_step_quota", 0)),
        }
        for batch_key, stat_key in (
            ("success_terminal_value_mean", "replay_success_terminal_value_mean"),
            ("success_terminal_value_max", "replay_success_terminal_value_max"),
            ("success_terminal_negative_frac", "replay_success_terminal_negative_frac"),
            (
                "success_terminal_added_value_mean",
                "replay_success_terminal_added_value_mean",
            ),
            ("aligned_candidate_envs", "replay_aligned_candidate_envs"),
            ("aligned_selected_trajectories", "replay_aligned_selected_trajectories"),
            ("aligned_dataset_steps", "replay_aligned_dataset_steps"),
            (
                "aligned_mean_trajectories_per_env",
                "replay_aligned_mean_trajectories_per_env",
            ),
            ("aligned_fastest_source_frac", "replay_aligned_fastest_source_frac"),
            ("aligned_return_fill_frac", "replay_aligned_return_fill_frac"),
            ("aligned_short_trajectory_frac", "replay_aligned_short_trajectory_frac"),
            ("aligned_transition_wrap_frac", "replay_aligned_transition_wrap_frac"),
            ("aligned_replay_epochs", "replay_aligned_replay_epochs"),
        ):
            stats[stat_key] = float(batch.get(batch_key, 0.0))
        return stats

    def compute_joint_losses(self, agent, replay_context):
        if (
            not self.args.use_sil
            or not bool(getattr(self.args, "sil_train", True))
            or replay_context is None
        ):
            if replay_context is not None:
                replay_context["_last_sil_batch_stats"] = replay_context.pop(
                    "_temporal_model_update_stats",
                    {},
                )
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
                if getattr(self.args, "sil_audit_interval", 0):
                    from projects.TSIL.replay_audit import record_empty

                    record_empty(self.args, replay_context)
                replay_context["_last_sil_batch_stats"] = replay_context.pop(
                    "_temporal_model_update_stats",
                    {},
                )
                return (
                    torch.tensor(0.0, device=self.device),
                    torch.tensor(0.0, device=self.device),
                    0.0,
                    0.0,
                    0,
                )

            if getattr(self.args, "sil_actual_direction_diagnostics", False):
                replay_context["diagnostic_replay_batch"] = batch
            policy_loss, value_loss, positive_frac, batch_stats = (
                self._compute_batch_losses(
                    agent,
                    batch["obs"],
                    batch["state"],
                    batch["action"],
                    batch["returns"],
                    batch["task_ids"],
                    time_target_batch=batch.get("time_targets"),
                    episode_time_batch=batch.get("episode_times"),
                    sil_weight_batch=batch.get("sil_weights"),
                    importance_weight_batch=batch.get("sil_importance_weights"),
                    frozen_positive_gap_batch=batch.get("sil_frozen_positive_gap"),
                    frozen_actor_weight_batch=batch.get("sil_frozen_actor_weights"),
                    sil_success_mask=batch.get("sil_success_mask"),
                    sil_return_fill_mask=batch.get("sil_return_fill_mask"),
                    bootstrap_value_batch=batch.get("sil_bootstrap_values"),
                    bootstrap_mask_batch=batch.get("sil_bootstrap_mask"),
                    observed_return_batch=batch.get("observed_returns"),
                    env_id_batch=batch.get("env_ids"),
                    success_weight_state=(
                        replay_context.get("schedule_context") or {}
                    ).get("success_weight_state"),
                    return_stats=True,
                    audit_context=(batch, replay_context),
                )
            )
            batch_stats.update(self._coverage_stats_for_batch(batch, replay_context))
            replay_context["_last_sil_batch_stats"] = batch_stats
            batch_stats.update(replay_context.pop("_temporal_model_update_stats", {}))
            return (
                policy_loss,
                value_loss,
                positive_frac,
                positive_frac,
                int(batch["action"].shape[0]),
            )

        replay_context["_last_sil_batch_stats"] = replay_context.pop(
            "_temporal_model_update_stats",
            {},
        )
        return (
            torch.tensor(0.0, device=self.device),
            torch.tensor(0.0, device=self.device),
            0.0,
            0.0,
            0,
        )

    # Diagnostics below must not mutate training parameters or replay membership.
    def pre_step_landscape_due(self, iteration):
        if (
            not getattr(self.args, "sil_actual_direction_diagnostics", False)
            or getattr(self.args, "sil_direction_timing", "pre_update") != "pre_update"
            or int(getattr(self.args, "sil_analysis_interval", 0)) <= 0
        ):
            return False
        fraction = float(getattr(self.args, "sil_landscape_progress_interval", 0.05))
        total = int(getattr(self.args, "total_iters", 0))
        if fraction > 0 and total > 0:
            targets = {
                max(0, math.ceil(min(k * fraction, 1.0) * total) - 1)
                for k in range(1, math.ceil(1.0 / fraction) + 1)
            }
            return int(iteration) in targets
        return iteration > 0 and iteration % int(self.args.sil_analysis_interval) == 0

    def _positive_replay_reference(self, agent, batch, limit):
        """Diagnostic reference from active successful entries in the first slice.

        Membership is fixed at the origin, including throughout the grid sweep.
        This changes neither the training loss nor its measured gradient.
        """
        if batch is None or "sil_success_mask" not in batch:
            return None, 0
        with torch.no_grad():
            gap = batch.get("sil_frozen_positive_gap")
            if gap is None:
                gap = self._positive_gap_for_batch(
                    agent, batch["state"], batch["returns"], batch["task_ids"]
                )
            weights = batch.get(
                "sil_frozen_actor_weights",
                batch.get("sil_weights", torch.ones_like(gap)),
            )
            indices = torch.nonzero(
                batch["sil_success_mask"].bool() & (gap > 0) & (weights > 0),
                as_tuple=True,
            )[0]
            eligible = int(indices.numel())
            if not eligible:
                return None, 0
            if eligible > limit:
                indices = indices[
                    torch.linspace(0, eligible - 1, limit, device=indices.device)
                    .round()
                    .long()
                ]
            if "env_ids" in batch and "slot_ids" in batch:
                _, traj_ids = torch.unique(
                    torch.stack(
                        (batch["env_ids"][indices], batch["slot_ids"][indices]), dim=1
                    ),
                    dim=0,
                    return_inverse=True,
                )
            else:
                traj_ids = torch.arange(indices.numel(), device=indices.device)
            reference = {
                k: batch[k][indices]
                for k in ("obs", "state", "action", "returns", "task_ids")
            }
            reference.update(
                traj_ids=traj_ids,
                ref_count=int(traj_ids.max()) + 1,
                reference_kind="Positive-gap successful replay",
                reference_indices_sha256=hashlib.sha256(
                    indices.cpu().numpy().tobytes()
                ).hexdigest(),
            )
            return reference, eligible

    def capture_pre_step_landscape(
        self,
        agent,
        context,
        ppo_loss,
        replay_loss,
        iteration,
        indices,
        valid,
        epoch,
        minibatch,
    ):
        """Differentiate the actual first optimizer losses; never resample replay."""
        if context is None or not self.pre_step_landscape_due(iteration):
            return
        if context.get("_pre_landscape_attempted"):
            return
        context["_pre_landscape_attempted"] = True
        params = self._trainable_params(agent)

        def direction(loss):
            if loss is None or not loss.requires_grad:
                return torch.zeros_like(parameters_to_vector(params))
            return -self._grad_vector(loss, params, retain_graph=True).detach()

        probe = dict(ppo=direction(ppo_loss), replay=direction(replay_loss))
        replay = context.get("diagnostic_replay_batch")
        if getattr(self.args, "sil_normalization_audit", False):
            from core.training.algo.tsil.normalization_diagnostics import (
                record_normalization_audit,
            )

            record_normalization_audit(self, agent, context, iteration, probe["replay"])
        ids = torch.as_tensor(indices).detach().cpu().numpy().tobytes()
        metadata = dict(
            direction_semantics="pre_update_first_optimizer_slices_actual_replay_v1",
            epoch=int(epoch),
            minibatch=int(minibatch),
            ppo_slice_count=int(len(indices)),
            ppo_valid_count=int(valid.sum()),
            replay_slice_count=0 if replay is None else int(replay["action"].shape[0]),
            ppo_indices_sha256=hashlib.sha256(ids).hexdigest(),
            completed_rollout_fraction=(int(iteration) + 1)
            / max(int(getattr(self.args, "total_iters", 1)), 1),
            replay_gap_mode=(
                "frozen_update"
                if getattr(self.args, "sil_freeze_replay_gap", False)
                else "current_critic"
            ),
            replay_normalization_scope=getattr(
                self.args, "sil_replay_normalization_scope", "slice"
            ),
            reference_selection=getattr(
                self.args, "sil_landscape_reference", "fast_success"
            ),
            sil_coef=float(getattr(self.args, "sil_coef", 0.0)),
            reference_role="surface_and_coordinate_axis_only",
            parameter_time="before_first_optimizer_step",
        )
        # Reference construction is deterministic; fork RNG to keep diagnostics
        # from altering future training randomness if helpers gain stochasticity.
        devices = sorted({p.device.index for p in params if p.is_cuda})
        with torch.random.fork_rng(devices=devices):
            limit = max(int(getattr(self.args, "sil_analysis_batch_size", 1024)), 1)
            if metadata["reference_selection"] == "top1_gap_ranked":
                from core.training.algo.tsil.landscape_reference import (
                    top1_gap_ranked_reference,
                )

                reference = top1_gap_ranked_reference(self, agent, context)
                metadata["reference_membership"] = (
                    "top1_per_env_top5_signed_gap_fixed_at_origin"
                )
                reason = "no_fast_success_reference"
            elif metadata["reference_selection"] == "positive_replay_success":
                reference, eligible = self._positive_replay_reference(
                    agent, replay, limit
                )
                metadata["reference_eligible_positive_success_count"] = eligible
                metadata["reference_membership"] = "fixed_at_pre_update_origin"
                reason = "no_positive_success_in_first_replay_slice"
            else:
                reference = self._fastest_reference_batch(
                    context, max_total_steps=limit
                )
                reason = "no_fast_success_reference"
            if reference is None:
                folder = getattr(self.args, "trajectory_dir", None)
                if folder:
                    os.makedirs(folder, exist_ok=True)
                    path = os.path.join(
                        folder, f"sil_direction_landscape_iter{iteration:06d}.json"
                    )
                    with open(path, "w") as stream:
                        json.dump(
                            dict(
                                iteration=int(iteration),
                                available=False,
                                unavailable_reason=reason,
                                ppo_norm=float(probe["ppo"].norm()),
                                replay_norm=float(probe["replay"].norm()),
                                **metadata,
                            ),
                            stream,
                        )
                return
            metadata["reference_sample_count"] = int(reference["action"].shape[0])
            if "reference_indices_sha256" in reference:
                metadata["reference_indices_sha256"] = reference[
                    "reference_indices_sha256"
                ]
            self._update_direction_diagnostics(
                agent,
                context,
                None,
                None,
                iteration,
                reference,
                context.setdefault("stats", self.default_stats()),
                probe=probe,
                metadata=metadata,
            )

    def finalize_pre_step_displacement(self, context, displacement):
        basis = context.pop("_pre_landscape_basis", None)
        path = context.pop("_pre_landscape_path", None)
        if basis is None or path is None:
            return
        bx, by, span = basis
        norm = displacement.norm()
        unit = (
            displacement / norm if float(norm) > 0 else torch.zeros_like(displacement)
        )
        with open(path) as stream:
            payload = json.load(stream)
        payload.update(
            optimizer_update=[
                float(torch.dot(unit, bx) * 0.8 * span),
                float(torch.dot(unit, by) * 0.8 * span),
            ],
            optimizer_displacement_norm=float(norm),
            optimizer_displacement_semantics="net_actor_change_over_update_not_replay_only",
        )
        with open(path + ".tmp", "w") as stream:
            json.dump(payload, stream)
        os.replace(path + ".tmp", path)

    def update_diagnostics(
        self, agent, replay_context, rollout_batch, updater, global_update_iter
    ):
        if replay_context is not None:
            replay_context["audit_iteration"] = int(global_update_iter)
        interval = int(getattr(self.args, "sil_analysis_interval", 0))
        self._full_actor_diagnostics = (
            interval > 0 and int(global_update_iter) % interval == 0
        )
        if interval <= 0:
            return
        if (
            not self.args.use_sil
            or replay_context is None
            or not replay_context.get("has_tsil_memory")
        ):
            return

        stats = replay_context.setdefault("stats", self.default_stats())
        analysis_batch_size = max(
            int(getattr(self.args, "sil_analysis_batch_size", 1024)), 1
        )
        ref_batch = self._fastest_reference_batch(
            replay_context, max_total_steps=analysis_batch_size
        )
        if ref_batch is None:
            return

        logp_stats = self._reference_logp_stats(agent, ref_batch)
        stats.update(logp_stats)
        if (
            getattr(self.args, "sil_actual_direction_diagnostics", False)
            and global_update_iter % interval == 0
        ):
            memory = replay_context["tsil_memory"]
            valid = memory._archive_fastest_idx >= 0
            env_ids = torch.arange(memory._archive_num_envs, device=self.device)[
                :, None
            ].expand_as(valid)[valid]
            slots = memory._archive_fastest_idx[valid]
            times = memory._archive_eps_time[env_ids, slots]
            payload = dict(
                iteration=int(global_update_iter),
                steps=int(replay_context.get("global_step", 0)),
                buffer_time_mean=float(times.mean()),
                buffer_trajectory_count=int(times.numel()),
                memory_nll=logp_stats["sil_revisit_nll_topk_mean"],
                positive_gap_nll=(
                    logp_stats["sil_supervised_nll_topk_mean"]
                    if logp_stats["sil_supervised_weight_frac"] > 0
                    else None
                ),
                positive_gap_fraction=logp_stats["sil_supervised_weight_frac"],
                diagnostic_only=not bool(getattr(self.args, "sil_train", True)),
            )
            with open(
                os.path.join(
                    self.args.trajectory_dir, "fast_success_diagnostics.jsonl"
                ),
                "a",
            ) as stream:
                stream.write(json.dumps(payload) + "\n")
        if bool(getattr(self.args, "outcome_diagnostics", False)):
            current_success = float(
                replay_context.get("current_success_rate", 0.0) or 0.0
            )
            threshold = float(getattr(self.args, "fixed_success_anchor_threshold", 0.8))
            if self._fixed_success_reference is None and current_success >= threshold:
                self._fixed_success_reference = {
                    key: (
                        value.detach().clone()
                        if isinstance(value, torch.Tensor)
                        else value
                    )
                    for key, value in ref_batch.items()
                }
                self._fixed_success_reference_step = int(
                    replay_context.get("global_step", 0) or 0
                )
                self._fixed_success_reference_success = current_success
                initial_stats = self._fixed_reference_stats(
                    agent,
                    compute_gradients=False,
                )
                self._fixed_success_reference_initial_nll = initial_stats.get(
                    "sil_fixed_anchor_nll_mean",
                    0.0,
                )
            if self._fixed_success_reference is not None:
                current_ref = self._refresh_fixed_reference(replay_context)
                interval = max(int(getattr(self.args, "sil_analysis_interval", 0)), 1)
                fixed_stats = self._fixed_reference_stats(
                    agent,
                    compute_gradients=(
                        int(global_update_iter) > 0
                        and int(global_update_iter) % interval == 0
                    ),
                )
                stats.update(fixed_stats)
                if current_ref is not None:
                    current_stats = self._fixed_reference_stats(
                        agent, ref_batch=current_ref
                    )
                    stats.update(
                        {
                            key.replace(
                                "sil_fixed_anchor_", "sil_relabelled_anchor_"
                            ): value
                            for key, value in current_stats.items()
                        }
                    )
        fastest1_batch = self._fastest_reference_batch(
            replay_context,
            max_total_steps=analysis_batch_size,
            best_per_env_only=True,
        )
        if fastest1_batch is not None:
            stats.update(
                self._reference_logp_stats(agent, fastest1_batch, key_suffix="fastest1")
            )

        if (
            getattr(self.args, "sil_actual_direction_diagnostics", False)
            and getattr(self.args, "sil_direction_timing", "pre_update") == "pre_update"
        ):
            return
        interval = int(getattr(self.args, "sil_analysis_interval", 0))
        if (
            interval <= 0
            or int(global_update_iter) <= 0
            or int(global_update_iter) % interval != 0
        ):
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

    def _refresh_fixed_reference(self, replay_context):
        """Keep physical probes fixed and put both targets in today's units.

        The fixed-label probe uses the raw target at capture, isolating critic
        drift. The relabelled probe uses today's deadline in observations and
        temporal success bonus, matching how the archive would be replayed.
        These are successful episodes, so their timeout-bootstrap term is zero.
        """
        ref = self._fixed_success_reference
        if ref is None or "raw_returns" not in ref:
            return None
        memory = replay_context["tsil_memory"]
        normalizer = replay_context.get("reward_normalizer")
        normalize = lambda x: (
            normalizer(x, ref["task_ids"]) if self.args.norm_rew else x
        )
        ref["returns"] = normalize(ref["raw_returns"]).to(dtype=self.dtype)
        schedule = replay_context.get("schedule_context")
        current = dict(ref)
        if schedule is not None:
            env_ids, task_ids = ref["trajectory_env_ids"], ref["trajectory_task_ids"]
            deadline = memory._deadline_for_batch(env_ids, task_ids, schedule)
            current["obs"] = ref["obs"].clone()
            current["state"] = ref["state"].clone()
            transition_deadline = deadline[ref["traj_ids"]]
            memory._relabel_public_deadline(current["obs"], transition_deadline)
            memory._relabel_public_deadline(current["state"], transition_deadline)
            rewards = ref["raw_base_rewards"].clone()
            scale = float(schedule.get("r_epstime_scale", 0.0))
            if scale > 0:
                times = ref["trajectory_eps_times"]
                if schedule.get("ratio_range") is None:
                    bonus = torch.ones_like(times)
                    efficiency_bonus = schedule.get("temporal_efficiency_bonus")
                    if efficiency_bonus is None:
                        efficiency_bonus = schedule.get("update_ddl", False)
                    if efficiency_bonus:
                        bonus += (
                            deadline
                            / times.clamp_min(float(schedule.get("ctrl_dt", 1e-8)))
                        ).clamp(0, 1)
                else:
                    bonus = -(deadline - times).abs().clamp(max=5.0)
                rewards[
                    torch.arange(len(times), device=self.device),
                    ref["trajectory_lengths"] - 1,
                ] += (
                    scale * bonus
                )
            raw_returns = memory._discounted_returns_from_rewards(
                rewards, ref["trajectory_lengths"]
            )
            current["raw_returns"] = raw_returns[ref["traj_ids"], ref["step_indices"]]
            current["returns"] = normalize(current["raw_returns"]).to(dtype=self.dtype)
        if not self._fixed_success_reference_saved and getattr(
            self.args, "instance_dir", None
        ):
            path = os.path.join(self.args.instance_dir, "fixed_success_probes.pt")
            torch.save(
                {
                    key: value.cpu() if isinstance(value, torch.Tensor) else value
                    for key, value in ref.items()
                },
                path,
            )
            self._fixed_success_reference_saved = True
        return current

    def _fixed_reference_stats(self, agent, compute_gradients=False, ref_batch=None):
        ref_batch = self._fixed_success_reference if ref_batch is None else ref_batch
        if ref_batch is None:
            return {}

        with torch.no_grad():
            logprob = (
                agent.get_logprob(
                    ref_batch["obs"],
                    ref_batch["action"],
                    update=False,
                )
                .detach()
                .float()
                .view(-1)
            )
            value, _, _ = agent.get_value(
                ref_batch["state"],
                denorm=False,
                update=False,
            )
            value = value.detach().float().view(-1)
            returns = ref_batch["returns"].detach().float().view(-1)
            if self.args.norm_value:
                normalize_task_ids = (
                    ref_batch["task_ids"] if self.args.pertask_norm else None
                )
                returns = (
                    agent.normalize_value(
                        returns,
                        denorm=False,
                        update=False,
                        task_ids=normalize_task_ids,
                    )
                    .detach()
                    .float()
                    .view(-1)
                )
            raw_gap = returns - value
            gate_weights = torch.clamp(raw_gap, min=0.0)
            gate_weights = self._normalize_actor_gap_weights(gate_weights)
            all_fast_success = torch.ones_like(gate_weights, dtype=torch.bool)
            no_return_fill = torch.zeros_like(gate_weights, dtype=torch.bool)
            retained_weights, _, _, _ = self._apply_fast_success_retention(
                gate_weights,
                raw_gap,
                all_fast_success,
                no_return_fill,
            )
            success_floor = float(getattr(self.args, "sil_success_actor_floor", 0.0))
            effective_weights = (
                torch.maximum(
                    retained_weights, torch.full_like(retained_weights, success_floor)
                )
                if success_floor > 0.0
                else retained_weights
            )

        nll_mean = float((-logprob).mean().item())
        initial_nll = self._fixed_success_reference_initial_nll
        stats = {
            "sil_fixed_anchor_active": 1,
            "sil_fixed_anchor_capture_step": int(
                self._fixed_success_reference_step or 0
            ),
            "sil_fixed_anchor_capture_success": float(
                self._fixed_success_reference_success or 0.0
            ),
            "sil_fixed_anchor_transition_count": int(logprob.numel()),
            "sil_fixed_anchor_return_mean": float(returns.mean().item()),
            "sil_fixed_anchor_raw_reward_return_mean": float(
                ref_batch.get("raw_returns", ref_batch["returns"]).mean().item()
            ),
            "sil_fixed_anchor_reward_units_refreshed": int("raw_returns" in ref_batch),
            "sil_fixed_anchor_value_mean": float(value.mean().item()),
            "sil_fixed_anchor_raw_gap_mean": float(raw_gap.mean().item()),
            "sil_fixed_anchor_positive_frac": float(
                (raw_gap > 0).float().mean().item()
            ),
            "sil_fixed_anchor_weight_mean": float(gate_weights.mean().item()),
            "sil_fixed_anchor_effective_weight_mean": float(
                effective_weights.mean().item()
            ),
            "sil_fixed_anchor_nll_mean": nll_mean,
            "sil_fixed_anchor_nll_delta": (
                0.0 if initial_nll is None else nll_mean - float(initial_nll)
            ),
            "sil_fixed_anchor_gated_grad_norm": 0.0,
            "sil_fixed_anchor_effective_grad_norm": 0.0,
        }
        if not compute_gradients:
            return stats

        params = self._trainable_params(agent)
        if not params:
            return stats
        gated_logprob = agent.get_logprob(
            ref_batch["obs"],
            ref_batch["action"],
            update=False,
        ).view(-1)
        gated_loss = -(gate_weights.detach() * gated_logprob).mean()
        gated_grad = self._grad_vector(gated_loss, params)
        if gated_grad is not None:
            stats["sil_fixed_anchor_gated_grad_norm"] = float(gated_grad.norm().item())

        effective_logprob = agent.get_logprob(
            ref_batch["obs"],
            ref_batch["action"],
            update=False,
        ).view(-1)
        effective_loss = -(effective_weights.detach() * effective_logprob).mean()
        effective_grad = self._grad_vector(effective_loss, params)
        if effective_grad is not None:
            stats["sil_fixed_anchor_effective_grad_norm"] = float(
                effective_grad.norm().item()
            )
        return stats

    def _fastest_reference_batch(
        self,
        replay_context,
        max_total_steps=None,
        best_per_env_only=False,
        env_subset=None,
    ):
        global_per = replay_context.get("global_per")
        if global_per is not None:
            return self._global_per_reference_batch(
                replay_context,
                max_total_steps=max_total_steps,
                best_per_env_only=best_per_env_only,
            )
        memory = replay_context.get("tsil_memory")
        if memory is None or not getattr(memory, "_archive_allocated", False):
            return None

        fastest_idx = memory._archive_fastest_idx
        if best_per_env_only:
            fastest_idx = fastest_idx[:, :1]
        env_grid = (
            torch.arange(memory._archive_num_envs, device=self.device)
            .unsqueeze(1)
            .expand_as(fastest_idx)
        )
        if env_subset is not None:
            fastest_idx = fastest_idx[env_subset]
            env_grid = env_grid[env_subset]
        valid = fastest_idx >= 0
        if not bool(valid.any().item()):
            return None

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
            select_idx = (
                torch.linspace(
                    0, int(lengths.numel()) - 1, step_budget, device=self.device
                )
                .round()
                .to(torch.long)
            )
            env_ids = env_ids[select_idx]
            slot_ids = slot_ids[select_idx]
            lengths = lengths[select_idx]
        ref_count = int(lengths.numel())
        per_traj_limit = None
        if step_budget is not None:
            per_traj_limit = max(step_budget // max(ref_count, 1), 1)

        schedule_context = replay_context.get("schedule_context")
        reward_normalizer = replay_context.get("reward_normalizer")
        # Vectorize across trajectories: the previous reference path launched
        # a 150-step return recursion separately for every archived trajectory.
        # This uses the same deterministic, evenly spaced per-trajectory probes.
        counts = (
            lengths if per_traj_limit is None else lengths.clamp(max=per_traj_limit)
        )
        width = int(counts.max().item())
        columns = (
            torch.arange(width, device=self.device).unsqueeze(0).expand(ref_count, -1)
        )
        valid_steps = columns < counts.unsqueeze(1)
        step_matrix = torch.round(
            columns.float()
            * (lengths - 1).unsqueeze(1).float()
            / (counts - 1).clamp_min(1).unsqueeze(1).float()
        ).long()
        trajectory_rows = (
            torch.arange(ref_count, device=self.device)
            .unsqueeze(1)
            .expand_as(columns)[valid_steps]
        )
        steps = step_matrix[valid_steps]
        selected_envs, selected_slots = (
            env_ids[trajectory_rows],
            slot_ids[trajectory_rows],
        )
        task_ids = memory._archive_task_id[env_ids, slot_ids]
        obs = (
            memory._archive_obs[selected_envs, selected_slots, steps]
            .to(dtype=self.dtype)
            .clone()
        )
        state = (
            memory._archive_state[selected_envs, selected_slots, steps]
            .to(dtype=self.dtype)
            .clone()
        )
        original_obs, original_state = obs.clone(), state.clone()
        if schedule_context is not None:
            deadline = memory._deadline_for_batch(env_ids, task_ids, schedule_context)[
                trajectory_rows
            ]
            memory._relabel_public_deadline(obs, deadline)
            memory._relabel_public_deadline(state, deadline)
        base_rewards = memory._archive_base_reward[env_ids, slot_ids].clone()
        reward_seq = base_rewards.clone()
        if (
            schedule_context is not None
            and float(schedule_context.get("r_epstime_scale", 0.0)) > 0.0
        ):
            reward_seq[
                torch.arange(ref_count, device=self.device), lengths - 1
            ] += memory._timeaware_bonus_for_batch(
                env_ids, slot_ids, task_ids, schedule_context
            )
        raw_returns = memory._discounted_returns_from_rewards(reward_seq, lengths)
        if self.args.norm_rew and reward_normalizer is not None:
            reward_seq = (
                reward_normalizer(
                    reward_seq.reshape(-1),
                    task_ids.unsqueeze(1).expand_as(reward_seq).reshape(-1),
                )
                .view_as(reward_seq)
                .to(dtype=torch.float32, device=self.device)
            )
        returns = memory._discounted_returns_from_rewards(reward_seq, lengths)
        summary = memory.fastest_reference_summary()
        return {
            "obs": obs,
            "state": state,
            "action": memory._archive_action[selected_envs, selected_slots, steps].to(
                dtype=self.dtype
            ),
            "returns": returns[trajectory_rows, steps].to(dtype=self.dtype),
            "task_ids": task_ids[trajectory_rows],
            "traj_ids": trajectory_rows,
            "raw_returns": raw_returns[trajectory_rows, steps],
            "raw_base_rewards": base_rewards,
            "step_indices": steps,
            "trajectory_env_ids": env_ids.clone(),
            "trajectory_slot_ids_at_capture": slot_ids.clone(),
            "trajectory_lengths": lengths.clone(),
            "trajectory_task_ids": task_ids.clone(),
            "trajectory_eps_times": memory._archive_eps_time[env_ids, slot_ids].clone(),
            "original_obs": original_obs,
            "original_state": original_state,
            "ref_count": ref_count,
            "best_eps_time": summary.get("best_eps_time"),
            "best_steps": summary.get("best_steps"),
        }

    def _global_per_reference_batch(
        self, replay_context, max_total_steps=None, best_per_env_only=False
    ):
        """Build a deterministic trajectory-balanced probe from global SIL PER."""
        pool = replay_context.get("global_per")
        if pool is None or int(pool.size) <= 0 or not pool.data:
            return None
        size = int(pool.size)
        success = pool.data["success"][:size]
        eligible = success if bool(success.any().item()) else torch.ones_like(success)
        indices = torch.nonzero(eligible, as_tuple=False).view(-1)
        if not int(indices.numel()):
            return None

        if best_per_env_only:
            chosen_slots = []
            env_ids = pool.data["env_ids"][:size]
            slot_ids = pool.data["slot_ids"][:size]
            times = pool.data["episode_times"][:size]
            for env_id in env_ids[indices].unique(sorted=True):
                env_mask = eligible & (env_ids == env_id)
                env_indices = torch.nonzero(env_mask, as_tuple=False).view(-1)
                env_slots = slot_ids[env_indices].unique(sorted=True)
                best_slot, best_time = None, None
                for slot_id in env_slots:
                    slot_mask = env_mask & (slot_ids == slot_id)
                    slot_time = float(times[slot_mask][0].item())
                    if best_time is None or slot_time < best_time:
                        best_slot, best_time = slot_id, slot_time
                if best_slot is not None:
                    chosen_slots.append(best_slot)
            if chosen_slots:
                chosen = torch.stack(chosen_slots)
                eligible = eligible & (
                    slot_ids.unsqueeze(1) == chosen.unsqueeze(0)
                ).any(dim=1)
                indices = torch.nonzero(eligible, as_tuple=False).view(-1)

        step_budget = None if max_total_steps is None else max(int(max_total_steps), 1)
        if step_budget is not None and int(indices.numel()) > step_budget:
            positions = (
                torch.linspace(
                    0, int(indices.numel()) - 1, step_budget, device=self.device
                )
                .round()
                .to(torch.long)
            )
            indices = indices[positions]

        slot_ids = pool.data["slot_ids"][indices]
        _, traj_ids = torch.unique(slot_ids, sorted=True, return_inverse=True)
        task_ids = pool.data["task_ids"][indices]
        raw_returns = pool.data["raw_returns"][indices].float()
        returns = raw_returns
        reward_normalizer = replay_context.get("reward_normalizer")
        if self.args.norm_rew and reward_normalizer is not None:
            returns = reward_normalizer(raw_returns, task_ids).float()
        episode_times = pool.data["episode_times"][indices]
        return {
            "obs": pool.data["obs"][indices].to(dtype=self.dtype),
            "state": pool.data["state"][indices].to(dtype=self.dtype),
            "action": pool.data["action"][indices].to(dtype=self.dtype),
            "returns": returns.to(dtype=self.dtype),
            "task_ids": task_ids,
            "traj_ids": traj_ids,
            "raw_returns": raw_returns,
            "ref_count": int(traj_ids.max().item()) + 1,
            "best_eps_time": float(episode_times.min().item()),
            "best_steps": int(pool.data["time_idx"][indices].max().item()) + 1,
            "reference_kind": "Global PER",
        }

    def _reference_objective(self, agent, ref_batch):
        logprob = agent.get_logprob(
            ref_batch["obs"], ref_batch["action"], update=False
        ).view(-1)
        traj_ids = ref_batch["traj_ids"]
        ref_count = int(ref_batch["ref_count"])
        sums = torch.zeros(
            (ref_count,), dtype=logprob.dtype, device=logprob.device
        ).scatter_add(0, traj_ids, logprob)
        counts = torch.zeros(
            (ref_count,), dtype=logprob.dtype, device=logprob.device
        ).scatter_add(
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
                    )
                    .detach()
                    .float()
                    .view(-1)
                )
        logprob = torch.cat(logps, dim=0)
        traj_ids = ref_batch["traj_ids"]
        ref_count = int(ref_batch["ref_count"])
        sums = torch.zeros(
            (ref_count,), dtype=logprob.dtype, device=logprob.device
        ).scatter_add(0, traj_ids, logprob)
        counts = torch.zeros(
            (ref_count,), dtype=logprob.dtype, device=logprob.device
        ).scatter_add(
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
                value, _, _ = agent.get_value(
                    ref_batch["state"], denorm=False, update=False
                )
                returns = ref_batch["returns"]
                if self.args.norm_value:
                    normalize_task_ids = (
                        ref_batch["task_ids"] if self.args.pertask_norm else None
                    )
                    returns = agent.normalize_value(
                        returns,
                        denorm=False,
                        update=False,
                        task_ids=normalize_task_ids,
                    )
                weights = torch.clamp(
                    returns.to(dtype=logprob.dtype) - value.detach().float().view(-1),
                    min=0.0,
                )
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
                "sil_revisit_nll_topk_std": (
                    float(traj_nll.std(unbiased=False).item()) if ref_count > 1 else 0.0
                ),
                "sil_supervised_logp_topk_mean": float(supervised_logp.item()),
                "sil_supervised_nll_topk_mean": float((-supervised_logp).item()),
                "sil_supervised_weight_frac": supervised_weight_frac,
                "sil_archive_best_eps_time": (
                    0.0 if best_eps_time is None else float(best_eps_time)
                ),
                "sil_archive_best_steps": 0 if best_steps is None else int(best_steps),
            }
        return {
            f"sil_revisit_reference_count_{key_suffix}": ref_count,
            f"sil_revisit_logp_{key_suffix}_mean": float(traj_logp.mean().item()),
            f"sil_revisit_nll_{key_suffix}_mean": float(traj_nll.mean().item()),
            f"sil_revisit_nll_{key_suffix}_std": (
                float(traj_nll.std(unbiased=False).item()) if ref_count > 1 else 0.0
            ),
            f"sil_supervised_logp_{key_suffix}_mean": float(supervised_logp.item()),
            f"sil_supervised_nll_{key_suffix}_mean": float((-supervised_logp).item()),
            f"sil_supervised_weight_{key_suffix}_frac": supervised_weight_frac,
        }

    @staticmethod
    def _trainable_params(agent):
        return [param for param in agent.parameters() if param.requires_grad]

    @staticmethod
    def _grad_vector(objective, params, retain_graph=False):
        grads = torch.autograd.grad(
            objective, params, allow_unused=True, retain_graph=retain_graph
        )
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
        batch_size = min(
            max(int(getattr(self.args, "sil_analysis_batch_size", 1024)), 1), n_items
        )
        indices = torch.arange(
            batch_size, dtype=torch.long, device=rollout_batch["actions"].device
        )
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
        probe=None,
        metadata=None,
    ):
        params = self._trainable_params(agent)
        if not params:
            return

        ppo_dir = (
            probe["ppo"]
            if probe is not None
            else self._policy_update_direction(agent, rollout_batch, updater)
        )
        memory_dir = self._memory_update_direction(agent, ref_batch)
        if ppo_dir is None or memory_dir is None:
            return

        replay_dir = probe["replay"] if probe is not None else memory_dir
        if probe is None and getattr(
            self.args, "sil_actual_direction_diagnostics", False
        ):
            replay_dir = torch.zeros_like(ppo_dir)
            batch = replay_context.get("diagnostic_replay_batch")
            if bool(getattr(self.args, "sil_train", True)) and batch is not None:
                policy_loss, _, _, _ = self._compute_batch_losses(
                    agent,
                    batch["obs"],
                    batch["state"],
                    batch["action"],
                    batch["returns"],
                    batch["task_ids"],
                    time_target_batch=batch.get("time_targets"),
                    episode_time_batch=batch.get("episode_times"),
                    sil_weight_batch=batch.get("sil_weights"),
                    importance_weight_batch=batch.get("sil_importance_weights"),
                    frozen_positive_gap_batch=batch.get("sil_frozen_positive_gap"),
                    frozen_actor_weight_batch=batch.get("sil_frozen_actor_weights"),
                    sil_success_mask=batch.get("sil_success_mask"),
                    sil_return_fill_mask=batch.get("sil_return_fill_mask"),
                    bootstrap_value_batch=batch.get("sil_bootstrap_values"),
                    bootstrap_mask_batch=batch.get("sil_bootstrap_mask"),
                    observed_return_batch=batch.get("observed_returns"),
                    env_id_batch=batch.get("env_ids"),
                    return_stats=True,
                )
                if policy_loss.requires_grad:
                    replay_dir = -self._grad_vector(policy_loss, params).detach()
        joint_dir = ppo_dir + float(getattr(self.args, "sil_coef", 0.0)) * replay_dir
        stats["sil_cos_ppo_memory"] = self._cosine(ppo_dir, memory_dir)
        stats["sil_cos_sil_memory"] = self._cosine(replay_dir, memory_dir)
        stats["sil_cos_joint_memory"] = self._cosine(joint_dir, memory_dir)
        stats["sil_alignment_gain"] = (
            stats["sil_cos_joint_memory"] - stats["sil_cos_ppo_memory"]
        )

        def unavailable(reason):
            folder = getattr(self.args, "trajectory_dir", None)
            if probe is not None and folder:
                os.makedirs(folder, exist_ok=True)
                path = os.path.join(
                    folder, f"sil_direction_landscape_iter{global_update_iter:06d}.json"
                )
                with open(path, "w") as stream:
                    json.dump(
                        dict(
                            iteration=int(global_update_iter),
                            available=False,
                            unavailable_reason=reason,
                            ppo_norm=float(ppo_dir.norm()),
                            replay_norm=float(replay_dir.norm()),
                            **(metadata or {}),
                        ),
                        stream,
                    )

        bx = self._unit_vector(ppo_dir)
        if bx is None:
            unavailable("zero_ppo_direction")
            return
        by_raw = memory_dir - torch.dot(memory_dir, bx) * bx
        by = self._unit_vector(by_raw)
        if by is None and probe is not None:
            unavailable("degenerate_reference_axis")
            return
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
                        vector_to_parameters(
                            base_vector + x_val * bx + y_val * by, params
                        )
                        row.append(
                            float(
                                self._reference_objective(agent, ref_batch)
                                .detach()
                                .item()
                            )
                        )
                    surface.append(row)
        finally:
            vector_to_parameters(base_vector, params)

        trajectory_dir = getattr(self.args, "trajectory_dir", None)
        if not trajectory_dir:
            return
        os.makedirs(trajectory_dir, exist_ok=True)
        path = os.path.join(
            trajectory_dir, f"sil_direction_landscape_iter{global_update_iter:06d}.json"
        )
        reference_kind = str(ref_batch.get("reference_kind", "Fast-success"))
        with open(path, "w") as file_obj:
            json.dump(
                {
                    "iteration": int(global_update_iter),
                    "available": True,
                    "x": [float(v.item()) for v in xs],
                    "y": [float(v.item()) for v in ys],
                    "surface": surface,
                    "origin": [0.0, 0.0],
                    "ppo_update": project_arrow(ppo_dir),
                    "ppo_sil_update": project_arrow(joint_dir),
                    "replay_update": project_arrow(replay_dir),
                    "direction_semantics": (
                        "post_update_loss_probe_on_actual_replay_batch"
                        if getattr(self.args, "sil_actual_direction_diagnostics", False)
                        else "legacy_reference_gradient"
                    ),
                    "optimizer_update": project_arrow(
                        replay_context.get(
                            "optimizer_displacement", torch.zeros_like(ppo_dir)
                        )
                    ),
                    "ppo_norm": float(ppo_dir.norm()),
                    "replay_norm": float(replay_dir.norm()),
                    "joint_norm": float(joint_dir.norm()),
                    "scaled_replay_to_ppo_norm_ratio": float(
                        getattr(self.args, "sil_coef", 0.0)
                    )
                    * float(replay_dir.norm())
                    / max(float(ppo_dir.norm()), 1e-30),
                    "cos_ppo_replay": self._cosine(ppo_dir, replay_dir),
                    "cos_replay_reference": self._cosine(replay_dir, memory_dir),
                    "xlabel": "PPO update axis",
                    "ylabel": "Fast-success reference axis orthogonal to PPO",
                    "zlabel": f"{reference_kind} memory log-prob",
                    "title": f"SIL {reference_kind.lower()} memory landscape @ iter {global_update_iter}",
                    **(metadata or {}),
                },
                file_obj,
            )
        stats["sil_landscape_file"] = path
        if probe is not None:
            replay_context["_pre_landscape_basis"] = (bx.detach(), by.detach(), span)
            replay_context["_pre_landscape_path"] = path


__all__ = ["TsilReplayLoss"]
