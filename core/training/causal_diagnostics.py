"""Exact-rollout shadow updates for TSIL collapse diagnosis.

This module is diagnostic-only.  It replays alternative optimizer updates from
the same agent, optimizer, rollout batch, replay context, and RNG state, writes
their checkpoints and attribution tables, then restores training exactly to the
pre-diagnostic state.
"""

from contextlib import contextmanager
from copy import deepcopy
import csv
import json
import os
import random

import numpy as np
import torch


class CausalShadowDiagnostics:
    VARIANTS = (
        "ppo_only",
        "tsil_only",
        "joint",
        "ppo_mask_high_dense_failure",
        "ppo_no_dense_advantage",
    )

    def __init__(self, args):
        self.args = args
        self.enabled = bool(getattr(args, "causal_shadow_diagnostics", False))
        configured_root = getattr(args, "causal_diagnostic_dir", None)
        self.root = configured_root or os.path.join(args.trajectory_dir, "causal_diagnostics")
        self.root = os.path.join(self.root, str(args.run_name))
        self.peak_success = 0.0
        self.capture_count = 0
        self.max_captures = max(int(getattr(args, "causal_shadow_max_captures", 1)), 0)
        if self.enabled:
            os.makedirs(self.root, exist_ok=True)

    def should_capture(self, success_rate):
        success_rate = float(success_rate)
        self.peak_success = max(self.peak_success, success_rate)
        if not self.enabled or self.capture_count >= self.max_captures:
            return False
        peak_threshold = float(getattr(self.args, "causal_shadow_peak_threshold", 0.5))
        drop_threshold = float(getattr(self.args, "causal_shadow_drop_threshold", 0.1))
        return (
            self.peak_success >= peak_threshold
            and success_rate <= self.peak_success - drop_threshold
        )

    @staticmethod
    def _rng_state():
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    @staticmethod
    def _restore_rng(state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])

    @staticmethod
    def _memory_cursors(memory):
        scalar_names = (
            "_round_robin_cursor",
            "_success_round_robin_cursor",
            "_fallback_round_robin_cursor",
        )
        cursors = {name: int(getattr(memory, name, 0)) for name in scalar_names}
        for name in ("_archive_transition_cursor", "_archive_transition_offset"):
            value = getattr(memory, name, None)
            if isinstance(value, torch.Tensor):
                cursors[name] = value.clone()
        return cursors

    @staticmethod
    def _restore_memory_cursors(memory, cursors):
        for name, value in cursors.items():
            current = getattr(memory, name, None)
            if isinstance(value, torch.Tensor) and isinstance(current, torch.Tensor):
                current.copy_(value)
            else:
                setattr(memory, name, int(value))

    @contextmanager
    def _override_args(self, **updates):
        old = {key: getattr(self.args, key) for key in updates}
        try:
            for key, value in updates.items():
                setattr(self.args, key, value)
            yield
        finally:
            for key, value in old.items():
                setattr(self.args, key, value)

    @staticmethod
    def _copy_batch(batch):
        return {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    @staticmethod
    def _high_dense_failure_mask(batch):
        outcomes = batch.get("completed_episode_outcome")
        dense_returns = batch.get("completed_episode_dense_return")
        valid = batch["policy_sample_valid"]
        if outcomes is None or dense_returns is None:
            return torch.zeros_like(valid)
        failure = valid & (outcomes == 0) & torch.isfinite(dense_returns)
        if not bool(failure.any().item()):
            return torch.zeros_like(valid)
        threshold = torch.quantile(dense_returns[failure].float(), 0.9)
        return failure & (dense_returns >= threshold)

    @staticmethod
    def _per_dimension_logprob(agent, batch, chunk_size=4096):
        chunks = []
        with torch.no_grad():
            for start in range(0, int(batch["actions"].shape[0]), int(chunk_size)):
                end = min(start + int(chunk_size), int(batch["actions"].shape[0]))
                probs, _ = agent.build_distribution(batch["obs"][start:end], update=False)
                chunks.append(probs.log_prob(batch["actions"][start:end]).detach())
        return torch.cat(chunks, dim=0)

    @staticmethod
    def _group_masks(batch, high_dense_failure):
        valid = batch["policy_sample_valid"]
        outcomes = batch.get("completed_episode_outcome")
        groups = {"all": valid, "high_dense_failure": high_dense_failure}
        if outcomes is not None:
            groups["success"] = valid & (outcomes == 1)
            groups["failure"] = valid & (outcomes == 0)

        distance = batch.get("peg_target_distance")
        if distance is not None:
            finite = valid & torch.isfinite(distance)
            groups.update({
                "phase_far_ge_0p15": finite & (distance >= 0.15),
                "phase_approach_0p08_0p15": finite & (distance >= 0.08) & (distance < 0.15),
                "phase_near_0p05_0p08": finite & (distance >= 0.05) & (distance < 0.08),
                "phase_success_region_lt_0p05": finite & (distance < 0.05),
            })
        return groups

    @staticmethod
    def _safe_mean(values, mask):
        if not bool(mask.any().item()):
            return None
        return float(values[mask].detach().float().mean().item())

    def _variant_rows(self, variant, batch, pre_dim_logprob, post_dim_logprob, high_dense_failure):
        rows = []
        delta_dim = post_dim_logprob - pre_dim_logprob
        delta_joint = delta_dim.sum(dim=-1)
        groups = self._group_masks(batch, high_dense_failure)
        for group, mask in groups.items():
            count = int(mask.sum().item())
            if count == 0:
                continue
            base = {
                "variant": variant,
                "group": group,
                "count": count,
                "full_advantage_raw_mean": self._safe_mean(batch["advantages_raw"], mask),
                "full_advantage_mean": self._safe_mean(batch["advantages"], mask),
                "no_dense_advantage_raw_mean": self._safe_mean(
                    batch.get("advantages_no_dense_raw", torch.zeros_like(batch["advantages"])), mask,
                ),
                "no_dense_advantage_mean": self._safe_mean(
                    batch.get("advantages_no_dense", torch.zeros_like(batch["advantages"])), mask,
                ),
                "full_positive_advantage_frac": self._safe_mean(
                    (batch["advantages"] > 0).float(), mask,
                ),
                "joint_logprob_delta_mean": self._safe_mean(delta_joint, mask),
                "joint_logprob_increase_frac": self._safe_mean((delta_joint > 0).float(), mask),
            }
            for action_dim in range(int(delta_dim.shape[-1])):
                row = dict(base)
                row["action_dim"] = action_dim
                row["action_logprob_delta_mean"] = self._safe_mean(delta_dim[:, action_dim], mask)
                row["action_logprob_increase_frac"] = self._safe_mean(
                    (delta_dim[:, action_dim] > 0).float(), mask,
                )
                rows.append(row)
        return rows

    def _run_variant(
        self,
        variant,
        batch,
        trainer,
        initial_lstm_state,
        replay_context,
        pre_agent_state,
        pre_optimizer_state,
        pre_rng,
        pre_cursors,
        pre_replay_context,
    ):
        trainer.agent.load_state_dict(pre_agent_state)
        trainer.optimizer.load_state_dict(pre_optimizer_state)
        self._restore_rng(pre_rng)
        self._restore_memory_cursors(trainer.tsil_memory, pre_cursors)
        replay_context.clear()
        replay_context.update(pre_replay_context)

        variant_batch = self._copy_batch(batch)
        high_dense_failure = self._high_dense_failure_mask(variant_batch)
        overrides = {}
        if variant == "ppo_only":
            overrides["sil_coef"] = 0.0
        elif variant == "tsil_only":
            overrides.update({
                # The replay-off replicas are also used as exact counterfactual
                # starting states.  Explicitly enable the already-prepared
                # replay context for this shadow instead of silently producing
                # a no-update control.
                "sil_train": True,
                "warmup_iters": int(trainer.metrics.global_update_iter) + 1,
                "vf_coef": 0.0,
                "bounds_loss_coef": 0.0,
                "sil_vf_coef": 0.0,
                "sil_temporal_vf_coef": 0.0,
            })
        elif variant == "joint":
            # Preserve the configured joint update for TSIL runs and turn replay
            # on for a replay-off counterfactual.
            overrides["sil_train"] = True
        elif variant == "ppo_mask_high_dense_failure":
            overrides["sil_coef"] = 0.0
            variant_batch["policy_sample_valid"][high_dense_failure] = False
            variant_batch["advantages"][high_dense_failure] = 0.0
        elif variant == "ppo_no_dense_advantage":
            overrides["sil_coef"] = 0.0
            variant_batch["advantages"] = variant_batch["advantages_no_dense"].clone()

        with self._override_args(**overrides):
            result = trainer.policy_updater.run_minibatch_updates(
                variant_batch,
                trainer.agent,
                trainer.optimizer,
                initial_lstm_state,
                trainer.lr_scheduler,
                trainer.metrics.cur_ent,
                trainer.metrics.global_update_iter,
                global_step=trainer.metrics.global_step,
                replay_module=trainer.tsil_replay_loss,
                replay_context=replay_context,
            )
        if bool(result[0]):
            trainer.agent.load_state_dict(result[3])
            trainer.optimizer.load_state_dict(result[4])
        return variant_batch, result

    def capture(self, trainer, batch, initial_lstm_state, replay_context):
        capture_id = self.capture_count
        self.capture_count += 1
        tag = f"drop_{capture_id:02d}_step_{int(trainer.metrics.global_step):09d}"
        capture_dir = os.path.join(self.root, tag)
        os.makedirs(capture_dir, exist_ok=False)

        pre_agent_state = deepcopy(trainer.agent.state_dict())
        pre_optimizer_state = deepcopy(trainer.optimizer.state_dict())
        pre_rng = self._rng_state()
        pre_cursors = self._memory_cursors(trainer.tsil_memory)
        # Sampling only adds/replaces diagnostic and cursor metadata in the
        # context; the immutable dataset tensors are intentionally shared.
        pre_replay_context = dict(replay_context)
        pre_dim_logprob = self._per_dimension_logprob(
            trainer.agent,
            batch,
            chunk_size=getattr(self.args, "outcome_diagnostic_batch_size", 4096),
        )
        high_dense_failure = self._high_dense_failure_mask(batch)

        pre_suffix = f"shadow_{tag}_pre"
        trainer.agent.save_checkpoint(trainer.args.checkpoint_dir, suffix=pre_suffix)
        rows = self._variant_rows(
            "pre_no_update",
            batch,
            pre_dim_logprob,
            pre_dim_logprob,
            high_dense_failure,
        )
        variants = []
        for variant in self.VARIANTS:
            variant_batch, result = self._run_variant(
                variant,
                batch,
                trainer,
                initial_lstm_state,
                replay_context,
                pre_agent_state,
                pre_optimizer_state,
                pre_rng,
                pre_cursors,
                pre_replay_context,
            )
            post_dim_logprob = self._per_dimension_logprob(
                trainer.agent,
                batch,
                chunk_size=getattr(self.args, "outcome_diagnostic_batch_size", 4096),
            )
            rows.extend(self._variant_rows(
                variant,
                batch,
                pre_dim_logprob,
                post_dim_logprob,
                high_dense_failure,
            ))
            suffix = f"shadow_{tag}_{variant}"
            trainer.agent.save_checkpoint(trainer.args.checkpoint_dir, suffix=suffix)
            variants.append({
                "variant": variant,
                "checkpoint_suffix": suffix,
                "policy_diverged": bool(result[0]),
                "num_agent_updates": int(result[2]),
            })

        csv_path = os.path.join(capture_dir, "phase_action_shadow_metrics.csv")
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        manifest = {
            "capture_id": capture_id,
            "tag": tag,
            "global_step": int(trainer.metrics.global_step),
            "global_update_iter": int(trainer.metrics.global_update_iter),
            "current_success_rate": float(trainer.metrics.cur_success_rate),
            "peak_success_rate": float(self.peak_success),
            "high_dense_failure_transition_count": int(high_dense_failure.sum().item()),
            "pre_checkpoint_suffix": pre_suffix,
            "variants": variants,
            "causal_scope": (
                "All variants share the exact pre-update agent, optimizer, rollout batch, "
                "replay context, sampler cursors, and RNG state. Evaluation checkpoints "
                "must still be compared on a shared reset panel."
            ),
        }
        with open(os.path.join(capture_dir, "manifest.json"), "w") as json_file:
            json.dump(manifest, json_file, indent=2)

        trainer.agent.load_state_dict(pre_agent_state)
        trainer.optimizer.load_state_dict(pre_optimizer_state)
        self._restore_rng(pre_rng)
        self._restore_memory_cursors(trainer.tsil_memory, pre_cursors)
        replay_context.clear()
        replay_context.update(pre_replay_context)
        return manifest


__all__ = ["CausalShadowDiagnostics"]
