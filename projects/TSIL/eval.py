"""TSIL evaluation entrypoint."""

import json
import hashlib
import os
from pathlib import Path

# Isaac Gym must be imported by the shared evaluator before torch.
from core.evaluation.evaluator import BaseEvaluator, run_evaluation
import torch
from projects.TSIL.benchmarks.metaworld_targets import (
    apply_single_training_configuration,
    apply_targets_to_fixed_env,
    load_target_bank,
    sidecar_path,
)
from projects.TSIL.reset_cadence_audit import install_reset_cadence_diagnostic


class TSILEvaluator(BaseEvaluator):
    """Apply checkpoint-paired deadlines to held-out Meta-World resets."""

    project_name = "TSIL"

    def setup_project_evaluation(self):
        self.target_assignment = None
        self.reset_cadence_diagnostic = None
        if os.environ.get("TSIL_EVAL_TRAINING_RESET_CADENCE", "0") == "1":
            if getattr(self.args, "benchmark_backend", "metaworld") != "metaworld":
                raise ValueError("The reset-cadence diagnostic is Meta-World-only")
            self.reset_cadence_diagnostic = install_reset_cadence_diagnostic(self.envs)
        if getattr(self.args, "benchmark_backend", "metaworld") != "metaworld":
            return
        if getattr(self.args, "fix_priv", False) or not getattr(self.args, "time2end", True):
            self.target_assignment = {"status": "masked_non_temporal_method"}
            return
        self.target_debug_mode = os.environ.get('TSIL_EVAL_TARGET_MODE', 'knn5')
        if self.target_debug_mode not in {
            'knn5', 'horizon', 'constant', 'training_exact', 'training_knn5',
            'training_single',
        }:
            raise ValueError('Unknown evaluation target debug mode')
        if self.target_debug_mode in {'horizon', 'constant'}:
            self.target_assignment = {'status': self.target_debug_mode}
            return
        path = sidecar_path(self.args.checkpoint_path)
        if not path.is_file():
            if getattr(self.args, "metaworld_checkpoint_target_bank", False):
                raise FileNotFoundError(f"Expected checkpoint target bank: {path}")
            self.target_assignment = {"status": "legacy_full_horizon_missing_sidecar"}
            return
        self.target_bank = load_target_bank(self.args.checkpoint_path)
        if self.target_bank["checkpoint"] != Path(self.args.checkpoint_path).name:
            raise ValueError("Meta-World target bank belongs to a different checkpoint")
        if abs(float(self.target_bank["max_episode_seconds"]) - float(self.envs.max_eps_time)) > 1e-6:
            raise ValueError("Meta-World target bank and evaluation horizon differ")
        if not getattr(self.envs, "fixed", getattr(self.args, "fixed", False)):
            raise ValueError("Checkpoint KNN targets require fixed evaluation configurations")

    def before_evaluation_reset(self, goal_speed):
        mode = getattr(self, 'target_debug_mode', 'knn5')
        if mode in {'horizon', 'constant'}:
            value = float(self.envs.max_eps_time) if mode == 'horizon' else float(os.environ['TSIL_EVAL_CONSTANT_TARGET'])
            if not 0 < value <= self.envs.max_eps_time:
                raise ValueError('Evaluation target must be within the episode horizon')
            self.envs.obs_range['time2end'][0].fill_(-value)
            self.envs.obs_range['time2end'][1].fill_(value)
            self.envs.reset_all()
            self.target_assignment = dict(status=mode, assigned_deadline_mean=value)
            return
        if not hasattr(self, "target_bank"):
            return
        if mode == 'training_single':
            offset = int(os.environ.get('TSIL_EVAL_TRAINING_CONFIG_INDEX', '0'))
            assigned, selected = apply_single_training_configuration(
                self.envs, self.target_bank, source_offset=offset,
            )
            self.target_assignment = dict(
                status=mode,
                requested_source_offset=offset,
                assigned_deadline_mean=float(assigned.mean()),
                selected=selected,
            )
            return
        if mode.startswith('training_'):
            ids = torch.tensor(self.target_bank['task_ids'])
            features = torch.tensor(self.target_bank['features'])
            deadlines = torch.tensor(self.target_bank['deadline_seconds'])
            assigned = torch.empty(self.envs.num_envs, device=self.envs.device)
            for task in torch.unique(self.envs.task_indices):
                destination = (self.envs.task_indices == task).nonzero().flatten()
                source = (ids == int(task)).nonzero().flatten()
                take = source[torch.arange(destination.numel()) % source.numel()]
                self.envs.last_rand_vecs[destination] = features[take].to(self.envs.device)
                assigned[destination] = deadlines[take].to(self.envs.device)
            if mode == 'training_exact':
                self.envs.obs_range['time2end'][0].copy_(-assigned)
                self.envs.obs_range['time2end'][1].copy_(assigned)
                self.envs.reset_all()
                self.target_assignment = dict(status=mode, assigned_deadline_mean=float(assigned.mean()))
                return
        env = self.envs
        targets, diagnostics = apply_targets_to_fixed_env(env, self.target_bank)
        source_counts = diagnostics["source_counts"]
        neighbor_counts = diagnostics["neighbor_counts"]
        self.target_assignment = {
            "status": "checkpoint_knn5" if mode == "knn5" else mode,
            "checkpoint_target_bank": str(sidecar_path(self.args.checkpoint_path)),
            "bank_sha256": self.target_bank["sha256"],
            "evaluation_configurations": int(targets.numel()),
            "fallback_count": diagnostics["fallback_count"],
            "learned_source_min": int(source_counts.min().item()),
            "learned_source_max": int(source_counts.max().item()),
            "neighbors_min": int(neighbor_counts.min().item()),
            "neighbors_max": int(neighbor_counts.max().item()),
            "assigned_deadline_mean": float(targets.mean().item()),
            "assigned_deadline_min": float(targets.min().item()),
            "assigned_deadline_max": float(targets.max().item()),
            "goal_speed": float(goal_speed),
        }

    @staticmethod
    def _tensor_summary(value, include_first=False):
        value = value.detach()
        flat = value.reshape(-1)
        digest = hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest()
        result = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "all_finite": bool(torch.isfinite(value).all().item()),
            "min": float(flat.min().item()),
            "max": float(flat.max().item()),
            "mean": float(flat.float().mean().item()),
            "std": float(flat.float().std(unbiased=False).item()),
            "sha256": digest,
        }
        if include_first:
            result["first"] = value[0].detach().cpu().tolist()
        return result

    @staticmethod
    def _normalizer_hash(normalizer):
        digest = hashlib.sha256()
        for name, value in sorted(normalizer.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def after_evaluation_reset(self, goal_speed, phase, next_obs, next_state):
        if os.environ.get("TSIL_EVAL_OBSERVATION_AUDIT", "0") != "1":
            return
        if not self.args.saving:
            raise ValueError("TSIL_EVAL_OBSERVATION_AUDIT requires --saving")

        normalizer = getattr(self.agent, "obs_normalizer", None)
        norm_hash_before = self._normalizer_hash(normalizer) if normalizer is not None else None
        normalized = self.agent.preprocess_obs(next_obs, update=False)
        norm_hash_after = self._normalizer_hash(normalizer) if normalizer is not None else None
        record = {
            "phase": phase,
            "goal_speed": float(goal_speed),
            "agent_training": bool(self.agent.training),
            "agent_deterministic": bool(self.args.deterministic),
            "raw_observation": self._tensor_summary(next_obs, include_first=True),
            "normalized_observation": self._tensor_summary(normalized, include_first=True),
            "raw_state": self._tensor_summary(next_state, include_first=True),
            "normalizer": {
                "present": normalizer is not None,
                "state_sha256_before": norm_hash_before,
                "state_sha256_after": norm_hash_after,
                "unchanged_by_preprocessing": norm_hash_before == norm_hash_after,
            },
        }
        if normalizer is not None:
            record["normalizer"].update({
                "running_mean": self._tensor_summary(normalizer.running_mean),
                "running_var": self._tensor_summary(normalizer.running_var),
                "count": normalizer.count.detach().cpu().tolist(),
            })
        for name in ("last_rand_vecs", "obj_init_pos", "target_pos"):
            value = getattr(self.envs, name, None)
            if isinstance(value, torch.Tensor):
                record[name] = self._tensor_summary(value, include_first=True)
        for name in ("obj_states", "target_pos"):
            value = getattr(self.envs, "states", {}).get(name)
            if isinstance(value, torch.Tensor):
                record[f"policy_state_{name}"] = self._tensor_summary(value, include_first=True)
        time_range = getattr(self.envs, "obs_range", {}).get("time2end")
        if time_range is not None:
            record["time2end_range"] = [
                self._tensor_summary(value, include_first=True) for value in time_range
            ]

        path = Path(self.args.trajectory_dir) / "observation_audit.json"
        payload = json.loads(path.read_text()) if path.is_file() else []
        payload.append(record)
        path.write_text(json.dumps(payload, indent=2) + "\n")
        torch.save({
            "raw_observation": next_obs.detach().cpu(),
            "normalized_observation": normalized.detach().cpu(),
            "raw_state": next_state.detach().cpu(),
        }, Path(self.args.trajectory_dir) / f"observation_audit_{phase}.pt")

    def after_eval_step(self, action, **kwargs):
        if os.environ.get("TSIL_EVAL_OBSERVATION_AUDIT", "0") != "1":
            return
        path = Path(self.args.trajectory_dir) / "first_action_audit.json"
        if path.exists():
            return
        path.write_text(json.dumps({
            "evaluation_action_seed": getattr(self.args, "evaluation_action_seed", None),
            "action": self._tensor_summary(action, include_first=True),
        }, indent=2) + "\n")
        torch.save(
            {"action": action.detach().cpu()},
            Path(self.args.trajectory_dir) / "first_action_audit.pt",
        )

    def after_eval_results(self, goal_speed, results):
        if self.args.saving and self.target_assignment is not None:
            path = Path(self.args.trajectory_dir) / "target_assignment.json"
            path.write_text(json.dumps(self.target_assignment, indent=2) + "\n")
        if self.args.saving and self.reset_cadence_diagnostic is not None:
            path = Path(self.args.trajectory_dir) / "reset_cadence_diagnostic.json"
            path.write_text(json.dumps({
                "mode": "training_reset_cadence",
                "description": (
                    "Supply one scene-wide reset-style physics refresh on policy steps "
                    "where no lane was already pending reset."
                ),
                **self.reset_cadence_diagnostic,
            }, indent=2) + "\n")


def main():
    import sys
    if not any(x == "--index_episode" or x.startswith("--index_episode=") for x in sys.argv):
        sys.argv.extend(["--index_episode", "best_suc_tail"])
    if not any(x == "--deterministic" or x.startswith("--deterministic=") for x in sys.argv):
        sys.argv.extend(["--deterministic", "False"])
    run_evaluation(evaluator_cls=TSILEvaluator)


if __name__ == "__main__":
    main()
