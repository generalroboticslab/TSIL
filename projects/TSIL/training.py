"""TSIL training with episode-local reward normalization and project policies."""

from dataclasses import dataclass
from typing import Optional
import json
from pathlib import Path

from core.training.trainer import OnPolicyTrainer
from core.training.args import Args, parse_args as parse_shared_args


@dataclass
class TSILArgs(Args):
    # Diagnostic-only live-simulator evaluation after the final checkpoint.
    inprocess_checkpoint_audit: bool = False
    inprocess_audit_action_seed: int = 20260923
    sil_diagnostic_fast_memory: bool = False
    sil_actual_direction_diagnostics: bool = False
    sil_direction_timing: str = "pre_update"
    sil_landscape_progress_interval: float = 0.0
    # Opt-in ablation; historical checkpoints retain joint actor/critic clipping.
    gradient_clip_scope: str = "joint"
    sil_normalization_audit: bool = False
    sil_freeze_replay_gap: bool = True
    sil_replay_normalization_scope: str = "slice"
    sil_landscape_reference: str = "fast_success"
    # Environment-side action replacement: PPO stores the commanded action
    # and its matching log-probability while the environment executes noise.
    action_execution_noise_p: float = 0.0
    action_execution_noise_mode: str = "uniform_replacement"
    action_execution_noise_scale: float = 1.0
    # Retained only for explicitly labelled invalid-likelihood negative controls.
    action_logprob_mismatch_p: float = 0.0
    # Historical checkpoints remain reproducible. New campaigns opt in to the
    # lane-local cached reset contract explicitly.
    metaworld_reset_mode: str = "historical_scene_step"
    sil_audit_interval: int = 0
    sil_audit_sample_size: int = 4096
    observation_ablation_mode: str = "current"
    tsil_zero_masked_temporal_inputs: bool = True
    beta: Optional[bool] = None
    fix_linvel: Optional[bool] = None
    fix_limvel: Optional[bool] = None
    d2s_reset_critic: bool = False
    sil_replay_no_fallback: bool = False
    benchmark_backend: str = "metaworld"
    minigrid_suite: str = "mt7"
    minigrid_observation: Optional[str] = None
    minigrid_reward: Optional[str] = None
    minigrid_config_bank: str = ""
    tsil_drop_constant_inputs: Optional[bool] = None
    # Explicit opt-in preserves saved/legacy configs; enable for future clean runs.
    tsil_drop_critic_velocity_inputs: Optional[bool] = None
    ppo_boundary_mode: Optional[str] = None
    minigrid_base_learner: str = 'current'
    reward_normalizer_scale_mode: Optional[str] = None
    sil_success_actor_schedule: str = 'off'
    sil_success_weight_boost: float = 5.0
    sil_success_weight_decay: float = 10.0
    sil_success_extra_mean_cap: float = 0.1
    sil_success_weight_floor: float = 0.0
    sil_aligned_success_only: bool = False


    def __post_init__(self):
        # Resolve only omitted values: explicit historical options always win.
        mini = self.benchmark_backend == 'minigrid'
        defaults = dict(beta=mini, fix_linvel=not mini, fix_limvel=not mini,
            tsil_drop_constant_inputs=not mini, tsil_drop_critic_velocity_inputs=not mini,
            ppo_boundary_mode='research' if mini else 'exclude_value',
            reward_normalizer_scale_mode='std_magnitude_floor' if mini else 'std_only',
            minigrid_observation='partial' if mini else 'full',
            minigrid_reward='native_bonus' if mini else 'success')
        for name, value in defaults.items():
            if getattr(self, name) is None:
                setattr(self, name, value)


def validate_args(args):
    # The final minibatch validates the preceding update; it is not optimized.
    if args.num_minibatches < 2:
        raise ValueError("TSIL training needs at least two minibatches: one training batch and one KL-validation batch")
    if args.gradient_clip_scope not in {"joint", "actor_critic"}:
        raise ValueError("gradient_clip_scope must be joint or actor_critic")
    if args.sil_replay_normalization_scope not in {'slice', 'dataset'}:
        raise ValueError('Replay normalization scope must be slice or dataset')
    if args.sil_replay_normalization_scope == 'dataset':
        if not args.sil_freeze_replay_gap or not args.sil_normalize_gap:
            raise ValueError('Dataset normalization requires frozen replay gaps and normalization')
    if args.sil_landscape_reference not in {'fast_success', 'positive_replay_success', 'top1_gap_ranked'}:
        raise ValueError('Unknown landscape reference')
    if args.sil_landscape_reference in {'positive_replay_success', 'top1_gap_ranked'}:
        if args.sil_direction_timing != 'pre_update' or not args.sil_actual_direction_diagnostics:
            raise ValueError('Positive replay reference requires actual pre-update diagnostics')
        if args.sil_mode != 'sil' or args.sil_filter_mode != 'return':
            raise ValueError('Positive replay reference currently requires return-gap SIL')
    if args.sil_direction_timing not in {'pre_update', 'post_update'}:
        raise ValueError('sil_direction_timing must be pre_update or post_update')
    if not 0 <= args.sil_landscape_progress_interval <= 1:
        raise ValueError('sil_landscape_progress_interval must be between zero and one')
    if args.sil_freeze_replay_gap:
        if not args.sil_aligned_replay or args.sil_replay_build_scope != 'update':
            raise ValueError('Frozen replay gaps require aligned update-scope replay')
        if args.sil_mode != 'sil' or args.sil_filter_mode != 'return' or args.sil_global_per:
            raise ValueError('Frozen replay gaps currently support trajectory return-gap SIL only')
        if (args.sil_actor_weight_coef or args.sil_fast_success_retention_coef
                or args.sil_success_actor_floor or args.sil_success_actor_min_share
                or args.sil_success_actor_schedule != 'off' or args.sil_replay_weight_target_mean):
            raise ValueError('Frozen replay gaps cannot combine with actor rescue/weight schedules')
    if args.inprocess_checkpoint_audit:
        if (args.benchmark_backend != "metaworld" or len(set(args.tasks)) != 1
                or not args.saving or not args.metaworld_checkpoint_target_bank
                or not args.fixed):
            raise ValueError(
                "In-process checkpoint audit requires one fixed Meta-World task, "
                "saving, and checkpoint target banks"
            )
    if not 0 <= args.action_execution_noise_p <= 1:
        raise ValueError("action_execution_noise_p must lie in [0, 1]")
    if not 0 <= args.action_logprob_mismatch_p <= 1:
        raise ValueError("action_logprob_mismatch_p must lie in [0, 1]")
    if args.action_execution_noise_mode not in {
            "uniform_replacement", "additive_uniform", "categorical_mixture"}:
        raise ValueError("Unknown action_execution_noise_mode")
    if not 0 <= args.action_execution_noise_scale <= 1:
        raise ValueError("action_execution_noise_scale must lie in [0, 1]")
    if args.metaworld_reset_mode not in {
            "historical_scene_step", "cached_initial_state"}:
        raise ValueError("Unknown metaworld_reset_mode")
    if (args.benchmark_backend != "metaworld"
            and args.metaworld_reset_mode != "historical_scene_step"):
        raise ValueError("metaworld_reset_mode applies only to Meta-World")
    if args.action_execution_noise_p and args.action_logprob_mismatch_p:
        raise ValueError("Valid action execution noise and likelihood mismatch are mutually exclusive")
    if (args.action_execution_noise_p or args.action_logprob_mismatch_p) and (
            args.random_policy or args.use_lstm):
        raise ValueError("Action stress requires feedforward policy training")
    if args.sil_trajectory_weight_mode not in {'none', 'episode_limit', 'adaptive_deadline'}:
        raise ValueError('Unknown TSIL trajectory weighting mode')
    if args.sil_audit_interval < 0 or args.sil_audit_sample_size <= 0:
        raise ValueError("Invalid replay audit interval or sample size")
    if args.observation_ablation_mode not in {"current", "original_full"}:
        raise ValueError("observation_ablation_mode must be current or original_full")
    if args.observation_ablation_mode == "original_full":
        if (args.tsil_drop_constant_inputs or args.tsil_drop_critic_velocity_inputs
                or args.task_embedding_dim not in {1, 50} or len(set(args.tasks)) != 1
                or args.beta or args.use_cost or args.use_timeawareness
                or args.use_lstm or args.use_grpo or args.benchmark_backend != "metaworld"
                or args.checkpoint is not None):
            raise ValueError("Original-input audit requires single-task scratch Gaussian ATTL, task_embedding_dim=1 (legacy) or 50 (preserved task label), and full inputs")

    if args.sil_audit_interval and not (args.use_sil and args.sil_train
            and args.sil_aligned_replay and args.sil_source == 'fastest'
            and args.sil_mode == 'sil' and args.sil_filter_mode == 'return'):
        raise ValueError('Replay audit requires aligned fastest-source return-filter SIL')

    if args.d2s_reset_critic and (args.benchmark_backend != 'minigrid'
            or args.dense_reward_mode != 'hard' or args.dense_reward_scale != [1., 0.]
            or args.use_sil or args.checkpoint is not None):
        raise ValueError('D2S critic reset requires fresh MiniGrid hard-switch D2S without replay')
    if args.sil_replay_no_fallback and not (args.use_sil and args.sil_train
            and args.sil_aligned_replay and args.sil_source == 'fastest'
            and args.sil_mode == 'sil' and args.sil_filter_mode == 'return'):
        raise ValueError('No-fallback requires aligned fastest-source return-filter SIL')
    if args.sil_aligned_success_only and (not args.use_sil or not args.sil_aligned_replay
            or args.sil_source != 'fastest' or args.sil_aligned_success_first):
        raise ValueError('Success-only replay requires aligned fastest TSIL without success-first fallback')
    if args.ppo_boundary_mode not in {'research', 'valid_only', 'legacy', 'exclude_value'}:
        raise ValueError('Unsupported PPO boundary mode')
    if args.ppo_boundary_mode != 'research' and (args.use_lstm or args.use_grpo or args.use_cost or args.use_timeawareness):
        raise ValueError('Boundary modes require feedforward PPO/TSIL without auxiliary critics')
    if args.reward_normalizer_scale_mode not in {'std_magnitude_floor', 'std_only'}:
        raise ValueError('Unknown reward divisor mode')
    if args.reward_normalizer_scale_mode == 'std_only' and args.reward_normalizer_mode != 'per_environment':
        raise ValueError('Std-only requires corrected per-environment histories')
    if args.minigrid_base_learner != 'current':
        from projects.TSIL.benchmarks.base_learners import BASE_LEARNERS
        if args.benchmark_backend != 'minigrid' or args.minigrid_base_learner not in BASE_LEARNERS:
            raise ValueError('Unknown MiniGrid base learner')
        for name, expected in BASE_LEARNERS[args.minigrid_base_learner].items():
            if getattr(args, name) != expected:
                raise ValueError(f'{args.minigrid_base_learner} requires {name}={expected}')
    if args.sil_success_actor_schedule not in {'off', 'constant', 'automatic'}:
        raise ValueError('Unknown success weighting mode')
    if args.sil_success_actor_schedule != 'off':
        if args.benchmark_backend != 'minigrid' or not args.minigrid_config_bank or not args.use_sil:
            raise ValueError('Success weighting requires MiniGrid TSIL with fixed configurations')
        if args.sil_success_weight_boost < 1 or args.sil_success_weight_decay <= 0 or args.sil_success_extra_mean_cap <= 0:
            raise ValueError('Invalid success weight parameters')
        if args.sil_success_weight_floor < 0 or args.sil_microbatch_size or args.sil_replay_weight_target_mean:
            raise ValueError('Success weighting needs nonnegative floor, full actor forward, no mean rescaling')
    if (args.tsil_drop_constant_inputs or args.tsil_drop_critic_velocity_inputs) and args.benchmark_backend != "minigrid":
        from projects.TSIL.agents import validate_constant_input_config
        validate_constant_input_config(args)
    if args.benchmark_backend not in {"metaworld", "minigrid"}:
        raise ValueError("benchmark_backend must be metaworld or minigrid")
    if args.benchmark_backend != "minigrid":
        return
    if args.tsil_drop_critic_velocity_inputs:
        raise ValueError("Critic velocity removal is specific to MetaWorld")
    if args.tsil_drop_constant_inputs and len(set(args.tasks)) != 1:
        raise ValueError("Removing task inputs requires a single-task MiniGrid policy")
    if args.minigrid_observation not in {"full", "partial"}:
        raise ValueError("minigrid_observation must be full or partial")
    if args.minigrid_reward not in {"success", "native_bonus", "native_no_success"}:
        raise ValueError("minigrid_reward must be success, native_bonus or native_no_success")
    if args.minigrid_config_bank and (args.minigrid_suite not in {'mt7', 'mt10_ff', 'mt15_ff', 'mt15_selection', 'mt15_refinement', 'mt15_control', 'mt15_fixed', 'mt15_gentle', 'mt15_repair_v1', 'mt15_repair_v2', 'mt15_repair_v3', 'mt15_repair_v4'} or args.minigrid_observation != 'partial'):
        raise ValueError('Fixed configuration banks require partial MT7, MT10-FF or MT15-FF')
    from projects.TSIL.benchmarks.suites import get_tsil_minigrid_suite
    suite = get_tsil_minigrid_suite(args.minigrid_suite)
    if any(task not in range(suite.task_count) for task in args.tasks):
        raise ValueError(f"{args.minigrid_suite} task IDs must be in [0, {suite.task_count - 1}]")
    if len(args.tasks) != len(args.task_counts) or any(n <= 0 for n in args.task_counts):
        raise ValueError("MiniGrid needs one positive task_count per task")
    unsupported = [name for name in ("beta", "use_lstm", "use_cost", "use_timeawareness", "use_grpo")
                   if getattr(args, name)]
    if unsupported or args.seq_length != 1:
        raise ValueError(f"MiniGrid supports feedforward categorical PPO/TSIL only: {unsupported}")
    if args.sil_filter_mode != "return" or args.temporal_deadline_mode != "off":
        raise ValueError("MiniGrid currently supports original return-filter TSIL without hard deadline resets")
    if args.ratio_range is not None or args.explicit_remaining_time_obs:
        raise ValueError("MiniGrid uses unit step time and the original three temporal fields")
    if args.checkpoint is not None:
        raise ValueError("MiniGrid training currently supports scratch runs; use its checkpoint evaluator for evaluation")
    if args.task_embedding_dim not in {0, suite.task_count}:
        raise ValueError("task_embedding_dim must match the MiniGrid suite")
    # Buffer sizing/config metadata follow native limits, not MetaWorld's 150.
    args.episodeLength = max(suite.tasks[task].max_steps for task in args.tasks)


def parse_args(argv=None):
    return parse_shared_args(project_name="TSIL", argv=argv, args_cls=TSILArgs,
                             project_validate_fn=validate_args)


class TSILTrainer(OnPolicyTrainer):
    project_name = "TSIL"

    def setup_project_training(self):
        if self.args.d2s_reset_critic:
            from projects.TSIL.benchmarks.d2s import maybe_reset_critic
            self.rollout_collector.before_step = lambda step: maybe_reset_critic(self, step)

    def after_training_complete(self):
        if self.args.inprocess_checkpoint_audit:
            from projects.TSIL.live_checkpoint_audit import run_live_checkpoint_audit
            run_live_checkpoint_audit(self, checkpoint_name="last")

    def _make_reward_normalizer(self):
        if self.args.reward_normalizer_mode != 'per_environment':
            return super()._make_reward_normalizer()
        if self.args.reward_normalizer_scale_mode == 'std_only':
            from projects.TSIL.reward_normalization import StdOnlyReturnRewardNormalizer
            return StdOnlyReturnRewardNormalizer(self.unique_task_ids, self.task_indices,
                                                  self.args.gamma, self.device)
        return super()._make_reward_normalizer()

    def after_rollout_complete(self, **kwargs):
        super().after_rollout_complete(**kwargs)
        if (not self.args.saving or self.reward_normalizer is None
                or self.args.reward_normalizer_mode != "per_environment"):
            return
        norm = self.reward_normalizer
        std = norm.G_rms._std.flatten().detach()
        bound = norm.G_r_max.detach() / norm.g_max
        valid = ~self.storage.dones.bool()
        row = {
            "steps": int(self.metrics.global_step),
            "valid_rows": int(valid.sum().item()), "all_rows": valid.numel(),
            "std": std.cpu().tolist(), "magnitude_divisor": bound.cpu().tolist(),
            "scale_mode": self.args.reward_normalizer_scale_mode,
            "magnitude_floor_would_bind": (bound > std).cpu().tolist(),
            "floor_active": ((bound > std).cpu().tolist()
                if self.args.reward_normalizer_scale_mode == "std_magnitude_floor"
                else [False] * std.numel()),
            "used_divisor": (std.maximum(bound) + norm.epsilon
                if self.args.reward_normalizer_scale_mode == "std_magnitude_floor"
                else std + norm.epsilon).cpu().tolist(),
        }
        with (Path(self.args.instance_dir) / "normalizer_trace.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")


    def _setup_environment(self):
        if self.args.benchmark_backend != "minigrid":
            return super()._setup_environment()
        from core.common.task_layout import build_task_id_lookup
        from projects.TSIL.benchmarks.minigrid import TSILMiniGridEnv
        self.envs = TSILMiniGridEnv(self.args, self.device)
        self.task_indices = self.envs.task_indices
        self.unique_task_ids = self.envs.unique_task_ids
        self.num_tasks = len(self.unique_task_ids)
        self.is_multi_task = self.num_tasks > 1
        self.tid_to_tidx = build_task_id_lookup(self.unique_task_ids, device=self.device)

    def _setup_agent(self):
        if self.args.benchmark_backend != "minigrid":
            return super()._setup_agent()
        from projects.TSIL.benchmarks.minigrid import MiniGridAgent
        self.agent = self.rollout_agent = MiniGridAgent(self.envs, self.args, device=self.device)
        self.agent.set_mode("train")

    def _sil_schedule_context(self):
        context = super()._sil_schedule_context()
        if self.args.benchmark_backend == "minigrid" and context is not None:
            context["episode_limits_by_env"] = self.envs.episode_limits
            if self.args.sil_success_actor_schedule != 'off':
                import torch
                records = self.envs.configuration_records
                identities = {identity: i for i, identity in enumerate(sorted({r['id'] for r in records}))}
                context['success_weight_state'] = dict(
                    config_ids=torch.tensor([identities[r['id']] for r in records], device=self.device),
                    success_counts=torch.tensor([r['success_count'] for r in records], device=self.device),
                )
        return context

    def save_project_meta(self):
        super().save_project_meta()
        if self.args.tsil_drop_constant_inputs and self.args.saving:
            obs_width, state_width = self.agent.network_input_dims()
            (Path(self.args.instance_dir) / "policy_input_contract.json").write_text(json.dumps({
                "agent_class": self.agent.__class__.__name__,
                "raw_observation_width": self.agent.obs_dim,
                "raw_state_width": self.agent.state_dim,
                "actor_input_width": obs_width,
                "critic_input_width": state_width,
                "removed_task_entries": self.agent.task_embedding_dim,
                "removed_speed_ratio": True,
                "removed_critic_fields": getattr(self.agent, "removed_critic_fields", []),
                "critic_retained_raw_indices": getattr(self.agent, "critic_input_indices", None),
                "temporal_inputs_masked": bool(self.args.fix_priv or not self.args.time2end),
                "temporal_mask_input_value": (0.0 if self.args.tsil_zero_masked_temporal_inputs
                    or self.args.benchmark_backend == 'minigrid' else 1.0 / self.envs.max_eps_time)
                    if (self.args.fix_priv or not self.args.time2end) else None,
                "retained_time_fields": ["elapsed/max_episode_time", "target/max_episode_time"],
                "reward_normalizer_mode": self.args.reward_normalizer_mode,
            }, indent=2) + "\n")
        if self.args.benchmark_backend == "minigrid" and self.args.saving:
            env = self.envs
            deadline = {
                str(int(task)): float(env.obs_range["time2end"][1][env.task_indices == task].min().item())
                for task in env.unique_task_ids
            }
            (Path(self.args.instance_dir) / "minigrid_state.json").write_text(json.dumps({
                "task_metadata": env.task_metadata,
                "observation_mode": env.minigrid_observation,
                "reward_mode": self.args.minigrid_reward,
                "image_shape": env.image_shape,
                "drop_constant_network_inputs": self.args.tsil_drop_constant_inputs,
                "cnn_head_input_width": self.agent.actor.head.mlp[0].in_features,
                "gpu_peak_allocated_mib": self._gpu_memory_peak("allocated"),
                "gpu_peak_reserved_mib": self._gpu_memory_peak("reserved"),
                "deadline_by_task": deadline,
                "valid_env_steps": env.valid_env_steps,
                "nominal_env_steps": self.metrics.global_step,
                "evaluation_deadline_rule": ("checkpoint-paired configuration target; heldout uses successful-training KNN5"
                    if env.configuration_bank is not None else "minimum learned bound per task, frozen for evaluation"),
                "configuration_counts": ({task: {split: len(data[split]) for split in ('train', 'heldout')}
                    for task, data in env.configuration_bank['tasks'].items()}
                    if env.configuration_bank is not None else None),
            }, indent=2) + "\n")

    def _gpu_memory_peak(self, kind):
        if self.device.type != "cuda":
            return 0.0
        import torch
        measure = torch.cuda.max_memory_allocated if kind == "allocated" else torch.cuda.max_memory_reserved
        return measure(self.device) / (1024 ** 2)
