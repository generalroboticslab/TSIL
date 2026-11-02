import os
import argparse
from dataclasses import dataclass, field
from simple_parsing import ArgumentParser
import datetime
import json
from distutils.util import strtobool
from math import ceil, isfinite
from typing import Callable, List, Optional, Type
import torch

from core.common.checkpointing import (
    infer_task_name_from_run_dir,
    resolve_checkpoint_run_dir,
)
from core.common.project_paths import normalize_project_result_root


@dataclass
class Args:
    # Env hyper parameters
    env_name: str = "MetaWorldV2" # Environment name controls the env class
    task_name: str = "MT01_T0" # Task name is used for grouping trainings
    tasks: List[int] = field(default_factory=lambda: [0])
    task_counts: List[int] = field(default_factory=lambda: [512])
    termination_on_success: bool = True
    fixed: bool = True # Fix the spatial randomization of each environment
    same_init_config_per_task: bool = False
    episodeLength: int = 150
    specific_idx: int = None
    
    # IsaacGym specific arguments
    use_gpu_pipeline: bool = True
    sim_device: str = "cuda:0"
    graphics_device_id: int = -1
    buffer_multiplier: float = 4.0
    headless: bool = False
    num_threads: int = 0
    subscenes: int = 0
    dt: float = 1/60
    
    # Training hyper parameters
    saving: bool = False
    rendering: bool = False
    quiet: bool = True
    seq_length: int = 1 # seq_length will directly control the sub env
    ratio_range: List[float] = None

    epstimeRewardScale: List[float] = field(default_factory=lambda: [0., 0.])
    scevelRewardScale: List[float] = field(default_factory=lambda: [0., 0.])
    sceaccRewardScale: float = 0.
    actvelPenaltyScale: float = 0.
    actaccPenaltyScale: float = 0.
    exp_scheduler: bool = False
    scevelSchedule: float = 1.
    vel_match: bool = True
    no_dense: bool = False

    time2end: bool = True
    explicit_remaining_time_obs: bool = False
    temporal_efficiency_bonus: Optional[bool] = None
    update_ddl: bool = False
    anneal_ddl: bool = False
    ddl_anneal_alpha: float = 0.2
    time2end_init_frac: float = 1.0
    metaworld_checkpoint_target_bank: bool = False
    temporal_deadline_mode: str = "off"  # off | hard_reset
    temporal_deadline_grace_frac: float = 0.1
    temporal_deadline_terminal: bool = True
    temporal_deadline_min_seconds: float = 0.0
    time_ratio: bool = True
    fix_linvel: bool = False
    fix_limvel: bool = False
    fix_priv: bool = False
    reset_critic: bool = False

    control_freq_inv: int = 1 # 3
    gripper_freq_inv: int = 1 # 10
    max_vel_subtract: float = 0. # 0.7
    limit_gripper_vel: bool = False # True
    control_type: str = "ik"
    
    delayed_stress: bool = False
    stress_start_iter: int = -1
    stress_start_on_sil_success: bool = True
    stress_require_all_tasks: bool = False
    init_curri_ratio: float = None
    success_threshold: float = 0. # > 0 to start curriculum
    curriculum_step: float = 0.03
    curri_hold_iters: int = 10
    curri_rate: int = 1
    init_success: float = 0.98
    successRewardScale: float = 100.
    native_success_reward: float = 0.0
    task_dense_reward_scales: str = "{}"
    reward_scale: float = 0.1
    lever_pull_dense_reward_scale: float = 1.0
    peg_insert_progress_reward_scale: float = 1.0
    step_cost: float = 0.0
    step_cost_mode: str = "fixed"  # fixed | horizon: coefficient * success reward / max steps
    dense_reward_scale: List[float] = field(default_factory=lambda: [1.0, 1.0])
    dense_reward_mode: str = "none"  # none | hard | soft
    dense_reward_dropout_p: float = 0.0
    dense_reward_dropout_mode: str = "step"  # step | episode
    hard_clip: float = 0.5
    
    # Environment-shaping parameters
    constrain_grasp: bool = False
    num_gms: int = 1
    use_potential_r: bool = False
    
    # I/O hyper parameter
    debug: bool = False
    result_dir: str = "results"  # expands to results/<project>/train_res
    result_benchmark_name: str = None  # Optional benchmark folder when using launcher-managed result layouts
    result_experiment_name: str = None  # Optional experiment folder when using launcher-managed result layouts
    result_training_stage_name: str = None  # Optional training_stage selector folder when using launcher-managed result layouts
    result_task_dir_name: str = None  # Optional task folder name distinct from task_name
    wandb: bool = False
    wandb_project: str = None
    wandb_entity: str = None
    force_name: str = None
    method_suffix: str = None
    train_stage: str = None
    script_time: str = None
    
    # Algorithm specific arguments
    beta: bool = True
    use_lstm: bool = False
    total_iters: int = None  # Optional override; resolved from total_timesteps when omitted
    total_timesteps: int = int(1e9)
    num_envs: int = 16384
    num_steps: int = 32
    minibatch_size: int = 8192
    update_epochs: int = 5
    activation: str = 'tanh'  # Activation function name: 'relu', 'tanh', 'elu', 'leaky_relu', 'selu', 'gelu', 'silu', 'mish', etc.
    use_layernorm: bool = True  # Whether to use LayerNorm in MLP hidden layers
    anneal_lr: bool = False
    scheduler: str = "linear"
    gae: bool = True
    gae_lambda: float = 0.95

    use_timeawareness: bool = False
    norm_obs: bool = True
    norm_rew: bool = True
    reward_normalizer_mode: str = "per_environment"
    norm_value: bool = True
    norm_cost: bool = True
    pertask_norm: bool = False  # Per-task normalisation for obs/state/value (vs global task-agnostic)
    pertask_norm_adv: bool = True  # Per-task advantage normalisation only (vs global advantage norm)
    use_cost: bool = False
    num_cost: int = 2
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: List[float] = field(default_factory=lambda: [0.005, 0.005])
    vf_coef: float = 4.0
    max_grad_norm: float = 1.0
    target_kl: Optional[float] = 2.5  # Optional hard approximate-KL rollback guard
    ppo_kl_penalty_coef: float = 0.0  # Optional soft KL(old rollout policy || current policy)
    ppo_kl_activation_success: float = 0.0  # Zero: immediate; positive: latch on once success reaches threshold
    value_bootstrap: bool = True
    use_grpo: bool = False  # Use GRPO (reward-to-go advantages, no critic) instead of PPO (GAE)
    grpo_use_clipping: bool = True  # Use PPO-style clipping in GRPO actor loss (set False for pure REINFORCE)
    policy_grad_noise_scale: float = 0.0
    policy_grad_noise_mode: str = "additive"  # additive | norm_preserved_direction
    bounds_loss_coef: float = 0.0001
    deterministic: bool = False
    gamma: float = 0.995
    c_gamma: List[float] = field(default_factory=lambda: [1, 0.99])
    c_scale: List[float] = field(default_factory=lambda: [0, 1])
    lr: float = 5e-4
    seed: int = 123456

    lstm_hidden_size: int = 256
    hidden_size: List[int] = field(default_factory=lambda: [512, 256, 128, 64, 32])
    task_embedding_dim: int = 0  # 0 = resolve from the environment observation contract

    ckpt_task: str = None
    checkpoint: str = None
    index_episode: str = "last"
    freeze: bool = False # Freeze the agent parameters apart from the last layer
    random_policy: bool = False
    save_iter: int = 10
    save_periodic_pct: float = 0  # Save checkpoints at this percentage interval (e.g. 0.1 = every 10 percent); 0 disables
    best_suc_tail_frac: float = 0.1
    running_len: Optional[int] = None
    warmup_iters: int = 0
    best_only: bool = True
    last_only: bool = True

    # Env0 debug trajectory export
    record_env0_trajectory: bool = False
    trajectory_topk: int = 5
    track_signal_metrics: bool = True
    signal_metrics_window: int = None
    signal_gradient_interval: int = 0
    dense_tail_frac: float = 0.1
    outcome_diagnostics: bool = False
    outcome_diagnostic_batch_size: int = 4096
    fixed_success_anchor_threshold: float = 0.8
    causal_shadow_diagnostics: bool = False
    causal_diagnostic_dir: str = None
    causal_shadow_peak_threshold: float = 0.5
    causal_shadow_drop_threshold: float = 0.1
    causal_shadow_max_captures: int = 1
    update_diagnostic_trace: bool = False
    update_event_capture: bool = False
    update_event_peak_threshold: float = 0.2
    update_event_kl_threshold: float = 0.1
    
    # SIL replay
    use_sil: bool = False
    sil_train: bool = True
    sil_mode: str = "sil"  # sil | bc
    sil_source: str = "fastest"  # fastest | return
    sil_filter_mode: str = "return"  # return | temporal | both
    sil_temporal_model: str = "standalone"  # standalone agent temporal critic | time-aware critic
    sil_temporal_target_mode: str = "mc"  # mc | td | td_lambda
    sil_temporal_lambda: float = 0.95
    sil_temporal_vf_coef: float = 0.25
    sil_temporal_success_only: bool = True
    sil_temporal_ready_samples: int = 1024
    sil_topk: int = 512
    sil_coef: float = 0.1
    sil_vf_coef: float = 0.05
    sil_batch_size: int = 2048
    sil_sample_unit: str = "transition"  # transition | trajectory
    sil_start_iter: int = 0
    sil_normalize_gap: bool = False
    sil_normalized_gap_clip: float = 1.0  # Upper cap after gap std division; 0 disables. Ignored without normalization.
    # Historical target ablations; excluded from the unified active registry.
    sil_relabel_reward_scale: float = -1.0
    sil_success_bootstrap: bool = False
    sil_gap_normalization: str = "std"  # std | max | clip (original-style SIL)
    sil_per_beta: float = 0.0  # Sampling correction; opt-in replacement/global PER only.
    sil_per_alpha: float = 1.0
    sil_global_per: bool = False
    sil_global_per_capacity: int = 50000
    sil_loss_denominator: str = "all"  # all | positive; shared actor/critic reduction
    sil_transition_selection: str = "window"  # window | least_selected_positive | gap_proportional_positive | gap_proportional_replacement
    sil_microbatch_size: int = 0  # Opt-in recomputed active replay; 0 keeps dense autograd
    sil_replay_weight_target_mean: float = 0.0
    sil_update_interval: int = 1
    sil_separate_updates: int = 0  # 0 = joint PPO+SIL; >0 = PPO first, then this many SIL-only updates
    sil_success_fraction: float = -1.0  # canonical: <0 legacy sampling; [0,1] max fast-success replay share
    sil_success_sample_frac: float = -1.0  # deprecated alias for sil_success_fraction
    sil_success_repeat: str = "to_fraction"  # to_fraction | to_floor | none
    sil_success_floor: float = 0.0  # minimum success share used only by to_floor
    sil_env_step_quota: int = 0  # 0 disables; -1 uses trajectory-safe automatic quota; >0 is an explicit quota
    sil_replay_build_scope: str = "minibatch"  # minibatch | update
    sil_shuffle_replay_transitions: bool = False
    sil_aligned_replay: bool = False
    sil_replay_trajectories_per_env: int = 5
    sil_replay_rotate_trajectories: bool = False
    # For aligned fastest-source replay, use a successful trajectory whenever
    # that environment has one; use its high-return archive only otherwise.
    sil_aligned_success_first: bool = False
    sil_replay_transitions_per_trajectory: int = 0  # 0 follows PPO num_steps
    sil_replay_epochs: int = 0  # 0 follows PPO update_epochs
    sil_success_weight_mode: str = "adaptive"  # adaptive | fixed
    sil_fixed_success_weight: float = 2.0
    # episode_limit preserves completed campaigns; adaptive_deadline implements
    # manuscript Sec. 3.3: 1 + 1_success * min(D_i / T_success, 1).  none is
    # the single-factor ablation that assigns every source trajectory weight 1.
    sil_trajectory_weight_mode: str = "episode_limit"  # none | episode_limit | adaptive_deadline
    # Global transition sampling normally gives every selected transition unit
    # source weight. TSIL-Global opts in to the source trajectory's efficiency
    # weight without restoring a per-trajectory sampling quota.
    sil_global_trajectory_weighting: bool = False
    sil_speed_priority_min: float = 1.0
    sil_speed_priority_max: float = 2.0
    sil_actor_weight_coef: float = 0.0
    sil_actor_weight_max: float = 3.0
    sil_actor_weight_mode: str = "multiply"  # multiply | add | add_norm | rescue_norm
    sil_actor_gap_alpha: float = 0.0
    sil_actor_temporal_margin: float = 0.0
    sil_success_actor_floor: float = 0.0
    # Optional actor-loss balancing after the SIL gate.  This changes weights,
    # not which replay transitions are selected.
    sil_success_actor_min_share: float = 0.0
    sil_success_actor_boost_cap: float = 16.0
    sil_fast_success_retention_coef: float = 0.0
    sil_fast_success_retention_temperature: float = 0.05
    double_value_critic: bool = False
    double_value_mode: str = "minimum"  # minimum | primary | timeout_minimum
    sil_analysis_interval: int = 0  # 0 disables optional SIL reference/actor analysis.
    sil_analysis_batch_size: int = 1024
    sil_landscape_grid: int = 9
    sil_landscape_span: float = 0.05

    # PyTorch specific arguments
    cuda: bool = True
    cpus: List[int] = field(default_factory=list)
    torch_deterministic: bool = True


def parse_args(
    project_name: str = "TSIL",
    argv: List[str] = None,
    args_cls: Type[Args] = Args,
    project_validate_fn: Optional[Callable[[Args], None]] = None,
    project_setup_dirs_fn: Optional[Callable[[Args], None]] = None,
):
    parser = ArgumentParser(description='Time-aware Policy Learning')
    parser.add_arguments(args_cls, dest="args")
    args = parser.parse_args(argv).args
    args.project_name = project_name
    args.result_dir = normalize_project_result_root(args.result_dir, project_name, "train_res")
    has_benchmark_name = bool(args.result_benchmark_name)
    has_task_dir_name = bool(args.result_task_dir_name)
    has_experiment_name = bool(args.result_experiment_name)
    has_training_stage_name = bool(args.result_training_stage_name)
    if has_benchmark_name != has_task_dir_name:
        raise ValueError(
            "result_benchmark_name and result_task_dir_name must be provided together."
        )
    if has_experiment_name != has_training_stage_name:
        raise ValueError(
            "result_experiment_name and result_training_stage_name must be provided together."
        )
    if (has_experiment_name or has_training_stage_name) and not (has_benchmark_name and has_task_dir_name):
        raise ValueError(
            "result_benchmark_name and result_task_dir_name must be provided when "
            "result_experiment_name/result_training_stage_name are used."
        )
    
    # Post-processing and validation
    if args.reward_normalizer_mode not in {"per_environment", "legacy_task"}:
        raise ValueError("reward_normalizer_mode must be per_environment or legacy_task")
    # 1. Compute batch sizes
    args.num_envs = sum(args.task_counts) if args.task_counts is not None else args.num_envs
    if args.running_len is None:
        args.running_len = 5 * args.num_envs
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = args.batch_size if args.minibatch_size is None else args.minibatch_size
    if args.minibatch_size <= 0:
        raise ValueError(f"minibatch_size must be positive, got {args.minibatch_size}")
    if args.batch_size < args.minibatch_size:
        raise ValueError(
            f"Batch size {args.batch_size} must be greater than or equal to "
            f"minibatch size {args.minibatch_size}"
        )
    if not 0.0 < args.best_suc_tail_frac <= 1.0:
        raise ValueError(f"best_suc_tail_frac must be in (0, 1], got {args.best_suc_tail_frac}")
    args.num_minibatches = max(ceil(args.batch_size / args.minibatch_size), 1)
    if len(set(args.tasks)) != len(args.tasks):
        raise ValueError(f"tasks list has duplicate entries: {args.tasks}")
    if args.total_iters is not None:
        if args.total_iters <= 0:
            raise ValueError(f"total_iters must be positive when set, got {args.total_iters}")
        args.total_timesteps = int(args.batch_size * args.total_iters)
    else:
        args.total_iters = max(int(args.total_timesteps // args.batch_size), 1)
    if args.task_embedding_dim < 0:
        raise ValueError(
            f"task_embedding_dim must be non-negative, got {args.task_embedding_dim}"
        )
    args.double_value_mode = str(args.double_value_mode).lower().replace("-", "_")
    if args.double_value_mode not in {"minimum", "primary", "timeout_minimum"}:
        raise ValueError(
            "double_value_mode must be 'minimum', 'primary', or 'timeout_minimum', "
            f"got {args.double_value_mode!r}"
        )
    if not args.double_value_critic and args.double_value_mode != "minimum":
        raise ValueError("double_value_mode only applies when double_value_critic=True")
    
    if args.cpus:
        import psutil

        print('Running on specific CPUS:', args.cpus)
        process = psutil.Process()
        process.cpu_affinity(args.cpus)
    
    if args.use_timeawareness:
        if args.termination_on_success is not True:
            raise ValueError("Time-awareness requires termination_on_success to be True")
    if not 0.0 < args.ddl_anneal_alpha <= 1.0:
        raise ValueError(f"ddl_anneal_alpha must be in (0, 1], got {args.ddl_anneal_alpha}")
    if not 0.0 < args.time2end_init_frac <= 1.0:
        raise ValueError(f"time2end_init_frac must be in (0, 1], got {args.time2end_init_frac}")
    args.temporal_deadline_mode = str(args.temporal_deadline_mode).lower()
    if args.temporal_deadline_mode not in {"off", "hard_reset"}:
        raise ValueError(
            "temporal_deadline_mode must be 'off' or 'hard_reset', "
            f"got {args.temporal_deadline_mode}"
        )
    if args.temporal_deadline_grace_frac < 0:
        raise ValueError(
            "temporal_deadline_grace_frac must be non-negative, "
            f"got {args.temporal_deadline_grace_frac}"
        )
    if args.temporal_deadline_min_seconds < 0:
        raise ValueError(
            "temporal_deadline_min_seconds must be non-negative, "
            f"got {args.temporal_deadline_min_seconds}"
        )
    if not 0.0 < args.dense_tail_frac <= 1.0:
        raise ValueError(f"dense_tail_frac must be in (0, 1], got {args.dense_tail_frac}")
    if args.signal_metrics_window is None:
        args.signal_metrics_window = args.running_len
    if args.signal_metrics_window <= 0:
        raise ValueError(f"signal_metrics_window must be positive, got {args.signal_metrics_window}")
    if args.step_cost_mode not in {"fixed", "horizon"}:
        raise ValueError("step_cost_mode must be fixed or horizon")
    if args.step_cost < 0:
        raise ValueError(f"step_cost must be non-negative, got {args.step_cost}")
    from core.common.task_rewards import parse_dense_scales
    parse_dense_scales(args.task_dense_reward_scales)
    if not isfinite(args.native_success_reward) or (args.native_success_reward < 0 and args.native_success_reward != -1):
        raise ValueError("native_success_reward must be -1 (historical) or finite nonnegative")
    if args.lever_pull_dense_reward_scale < 0:
        raise ValueError(
            "lever_pull_dense_reward_scale must be non-negative, "
            f"got {args.lever_pull_dense_reward_scale}"
        )
    if not 0.0 <= args.peg_insert_progress_reward_scale <= 1.0:
        raise ValueError(
            "peg_insert_progress_reward_scale must be in [0, 1], "
            f"got {args.peg_insert_progress_reward_scale}"
        )
    if len(args.dense_reward_scale) != 2:
        raise ValueError(f"dense_reward_scale must have two values, got {args.dense_reward_scale}")
    args.dense_reward_mode = str(args.dense_reward_mode).lower()
    if args.dense_reward_mode not in {"none", "hard", "soft"}:
        raise ValueError(f"dense_reward_mode must be 'none', 'hard', or 'soft', got {args.dense_reward_mode}")
    if not 0.0 <= args.dense_reward_dropout_p <= 1.0:
        raise ValueError(f"dense_reward_dropout_p must be in [0, 1], got {args.dense_reward_dropout_p}")
    args.dense_reward_dropout_mode = str(args.dense_reward_dropout_mode).lower()
    if args.dense_reward_dropout_mode not in {"step", "episode"}:
        raise ValueError(
            f"dense_reward_dropout_mode must be 'step' or 'episode', got {args.dense_reward_dropout_mode}"
        )
    if not 0.0 <= args.hard_clip <= 1.0:
        raise ValueError(f"hard_clip must be in [0, 1], got {args.hard_clip}")
    if args.policy_grad_noise_scale < 0:
        raise ValueError(
            f"policy_grad_noise_scale must be non-negative, got {args.policy_grad_noise_scale}"
        )
    args.policy_grad_noise_mode = str(args.policy_grad_noise_mode).lower()
    if args.policy_grad_noise_mode not in {"additive", "norm_preserved_direction"}:
        raise ValueError(
            "policy_grad_noise_mode must be 'additive' or "
            f"'norm_preserved_direction', got {args.policy_grad_noise_mode!r}"
        )
    if args.stress_start_iter < -1:
        raise ValueError(f"stress_start_iter must be >= -1, got {args.stress_start_iter}")
    args.stress_active = not bool(args.delayed_stress)
    args.effective_dense_reward_dropout_p = (
        float(args.dense_reward_dropout_p) if args.stress_active else 0.0
    )
    args.effective_policy_grad_noise_scale = (
        float(args.policy_grad_noise_scale) if args.stress_active else 0.0
    )
    if args.sil_mode not in {"sil", "bc"}:
        raise ValueError(f"sil_mode must be 'sil' or 'bc', got {args.sil_mode}")
    if args.sil_source not in {"fastest", "return"}:
        raise ValueError(
            f"sil_source must be 'fastest' or 'return', got {args.sil_source}"
        )
    args.sil_filter_mode = str(args.sil_filter_mode).lower()
    if args.sil_filter_mode not in {"return", "temporal", "both"}:
        raise ValueError(
            "sil_filter_mode must be 'return', 'temporal', or 'both', "
            f"got {args.sil_filter_mode}"
        )
    args.sil_temporal_model = str(args.sil_temporal_model).lower()
    if args.sil_temporal_model not in {"standalone", "critic"}:
        raise ValueError(
            "sil_temporal_model must be 'standalone' or 'critic', "
            f"got {args.sil_temporal_model}"
        )
    args.sil_temporal_target_mode = str(args.sil_temporal_target_mode).lower()
    if args.sil_temporal_target_mode not in {"mc", "td", "td_lambda"}:
        raise ValueError(
            "sil_temporal_target_mode must be 'mc', 'td', or 'td_lambda', "
            f"got {args.sil_temporal_target_mode}"
        )
    if not 0.0 <= args.sil_temporal_lambda <= 1.0:
        raise ValueError(
            "sil_temporal_lambda must be in [0, 1], "
            f"got {args.sil_temporal_lambda}"
        )
    if args.signal_gradient_interval < 0:
        raise ValueError(
            "signal_gradient_interval must be non-negative, "
            f"got {args.signal_gradient_interval}"
        )
    if args.outcome_diagnostic_batch_size <= 0:
        raise ValueError(
            "outcome_diagnostic_batch_size must be positive, "
            f"got {args.outcome_diagnostic_batch_size}"
        )
    if not 0.0 <= args.fixed_success_anchor_threshold <= 1.0:
        raise ValueError(
            "fixed_success_anchor_threshold must be in [0, 1], "
            f"got {args.fixed_success_anchor_threshold}"
        )
    if (
        args.use_sil
        and args.sil_filter_mode in {"temporal", "both"}
        and args.sil_temporal_model == "critic"
        and not args.use_timeawareness
    ):
        raise ValueError("critic temporal SIL filtering requires use_timeawareness=True")
    if args.sil_temporal_vf_coef < 0:
        raise ValueError(f"sil_temporal_vf_coef must be non-negative, got {args.sil_temporal_vf_coef}")
    if args.sil_success_actor_floor < 0:
        raise ValueError(
            "sil_success_actor_floor must be non-negative, "
            f"got {args.sil_success_actor_floor}"
        )
    if args.sil_fast_success_retention_coef < 0:
        raise ValueError(
            "sil_fast_success_retention_coef must be non-negative, "
            f"got {args.sil_fast_success_retention_coef}"
        )
    if args.sil_fast_success_retention_temperature <= 0:
        raise ValueError(
            "sil_fast_success_retention_temperature must be positive, "
            f"got {args.sil_fast_success_retention_temperature}"
        )
    if args.sil_temporal_ready_samples < 0:
        raise ValueError(
            "sil_temporal_ready_samples must be non-negative, "
            f"got {args.sil_temporal_ready_samples}"
        )
    args.sil_sample_unit = str(args.sil_sample_unit).lower()
    if args.sil_sample_unit not in {"transition", "trajectory"}:
        raise ValueError(
            f"sil_sample_unit must be 'transition' or 'trajectory', got {args.sil_sample_unit}"
        )
    if args.sil_topk <= 0:
        raise ValueError(f"sil_topk must be positive, got {args.sil_topk}")
    if args.use_sil and args.use_lstm:
        raise ValueError("use_sil is not supported when use_lstm=True")
    if args.use_sil and args.use_grpo:
        raise ValueError("use_sil is not supported when use_grpo=True")
    if args.sil_batch_size <= 0:
        raise ValueError(f"sil_batch_size must be positive, got {args.sil_batch_size}")
    if not isfinite(args.sil_normalized_gap_clip) or args.sil_normalized_gap_clip < 0:
        raise ValueError("sil_normalized_gap_clip must be finite and non-negative (0 disables the cap)")
    if args.sil_replay_weight_target_mean < 0:
        raise ValueError(
            "sil_replay_weight_target_mean must be non-negative, "
            f"got {args.sil_replay_weight_target_mean}"
        )
    if args.sil_start_iter < 0:
        raise ValueError(f"sil_start_iter must be non-negative, got {args.sil_start_iter}")
    if args.sil_update_interval <= 0:
        raise ValueError(f"sil_update_interval must be positive, got {args.sil_update_interval}")
    if args.sil_loss_denominator not in {"all", "positive"}:
        raise ValueError("sil_loss_denominator must be all or positive")
    if args.sil_gap_normalization not in {"std", "max", "clip"}:
        raise ValueError("Unknown sil_gap_normalization")
    if not isfinite(args.sil_per_beta) or not 0 <= args.sil_per_beta <= 1:
        raise ValueError("sil_per_beta must be in [0,1]")
    if not isfinite(args.sil_per_alpha) or args.sil_per_alpha <= 0:
        raise ValueError("sil_per_alpha must be finite and positive")
    if args.sil_global_per_capacity <= 100:
        raise ValueError("sil_global_per_capacity must exceed 100")
    if args.sil_per_beta > 0 and not (args.sil_global_per or args.sil_transition_selection == "gap_proportional_replacement"):
        raise ValueError("PER correction requires replacement sampling")
    if args.sil_global_per and (not args.sil_aligned_replay or args.sil_replay_build_scope != "update"
            or args.sil_source != "return" or not args.fix_priv or args.update_ddl
            or args.temporal_efficiency_bonus or args.sil_replay_no_fallback
            or args.sil_gap_normalization != "clip" or not args.sil_normalize_gap
            or args.sil_loss_denominator != "positive" or args.use_lstm):
        raise ValueError("Global SIL PER requires feedforward, return-only aligned PPO+SIL without temporal shaping")
    if args.sil_transition_selection not in {
        "window", "least_selected_positive", "gap_proportional_positive",
        "gap_proportional_replacement", "global_gap_proportional_positive",
    }:
        raise ValueError("Unknown sil_transition_selection")
    if not isfinite(args.sil_relabel_reward_scale) or (args.sil_relabel_reward_scale < 0 and args.sil_relabel_reward_scale != -1):
        raise ValueError("sil_relabel_reward_scale must be -1 or finite non-negative")
    if args.sil_success_bootstrap or args.sil_relabel_reward_scale >= 0:
        if not (args.use_sil and args.sil_train and args.sil_mode == "sil"
                and args.sil_filter_mode == "return" and args.sil_aligned_replay
                and args.sil_replay_no_fallback and args.sil_source == "fastest"
                and args.value_bootstrap and not args.use_lstm):
            raise ValueError("Success target variants require IH aligned success-only feedforward SIL")
    if args.sil_loss_denominator == "positive" and (
        args.sil_mode != "sil" or args.sil_filter_mode != "return"
        or args.sil_success_actor_floor != 0 or args.sil_fast_success_retention_coef != 0
        or args.sil_actor_weight_coef != 0
    ):
        raise ValueError("Positive denominator requires return-filter SIL without gate rescue")
    if args.sil_transition_selection in {
        "least_selected_positive", "gap_proportional_positive",
        "gap_proportional_replacement", "global_gap_proportional_positive",
    } and (
        not args.sil_aligned_replay or args.sil_replay_build_scope != "update"
        or args.sil_loss_denominator != "positive" or args.use_lstm
    ):
        raise ValueError("Positive selection requires aligned update-scope, positive-denominator feedforward SIL")
    if args.sil_microbatch_size < 0:
        raise ValueError("sil_microbatch_size must be non-negative")
    if args.ppo_kl_penalty_coef < 0:
        raise ValueError("ppo_kl_penalty_coef must be non-negative")
    if not 0.0 <= args.ppo_kl_activation_success <= 1.0:
        raise ValueError("ppo_kl_activation_success must be between zero and one")
    if args.sil_microbatch_size and (
        args.sil_mode != "sil" or args.sil_filter_mode != "return"
        or args.use_lstm or args.use_cost or args.use_timeawareness
        or args.double_value_critic or args.sil_actor_weight_coef != 0
        or args.sil_success_actor_floor != 0 or args.sil_fast_success_retention_coef != 0
    ):
        raise ValueError("sil_microbatch_size requires feedforward return-filter SIL without actor rescue")
    if args.sil_actor_weight_coef < 0:
        raise ValueError(f"sil_actor_weight_coef must be non-negative, got {args.sil_actor_weight_coef}")
    if args.sil_actor_weight_max < 1:
        raise ValueError(f"sil_actor_weight_max must be at least 1, got {args.sil_actor_weight_max}")
    args.sil_actor_weight_mode = str(args.sil_actor_weight_mode).lower()
    if args.sil_actor_weight_mode not in {"multiply", "add", "add_norm", "rescue_norm"}:
        raise ValueError(
            "sil_actor_weight_mode must be 'multiply', 'add', 'add_norm', or 'rescue_norm', "
            f"got {args.sil_actor_weight_mode}"
        )
    if args.sil_actor_gap_alpha < 0:
        raise ValueError(f"sil_actor_gap_alpha must be non-negative, got {args.sil_actor_gap_alpha}")
    if args.sil_actor_temporal_margin < 0:
        raise ValueError(
            f"sil_actor_temporal_margin must be non-negative, got {args.sil_actor_temporal_margin}"
        )
    if not 0.0 <= args.sil_success_actor_min_share < 1.0:
        raise ValueError(
            "sil_success_actor_min_share must be in [0, 1), "
            f"got {args.sil_success_actor_min_share}"
        )
    if args.sil_success_actor_boost_cap < 1.0:
        raise ValueError(
            "sil_success_actor_boost_cap must be at least 1, "
            f"got {args.sil_success_actor_boost_cap}"
        )
    if args.sil_separate_updates < 0:
        raise ValueError(f"sil_separate_updates must be non-negative, got {args.sil_separate_updates}")
    if args.sil_analysis_interval < 0:
        raise ValueError(f"sil_analysis_interval must be non-negative, got {args.sil_analysis_interval}")
    if args.sil_trajectory_weight_mode not in {"none", "episode_limit", "adaptive_deadline"}:
        raise ValueError(
            "sil_trajectory_weight_mode must be 'none', 'episode_limit' or 'adaptive_deadline', "
            f"got {args.sil_trajectory_weight_mode}"
        )
    if args.sil_global_trajectory_weighting and (
        args.sil_transition_selection != "global_gap_proportional_positive"
        or args.sil_source != "fastest"
    ):
        raise ValueError(
            "sil_global_trajectory_weighting requires global fastest-source replay"
        )
    if args.sil_analysis_batch_size <= 0:
        raise ValueError(f"sil_analysis_batch_size must be positive, got {args.sil_analysis_batch_size}")
    if args.sil_landscape_grid < 3:
        raise ValueError(f"sil_landscape_grid must be at least 3, got {args.sil_landscape_grid}")
    if args.sil_landscape_span <= 0:
        raise ValueError(f"sil_landscape_span must be positive, got {args.sil_landscape_span}")
    for field_name in ("sil_success_fraction", "sil_success_sample_frac"):
        value = float(getattr(args, field_name))
        if value > 1.0:
            raise ValueError(f"{field_name} must be <= 1.0, got {value}")
    if args.sil_success_fraction >= 0.0 and args.sil_success_sample_frac >= 0.0:
        if abs(float(args.sil_success_fraction) - float(args.sil_success_sample_frac)) > 1e-8:
            raise ValueError(
                "sil_success_fraction and deprecated sil_success_sample_frac disagree: "
                f"{args.sil_success_fraction} != {args.sil_success_sample_frac}"
            )
    args.sil_success_repeat = str(args.sil_success_repeat).strip().lower()
    if args.sil_success_repeat not in {"to_fraction", "to_floor", "none"}:
        raise ValueError(
            "sil_success_repeat must be 'to_fraction', 'to_floor', or 'none', "
            f"got {args.sil_success_repeat}"
        )
    if not 0.0 <= args.sil_success_floor <= 1.0:
        raise ValueError(f"sil_success_floor must be in [0, 1], got {args.sil_success_floor}")
    if args.sil_env_step_quota < -1:
        raise ValueError(
            "sil_env_step_quota must be -1 (automatic), 0 (off), or positive, "
            f"got {args.sil_env_step_quota}"
        )
    args.sil_replay_build_scope = str(args.sil_replay_build_scope).strip().lower()
    if args.sil_replay_build_scope not in {"minibatch", "update"}:
        raise ValueError(
            "sil_replay_build_scope must be 'minibatch' or 'update', "
            f"got {args.sil_replay_build_scope}"
        )
    if args.sil_replay_transitions_per_trajectory == 0:
        args.sil_replay_transitions_per_trajectory = args.num_steps
    if args.sil_replay_epochs == 0:
        args.sil_replay_epochs = args.update_epochs
    if args.sil_replay_trajectories_per_env <= 0:
        raise ValueError(
            "sil_replay_trajectories_per_env must be positive, "
            f"got {args.sil_replay_trajectories_per_env}"
        )
    if args.sil_replay_transitions_per_trajectory <= 0:
        raise ValueError(
            "sil_replay_transitions_per_trajectory must be positive, "
            f"got {args.sil_replay_transitions_per_trajectory}"
        )
    if args.sil_replay_epochs <= 0:
        raise ValueError(f"sil_replay_epochs must be positive, got {args.sil_replay_epochs}")
    if args.sil_aligned_replay:
        if args.sil_replay_trajectories_per_env > args.sil_topk:
            raise ValueError(
                "sil_replay_trajectories_per_env cannot exceed sil_topk, "
                f"got {args.sil_replay_trajectories_per_env} > {args.sil_topk}"
            )
        if args.sil_sample_unit != "trajectory":
            raise ValueError("sil_aligned_replay requires sil_sample_unit='trajectory'")
        if args.sil_replay_build_scope != "update":
            raise ValueError("sil_aligned_replay requires sil_replay_build_scope='update'")
        if args.sil_success_fraction >= 0.0 or args.sil_success_sample_frac >= 0.0:
            raise ValueError(
                "sil_aligned_replay uses per-environment fastest-first selection and is incompatible "
                "with a global success fraction"
            )
    elif args.sil_replay_rotate_trajectories:
        raise ValueError("sil_replay_rotate_trajectories requires sil_aligned_replay=True")
    if args.sil_aligned_success_first:
        if not args.sil_aligned_replay:
            raise ValueError("sil_aligned_success_first requires sil_aligned_replay=True")
        if args.sil_source != "fastest":
            raise ValueError("sil_aligned_success_first requires sil_source='fastest'")
    if args.sil_fast_success_retention_coef > 0:
        if not args.use_sil or args.sil_mode != "sil":
            raise ValueError(
                "sil_fast_success_retention_coef requires use_sil=True and sil_mode='sil'"
            )
        if not args.sil_aligned_replay or args.sil_source != "fastest":
            raise ValueError(
                "sil_fast_success_retention_coef requires PPO-aligned fastest-source replay"
            )
        if args.sil_filter_mode != "return":
            raise ValueError(
                "sil_fast_success_retention_coef currently requires sil_filter_mode='return'"
            )
    args.sil_success_weight_mode = str(args.sil_success_weight_mode).lower()
    if args.sil_success_weight_mode not in {"adaptive", "fixed"}:
        raise ValueError(
            "sil_success_weight_mode must be 'adaptive' or 'fixed', "
            f"got {args.sil_success_weight_mode}"
        )
    if args.sil_fixed_success_weight < 0:
        raise ValueError(
            "sil_fixed_success_weight must be non-negative, "
            f"got {args.sil_fixed_success_weight}"
        )
    if args.sil_speed_priority_min < 0:
        raise ValueError(f"sil_speed_priority_min must be non-negative, got {args.sil_speed_priority_min}")
    if args.sil_speed_priority_max < args.sil_speed_priority_min:
        raise ValueError(
            "sil_speed_priority_max must be greater than or equal to "
            f"sil_speed_priority_min, got {args.sil_speed_priority_max} < {args.sil_speed_priority_min}"
        )
    if args.sil_vf_coef < 0:
        raise ValueError(f"sil_vf_coef must be non-negative, got {args.sil_vf_coef}")
    if args.init_curri_ratio is None:
        raise ValueError("init_curri_ratio must be specified explicitly.")
    if project_validate_fn is not None:
        project_validate_fn(args)
    if args.script_time is None:
        args.script_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.checkpoint is not None:
        _load_checkpoint_config(args)

    args.train_stage = _build_train_stage(args)
    args.method_name = _build_method_name(args)

    # Build naming convention
    _build_experiment_name(args)
    # Setup directories
    _setup_directories(args)
    if project_setup_dirs_fn is not None:
        project_setup_dirs_fn(args)
    
    return args


def _build_train_stage(args):
    """Build the high-level training stage label.

    Stage definitions:
    - ``PreT``: training from scratch, i.e. no checkpoint is provided.
    - ``FT``: standard fine-tuning from a checkpoint without time-ratio randomization.
    - ``TW``: time-aware fine-tuning from a checkpoint with ``ratio_range`` enabled.
    """
    args.scratch = args.checkpoint is None
    args.ft_train = (args.checkpoint is not None) and (args.ratio_range is None)
    args.tw_train = (args.checkpoint is not None) and (args.ratio_range is not None)

    if args.tw_train:
        stage_name = "TW"
    elif args.ft_train:
        stage_name = "FT"
    elif args.scratch:
        stage_name = "PreT"
    else:
        raise ValueError("Unable to infer train_stage from the current argument combination.")

    return stage_name


def _build_method_name(args):
    parts = []

    # Architecture / algorithm family
    if args.use_grpo:
        parts.append("GRPO")
    elif args.use_cost:
        parts.append("P3O")
    else:
        parts.append("PPO")
    suffix = str(args.method_suffix or "").strip().strip("_")
    suffix_tokens = {token for token in suffix.split("_") if token}
    hard_deadline = str(getattr(args, "temporal_deadline_mode", "off")).lower() == "hard_reset"
    named_temporal_method = bool(args.update_ddl or hard_deadline or "FTTL" in suffix_tokens)

    # Variant methods
    if args.fix_linvel and args.fix_limvel and not named_temporal_method:
        parts.append("FH")
    if args.use_timeawareness:
        parts.append("TC")
    if args.update_ddl:
        parts.append("ATTL_EMA" if args.anneal_ddl else "ATTL")
    if hard_deadline:
        parts.append("HDDL")
    sil_filter_mode = str(getattr(args, "sil_filter_mode", "return")).lower()
    if getattr(args, "use_sil", False) and sil_filter_mode == "temporal":
        parts.append("TFilter")
    elif getattr(args, "use_sil", False) and sil_filter_mode == "both":
        parts.append("RTFilter")

    # Optional architecture modifiers
    if args.use_lstm:
        parts.append("LSTM")

    method_name = "_".join(parts)
    if suffix:
        method_name = f"{method_name}_{suffix}"

    return method_name


def _build_experiment_name(args):
    """Build experiment name based on configuration."""
    additional = f'_{args.task_name}_{args.train_stage}_{args.method_name}'
    
    # Load checkpoint configs
    if args.checkpoint is not None:
        if min(args.epstimeRewardScale) < 0:
            raise ValueError(
                f"epstimeRewardScale must be non-negative when loading a checkpoint, "
                f"got {args.epstimeRewardScale}"
            )
        additional += '_Rcritic' if args.reset_critic else ''
    
    # Algorithm specific
    additional += '_Beta' if args.beta else '_Normal'
    additional += '_VB' if args.value_bootstrap else ''
    if not args.fix_priv:
        if args.time2end:
            additional += '_T2E'
        if args.time_ratio:
            additional += 'ratio'
        if args.explicit_remaining_time_obs:
            additional += '_Tremain'
    
    if args.ratio_range:
        additional += f'_TRatio_{args.ratio_range[0]}to{args.ratio_range[1]}'
    
    # Weight and frequency settings
    additional += f'_gam{args.gamma}'
    if args.use_cost:
        additional += f'_cGam{args.c_gamma[0]}_{args.c_gamma[1]}'
        additional += f'_cScale{args.c_scale[0]}_{args.c_scale[1]}'
    if args.no_dense:
        additional += '_noDense'
    if args.step_cost > 0:
        additional += f'_stepCost{args.step_cost:g}'
        if args.step_cost_mode == 'horizon':
            additional += '_horizon'
    if args.lever_pull_dense_reward_scale != 1.0:
        additional += f'_leverDense{args.lever_pull_dense_reward_scale:g}'
    if args.peg_insert_progress_reward_scale != 1.0:
        additional += f'_pegProgress{args.peg_insert_progress_reward_scale:g}'
    if args.dense_reward_scale != [1.0, 1.0]:
        additional += f'_denseR{args.dense_reward_scale[0]:g}to{args.dense_reward_scale[1]:g}'
    if args.dense_reward_dropout_p > 0:
        additional += f'_denseDrop{args.dense_reward_dropout_p:g}'
        if args.dense_reward_dropout_mode != "step":
            additional += f'_{args.dense_reward_dropout_mode}'
    if args.policy_grad_noise_scale > 0:
        noise_tag = (
            'pGradDirN'
            if args.policy_grad_noise_mode == 'norm_preserved_direction'
            else 'pGradN'
        )
        additional += f'_{noise_tag}{args.policy_grad_noise_scale:g}'
    for tag in getattr(args, "project_run_tags", []):
        additional += f'_{tag}'
    if args.delayed_stress:
        additional += '_delayedStress'
        if args.stress_start_iter >= 0:
            additional += f'_stressIter{args.stress_start_iter}'
        if args.stress_require_all_tasks:
            additional += '_allTasks'
    if args.update_ddl and args.anneal_ddl:
        additional += f'_ddlEMA{args.ddl_anneal_alpha:g}'
    if args.time2end_init_frac != 1.0:
        additional += f'_t2eFrac{args.time2end_init_frac:g}'
    if args.temporal_deadline_mode == "hard_reset":
        additional += f'_HDDLgrace{args.temporal_deadline_grace_frac:g}'
        if args.temporal_deadline_min_seconds > 0:
            additional += f'_minT{args.temporal_deadline_min_seconds:g}'
        if not args.temporal_deadline_terminal:
            additional += '_deadlineTimeout'
    if args.use_sil and args.sil_filter_mode != "return":
        additional += f'_silFilter{args.sil_filter_mode}'
        additional += f'_silTModel{args.sil_temporal_model}'
        if args.sil_temporal_target_mode != "mc":
            additional += f'_silTTarget{args.sil_temporal_target_mode}'
            if args.sil_temporal_target_mode == "td_lambda":
                additional += f'{args.sil_temporal_lambda:g}'
    if args.double_value_critic:
        double_value_tags = {
            "minimum": "DoubleVMin",
            "primary": "DoubleVPrimary",
            "timeout_minimum": "DoubleVTimeoutMin",
        }
        additional += f'_{double_value_tags[args.double_value_mode]}'
    if args.use_sil and args.sil_actor_weight_coef > 0:
        additional += f'_silActorW{args.sil_actor_weight_coef:g}'
        if args.sil_actor_weight_mode != "multiply":
            additional += f'_{args.sil_actor_weight_mode}'
            if args.sil_actor_gap_alpha > 0:
                additional += f'Alpha{args.sil_actor_gap_alpha:g}'
        if args.sil_actor_temporal_margin > 0:
            additional += f'_actorMargin{args.sil_actor_temporal_margin:g}'
    if args.use_sil and args.sil_success_actor_floor > 0:
        additional += f'_silSucFloor{args.sil_success_actor_floor:g}'
    if args.use_sil and args.sil_aligned_success_first:
        additional += '_silSuccessFirst'
    if args.use_sil and args.sil_success_actor_min_share > 0:
        additional += (
            f'_silSucShare{args.sil_success_actor_min_share:g}'
            f'Cap{args.sil_success_actor_boost_cap:g}'
        )
    if args.use_sil and args.sil_fast_success_retention_coef > 0:
        additional += (
            f'_silFastRet{args.sil_fast_success_retention_coef:g}'
            f'T{args.sil_fast_success_retention_temperature:g}'
        )
    if args.use_sil and args.sil_success_weight_mode == "fixed":
        additional += f'_silWfixed{args.sil_fixed_success_weight:g}'
    
    # Robot control
    additional += f"_{args.control_type.upper()}"
    if args.control_freq_inv > 0:
        if 1 / args.dt % args.control_freq_inv != 0:
            raise ValueError(
                f"control_freq_inv={args.control_freq_inv} must evenly divide 1/dt={1 / args.dt}"
            )
        control_freq = int(1 / args.dt // args.control_freq_inv)
        additional += f'_Ctrl{control_freq}Hz'
    if args.gripper_freq_inv > 0:
        if 1 / args.dt / args.control_freq_inv % args.gripper_freq_inv != 0:
            raise ValueError(
                f"gripper_freq_inv={args.gripper_freq_inv} must evenly divide "
                f"control frequency {1 / args.dt / args.control_freq_inv}"
            )
        gripper_freq = int(1 / args.dt / args.control_freq_inv // args.gripper_freq_inv)
        additional += f'_Grip{gripper_freq}Hz'
    
    # Time aware constraints
    if args.max_vel_subtract > 0:
        additional += f'_maxVel{1-args.max_vel_subtract:.1f}'
    if args.limit_gripper_vel:
        additional += f'_LimGrip'
    if max(args.epstimeRewardScale) > 0:
        additional += f'_epsT{args.epstimeRewardScale[0]}to{args.epstimeRewardScale[1]}' \
              if args.epstimeRewardScale[0] != args.epstimeRewardScale[1] else f'_epsT{args.epstimeRewardScale[0]}'
    if args.ratio_range:
        if args.fix_linvel:
            additional += '_NoLinVel'
        if args.fix_limvel:
            additional += '_NoLimGt'
        if max(args.scevelRewardScale) > 0 and not args.use_cost:
            additional += f'_sceVel{args.scevelRewardScale[0]}to{args.scevelRewardScale[1]}'
        if args.sceaccRewardScale > 0:
            additional += f'_sceAcc{args.sceaccRewardScale}'
        if args.scevelSchedule < 1:
            additional += f'_sceMul{args.scevelSchedule}'
        if args.exp_scheduler:
            if args.scevelSchedule < 1:
                raise ValueError(
                    f"exp_scheduler requires scevelSchedule >= 1, got {args.scevelSchedule}"
                )
            additional += f'_sceExp{args.scevelSchedule}'
        if args.actvelPenaltyScale > 0:
            additional += f'_actVel{args.actvelPenaltyScale}'
        if args.actaccPenaltyScale > 0:
            additional += f'_actAcc{args.actaccPenaltyScale}'
        if args.curri_rate > 1:
            additional += f'_curr{args.curri_rate}'
    
    additional += f'_step{args.num_steps}'
    additional += f'_seq{args.seq_length}'
    additional += f'_ent{args.ent_coef[0]}' if args.ent_coef[0] == args.ent_coef[1] else f'_entropy_{args.ent_coef[0]}to{args.ent_coef[1]}'
    additional += f'_seed{args.seed}'
    
    args.timer = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.random_policy:
        args.run_name = args.timer + additional.replace('-train', '-random_policy')
    elif args.force_name:
        args.run_name = args.force_name + args.timer
    else:
        args.run_name = args.timer + additional


def _load_checkpoint_config(args):
    """Load configuration from checkpoint."""
    ckpt_task_name = args.ckpt_task
    args.ckpt_dir = resolve_checkpoint_run_dir(args.result_dir, ckpt_task_name, args.checkpoint)
    if args.ckpt_task is None:
        args.ckpt_task = infer_task_name_from_run_dir(args.result_dir, args.ckpt_dir)
    ckpt_json_file_path = os.path.join(args.ckpt_dir, 'config.json')
    with open(ckpt_json_file_path, 'r') as json_obj:
        ckpt_json = json.load(json_obj)

    synced_keys = [
        # Agent related
        "beta",
        "use_lstm",
        "use_grpo",
        "activation",
        "use_layernorm",
        "hidden_size",
        "lstm_hidden_size",
        
        # Normalization related
        "norm_obs",
        "norm_rew",
        "reward_normalizer_mode",
        "norm_value",
        "norm_cost",
        "pertask_norm",
        "pertask_norm_adv",
        "task_embedding_dim",
        "explicit_remaining_time_obs",
        "double_value_critic",
        "double_value_mode",
        
        # Task related
        "control_freq_inv",
        "gripper_freq_inv",
        "max_vel_subtract",
        "limit_gripper_vel",
        "fix_linvel",
        "fix_limvel",
    ]
    for key in synced_keys:
        if key in ckpt_json:
            setattr(args, key, ckpt_json[key])

    args.par_checkpoint = ckpt_json["checkpoint"]
    args.par_index_episode = ckpt_json["index_episode"]
    ckpt_control_type = ckpt_json.get("control_type", args.control_type)
    if args.control_type != ckpt_control_type:
        raise ValueError(
            f"Checkpoint control_type {ckpt_control_type!r} does not match requested "
            f"control_type {args.control_type!r}"
        )


def _setup_directories(args):
    """Setup result directories."""
    if (
        args.result_benchmark_name
        and args.result_experiment_name
        and args.result_training_stage_name
        and args.result_task_dir_name
    ):
        args.result_dir = os.path.join(
            args.result_dir,
            args.result_benchmark_name,
            args.result_experiment_name,
            args.result_training_stage_name,
            args.result_task_dir_name,
            args.method_name,
            args.script_time,
        )
    elif args.result_benchmark_name and args.result_task_dir_name:
        args.result_dir = os.path.join(
            args.result_dir,
            args.train_stage,
            args.result_benchmark_name,
            args.result_task_dir_name,
            args.method_name,
            args.script_time,
        )
    else:
        args.result_dir = os.path.join(
            args.result_dir,
            args.task_name,
            args.train_stage,
            args.method_name,
            args.script_time,
        )
    args.instance_dir = os.path.join(args.result_dir, args.run_name)
    args.checkpoint_dir = os.path.join(args.instance_dir, 'checkpoints')
    args.trajectory_dir = os.path.join(args.instance_dir, 'trajectories')
    args.csv_file_path = os.path.join(args.instance_dir, 'data.csv')
    args.json_file_path = os.path.join(args.instance_dir, 'config.json')
    
    if args.saving:
        os.makedirs(args.result_dir, exist_ok=True)
        os.makedirs(args.instance_dir, exist_ok=False)
        os.makedirs(args.checkpoint_dir, exist_ok=False)
        os.makedirs(args.trajectory_dir, exist_ok=False)
