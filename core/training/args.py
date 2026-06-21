import os
import argparse
from dataclasses import dataclass, field
from simple_parsing import ArgumentParser
import datetime
import json
import psutil
from distutils.util import strtobool
from math import ceil
from typing import List, Optional
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
    update_ddl: bool = False
    anneal_ddl: bool = False
    ddl_anneal_alpha: float = 0.2
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
    reward_scale: float = 0.1
    step_cost: float = 0.0
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
    value_bootstrap: bool = True
    use_grpo: bool = False  # Use GRPO (reward-to-go advantages, no critic) instead of PPO (GAE)
    grpo_use_clipping: bool = True  # Use PPO-style clipping in GRPO actor loss (set False for pure REINFORCE)
    policy_grad_noise_scale: float = 0.0
    
    bounds_loss_coef: float = 0.0001
    deterministic: bool = False
    gamma: float = 0.995
    c_gamma: List[float] = field(default_factory=lambda: [1, 0.99])
    c_scale: List[float] = field(default_factory=lambda: [0, 1])
    lr: float = 5e-4
    seed: int = 123456

    lstm_hidden_size: int = 256
    hidden_size: List[int] = field(default_factory=lambda: [512, 256, 128, 64, 32])
    task_embedding_dim: int = 0  # 0 = auto-set to len(tasks) in parse_args; set explicitly to override

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
    dense_tail_frac: float = 0.1
    
    # SIL replay
    use_sil: bool = False
    sil_train: bool = True
    sil_mode: str = "sil"  # sil | bc
    sil_source: str = "fastest"  # fastest | return
    sil_topk: int = 512
    sil_coef: float = 0.1
    sil_vf_coef: float = 0.05
    sil_batch_size: int = 2048
    sil_sample_unit: str = "transition"  # transition | trajectory
    sil_start_iter: int = 0
    sil_normalize_gap: bool = False
    sil_update_interval: int = 1
    sil_separate_updates: int = 0  # 0 = joint PPO+SIL; >0 = PPO first, then this many SIL-only updates
    sil_success_sample_frac: float = -1.0  # <0 keeps legacy per-env fast-success sampling; [0,1] uses global success/fallback mix
    sil_speed_priority_min: float = 1.0
    sil_speed_priority_max: float = 2.0
    sil_analysis_interval: int = 0
    sil_analysis_batch_size: int = 1024
    sil_landscape_grid: int = 9
    sil_landscape_span: float = 0.05

    # PyTorch specific arguments
    cuda: bool = True
    cpus: List[int] = field(default_factory=list)
    torch_deterministic: bool = True


def parse_args(project_name: str = "TSIL", argv: List[str] = None):
    parser = ArgumentParser(description='Time-aware Policy Learning')
    parser.add_arguments(Args, dest="args")
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
    if args.task_embedding_dim == 0:
        args.task_embedding_dim = len(args.tasks)
    
    if args.cpus:
        print('Running on specific CPUS:', args.cpus)
        process = psutil.Process()
        process.cpu_affinity(args.cpus)
    
    if args.use_timeawareness:
        if args.termination_on_success is not True:
            raise ValueError("Time-awareness requires termination_on_success to be True")
    if not 0.0 < args.ddl_anneal_alpha <= 1.0:
        raise ValueError(f"ddl_anneal_alpha must be in (0, 1], got {args.ddl_anneal_alpha}")
    if not 0.0 < args.dense_tail_frac <= 1.0:
        raise ValueError(f"dense_tail_frac must be in (0, 1], got {args.dense_tail_frac}")
    if args.signal_metrics_window is None:
        args.signal_metrics_window = args.running_len
    if args.signal_metrics_window <= 0:
        raise ValueError(f"signal_metrics_window must be positive, got {args.signal_metrics_window}")
    if args.step_cost < 0:
        raise ValueError(f"step_cost must be non-negative, got {args.step_cost}")
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
    if args.sil_start_iter < 0:
        raise ValueError(f"sil_start_iter must be non-negative, got {args.sil_start_iter}")
    if args.sil_update_interval <= 0:
        raise ValueError(f"sil_update_interval must be positive, got {args.sil_update_interval}")
    if args.sil_separate_updates < 0:
        raise ValueError(f"sil_separate_updates must be non-negative, got {args.sil_separate_updates}")
    if args.sil_analysis_interval < 0:
        raise ValueError(f"sil_analysis_interval must be non-negative, got {args.sil_analysis_interval}")
    if args.sil_analysis_batch_size <= 0:
        raise ValueError(f"sil_analysis_batch_size must be positive, got {args.sil_analysis_batch_size}")
    if args.sil_landscape_grid < 3:
        raise ValueError(f"sil_landscape_grid must be at least 3, got {args.sil_landscape_grid}")
    if args.sil_landscape_span <= 0:
        raise ValueError(f"sil_landscape_span must be positive, got {args.sil_landscape_span}")
    if args.sil_success_sample_frac > 1.0:
        raise ValueError(f"sil_success_sample_frac must be <= 1.0, got {args.sil_success_sample_frac}")
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
    named_temporal_method = bool(args.update_ddl or "FTTL" in suffix_tokens)

    # Variant methods
    if args.fix_linvel and args.fix_limvel and not named_temporal_method:
        parts.append("FH")
    if args.use_timeawareness:
        parts.append("TC")
    if args.update_ddl:
        parts.append("ATTL_EMA" if args.anneal_ddl else "ATTL")

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
    if args.dense_reward_scale != [1.0, 1.0]:
        additional += f'_denseR{args.dense_reward_scale[0]:g}to{args.dense_reward_scale[1]:g}'
    if args.dense_reward_dropout_p > 0:
        additional += f'_denseDrop{args.dense_reward_dropout_p:g}'
        if args.dense_reward_dropout_mode != "step":
            additional += f'_{args.dense_reward_dropout_mode}'
    if args.policy_grad_noise_scale > 0:
        additional += f'_pGradN{args.policy_grad_noise_scale:g}'
    if args.delayed_stress:
        additional += '_delayedStress'
        if args.stress_start_iter >= 0:
            additional += f'_stressIter{args.stress_start_iter}'
        if args.stress_require_all_tasks:
            additional += '_allTasks'
    if args.update_ddl and args.anneal_ddl:
        additional += f'_ddlEMA{args.ddl_anneal_alpha:g}'
    
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
        "norm_value",
        "norm_cost",
        "pertask_norm",
        "pertask_norm_adv",
        "task_embedding_dim",
        
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
