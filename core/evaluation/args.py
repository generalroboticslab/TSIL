
import os
from dataclasses import dataclass, field
from simple_parsing import ArgumentParser
from typing import List
from core.common.checkpointing import (
    check_file_exist,
    infer_task_name_from_run_dir,
    resolve_checkpoint_run_dir,
)
from core.common.io import read_json
from core.common.project_paths import normalize_project_result_root
import numpy as np


@dataclass
class EvalArgs:
    """Arguments for evaluating trained models.
    
    Organization:
    - Evaluation-only parameters: Used only during evaluation
    - Override parameters: Override training config values (None = use training default)
    """
    
    # ==================== Evaluation-Only Parameters ====================
    # These are specific to evaluation and don't exist in training config
    
    # Basic evaluation setup
    checkpoint: str = None  # Required unless random_policy/heuristic_policy
    index_episode: str = "last"
    result_dir: str = "results"  # expands to results/<project>/train_res
    eval_dir: str = "results"  # expands to results/<project>/eval_res
    task_name: str = None  # Inferred from checkpoint if not provided
    
    # Evaluation environment
    num_envs: int = 10
    sim_device: str = "cuda:0"
    rendering: bool = False
    graphics_device_id: int = 0
    
    # Evaluation control
    seed: int = 123456
    target_episodes: int = 20000
    target_success_eps: int = None
    warmup_episodes: int = None  # Default: num_envs * 5
    deterministic: bool = True
    strict_eval: bool = False
    saving: bool = False
    eval_result: bool = True
    
    # Goal speed evaluation
    goal_speed: float = None  # Single speed to evaluate
    goal_ratio_range: List[float] = field(default_factory=list)  # Range [start, end, step]
    goal_time: float = None  # Fixed goal time instead of speed
    budget_portion: List[float] = None
    speed_describe: List[str] = field(default_factory=list)
    act_scale: float = 1.0
    
    # Special evaluation modes
    random_policy: bool = False
    heuristic_policy: bool = False
    use_par_checkpoint: bool = False
    
    # Visualization
    quiet: bool = True
    realtime: bool = False
    inspect_trajectory: bool = False
    debug_replay_first_step: bool = False
    record_eval_trajectories: bool = False
    eval_trajectory_samples: int = 5
    eval_trajectory_success_only: bool = False
    eval_trajectory_max_attempts: int = 0
    eval_trajectory_post_success_steps: int = 0
    eval_trajectory_stop_arm_after_success: bool = False
    eval_trajectory_same_config: bool = False
    eval_trajectory_start_snapshot: str = None
    trajectory_bucket: str = "topk"
    trajectory_task_id: int = None
    trajectory_rank: int = 1
    trajectory_cycles: int = 2
    trajectory_source_dir: str = None
    trajectory_output_dir: str = None
    trajectory_record_video: bool = False
    trajectory_video_fps: int = None
    trajectory_play_speed: float = 1.0
    trajectory_export_frames: bool = False
    trajectory_frame_count: int = 6
    trajectory_include_initial: bool = True
    trajectory_camera_preset: str = "default"
    trajectory_camera_fov: float = None
    trajectory_clean_background: bool = False
    trajectory_force_cpu_capture: bool = None
    trajectory_graphics_device_id: int = None
    record_videos: bool = False
    
    # Baseline specific
    interpolate_joints: int = 1
    
    # ==================== Override Parameters ====================
    # These override values from training config (None = use training default)
    
    # Environment overrides
    episodeLength: int = None  # Override episode length
    gripper_freq_inv: int = None  # Override gripper control interval
    away_dist: float = None  # Override away distance
    specific_idx: int = None  # Specific config index to use
    fixed_configs: bool = None  # Override fixed configs
    global_configs: bool = None  # Override global configs
    par_configs: bool = None  # Override par configs
    apply_noise: bool = None  # Override noise application (default True)
    init_curri_ratio: float = None  # Override curriculum ratio
    termination_on_success: bool = None  # Override termination behavior
    scale_actions: bool = None  # Override action scaling
    
    # Multi-task overrides
    tasks: List[int] = None  # Override task list
    task_counts: List[int] = None  # Override task counts
    env_name: str = None  # Override environment name


# Default training config values (used when no checkpoint is provided)
DEFAULT_TRAINING_CONFIG = {
    "cuda": True,
    "torch_deterministic": True,
    "beta": True,
    "use_lstm": False,
    "time2end": True,
    "time_ratio": False,
    "fix_priv": False,
    "hidden_size": [256, 128, 64],
    "norm_obs": True,
    "norm_rew": True,
    "norm_value": True,
    "control_type": "ik",
    "control_freq_inv": 1,
    "gripper_freq_inv": 1,
    "max_vel_subtract": 0.0,
    "limit_gripper_vel": False,
    "num_gms": 1,
    "fixed_configs": False,
    "global_configs": False,
    "par_configs": False,
    "specific_idx": None,
    "episodeLength": 300,
    "away_dist": 0.1,
    "act_scale": 1.0,
    "apply_noise": True,
    "init_curri_ratio": 1.0,
    "termination_on_success": True,
    "scale_actions": False,
    "tasks": [0],
    "task_counts": [10],
    "env_name": "MetaWorldV2",
}


def get_args(project_name: str = "TSIL"):
    parser = ArgumentParser(description='Evaluate the Trained Model')
    parser.add_arguments(EvalArgs, dest="args")
    args = parser.parse_args().args
    args.project_name = project_name
    args.result_dir = normalize_project_result_root(args.result_dir, project_name, "train_res")
    args.eval_dir = normalize_project_result_root(args.eval_dir, project_name, "eval_res")
    
    # Validate required arguments
    if args.checkpoint is None and not args.random_policy and not args.heuristic_policy:
        raise ValueError("--checkpoint is required unless using --random_policy or --heuristic_policy")
    
    _process_args(args)
    _setup_directories(args)
    
    return args


def _process_args(args):
    """Process and validate arguments.
    
    Flow:
    1. Load training config (from checkpoint or defaults)
    2. Apply eval overrides (args with non-None values override training config)
    3. Validate and build derived values
    """
    
    # Handle random/heuristic policy case (no checkpoint needed)
    if args.random_policy or args.heuristic_policy:
        if args.task_name is None:
            raise ValueError("--task_name is required when using --random_policy or --heuristic_policy")
        args.checkpoint_path = None
        args.json_file_path = None
        training_config = DEFAULT_TRAINING_CONFIG.copy()
        args.run_name = "random_policy" if args.random_policy else "heuristic_policy"
    else:
        checkpoint_folder = resolve_checkpoint_run_dir(args.result_dir, args.task_name, args.checkpoint)
        args.checkpoint_run_dir = checkpoint_folder
        if args.task_name is None:
            args.task_name = infer_task_name_from_run_dir(args.result_dir, checkpoint_folder)

        args.json_file_path = os.path.join(checkpoint_folder, 'config.json')
        args.meta_path = os.path.join(checkpoint_folder, "trajectories", "meta_data.json")
        args.checkpoint_path = os.path.join(checkpoint_folder, 'checkpoints', 'eps_' + args.index_episode)
        if not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint path {args.checkpoint_path} does not exist")
        
        training_config = read_json(args.json_file_path)
        
        # Handle parent checkpoint if needed
        if args.use_par_checkpoint:
            parent_checkpoint = training_config.get("checkpoint")
            if parent_checkpoint is None:
                raise ValueError("use_par_checkpoint requires the training config to include a parent checkpoint")
            par_task_name = training_config.get("ckpt_task", None)
            par_checkpoint_folder = resolve_checkpoint_run_dir(
                args.result_dir, par_task_name, parent_checkpoint
            )
            args.checkpoint_run_dir = par_checkpoint_folder
            par_json_file_path = os.path.join(par_checkpoint_folder, 'config.json')
            args.checkpoint_path = os.path.join(par_checkpoint_folder, 'checkpoints', 'eps_' + training_config["index_episode"])
            args.run_name = training_config["run_name"]
            training_config = read_json(par_json_file_path)
    
    # Store original eval args before loading training config
    eval_overrides = {k: v for k, v in args.__dict__.items() if v is not None}
    # Apply training config to args
    args.__dict__.update(training_config)
    # Re-apply eval overrides (these take precedence over training config)
    args.__dict__.update(eval_overrides)
    
    # Set graphics device based on viewer rendering or headless camera capture
    needs_offscreen_graphics = (
        (args.inspect_trajectory or args.record_eval_trajectories)
        and (args.trajectory_record_video or args.trajectory_export_frames)
    )
    if args.rendering:
        args.graphics_device_id = 2
    elif needs_offscreen_graphics:
        args.graphics_device_id = 2 if args.trajectory_graphics_device_id is None else args.trajectory_graphics_device_id
    else:
        args.graphics_device_id = -1
    
    # Set warmup_episodes default (depends on num_envs, so can't be in dataclass)
    if args.warmup_episodes is None:
        args.warmup_episodes = args.num_envs * 5
    
    # Validate and process argument relationships
    _validate_and_process(args, training_config)
    
    # Build evaluation lists
    _build_evaluation_lists(args)
    
    # Validate constraints
    _validate_constraints(args)
    
    # Build experiment name
    _build_experiment_name(args)


def _validate_and_process(args, training_config):
    """Validate argument relationships and process special cases."""
    if args.inspect_trajectory or args.debug_replay_first_step or args.record_eval_trajectories:
        if args.debug_replay_first_step:
            args.trajectory_record_video = False
        elif args.trajectory_record_video or args.trajectory_export_frames:
            args.trajectory_record_video = True
            args.record_videos = True
        if args.trajectory_task_id is not None:
            args.tasks = [int(args.trajectory_task_id)]
        else:
            args.tasks = [int(task_id) for task_id in args.tasks]
        args.task_counts = [1] * len(args.tasks)
        args.num_envs = len(args.tasks)
        args.strict_eval = False

    # Specific index implies fixed configs
    if args.specific_idx is not None:
        args.fixed_configs = True

    # Par configs mode
    if args.par_configs:
        args.par_configs = True
        if not args.global_configs:
            if not args.fixed_configs:
                raise ValueError("par_configs requires fixed_configs when not using global_configs")
            if training_config.get("checkpoint") is None:
                raise ValueError("Par configs evaluation requires a parent checkpoint with fixed configs")
            args.par_checkpoint = training_config["checkpoint"]
            args.par_index_episode = training_config["index_episode"]
        else:
            args.fixed_configs = True
    
    # Strict eval validation
    if args.strict_eval:
        if args.num_envs != args.target_success_eps:
            raise ValueError(
                f"strict_eval requires num_envs ({args.num_envs}) == "
                f"target_success_eps ({args.target_success_eps})"
            )
    

def _build_evaluation_lists(args):
    """Build evaluation parameter lists."""
    # Goal speed list
    args.goal_speed_lst = [1]
    if len(args.goal_ratio_range) != 0:
        if len(args.goal_ratio_range) != 3:
            raise ValueError(
                f"goal_ratio_range must have three values: start, stop, step; got {args.goal_ratio_range}"
            )
        max_ratio = args.goal_ratio_range[1]
        args.goal_speed_lst = np.arange(*args.goal_ratio_range).tolist()
        args.goal_speed_lst += [max_ratio] if max_ratio not in args.goal_speed_lst else []
    args.goal_speed_lst = [args.goal_speed] if args.goal_speed is not None else args.goal_speed_lst


def _validate_constraints(args):
    """Validate argument constraints."""
    if args.goal_time is not None:
        if args.goal_speed is not None or args.goal_ratio_range != []:
            raise ValueError("goal_time cannot be combined with goal_speed or goal_ratio_range")
    
    if args.budget_portion is not None:
        if args.goal_time is None and args.goal_speed is None:
            raise ValueError("budget_portion requires either goal_time or goal_speed")
        if not np.allclose(sum(args.budget_portion), 1):
            raise ValueError(f"budget_portion must sum to 1, got {args.budget_portion}")
        if len(args.speed_describe) != len(args.budget_portion):
            raise ValueError(
                "speed_describe must have the same length as budget_portion, got "
                f"{len(args.speed_describe)} and {len(args.budget_portion)}"
            )
    
def _build_experiment_name(args):
    """Build experiment name based on configuration."""
    eval_config = ''
    
    if args.random_policy:
        args.run_name = f'EVAL_RandPolicy'
    elif args.heuristic_policy:
        args.run_name = f'EVAL_HeurPolicy'
    else:
        eval_config += '_EVAL_' + args.index_episode
    
    if args.interpolate_joints != 1:
        eval_config += f'_Intp{args.interpolate_joints}'
    if args.goal_time is not None:
        eval_config += f'_RT{args.goal_time}'
    if args.specific_idx:
        eval_config += f'_Idx{args.specific_idx}'
    
    if args.budget_portion is not None:
        eval_config += f'_Staged'
    
    temp_filename = args.run_name + eval_config
    
    maximum_name_len = 250
    if len(temp_filename) > maximum_name_len:
        shorten_name_range = len(temp_filename) - maximum_name_len
        args.run_name = args.run_name[:-shorten_name_range]
    args.run_name = args.run_name + eval_config
    
    print('Uniform name is:', args.run_name)


def _setup_directories(args):
    """Setup result directories."""
    trajectory_mode = (
        args.inspect_trajectory or args.debug_replay_first_step or args.record_eval_trajectories
    )
    if trajectory_mode and getattr(args, "checkpoint_run_dir", None) is not None:
        args.save_dir = args.checkpoint_run_dir
        args.instance_dir = args.checkpoint_run_dir
        args.trajectory_dir = os.path.join(args.instance_dir, 'traj_vis')
    else:
        args.save_dir = os.path.join(args.eval_dir, args.task_name)
        args.instance_dir = os.path.join(args.save_dir, args.run_name)
        args.trajectory_dir = os.path.join(args.instance_dir, 'trajectories')
    if trajectory_mode and getattr(args, "trajectory_output_dir", None):
        args.trajectory_dir = args.trajectory_output_dir
    args.csv_file_path = os.path.join(args.instance_dir, 'data.csv')
    args.json_file_path = os.path.join(args.instance_dir, 'config.json')
    
    if args.saving:
        check_file_exist(args.csv_file_path)
        check_file_exist(args.trajectory_dir)
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.instance_dir, exist_ok=True)
        os.makedirs(args.trajectory_dir, exist_ok=True)
    elif trajectory_mode:
        os.makedirs(args.trajectory_dir, exist_ok=True)
