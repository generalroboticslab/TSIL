import json
import os
import random
import sys
import time
import traceback
import numpy as np
try:
    import isaacgym  # Must precede torch when installed.
except ModuleNotFoundError:
    isaacgym = None
import torch
import torch.optim as optim
from tabulate import tabulate
import threadpoolctl as tpc
import multiprocessing

from MTBench import isaacgymenvs  # Registers shared Hydra resolvers without creating a simulator.
from core.agents.agent import get_agent
from core.agents.utils import (
    AdaptiveScheduler,
    LinearScheduler,
    PerTaskRewardNormalizer,
    NormalizeReward,
    load_checkpoint,
    save_checkpoint,
)
from core.common.task_layout import build_task_id_lookup
from core.agents.reward_normalization import PerEnvironmentReturnRewardNormalizer
from core.training.args import parse_args
from core.training.algo import PolicyUpdater
from core.training.algo.tsil.loss import TsilReplayLoss
from core.training.rollout import RolloutCollector
from core.training.algo.tsil.memory import TsilTrajectoryMemory
from core.training.metrics import MetricTracker
from core.training.storage import RolloutStorage
from core.training.logger import TrainingLogger
from core.training.causal_diagnostics import CausalShadowDiagnostics
from core.training.update_events import UpdateEventDiagnostics
from hydra import compose, initialize, initialize_config_dir
from omegaconf import OmegaConf


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HYDRA_CONFIG_DIR = os.path.join(REPO_ROOT, "MTBench", "isaacgymenvs", "cfg")


class OnPolicyTrainer:
    """On-policy trainer loop for PPO/GRPO-style updates with optional SIL."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.tensor_dtype = torch.float32
        
        # Initialize core state
        self._setup_seeding()
        self._setup_environment()
        self._stress_active = None
        self._set_effective_stress(active=not bool(getattr(args, "delayed_stress", False)))
        self._setup_agent()
        self._setup_optimizer()

        # Create component: RolloutStorage
        obs_shape = self.envs.obs_space.shape
        state_shape = self.envs.state_space.shape
        act_shape = self.envs.act_space.shape
        self.storage = RolloutStorage(
            num_steps=args.num_steps,
            num_envs=args.num_envs,
            obs_shape=obs_shape,
            state_shape=state_shape,
            act_shape=act_shape,
            num_cost=args.num_cost,
            device=self.device,
            dtype=self.tensor_dtype,
            use_lstm=args.use_lstm,
            rollout_agent=self.rollout_agent,
        )
        self.storage.init_from_env_reset(self.envs)

        # Create component: RolloutCollector
        self.rollout_collector = RolloutCollector(
            envs=self.envs,
            args=args,
            device=self.device,
            dtype=self.tensor_dtype,
            task_indices=self.task_indices,
        )

        # Create component: PolicyUpdater
        self.policy_updater = self.policy_updater_class()(
            args=args,
            device=self.device,
            dtype=self.tensor_dtype,
        )

        # Setup normalizers
        self._setup_normalizers()

        # Create component: MetricTracker
        reward_settings = getattr(self.envs, "reward_settings", {}) or {}
        self.metrics = MetricTracker(
            args=args,
            num_tasks=self.num_tasks,
            unique_task_ids=self.unique_task_ids,
            device=self.device,
            dtype=self.tensor_dtype,
            max_eps_time=self.envs.max_eps_time,
            avg_time2end_upper=self.envs._get_avg_time2end_upper(),
            reward_settings=reward_settings,
        )
        self.envs.cfg['r_epstime_scale'] = args.epstimeRewardScale[0]
        self.envs.cfg['r_scene_vel_scale'] = args.scevelRewardScale[0]

        # Project hooks
        self.setup_project_training()
        self.after_tracking_setup()

        # Create component: TsilTrajectoryMemory
        self.tsil_memory = TsilTrajectoryMemory(
            args=args,
            unique_task_ids=self.unique_task_ids,
            device=self.device,
            envs=self.envs,
            storage=self.storage,
        )

        # Create component: TsilReplayLoss
        self.tsil_replay_loss = TsilReplayLoss(args=args, device=self.device, dtype=self.tensor_dtype)

        # Create component: TrainingLogger
        self.training_logger = TrainingLogger(args=args)
        self.causal_shadow_diagnostics = CausalShadowDiagnostics(args)
        self.update_event_diagnostics = UpdateEventDiagnostics(args)

        self._pending_ddl_updates = []
        
        self._print_configuration()

    def setup_project_training(self):
        """Project hook for training-specific setup."""

    def policy_updater_class(self):
        """Return the updater implementation used by this trainer."""
        return PolicyUpdater

    def after_tracking_setup(self):
        """Project hook after shared metric/tracking setup."""

    def after_rollout_complete(self, *args, **kwargs):
        """Project hook after the rollout is fully collected."""

    def before_policy_update(self, *args, **kwargs):
        """Project hook right before the optimizer/update step."""

    def after_policy_update(self, *args, **kwargs):
        """Project hook right after the optimizer/update step."""

    def after_training_complete(self):
        """Project hook after the final checkpoint and before environment close."""

    def before_checkpoint_save(self):
        """Project hook before checkpoint/meta persistence."""

    def save_project_meta(self):
        """Project hook for metadata persistence."""
        self.metrics.save_meta_data_snapshot(self.tsil_memory)

    def _stress_gate_active(self):
        if not bool(getattr(self.args, "delayed_stress", False)):
            return True

        fixed_iter = int(getattr(self.args, "stress_start_iter", -1))
        if fixed_iter >= 0 and self.metrics.global_update_iter >= fixed_iter:
            return True

        if not bool(getattr(self.args, "stress_start_on_sil_success", True)):
            return False

        if bool(getattr(self.args, "stress_require_all_tasks", False)):
            return self.tsil_memory.num_tasks_with_success_archive() >= self.num_tasks

        summary = self.tsil_memory.fastest_reference_summary()
        return int(summary.get("count", 0) or 0) > 0

    def _set_effective_stress(self, active):
        active = bool(active)
        self.args.stress_active = active
        self.args.effective_dense_reward_dropout_p = (
            float(getattr(self.args, "dense_reward_dropout_p", 0.0)) if active else 0.0
        )
        self.args.effective_policy_grad_noise_scale = (
            float(getattr(self.args, "policy_grad_noise_scale", 0.0)) if active else 0.0
        )
        self._stress_active = active

    def _update_stress_gate(self):
        self._set_effective_stress(self._stress_gate_active())
    
    
    def _setup_environment(self):
        """Initialize Isaac Gym environment."""
        self.args.graphics_device_id = 2 if self.args.rendering else -1
        from MTBench import isaacgymenvs

        self.envs = isaacgymenvs.make(
            seed=self.args.seed,
            task=self.args.env_name,
            num_envs=self.args.num_envs,
            sim_device=self.args.sim_device,
            rl_device=self.args.sim_device,
            graphics_device_id=self.args.graphics_device_id,
            headless=self.args.graphics_device_id == -1,
            force_render=self.args.rendering,
            custom_args=self.args,
            tasks=self.args.tasks,
            taskEnvCount=self.args.task_counts,
            cfg=self.args
        )

        # task_indices is a tensor of shape [num_envs] mapping each env to its task_id
        if hasattr(self.envs, 'extras') and 'task_indices' in self.envs.extras:
            self.task_indices = self.envs.extras['task_indices'].to(self.device)
        elif hasattr(self.envs, 'task_idx') and hasattr(self.envs, 'task_env_count'):
            task_indices = sum([[tid] * count for tid, count in zip(self.envs.task_idx, self.envs.task_env_count)], [])
            self.task_indices = torch.tensor(task_indices, dtype=torch.long, device=self.device)
        else:
            # Single-task case
            self.task_indices = torch.zeros(self.args.num_envs, dtype=torch.long, device=self.device)

        # Per-task tracking - uses batch operations for efficiency
        self.unique_task_ids = self.envs.unique_task_ids = torch.unique(self.task_indices)
        self.num_tasks = len(self.unique_task_ids)
        self.is_multi_task = self.num_tasks > 1
        
        # Map task_id to row index (0, 3, 7, ...) -> (0, 1, 2, ...) for efficient tensor lookup
        self.tid_to_tidx = build_task_id_lookup(self.unique_task_ids, device=self.device)

    
    def _setup_seeding(self):
        """Set random seeds for reproducibility."""
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic
    
    
    def _setup_agent(self):
        """Initialize agent and load checkpoint if provided."""
        self.agent = self.rollout_agent = get_agent(self.envs, self.args, self.device)
        
        if self.args.checkpoint is not None:
            checkpoint_folder = os.path.join(self.args.ckpt_dir, "checkpoints")
            self.args.checkpoint_path = os.path.join(checkpoint_folder, f"eps_{self.args.index_episode}")
            if not os.path.exists(self.args.checkpoint_path):
                raise FileNotFoundError(f"Checkpoint path {self.args.checkpoint_path} does not exist")
            self.rollout_agent.load_checkpoint(self.args.checkpoint_path, map_location=self.device, reset_critic=self.args.reset_critic)
            
        self.agent.set_mode('train')
    
    
    def _setup_optimizer(self):
        """Initialize optimizer and learning rate scheduler."""
        params = self.agent.ppo_parameters() if hasattr(self.agent, "ppo_parameters") else self.agent.parameters()
        self.optimizer = optim.Adam(params, lr=self.args.lr, eps=1e-8)
        
        if self.args.scheduler == 'adapt':
            self.lr_scheduler = AdaptiveScheduler(kl_threshold=1.6e-2)
        elif self.args.scheduler == 'linear':
            self.lr_scheduler = LinearScheduler(start_lr=self.args.lr, max_steps=self.args.total_timesteps)
        else:
            raise NotImplementedError(f"Scheduler {self.args.scheduler} is not implemented")
    
    
    def _setup_normalizers(self):
        """Initialize reward and cost normalizers."""
        # Each environment keeps its own reward history; scale is shared by task.
        self.reward_normalizer = None
        if self.args.norm_rew:
            self.reward_normalizer = self._make_reward_normalizer().to(self.device)
            # Load checkpoint if available
            if self.args.checkpoint is not None and not self.args.reset_critic:
                checkpoint_folder = os.path.join(self.args.ckpt_dir, "checkpoints")
                rew_ckpt_path = os.path.join(checkpoint_folder, f"rew_norm_eps_{self.args.index_episode}")
                self.reward_normalizer = load_checkpoint(self.reward_normalizer, rew_ckpt_path, evaluate=False, map_location=self.device)
                if isinstance(self.reward_normalizer, PerEnvironmentReturnRewardNormalizer):
                    # The simulator starts fresh; retain scale statistics, but
                    # do not carry unfinished checkpoint episodes into new ones.
                    self.reward_normalizer.reset_history()
        
        self.cost_normalizer = None
        if self.args.use_cost and self.args.norm_cost:
            c_gamma = torch.tensor(self.args.c_gamma, dtype=self.tensor_dtype, device=self.device)
            self.cost_normalizer = NormalizeReward(self.args.num_envs, gamma=c_gamma, insize=self.args.num_cost, device=self.device)

    def _make_reward_normalizer(self):
        """Shared corrected default for every method; explicit legacy replay only."""
        mode = getattr(self.args, "reward_normalizer_mode", "per_environment")
        if mode == "legacy_task":
            return PerTaskRewardNormalizer(self.unique_task_ids, self.args.gamma, self.device)
        if mode != "per_environment":
            raise ValueError("reward_normalizer_mode must be per_environment or legacy_task")
        return PerEnvironmentReturnRewardNormalizer(
            self.unique_task_ids, self.task_indices, self.args.gamma, self.device,
        )
    
    
    # NOTE: _setup_storage removed — RolloutStorage is created in __init__
    # NOTE: _setup_tracking removed — MetricTracker is created in __init__
    # NOTE: _setup_trajectory_archive removed — TsilTrajectoryMemory is created in __init__
    # NOTE: _setup_wandb removed — TrainingLogger is created in __init__
    # NOTE: All trajectory archive, signal metric, replay, logging, curriculum,
    #       and performance metric methods removed — now on MetricTracker,
    #       TsilTrajectoryMemory, TsilReplayLoss, and TrainingLogger components.
    
    
    def _print_configuration(self):
        """Print training configuration."""
        trainable_params = sum(p.numel() for p in self.agent.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.agent.parameters())
        ddl_update_mode = "Disabled"
        if self.args.update_ddl:
            ddl_update_mode = "Direct"
            if self.args.anneal_ddl:
                ddl_update_mode = f"EMA(alpha={self.args.ddl_anneal_alpha:g})"
        raw_obs_shape_data = [
            ["Summary", ""],
            ["Task Ids", self.envs.task_idx],
            ["Num Envs Per Task", self.envs.task_env_count],
            ["Num Envs", self.envs.num_envs],
            ["Sequence Len", self.envs.seq_length],
            ["Observation Shape", self.envs.observation_space.shape],
            ["State Shape", self.envs.state_space.shape],
            ["Action Shape", self.envs.action_space.shape],
            ["Max Episode Time", f"{self.envs.max_eps_time:.2f}"],
            ["DDL Update", ddl_update_mode],
            ["Agent Class", self.agent.__class__.__name__],
            ["Hidden Size", self.args.hidden_size],
            ["Trainable Params", f"{trainable_params:,} / {total_params:,}"],
        ]
        print(tabulate(raw_obs_shape_data, headers="firstrow", tablefmt="grid"))
        
        print(f"########### ATTENTION ###########\n"
              f"Uniform Name: {self.args.run_name}\n\n"
              f"Batch Size: {self.args.batch_size}, MiniBatchSize: {self.args.minibatch_size}, "
              f"Num Minibatches: {self.args.num_minibatches}, Num UpdateEpochs: {self.args.update_epochs}\n\n"
              f"Local Instance Dir Name: {self.args.instance_dir}\n"
              f"#################################\n")
    
    
    # NOTE: _update_performance_metrics, update_curriculum, log_episode_metrics,
    #       _log_training_metrics removed — now on MetricTracker / TrainingLogger.
    
    
    
    def save_checkpoints(self):
        """Save model checkpoints based on performance."""
        self.metrics.save_training_checkpoints(
            self.agent, self.args, self.reward_normalizer,
            self.before_checkpoint_save, self.save_project_meta,
        )

    # ------------------------------------------------------------------
    # Rollout callbacks — override in project subclass for custom
    # episode-done or step-recording behaviour.
    # ------------------------------------------------------------------

    def _episodes_callback(self, terminal_ids, success_buf, terminal_metrics, infos):
        """Handle episode completions during rollout collection.

        Override in a project trainer subclass to customise episode-done
        behaviour (e.g. extra logging, curriculum changes).
        """
        m = self.metrics
        success_ids = terminal_ids[success_buf]

        # DDL updates are committed between rollouts so each episode keeps a
        # single deadline schedule from reset to termination.
        if self.args.update_ddl and len(success_ids) > 0 and "eps_time" in infos:
            success_times = (
                infos["eps_time"][self.storage.next_done.bool()]
                .float()
                .to(self.device)[success_buf]
            )
            self._pending_ddl_updates.append((
                success_ids.detach().clone(),
                success_times.detach().clone(),
            ))

        # Metric tracking (counters, per-task buffers, global averages)
        m.on_episodes_done(
            terminal_ids, success_buf, terminal_metrics,
            self.task_indices, self.tid_to_tidx,
        )
        current_step = int(getattr(self.rollout_collector, "global_step", m.global_step))

        # Signal metric recording for completed episodes
        if self.args.track_signal_metrics:
            for local_idx, env_tensor in enumerate(terminal_ids):
                env_id = int(env_tensor.item())
                task_idx_val = int(self.tid_to_tidx[self.task_indices[env_id]].item())
                eps_dense_return = float(terminal_metrics["eps_dense_return"][local_idx].item())
                success = bool(success_buf[local_idx].item())
                eps_time = float(terminal_metrics["eps_time"][local_idx].item())
                episode_id = self.storage.rollout_completed_episode_id_by_env.get(env_id)
                if episode_id is not None:
                    self.storage.rollout_completed_episode_horizon[int(episode_id)] = (
                        float(self.envs.episode_limits[env_id]) if hasattr(self.envs, 'episode_limits') else float(self.envs.max_eps_time))
                    self.storage.rollout_completed_episode_dense_return[int(episode_id)] = eps_dense_return
                    self.storage.rollout_completed_episode_return[int(episode_id)] = float(
                        terminal_metrics["eps_sum_rew"][local_idx].item()
                    )
                    timed_out = bool(infos.get("time_outs", torch.zeros_like(self.storage.next_done))[env_id].item())
                    env_terminated = bool(infos.get("terminated", torch.zeros_like(self.storage.next_done))[env_id].item())
                    termination = (
                        "success" if success else
                        "timeout" if timed_out else
                        "terminated_failure" if env_terminated else
                        "failure"
                    )
                    self.storage.rollout_completed_episode_termination[int(episode_id)] = termination
                m.record_completed_episode_signal(
                    task_idx_val,
                    eps_dense_return,
                    success,
                    eps_time,
                    steps=current_step,
                    iteration=m.global_update_iter,
                    episode_id=episode_id,
                    horizon=float(self.envs.episode_limits[env_id]) if hasattr(self.envs, "episode_limits") else float(self.envs.max_eps_time),
                )

        # Archive finalization
        self.tsil_memory.finalize_archived_episodes(
            terminal_ids, success_buf, terminal_metrics,
            self.task_indices, current_step, m.global_episodes, m.global_update_iter,
        )

    def _steps_callback(self, step_idx, step_action, org_reward, reward, org_cost, infos):
        """Per-step callback for trajectory archiving.

        Override in a project trainer subclass to add custom per-step
        recording logic.
        """
        self.tsil_memory.record_rollout_step(
            step_idx, step_action, org_reward, reward, org_cost, infos, self.storage, self.envs,
        )

    def _commit_pending_ddl_updates(self):
        if not self._pending_ddl_updates:
            return

        for success_ids, success_times in self._pending_ddl_updates:
            if len(success_ids) == 0:
                continue
            updated_upper = RolloutCollector.compute_time2end_upper_update(
                self.envs, self.args, success_ids, success_times,
            )
            self.envs._update_time2end_upper(success_ids, updated_upper)

        self._pending_ddl_updates.clear()
        self.metrics.avg_time2end_upper = self.envs._get_avg_time2end_upper()

    def _sil_schedule_context(self):
        if not hasattr(self.envs, "real_time2end_init"):
            return None

        upper = self.envs.real_time2end_init.detach().to(self.device)
        time2end_by_task = {}
        for task_id in self.unique_task_ids.tolist():
            task_id = int(task_id)
            mask = self.task_indices == task_id
            if mask.any():
                time2end_by_task[task_id] = float(upper[mask].min().item())

        if not time2end_by_task:
            return None

        return {
            "time2end_by_env": upper.clone(),
            "time2end_by_task": time2end_by_task,
            "ctrl_dt": float(getattr(self.envs, "ctrl_dt", getattr(self.envs, "dt", 1.0))),
            "max_eps_time": float(getattr(self.envs, "max_eps_time", 1.0)),
            "r_epstime_scale": float(self.envs.cfg.get("r_epstime_scale", 0.0)),
            "update_ddl": bool(getattr(self.args, "update_ddl", False)),
            "temporal_efficiency_bonus": getattr(self.args, "temporal_efficiency_bonus", None),
            "ratio_range": getattr(self.envs, "ratio_range", None),
        }

    # ------------------------------------------------------------------
    # Policy update orchestration
    # ------------------------------------------------------------------

    def update_policy(self, returns, advantages, returns_c, advantages_c,
                      returns_t, advantages_t, initial_lstm_state):
        """Orchestrate a full policy update iteration.

        Calls into PolicyUpdater for the math, then handles project hooks,
        replay integration, divergence recovery, and logging via TrainingLogger.
        """
        m = self.metrics

        # Prepare batch
        batch = self.policy_updater.prepare_batch(
            self.storage, self.envs, self.agent,
            self.task_indices, self.unique_task_ids,
            returns, advantages, returns_c, advantages_c, returns_t,
        )

        # Signal metrics
        m.compute_advantage_magnitude_metrics(batch["advantages_raw"])
        m.set_rollout_bootstrap_diagnostics(batch.get("_rollout_bootstrap_diagnostics", {}))
        m.refresh_dense_tail_metrics()
        m.refresh_success_eps_time_metric()
        m.refresh_sil_revisit_episode_metrics(self.tsil_memory)

        # Project hook
        if self.before_policy_update(batch=batch) == "skip_update":
            self._commit_pending_ddl_updates()
            return False

        replay_context = self.tsil_replay_loss.prepare_iteration(
            tsil_memory=self.tsil_memory,
            unique_task_ids=self.unique_task_ids,
            reward_normalizer=self.reward_normalizer,
            global_update_iter=m.global_update_iter,
            schedule_context=self._sil_schedule_context(),
        )
        replay_context["current_success_rate"] = float(m.cur_success_rate)
        replay_context["global_step"] = int(m.global_step)

        if self.causal_shadow_diagnostics.should_capture(m.cur_success_rate):
            manifest = self.causal_shadow_diagnostics.capture(
                trainer=self,
                batch=batch,
                initial_lstm_state=initial_lstm_state,
                replay_context=replay_context,
            )
            print(
                "[causal_shadow] captured exact-rollout shadow updates at "
                f"step={manifest['global_step']} success={manifest['current_success_rate']:.3f} "
                f"peak={manifest['peak_success_rate']:.3f}",
                flush=True,
            )

        # Minibatch updates
        pre_direction = (getattr(self.args, "sil_actual_direction_diagnostics", False)
            and getattr(self.args, "sil_direction_timing", "pre_update") == "pre_update")
        direction_capture = (self.tsil_replay_loss.pre_step_landscape_due(m.global_update_iter)
            if pre_direction else (getattr(self.args, "sil_actual_direction_diagnostics", False)
            and self.args.sil_analysis_interval > 0
            and m.global_update_iter % self.args.sil_analysis_interval == 0))
        direction_before = None
        if direction_capture:
            direction_before = torch.nn.utils.parameters_to_vector(
                self.tsil_replay_loss._trainable_params(self.agent)).detach().clone()
        self.update_event_diagnostics.before(self, batch, replay_context=replay_context)
        (
            policy_diverged,
            valid_met,
            num_agent_updates,
            agent_params_store,
            optim_params_store,
            illed_met,
            replay_stats,
        ) = (
            self.policy_updater.run_minibatch_updates(
                batch, self.agent, self.optimizer, initial_lstm_state,
                self.lr_scheduler, m.cur_ent, m.global_update_iter,
                global_step=m.global_step,
                replay_module=self.tsil_replay_loss,
                replay_context=replay_context,
                current_success_rate=float(m.cur_success_rate),
            )
        )
        if int(getattr(self.args, "sil_separate_updates", 0)) > 0 and not policy_diverged:
            replay_stats = self.policy_updater.run_separate_replay_updates(
                batch, self.agent, self.optimizer, self.tsil_replay_loss,
                replay_context, m.global_update_iter,
            )
        # Handle divergence
        if policy_diverged:
            self.agent.load_state_dict(agent_params_store)
            self.optimizer.load_state_dict(optim_params_store)
            m.skipped_update_iter += 1
            if self.args.saving and self.args.wandb:
                import wandb

                wandb.log({
                    "misc/iterations": m.global_update_iter,
                    "misc/episodes": m.global_episodes,
                    "misc/steps": m.global_step,
                    "debug/skipped_update_iter": m.skipped_update_iter,
                    "debug/illed_kl": illed_met["approx_kl"].item(),
                    "debug/illed_adv": illed_met["mb_advantages"].mean().item(),
                    "debug/illed_ratio": illed_met["ratio"].mean().item(),
                    "debug/illed_entropy": illed_met["entropy_mean"].item(),
                    "debug/valid_update_iters": num_agent_updates,
                }, commit=False)

        if getattr(self.args, "sil_actual_direction_diagnostics", False):
            if direction_before is not None:
                replay_context["optimizer_displacement"] = (
                    torch.nn.utils.parameters_to_vector(self.tsil_replay_loss._trainable_params(self.agent)).detach()
                    - direction_before)
            if pre_direction:
                if direction_before is not None:
                    self.tsil_replay_loss.finalize_pre_step_displacement(
                        replay_context, replay_context["optimizer_displacement"])
            # Keep post-step reference statistics; this helper skips landscape
            # recomputation when the pre-step diagnostic path is selected.
            self.tsil_replay_loss.update_diagnostics(self.agent, replay_context, batch,
                self.policy_updater, m.global_update_iter)
            replay_stats.update({k: v for k, v in replay_context.get("stats", {}).items() if k.startswith("sil_")})
        self.update_event_diagnostics.after(self)
        seen_temporal = getattr(self.agent, "temporal_critic_seen_samples", None)
        if seen_temporal is not None and num_agent_updates > 0 and "temporal_model_targets" in batch:
            valid_count = torch.isfinite(batch["temporal_model_targets"]).sum()
            seen_temporal.add_(valid_count.to(device=seen_temporal.device, dtype=seen_temporal.dtype))
            seen_samples = int(seen_temporal.item())
            replay_stats["replay_temporal_model_seen_samples"] = seen_samples
            replay_stats["replay_temporal_model_ready"] = int(
                seen_samples >= int(getattr(self.args, "sil_temporal_ready_samples", 0))
            )
        replay_stats.update(
            self.policy_updater.temporal_model_evaluation(self.agent, batch),
        )
        batch.setdefault("_rollout_bootstrap_diagnostics", {}).update({
            "ppo_kl_penalty_effective_coef": self.policy_updater.last_effective_kl_penalty,
            "ppo_kl_penalty_active": int(self.policy_updater.last_effective_kl_penalty > 0.0),
            "ppo_kl_activation_success": float(getattr(self.args, "ppo_kl_activation_success", 0.0)),
        })
        m.set_rollout_bootstrap_diagnostics(batch.get("_rollout_bootstrap_diagnostics", {}))
        m.last_replay_stats = replay_stats

        logprob_delta = self.policy_updater.policy_logprob_delta(self.agent, batch)
        m.completed_episode_horizon = getattr(self.storage, "rollout_completed_episode_horizon", {})
        m.compute_success_positive_advantage_metrics(
            batch["advantages"], self.storage.step_episode_ids,
            self.storage.rollout_completed_episode_success,
            completed_episode_time=self.storage.rollout_completed_episode_time,
            completed_episode_dense_return=self.storage.rollout_completed_episode_dense_return,
            completed_episode_return=self.storage.rollout_completed_episode_return,
            completed_episode_termination=self.storage.rollout_completed_episode_termination,
            completed_episode_task_id=self.storage.rollout_completed_episode_task_id,
            advantages_raw=batch["advantages_raw"],
            logprob_delta=logprob_delta,
            policy_sample_valid=batch["policy_sample_valid"],
        )

        if getattr(self.args, "sil_actual_direction_diagnostics", False):
            m.append_episode_signal_history()

        self._commit_pending_ddl_updates()

        # Explained variance
        if self.args.use_grpo:
            explained_var = np.nan
        else:
            y_pred = batch["values"].to(torch.float32).cpu().numpy()
            y_true = batch["returns"].to(torch.float32).cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Log metrics
        if num_agent_updates > 0:
            avg_met = {
                k: (v / num_agent_updates) if v is not None else None
                for k, v in valid_met.items()
                if not k.startswith("_")
            }
            def _scalar_metric(value):
                if hasattr(value, "item"):
                    return value.item()
                return float(value) if value is not None else 0.0

            self.training_logger.log_training_metrics(
                m, self.agent, self.optimizer, self.envs,
                avg_met["pg_loss"], avg_met["v_loss"], avg_met["entropy"],
                avg_met["approx_kl"], avg_met["mb_advantages"], explained_var,
                avg_met.get("v_loss_c"), avg_met.get("v_loss_t"),
                avg_met.get("cost_loss"), avg_met.get("bound_loss", 0.0),
                extra_metrics={
                    "debug/skipped_update_iter": int(m.skipped_update_iter),
                    "train/policy_grad_norm_pre_noise": _scalar_metric(avg_met.get("policy_grad_norm_pre_noise", 0.0)),
                    "train/policy_grad_norm_post_noise": _scalar_metric(avg_met.get("policy_grad_norm_post_noise", 0.0)),
                    "train/policy_grad_norm_post_clip": _scalar_metric(avg_met.get("policy_grad_norm_post_clip", 0.0)),
                    "train/policy_grad_noise_norm": _scalar_metric(avg_met.get("policy_grad_noise_norm", 0.0)),
                    "train/policy_grad_norm_raw_noise": _scalar_metric(avg_met.get("policy_grad_norm_raw_noise", 0.0)),
                    "train/policy_grad_norm_restored_noise": _scalar_metric(avg_met.get("policy_grad_norm_restored_noise", 0.0)),
                    "train/policy_grad_clean_corrupted_cosine": _scalar_metric(avg_met.get("policy_grad_clean_corrupted_cosine", 0.0)),
                    "train/policy_grad_nonppo_norm": _scalar_metric(avg_met.get("policy_grad_nonppo_norm", 0.0)),
                    "train/policy_grad_final_pre_clip_norm": _scalar_metric(avg_met.get("policy_grad_final_pre_clip_norm", 0.0)),
                    "train/policy_grad_corrupted_nonppo_cosine": _scalar_metric(avg_met.get("policy_grad_corrupted_nonppo_cosine", 0.0)),
                    "train/policy_grad_final_nonppo_cosine": _scalar_metric(avg_met.get("policy_grad_final_nonppo_cosine", 0.0)),
                    "train/policy_grad_final_clean_ppo_cosine": _scalar_metric(avg_met.get("policy_grad_final_clean_ppo_cosine", 0.0)),
                },
            )

        elif getattr(self.args, "sil_actual_direction_diagnostics", False):
            self.training_logger.log_skipped_training_update(m)

        # Project hook
        self.after_policy_update(batch=batch)

        return policy_diverged

    def train(self):
        """Main training loop."""
        m = self.metrics
        n_cpu_cores = multiprocessing.cpu_count()
        n_gpu_used = 1
        thread_limits = max(4, int(n_cpu_cores * n_gpu_used / self.args.num_envs))
        
        with tpc.threadpool_limits(limits=thread_limits):
            torch.cuda.empty_cache()

            num_iters = int(self.args.total_iters)
            m.training_start_time = time.time()
            
            for iter in range(num_iters):
                iter_start_time = time.perf_counter()
                self._update_stress_gate()
                
                # Step 0: Collect rollout via RolloutCollector
                self.rollout_collector.global_step = m.global_step
                initial_lstm_state = self.rollout_collector.collect(
                    agent=self.rollout_agent,
                    storage=self.storage,
                    reward_normalizer=self.reward_normalizer,
                    cost_normalizer=self.cost_normalizer,
                    record_step_fn=self._steps_callback if self.tsil_memory.archive_enabled else None,
                    on_episodes_done_fn=self._episodes_callback,
                    update_reward_stats=True,
                )
                m.global_step = self.rollout_collector.global_step
                m.rollout_time = self.rollout_collector.rollout_time
                m.rollout_env_step_time = self.rollout_collector.env_step_time
                self.tsil_memory.flush_pending_records()
                self._update_stress_gate()

                self.after_rollout_complete(initial_lstm_state=initial_lstm_state)
                
                # Log episode metrics
                self.training_logger.log_episode_metrics(m, self.is_multi_task)

                # Save checkpoints based on performance before update
                self.save_checkpoints()

                # Skip training for random policy
                if self.args.random_policy:
                    self._commit_pending_ddl_updates()
                    m.update_time = 0.0
                    m.update_performance_metrics(self.args.num_steps, self.args.num_envs)
                    m.track_iter_time(iter_start_time)
                    m.print_status(iter, num_iters, self.is_multi_task,
                                   self.tsil_memory.num_tasks_with_success_archive())
                    continue
                
                # Compute advantages
                update_start_time = time.perf_counter()
                if self.args.use_grpo:
                    returns, advantages, returns_c, advantages_c, return_t, advantages_t = (
                        self.policy_updater.compute_grpo_returns(self.storage, self.agent, self.envs)
                    )
                else:
                    returns, advantages, returns_c, advantages_c, return_t, advantages_t = (
                        self.policy_updater.compute_advantages(self.storage, self.agent, self.envs)
                    )
                
                # Update policy
                policy_diverged = self.update_policy(
                    returns, advantages, 
                    returns_c, advantages_c, 
                    return_t, advantages_t, 
                    initial_lstm_state)
                
                # Update counters and curriculum
                m.update_time = time.perf_counter() - update_start_time
                m.update_performance_metrics(self.args.num_steps, self.args.num_envs)
                m.global_update_iter += 1
                m.curri_update_iters += 1
                
                # Track update time for ETA estimation
                m.track_iter_time(iter_start_time)
                m.print_status(iter, num_iters, self.is_multi_task,
                               self.tsil_memory.num_tasks_with_success_archive())
            
            # Save final checkpoint
            self.tsil_memory.flush_pending_records()
            if self.args.saving and not self.args.random_policy:
                self.agent.save_checkpoint(folder_path=self.args.checkpoint_dir, suffix='last')
                save_checkpoint(self.reward_normalizer, self.args.checkpoint_dir, ckpt_name="rew_norm_eps", suffix='last')
                self.save_project_meta()

            self.after_training_complete()
            
            print('\nProcess Over here')
            if hasattr(self.envs, 'close'):
                self.envs.close()
            TrainingLogger.finish()


def run_training(trainer_cls=None, args_fn=None):
    """Shared training bootstrap used by project entrypoints and wrappers."""
    trainer_class = OnPolicyTrainer if trainer_cls is None else trainer_cls
    project_name = getattr(trainer_class, "project_name", "TSIL")
    # Print the config
    args = args_fn() if args_fn is not None else parse_args(project_name=project_name)

    overrides = [
        f"task_id={args.tasks}",
        f"task_counts={args.task_counts}",
        f"task.env.episodeLength={args.episodeLength}",
        f"fixed={args.fixed}",
        f"same_init_config_per_task={args.same_init_config_per_task}",
        f"termination_on_success={args.termination_on_success}",
        f"reward_scale={args.reward_scale}",
    ]
    with initialize_config_dir(version_base=None, config_dir=HYDRA_CONFIG_DIR):
        cfg = compose(config_name="custom_config", overrides=overrides)
    
    print("\n########## Summary ##########\n", 
          OmegaConf.to_yaml(cfg),
          "\n####################\n")

    # Convert to dictionary
    cfg = OmegaConf.to_container(cfg, resolve=True)
    
    # merge to args
    for key, value in cfg.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    # Your training code here
    trainer = trainer_class(args)
    
    # Save the training configuration after trainer init (includes wandb info)
    if args.saving:
        with open(args.json_file_path, 'w') as json_obj:
            json.dump(vars(args), json_obj, indent=4)
    
    try:
        trainer.train()
    except Exception as e:
        traceback.print_exc()
        if (
            not isinstance(e, KeyboardInterrupt)
            and args.debug
            and sys.stdin.isatty()
            and os.environ.get("TIMEAWARE_SWEEP") != "1"
        ):
            import ipdb
            ipdb.post_mortem()
        raise


def main():
    run_training()


if __name__ == "__main__":
    main()
