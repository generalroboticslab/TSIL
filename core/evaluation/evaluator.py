import json
import numpy as np
import os
import random
import time

import isaacgym
from MTBench import isaacgymenvs
import torch

from core.agents.agent import get_agent
from core.evaluation.args import get_args
from core.evaluation.metrics import EvalMetrics
from core.evaluation.replay import TrajectoryReplayer


class BaseEvaluator:
    """Base evaluator loop for trained RL policies."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.tensor_dtype = torch.float32
        
        # Setup
        self._setup_seeding()
        self._setup_environment()
        self._setup_agent()

        # Create components
        self.eval_metrics = EvalMetrics(
            args, self.device, self.num_tasks, self.unique_task_ids,
            self.tid_to_tidx, self.is_multi_task, self.tensor_dtype,
        )
        self.replayer = TrajectoryReplayer(
            self.envs, args, self.device, self.task_indices,
        )

        self.setup_project_eval_exts()

    def setup_project_eval_exts(self):
        """Project hook for evaluation-specific setup."""

    def handle_special_eval_modes(self):
        """Project hook for special evaluation modes.

        Returns True when the hook handled the full run.
        """
        if getattr(self.args, "debug_replay_first_step", False):
            self.replayer.debug_replay_first_step()
            return True
        if self.args.inspect_trajectory:
            self.replayer.inspect()
            return True
        if getattr(self.args, "record_eval_trajectories", False):
            self.replayer.record_eval_samples(self.agent)
            return True
        return False

    def after_eval_step(self, *args, **kwargs):
        """Project hook after each evaluation environment step."""

    def after_eval_results(self, *args, **kwargs):
        """Project hook after one goal-speed evaluation finishes."""
    
    
    def _setup_seeding(self):
        """Set random seeds for reproducibility."""
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic
    
    
    def _setup_environment(self):
        """Initialize Isaac Gym environment."""
        # Compute total num_envs from task_counts if provided
        if hasattr(self.args, 'task_counts') and self.args.task_counts is not None:
            self.args.num_envs = sum(self.args.task_counts)
        
        self.envs = isaacgymenvs.make(
            seed=self.args.seed,
            task=self.args.env_name,
            num_envs=self.args.num_envs,
            sim_device=self.args.sim_device,
            rl_device=self.args.sim_device,
            graphics_device_id=self.args.graphics_device_id,
            headless=not self.args.rendering,
            force_render=self.args.rendering,
            custom_args=self.args,
            tasks=self.args.tasks,
            taskEnvCount=self.args.task_counts,
            cfg=self.args
        )
        
        # Setup task indices for multi-task tracking (similar to training)
        if hasattr(self.envs, 'extras') and 'task_indices' in self.envs.extras:
            self.task_indices = self.envs.extras['task_indices'].to(self.device)
        elif hasattr(self.envs, 'task_idx') and hasattr(self.envs, 'task_env_count'):
            task_indices = sum([[tid] * count for tid, count in zip(self.envs.task_idx, self.envs.task_env_count)], [])
            self.task_indices = torch.tensor(task_indices, dtype=torch.long, device=self.device)
        else:
            # Single-task case
            self.task_indices = torch.zeros(self.args.num_envs, dtype=torch.long, device=self.device)
        
        # Per-task tracking setup
        self.unique_task_ids = torch.unique(self.task_indices)
        self.num_tasks = len(self.unique_task_ids)
        self.is_multi_task = self.num_tasks > 1
        
        # Map task_id to row index for efficient tensor lookup
        max_task_id = self.unique_task_ids.max().item() + 1
        self.tid_to_tidx = torch.zeros(max_task_id, dtype=torch.long, device=self.device)
        for i, tid in enumerate(self.unique_task_ids):
            self.tid_to_tidx[tid] = i
        self.task_cont_indices = self.tid_to_tidx[self.task_indices]
    
    
    def _setup_agent(self):
        """Initialize agent and load checkpoint."""
        self.agent = None
        if not self.args.random_policy and not self.args.heuristic_policy:
            self.agent = get_agent(self.envs, self.args, self.device)
            self.agent.load_checkpoint(self.args.checkpoint_path, evaluate=True, map_location=self.device)
            self.agent.deterministic = self.args.deterministic
    
    
    def reset_obs_done(self):
        """Reset observation and done flags."""
        next_obs_dict = self.envs.reset()
        next_obs = torch.Tensor(next_obs_dict["obs"]).to(self.device)
        next_state = torch.Tensor(next_obs_dict["states"]).to(self.device)
        next_done = torch.zeros(self.args.num_envs).to(self.device)
        next_lstm_state = None
        
        if self.args.use_lstm:
            next_lstm_state = (
                torch.zeros(self.agent.crt_lstm.num_layers, self.args.num_envs, self.agent.crt_lstm.hidden_size).to(self.device),
                torch.zeros(self.agent.crt_lstm.num_layers, self.args.num_envs, self.agent.crt_lstm.hidden_size).to(self.device),
                torch.zeros(self.agent.act_lstm.num_layers, self.args.num_envs, self.agent.act_lstm.hidden_size).to(self.device),
                torch.zeros(self.agent.act_lstm.num_layers, self.args.num_envs, self.agent.act_lstm.hidden_size).to(self.device),
            )
        
        return next_obs, next_state, next_done, next_lstm_state
    
    
    def evaluate_simulation(self):
        """Run simulation-based evaluation over goal speed range."""
        with torch.no_grad():
            for goal_speed in self.args.goal_speed_lst:
                results = self._run_evaluation_loop(goal_speed)
                self._log_and_save_results(goal_speed, results)
    
    
    def _run_evaluation_loop(self, goal_speed):
        """Run evaluation loop for a single goal speed setting.
        
        Returns:
            dict: Results containing num_episodes, num_success_eps, machine_time, etc.
        """
        # Setup
        self.envs.goal_speed = goal_speed
        self.envs.reset_all()
        next_obs, next_state, next_done, next_lstm_state = self.reset_obs_done()
        self.agent.deterministic = self.args.deterministic
        
        # Reset metrics
        self.eval_metrics.reset_evaluation_metrics()
        
        # State tracking
        num_episodes = 0
        num_success_eps = 0
        valid_env_ids = torch.arange(self.args.num_envs, device=self.device)
        warmup_done = self.args.warmup_episodes <= 0 or self.args.saving
        warmup_count = 0
        
        requested_episodes = getattr(self.args, "requested_target_episodes", self.args.target_episodes)
        max_trials = self.args.target_episodes
        trial_msg = f"{requested_episodes} Trials"
        if max_trials != requested_episodes:
            trial_msg += f" | Max Trials: {max_trials}"
        print(f"Start Evaluating: {trial_msg} | "
              f"Goal Speed: {goal_speed} | {self.args.target_success_eps} Success Trials Required")
        if not warmup_done:
            print(f"Running Warmup Episodes {self.args.warmup_episodes}...")
        
        start_time = time.perf_counter()
        
        while num_episodes < self.args.target_episodes:
            # Get action from policy
            action, next_lstm_state = self._get_action(next_obs, next_done, next_lstm_state)
            
            # Step environment
            next_obs_dict, reward, done, infos = self.envs.step(action)
            next_obs = next_obs_dict["obs"].to(self.device)
            next_done = done.to(self.device)
            self.eval_metrics.step_metrics['eps_r'] += reward.to(self.device).view(-1)
            self.after_eval_step(
                next_obs=next_obs,
                reward=reward,
                done=done,
                infos=infos,
                goal_speed=goal_speed,
                num_episodes=num_episodes,
            )
            
            # Handle episode terminations
            terminal_mask = (done == 1)
            
            if terminal_mask.any():
                if not warmup_done:
                    warmup_count += terminal_mask.sum().item()
                    if warmup_count >= self.args.warmup_episodes:
                        warmup_done = True
                        print(f"End Warmup: {warmup_count}/{self.args.warmup_episodes}")
                    else:
                        self.eval_metrics.reset_step_metrics()
                        continue
                
                # Handle strict eval (each env evaluated once)
                if self.args.strict_eval:
                    terminal_mask = terminal_mask & (valid_env_ids != -1)
                    valid_env_ids[terminal_mask] = -1
                    if not terminal_mask.any():
                        continue
                
                # Update metrics
                terminal_count = terminal_mask.sum().item()
                num_episodes += terminal_count
                num_success_eps += self.eval_metrics.update_episode_metrics(terminal_mask, infos, self.task_indices)
                
                # Progress logging
                self._print_progress(num_episodes, num_success_eps)
                
                # Check termination conditions
                if self._should_stop(num_success_eps, valid_env_ids):
                    break
        
        machine_time = time.perf_counter() - start_time
        return {
            "num_episodes": num_episodes,
            "num_success_eps": num_success_eps,
            "machine_time": machine_time,
            "infos": infos
        }
    
    
    def _get_action(self, obs, done, lstm_state):
        """Get action from policy."""
        if self.args.random_policy or self.args.heuristic_policy:
            action = torch.rand((self.args.num_envs, self.envs.num_actions), device=self.device)
            return action, lstm_state
        
        if self.args.use_lstm:
            action, _, lstm_state = self.agent.get_action_and_value(
                obs, lstm_state, done, action_only=True)
        else:
            action, _ = self.agent.get_action_and_value(obs, action_only=True)
        return action, lstm_state
    
    
    def _print_progress(self, num_episodes, num_success_eps):
        """Print evaluation progress."""
        msg = f"Episodes: {num_episodes} | Total Success: {num_success_eps}"
        print(msg)
    
    
    def _should_stop(self, num_success_eps, valid_env_ids):
        """Check if evaluation should stop early."""
        if self.args.target_success_eps is not None and num_success_eps >= self.args.target_success_eps:
            return True
        
        if self.args.strict_eval and (valid_env_ids == -1).all():
            print("All envs evaluated, stopping")
            return True
        
        return False
    
    
    def _log_and_save_results(self, goal_speed, results):
        """Log results and save to file."""
        num_episodes = results["num_episodes"]
        num_success_eps = results["num_success_eps"]
        machine_time = results["machine_time"]
        infos = results["infos"]
        
        # Compute averages
        self.eval_metrics.compute_average_metrics(num_episodes, num_success_eps)
        self.eval_metrics.update_speed_time_dict(goal_speed)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Goal Speed: {goal_speed} | Episodes: {num_episodes}")
        m_avg = self.eval_metrics.eps_metrics_avg
        m_std = self.eval_metrics.eps_metrics_std
        print(f"Success Rate: {m_avg['eps_success'] * 100:.2f}%")
        print(f"Avg Reward: {m_avg['eps_r']:.3f}")
        print(f"Avg Time - Used: {m_avg['eps_time']:.3f}s | "
              f"Goal: {m_avg['eps_time_goal']:.3f}s | "
              f"Mismatch: {m_avg['eps_time_p']:.3f}s")
        print(f"Instability - Sum: {m_avg['eps_sum_inst']:.3f} ± {m_std['eps_sum_inst']:.3f} | "
              f"Max: {m_avg['eps_max_inst']:.3f} | "
              f"Threshold: {infos['scene_linvel_lim'].item():.3f}")
        print(f"Manipulation Time: {m_avg['interaction_time']:.3f}s | "
              f"Wall Time: {machine_time:.2f}s | Envs: {self.args.num_envs}")
        
        print(f"{'='*60}\n")
        
        # Print per-task results
        self.eval_metrics.print_per_task_results()
        
        # Save
        self.eval_metrics.save_results(num_episodes, machine_time, infos)
        self.after_eval_results(goal_speed=goal_speed, results=results)
    
    
    def run(self):
        """Main evaluation entry point."""
        if not self.handle_special_eval_modes():
            self.evaluate_simulation()
        
        print('Process Over')

def run_evaluation(evaluator_cls=None):
    """Shared evaluation bootstrap used by project entrypoints and wrappers."""
    cur_pid = os.getpid()
    print(f"###### Evaluation PID is {cur_pid} ######")

    evaluator_class = BaseEvaluator if evaluator_cls is None else evaluator_cls
    project_name = getattr(evaluator_class, "project_name", "TSIL")
    args = get_args(project_name=project_name)

    if args.saving:
        with open(args.json_file_path, 'w') as json_obj:
            json.dump(vars(args), json_obj, indent=4)

    evaluator = evaluator_class(args)
    evaluator.run()


if __name__ == "__main__":
    run_evaluation()
