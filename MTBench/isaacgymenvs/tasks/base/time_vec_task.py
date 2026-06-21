import os
import json
import abc
from typing import Dict, Any, Tuple, List
import time

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

from isaacgym import gymtorch, gymapi, gymutil
from isaacgymenvs.utils.torch_jit_utils import (
    to_torch, torch_rand_tensor, torch_rand_float, normalize, quat_mul, 
    axisangle2quat, quat_from_euler_xyz
)
from isaacgymenvs.utils.task_utils import mix_clone, to_numpy, unify_quat, unify_quat_np
from isaacgymenvs.tasks.base.vec_task import VecTask


class TimeVecTask(VecTask):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 24}

    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture: bool = False, force_render: bool = False): 
        """Initialise the `VecTask`."""
        self.custom_variable_init() 
        self.obs_act_rew_init() 
        super().__init__(config, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self.common_data_init() 
        self.timeaware_init()   
        self.allocate_time_buffers() 
        self.init_cur_dr_params(init_curri_ratio=self.cfg.get("init_curri_ratio", 0.0))

    # ----------------------------------------------------------------------
    # 1. Simulation Setup & Allocation
    # ----------------------------------------------------------------------
    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))


    def allocate_time_buffers(self):
        """Allocate the observation, states, etc. buffers."""
        self.extras, self.state_slice, self.obs_slice = {}, {}, {}
        self.state_names, self.obs_names, self.task_priv_obs_names = {}, {}, {}
        self.agent = None  # To be set when attaching agent later

        # Main Buffers
        self.states_buf = torch.zeros((self.num_envs, self.num_states), device=self.device)
        self.obs_buf = torch.zeros((self.num_envs, self.num_observations), device=self.device)

        # Previous State Buffers
        self.prev_tgtq = torch.zeros((self.num_envs, self.num_franka_dofs-2), device=self.device)
        self.prev_tgtq_gripper = torch.zeros((self.num_envs, 2), device=self.device)
        self.prev_dq = torch.zeros((self.num_envs, self.num_franka_dofs-2), device=self.device)

        # Reward & Status Buffers
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.success_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.timeout_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.continuous_check_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.force_has_applied = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # Time-Aware Buffers
        self.time_ratio_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.float)
        self.prev_time_ratio_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.float)
        self.time2end = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.time2end_init = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.real_timecur = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.real_time2end = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.real_time2end_init = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.interaction_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        # Stability Buffers
        self.max_linvel_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.sce_linvel_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.sce_linacc_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.linvel_max_gt = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.linvel_max_gt_init = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.accm_instability = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        self.env2index = -torch.ones(self.num_envs, device=self.device, dtype=torch.long)

        # Constants
        self.unit_z = torch.tensor([0., 0., 1.], device=self.device).repeat(self.num_envs, 1)
        self.unit_quat = torch.tensor([0., 0., 0., 1.], device=self.device).repeat(self.num_envs, 1)


    def custom_variable_init(self):
        """General configs reading and placeholders initialization."""
        self.dt = self.cfg.get("dt", 1/60)
        self.ratio_range = self.cfg.get("ratio_range", None)
        self.goal_speed = self.cfg.get("goal_speed", None)
        self.goal_time = self.cfg.get("goal_time", None)
        self.disturbance_v = self.cfg.get("disturbance_v", None)
        self.use_beta = self.cfg.get("beta", False)
        self.act_scale = self.cfg.get("act_scale", 1.)
        self.training = not self.cfg.get("eval_result", False)
        self.add_obs_noise = (self.training and self.cfg.get("add_obs_noise", False)) or self.cfg.get("apply_noise_eval", False)
        self.add_act_noise = (self.training and self.cfg.get("add_act_noise", False)) or self.cfg.get("apply_noise_eval", False)
        self.obj_target_obs_noise_scale = float(self.cfg.get("obj_target_obs_noise_scale", 0.0) or 0.0)
        self.obj_target_obs_noise_dist = str(self.cfg.get("obj_target_obs_noise_dist", "gaussian")).lower()
        if not (self.training or self.cfg.get("apply_noise_eval", False)):
            self.obj_target_obs_noise_scale = 0.0
  
        # Placeholders
        self.states, self.prev_states, self.world_states = {}, {}, {}
        self.states_real, self.prev_states_real = {}, {}
        self.link_handles, self.debug_info = {}, {}
        self.num_dofs, self.actions = None, None

        # Tensors
        self._root_state, self._contact_forces = None, None
        self._dof_state, self._q, self._qd = None, None, None
        self._rigid_body_state = None
        self._pos_control, self._effort_control = None, None
        self._global_indices = None

        # Env config
        self.up_axis, self.up_axis_idx = "z", 2
        self.control_freq_inv = self.cfg.get("control_freq_inv", 1)
        self.control_freq = 1. / self.dt / self.control_freq_inv
        self.ctrl_dt = self.dt * self.control_freq_inv
        self.MAX_CONTACT_FORCE_NORM = 20.
        self.MAX_VEL_NORM = 10.
        
        self.robot_variable_init()
        self.domain_randomization_init()


    def domain_randomization_init(self):
        self.dr_settings = {
            "spatial": {"franka_dof": [0., self.franka_dof_noise]},
            "properties": {},
            "controller": {
                "max_vel_subtract": [0., self.cfg.get("max_vel_subtract", 0.)],
                "alpha": [0., 0.9], 
                "gripper_delay": [0., 0.3], # We do add gripper delay for now
            },
            "noise": {
                "eef_pos": [0., 0.005], 
                "eef_quat": [0., np.pi / 60],
                "q_gripper": [0., 0.005], 
                "q": [0., np.pi / 60],
                "action": [0., 0.1], 
                "inertial_mat": [0., 0.1],
                "delta_Kp": [0, 25], 
                
                "v_gripper": [0., 0.005], 
                "gripper_delay": [0., 0.05],
            },
        }
        # cur_dr_params: Dict[task_id, Dict[category, Dict[param, value]]]
        # For single-task, task_id=0. For multi-task, task_id from self.task_idx
        self.cur_dr_params = {}
        self.curri_ratio_per_task = {}


    def common_data_init(self):
        self.all_env_ids = torch.arange(self.num_envs, device=self.device)
        self.num_dofs = self.gym.get_sim_dof_count(self.sim)
        self.num_actors = self.gym.get_sim_actor_count(self.sim)
        self.num_bodies = self.gym.get_sim_rigid_body_count(self.sim)

        self._root_state = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        self._dof_state = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        self._rigid_body_state = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        self._contact_forces = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim))
        self._refresh()


    def obs_act_rew_init(self):
        # Obs & States
        self.cfg["env"]["numObservations"] += 3 # time_ratio, time_cur, ddl
        self.cfg["env"]["numStates"] = self.cfg["env"]["numObservations"]
        self.cfg["env"]["numStates"] += 2 # time_ratio, time_cur, ddl, sce_linvel, lim_linvel
        self.task_priv_obs_init()
        
        self.numSingleObs = self.cfg["env"]["numObservations"]
        self.numSingleState = self.cfg["env"]["numStates"]
        self.seq_length = self.cfg.get("seq_length", 1)
        
        # Rewards
        self.reward_settings = {}
        self.reward_settings.update({
            "r_success": self.cfg.get("successRewardScale", 1000.),
            "r_violate": self.cfg.get("violateRewardScale", 0.),
            "r_action_penalty_scale": self.cfg["env"].get("actionPenaltyScale", 0.),
            "r_force_penalty_scale": self.cfg["env"].get("forcePenaltyScale", 0.),
            "r_hold_scale": self.cfg["env"].get("holdRewardScale", 0.),
            "r_epstime_scale": self.cfg.get("epstimeRewardScale", [0.])[0],
            "r_scene_vel_scale": self.cfg.get("scevelRewardScale", [0.])[0],
            "r_steptime_scale": self.cfg.get("steptimeRewardScale", 0.),
            "r_scene_acc_scale": self.cfg.get("sceaccRewardScale", 0.),
            "r_arm_vel_scale": self.cfg.get("actvelPenaltyScale", 0.),
            "r_arm_acc_scale": self.cfg.get("actaccPenaltyScale", 0.),
        })
        if self.cfg.get("no_dense", False):
            self.cfg["env"]["sparse_reward"] = True

    # ----------------------------------------------------------------------
    # 2. Simulation Loop (Step, Reset, Refresh)
    # ----------------------------------------------------------------------
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        self.pre_physics_step(action_tensor)

        for i in range(self.num_inter_steps):
            self.deploy_joint_command(i)
            for j in range(self.control_freq_inv):
                if self.force_render: self.render()
                self.gym.simulate(self.sim)
            self.progress_buf += 1
        self.control_steps += 1

        # Match VecTask's camera tensor lifecycle so GPU camera tensors are read
        # only inside Isaac Gym's access window.
        self.gym.fetch_results(self.sim, True)

        if self.camera_rendering:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
        
        self.post_physics_step()

        if self.camera_rendering:
            self.gym.end_access_image_tensors(self.sim)
        self.timeout_buf = (
            (self.progress_buf >= self.max_episode_length - 1)
            & (self.reset_buf != 0)
            & (self.success_buf != 1)
        )

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)
        self.extras["success"] = self.success_buf.to(self.rl_device)
        self.update_observations_dict()

        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras


    def reset(self):
        self.update_observations_dict()
        return self.obs_dict


    def reset_all(self):
        reset_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self.success_counts[reset_ids] = self.save_threshold 
        self._update_props(reset_ids)
        self.reset_idx(reset_ids)
        self.compute_observations(reset_ids=reset_ids)


    def reset_done(self):
        done_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(done_env_ids) > 0:
            self.reset_idx(done_env_ids)
        self.update_observations_dict()
        return self.obs_dict, done_env_ids


    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        if self.force_render:
            self.gym.render_all_camera_sensors(self.sim)


    def _reset_bufs(self, env_ids):
        self.progress_buf[env_ids] = 0
        self.success_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.continuous_check_buf[env_ids] = 0
        self.force_has_applied[env_ids] = 0
        
        self.obs_buf[env_ids] = 0.
        self.states_buf[env_ids] = 0.
        self._reset_task_bufs(env_ids)

    
    def _reset_timeaware(self, env_ids):
        self._reset_timeaware_states(env_ids, use_pred_nets=self.cfg.get("use_pred_nets", False))
        self._reset_timeaware_bufs(env_ids)


    def _warmup_env(self):
        for _ in range(2): self.reset_idx(self.all_env_ids)
        for i in range(10): self.gym.simulate(self.sim)
        self.compute_observations(reset_ids=torch.arange(self.num_envs, device=self.device))

    # ----------------------------------------------------------------------
    # 3. State & Observation Management
    # ----------------------------------------------------------------------
    def compute_observations(self, reset_ids=[]):
        self._refresh()
        self._custom_refresh()
        self._update_states(reset_ids)
        self._collect_process_obs(reset_ids)
        self._collect_process_states()


    def _update_states(self, reset_ids=[]):
        self._update_robot_states()
        self._update_task_states() # mt50 some task states (sweep) depend on robot states
        self._reset_prev_states(reset_ids)
        self._update_diff_states()
        self._update_prev_states()

        self._update_timeaware_states()
        self._unify_quat_states()


    def _collect_process_obs(self, reset_ids=[]):
        obs_names = self.add_timeaware_obs(self.get_taskobs_names())
        cur_obs = self._stacking_obs(obs_names, self.add_obs_noise)
        cur_obs = self.maskout_buf(cur_obs, self.obs_slice, inplace=True)
        self.obs_buf[:] = cur_obs

        # TODO: Consider to move to the reset function later
        if len(reset_ids) > 0: # Unfortunately, we need observation to get the remaining time reset time states
            self.obs_buf[reset_ids] = self.maskout_all_timeaware(self.obs_buf[reset_ids], self.obs_slice)
            self._reset_timeaware(reset_ids)
            self.update_partial_obs(self.obs_buf, reset_ids, self.add_timeaware_obs([]), inplace=True)
            self.maskout_buf(self.obs_buf, self.obs_slice, inplace=True)


    def _collect_process_states(self):
        state_names = self.add_priv_timeaware_obs(self.add_priv_taskobs(self.get_taskobs_names()))
        cur_state = self._stacking_states(self.add_timeaware_obs(state_names))
        cur_state = self.maskout_buf(cur_state, self.state_slice, inplace=True)
        self.states_buf[:] = cur_state


    def _stacking_obs(self, obs_names, add_obs_noise=False):
        obs, index = [], 0
        for name in obs_names:
            piece = self.states[name].clone()
            self._apply_obj_target_obs_noise(name, piece)
            if add_obs_noise:
                # Apply per-task noise
                for tid in self.cur_dr_params.keys():
                    if name not in self.cur_dr_params[tid]["noise"]: continue
                    scale = self.cur_dr_params[tid]["noise"][name]
                    if scale == 0: continue

                    task_env_ids = (self.task_indices == tid).nonzero(as_tuple=True)[0]
                    if "quat" in name:
                        noise = quat_from_euler_xyz(*torch_rand_float(-scale, scale, (3, len(task_env_ids)), device=self.device))
                        piece[task_env_ids] = normalize(quat_mul(piece[task_env_ids], noise))
                    elif "time2end" in name:
                        piece[task_env_ids] = piece[task_env_ids] * (1 + torch_rand_float(-scale, scale, piece[task_env_ids].shape, device=self.device))
                    else:
                        piece[task_env_ids] = piece[task_env_ids] + torch_rand_float(-scale, scale, piece[task_env_ids].shape, device=self.device)
            
            obs.append(piece)
            self.obs_slice[name] = slice(index, index + piece.shape[-1])
            index += piece.shape[-1]
            if self.debug_viz: self.states[name] = piece
        return torch.cat(obs, dim=-1)

    def _apply_obj_target_obs_noise(self, name, piece):
        scale = self.obj_target_obs_noise_scale
        if scale <= 0.0:
            return
        if name == "target_pos":
            piece.add_(self._sample_obj_target_obs_noise(piece, scale))
        elif name == "obj_states":
            if piece.shape[-1] >= 3:
                piece[:, 0:3].add_(self._sample_obj_target_obs_noise(piece[:, 0:3], scale))
            if piece.shape[-1] >= 10:
                piece[:, 7:10].add_(self._sample_obj_target_obs_noise(piece[:, 7:10], scale))

    def _sample_obj_target_obs_noise(self, piece, scale):
        if self.obj_target_obs_noise_dist == "uniform":
            return torch_rand_float(-scale, scale, piece.shape, device=self.device).to(piece.dtype)
        return torch.randn_like(piece) * scale

    def set_obj_target_obs_noise(self, scale, dist=None, refresh_current_obs=False):
        self.obj_target_obs_noise_scale = float(scale or 0.0)
        if dist is not None:
            self.obj_target_obs_noise_dist = str(dist).lower()
        if refresh_current_obs:
            self._collect_process_obs(reset_ids=[])
            self.update_observations_dict()


    def _stacking_states(self, state_names):
        states, index = [], 0
        for name in state_names:
            piece = self.states[name]
            states.append(piece)
            self.state_slice[name] = slice(index, index + piece.shape[-1])
            index += piece.shape[-1]
        return torch.cat(states, dim=-1)


    def update_memory_buf(self, cur_obs):
        # TODO: use self.obs_buf to get the memory instead of list
        pass


    def _update_prev_states(self, real=False):
        states, prev_states = self.get_states_dict(real=real)
        for name in (self.get_robot_prev_state_names() + self.get_task_prev_state_names()):
            if name in states: prev_states[name] = mix_clone(states[name])


    def _reset_prev_states(self, reset_ids):
        if len(reset_ids) > 0:
            for key in [k for k in self.prev_states if k in self.states]:
                self.prev_states[key][reset_ids] = self.states[key][reset_ids].clone()
            for key in [k for k in self.prev_states if k not in self.states]:
                self.prev_states.pop(key)


    def _unify_quat_states(self, real=False):
        states, _ = self.get_states_dict(real=real)
        func = unify_quat if not real else unify_quat_np
        for key in [k for k in states.keys() if "quat" in k]:
            states[key] = func(states[key])


    def get_states_dict(self, real=False):
        return (self.states, self.prev_states) if not real else (self.states_real, self.prev_states_real)
    

    def update_observations_dict(self):
        self.obs_dict["states"] = torch.clamp(self.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    # ----------------------------------------------------------------------
    # 4. Rewards & Actions
    # ----------------------------------------------------------------------
    def compute_reward(self, actions):
        rewards, reset_buf, success_buf = self.compute_task_reward(actions)
        rewards = self.apply_timeaware_rewards(rewards, reset_buf, success_buf)
        self.rew_buf[:], self.reset_buf[:], self.success_buf[:] = rewards, reset_buf, success_buf
        self.num_episodes += self.reset_buf.sum().item()
        self._update_common_info()


    def apply_timeaware_rewards(self, rewards, reset_buf, success_buf):
        rs = self.reward_settings
        cost = torch.zeros(self.num_envs, self.cfg.get("num_cost", 2), device=self.device)

        # Stability costs
        rob_qvel_norm = self.states.get("rob_qvel_norm", torch.zeros((self.num_envs,), device=self.device))
        rob_qacc_norm = self.states.get("rob_qacc_norm", torch.zeros((self.num_envs,), device=self.device))
        
        focus_linvel = self.states["sce_linvel"].flatten()
        focus_linacc = self.states["sce_linacc"].flatten()
        sce_linvel_sum = torch.clamp((focus_linvel - self.linvel_max_gt if self.cfg.get("vel_match", False) else focus_linvel), min=0., max=self.MAX_VEL_NORM)
        
        rec_ids = self.all_env_ids if not self.use_staged_ctrl else torch.where(self.speed_describe_tensor[self.all_env_ids, self.cur_stage] == 0)[0]
        self.accm_instability[rec_ids] += focus_linvel[rec_ids]

        if not self.cfg.get("use_cost", False) and rs["r_scene_vel_scale"] > 0:
            rewards -= (rs["r_scene_vel_scale"] * sce_linvel_sum + rs["r_arm_vel_scale"] * rob_qvel_norm)
            rewards -= (rs["r_scene_acc_scale"] * focus_linacc + rs["r_arm_acc_scale"] * rob_qacc_norm)
            rewards -= rs["r_scene_vel_scale"] * self.accm_instability

        cost[:, 1] = sce_linvel_sum

        reward_without_timeaware = rewards.clone()

        # Time rewards are applied last so SIL can relabel only this component.
        timeaware_reward_bonus = torch.zeros_like(rewards)
        success_mask = success_buf.to(dtype=self.real_timecur.dtype)
        time_used = self.real_timecur
        time_ddl = self.real_time2end_init
        time_to_ddl = time_ddl - time_used
        abs_eps_time_diff = success_mask * torch.abs(time_to_ddl)
        
        if rs["r_epstime_scale"] > 0:
            if self.ratio_range is None: # Ensure the success reward is larger than epstimeRewardScale
                eps_time_reward = success_mask
                if self.cfg.get("update_ddl", False):
                    safe_time_used = torch.clamp(time_used, min=self.ctrl_dt)
                    eps_time_reward += success_mask * torch.clamp(time_ddl / safe_time_used, min=0.0, max=1.0).to(dtype=success_mask.dtype)
                # eps_time_reward = success_mask * (1 + (self.max_eps_time - time_used) / self.max_eps_time)
                # if self.cfg.get("update_ddl", False):
                #     tol = time_ddl * 0.1 # 10% fastest time as tolerance
                #     eps_time_reward += success_mask * (time_used <= time_ddl + tol).to(dtype=eps_time_reward.dtype)
            else:
                eps_time_reward = -torch.clamp(abs_eps_time_diff, max=5.) # Maximum time mismatch is 5 seconds
            timeaware_reward_bonus = rs["r_epstime_scale"] * eps_time_reward
            rewards = torch.clamp(rewards + timeaware_reward_bonus, min=0.)

        self.extras.update({
            "eps_time_p": abs_eps_time_diff,
            "cost": cost,
            "reward_without_timeaware": reward_without_timeaware,
            "timeaware_reward_bonus": timeaware_reward_bonus,
            "scene_linvel": focus_linvel, "accm_instability": self.accm_instability, 
            "scene_linvel_penalty": sce_linvel_sum, "scene_linacc_penalty": focus_linacc, 
            "rob_qvel_norm": rob_qvel_norm, "rob_qacc_norm": rob_qacc_norm
        })

        return rewards

    # ----------------------------------------------------------------------
    # 4.5 Actions
    # ----------------------------------------------------------------------
    def convert_actions(self, actions):
        actions = actions.to(self.device)

        if self.use_beta: 
            actions = actions * 2.0 - 1.0
        if self.training and self.add_act_noise:
            # Apply per-task action noise
            for tid in self.cur_dr_params.keys():
                task_env_ids = (self.task_indices == tid).nonzero(as_tuple=True)[0]
                noise_scale = self.cur_dr_params[tid]["noise"]["action"]
                if noise_scale == 0: continue
                actions[task_env_ids, :-1] = actions[task_env_ids, :-1] * (1 + torch_rand_float(-1, 1., actions[task_env_ids, :-1].shape, device=self.device) * noise_scale)
        actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        
        if (self.ratio_range is not None or self.goal_speed is not None) and self.cfg.get("scale_actions", False):
            scale_idx = self.time_ratio_buf < 1.
            if scale_idx.any(): actions[scale_idx, :-1] *= self.time_ratio_buf[scale_idx].unsqueeze(-1)

        self.raw_actions = actions.clone()
        self.actions = self.raw_actions.clone()
        return self.raw_actions

    # ----------------------------------------------------------------------
    # 5. Time-Aware & Curriculum Logic
    # ----------------------------------------------------------------------
    def timeaware_init(self):
        self.argument_related()
        self.maskout_related()
        self.stage_wise_ctrl_related()


    def argument_related(self):
        time2end_upper = self.max_episode_length * self.dt # max obs time2end
        if self.cfg.get("tw_train", False):
            self.max_episode_length = max(self.max_episode_length, int(self.max_episode_length / self.ratio_range[0]))
        self.max_eps_time = self.max_episode_length * self.dt # max simulation time

        self.num_episodes = 0
        self.save_threshold = self.cfg.get("save_threshold", 10)
        self.success_counts = torch.ones(self.num_envs, device=self.device) * self.save_threshold
        self.failure_counts = torch.zeros(self.num_envs, device=self.device)
        self.saved_eps, self.time_used_accm, self.max_linvel_accm, self.sum_linvel_accm = [], [], [], []
        self.obs_range = {
            "time2end": [
                torch.full((self.num_envs,), -time2end_upper, device=self.device),
                torch.full((self.num_envs,), time2end_upper, device=self.device),
            ],
            "time_ratio": torch.tensor([0.2, 1.], device=self.device),
            "max_linvel": torch.tensor([0., 3.], device=self.device),
        }
        # time_cur is at index 1; sce_linvel follows the public time fields.
        self.extras["tobs_idx"], self.extras["vobs_idx"] = 1, 3


    def maskout_related(self):
        self.maskout_names_full = ["time_ratio", "time_cur", "ddl", "sce_linvel", "lim_linvel"]
        maskout_set = set()
        
        # Mask all time-aware features during training from scratch or when fixing privileged info
        if self.cfg.get("fix_priv", False):
            maskout_set.update(self.maskout_names_full)
        # Mask time features if not using time2end
        if not self.cfg.get("time2end", False):
            maskout_set.update(["time_ratio", "time_cur", "ddl"])
        
        # Mask stability features based on config
        if self.cfg.get("fix_linvel", False):
            maskout_set.add("sce_linvel")
        if self.cfg.get("fix_limvel", False):
            maskout_set.add("lim_linvel")
        
        self.maskout_names = list(maskout_set)


    def stage_wise_ctrl_related(self):
        self.use_staged_ctrl = self.cfg.get("budget_portion", None) is not None
        if self.use_staged_ctrl:
            self.budget_portion = bp = self.cfg["budget_portion"]
            self.speed_describe = sd = self.cfg["speed_describe"]
            self.fast_portion = sum(bp[i] for i in range(len(bp)) if sd[i] == 1)
            self.slow_portion = sum(bp[i] for i in range(len(bp)) if sd[i] != 1)
            assert np.allclose(sum(bp), 1) and self.fast_portion > 0 and self.slow_portion > 0
            self.stage_time_ratio_buf = torch.zeros((self.num_envs, len(bp)), device=self.device)
            self.real_time_milestone = torch.tensor([sum(bp[:i+1]) * self.goal_time for i in range(len(bp))]).repeat(self.num_envs, 1).to(self.device)
            self.speed_describe_tensor = torch.tensor(sd, device=self.device).repeat(self.num_envs, 1)
            self.cur_stage = torch.zeros((self.num_envs, ), device=self.device, dtype=torch.long)


    def _update_timeaware_states(self, real=False):
        states, _ = self.get_states_dict(real=real)
        run_idx = self.continuous_check_buf == 0
        self.real_timecur[run_idx] += self.ctrl_dt
        self.real_time2end[run_idx] -= self.ctrl_dt
        
        self._update_stage_wise_time_ratio()
        # Update time to end
        if self.cfg.get("time_ratio", False):
            self.time2end[run_idx] -= (self.ctrl_dt * self.time_ratio_buf[run_idx])
        else:
            self.time2end[run_idx] = self.time2end[run_idx] * (self.prev_time_ratio_buf[run_idx] / self.time_ratio_buf[run_idx]) - self.ctrl_dt
        
        if not self.training and self.cfg.get("real_robot", False):
            print(f"CurTime2End: {self.time2end[0].item():.5f}; TimeBudgetReal: {self.real_time2end[0].item():.5f}")

        # Update scene instability (scevel is the raw stability return by task)
        self.sce_linvel_buf[:] = states.get("scevel", self.sce_linvel_buf)
        self.max_linvel_buf[:] = torch.max(self.max_linvel_buf, self.sce_linvel_buf)
        
        # Update manipulation time
        run_ids = run_idx.nonzero(as_tuple=False).flatten()
        if len(run_ids) > 0:
            man_ids = run_ids[self.sce_linvel_buf[run_ids] >= 0.01]
            if len(man_ids) > 0: self.interaction_time[man_ids] += self.ctrl_dt

        # Update dict
        ta_states = {
            "time_cur": self.real_timecur, "time_ratio": self.time_ratio_buf,
            "time2end": self.time2end, "ddl": self.real_time2end_init,
            "sce_linvel": self.sce_linvel_buf, "max_linvel": self.max_linvel_buf, "lim_linvel": self.linvel_max_gt,
            "sce_linacc": self.sce_linacc_buf
        }
        for k, v in ta_states.items():
            states[k] = v[0].cpu().numpy() if real else (v.view(-1, 1) if v.dim() == 1 else v)


    def _update_stage_wise_time_ratio(self):
        if self.use_staged_ctrl:
            ids = torch.arange(self.cur_stage.size(0), device=self.device)
            next_ids = torch.where(self.real_timecur > self.real_time_milestone[ids, self.cur_stage])[0]
            self.cur_stage[next_ids] = torch.clamp(self.cur_stage[next_ids] + 1, max=len(self.budget_portion) - 1)
            self.update_time_ratio_buf(self.stage_time_ratio_buf[ids, self.cur_stage], env_ids=ids)
            self.update_linvel_gt()


    def _reset_timeaware_bufs(self, env_ids):
        for buf in [self.real_timecur, self.interaction_time, self.sce_linvel_buf, self.max_linvel_buf, self.sce_linacc_buf, self.accm_instability]:
            buf[env_ids] = 0.
        self.time2end[env_ids] = self.time2end_init[env_ids]
        self.real_time2end[env_ids] = self.real_time2end_init[env_ids]
        self.prev_time_ratio_buf[env_ids] = self.time_ratio_buf[env_ids].clone()
        self.recompute_staged_time_ratio(env_ids)


    def _reset_timeaware_states(self, env_ids, use_pred_nets=False):
        # Speed Ratio
        if self.goal_speed is not None:
            self.time_ratio_buf[env_ids] = self.goal_speed
        elif self.cfg.get("warmup_rand", False):
            self.time_ratio_buf[env_ids] = torch_rand_float(*self.obs_range["time_ratio"], (1, len(env_ids)), device=self.device)
        elif self.ratio_range is not None:
            probs = torch.rand(len(env_ids), device=self.device)
            # Extreme values
            self.time_ratio_buf[env_ids[probs <= 0.]] = self.ratio_range[0]
            self.time_ratio_buf[env_ids[probs >= 1.]] = self.ratio_range[1]
            mid = (probs > 0.) & (probs < 1.)
            if mid.any(): self.time_ratio_buf[env_ids[mid]] = torch_rand_float(self.ratio_range[0], self.ratio_range[1], (1, mid.sum()), device=self.device)

        # Time2End & LimVel
        if use_pred_nets and self.agent is not None:
            init_t2e = self.agent.predict_remaining_time(self.obs_buf[env_ids])
            init_limvel = self.agent.predict_max_instability(self.obs_buf[env_ids])
            if self.goal_time is not None: self.time_ratio_buf[env_ids] = init_t2e / self.goal_time
            self._init_t2ebuf_linvelbuf(init_t2e, init_limvel, env_ids)
        elif self.cfg.get("warmup_rand", False):
            init_t2e = self.sample_time2end(env_ids)
            init_limvel = torch_rand_float(*self.obs_range["max_linvel"], (1, len(env_ids)), device=self.device)
            self._init_t2ebuf_linvelbuf(init_t2e, init_limvel, env_ids)
        else:
            _, time2end_upper = self.get_time2end_bounds(env_ids)
            self.time2end_init[env_ids] = time2end_upper
            self.real_time2end_init[env_ids] = self.time2end_init[env_ids]


    def _update_time2end_upper(self, env_ids, new_upper):
        new_upper = torch.clamp(new_upper, min=self.ctrl_dt, max=self.max_eps_time)
        self.obs_range["time2end"][0][env_ids] = -new_upper
        self.obs_range["time2end"][1][env_ids] = new_upper

    
    def _get_avg_time2end_upper(self):
        return self.obs_range["time2end"][1].mean().item()


    def _init_t2ebuf_linvelbuf(self, init_t2e, init_limvel, env_ids):
        # T2E
        if self.cfg.get("time_ratio", False):
            self.time2end_init[env_ids] = init_t2e
            self.real_time2end_init[env_ids] = init_t2e / self.time_ratio_buf[env_ids]
        else:
            self.time2end_init[env_ids] = init_t2e / self.time_ratio_buf[env_ids]
            self.real_time2end_init[env_ids] = self.time2end_init[env_ids]
        
        # LimVel
        self.linvel_max_gt_init[env_ids] = init_limvel
        self.update_linvel_gt(env_ids)


    def recompute_staged_time_ratio(self, reset_ids=[]):
        if self.use_staged_ctrl and len(reset_ids) > 0:
            avg = self.time_ratio_buf[reset_ids].clone()
            if ((avg < self.ratio_range[0]) | (avg > self.ratio_range[1])).any():
                raise ValueError(f"Time usage range issue: [{self.time2end_init[reset_ids].min()}, {self.time2end_init[reset_ids].max()}]")
            
            if self.cfg.get("use_avg_speed", False):
                fast, slow = avg, avg
            else:
                ratio = self.slow_portion / self.fast_portion
                d_slow = torch.min(avg - self.ratio_range[0], (self.ratio_range[1] - avg) / ratio)
                fast, slow = avg + d_slow * ratio, avg - d_slow
            
            for i in range(len(self.budget_portion)):
                self.stage_time_ratio_buf[reset_ids, i] = fast if self.speed_describe[i] == 1 else slow
            self.cur_stage[reset_ids] = 0


    def update_linvel_gt(self, env_ids=None):
        ids = env_ids if env_ids is not None else self.all_env_ids
        sched = self.cfg.get("scevelSchedule", 1.0)
        scaler = self.time_ratio_buf[ids] ** sched if self.cfg.get("exp_scheduler", False) else self.time_ratio_buf[ids] * sched
        self.linvel_max_gt[ids] = scaler * self.linvel_max_gt_init[ids]


    def update_time_ratio_buf(self, new_time_ratio, env_ids=None):
        ids = env_ids if env_ids is not None else self.all_env_ids
        self.prev_time_ratio_buf[ids] = self.time_ratio_buf[ids].clone()
        self.time_ratio_buf[ids] = new_time_ratio


    def add_timeaware_obs(self, obs_names): return ["time_ratio", "time_cur", "ddl"] + obs_names
    def add_priv_timeaware_obs(self, state_names): return ["sce_linvel", "lim_linvel"] + state_names
    
    def update_partial_obs(self, cur_obs, env_ids, names, inplace=False):
        res_obs = cur_obs if inplace else cur_obs.clone()
        for name in names:
            piece = self.states[name]
            res_obs[env_ids, self.obs_slice[name]] = piece[env_ids]
        return res_obs

    def maskout_buf(self, buf, buf_slice, inplace=False):
        res_buf = buf if inplace else buf.clone()
        for name in [n for n in self.maskout_names if n in buf_slice]: res_buf[..., buf_slice[name]] = 1.
        return res_buf
    def maskout_all_timeaware(self, buf, buf_slice, inplace=False):
        res_buf = buf if inplace else buf.clone()
        for name in [n for n in self.maskout_names_full if n in buf_slice]: res_buf[..., buf_slice[name]] = 1.
        return res_buf

    # ----------------------------------------------------------------------
    # 6. Real Robot Interface
    # ----------------------------------------------------------------------
    def _init_real_robot_mode(self, state_estimator, robot):
        self._update_states_real(state_estimator, robot, max_trials=3)
        if hasattr(self, "env_configs") and self.env_configs is not None:
            self._reset_timeaware_states_real(self._estimate_minimum_time2end(self.env_configs, self.states_real))
        obs_buf, extras = self.compute_observations_real(state_estimator, robot)
        
        diff = np.abs(self._dof_pos.cpu().numpy()[0] - self.states_real["q"])
        if np.max(diff) > 0.01: raise ValueError(f"Sim/Real mismatch. Max diff: {np.max(diff):.4f}")
        return obs_buf, extras


    def compute_observations_real(self, state_estimator, robot, max_trials=3):
        self._update_states_real(state_estimator, robot, max_trials=max_trials)
        obs_names = self.add_timeaware_obs(self.get_taskobs_names())
        cur_obs = self._stacking_obs_real(obs_names)
        cur_obs = self.maskout_buf(cur_obs, self.obs_slice, inplace=True)
        self.map_real2sim()
        return cur_obs, self.extras
    

    def compute_reward_real(self):
        done = False
        if self.goal_speed is not None or self.goal_time is not None:
            grace = 5 if self.cfg.get("task_name", "") == "FrankaGmPour" else 2
            done = (self.time2end[0] <= -grace)
        return to_torch([0.0], device=self.device), to_torch([done], device=self.device), to_torch([False], device=self.device)
    

    def _estimate_minimum_time2end(self, data_dict, new_config, k: int = 5, use_normalization: bool = False):
        features = self._compute_init_config_features(data_dict).cpu().numpy()
        times = self.env_configs["time_used"].cpu().numpy()
        new_feat = self.extract_features(new_config, self._time_related_state_names())
        
        if use_normalization:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
            new_feat = scaler.transform(new_feat.reshape(1, -1)).flatten()
        
        dists = self.calculate_weighted_distances(new_feat, features)
        k_idx = np.argpartition(dists, k)[:k]
        # Sort by distance
        sorted_k = k_idx[np.argsort(dists[k_idx])]
        est_time = np.mean(times[sorted_k]).item()
        print(f"\nEstimate Real World Time2End: {est_time:.3f}s using k-NN with k={k}\n")
        return est_time


    def _reset_timeaware_states_real(self, init_t2e):
        if self.goal_speed is not None: self.time_ratio_buf[:] = self.goal_speed
        if self.goal_time is not None: self.time_ratio_buf[:] = init_t2e / self.goal_time
        
        val = init_t2e if self.cfg.get("time_ratio", False) else init_t2e / self.time_ratio_buf
        self.time2end_init[:] = val
        self.real_time2end_init[:] = init_t2e / self.time_ratio_buf if self.cfg.get("time_ratio", False) else val
        self._reset_bufs(self.all_env_ids)


    def _update_states_real(self, state_estimator, robot, max_trials, reset_ids=[]):
        obj, arm = self._get_states_with_retry(state_estimator, robot, max_trials)
        self.states_real.update({**obj, **arm})
        self._update_diff_states(real=True)
        self._update_timeaware_states(real=True)
        self._update_prev_states(real=True)
        self._unify_quat_states(real=True)
        self._update_common_info_real()
        self._update_debug_info_real()


    def _get_states_with_retry(self, state_estimator, robot, max_trials):
        for _ in range(max_trials):
            o, a = self._update_task_states_real(state_estimator), self._update_robot_states_real(robot)
            if o is not None and a is not None: return o, a
        if o is None: raise TimeoutError("Failed to get cube poses.")
        if a is None: raise TimeoutError("Failed to get robot state.")


    def _update_robot_states_real(self, robot):
        for _ in range(100):
            st = robot.get_state()
            if st is not None: return self._preprocess_robot_states_real(st)
        if self.cfg.get("use_sim_pure", False) and not self.cfg.get("use_fk_replay", False): return None
        raise ValueError("Failed to get robot state.")


    def _stacking_obs_real(self, obs_names):
        obs, index = [], 0
        for name in obs_names:
            p = to_numpy(self.states_real[name])
            if len(p.shape) == 0: p = np.expand_dims(p, axis=0)
            obs.append(p)
            self.obs_slice[name] = slice(index, index + p.shape[-1])
            index += p.shape[-1]
        return to_torch(np.concatenate(obs), device=self.device).unsqueeze(0)


    def update_memory_buf_real(self, cur_obs):
        # TODO: use self.obs_buf to get the memory
        pass


    def _update_common_info_real(self):
        self.extras.update({
            "observed_time2end": self.time2end[0].clone(), "real_time2end": self.real_time2end[0].clone(),
            "eps_time_goal": self.real_time2end_init.clone(), "time_ratio": self.time_ratio_buf.clone(),
            "scene_linvel": self.states_real["sce_linvel"], "scene_linvel_lim": self.states_real["lim_linvel"],
        })
        self._update_task_common_info_real()

    # ----------------------------------------------------------------------
    # 7. Utils, Logging & Abstract Methods
    # ----------------------------------------------------------------------
    def get_time2end_bounds(self, env_ids=None):
        lower, upper = self.obs_range["time2end"]
        if env_ids is None:
            return lower, upper
        return lower[env_ids], upper[env_ids]


    def sample_time2end(self, env_ids):
        lower, upper = self.get_time2end_bounds(env_ids)
        return torch_rand_tensor(lower, upper, device=self.device)
    
    
    def init_cur_dr_params(self, init_curri_ratio=0.):
        """Initialize DR params for all tasks. For single-task, task_id=0."""
        # Get task list: use task_idx if multi-task, otherwise [0]
        task_list = self.task_idx if hasattr(self, 'task_idx') and self.task_idx else [0]
        
        for tid in task_list:
            self.curri_ratio_per_task[tid] = init_curri_ratio
            self.cur_dr_params[tid] = {k: {} for k in self.dr_settings.keys()}
            for dname, ddict in self.dr_settings.items():
                for dk, dv in ddict.items():
                    if self.training or (self.cfg.get("apply_noise_eval", False) or dname in ["spatial", "controller"]):
                        self.cur_dr_params[tid][dname][dk] = dv[0] + init_curri_ratio * (dv[1] - dv[0])
                    else:
                        self.cur_dr_params[tid][dname][dk] = dv[0]

            self.update_max_joint_velocity(task_id=tid)


    def update_dr_params(self, curri_ratio=0., task_id=None):
        """Update DR params. If task_id=None, update all tasks; otherwise update specific task."""
        task_list = [task_id] if task_id is not None else list(self.cur_dr_params.keys())
        
        for tid in task_list:
            self.curri_ratio_per_task[tid] = curri_ratio
            for dname, ddict in self.dr_settings.items():
                for dk, dv in ddict.items():
                    self.cur_dr_params[tid][dname][dk] = dv[0] + curri_ratio * (dv[1] - dv[0])
        
            self.update_max_joint_velocity(task_id=tid)


    def _update_common_info(self):
        self.extras.update({
            "eps_time": self.real_timecur, "eps_horizon": self.progress_buf,
            "eps_time_p": self.extras.get("eps_time_p", torch.zeros_like(self.rew_buf)),
            "scene_linvel_penalty": self.extras.get("scene_linvel_penalty", torch.zeros_like(self.rew_buf)),
            "scene_linacc_penalty": self.extras.get("scene_linacc_penalty", torch.zeros_like(self.rew_buf)),
            "eps_lim_scevel": self.states["lim_linvel"].flatten(), "eps_max_scevel": self.states["max_linvel"].flatten(),
            "eps_sum_inst": self.extras.get("accm_instability", torch.zeros_like(self.rew_buf)),
        })
        if not self.training:
            env_id = 0
            self.extras.update({
                "observed_time2end": self.time2end[env_id].clone(), "real_time2end": self.real_time2end[env_id].clone(),
                "real_cur_time": self.real_timecur[env_id].clone(), "time_ratio": self.time_ratio_buf[env_id].clone(),
                "eps_time_goal": self.real_time2end_init.clone(), "interaction_time": self.interaction_time.flatten(),
                "scene_linvel": self.states["sce_linvel"][env_id].clone(), "scene_linvel_lim": self.states["lim_linvel"][env_id].clone(),
            })
            self._update_task_common_info()


    def get_config_idx(self, env_ids):
        if not self.cfg.get("fixed_configs", False): return None
        if self.cfg.get("specific_idx", None) is not None: return torch.tensor([self.cfg["specific_idx"]], device=self.device, dtype=torch.long)
        if self.cfg.get("update_configs", False): return self.config_ids[env_ids]
        return torch.randint(0, self.num_configs, (len(env_ids),), device=self.device)


    def filter_env_ids(self, env_ids, config_index):
        if self.is_ready_to_record(env_ids):
            failed = (self.reset_buf * (1 - self.success_buf)).nonzero(as_tuple=False).squeeze(-1)
            self.failure_counts[failed] += 1
            new_sub = ((self.success_counts[env_ids] >= self.save_threshold) | (self.failure_counts[env_ids] >= self.save_threshold)).nonzero(as_tuple=False).squeeze(-1)
            new_env_ids = env_ids[new_sub]
            config_index = config_index[new_sub] if config_index is not None else None
            self.success_counts[new_env_ids] = 0
            self.failure_counts[new_env_ids] = 0
            return new_env_ids, config_index
        return env_ids, config_index


    def record_init_configs(self, env_ids):
        if "init_configs" not in self.extras:
            self.extras["init_configs"] = {"time_used": [], "max_linvel": [], "sum_linvel": []}
            self._init_task_configs_buf()
        
        ic = self.extras["init_configs"]
        cur = len(ic["time_used"])
        self._record_task_init_configs(env_ids)
        for k in ["time_used", "max_linvel", "sum_linvel"]: ic[k].extend([0] * len(env_ids))
        self.time_used_accm.extend([0] * len(env_ids))
        self.max_linvel_accm.extend([0] * len(env_ids))
        self.sum_linvel_accm.extend([0] * len(env_ids))
        self.env2index[env_ids] = torch.arange(cur, cur + len(env_ids), dtype=torch.long, device=self.device)


    def record_post_config(self, success_ids):
        if "init_configs" not in self.extras: return
        ic = self.extras["init_configs"]
        buf_idx = self.env2index[success_ids]
        vals = [self.real_timecur[success_ids].cpu().tolist(), self.max_linvel_buf[success_ids].cpu().tolist(), self.accm_instability[success_ids].cpu().tolist()]
        
        for i, sid in enumerate(success_ids):
            idx = buf_idx[i]
            if idx == -1 or (self.cfg.get("update_configs", False) and (sid >= self.num_configs or sid in self.saved_eps)): continue
            
            self.success_counts[sid] += 1
            self.time_used_accm[idx] += vals[0][i]
            self.max_linvel_accm[idx] += vals[1][i]
            self.sum_linvel_accm[idx] += vals[2][i]
            
            if self.success_counts[sid] >= self.save_threshold:
                count = self.success_counts[sid].item()
                ic["time_used"][idx] = self.time_used_accm[idx] / count
                ic["max_linvel"][idx] = self.max_linvel_accm[idx] / count
                ic["sum_linvel"][idx] = self.sum_linvel_accm[idx] / count
                self.saved_eps.append(sid)
        
        self.extras["num_eps_recorded"] = len(self.saved_eps)
        if self.cfg.get("update_configs", False): self.extras["update_done"] = len(self.saved_eps) == self.num_configs


    def _reset_init_cube_state(self, cube_name, cube_sizes, surface2cube_z, other_cube_state, other_cube_sizes, env_ids, check_valid=True):
        if env_ids is None: env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        num = len(env_ids)
        sampled = torch.zeros(num, 13, device=self.device)
        min_dists = (cube_sizes[:, 0] + other_cube_sizes[:, 0])[env_ids] * np.sqrt(2) * 0.8
        
        sign = 1.0 if "B" in cube_name else -1.0
        center = self._ws_surface_pos[:2] + torch.tensor([0, sign * torch.max(cube_sizes[:, 0][env_ids], other_cube_sizes[:, 0][env_ids]) * np.sqrt(2)], device=self.device)
        sampled[:, 2], sampled[:, 6] = self._ws_surface_pos[2] + surface2cube_z[env_ids], 1.0
        
        active = torch.arange(num, device=self.device)
        # Use first task's spatial params (spatial randomization is typically uniform)
        first_tid = list(self.cur_dr_params.keys())[0]
        for _ in range(200):
            delta = sign * self.cur_dr_params[first_tid]["spatial"][f"{cube_name}_pos"] * (torch.ones_like(sampled[active, :2]) if self.cfg.get("max_dist", False) else 2.0 * (torch.rand_like(sampled[active, :2]) - 0.5))
            sampled[active, :2] = torch.clamp(center[active] + delta, self._ws_surface_pos[:2]-self._ws_upper_bounds[:2], self._ws_surface_pos[:2]+self._ws_upper_bounds[:2])
            if not check_valid: break
            active = torch.nonzero(torch.linalg.norm(sampled[:, :2] - other_cube_state[env_ids, :2], dim=-1) < min_dists, as_tuple=True)[0]
            if len(active) == 0: break
        
        if self.cur_dr_params[first_tid]["spatial"][f"{cube_name}_quat"] > 0:
            rot = torch.zeros(num, 3, device=self.device)
            rot[:, 2] = self.cur_dr_params[first_tid]["spatial"][f"{cube_name}_quat"] * 2.0 * (torch.rand(num, device=self.device) - 0.5)
            sampled[:, 3:7] = quat_mul(axisangle2quat(rot), sampled[:, 3:7])
        return sampled


    def apply_disturbances(self):
        ready = torch.where(self.states["approaching_dist"] < 0.05, 1, 0) * (self.force_has_applied == 0)
        ids = torch.nonzero(ready, as_tuple=True)[0]
        if len(ids) > 0:
            f = torch_rand_float(-1, 1, (len(ids), 3), device=self.device)
            f[:, 2] = 0.
            self.apply_rigid_body_force(ids, f / torch.linalg.norm(f, dim=-1, keepdim=True) * self.disturbance_v)
            self.force_has_applied[ids] = 1


    def apply_rigid_body_force(self, env_id, forces, body_handle=None):
        h = self.link_handles[self.apply_force_handle] if body_handle is None else body_handle
        af = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        af[env_id, h] = forces
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(af), None, gymapi.ENV_SPACE)


    def adjust_viewer(self, cam_pos=None, cam_target=None):
        if cam_pos and cam_target and not self.headless:
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)


    def get_viewer_image(self, env_id=0):
        return self.gym.get_camera_image(self.sim, self.envs[env_id], self.camera, gymapi.IMAGE_COLOR).reshape(self.camera_h, self.camera_w, 4)


    def draw_point(self, pos, ori=None):
        g = gymutil.WireframeSphereGeometry(0.01, 12, 12, gymapi.Transform(r=gymapi.Quat(*(ori or [0,0,0,1]))), color=(1, 1, 0))
        gymutil.draw_lines(g, self.gym, self.viewer, self.envs[0], gymapi.Transform(p=gymapi.Vec3(*pos)))


    def draw_axes(self, pos, ori=None):
        g = gymutil.AxesGeometry(0.1, gymapi.Transform(r=gymapi.Quat(*(ori or [0,0,0,1]))))
        gymutil.draw_lines(g, self.gym, self.viewer, self.envs[0], gymapi.Transform(p=gymapi.Vec3(*pos)))


    def clear_lines(self): self.gym.clear_lines(self.viewer)
    def update_debug_info(self, name, value): self.debug_info.setdefault(name, []).append(value)
    def is_warmup_done(self): return self.num_episodes >= self.cfg.get("warmup_episodes", 0)
    def is_ready_to_record(self, env_ids): return self.cfg.get("record_init_configs", False) and self.is_warmup_done() and len(env_ids) > 0
    

    # Empty / Abstract Placeholders
    def curriculum_noise(self, dr_params, env): pass
    def curriculum_properties_dr(self, dr_params, env): pass
    def curriculum_ctrl_freq(self, dr_params, env): pass
    def priv_obs_init(self): self.cfg["env"]["numStates"] += 2; self.task_priv_obs_init()
    def _compute_init_config_features(self, data_dict): pass
    def _time_related_state_names(self): pass
    def map_real2sim(self): pass
    def post_physics_step_real(self, actions): pass
    def _update_debug_info_real(self): pass
    def _update_props(self, reset_ids): pass
    def _update_task_common_info(self): pass
    @abc.abstractmethod
    def compute_task_reward(self): """Compute task reward."""
    
    @abc.abstractmethod
    def get_taskobs_names(self): """Observation names from states dict."""
    def add_priv_taskobs(self, task_obs_names): return []
    def get_robot_prev_state_names(self): return []
    def get_task_prev_state_names(self): return []

    @abc.abstractmethod
    def deploy_joint_command(self, index=-1): """Deploy joint commands."""
    @abc.abstractmethod
    def _update_task_states_real(self, state_estimator): """Update task states real."""
    @abc.abstractmethod
    def pre_physics_step_real(self, actions): """Pre physics step real."""
    @abc.abstractmethod
    def _update_task_common_info_real(self): """Update task info real."""
    @abc.abstractmethod
    def _create_ground_plane(self): """Create ground plane."""
    @abc.abstractmethod
    def _custom_refresh(self): """Refresh sliced tensor."""
    @abc.abstractmethod
    def _create_envs(self, num_envs, spacing, num_per_row): """Create envs."""
    @abc.abstractmethod
    def robot_variable_init(self): """Init robot variables."""
    @abc.abstractmethod
    def _update_task_states(self): """Update task states."""
    @abc.abstractmethod
    def _update_robot_states(self): """Update robot states."""
    @abc.abstractmethod
    def _update_diff_states(self, real=False): """Update diff states."""
