from collections import defaultdict
from typing import Any, Dict, Tuple
import isaacgym
import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import to_torch, tensor_clamp, tf_combine, quat_apply, tf_apply, torch_rand_float, tf_inverse
from isaacgymenvs.utils.ctrl_utils import FrankaController
from isaacgymenvs.utils.task_utils import mix_clone
from isaacgymenvs.tasks.base.time_vec_task import TimeVecTask
from isaacgymenvs.tasks.franka.vec_task import task_fns

from skrl.utils.isaacgym_utils import ik

TASK_IDX_TO_NAME = {
    0:  "assembly",
    1:  "basketball",
    2:  "bin_picking",
    3:  "box_close",
    4:  "button_press_topdown",
    5:  "button_press_topdown_wall",
    6:  "button_press",
    7:  "button_press_wall",
    8:  "coffee_button",
    9:  "coffee_pull",
    10: "coffee_push",
    11: "dial_turn",
    12: "disassemble",
    13: "door_close",
    14: "door_lock",
    15: "door_unlock",
    16: "door_open",
    17: "drawer_close",
    18: "drawer_open",
    19: "faucet_close",
    20: "faucet_open",
    21: "hammer",
    22: "hand_insert",
    23: "handle_press_side",
    24: "handle_press",
    25: "handle_pull_side",
    26: "handle_pull",
    27: "lever_pull",
    28: "peg_insert_side",
    29: "peg_unplug_side",
    30: "pick_out_of_hole",
    31: "pick_place",
    32: "pick_place_wall",
    33: "plate_slide_back_side",
    34: "plate_slide_back",
    35: "plate_slide_side",
    36: "plate_slide",
    37: "push_back",
    38: "push",
    39: "push_wall",
    40: "reach",
    41: "reach_wall",
    42: "shelf_place",
    43: "soccer",
    44: "stick_pull",
    45: "stick_push",
    46: "sweep_into_goal",
    47: "sweep",
    48: "window_close",
    49: "window_open",
    50: "cube_stack",
    51: "gm_pouring",
    52: "drawer_opening"
}
FRANKA_DEFAULT_DOF_STATES = [
    [-0.25, .4, 0, -2.2, -0.17, 2.6, -0.7, 0.04, 0.04], # 0
    [0,-.3, 0, -2.8, -.17, 2.6, 0.7, 0.04, 0.04],
    [.15, 0.3, 0, -2, 0, 2.45, 0.7, 0.04, 0.04],
    [0, -.4, 0, -2.9, -0.17, 2.6, -0.7, 0.04, 0.04],
    [0,-.1, 0, -2.6, -.17, 2.6, 0.7, 0.04, 0.04],
    [0,-.1, 0, -2.6, -.17, 2.6, 0.7, 0.04, 0.04],
    [0,-.2, 0, -2.9, -.17, 2.6, 0.7, 0.04, 0.04],
    [0,-.2, 0, -2.9, -.17, 2.6, 0.7, 0.04, 0.04],
    [0, -0.2, 0, -2.7, -.17, 2.6, 0.7, 0.04, 0.04],
    [0, 0.1, 0, -2.5, -.17, 2.6, 0.7, 0.04, 0.04],
    [0, 0.1, 0, -2.5, -.17, 2.6, 0.7, 0.04, 0.04], # 10
    [0, 0.6, 0, -1.7, 0, 2.4, .7, 0.04, 0.04],
    [-0.25, .3, 0, -2.4, -0.17, 2.6, -0.7, 0.04, 0.04],
    [.45, .85, 0.14485, -1.25, -0.2897, 2, -0.1, 0.04, 0.04], # door_close has a different init pos because it must not intersect with the door
    [0, 0, 0, -2.7, -.17, 2.8, 0.7, 0.04, 0.04],
    [0, 0, 0, -2.7, -.17, 2.8, 0.7, 0.04, 0.04],
    [0, 0, 0, -2.7, -.17, 2.8, 0.7, 0.04, 0.04],
    [0, -0.2, 0, -2.9, -.17, 2.8, 0.7, 0.04, 0.04],
    [0, -0.2, 0, -2.9, -.17, 2.8, 0.7, 0.04, 0.04],
    [0, .2, 0, -2.5, -.17, 2.8, 0.7, 0.04, 0.04],
    [0, .2, 0, -2.5, -.17, 2.8, 0.7, 0.04, 0.04],  # 20
    [.2, 0.1, 0, -2.6, -0.17, 2.6, -0.7, 0.04, 0.04],
    [0,.2, 0, -2.5, -.17, 2.6, 0.7, 0.04, 0.04],
    [0, -0.3, 0, -2.5, -.17, 2.6, 0, 0.04, 0.04],
    [0, 0.1, 0, -2.5, -.17, 2.6, 0.7, 0.04, 0.04],
    [0, -0.3, 0, -2.5, -.17, 2.6, 0, 0.04, 0.04],
    [0, 0.1, 0, -2.5, -.17, 2.6, 0.7, 0.04, 0.04],
    [0, -0.2, 0, -2.9, -.17, 2.8, 0.7, 0.04, 0.04],
    [0, 0.4, 0, -2.2, 0, 2.5, -0.7, 0.04, 0.04],
    [0,-.1, 0, -2.6, -.17, 2.6, 0.7, 0.04, 0.04], # peg unplug
    [0, 0.75, 0, -1.6, -0.17, 2.5, .8, 0.04, 0.04], # 30
    [0,.2, 0, -2.5, -.17, 2.6, -0.7, 0.04, 0.04],
    [0,.2, 0, -2.5, -.17, 2.6, -0.7, 0.04, 0.04],
    [0,.25, 0, -2.6, -.17, 2.8, -0.7, 0.04, 0.04],
    [0, .2, 0, -2.5, -.17, 2.7, 0.7, 0.04, 0.04],
    [0,.25, 0, -2.6, -.17, 2.8, -0.7, 0.04, 0.04],
    [0,.25, 0, -2.6, -.17, 2.8, -0.7, 0.04, 0.04],
    [0, .2, 0, -2.5, -.17, 2.6, -0.7, 0.04, 0.04],
    [0, .2, 0, -2.5, -.17, 2.6, -0.7, 0.04, 0.04],
    [0, .2, 0, -2.5, -.17, 2.6, -0.7, 0.04, 0.04],
    [0, .2, 0, -2.5, -.17, 2.6, -0.7, 0.04, 0.04], # 40
    [0, .2, 0, -2.5, -.17, 2.6, -0.7, 0.04, 0.04],
    [0, .2, 0, -2.5, -.17, 2.6, 0.7, 0.04, 0.04],
    [0, -.3, 0, -2.8, -.17, 2.6, 0.7, 0.04, 0.04],
    [0, 0.6, 0, -1.45, 0, 2.06, -.87, 0.04, 0.04],
    [0, 0.6, 0, -1.45, 0, 2.06, -.87, 0.04, 0.04], # 45
    [0, .2, 0, -2.5, -.17, 2.6, -0.7, 0.04, 0.04],
    [0, .2, 0, -2.5, -.17, 2.6, -0.7, 0.04, 0.04],
    [0, .2, 0, -2.5, -.17, 2.8, 0.7, 0.04, 0.04],
    [0, .2, 0, -2.5, -.17, 2.8, 0.7, 0.04, 0.04], # 49
    [0, 0, 0, -2.3180, 0, 2.4416, 0.7854, 0.04, 0.04],
    [0, 0, 0, -2.3180, 0, 2.4416, 0.7854, 0.04, 0.04],
    [0, 0, 0, -2.3180, 0, 2.4416, 0.7854, 0.04, 0.04]
]

# FRANKA_DEFAULT_DOF_STATES = [[0,-.2, 0, -2.9, -.17, 2.6, 0.7, 0.04, 0.04] for _ in range(len(TASK_IDX_TO_NAME))]

# Task-specific kwargs configuration
# Values can be:
#   - None: will remain None (for values set dynamically during compute_observations)
#   - list/tuple: will be converted to tensor with to_torch()
#   - other: will be used as-is
TASK_SPECIALIZED_KWARGS = {
    'basketball': {
        'scale': [1.0, 1.0, 1.0],
    },
    'box_close': {
        'error_scale': [1.0, 1.0, 3.0],
    },
    'coffee_pull': {
        'scale': [2.0, 2.0, 1.0],
    },
    'coffee_push': {
        'scale': [2.0, 2.0, 1.0],
    },
    'dial_turn': {
        'dial_push_position_init': None,
    },
    'door_lock': {
        'scale': [1.0, 0.25, 0.5],
    },
    'door_unlock': {
        'scale': [1.0, 1.0, 1.0],
    },
    'drawer_open': {
        'scale': [1.0, 1.0, 1.0],
    },
    'handle_pull_side': {
        'scale': [1.0, 1.0, 1.0],
    },
    'lever_pull': {
        'scale': [1.0, 4.0, 4.0],
        'offset': [0, 0, 0.07],
        'lever_pos_init': None,
    },
    'peg_insert_side': {
        'peg_head_pos_init': None,
        'scale': [2.0, 1.0, 2.0],
    },
    'pick_place_wall': {
        'in_place_scaling': [1.0, 1.0, 3.0],
    },
    'push_wall': {
        'in_place_scaling': [3.0, 1.0, 1.0],
        'midpoint': [0.17, 0.1, 1.0470],
    },
    'soccer': {
        'scale': [1.0, 3.0, 1.0],
    },
    'stick_push': {
        'stick_init_pos': None,
        'thermos_dof_pos': None,
    },
    'stick_pull': {
        'stick_init_pos': None,
        'yz_scaling': [1.0, 1.0, 2.0],
        'thermos_dof_pos': None,
        'thermos_insertion_pos': None,
        'thermos_insertion_pos_init': None,
    },
    'sweep': {
        'init_left_pad': None,
        'init_right_pad': None,
        'scale': [1.0, 1.0, 1.0],
    },
}

class FrankaBaseEnvV2(TimeVecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        if self.cfg["env"]["seed"] > 0:
            np.random.seed(self.cfg["env"]["seed"])
            torch.manual_seed(self.cfg["env"]["seed"])

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.action_scale = self.cfg["env"]["actionScale"]
        self.reset_noise = self.cfg["env"].get("resetNoise", .1)

        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.camera_rendering_interval = self.cfg["env"]["cameraRenderingInterval"]
        self.camera_capture_step_interval = 1
        force_cpu_capture = self.cfg.get("trajectory_force_cpu_capture", None)
        self.camera_force_cpu_capture = bool(self.debug_viz) if force_cpu_capture is None else bool(force_cpu_capture)
        self._replay_reset_snapshot_enabled = False
        self._replay_reset_snapshot_env_ids = set()
        self._replay_reset_snapshots = {}

        self.ml_one_enabled = self.cfg["env"].get("metaLearningEnabled", None)
        self.meta_batch_size = self.cfg["env"].get("metaBatchSize", None)

        # is the object position and target fixed per reset?
        self.fixed = self.cfg["env"]["fixed"]
        self.same_init_config_per_task = self.cfg["env"].get("same_init_config_per_task", False)

        num_robot_obs = 4 * 2 # eef pos(3) + gripper mode(1); and previous
        num_task_obs = (7 + 7) * 2 # obj1 pos(3) + obj1 quat(4) + obj2 pos(3) + obj2 quat(4); and previous
        if not self.ml_one_enabled:
            num_task_obs += 3 # target pos(3)
        self.task_embedding_dim = int(self.cfg["env"].get("taskEmbeddingDim", 50))
        if self.cfg["env"]["taskEmbedding"]: # One-hot task embedding
            max_task_id = max(self.cfg["env"]["tasks"]) if self.cfg["env"]["tasks"] else -1
            if max_task_id >= self.task_embedding_dim:
                raise ValueError(
                    f"taskEmbeddingDim={self.task_embedding_dim} is too small for task id {max_task_id}. "
                    "Increase taskEmbeddingDim or reduce the task ids."
                )
            num_task_obs += self.task_embedding_dim
        num_obs = num_robot_obs + num_task_obs
        
        if self.ml_one_enabled:
            self.num_envs_per_task = self.cfg["env"]["numEnvs"] // self.meta_batch_size
            assert self.cfg["env"]["numEnvs"] % self.meta_batch_size == 0, "numEnvs should be divisible by metaBatchSize"
        
        # change in 3D space of the end-effector followed by a normalized torque that the gripper fingers should appl
        num_actions = 3 + 1 # 6 for delta pose, 1 for gripper open/close (meta world uses 3 + 1). Lets use 3 + 1 for now

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numActions"] = num_actions

        if "actionSpace" in self.cfg["env"]:
            action_space_type = self.cfg["env"]["actionSpace"]["type"]
            if action_space_type == "multi_discrete":
                self.actions_num = self.cfg["env"]["actionSpace"]["binsPerDim"] # number of bins per dimension
                self.action_dim = num_actions # number of actions
            elif action_space_type == "discrete":
                self.actions_num = self.cfg["env"]["actionSpace"]["numActions"] # number of choices per action dim (which is 1)
                self.action_dim = 1 # number of actions
            else: # continuous
                self.action_dim = num_actions
            
        # multi-task related
        self.task_idx = self.cfg["env"]["tasks"]  # list of tasks indexes from 0-49 (50 different possible tasks)
        self.num_tasks = len(self.task_idx)
        self.task_env_count = self.cfg["env"]["taskEnvCount"]  # list of number of envs per task
        self.task_indices = []
        for i, count in zip(self.task_idx, self.task_env_count):
            self.task_indices.extend([i]*count)
        self.task_indices = torch.tensor(self.task_indices, device=sim_device)

        self.init_at_random_progress = self.cfg['env']['init_at_random_progress']
        self.termination_on_success = self.cfg["env"]["termination_on_success"]
        self.reward_scale = self.cfg["env"]["reward_scale"]
        self.exempted_init_at_random_progress_tasks = self.cfg["env"]["exemptedInitAtRandomProgressTasks"]
        assert sum(self.task_env_count) == self.cfg["env"]["numEnvs"], \
            f"Sum of taskEnvCount {self.task_env_count} should be equal to num_envs {self.cfg['env']['numEnvs']}"
        assert len(self.task_idx) == len(self.task_env_count), \
            f"Length of task_idx {len(self.task_idx)} should be equal to length of task_env_count {len(self.task_env_count)}"

        # Dense reward scaling should be included here
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        """
        Seven revolute joints for the arm:
        panda_joint1
        panda_joint2
        panda_joint3
        panda_joint4
        panda_joint5
        panda_joint6
        panda_joint7

        Fixed joints:
        8. panda_hand_joint
        9. panda_hand_y_axis_joint
        10. panda_hand_z_axis_joint
        11. panda_grip_vis_joint
        12. panda_leftfinger_tip_joint
        13. panda_rightfinger_tip_joint

        Prismatic joints for the gripper:
        14. panda_finger_joint1
        15. panda_finger_joint2
        """

        self.control_type = self.cfg.get("control_type", "ik")
        assert self.control_type in {"osc", "ik", "jp"},\
            "Invalid control type specified. Must be one of: {osc, ik, jp}"
        self.franka_ctrller = FrankaController(
            self.num_envs, 
            self.device, 
            franka_dof_lower_limits=self.franka_dof_lower_limits,
            franka_dof_upper_limits=self.franka_dof_upper_limits,
            ctrl_dt=self.ctrl_dt
        )
        self._init_robot_tensor()
        self._init_data()
        self._warmup_env()


    def pre_physics_step(self, actions):
        # Assuming actions are [dx, dy, dz, normalized_torque]
        self.convert_actions(actions)
        self.command_arm()
        self.command_gripper()


    def deploy_joint_command(self, index=-1):
        if self.control_type == "ik":
            self._arm_pos_control[:] = self.tgt_q_seq[index]
            self._gripper_control[:] = self.tgt_gripper_seq[index]
        
        self.dof_targets_all.flatten().index_copy_(0, self.franka_dof_idx, self._pos_control.flatten())
        self.effort_control_all.flatten().index_copy_(0, self.franka_dof_idx, self._effort_control.flatten())
        
        # Deploy actions. Some objects joints seems remain controllable in the meta world envs.
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_targets_all),
            gymtorch.unwrap_tensor(self.franka_actor_idx),
            len(self.franka_actor_idx)
        )
        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.effort_control_all),
            gymtorch.unwrap_tensor(self.franka_actor_idx),
            len(self.franka_actor_idx)
        )

        # if self.cfg.get("apply_disturbances", False):
        #     self.apply_disturbances()


    def post_physics_step(self):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        success_env_ids = self.success_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            if self.is_ready_to_record(success_env_ids):
                self.record_post_config(success_env_ids)
            self.reset_idx(env_ids)
            self._capture_replay_reset_snapshots(env_ids)

        self.compute_observations(env_ids)
        self.compute_reward(self.actions)

        if self.camera_rendering:
            self.render_headless()

        if self.control_steps % self.camera_rendering_interval == 0:
            self.camera_request_visual = True

        if self.debug_viz and self.camera_request_visual:
            self.extras["debug_visual"] = self.get_debug_viz()
        else:
            self.extras["debug_visual"] = None

        # Restore episode metrics for training observers so success-rate
        # statistics can be logged alongside rewards.
        # Note: these averages are computed from tasks with episodes finishing on the current step, so they can very differ from tw_training.py's running per-task metrics.
        self.cumulatives["reward"] += self.rew_buf
        self.cumulatives["success"] += self.success_buf.float()
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            self.extras['episode'] = {}
            task_success_rates = []
            task_reward_means = []

            success_count = self.cumulatives["success"][env_ids].clone()
            finished_success = (success_count > 0).float()
            self.extras['episode']["average_environment_success_rate"] = finished_success.mean()
            self.extras['episode']["success"] = finished_success
            self.extras['episode']["success_count_per_episode"] = success_count

            for tid in self.task_idx:
                env_ids_counted = torch.logical_and(self.task_indices == tid, self.reset_buf).nonzero(as_tuple=False).squeeze(-1)

                if len(env_ids_counted) > 0:
                    task_success_count = self.cumulatives["success"][env_ids_counted].clone()
                    task_success = (task_success_count > 0).float()
                    task_reward = self.cumulatives["reward"][env_ids_counted].clone()
                    task_success_rates.append(task_success.mean())
                    task_reward_means.append(task_reward.mean())
                    self.extras['episode'][f"task_{tid}_reward"] = task_reward
                    self.extras['episode'][f"task_{tid}_success"] = task_success
                    self.extras['episode'][f"task_{tid}_eplength"] = self.progress_buf[env_ids_counted].float().clone()
                    self.extras['episode'][f"task_{tid}_success_count_per_episode"] = task_success_count

            # Calculate macro average metrics across tasks
            if task_reward_means:
                self.extras['episode']["average_task_reward"] = torch.stack(task_reward_means).mean()
            if task_success_rates:
                self.extras['episode']["average_task_success_rate"] = torch.stack(task_success_rates).mean()

            self.cumulatives["reward"][env_ids] = 0
            self.cumulatives["success"][env_ids] = 0


    def compute_task_reward(self, actions):
        total_count = 0
        env_ids = torch.arange(self.num_envs, device=self.device)
        franka_dof_pos = self._dof_state[self.franka_dof_idx, :].view(self.num_envs, -1, 2)[...,0] # get franka dof pos
        for tid, env_count in zip(self.task_idx, self.task_env_count):
            task_name = self.task_idx2name[tid]
            reward_fn = getattr(task_fns, task_name).compute_reward
            ids = env_ids[total_count:total_count+env_count]
            self.rew_buf[ids], self.reset_buf[ids], self.success_buf[ids] = reward_fn(
                self.reset_buf[ids],self.progress_buf[ids], self.actions[ids], franka_dof_pos[ids],
                self.world_states["eef_lf_pos"][ids], self.world_states["eef_rf_pos"][ids], self.max_episode_length,
                self.tcp_init[ids], self.target_pos[ids], self.obj_pos[ids], self.obj_init_pos[ids], self.specialized_kwargs[task_name])
            total_count += env_count

        dense_reward = torch.where(self.success_buf, torch.zeros_like(self.rew_buf), self.rew_buf)
        if self.cfg["env"]["sparse_reward"]:
            dense_reward.zero_()

        if not self.termination_on_success:
            self.reset_buf[:] = (self.progress_buf >= self.max_episode_length - 1)
 
        if self.termination_on_success:
            self.rew_buf[:] += self.reward_settings["r_success"] * self.success_buf[:].float()
        if self.cfg["env"]["sparse_reward"]:
            self.rew_buf[:] = self.reward_settings["r_success"] * self.success_buf[:].float()

        # Dense reward scaling, but not for final success reward
        scaling_factor = torch.ones_like(self.rew_buf) * self.reward_scale
        scaling_factor[self.success_buf] = 1.0
        self.rew_buf *= scaling_factor
        self.extras["dense_reward"] = dense_reward * self.reward_scale

        # TODO Remove this part after debugging
        success_and_short = self.success_buf & (self.progress_buf <= 1)
        if success_and_short.any():
            ill_env_ids = success_and_short.nonzero(as_tuple=False).squeeze(-1)
            ill_task_ids = self.task_indices[ill_env_ids].unique()
            print(f"Task {ill_task_ids} has some problems; Envs {ill_env_ids} got success immediately when reset.")

        return self.rew_buf, self.reset_buf, self.success_buf


    def _custom_refresh(self):
        # Pre-slice Franka specific tensors to avoid complex indexing
        # Franka Root
        self._franka_root_state = self._root_state[self.franka_actor_idx, :]

        # Franka DOF
        self.franka_dof_states = self._dof_state[self.franka_dof_idx, :].view(self.num_envs, -1, 2) # creates a copy!
        self.franka_dof_pos = self._q = self.franka_dof_states[..., 0]
        self.franka_dof_vel = self._qd = self.franka_dof_states[..., 1]

        # Franka Rigid Body
        self.franka_rigid_body_states = self._rigid_body_state[self.franka_rigid_body_idx, :].view(self.num_envs, -1, 13) # creates a copy!
        self._eef_state = self.franka_rigid_body_states[:, self.fr_linkname2id["panda_grip_site"], :]
        self._eef_lf_state = self.franka_rigid_body_states[:, self.fr_linkname2id["panda_leftfinger_tip"], :]
        self._eef_rf_state = self.franka_rigid_body_states[:, self.fr_linkname2id["panda_rightfinger_tip"], :]

        # Franka Rigid Body
        self._pos_control = self.dof_targets_all[self.franka_dof_idx].view(self.num_envs, -1) # creates a copy!
        self._effort_control = self.effort_control_all[self.franka_dof_idx].view(self.num_envs, -1) # creates a copy!
        self._arm_pos_control = self._pos_control[:, :7]
        self._gripper_control = self._pos_control[:, 7:9]
        self._arm_control = self._effort_control[:, :7]

        # self._table_contact_forces = self._contact_forces[:, self.link_handles["table"], :]
        # self._arm_contact_forces = self._contact_forces[:, :self.gym.get_actor_rigid_body_count(env_ptr, franka_handle), :]
    
    
    ################## Observation and State Names ###################
    def get_taskobs_names(self):
        task_obs_names = ["eef_pos", "gripper_openess", "obj_states"]
        task_obs_names += ["prev_memory"] # Must be the end
        
        if not self.ml_one_enabled:
            task_obs_names += ["target_pos"]
        if self.cfg["env"]["taskEmbedding"]:
            task_obs_names += ["taskEmbedding"]

        return task_obs_names
    

    def add_priv_taskobs(self, task_obs_names):
        # task_obs_names += ["priv_obs"]
        return task_obs_names


    def get_robot_prev_state_names(self):
        return ["q", "q_vel", "q_gripper", "q_gripper_vel"]
    

    def get_task_prev_state_names(self):
        return ["obj_states", "target_pos"]

    
    def _update_task_states(self):
        # task specific observation: object1 pos rot and object2 pos rot (3 + 4 + 3 + 4)
        # The problem: object states are in the world frame, we need to convert them to the franka base frame
        total_count = 0
        env_ids = torch.arange(self.num_envs, device=self.device)
        object_states = []
        for tid, env_count in zip(self.task_idx, self.task_env_count):
            task_name = self.task_idx2name[tid]
            obs_fn = getattr(task_fns, task_name).compute_observations
            object_state = obs_fn(self, env_ids[total_count:total_count+env_count])
            object_states.append(object_state)
            total_count += env_count
            
        object_states = torch.cat(object_states, dim=0)
        self.obj_pos = object_states[:, 0:3]

        # Convert object states from world frame to franka base frame
        pan2obj1_quat, pan2obj1_pos = self.point2frankaBase(object_states[:, 3:7], object_states[:, 0:3])
        pan2obj2_quat, pan2obj2_pos = self.point2frankaBase(object_states[:, 10:14], object_states[:, 7:10])
        object_states = torch.cat([pan2obj1_pos, pan2obj1_quat, pan2obj2_pos, pan2obj2_quat], dim=-1)
        _, pan2tgt_pos = self.point2frankaBase(self.unit_quat, self.target_pos)

        # Prev object states
        # Offset by 3 because [time_ratio, time_cur, ddl] are prepended.
        prev_memory = self.obs_buf[:, 3:21].clone()

        self.states.update({
            # Object
            "prev_memory": prev_memory,
            "obj_states": object_states,
            "target_pos": pan2tgt_pos,
            "taskEmbedding": self.task_embedding
        })

    
    def _update_robot_states(self):
        # convert transformations from the world frame to the robot base frame
        eef_quat, eef_pos = self.point2frankaBase(self._eef_state[:, 3:7], self._eef_state[:, :3])
        _, eef_lf_pos = self.point2frankaBase(self._eef_lf_state[:, 3:7], self._eef_lf_state[:, :3])
        _, eef_rf_pos = self.point2frankaBase(self._eef_rf_state[:, 3:7], self._eef_rf_state[:, :3])
        _, eef_unitz_pos = tf_combine(eef_quat, eef_pos, self.unit_quat, self.unit_z)
        
        # We assume the franka base orientation is the same as the world frame; so we do not convert the velocity
        # eef_linvel = self.vec2frankaBase(self._eef_state[:, 7:10])
        # eef_angvel = self.vec2frankaBase(self._eef_state[:, 10:])
        # eef_vel = torch.cat([eef_linvel, eef_angvel], dim=-1)
        eef_vel = self._eef_state[:, 7:]

        self.states.update({
            # Franka
            "q": self._q[:, :self.num_franka_dofs-2],
            "q_gripper": self._q[:, self.num_franka_dofs-2:self.num_franka_dofs],
            "q_vel": self._qd[:, :self.num_franka_dofs-2],
            "q_gripper_vel": self._qd[:, self.num_franka_dofs-2:self.num_franka_dofs],
            "prev_tgtq": self.prev_tgtq,
            "prev_dq": self.prev_dq,
            "eef_pos": eef_pos,
            "eef_quat": eef_quat,
            "eef_vel": eef_vel,
            "eef_lf_pos": eef_lf_pos,
            "eef_rf_pos": eef_rf_pos,
            "eef_unitz_pos": eef_unitz_pos,
            "j_eef": self._j_eef,
            # Force
            # "table_contact_forces": self._table_contact_forces,
            # "arm_contact_forces": self._arm_contact_forces,
            # Controller
            "gripper_mode": self._gripper_mode_temp.view(-1, 1),
            "gripper_waiting": self.gripper_ctrl_counts.view(-1, 1),
        })

        gripper_openess = torch.clip(self.states["q_gripper"].sum(dim=-1) / sum(self.franka_dof_upper_limits[7:]), 0.0, 1.0)
        self.states.update({
            "gripper_openess": gripper_openess.view(-1, 1)
        })

        self.world_states.update({
            # Franka World Frame
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "eef_lf_pos": self._eef_lf_state[:, :3],
            "eef_rf_pos": self._eef_rf_state[:, :3],
        })

        if self.tcp_init is None:     
            self.tcp_init = (self._eef_lf_state[:, :3] + self._eef_rf_state[:, :3]) / 2


    def _update_diff_states(self, real=False):
        if "obj_states" in self.prev_states:
            prev_obj_states = self.prev_states["obj_states"]
            cur_obj_states = self.states["obj_states"]
            prev_obj_pos1, prev_obj_pos2 = prev_obj_states[:, 0:3], prev_obj_states[:, 7:10]
            cur_obj_pos1, cur_obj_pos2 = cur_obj_states[:, 0:3], cur_obj_states[:, 7:10]
            obj1_scevel = torch.norm(cur_obj_pos1 - prev_obj_pos1, dim=-1) / self.ctrl_dt
            obj2_scevel = torch.norm(cur_obj_pos2 - prev_obj_pos2, dim=-1) / self.ctrl_dt
            scevel = obj1_scevel + obj2_scevel
            self.states["scevel"] = scevel

        if "q_vel" in self.prev_states:
            arm_qvel = self.states["q_vel"]
            arm_qacc = (arm_qvel - self.prev_states["q_vel"]) / self.ctrl_dt
            arm_qvel_p, arm_qacc_p = torch.norm(arm_qvel, dim=-1), torch.norm(arm_qacc, dim=-1)
            self.states["rob_qvel_norm"] = arm_qvel_p
            self.states["rob_qacc_norm"] = arm_qacc_p


    def update_max_joint_velocity(self, task_id):
        """Update max joint velocity limits. It is better not to change the joint velocity limits for various tasks"""
        task_env_ids = (self.task_indices == task_id).nonzero(as_tuple=False).squeeze(-1)
        max_vel_subtract = self.cur_dr_params[task_id]["controller"]["max_vel_subtract"]
        self.cur_franka_velocity_limits[task_env_ids, :7] = self.franka_velocity_limits[:7].repeat(len(task_env_ids), 1) * (1 - max_vel_subtract)
        if self.cfg.get("limit_gripper_vel", True):
            self.cur_franka_velocity_limits[task_env_ids, 7:] = self.franka_velocity_limits[7:].repeat(len(task_env_ids), 1) * (1 - max_vel_subtract)
            self.cur_franka_velocity_limits[task_env_ids, 7:] = torch.clamp(self.cur_franka_velocity_limits[task_env_ids, 7:], min=0.05) # minimum 0.05 m/s gripper


    def _reset_task_bufs(self, env_ids):
        self._gripper_mode[env_ids] = -1
        self._gripper_mode_temp[env_ids] = -1
        self.prev_tgtq[env_ids] = self._arm_pos_control[env_ids]
        self.prev_tgtq_gripper[env_ids] = self._gripper_control[env_ids]
        self.prev_dq[env_ids] = 0.
        self.prev_u_gripper_real = -1
    

    def reset_idx(self, env_ids):
        # Vectorized: map env_ids to task_ids using pre-built tensor mapping
        task_ids_for_envs = self.task_indices[env_ids]
        
        # Group env_ids by task_id using vectorized operations
        tids = {}
        for tid in torch.unique(task_ids_for_envs):
            tid_item = tid.item()
            mask = (task_ids_for_envs == tid)
            tids[tid_item] = env_ids[mask]

        dof_multi_env_ids_int32 = []
        actor_multi_env_ids_int32 = []
        # reset each task's envs
        for tid in tids:
            task_name = self.task_idx2name[tid]
            ids = tids[tid]
            reset_fn = getattr(task_fns, task_name).reset_env
            # return the actor indices whose dof needs to be reset and actor indices whose pose needs to be reset
            reset_dof_actor_indices, reset_root_actor_indices = reset_fn(self, tid, ids, self.random_reset_space_task[tid])

            dof_multi_env_ids_int32.append(reset_dof_actor_indices)
            actor_multi_env_ids_int32.append(reset_root_actor_indices)

        dof_multi_env_ids_int32 = torch.hstack(dof_multi_env_ids_int32).flatten().to(dtype=torch.int32)
        actor_multi_env_ids_int32 = torch.hstack(actor_multi_env_ids_int32).flatten().to(dtype=torch.int32)
        franka_env_ids_int32 = self.franka_actor_idx[env_ids].to(dtype=torch.int32)
                
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.dof_targets_all),
                                                        gymtorch.unwrap_tensor(franka_env_ids_int32), 
                                                        len(franka_env_ids_int32))
        
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.effort_control_all),
                                                        gymtorch.unwrap_tensor(franka_env_ids_int32),
                                                        len(franka_env_ids_int32))
        
        self.gym.set_actor_root_state_tensor_indexed(self.sim, 
                                                        gymtorch.unwrap_tensor(self._root_state),
                                                        gymtorch.unwrap_tensor(actor_multi_env_ids_int32), 
                                                        len(actor_multi_env_ids_int32))
        
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(dof_multi_env_ids_int32), 
                                              len(dof_multi_env_ids_int32))
        
        self._reset_bufs(env_ids)
        self.extras.pop('episode', None)

        # Need one step to refresh the dof states; Otherwise, the states (such as eef pose) will be outdated
        self.gym.simulate(self.sim)
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
        if self.force_render:
            self.render()


    def _reset_specialized_kwargs_for_task(self, task_name: str):
        self.specialized_kwargs[task_name] = {}
        if task_name in TASK_SPECIALIZED_KWARGS:
            for key, value in TASK_SPECIALIZED_KWARGS[task_name].items():
                if isinstance(value, (list, tuple)):
                    self.specialized_kwargs[task_name][key] = to_torch(value, device=self.device)
                else:
                    self.specialized_kwargs[task_name][key] = value


    def enable_replay_reset_snapshot_capture(self, env_ids=None):
        self._replay_reset_snapshot_enabled = True
        self._replay_reset_snapshot_env_ids = (
            None if env_ids is None else {int(env_id) for env_id in env_ids}
        )
        self._replay_reset_snapshots = {}

    def pop_replay_reset_snapshots(self):
        snapshots = self._replay_reset_snapshots
        self._replay_reset_snapshots = {}
        return snapshots

    def _capture_replay_reset_snapshots(self, env_ids):
        if not getattr(self, "_replay_reset_snapshot_enabled", False):
            return
        target_env_ids = getattr(self, "_replay_reset_snapshot_env_ids", None)
        for env_id in env_ids.detach().cpu().tolist():
            env_id = int(env_id)
            if target_env_ids is not None and env_id not in target_env_ids:
                continue
            self._replay_reset_snapshots[env_id] = self.export_replay_snapshot(env_id)

    def export_replay_snapshot(self, env_id: int) -> Dict[str, Any]:
        env_id = int(env_id)
        task_id = int(self.task_indices[env_id].item())
        actor_start = int(self.franka_actor_idx[env_id].item())
        actor_count = int(self.env_actor_count[env_id].item())
        dof_start = int(self.franka_dof_start_idx[env_id].item())
        dof_count = int(self.env_dof_count[env_id].item())

        return {
            "env_id": env_id,
            "task_id": task_id,
            "last_rand_vec": self.last_rand_vecs[env_id].detach().cpu().numpy().astype(np.float32),
            "target_pos": self.target_pos[env_id].detach().cpu().numpy().astype(np.float32),
            "obj_init_pos": self.obj_init_pos[env_id].detach().cpu().numpy().astype(np.float32),
            "franka_init_dof_pos": self.franka_dof_pos[env_id].detach().cpu().numpy().astype(np.float32),
            "root_state": self._root_state[actor_start:actor_start + actor_count].detach().cpu().numpy().astype(np.float32),
            "dof_state": self._dof_state[dof_start:dof_start + dof_count].detach().cpu().numpy().astype(np.float32),
            "dof_targets": self.dof_targets_all[dof_start:dof_start + dof_count].detach().cpu().numpy().astype(np.float32),
            "effort_control": self.effort_control_all[dof_start:dof_start + dof_count].detach().cpu().numpy().astype(np.float32),
            "prev_tgtq": self.prev_tgtq[env_id].detach().cpu().numpy().astype(np.float32),
            "prev_tgtq_gripper": self.prev_tgtq_gripper[env_id].detach().cpu().numpy().astype(np.float32),
            "prev_dq": self.prev_dq[env_id].detach().cpu().numpy().astype(np.float32),
            "gripper_mode": float(self._gripper_mode[env_id].item()),
            "gripper_mode_temp": float(self._gripper_mode_temp[env_id].item()),
            "gripper_ctrl_counts": float(self.gripper_ctrl_counts[env_id].item()),
            "gripper_delay_timer": float(self.gripper_delay_timer[env_id].item()),
            "progress_buf": int(self.progress_buf[env_id].item()),
        }

    def _restore_task_replay_buffers(self, env_id: int, task_id: int, task_name: str, snapshot: Dict[str, Any]):
        self.last_rand_vecs[env_id] = torch.as_tensor(
            snapshot["last_rand_vec"],
            device=self.device,
            dtype=self.last_rand_vecs.dtype,
        )

        if "target_pos" in snapshot and "obj_init_pos" in snapshot:
            self.target_pos[env_id] = torch.as_tensor(
                snapshot["target_pos"],
                device=self.device,
                dtype=self.target_pos.dtype,
            )
            self.obj_init_pos[env_id] = torch.as_tensor(
                snapshot["obj_init_pos"],
                device=self.device,
                dtype=self.obj_init_pos.dtype,
            )
            return

        # Legacy trajectory archives only stored `last_rand_vec`, while task
        # rewards read derived buffers such as `target_pos` and `obj_init_pos`.
        # Re-run the task reset helper in a fixed configuration so those
        # buffers are reconstructed from the archived task sample, then
        # overwrite the actual simulator tensors with the archived snapshot.
        original_fixed = self.fixed
        try:
            self.fixed = True
            reset_fn = getattr(task_fns, task_name).reset_env
            env_ids = torch.tensor([env_id], device=self.device, dtype=torch.long)
            reset_fn(self, task_id, env_ids, self.random_reset_space_task[task_id])
            self.last_rand_vecs[env_id] = torch.as_tensor(
                snapshot["last_rand_vec"],
                device=self.device,
                dtype=self.last_rand_vecs.dtype,
            )
        finally:
            self.fixed = original_fixed

    def load_replay_snapshot(self, env_id: int, snapshot: Dict[str, Any]):
        env_id = int(env_id)
        task_id = int(snapshot["task_id"])
        if int(self.task_indices[env_id].item()) != task_id:
            raise ValueError(
                f"Snapshot task_id {task_id} does not match env task_id {int(self.task_indices[env_id].item())}"
            )

        task_name = self.task_idx2name[task_id]
        self._reset_specialized_kwargs_for_task(task_name)
        self._restore_task_replay_buffers(env_id, task_id, task_name, snapshot)

        actor_start = int(self.franka_actor_idx[env_id].item())
        actor_count = int(self.env_actor_count[env_id].item())
        dof_start = int(self.franka_dof_start_idx[env_id].item())
        dof_count = int(self.env_dof_count[env_id].item())

        root_state = torch.as_tensor(snapshot["root_state"], device=self.device, dtype=self._root_state.dtype)
        dof_state = torch.as_tensor(snapshot["dof_state"], device=self.device, dtype=self._dof_state.dtype)
        dof_targets = torch.as_tensor(snapshot["dof_targets"], device=self.device, dtype=self.dof_targets_all.dtype)
        effort_control = torch.as_tensor(snapshot["effort_control"], device=self.device, dtype=self.effort_control_all.dtype)

        self._root_state[actor_start:actor_start + actor_count] = root_state
        self._dof_state[dof_start:dof_start + dof_count] = dof_state
        self.dof_targets_all[dof_start:dof_start + dof_count] = dof_targets
        self.effort_control_all[dof_start:dof_start + dof_count] = effort_control

        self.prev_tgtq[env_id] = torch.as_tensor(snapshot["prev_tgtq"], device=self.device, dtype=self.prev_tgtq.dtype)
        self.prev_tgtq_gripper[env_id] = torch.as_tensor(snapshot["prev_tgtq_gripper"], device=self.device, dtype=self.prev_tgtq_gripper.dtype)
        self.prev_dq[env_id] = torch.as_tensor(snapshot["prev_dq"], device=self.device, dtype=self.prev_dq.dtype)
        self._gripper_mode[env_id] = float(snapshot["gripper_mode"])
        self._gripper_mode_temp[env_id] = float(snapshot["gripper_mode_temp"])
        self.gripper_ctrl_counts[env_id] = float(snapshot["gripper_ctrl_counts"])
        self.gripper_delay_timer[env_id] = float(snapshot["gripper_delay_timer"])

        self.progress_buf[env_id] = int(snapshot.get("progress_buf", 0))
        self.reset_buf[env_id] = 0
        self.success_buf[env_id] = 0
        self.timeout_buf[env_id] = 0
        self.continuous_check_buf[env_id] = 0
        self.force_has_applied[env_id] = 0

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_state))
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self._dof_state))
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_targets_all))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.effort_control_all))

        self.gym.simulate(self.sim)
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
        if self.force_render:
            self.render()

        reset_ids = torch.tensor([env_id], device=self.device, dtype=torch.long)
        self.compute_observations(reset_ids=reset_ids)
        self.update_observations_dict()
        self.extras.pop('episode', None)

    
    def command_arm(self):
        # Split arm and gripper command
        dpos = self.actions[:, :-1]
        dori = torch.zeros_like(dpos[:, :3])  # only use position control for now
        dpose = torch.cat([dpos, dori], dim=-1)
        dpose = dpose * self.action_scale * self.act_scale

        # Control arm
        if self.control_type == "ik":
            p_arm, self.tgt_q_seq, self.prev_dq[:] = self.franka_ctrller.differential_ik(
                dpose=dpose,
                q=self._arm_pos_control,
                j_eef=self._j_eef,
                velocity_limits=self.cur_franka_velocity_limits[:, :7],
                prev_dq=self.prev_dq,
                alpha=0.
            )
            self._arm_pos_control[:] = self.prev_tgtq[:] =  p_arm
        elif self.control_type == "osc":
            u_arm = self.franka_ctrller.compute_osc_torques(dpose=dpose)
            self._arm_control[:] = u_arm
        elif self.control_type == "jp":
            u_arm = self.franka_ctrller.compute_joint_pos(dpose=dpose)
            self._arm_pos_control[:] = u_arm
        else:
            raise ValueError(f"Unknown control type: {self.control_type}")
        
    
    def command_gripper(self):
        u_gripper = self.actions[:, -1]
        # Control gripper
        self.gripper_ctrl_counts += 1
        gripper_ctrl_idx = self.gripper_ctrl_counts >= self.gripper_freq_inv

        # Control Delay Logic
        if gripper_ctrl_idx.any():
            gripper_ctrl_ids = gripper_ctrl_idx.nonzero().flatten()
            self.gripper_ctrl_counts[gripper_ctrl_idx] = 0
            
            # u_gripper > 0 is close (gripper_mode = 1), u_gripper <= 0 is open (gripper_mode = 0)
            cur_gripper_mode_temp = torch.where(u_gripper[gripper_ctrl_idx]>0., 1., -1.)
            prev_gripper_mode_temp = self._gripper_mode_temp[gripper_ctrl_idx]
            # find the different idx with the current gripper mode
            diff_idx = (cur_gripper_mode_temp != prev_gripper_mode_temp)
            diff_ids = gripper_ctrl_ids[diff_idx]
            # If the gripper mode is changed, we need to reset the delay timer. For ids we can not use .any()!!
            if len(diff_ids)>0:
                # Get DR params for each environment
                for tid in self.cur_dr_params.keys():
                    task_env_ids = (self.task_indices[diff_ids] == tid).nonzero(as_tuple=True)[0]
                    if len(task_env_ids) == 0: continue
                    actual_ids = diff_ids[task_env_ids]
                    delay_time = self.cur_dr_params[tid]["controller"]["gripper_delay"] + \
                                 torch_rand_float(-1, 1, (len(actual_ids), 1), device=self.device) * self.cur_dr_params[tid]["noise"]["gripper_delay"]
                    delay_steps = (torch.clamp(delay_time, min=0.) / self.ctrl_dt).ceil().flatten()
                    self.gripper_delay_timer[actual_ids] = delay_steps
            # Update the gripper mode temp
            self._gripper_mode_temp[gripper_ctrl_idx] = cur_gripper_mode_temp
        
        self.gripper_delay_timer = torch.clamp(self.gripper_delay_timer-1, min=0)
        delay_done_idx = self.gripper_delay_timer == 0
        if delay_done_idx.any():
            # Delay has been done, we can update the gripper mode
            self._gripper_mode[delay_done_idx] = self._gripper_mode_temp[delay_done_idx]
        
        # Gripper Velocity Random Logic
        v_gripper = self.cur_franka_velocity_limits[:, -2:] * 20 # TODO: Remove this cheating that 1 sim step to open or close the gripper.
        if self.add_act_noise:
            # Add gripper velocity noise per task
            for tid in self.cur_dr_params.keys():
                task_env_ids = (self.task_indices == tid).nonzero(as_tuple=True)[0]
                v_gripper[task_env_ids] = v_gripper[task_env_ids] + \
                                       torch_rand_float(-1, 1., (len(task_env_ids), 1), device=self.device) * self.cur_dr_params[tid]["noise"]["v_gripper"]
        
        u_fingers = torch.where(self._gripper_mode==1.,
                                -v_gripper[:, 0] * self.ctrl_dt,
                                v_gripper[:, 1] * self.ctrl_dt)
        
        p_fingers = self.prev_tgtq_gripper + u_fingers.unsqueeze(1)
        p_fingers = tensor_clamp(p_fingers, 
                                 self.franka_dof_lower_limits[-2:].unsqueeze(0), 
                                 self.franka_dof_upper_limits[-2:].unsqueeze(0))
        self.tgt_gripper_seq = [self.prev_tgtq_gripper] * (self.num_inter_steps-1) + [p_fingers]  # Update the target gripper sequence with the final position
        # Write gripper command to appropriate tensor buffer
        self.prev_tgtq_gripper[:] = p_fingers
    
    
    def _update_task_common_info(self):
        gripper_vel = self.states["q_gripper_vel"][0].clone()
        gripper_v_fix = (self.states["q_gripper"][0] - self.prev_states["q_gripper"][0]).clone() / self.ctrl_dt # gripper vel does not make sense in pos control mode. We use the difference of pos to get the vel
        gripper_vel[:] = gripper_v_fix
        self.extras.update({
            # Franka Related
            "joint_tgt_q": self._arm_pos_control[0].clone(),
            "joint_torqs": self._effort_control[0].clone(),
            "joint_poss": self.states["q"][0].clone(),
            "joint_vels": self.states["q_vel"][0].clone(),
            "joint_accs": (self.states["q_vel"][0] - self.prev_states["q_vel"][0]) / self.ctrl_dt, # refer torque is better
            "joint_gripper_poss": self.states["q_gripper"][0].clone(),
            "joint_gripper_vels": gripper_vel,
            "joint_velocity_limits": self.cur_franka_velocity_limits[0].clone(),
            "rob_qvel_norm": self.extras.get("rob_qvel_norm", torch.zeros_like(self.rew_buf)) if self.reward_settings["r_arm_vel_scale"]!=0 else torch.zeros_like(self.rew_buf),
        })

        if self.cfg.get("use_fk_replay", False):
            joint_tgt_q = self.extras["joint_tgt_q"].tolist()
            gripper_mode = self._gripper_mode_temp[0].cpu().item()
            u_gripper = 0 if gripper_mode==self.prev_u_gripper_real else gripper_mode
            self.prev_u_gripper_real = gripper_mode
            ctrl_cmd = [*joint_tgt_q, u_gripper]
            self.extras["fk_replay_cmd"] = ctrl_cmd


    @property
    def task_idx2name(self):
        return TASK_IDX_TO_NAME
    
    #################### Initializations ####################
    def _trajectory_clean_background(self):
        return bool(self.cfg.get("trajectory_clean_background", False))

    def _create_ground_plane(self):
        if self._trajectory_clean_background():
            return
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)


    def _create_franka_assets(self, asset_root):
        franka_asset_file = "urdf/franka_description/robots/franka_panda_gripper.urdf"
        
        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)
        
        # load franka asset
        franka_asset_options = gymapi.AssetOptions()
        franka_asset_options.flip_visual_attachments = True
        franka_asset_options.fix_base_link = True
        franka_asset_options.collapse_fixed_joints = False
        franka_asset_options.disable_gravity = True
        franka_asset_options.thickness = 0.001
        franka_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        franka_asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, franka_asset_options)

        franka_dof_stiffness = to_torch([1600, 1600, 1600, 1600, 1600, 1600, 1600, 1.0e6, 1.0e6], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)
        self.num_franka_rigid_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self.franka_velocity_limits = []
        self.franka_effort_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS # We only use joint impedance control
            
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
            else:
                franka_dof_props['stiffness'][i] = 7000.0
                franka_dof_props['damping'][i] = 50.0

            # Constrain the gripper velocity to be 0.053 m/s (1.5s required for 0.08m open/close)
            # IsaacGym Bug: Setting the gripper velocity will affect other dofs instead of just the gripper! Also, check the gripper initial position is open?
            # franka_dof_props['velocity'][i] = 0.053 if i > 6 else franka_dof_props['velocity'][i]
            
            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
            self.franka_velocity_limits.append(franka_dof_props['velocity'][i])
            self.franka_effort_limits.append(franka_dof_props['effort'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self.franka_velocity_limits = to_torch(self.franka_velocity_limits, device=self.device)
        self.cur_franka_velocity_limits = self.franka_velocity_limits.clone().repeat(self.num_envs, 1) # Each task might have different velocity limits
        self.franka_effort_limits = to_torch(self.franka_effort_limits, device=self.device)
        self.franka_effort_limits[7:] = 80
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[7:] = 0.1
        franka_dof_props['effort'][7:] = 80.0

        return franka_asset, franka_dof_props

    def _debug_camera_pose(self):
        preset = str(self.cfg.get("trajectory_camera_preset", "default") or "default")
        camera_presets = {
            "mt15_assembly": ((0.60, -0.42, 1.43), (-0.02, 0.00, 1.13)),
            "mt15_basketball": ((-0.42, -0.82, 1.50), (0.05, -0.02, 1.08)),
            "mt15_bin_picking": ((0.58, -0.40, 1.46), (0.08, 0.00, 1.11)),
            "mt15_box_close": ((0.54, -0.36, 1.51), (0.08, 0.00, 1.11)),
            "mt15_door_lock": ((-0.42, -0.84, 1.50), (0.10, 0.03, 1.08)),
            "mt15_drawer_open": ((-0.50, -0.78, 1.50), (0.04, 0.02, 1.07)),
            "mt15_hammer": ((-0.42, -0.78, 1.42), (0.04, 0.00, 1.06)),
            "mt15_hand_insert": ((-0.38, -0.78, 1.50), (0.04, 0.00, 1.06)),
            "mt15_lever_pull": ((-0.58, -0.84, 1.55), (0.04, 0.00, 1.06)),
            "mt15_peg_unplug_side": ((-0.50, -1.00, 1.50), (0.00, 0.00, 1.00)),
            "mt15_pick_out_of_hole": ((0.56, -0.36, 1.45), (0.05, 0.00, 1.12)),
            "mt15_soccer": ((-0.42, -0.82, 1.42), (0.06, -0.02, 1.08)),
            "mt15_stick_pull": ((0.48, -0.82, 1.46), (0.07, 0.00, 1.09)),
            "mt15_sweep_into_goal": ((-0.42, -0.82, 1.42), (0.05, -0.02, 1.08)),
        }
        if preset in camera_presets:
            cam_pos, cam_target = camera_presets[preset]
            return gymapi.Vec3(*cam_pos), gymapi.Vec3(*cam_target)
        if preset in {"teaser", "teaser_peg_insert"}:
            return gymapi.Vec3(0.50, -0.24, 1.39), gymapi.Vec3(0.04, 0.01, 1.14)
        return gymapi.Vec3(-0.5, -1.0, 1.5), gymapi.Vec3(0.0, 0.0, 1.0)

    def _debug_camera_fov(self):
        fov = self.cfg.get("trajectory_camera_fov", None)
        if fov is not None:
            return float(fov)
        preset = str(self.cfg.get("trajectory_camera_preset", "default") or "default")
        preset_fovs = {
            "mt15_assembly": 42.0,
            "mt15_basketball": 44.0,
            "mt15_bin_picking": 48.0,
            "mt15_box_close": 48.0,
            "mt15_door_lock": 45.0,
            "mt15_drawer_open": 44.0,
            "mt15_hammer": 46.0,
            "mt15_hand_insert": 40.0,
            "mt15_lever_pull": 43.0,
            "mt15_peg_unplug_side": 50.0,
            "mt15_pick_out_of_hole": 50.0,
            "mt15_soccer": 44.0,
            "mt15_stick_pull": 45.0,
            "mt15_sweep_into_goal": 40.0,
        }
        if preset in preset_fovs:
            return preset_fovs[preset]
        if preset in {"teaser", "teaser_peg_insert"}:
            return 40.0
        return None

        
    def _create_envs(self, num_envs, spacing, num_per_row):
        #------------------- prepare Franka and table assets shared by all envs -------------------#
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../assets")
        franka_asset, franka_dof_props = self._create_franka_assets(asset_root=asset_root)
        # Create table asset
        table_pos = [0.0, 0.0, 1.0]
        table_thickness = 0.054
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[1.2, 1.2, table_thickness], table_opts)
        
        # Create table stand asset
        table_stand_height = 0.01
        table_stand_pos = [-0.6, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_opts)
        
        # define start pose for franka
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(-0.55, 0.0, 1.0 + table_thickness / 2 + table_stand_height)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])

        # Define start pose for table stand
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for ws stand
        ws_stand_pos = [0.035, 0.0, 1.0 + table_thickness / 2]
        ws_stand_start_pose = gymapi.Transform()
        ws_stand_start_pose.p = gymapi.Vec3(*ws_stand_pos)
        ws_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._ws_surface_pos = np.array(ws_stand_pos)
        
        self.num_table_rigid_bodies = self.gym.get_asset_rigid_body_count(table_asset) \
            + self.gym.get_asset_rigid_body_count(table_stand_asset)
        #-------------------------------------------------------------------------------------#
        
        self.frankas = []
        self.envs = []
        self.objects = []
        self.random_reset_space_task = [None]*len(FRANKA_DEFAULT_DOF_STATES)
        self.franka_actor_idx = torch.zeros(self.num_envs, device=self.device).to(dtype=torch.int32)
        self.env_actor_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        # where the first franka dof starts in each env 
        self.franka_dof_start_idx = torch.zeros(self.num_envs, device=self.device).to(dtype=torch.int32)
        self.env_dof_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        # where the first franka rigid body starts in each env 
        self.franka_rigid_body_start_idx = torch.zeros(self.num_envs, device=self.device).to(dtype=torch.int32)
        i = 0  # count the current env idx
        
        total_actor_count = 0
        total_dof_count = 0
        total_rigid_body_count = 0
        for tid, env_count in zip(self.task_idx, self.task_env_count):
            task_name = self.task_idx2name[tid]
            print("Creating %d Franka environments for task %s" % (env_count, task_name))
            # create envs
            env_fn = getattr(task_fns, task_name).create_envs
            envs, franks, objects, random_reset_space, num_task_actors, num_task_dof, num_task_rigid_bodies = env_fn(
                self, env_count, spacing, num_per_row,
                franka_asset, franka_start_pose, franka_dof_props,
                table_asset, table_start_pose, table_stand_asset, table_stand_start_pose,    
            )
            self.envs.extend(envs)
            self.frankas.extend(franks)
            self.objects.extend(objects)
            if self.same_init_config_per_task and self.fixed:
                center = (random_reset_space.low + random_reset_space.high) / 2.0
                random_reset_space.low[...] = center
                random_reset_space.high[...] = center
            self.random_reset_space_task[tid] = random_reset_space
            
            for _ in range(env_count):
                self.franka_actor_idx[i] = total_actor_count
                self.env_actor_count[i] = num_task_actors + 3
                total_actor_count += num_task_actors + 3
                
                self.franka_dof_start_idx[i] = total_dof_count
                self.env_dof_count[i] = num_task_dof + self.num_franka_dofs
                total_dof_count += num_task_dof + self.num_franka_dofs
                
                self.franka_rigid_body_start_idx[i] = total_rigid_body_count
                total_rigid_body_count += \
                    num_task_rigid_bodies + self.num_franka_rigid_bodies + self.num_table_rigid_bodies
                
                i += 1
            
        # Franka / Table handles for transformations
        env_ptr = self.envs[0]
        franka_actor = self.frankas[0]
        self.fr_linkname2id = self.gym.get_actor_rigid_body_dict(env_ptr, franka_actor)
        self.fr_linkid2name = {v: k for k, v in self.fr_linkname2id.items()}
        self.fr_jointname2id = self.gym.get_actor_joint_dict(env_ptr, franka_actor)
        self.fr_jointid2name = {v: k for k, v in self.fr_jointname2id.items()}

        # add camera
        # TODO: need to add camera to each env for visual based tasks
        if self.debug_viz:
            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = self.cfg["env"]["cameraWidth"]
            self.camera_props.height = self.cfg["env"]["cameraHeight"]
            self.camera_props.enable_tensors = not self.camera_force_cpu_capture
            camera_fov = self._debug_camera_fov()
            if camera_fov is not None:
                self.camera_props.horizontal_fov = camera_fov
            self.rendering_cameras = []
            self.torch_camera_tensors = []
            for env_ptr in self.envs:
                rendering_camera = self.gym.create_camera_sensor(env_ptr, self.camera_props)
                self.rendering_cameras.append(rendering_camera)
                if rendering_camera != -1:
                    cam_pos, cam_target = self._debug_camera_pose()
                    self.gym.set_camera_location(rendering_camera, env_ptr, cam_pos, cam_target)

                    if self.camera_force_cpu_capture:
                        torch_camera_tensor = None
                    else:
                        video_frame_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, rendering_camera, gymapi.IMAGE_COLOR)
                        torch_camera_tensor = gymtorch.wrap_tensor(video_frame_tensor).view((self.camera_props.height, self.camera_props.width, 4))
                else:
                    torch_camera_tensor = None
                self.torch_camera_tensors.append(torch_camera_tensor)
            self.rendering_camera = self.rendering_cameras[-1]
            self.torch_camera_tensor = self.torch_camera_tensors[-1]

            img_dir = "debug"
            if not os.path.exists(img_dir):
                os.mkdir(img_dir)
    

    def _init_robot_tensor(self):
        """
        Initialize values for the franka. 
        E.g., the default DoF position, velocity, and target tensors.
        """
        # Franka DOF
        self.franka_default_dof_pos = [to_torch(FRANKA_DEFAULT_DOF_STATES[i],device=self.device) for i in range(len(FRANKA_DEFAULT_DOF_STATES))]
        self.franka_dof_idx = torch.stack([self.franka_dof_start_idx + i for i in range(self.num_franka_dofs)], dim=1).flatten().to(dtype=torch.long)
        
        # Franka Rigid Body
        self.franka_rigid_body_idx = torch.stack([self.franka_rigid_body_start_idx + i for i in range(self.num_franka_rigid_bodies)], dim=1).flatten().to(dtype=torch.long)

        # Franka Kinematics & Dynamics
        jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        self._jacobian = gymtorch.wrap_tensor(jacobian_tensor)
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._j_eef = self._jacobian[:, self.fr_jointname2id['panda_hand_joint'], :, :7]
        self._mm = mm[:, :7, :7]

        # Initialize control
        self.dof_targets_all = torch.zeros((self.gym.get_sim_dof_count(self.sim), 1),dtype=torch.float, device=self.device)
        self._pos_control = self.dof_targets_all[self.franka_dof_idx].view(self.num_envs, -1) # creates a copy!
        self.effort_control_all = torch.zeros((self.gym.get_sim_dof_count(self.sim), 1), dtype=torch.float, device=self.device)
        self._effort_control = self.effort_control_all[self.franka_dof_idx].view(self.num_envs, -1) # creates a copy!
        
        # Initialize indices
        self._gripper_mode = -torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        self._gripper_mode_temp = -torch.ones(self.num_envs, dtype=torch.float, device=self.device)  # Temporary gripper mode for control
        self.gripper_ctrl_counts = torch.zeros(self.num_envs, device=self.device)
        self.gripper_delay_timer = torch.zeros(self.num_envs, device=self.device)
        self.gripper_ctrl_counts_real = 0

        # Update custom slice indices.
        self._custom_refresh()

        # Task embedding
        if self.cfg["env"]["taskEmbedding"]:
            self.task_embedding = torch.nn.functional.one_hot(
                self.task_indices.long(),
                self.task_embedding_dim,
            ).float()

        self.extras["episode_cumulative"] = {}
        self.extras["task_indices"] = self.task_indices
        # record task names
        ordered_task_names = [self.task_idx2name[tid.item()] for tid in torch.unique(self.task_indices)]
        self.extras["ordered_task_names"] = list(map(lambda s: s.replace("_", "-"), ordered_task_names))

        self.cumulatives = defaultdict(lambda: torch.zeros(self.num_envs, device=self.device))

        if self.debug_viz:
            self.camera_frames = []
            self.manual_camera_frames = []
            self.manual_camera_capture = False
            self.camera_render_env_id = self.num_envs - 1
            # GPU camera tensors are only safe when the rendering device and the
            # CUDA compute device refer to the same ordinal that PyTorch can see.
            # On some setups Vulkan graphics device IDs do not match CUDA ordinals
            # even when there is only one physical GPU, so force CPU readback.
            self.camera_force_cpu_capture = self.camera_force_cpu_capture or (
                self.graphics_device_id < 0 or self.graphics_device_id != self.device_id
            )

        self.camera_rendering = False  # by default, don't render camera except for viewer camera, which is controlled by headless flag
        self.camera_request_visual = False

        # Custom 
        self._global_indices = torch.arange(self.num_actors, dtype=torch.long, device=self.device)
        self.action_scale *= self.cfg.get("control_freq_inv", 1) # 0.01 * 60Hz = 0.6; 0.03 * 20Hz = 0.6
        self.extras["franka"] = {
            "dof_lower": self.franka_dof_lower_limits.cpu().numpy(),
            "dof_upper": self.franka_dof_upper_limits.cpu().numpy(),
            "velocity_limits": self.franka_velocity_limits.cpu().numpy(),
            "effort_limits": self.franka_effort_limits.cpu().numpy(),
        }
        # Assume franka base will not change
        self._franka2W_quat, self._franka2W_pos = tf_inverse(self._franka_root_state[:, 3:7], self._franka_root_state[:, :3])

    
    def _init_data(self):
        self.tcp_init = None
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.obj_init_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.specialized_kwargs = defaultdict(dict)  # simplified: task_name -> {key: value}
        self.env_id_to_task_id = [-1 for _ in range(self.num_envs)]

        self.last_rand_vecs = torch.zeros((self.num_envs,6),device=self.device)
        self.last_rand_vecs_test = torch.zeros((self.num_envs,6),device=self.device)

        # Initialize specialized_kwargs from configuration to torch tensors
        for tid, env_count in zip(self.task_idx, self.task_env_count):
            task_name = self.task_idx2name[tid]
            if task_name in TASK_SPECIALIZED_KWARGS:
                for key, value in TASK_SPECIALIZED_KWARGS[task_name].items():
                    if isinstance(value, (list, tuple)):
                        self.specialized_kwargs[task_name][key] = to_torch(value, device=self.device)
                    else:
                        self.specialized_kwargs[task_name][key] = value
            
        total_count = 0
        for tid, env_count in zip(self.task_idx, self.task_env_count):
            self.env_id_to_task_id[total_count:total_count+env_count] = [tid] * env_count
            # exempted_init_at_random_progress_tasks is a list of task ids that should not be initialized at random progress
            if not self.init_at_random_progress or tid in self.exempted_init_at_random_progress_tasks:
                self.progress_buf[total_count:total_count+env_count] = 0
            else:
                self.progress_buf[total_count:total_count+env_count] = torch.randint_like(self.progress_buf[total_count:total_count+env_count], high=int(self.max_episode_length))
            total_count += env_count

        # Work space limits
        self._ws_surface_pos = to_torch(self._ws_surface_pos, device=self.device)
        self._ws_upper_bounds = to_torch([0.2] * 2, device=self.device)
        
        #------------------- initialize special variables required for some reward functions -------------------#
        # self.specialized_kwargs['basketball']['scale'] = to_torch([1.0, 1.0, 1.0], device=self.device)
        
        # self.specialized_kwargs['box_close']['error_scale'] = to_torch([1.0, 1.0, 3.0], device=self.device)
        
        # self.specialized_kwargs['coffee_pull']['scale'] = to_torch([2.0, 2.0, 1.0],device=self.device)
        
        # self.specialized_kwargs['coffee_push']['scale'] = to_torch([2.0, 2.0, 1.0],device=self.device)
        
        # self.specialized_kwargs['dial_turn']['dial_push_position_init'] = None
        
        # self.specialized_kwargs['door_lock']['scale'] = to_torch([1,.25,.5], device=self.device)
        
        # self.specialized_kwargs['door_unlock']['scale'] = to_torch([1.0, 1.0, 1.0], device=self.device)

        # self.specialized_kwargs['drawer_open']['scale'] = to_torch([1.0, 1.0, 1.0],device=self.device)

        # self.specialized_kwargs['handle_pull_side']['scale'] = to_torch([1.0, 1.0, 1.0],device=self.device)

        # self.specialized_kwargs['lever_pull']['scale'] =  to_torch([1.0,4.0,4.0],device=self.device)
        # self.specialized_kwargs['lever_pull']['offset'] = to_torch([0, 0, 0.07],device=self.device)
        # self.specialized_kwargs['lever_pull']['lever_pos_init'] = None

        # self.specialized_kwargs['peg_insert_side']['peg_head_pos_init'] = None
        # self.specialized_kwargs['peg_insert_side']['scale'] = to_torch([2.0, 1.0, 2.0], device=self.device)
        
        # self.specialized_kwargs['pick_place_wall']['in_place_scaling'] = to_torch([3.0, 1.0, 1.0], device=self.device)

        # self.specialized_kwargs["push_wall"]["in_place_scaling"] = to_torch([3.0, 1.0, 1.0], device=self.device)
        # self.specialized_kwargs["push_wall"]["midpoint"] = to_torch([0.17, .1, 1.0470], device=self.device)

        # self.specialized_kwargs['soccer']['scale'] = to_torch([1.0, 3.0, 1.0],device=self.device)
        
        # self.specialized_kwargs['stick_push']['stick_init_pos'] = None
        # self.specialized_kwargs['stick_push']['thermos_dof_pos'] = None

        # self.specialized_kwargs['stick_pull']['stick_init_pos'] = None
        # self.specialized_kwargs['stick_pull']['yz_scaling'] = to_torch([1.0, 1.0, 2.0], device=self.device)
        # self.specialized_kwargs['stick_pull']['thermos_dof_pos'] = None
        # self.specialized_kwargs['stick_pull']['thermos_insertion_pos'] = None
        # self.specialized_kwargs['stick_pull']['thermos_insertion_pos_init'] = None

        # self.specialized_kwargs['sweep']['init_left_pad'] = None
        # self.specialized_kwargs['sweep']['init_right_pad'] = None
        # self.specialized_kwargs['sweep']['scale'] = to_torch([1.0, 1.0, 1.0],device=self.device)

    
    def task_priv_obs_init(self):
        for tid, env_count in zip(self.task_idx, self.task_env_count):
            task_name = self.task_idx2name[tid]
            task_class = getattr(task_fns, task_name)
            priv_obs_init_fn = getattr(task_class, "task_priv_obs_init", None)
            if priv_obs_init_fn is None:
                continue
            task_priv_obs_names = priv_obs_init_fn(self)
            self.task_priv_obs_names[task_name] = task_priv_obs_names


    def robot_variable_init(self):
        self._eef_state = None  # end effector state (at grasping point)
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)

        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._arm_pos_control = None  # Position actions
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._gripper_mode = None  # Gripper mode (open/close)
        self.franka_effort_limits = None        # Actuator effort limits for franka
        self.franka_velocity_limits = None      # Actuator velocity limits for franka
        self.cur_franka_velocity_limits = None      # Actuator velocity limits for franka in the real world (slowly curriculum)
        self.franka_dof_lower_limits = None        # Actuator lower limits for franka
        self.franka_dof_upper_limits = None        # Actuator upper limits for franka

        # self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        # self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        # self.franka_position_noise = self.cfg["env"]["frankaPositionNoise"]
        # self.franka_rotation_noise = self.cfg["env"]["frankaRotationNoise"]
        self.franka_dof_noise = self.cfg["env"]["frankaDofNoise"] if "frankaDofNoise" in self.cfg["env"] else 0.0

        self.gripper_freq_inv = self.cfg.get("gripper_freq_inv", 1)
        self.gripper_freq = self.control_freq // self.gripper_freq_inv
        assert self.control_freq % self.gripper_freq_inv == 0, \
            f"Gripper inv must be a multiple of control frequency, but got control: {self.control_freq} and gripper: {self.gripper_freq_inv}"
        # assert self.gripper_freq <= 2, \
        #     f"Gripper frequency must be less than 2Hz, but got {self.gripper_freq}"
        
        # Number of Interpolate Joints for each control command 
        self.num_inter_steps = self.cfg.get("interpolate_joints", 1)
        assert self.num_inter_steps >= 1, "Number of interpolation steps must be at least 1."
        if self.num_inter_steps > 1:
            if self.training:
                # Because the control dt is incorrect in this case
                raise ValueError("Interpolation is not supported during training. Please set 'interpolate_joints' to 1.")
            if self.control_type == "osc":
                raise ValueError("Interpolation is not supported for OSC control type. Please set 'interpolate_joints' to 1.")
        self.max_episode_length *= self.num_inter_steps

    
    ###################################### Utils ##############################################
    def set_meta_parameters(self,train, num_total_tasks):
        """ Randomly sample meta_batch_size tasks from the set of all 50 tasks for either training or testing,
            and set the last_rand_vecs to the sampled training/testing tasks.
        MUST call reset after this function is called to reset the envs to the new tasks.

        Returns the sampled tasks as indexes
        """
        assert self.meta_batch_size <= num_total_tasks, "meta_batch_size should be less than or equal to num_total_tasks"

        task_indices = np.random.permutation(num_total_tasks)[:self.meta_batch_size] # permute [0,...,49]
        # task_indices = np.arange(num_total_tasks)[:self.meta_batch_size]
        if train:
            for i, task_idx in enumerate(task_indices):
                self.last_rand_vecs[i*self.num_envs_per_task:(i+1)*self.num_envs_per_task] = self.all_train_tasks[task_idx]
            return task_indices
        else:
            self.last_rand_vecs[:] = self.all_train_tasks[task_indices[0]]
            return [task_indices[0]]
        

    def render_headless(self):
        if self.control_steps % self.camera_capture_step_interval == 0:
            camera_idx = self.camera_render_env_id
            camera_tensor = self.torch_camera_tensors[camera_idx]
            frame = None
            if camera_tensor is not None and not self.camera_force_cpu_capture:
                frame = camera_tensor.clone()

            if frame is None:
                rendering_camera = self.rendering_cameras[camera_idx]
                if rendering_camera == -1:
                    return
                frame = self.gym.get_camera_image(
                    self.sim,
                    self.envs[camera_idx],
                    rendering_camera,
                    gymapi.IMAGE_COLOR,
                )
                if frame is None:
                    return
                frame = torch.from_numpy(np.asarray(frame).reshape(
                    self.camera_props.height,
                    self.camera_props.width,
                    4,
                ).copy())

            if frame is None:
                return
            if self.manual_camera_capture:
                self.manual_camera_frames.append(frame)
            else:
                self.camera_frames.append(frame)


    def capture_manual_camera_frame(self):
        if not self.debug_viz or not self.manual_camera_capture:
            return False
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        if self.camera_force_cpu_capture:
            self.render_headless()
            return True
        self.gym.start_access_image_tensors(self.sim)
        try:
            self.render_headless()
        finally:
            self.gym.end_access_image_tensors(self.sim)
        return True


    def begin_manual_camera_capture(self, env_id: int):
        if not self.debug_viz:
            return False
        self.camera_render_env_id = int(env_id)
        self.manual_camera_frames = []
        self.manual_camera_capture = True
        self.camera_rendering = True
        return True


    def end_manual_camera_capture(self):
        if not self.debug_viz:
            return []
        self.manual_camera_capture = False
        self.camera_rendering = False
        frames = self.manual_camera_frames
        self.manual_camera_frames = []
        return frames

    
    def get_debug_viz(self):
        if not self.debug_viz or not self.camera_request_visual:
            return None

        if self.camera_request_visual:
            if len(self.camera_frames) == 0:
                if self.progress_buf[-1].item() == 1:
                    # only start rendering at first time step
                    self.camera_rendering = True
                return None
        
            if len(self.camera_frames) < self.max_episode_length / 5:
                return None
            else:
                self.camera_rendering = False
                self.camera_request_visual = False
                camera_frames = torch.stack(self.camera_frames, dim=0).permute(0, 3, 1, 2).unsqueeze(0)
                self.camera_frames = []
                return camera_frames
        else:
            return None
    
    
    def point2frankaBase(self, quat, pos):
        """
        Convert the obj center axes from the world frame to the franka base frame.
        pos: (n_envs, 3), quat: (n_envs, 4).
        Return: quat, pos
        """
        return tf_combine(self._franka2W_quat, self._franka2W_pos, quat, pos)
    
    
    def vec2frankaBase(self, vec):
        """
        Convert the vector from the world frame to the franka base frame.
        vec: (n_envs, 3).
        Return: vec
        """
        return quat_apply(self._franka2W_quat, vec)
    
    
    def points2frankaBase(self, pos, quat=None):
        """
        Convert obj center points from the world frame to the franka base frame.
        pos: (n_envs, n_points, 3).
        Return: pos
        """
        _, m, _ = pos.shape
        _franka2W_quat = self._franka2W_quat.unsqueeze(1).repeat(1, m, 1)
        _franka2W_pos = self._franka2W_pos.unsqueeze(1).repeat(1, m, 1)
        return tf_apply(_franka2W_quat, _franka2W_pos, pos)

    
    ###################################### Real Robot ##############################################
    def _update_task_states_real(self, state_estimator):
        pass


    def _preprocess_robot_states_real(self, franka_states_real):
        franka_states = {}
        franka_states.update({
            "eef_pos": np.array(franka_states_real["eef_pos"]),
            "eef_quat": np.array(franka_states_real["eef_quat"]),
            "q": franka_states_real["q"],
            "qd": franka_states_real["qd"],
            "q_gripper": franka_states_real["q_gripper"],
            "eef_vel": franka_states_real["eef_vel"],
            "mm": np.array(franka_states_real["mm"]).reshape(7, 7),
            "j_eef": np.array(franka_states_real["j_eef"]).reshape(6, 7),

            "prev_tgtq": self.prev_tgtq[0].cpu().numpy(),
            "prev_dq": self.prev_dq[0].cpu().numpy(),
            "gripper_mode": self._gripper_mode_temp[0].cpu().numpy(),
        })

        return franka_states
    

    def pre_physics_step_real(self, actions):
        self.convert_actions(actions)
        u_arm = self.command_arm_real().flatten()
        u_gripper = self.command_gripper_real().flatten()
        u = torch.cat([u_arm, u_gripper], dim=-1)
        
        return u.cpu().tolist()
    

    def command_arm_real(self):
        # Split arm and gripper command (keep the dim)
        dpose = self.actions[:, :-1]
        # Scale arm value first
        dpose = dpose * self.action_scale * self.act_scale

        states = self.states_real
        if self.control_type == "osc":
            eef_vel = to_torch(states["eef_vel"], device=self.device).unsqueeze(0)
            q = to_torch(states["q"], device=self.device).unsqueeze(0)
            qd = to_torch(states["qd"], device=self.device).unsqueeze(0)
            mm = to_torch(states["mm"], device=self.device).unsqueeze(0)
            j_eef = to_torch(states["j_eef"], device=self.device).unsqueeze(0)
            u_arm = self._compute_osc_torques(
                dpose=dpose, 
                eef_vel=eef_vel, 
                q=q, 
                qd=qd, 
                j_eef=j_eef, 
                mm=mm,
                kp=self.kp_real,
                kd=self.kd_real,
                kp_null=self.kp_null_real,
                kd_null=self.kd_null_real,
            )
            self._arm_control[:] = u_arm

        elif self.control_type == "ik":
            q = to_torch(states["q"], device=self.device).unsqueeze(0)
            j_eef = to_torch(states["j_eef"], device=self.device).unsqueeze(0)
            u_arm = self._differential_ik(dpose=dpose, q=q, j_eef=j_eef)

        elif self.control_type == "jp":
            q = to_torch(states["q"], device=self.device).unsqueeze(0)
            u_arm = self._joint_fk(dpose=dpose, q=q)
            self._arm_pos_control[:] = u_arm
        
        return u_arm


    def command_gripper_real(self):
        raw_gripper = self.actions[:, -1]
        # Gripper control
        self.gripper_ctrl_counts_real += 1
        if self.gripper_ctrl_counts_real >= self.gripper_freq_inv:
            self.gripper_ctrl_counts_real = 0
            cmd_gripper = torch.where(raw_gripper > 0., 1., -1.)
            u_gripper = torch.where(cmd_gripper==self.prev_u_gripper_real, 0., cmd_gripper)
            self.prev_u_gripper_real = self._gripper_mode[0] = self._gripper_mode_temp[0] = cmd_gripper
        else:
            u_gripper = torch.zeros_like(raw_gripper)

        return u_gripper
    

    def _update_task_common_info_real(self):
        self.extras.update({
            # Franka Related
            "joint_tgt_q": self._arm_pos_control[0].clone(),
            "joint_torqs": self._effort_control[0].clone(),
            "joint_poss": self.states_real["q"],
            "joint_vels": self.states_real["qd"], # Sensor reading velocity is kind of noisy!
            "joint_gripper_poss": self.states_real["q_gripper"],
            "joint_velocity_limits": self.cur_franka_velocity_limits[0].clone(),
        })
