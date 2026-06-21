"""Robot controller implementations."""

import torch
import numpy as np
from typing import Optional
from isaacgymenvs.utils.task_utils import tensor_clamp, ema_filter


def pinv_analytical(j_eef: torch.Tensor, damping_factor: float = 0.05) -> torch.Tensor:
    """
    Analytical pseudo-inverse for wide, full-row-rank Jacobian matrices.
    
    Args:
        j_eef: Jacobian tensor of shape (N, 6, 7)
        eps: Regularization term for numerical stability
    
    Returns:
        Pseudo-inverse of shape (N, 7, 6)
    """
    jjt = j_eef @ j_eef.transpose(-1, -2)  # (N, 6, 6)
    eye = torch.eye(jjt.shape[-1], device=j_eef.device).expand_as(jjt)
    jjt_reg = jjt + damping_factor**2 * eye
    jjt_inv = torch.inverse(jjt_reg)
    return j_eef.transpose(-1, -2) @ jjt_inv  # (N, 7, 6)


class FrankaController:
    """Controller for Franka Panda robot."""
    
    def __init__(self, 
                 num_envs: int,
                 device: str,
                 franka_dof_lower_limits,
                 franka_dof_upper_limits,
                 ctrl_dt: float = 0.01,
                 ):
        """
        Initialize controller.
        
        Args:
            num_envs: Number of parallel environments
            device: Device to run on
            ctrl_dt: Control timestep
        """
        self.num_envs = num_envs
        self.device = device
        self.franka_dof_lower_limits = franka_dof_lower_limits
        self.franka_dof_upper_limits = franka_dof_upper_limits
        self.ctrl_dt = ctrl_dt
        
        # OSC gains
        self.kp_init = 150
        self.kp = torch.ones((num_envs, 6), device=device) * self.kp_init
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = torch.ones((num_envs, 7), device=device) * (self.kp_init / 15)
        self.kd_null = 2 * torch.sqrt(self.kp_null)
        
        # Real robot OSC gains
        self.kp_real_init = 100
        self.kp_real = torch.ones((1, 6), device=device) * self.kp_real_init
        self.kd_real = 2 * torch.sqrt(self.kp_real)
        self.kp_null_real = torch.ones((1, 7), device=device) * (self.kp_real_init / 15)
        self.kd_null_real = 2 * torch.sqrt(self.kp_null_real)
        
        # Joint position PD gains
        self.kp_jp_init = 100
        self.kp_jp = torch.ones((num_envs, 7), device=device) * self.kp_jp_init
        self.kd_jp = 2 * torch.sqrt(self.kp_jp)
        
        # Neutral joint configuration
        self.default_dof_pos = torch.tensor(
            [0, 0, 0, -2.3180, 0, 2.4416, 0.7854, 0.04, 0.04], 
            device=device
        )
    

    def compute_osc_torques(self,
                           dpose: torch.Tensor,
                           eef_vel: torch.Tensor,
                           q: torch.Tensor,
                           qd: torch.Tensor,
                           j_eef: torch.Tensor,
                           mm: torch.Tensor,
                           effort_limits: torch.Tensor,
                           kp: Optional[torch.Tensor] = None,
                           kd: Optional[torch.Tensor] = None,
                           kp_null: Optional[torch.Tensor] = None,
                           kd_null: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute operational space control torques.
        
        Args:
            dpose: Desired pose change (num_envs, 6)
            eef_vel: End-effector velocity (num_envs, 6)
            q: Joint positions (num_envs, 7)
            qd: Joint velocities (num_envs, 7)
            j_eef: End-effector Jacobian (num_envs, 6, 7)
            mm: Mass matrix (num_envs, 7, 7)
            effort_limits: Torque limits (7,)
            kp, kd, kp_null, kd_null: Optional gain overrides
        
        Returns:
            Joint torques (num_envs, 7)
        """
        kp = self.kp if kp is None else kp
        kd = self.kd if kd is None else kd
        kp_null = self.kp_null if kp_null is None else kp_null
        kd_null = self.kd_null if kd_null is None else kd_null
        
        # Compute operational space inertia matrix
        mm_inv = torch.inverse(mm)
        m_eef_inv = j_eef @ mm_inv @ torch.transpose(j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)
        
        # Task space control torques
        u = torch.transpose(j_eef, 1, 2) @ m_eef @ (
            kp * dpose - kd * eef_vel
        ).unsqueeze(-1)
        
        # Nullspace control (keeps arm in comfortable configuration)
        j_eef_inv = m_eef @ j_eef @ mm_inv
        u_null = kp_null * (
            (self.default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi
        ) - kd_null * qd
        u_null = mm @ u_null.unsqueeze(-1)
        u_null = (
            torch.eye(7, device=self.device).unsqueeze(0) - 
            torch.transpose(j_eef, 1, 2) @ j_eef_inv
        ) @ u_null
        
        u += u_null
        
        # Clip to effort limits
        u = tensor_clamp(
            u.squeeze(-1),
            -effort_limits.unsqueeze(0),
            effort_limits.unsqueeze(0)
        )
        
        return u
    

    def differential_ik(self,
                       dpose: torch.Tensor,
                       q: torch.Tensor,
                       j_eef: torch.Tensor,
                       prev_dq: torch.Tensor,
                       velocity_limits: torch.Tensor,
                       alpha: float = 0.,
                       num_inter_steps: int = 1) -> tuple:
        """
        Compute differential inverse kinematics.
        
        Args:
            dpose: Desired pose change (num_envs, 6)
            q: Current joint positions (num_envs, 7)
            j_eef: End-effector Jacobian (num_envs, 6, 7)
            velocity_limits: Joint velocity limits (num_envs, 7)
            dof_limits_lower: Lower joint limits (7,)
            dof_limits_upper: Upper joint limits (7,)
            prev_dq: Previous joint velocity (num_envs, 7)
            alpha: EMA filter coefficient
            num_inter_steps: Number of interpolation steps
        
        Returns:
            target_q: Target joint positions (num_envs, 7)
            dq: Joint velocities (num_envs, 7)
            tgt_q_seq: Sequence of interpolated targets

        Warning:
            We can not clamp dq here, as it will make the ik computation totally wrong.
        """
        # Compute pseudo-inverse and delta joint angles
        j_eef_inv = pinv_analytical(j_eef, damping_factor=0.05)
        dq = (j_eef_inv @ dpose.unsqueeze(-1)).squeeze(-1)
        max_dq = velocity_limits * self.ctrl_dt

        # UNIFORM SCALING for acceleration limits (similar as ema filter)
        # dq_filtered = alpha * prev_dq + (1 - alpha) * dq
        # ddq = dq_filtered - prev_dq = (1 - alpha) * (dq - prev_dq)
        # Warning, this ddq constraint will still distort the ik solution but it is better than ema.
        if alpha > 0:
            ddq = dq - prev_dq
            max_ddq = (1 - alpha) * max_dq
            ddq_violation_ratio = torch.abs(ddq) / max_ddq
            ddq_violation_ratio = torch.max(ddq_violation_ratio, dim=-1, keepdim=True)[0]
            
            # Scale uniformly if any joint exceeds acceleration limit
            ddq_scale = torch.clamp(ddq_violation_ratio, min=1.0)
            ddq = ddq / ddq_scale
            dq = prev_dq + ddq

        # Or Apply EMA filter
        # dq = ema_filter(dq, prev_dq, alpha=alpha)
        
        # Limit velocity with UNIFORM scaling
        dq_violation_ratio = torch.abs(dq) / max_dq
        dq_violation_ratio = torch.max(dq_violation_ratio, dim=-1, keepdim=True)[0]
        dq = dq / torch.clamp(dq_violation_ratio, min=1.0)
        
        # Interpolate trajectory
        d_dq = dq / num_inter_steps
        tgt_q_seq = []
        for j in range(1, num_inter_steps + 1):
            step_tgt_q = q + j * d_dq
            step_tgt_q = tensor_clamp(
                step_tgt_q,
                self.franka_dof_lower_limits[:7].unsqueeze(0),
                self.franka_dof_upper_limits[:7].unsqueeze(0)
            )
            tgt_q_seq.append(step_tgt_q)
        
        target_q = tgt_q_seq[-1]

        return target_q, tgt_q_seq, dq 
    

    def compute_im_torques(self, 
                            dpose: torch.Tensor,
                            eef_vel: torch.Tensor,
                            j_eef: torch.Tensor,
                            effort_limits: torch.Tensor,
                            kp: Optional[torch.Tensor] = None,
                            kd: Optional[torch.Tensor] = None,
                            kp_null: Optional[torch.Tensor] = None,
                            kd_null: Optional[torch.Tensor] = None,
                            q: Optional[torch.Tensor] = None,
                            qd: Optional[torch.Tensor] = None,
                            default_dof_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        kp = self.kp if kp is None else kp
        kd = self.kd if kd is None else kd
        kp_null = self.kp_null if kp_null is None else kp_null
        kd_null = self.kd_null if kd_null is None else kd_null
        # Desired task-space force using PD law (first half is the spring force while the last half is the damping force)
        F_eef = kp * dpose - kd * eef_vel
        # Clip the values to be within valid effort range
        u = torch.transpose(j_eef, 1, 2) @ F_eef.unsqueeze(-1)
        
        # Add null space torques if joint states and default positions are provided
        if (q is not None) and (qd is not None) and (default_dof_pos is not None):
            q, qd = q[:, :7], qd[:, :7]
            # pseudo-inverse of the jacobian; torch.linalg.pinv Takes huge amount of time to compute (~0.2s for 1024 envs)
            # j_eef_inv = torch.linalg.pinv(self._j_eef, rcond=1e-2)
            j_eef_inv = pinv_analytical(j_eef)
            F_null = self.kp_null * ((default_dof_pos - q + np.pi) % (2 * np.pi) - np.pi) - kd_null * qd
            # Compute the null space projector: N = I - J_pinv * J (7x7)
            N_mat = torch.eye(7, device=self.device).unsqueeze(0) - j_eef_inv @ j_eef
            u_null = N_mat @ F_null.unsqueeze(-1)
            u += u_null

        u = tensor_clamp(u.squeeze(-1),
                         -effort_limits[:7].unsqueeze(0), 
                         effort_limits[:7].unsqueeze(0))

        return u
    

    
    def joint_position_control(self,
                               dpose: torch.Tensor,
                               q: torch.Tensor,
                               velocity_limits: torch.Tensor,
                               prev_dq: torch.Tensor,
                               alpha: float = 0.9) -> tuple:
        """
        Compute joint position control.
        
        Args:
            dpose: Normalized action (num_envs, 7) in [-1, 1]
            q: Current joint positions (num_envs, 7)
            velocity_limits: Joint velocity limits (7,)
            dof_limits_lower: Lower joint limits (7,)
            dof_limits_upper: Upper joint limits (7,)
            prev_dq: Previous joint velocity (num_envs, 7)
            alpha: EMA filter coefficient
        
        Returns:
            target_q: Target joint positions (num_envs, 7)
            dq: Joint velocities (num_envs, 7)
        """
        dq_max_abs = velocity_limits.unsqueeze(0) * self.ctrl_dt
        dq = dpose * dq_max_abs
        
        # Apply EMA filter
        dq = ema_filter(dq, prev_dq, alpha=alpha)
        dq = tensor_clamp(dq, -dq_max_abs, dq_max_abs)
        
        target_q = q + dq
        target_q = tensor_clamp(target_q, q - dq_max_abs, q + dq_max_abs)
        target_q = tensor_clamp(
            target_q,
            self.franka_dof_lower_limits.unsqueeze(0),
            self.franka_dof_upper_limits.unsqueeze(0)
        )
        
        return target_q, dq
    
    
    def compute_joint_pos(self, 
                          dpose, 
                          q,
                          prev_dq,
                          prev_tgtq,
                          velocity_limits,
                          alpha,
                          delta=False):
        """
        Forward Kinematics to compute the end-effector pose given joint angles.
        
        Inputs:
          q: Tensor of shape (batch_size, 7) representing the joint angles.
        
        Output:
          eef_pos: Tensor of shape (batch_size, 3) representing the end-effector position.
          eef_quat: Tensor of shape (batch_size, 4) representing the end-effector orientation as a quaternion.
        """
        # Compute forward kinematics using the robot's URDF model.
        q = q[:, :7]
        dq_max_abs = velocity_limits[:7].unsqueeze(0) * self.ctrl_dt
        dq = dpose * dq_max_abs
        if delta:
            return dq

        dq = ema_filter(dq, prev_dq, alpha=alpha)
        dq = tensor_clamp(dq, dq_max_abs, dq_max_abs)
        prev_dq[:] = dq.clone()

        tgt_q = prev_tgtq + dq
        tgt_q = tensor_clamp(tgt_q, q - dq_max_abs, q + dq_max_abs)
        tgt_q = tensor_clamp(tgt_q,
                             self.franka_dof_lower_limits[:7].unsqueeze(0),
                             self.franka_dof_upper_limits[:7].unsqueeze(0))
        prev_tgtq[:] = tgt_q.clone()
        
        return tgt_q, prev_dq, prev_tgtq
    

    def compute_joint_torque(self, dq, qd, effort_limits):
        u = self.kp_jp * dq - self.kd_jp * qd
        u = tensor_clamp(u,
                         -effort_limits[:7].unsqueeze(0), 
                         effort_limits[:7].unsqueeze(0))
        return u