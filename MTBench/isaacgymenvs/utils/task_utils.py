"""Utility functions for Isaac Gym environments."""

import numpy as np
import torch
from typing import Union, Optional
from isaacgymenvs.utils.torch_jit_utils import quat_mul, axisangle2quat


def tensor_clamp(x: torch.Tensor, 
                 lower: torch.Tensor, 
                 upper: torch.Tensor) -> torch.Tensor:
    """Clamp tensor values element-wise."""
    return torch.max(torch.min(x, upper), lower)


def mix_norm(x: Union[torch.Tensor, np.ndarray], 
             dim: int = -1, 
             keepdim: bool = False) -> Union[torch.Tensor, np.ndarray]:
    """Compute norm for both torch tensors and numpy arrays."""
    if isinstance(x, torch.Tensor):
        return torch.norm(x, dim=dim, keepdim=keepdim)
    elif isinstance(x, np.ndarray):
        return np.linalg.norm(x, axis=dim, keepdims=keepdim)
    else:
        raise TypeError(f"Unsupported type {type(x)}")


def mix_clone(x: Union[torch.Tensor, np.ndarray, list]):
    """Clone for both torch tensors and numpy arrays."""
    if isinstance(x, torch.Tensor):
        return x.clone()
    elif isinstance(x, np.ndarray) or isinstance(x, list):
        return x.copy()
    else:
        raise TypeError(f"Unsupported type {type(x)} for cloning.")


def to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)


def to_torch(x, device: str, dtype=torch.float) -> torch.Tensor:
    """Convert to torch tensor."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def ema_filter(x: torch.Tensor, prev_x: torch.Tensor, alpha: float = 0.9) -> torch.Tensor:
    """Exponential moving average filter."""
    return alpha * prev_x + (1 - alpha) * x


def unify_quat(quat: torch.Tensor) -> torch.Tensor:
    """
    Unify quaternion representation to avoid ambiguity.
    Makes the largest absolute element positive.
    
    Args:
        quat: (N, 4) quaternion tensor (x, y, z, w)
    """
    max_idx = torch.argmax(torch.abs(quat), dim=-1)
    sign = torch.sign(quat[torch.arange(len(quat)), max_idx])
    return quat * sign.unsqueeze(-1)


def unify_quat_np(quat: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Numpy version of unify_quat."""
    if isinstance(quat, torch.Tensor):
        quat = quat.cpu().numpy()
    elif not isinstance(quat, np.ndarray):
        quat = np.array(quat)

    max_idx = np.argmax(np.abs(quat))
    sign = np.sign(quat[max_idx])
    return quat * sign


def project_point_on_segment(x: torch.Tensor, 
                             y: torch.Tensor, 
                             z: torch.Tensor) -> tuple:
    """
    Project point z onto line segment connecting x and y.
    
    Args:
        x: Start point (num_envs, d)
        y: End point (num_envs, d)
        z: Points to project (num_envs, n, d) or (num_envs, d)
    
    Returns:
        proj: Projection points (num_envs, n, d)
        dist: Distance to segment (num_envs, n)
        p_scale: Parametric position along segment (num_envs, n)
    """
    if z.dim() == 2:
        z = z.unsqueeze(1)
    
    num_gms = z.size(1)
    x, y = x.unsqueeze(1), y.unsqueeze(1)
    
    v = (y - x).repeat(1, num_gms, 1)
    w = z - x
    v_norm_sq = torch.clamp(torch.sum(v * v, dim=-1, keepdim=True), min=1e-8)
    p_scale = torch.sum(w * v, dim=-1, keepdim=True) / v_norm_sq
    
    proj = x + p_scale * v
    dist = torch.norm(z - proj, dim=-1)
    
    return proj, dist, p_scale.squeeze(-1)


def is_in_cup(gms_pos: torch.Tensor, 
              cup_pos: torch.Tensor, 
              cup_rimpos: torch.Tensor, 
              cup_size: torch.Tensor) -> torch.Tensor:
    """
    Check if points are inside a cup.
    
    Args:
        gms_pos: Point positions (num_envs, num_points, 3) or (num_envs, 3)
        cup_pos: Cup bottom positions (num_envs, 3)
        cup_rimpos: Cup rim positions (num_envs, 3)
        cup_size: Cup dimensions [radius, height, thickness] (num_envs, 3)
    
    Returns:
        Boolean tensor indicating if points are in cup
    """
    cup_radius = cup_size[:, 0] / 2
    cup_thickness = cup_size[:, 2]
    
    if gms_pos.dim() == 2:
        gms_pos = gms_pos.unsqueeze(1)
    
    _, dist, p_scale = project_point_on_segment(cup_pos, cup_rimpos, gms_pos)
    
    in_radius = dist <= (cup_radius - cup_thickness).unsqueeze(-1)
    in_height = (0 <= p_scale) & (p_scale <= 1)
    
    return in_radius & in_height


def is_under_valid_vel(linvel_norm, vel_limit=10.):
    """
    Check if the object is under the velocity limit
    Input:
        linvel_norm: (num_envs, 1)
        vel_limit: float
    Output:
        in_vel: (num_envs, 1)
    """
    # Check if the object is under the velocity limit
    return linvel_norm <= vel_limit


def is_under_valid_contact(contact_forces_norm, contact_limit=10.):
    """
    Check if the contact forces are under the limit
    Input:
        contact_forces: (num_envs, num_contacts, 3)
        contact_limit: float
    Output:
        in_contact: (num_envs, num_contacts)
    """
    # Check if the contact forces are under the limit
    return contact_forces_norm <= contact_limit


def extract_features(config: dict, state_names: list) -> np.ndarray:
    """
    Extract and concatenate features from a configuration.
    
    Args:
        config: Dictionary containing state values
        state_names: List of state names to extract
    
    Returns:
        Concatenated feature vector
    """
    features = [to_numpy(config[key]) for key in state_names]
    return np.concatenate(features, axis=-1).flatten()


def calculate_weighted_distances(new_features: np.ndarray, 
                                 features_array: np.ndarray) -> np.ndarray:
    """
    Calculate L2 distances between one feature vector and multiple vectors.
    
    Args:
        new_features: Single feature vector (d,)
        features_array: Multiple feature vectors (N, d)
    
    Returns:
        Array of distances (N,)
    """
    new_features_broadcast = new_features[np.newaxis, :]
    return np.linalg.norm(new_features_broadcast - features_array, axis=1)


def _apply_cube_state_noise(template_cube_state, env_ids, noise_scale=0.01):
    """
    Apply small spatial random noise to the cube state, when using fixed configurations.
    """
    device = template_cube_state.device
    num_resets = len(env_ids)
    sampled_cube_state = template_cube_state.repeat(num_resets, 1)
    # We just directly sample
    sampled_cube_state[:, :2] = sampled_cube_state[:, :2] + \
                                2.0 * noise_scale * (
                                torch.rand(num_resets, 2, device=device) - 0.5)
    aa_rot = torch.zeros(num_resets, 3, device=device)
    aa_rot[:, 2] = 2.0 * noise_scale * (torch.rand(num_resets, device=device) - 0.5)
    sampled_cube_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_cube_state[:, 3:7])
    return sampled_cube_state
    
    
def unify_quat(quat):
    """
    We make sure the abs value of largest element of the quaternion is positive to avoid any ambiguity.
    quat: (N, 4); (x, y, z, w)
    """
    max_idx = torch.argmax(torch.abs(quat), dim=-1)
    sign = torch.sign(quat[torch.arange(len(quat)), max_idx])
    quat = quat * sign.unsqueeze(-1)
    return quat


def unify_quat_np(quat):
    """
    numpy version of unify_quat
    We make sure the abs value of largest element of the quaternion is positive to avoid any ambiguity.
    quat: (4, ); (x, y, z, w)
    """
    if isinstance(quat, torch.Tensor):
        quat = quat.cpu().numpy()
    elif not isinstance(quat, np.ndarray):
        quat = np.array(quat)

    max_idx = np.argmax(np.abs(quat))
    sign = np.sign(quat[max_idx])
    quat = quat * sign
    return quat