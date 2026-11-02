import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from core.common.task_layout import build_task_id_lookup
from core.common.tensor import get_args_attr


# ---------------------------------------------------------------------------
# Schedulers
# ---------------------------------------------------------------------------

class AdaptiveScheduler:
    def __init__(self, kl_threshold=0.008):
        super().__init__()
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, current_lr, kl_dist):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr


class LinearScheduler(AdaptiveScheduler):
    def __init__(self, start_lr, min_lr=1e-6, max_steps=1000000, apply_to_entropy=False, **kwargs):
        super().__init__()

        self.start_lr = start_lr
        self.min_lr = min_lr
        self.max_steps = max_steps
        self.apply_to_entropy = apply_to_entropy
        if apply_to_entropy:
            self.start_entropy_coef = kwargs.pop("start_entropy_coef", 0.01)
            self.min_entropy_coef = kwargs.pop("min_entropy_coef", 0.0001)

    def update(self, steps, entropy_coef=0.0):
        mul = max(0, self.max_steps - steps) / self.max_steps
        lr = self.min_lr + (self.start_lr - self.min_lr) * mul
        if self.apply_to_entropy:
            entropy_coef = self.min_entropy_coef + (self.start_entropy_coef - self.min_entropy_coef) * mul

        return lr, entropy_coef


def linear_amplifier(start_v, end_v, cur_step, max_steps, curri_rate=1):
    alpha = min(curri_rate * cur_step / max_steps, 1)
    next_v = (1 - alpha) * start_v + alpha * end_v
    return next_v


# ---------------------------------------------------------------------------
# Network building blocks
# ---------------------------------------------------------------------------

ACTIVATION_DICT = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "selu": nn.SELU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "softplus": nn.Softplus,
    "mish": nn.Mish,
    "identity": nn.Identity,
}


def get_activation(activation):
    """Get an activation function instance from a string name."""
    if not isinstance(activation, str):
        raise TypeError(f"activation must be a str, got {type(activation)}")
    key = activation.lower()
    if key not in ACTIVATION_DICT:
        raise ValueError(f"Unknown activation '{activation}'. Choose from: {list(ACTIVATION_DICT.keys())}")
    return ACTIVATION_DICT[key]()


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=None, activation="tanh", use_layernorm=True, output_layernorm=False, output_softplus=False, init_std=1.0):
        super().__init__()
        self.activation = get_activation(activation)
        self.mlp = nn.Sequential()

        if type(hidden_size) not in [list, tuple]:
            assert num_hidden_layers is not None, f"num_hidden_layers must be specified if hidden_size is an int number {hidden_size}"
            hidden_size = [hidden_size] * num_hidden_layers
        else:
            num_hidden_layers = len(hidden_size)

        for i in range(num_hidden_layers):
            input_shape = input_size if i == 0 else hidden_size[i - 1]
            self.mlp.append(layer_init(nn.Linear(input_shape, hidden_size[i])))
            if use_layernorm:
                self.mlp.append(nn.LayerNorm(hidden_size[i]))
            self.mlp.append(deepcopy(self.activation))

        last_hidden_size = input_size if num_hidden_layers == 0 else hidden_size[-1]
        self.mlp.append(layer_init(nn.Linear(last_hidden_size, output_size), std=init_std))

        if output_layernorm:
            self.mlp.append(nn.LayerNorm(output_size))
        if output_softplus:
            self.mlp.append(nn.Softplus())

    def forward(self, x):
        return self.mlp(x)


# ---------------------------------------------------------------------------
# Observation, value, and reward normalization
# ---------------------------------------------------------------------------

class PerTaskEmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values for each task."""

    def __init__(self, num_tasks: int, shape: tuple, device: torch.device, eps: float = 1e-2, until: int = None):
        super().__init__()
        if not isinstance(shape, tuple):
            shape = (shape,)
        self.num_tasks = num_tasks
        self.shape = shape
        self.eps = eps
        self.until = until
        self.device = device

        self.register_buffer("_mean", torch.zeros(num_tasks, *shape).to(device))
        self.register_buffer("_var", torch.ones(num_tasks, *shape).to(device))
        self.register_buffer("_std", torch.ones(num_tasks, *shape).to(device))
        self.register_buffer("count", torch.zeros(num_tasks, dtype=torch.long).to(device))

    def forward(self, x: torch.Tensor, task_ids: torch.Tensor, center: bool = True) -> torch.Tensor:
        if x.shape[1:] != self.shape:
            raise ValueError(f"Expected input shape (*, {self.shape}), got {x.shape}")
        if x.shape[0] != task_ids.shape[0]:
            raise ValueError("Batch size of x and task_ids must match.")

        view_shape = (task_ids.shape[0],) + (1,) * len(self.shape)
        task_ids_expanded = task_ids.view(view_shape).expand_as(x)

        mean = self._mean.gather(0, task_ids_expanded)
        std = self._std.gather(0, task_ids_expanded)

        if self.training:
            self.update(x, task_ids)

        if center:
            return (x - mean) / (std + self.eps)
        return x / (std + self.eps)

    @torch.jit.unused
    def update(self, x: torch.Tensor, task_ids: torch.Tensor):
        unique_tasks = torch.unique(task_ids)

        for task_id in unique_tasks:
            if self.until is not None and self.count[task_id] >= self.until:
                continue

            mask = task_ids == task_id
            x_task = x[mask]
            batch_size = x_task.shape[0]

            if batch_size == 0:
                continue

            old_count = self.count[task_id].clone()
            new_count = old_count + batch_size

            task_mean = self._mean[task_id]
            batch_mean = torch.mean(x_task, dim=0)
            delta = batch_mean - task_mean
            self._mean[task_id].copy_(task_mean + (batch_size / new_count) * delta)

            if old_count > 0:
                batch_var = torch.var(x_task, dim=0, unbiased=False)
                m_a = self._var[task_id] * old_count
                m_b = batch_var * batch_size
                m2 = m_a + m_b + (delta ** 2) * (old_count * batch_size / new_count)
                self._var[task_id].copy_(m2 / new_count)
            else:
                self._var[task_id].copy_(torch.var(x_task, dim=0, unbiased=False))

            self._std[task_id].copy_(torch.sqrt(self._var[task_id]))
            self.count[task_id].copy_(new_count)


class PerTaskRewardNormalizer(nn.Module):
    """Per-task reward normalizer copied from MTBench for local use."""

    def __init__(self, num_tasks, gamma: float, device: torch.device, g_max: float = 10.0, epsilon: float = 1e-8):
        super().__init__()
        self.gamma = gamma
        self.g_max = g_max
        self.epsilon = epsilon
        self.device = device

        if isinstance(num_tasks, torch.Tensor):
            unique_task_ids = num_tasks.to(device=device, dtype=torch.long)
        else:
            unique_task_ids = torch.arange(num_tasks, device=device, dtype=torch.long)
        self.num_tasks = len(unique_task_ids)
        self.register_buffer("tid_to_tidx", build_task_id_lookup(unique_task_ids, device=device))

        self.register_buffer("G", torch.zeros(self.num_tasks, device=device))
        self.register_buffer("G_r_max", torch.zeros(self.num_tasks, device=device))
        self.G_rms = PerTaskEmpiricalNormalization(num_tasks=self.num_tasks, shape=(1,), device=device)

    def _task_rows(self, task_ids: torch.Tensor) -> torch.Tensor:
        return self.tid_to_tidx[task_ids]

    def _scale_reward(self, rewards: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        task_rows = self._task_rows(task_ids)
        std_for_batch = self.G_rms._std.gather(0, task_rows.unsqueeze(-1)).squeeze(-1)
        g_r_max_for_batch = self.G_r_max.gather(0, task_rows)

        var_denominator = std_for_batch + self.epsilon
        min_required_denominator = g_r_max_for_batch / self.g_max
        denominator = torch.maximum(var_denominator, min_required_denominator)
        return rewards / (denominator + self.epsilon)

    def update_stats(self, rewards: torch.Tensor, dones: torch.Tensor, task_ids: torch.Tensor):
        if not (rewards.shape == dones.shape == task_ids.shape):
            raise ValueError("rewards, dones, and task_ids must have the same shape.")

        task_rows = self._task_rows(task_ids)
        prev_G = self.G.gather(0, task_rows)
        new_G = self.gamma * (1 - dones.float()) * prev_G + rewards
        self.G.scatter_(0, task_rows, new_G)
        self.G_rms.update(new_G.unsqueeze(-1), task_rows)

        prev_G_r_max = self.G_r_max.gather(0, task_rows)
        updated_G_r_max = torch.maximum(prev_G_r_max, torch.abs(new_G))
        self.G_r_max.scatter_(0, task_rows, updated_G_r_max)

    def forward(self, rewards: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        return self._scale_reward(rewards, task_ids)


class PerTaskRunningMeanStd(nn.Module):
    """Per-task running mean/std normalizer with vectorized forward pass."""

    def __init__(self, insize: int, unique_task_ids: torch.Tensor = None, center: bool = True, clip: bool = True, clip_range: float = 5.0, epsilon: float = 1e-8, dtype: torch.dtype = torch.float32, device: str = "cuda"):
        super().__init__()
        unique_task_ids = torch.tensor([0], dtype=torch.long, device=device) if unique_task_ids is None else unique_task_ids
        self.insize = insize
        self.epsilon = epsilon
        self.num_tasks = len(unique_task_ids)
        self.center = center
        self.clip = clip
        self.clip_range = clip_range
        self.dtype = dtype
        self.device = device

        self.register_buffer("tid_to_tidx", build_task_id_lookup(unique_task_ids, device=device))

        self.register_buffer("running_mean", torch.zeros(self.num_tasks, insize, dtype=dtype, device=device))
        self.register_buffer("running_var", torch.ones(self.num_tasks, insize, dtype=dtype, device=device))
        self.register_buffer("count", torch.ones(self.num_tasks, dtype=dtype, device=device))

    def reset(self):
        self.running_mean.zero_()
        self.running_var.fill_(1.0)
        self.count.fill_(1.0)

    @torch.no_grad()
    def _update(self, x: torch.Tensor, tidx: torch.Tensor):
        for i in range(self.num_tasks):
            mask = tidx == i
            if not mask.any():
                continue
            x_task = x[mask]
            batch_count = x_task.shape[0]
            old_count = self.count[i]
            new_count = old_count + batch_count

            batch_mean = x_task.mean(0)
            delta = batch_mean - self.running_mean[i]
            self.running_mean[i] += delta * (batch_count / new_count)

            batch_var = x_task.var(0, unbiased=False)
            m_a = self.running_var[i] * old_count
            m_b = batch_var * batch_count
            m2 = m_a + m_b + delta ** 2 * (old_count * batch_count / new_count)
            self.running_var[i] = m2 / new_count
            self.count[i] = new_count

    def forward(self, x: torch.Tensor, task_ids=None, denorm: bool = False, update: bool = False) -> torch.Tensor:
        with torch.no_grad():
            scalar_input = x.dim() == 1
            if scalar_input:
                x = x.unsqueeze(-1)

            if task_ids is None or self.num_tasks == 1:
                if update:
                    tidx = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
                    self._update(x, tidx)
                mean = self.running_mean[0]
                var = self.running_var[0]
            else:
                tidx = self.tid_to_tidx[task_ids]
                if update:
                    self._update(x, tidx)
                tidx_exp = tidx.unsqueeze(-1).expand_as(x)
                mean = self.running_mean.gather(0, tidx_exp)
                var = self.running_var.gather(0, tidx_exp)

            std = torch.sqrt(var + self.epsilon)
            if denorm:
                if self.clip:
                    x = torch.clamp(x, -self.clip_range, self.clip_range)
                y = x * std + mean if self.center else x * std
            else:
                y = (x - mean) / std if self.center else x / std
                if self.clip:
                    y = torch.clamp(y, -self.clip_range, self.clip_range)

            if scalar_input:
                y = y.squeeze(-1)
            return y


class RunningMeanStd(nn.Module):
    """Updates statistics from full-batch data."""

    def __init__(self, insize, epsilon=1e-8, per_channel=False, std_only=False, device="cuda"):
        super().__init__()
        self.insize = insize
        self.epsilon = epsilon
        self.std_only = std_only
        self.per_channel = per_channel
        if per_channel:
            if len(self.insize) == 3:
                self.axis = [0, 2, 3]
            if len(self.insize) == 2:
                self.axis = [0, 2]
            if len(self.insize) == 1:
                self.axis = [0]
            in_size = self.insize[0]
        else:
            self.axis = [0]
            in_size = insize

        self.register_buffer("running_mean", torch.zeros(in_size, dtype=torch.float64, device=device))
        self.register_buffer("running_var", torch.ones(in_size, dtype=torch.float64, device=device))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64, device=device))

    def reset(self, reset_slice=None):
        if reset_slice is None:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.count.fill_(0)
        else:
            self.running_mean[reset_slice].zero_()
            self.running_var[reset_slice].fill_(1)
            self.count[reset_slice].fill_(0)

    def update(self, x):
        with torch.no_grad():
            batch_mean = x.mean(self.axis)
            batch_var = x.var(self.axis)
            batch_count = x.size()[0]
            self.running_mean, self.running_var, self.count = self.update_mean_var_count_from_moments(
                self.running_mean,
                self.running_var,
                self.count,
                batch_mean,
                batch_var,
                batch_count,
            )

    def update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * count * batch_count / tot_count
        new_var = m2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    def forward(self, input, denorm=False, update=False):
        with torch.no_grad():
            prev_shape = None
            if len(input.shape) == 1:
                prev_shape = input.shape
                input = input.view(-1, 1)

            if update:
                self.update(input)

            if self.per_channel:
                if len(self.insize) == 3:
                    current_mean = self.running_mean.view([1, self.insize[0], 1, 1]).expand_as(input)
                    current_var = self.running_var.view([1, self.insize[0], 1, 1]).expand_as(input)
                if len(self.insize) == 2:
                    current_mean = self.running_mean.view([1, self.insize[0], 1]).expand_as(input)
                    current_var = self.running_var.view([1, self.insize[0], 1]).expand_as(input)
                if len(self.insize) == 1:
                    current_mean = self.running_mean.view([1, self.insize[0]]).expand_as(input)
                    current_var = self.running_var.view([1, self.insize[0]]).expand_as(input)
            else:
                current_mean = self.running_mean
                current_var = self.running_var

            if denorm:
                y = torch.clamp(input, min=-5.0, max=5.0)
                y = torch.sqrt(current_var.float() + self.epsilon) * y + current_mean.float()
            else:
                if self.std_only:
                    y = input / torch.sqrt(current_var.float() + self.epsilon)
                else:
                    y = (input - current_mean.float()) / torch.sqrt(current_var.float() + self.epsilon)
                    y = torch.clamp(y, min=-5.0, max=5.0)

            if prev_shape is not None:
                y = y.view(prev_shape)
            return y


class NormalizeReward(nn.Module):
    def __init__(self, num_envs, insize: int = 1, gamma: float = 0.99, epsilon: float = 1e-8, device="cuda"):
        super().__init__()
        print("Creating Normalizer NormalizeReward | Size: ", insize)
        self.return_rms = RunningMeanStd(insize, epsilon, std_only=True)
        self.returns = torch.zeros((num_envs, insize), dtype=torch.float32, device=device)
        self.gamma = gamma if insize == 1 else gamma.repeat(num_envs, 1)
        self.epsilon = epsilon

    def reset(self):
        self.return_rms.reset()
        self.returns = torch.zeros_like(self.returns)

    def normalize(self, rews, dones):
        org_shape = rews.shape
        rews = rews.view(self.returns.shape)
        self.returns = self.returns * self.gamma * (1 - dones).view(-1, 1) + rews
        self.return_rms.update(self.returns)
        post_rews = rews / torch.sqrt(self.return_rms.running_var + self.epsilon)
        return post_rews.view(org_shape)


# ---------------------------------------------------------------------------
# PPO loss and checkpoint helpers
# ---------------------------------------------------------------------------

def bound_loss(mu, soft_bound=1.1):
    mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
    mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
    b_loss = (mu_loss_low + mu_loss_high).sum(dim=-1).mean()
    return b_loss


def save_checkpoint(model, folder_path, ckpt_name="rew_eps", suffix="", ckpt_path=None, verbose=False):
    if model is None:
        return
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if ckpt_path is None:
        ckpt_path = "{}/{}_{}".format(folder_path, ckpt_name, suffix)
    if verbose:
        print("Saving models to {}".format(ckpt_path))
    filtered_state_dict = {k: v for k, v in model.state_dict().items()}
    torch.save(filtered_state_dict, ckpt_path)


def load_checkpoint(model, ckpt_path, evaluate=False, map_location="cuda:0"):
    print("Loading models from {}".format(ckpt_path))
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location=map_location)
        model.load_state_dict(checkpoint, strict=False)
    else:
        print(f"WARN: Can not find checkpoint path: {ckpt_path}")

    if evaluate:
        model.eval()
    else:
        model.train()

    return model


# ---------------------------------------------------------------------------
# Tensor buffer helpers
# ---------------------------------------------------------------------------

def update_tensor_buffer(buffer, new_v):
    len_v = len(new_v)
    if len_v == 0:
        return
    if len_v > len(buffer):
        buffer[:] = new_v[-len(buffer):]
    else:
        buffer[:-len_v] = buffer[len_v:].clone()
        buffer[-len_v:] = new_v
