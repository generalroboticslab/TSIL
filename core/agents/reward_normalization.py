"""Reward scaling from independent discounted reward traces per environment."""

import torch

from core.agents.utils import PerTaskRewardNormalizer


class PerEnvironmentReturnRewardNormalizer(PerTaskRewardNormalizer):
    """Pool reward-trace statistics per task without mixing episode histories.

    ``dones`` marks the transition whose reward is supplied in the same call.
    Its reward enters the completed episode's accumulator and statistics before
    that environment's accumulator resets. Rollout rows must retain environment
    identity across calls. Forward-only replay scaling may use arbitrary rows.
    This backward-discounted trace is a scale estimator, not the forward
    discounted episode return logged by RolloutCollector or a critic target.
    """

    def __init__(self, task_ids, environment_task_ids, gamma, device, g_max=10.0, epsilon=1e-8, scale_mode="std_magnitude_floor"):
        if scale_mode not in {"std_magnitude_floor", "std_only"}:
            raise ValueError("Invalid reward scale mode")
        self.scale_mode = scale_mode
        super().__init__(task_ids, gamma, device, g_max=g_max, epsilon=epsilon)
        environment_task_ids=torch.as_tensor(environment_task_ids,device=device,dtype=torch.long).clone()
        if environment_task_ids.ndim!=1 or not environment_task_ids.numel():
            raise ValueError('environment_task_ids must contain one task ID per environment')
        declared_task_ids=(task_ids.to(device=device,dtype=torch.long) if isinstance(task_ids,torch.Tensor)
                           else torch.arange(task_ids,device=device,dtype=torch.long))
        if not bool(torch.isin(environment_task_ids,declared_task_ids).all()):
            raise ValueError('All environment task IDs must belong to the declared task layout')
        task_rows=self._task_rows(environment_task_ids)
        if bool((task_rows<0).any()):
            raise ValueError('All environment task IDs must belong to the declared task layout')
        self.register_buffer("reward_scale_mode_id", torch.tensor(int(scale_mode == "std_only"),device=device,dtype=torch.long))
        self.G=torch.zeros(environment_task_ids.numel(),device=device)
        self.register_buffer('environment_task_ids',environment_task_ids)
        self.register_buffer('environment_task_rows',task_rows)
        self.register_buffer('normalizer_format_version',torch.tensor(1,device=device,dtype=torch.long))

    def _scale_reward(self, rewards, task_ids):
        rows = self._task_rows(task_ids)
        std = self.G_rms._std.gather(0, rows.unsqueeze(-1)).squeeze(-1)
        if self.scale_mode == "std_magnitude_floor":
            bound = self.G_r_max.gather(0, rows) / self.g_max
            std = torch.maximum(std, bound)
        return rewards / (std + self.epsilon)

    @torch.no_grad()
    def update_stats(self, rewards, dones, task_ids):
        if rewards.shape!=self.G.shape or dones.shape!=self.G.shape or task_ids.shape!=self.G.shape:
            raise ValueError('Reward updates require exactly one row per configured environment')
        if not torch.equal(task_ids,self.environment_task_ids):
            raise ValueError('Reward update task layout must match the configured environment rows')
        reward_trace=self.gamma*self.G+rewards
        self.G_rms.update(reward_trace[:,None],self.environment_task_rows)
        self.G_r_max.scatter_reduce_(0,self.environment_task_rows,reward_trace.abs(),
                                     reduce='amax',include_self=True)
        self.G.copy_(torch.where(dones.bool(),torch.zeros_like(reward_trace),reward_trace))

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        code = state_dict.get("reward_scale_mode_id")
        if code is None:
            if self.scale_mode != "std_magnitude_floor":
                raise ValueError("Cannot implicitly load a historical scale into std_only")
            state_dict = state_dict.copy()
            state_dict["reward_scale_mode_id"] = self.reward_scale_mode_id.clone()
        elif int(code) != int(self.reward_scale_mode_id):
            raise ValueError("Reward scaling mode differs from checkpoint")

        # The shared checkpoint loader uses strict=False. Reject incompatible
        # formats explicitly instead of silently mixing task and env histories.
        if 'normalizer_format_version' not in state_dict or 'environment_task_ids' not in state_dict:
            raise ValueError('Per-environment reward normalization requires its own checkpoint format; legacy statistics cannot be resumed implicitly')
        if int(state_dict['normalizer_format_version'])!=1:
            raise ValueError('Unsupported per-environment reward-normalizer checkpoint version')
        saved_ids=state_dict['environment_task_ids'].to(self.environment_task_ids.device)
        if not torch.equal(saved_ids,self.environment_task_ids):
            raise ValueError('Reward-normalizer checkpoint environment/task layout differs')
        saved_rows=state_dict.get('environment_task_rows')
        if saved_rows is None or not torch.equal(saved_rows.to(self.environment_task_rows.device),self.environment_task_rows):
            raise ValueError('Reward-normalizer checkpoint task-index mapping differs')
        return super().load_state_dict(state_dict,strict=True,**kwargs)

    @torch.no_grad()
    def reset_history(self):
        """Clear partial episodes after an external reset, retaining scale stats."""
        self.G.zero_()
