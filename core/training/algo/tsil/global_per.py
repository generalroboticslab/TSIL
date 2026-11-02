"""Opt-in original-style SIL: pooled FIFO transitions and cached PER priorities.

The common PPO/IH learner and five-epoch replay schedule remain matched. This
is not a full reproduction of the original optimizer/schedule. Global
replacement PER, capped gaps and sampling correction follow the released SIL
implementation. All completed episodes are admitted so signed- or
negative-reward environments are not filtered by reward sign. No
configuration/trajectory quotas.
"""
import json
from pathlib import Path

import torch


class GlobalTransitionPER:
    def __init__(self, args, device):
        self.args, self.device = args, device
        self.capacity = int(args.sil_global_per_capacity)
        self.size = self.cursor = self.episode_id = 0
        self.max_priority = 1.
        self.data = {}
        self.priorities = torch.zeros(self.capacity, dtype=torch.float64, device=device)
        self.rng = torch.Generator(device=device).manual_seed(int(args.seed) + 17041)

    @torch.no_grad()
    def add_episode(self, record):
        reward = record['reward'].to(device=self.device, dtype=torch.float32)
        length = len(reward)
        if not length:
            return False
        if any(len(record[key]) != length for key in ('obs', 'state', 'action')):
            raise ValueError('Global PER requires a complete aligned episode')
        # Compute the full episode's observed returns BEFORE FIFO tail eviction.
        discount = float(self.args.gamma) ** torch.arange(length, device=self.device, dtype=torch.float64)
        weighted = reward.double() * discount
        returns = weighted.flip(0).cumsum(0).flip(0) / discount
        summary = record.get('summary', {})
        success = bool(summary.get('success', False))
        timeout = bool(record.get('terminal_timeout', False)) and not success
        self.episode_id += 1
        values = dict(
            obs=record['obs'].to(self.device, torch.float16),
            state=record['state'].to(self.device, torch.float16),
            action=record['action'].to(self.device, torch.float16),
            raw_returns=returns.float(),
            terminal_state=record['terminal_next_state'].to(self.device, torch.float16).expand(length, -1),
            bootstrap_discount=(float(self.args.gamma) ** torch.arange(length, 0, -1, device=self.device)).float(),
            timeout=torch.full((length,), timeout, dtype=torch.bool, device=self.device),
            task_ids=torch.full((length,), int(record['task_id']), dtype=torch.long, device=self.device),
            env_ids=torch.full((length,), int(record['env_id']), dtype=torch.long, device=self.device),
            slot_ids=torch.full((length,), self.episode_id, dtype=torch.long, device=self.device),
            time_idx=torch.arange(length, device=self.device),
            episode_times=torch.full((length,), float(summary.get('eps_time', length)), device=self.device),
            success=torch.full((length,), success, dtype=torch.bool, device=self.device),
        )
        if not self.data:
            self.data = {key: torch.empty((self.capacity,) + tuple(value.shape[1:]),
                device=self.device, dtype=value.dtype) for key, value in values.items()}
        first = max(0, length - self.capacity)
        indices = (torch.arange(first, length, device=self.device) + self.cursor) % self.capacity
        for key, value in values.items():
            self.data[key][indices] = value[first:]
        self.priorities[indices] = self.max_priority
        self.cursor = (self.cursor + length) % self.capacity
        self.size = min(self.capacity, self.size + length)
        return True

    @torch.no_grad()
    def update_priorities(self, indices, advantages):
        priorities = advantages.double().clamp(min=1e-6, max=1.)
        # Repeated draws of the same state share its current gap. Deduplicate
        # explicitly rather than relying on undefined repeated-index writes.
        unique, inverse = torch.unique(indices, return_inverse=True)
        values = torch.zeros_like(unique, dtype=torch.float64)
        values.scatter_reduce_(0, inverse, priorities, reduce='amax', include_self=False)
        self.priorities[unique] = values
        if values.numel():
            self.max_priority = max(self.max_priority, float(values.max()))

    def context(self, memory, reward_normalizer, schedule_context, stats):
        counts = torch.bincount(self.data['env_ids'][:self.size], minlength=self.args.num_envs)
        ntraj = int(self.data['slot_ids'][:self.size].unique().numel())
        stats.update(replay_has_demo=int(self.size > 100), replay_dataset_steps=self.size,
                     replay_dataset_trajectories=ntraj, replay_unique_trajectories=ntraj)
        return dict(demos=[], dataset=None, has_tsil_memory=True, tsil_memory=memory,
                    global_per=self, source_counts=counts, reward_normalizer=reward_normalizer,
                    schedule_context=schedule_context, sample_batch_fn=memory.sample_sil_batch,
                    positive_filter_enabled=False, stats=stats)

    @torch.no_grad()
    def sample(self, memory, context):
        if self.size <= 100:
            return None
        # Match TSIL's maximum dataset budget and five-epoch reuse, without
        # imposing per-configuration allocation or trajectory ranking.
        budget = self.args.num_envs * self.args.sil_replay_trajectories_per_env * self.args.sil_replay_transitions_per_trajectory
        count = min(int(budget), self.size)
        probabilities = self.priorities[:self.size].pow(float(self.args.sil_per_alpha))
        probabilities /= probabilities.sum()
        old_rng = self.rng.get_state() if not context.get('_sil_building_replay_batch', False) else None
        indices = torch.multinomial(probabilities, count, replacement=True, generator=self.rng)
        if old_rng is not None:
            self.rng.set_state(old_rng)
        # (N*p)^-beta / max_buffer (N*p)^-beta == (p_min/p)^beta.
        importance = (probabilities.min() / probabilities[indices]).pow(float(self.args.sil_per_beta)).float()
        batch = {key: value[indices].clone() for key, value in self.data.items()}
        task_ids = batch['task_ids']
        returns = batch.pop('raw_returns')
        if self.args.norm_rew:
            # R2-S reward normalization is task-wise linear scaling, so scaling
            # cached raw discounted returns equals discounting scaled rewards.
            returns = context['reward_normalizer'](returns, task_ids).float()
        observed = returns.clone()
        for key in ('obs', 'state', 'terminal_state'):
            batch[key] = batch[key].to(memory.dtype)
            memory._relabel_public_deadline(batch[key], torch.zeros_like(returns))
        timeout = batch.pop('timeout') & bool(self.args.value_bootstrap)
        bootstrap = torch.zeros_like(returns)
        if timeout.any():
            agent = context['_sil_value_agent']
            bootstrap[timeout] = agent.get_value(batch['terminal_state'][timeout], denorm=True, update=False)[0].view(-1)
        returns += batch.pop('bootstrap_discount') * bootstrap
        batch.pop('terminal_state')
        success = batch.pop('success')
        batch.update(returns=returns, observed_returns=observed, sil_weights=torch.ones_like(returns),
            sil_importance_weights=importance, sil_sample_probabilities=probabilities[indices].float(),
            per_indices=indices, sil_success_mask=success, sil_fallback_mask=~success,
            sil_return_fill_mask=torch.ones_like(success), sil_bootstrap_values=bootstrap,
            sil_bootstrap_mask=timeout, trajectory_count=int(batch['slot_ids'].unique().numel()),
            aligned_dataset_steps=float(count), aligned_replay_epochs=float(self.args.sil_replay_epochs))
        batch['action'] = batch['action'].to(memory.dtype)
        if context.get('_sil_building_replay_batch', False) and getattr(self.args, 'instance_dir', None):
            stats = dict(global_step=context.get('global_step'), buffer_size=self.size,
                selected_count=count, unique_transitions=int(indices.unique().numel()),
                sampled_envs=int(batch['env_ids'].unique().numel()),
                importance_min=float(importance.min()), importance_mean=float(importance.mean()),
                alpha=float(self.args.sil_per_alpha), beta=float(self.args.sil_per_beta))
            with (Path(self.args.instance_dir) / 'global_per.jsonl').open('a') as handle:
                handle.write(json.dumps(stats) + '\n')
        return batch
