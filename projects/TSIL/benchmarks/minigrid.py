"""MiniGrid image observations with the original TSIL temporal/reward contract.

Task registrations and reset-boundary handling live in the local benchmark adapter.
The original partial 7x7 observations are the default; full views remain an
explicit option. The three public time fields precede the image, keeping replay
relabeling out of image pixels and task state.
"""

import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

from core.agents.agent import Agent
from core.agents.utils import MLP
from projects.TSIL.benchmarks.minigrid_policy import CategoricalPolicyMixin
from projects.TSIL.benchmarks.minigrid_policy import _cnn
from projects.TSIL.benchmarks.minigrid_adapter import (
    MiniGridVecEnv,
    MiniGridMissionEncoder,
)
from projects.TSIL.benchmarks.suites import get_tsil_minigrid_suite


# All registered TSIL MT7/20/50 maps fit this canvas. A fixed size also permits
# evaluating a subset of a checkpoint's tasks without changing its CNN shape.
FULL_GRID_SIZE = 25


def target_encoding(mission):
    """Six colors plus key/ball/box/door, supplied each step for MT15 missions."""
    words = set(mission.split())
    colors = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']
    objects = ['key', 'ball', 'box', 'door']
    return np.asarray([float(word in words) for word in colors + objects], dtype=np.float32)


class TSILMiniGridEnv(MiniGridVecEnv):
    """Finite episodes with timeout bootstrapping; deadlines never force resets."""

    def __init__(self, args, device="cpu", configuration_bank=None, configuration_split='train'):
        self.args = args
        suite = get_tsil_minigrid_suite(args.minigrid_suite)
        super().__init__(
            tasks=args.tasks, task_counts=args.task_counts, seed=args.seed,
            suite=suite.key, device=device, observation_mode="official_image",
            include_time_obs=False, seq_length=args.seq_length,
        )
        self.public_time_obs_names = ["time_ratio", "time_cur", "ddl"]
        self.ratio_range = None
        self.minigrid_observation = args.minigrid_observation
        if self.minigrid_observation not in {"full", "partial"}:
            self.close()
            raise ValueError("minigrid_observation must be full or partial")
        full = self.minigrid_observation == "full"
        self.image_shape = (3, FULL_GRID_SIZE, FULL_GRID_SIZE) if full else (3, 7, 7)
        self.image_dim = int(np.prod(self.image_shape))
        if full and any(max(env.unwrapped.width, env.unwrapped.height) > FULL_GRID_SIZE
                        for env in self._envs):
            self.close()
            raise ValueError("MiniGrid map exceeds the full-observation canvas; increase FULL_GRID_SIZE")
        self.mission_encoder = (MiniGridMissionEncoder()
                                if full or args.minigrid_suite not in {"mt7", "mt7_large", "mt10_ff", "mt15_ff", "mt15_selection", "mt15_refinement", "mt15_control", "mt15_fixed", "mt15_gentle", "mt15_repair_v1", "mt15_repair_v2", "mt15_repair_v3", "mt15_repair_v4"} else None)
        self.compact_target = args.minigrid_suite in {'mt15_ff', 'mt15_selection', 'mt15_refinement', 'mt15_control', 'mt15_fixed', 'mt15_gentle', 'mt15_repair_v1', 'mt15_repair_v2', 'mt15_repair_v3', 'mt15_repair_v4'} and not full
        self.aux_dim = 0 if self.mission_encoder is None else self.mission_encoder.output_dim + 4
        self.aux_dim += 10 if self.compact_target else 0
        # Full views add x/y, map width/height and the carried object's encoding.
        self.aux_dim += 7 if full else 0
        self.num_observations = 3 + self.image_dim + self.aux_dim + self.task_observation_embedding_dim
        self.num_states = self.num_observations
        from gymnasium.spaces import Box
        self.obs_space = self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.num_observations,), dtype=np.float32,
        )
        self.state_space = Box(
            low=-np.inf, high=np.inf, shape=(self.num_states,), dtype=np.float32,
        )
        self.episode_limits = torch.as_tensor(
            self._max_steps_by_env, dtype=torch.float32, device=self.device,
        )
        upper = self.episode_limits * args.time2end_init_frac
        self.obs_range = {"time2end": [-upper.clone(), upper.clone()]}
        self.real_time2end_init = upper.clone()
        self.reward_settings = {"r_success": float(args.successRewardScale)}
        self.cfg["r_epstime_scale"] = float(args.epstimeRewardScale[0])
        self.valid_env_steps = 0
        self.configuration_bank = None
        if configuration_bank is not None or getattr(args, 'minigrid_config_bank', ''):
            bank = configuration_bank if configuration_bank is not None else json.loads(
                Path(args.minigrid_config_bank).read_text())
            if bank['format_version'] != 1 or bank['suite'] != args.minigrid_suite:
                raise ValueError('Incompatible MiniGrid configuration bank')
            self.configuration_bank = json.loads(json.dumps(bank))
            self.configuration_bank['tasks'] = {str(t): self.configuration_bank['tasks'][str(t)] for t in args.tasks}
            self.configuration_records, self.configuration_families = [], []
            for task, count in zip(args.tasks, args.task_counts):
                task_bank = self.configuration_bank['tasks'][str(task)]
                if task_bank['env_id'] != suite.tasks[task].env_id or task_bank['horizon'] != suite.tasks[task].max_steps:
                    raise ValueError(f'Task {task} bank does not match the native task variant')
                records = task_bank[configuration_split]
                if not records:
                    raise ValueError(f'Task {task} has no {configuration_split} configurations')
                self.configuration_records.extend(records[i % len(records)] for i in range(count))
                self.configuration_families.extend([task_bank['family']] * count)
            self.configuration_split = configuration_split
            self._cue_rngs = [np.random.RandomState(bank['seed'] + 700000 + int(t) * 1000 + i)
                              for i, t in enumerate(self.task_indices.cpu().tolist())]
            self._cue_orders = [[] for _ in self._envs]

    def add_timeaware_obs(self, observations):
        return ["time_ratio", "time_cur", "ddl"] + list(observations)

    def _get_avg_time2end_upper(self):
        return float(self.obs_range["time2end"][1].mean().item())

    def _update_time2end_upper(self, env_ids, new_upper):
        upper = torch.minimum(new_upper.clamp(min=1.0), self.episode_limits[env_ids])
        self.obs_range["time2end"][0][env_ids] = -upper
        self.obs_range["time2end"][1][env_ids] = upper
        if self.configuration_bank is not None:
            # Duplicate workers share the configuration's bound, not a worker-specific label.
            for index, value in zip(env_ids.cpu().tolist(), upper.cpu().tolist()):
                record = self.configuration_records[index]
                record['target'] = min(record['target'], value)
            for index, record in enumerate(self.configuration_records):
                self.obs_range['time2end'][0][index] = -record['target']
                self.obs_range['time2end'][1][index] = record['target']

    def _reset_one(self, env_id):
        if self.configuration_bank is not None:
            self._initial_seeds[env_id] = self.configuration_records[env_id]['seed']
            self._has_seeded[env_id] = False
        observation = super()._reset_one(env_id)
        if self.configuration_bank is not None:
            from projects.TSIL.benchmarks.configurations import describe, fingerprint, vary_memory_cue
            family = self.configuration_families[env_id]
            env = self._envs[env_id].unwrapped
            if family == 'Memory':
                if not self._cue_orders[env_id]:
                    self._cue_orders[env_id] = self._cue_rngs[env_id].permutation(4).tolist()
                vary_memory_cue(env, self._cue_orders[env_id].pop())
                observation = env.gen_obs()
            task_bank = self.configuration_bank['tasks'][str(int(self.task_indices[env_id]))]
            if fingerprint(task_bank['env_id'], describe(env, family)) != self.configuration_records[env_id]['id']:
                raise RuntimeError('Configuration reset changed the saved identity; check MiniGrid version')
        # As in MetaWorld, learned bounds become the deadline at the next reset.
        self.real_time2end_init[env_id] = self.obs_range["time2end"][1][env_id]
        return observation

    def _encode_batch(self, observations):
        if self.minigrid_observation == "full":
            encoded = self._encode_full_batch(observations)
        else:
            encoded = super()._encode_batch(observations)
        if self.compact_target:
            # Only target-dependent tasks need an instruction; other tasks get zeros.
            aux = np.stack([target_encoding(obs['mission']) if self.suite.tasks[int(task)].family
                            in {'Fetch', 'GoToDoor', 'GoToObject'} else np.zeros(10, dtype=np.float32)
                            for obs, task in zip(observations, self.task_indices.cpu().tolist())])
            task_dim = self.task_observation_embedding_dim
            encoded = torch.cat((encoded[:, :-task_dim], torch.as_tensor(aux, device=self.device),
                                 encoded[:, -task_dim:]), dim=1)
        if self.minigrid_observation == "partial" and self.mission_encoder is not None:
            aux = np.stack([
                np.concatenate((np.eye(4, dtype=np.float32)[int(obs["direction"])],
                                self.mission_encoder.encode(obs["mission"])))
                for obs in observations
            ])
            task_dim = self.task_observation_embedding_dim
            encoded = torch.cat((encoded[:, :-task_dim], torch.as_tensor(
                aux, device=self.device), encoded[:, -task_dim:]), dim=1)
        elapsed = torch.as_tensor(self._episode_steps, dtype=torch.float32, device=self.device)
        temporal = torch.stack((torch.ones_like(elapsed), elapsed, self.real_time2end_init), dim=1)
        if self.args.fix_priv or not self.args.time2end:
            temporal.zero_()
        return torch.cat((temporal, encoded), dim=1)

    def _encode_full_batch(self, observations):
        rows = []
        task_ids = np.repeat(self.task_idx, self.task_env_count)
        for wrapped_env, observation, task_id in zip(self._envs, observations, task_ids):
            env = wrapped_env.unwrapped
            image = np.zeros(self.image_shape, dtype=np.float32)
            # Keep the tile under the agent: pose is supplied separately.
            image[:, :env.width, :env.height] = env.grid.encode().transpose(2, 0, 1)
            pose_and_extent = np.asarray(
                [*env.agent_pos, env.width, env.height], dtype=np.float32,
            ) / FULL_GRID_SIZE
            carrying = np.asarray(env.carrying.encode() if env.carrying is not None
                                  else (0, 0, 0), dtype=np.float32)
            task = np.zeros(self.task_observation_embedding_dim, dtype=np.float32)
            task[task_id] = 1
            rows.append(np.concatenate((
                image.reshape(-1), pose_and_extent,
                np.eye(4, dtype=np.float32)[int(observation["direction"])], carrying,
                self.mission_encoder.encode(observation["mission"]), task,
            )))
        return torch.as_tensor(np.stack(rows), dtype=torch.float32, device=self.device)

    def step(self, actions):
        self.valid_env_steps += int((~self._needs_reset).sum())
        observations, native_reward, done, info = super().step(actions)
        success = info["success"].float()
        if self.configuration_bank is not None and self.configuration_split == 'train':
            for index in info['success'].nonzero(as_tuple=True)[0].cpu().tolist():
                record = self.configuration_records[index]
                duration = float(info['eps_time'][index])
                record['success_count'] += 1
                previous = record['best_success_steps']
                record['best_success_steps'] = duration if previous is None else min(previous, duration)
        base_reward = self.args.successRewardScale * success
        if self.args.minigrid_reward == "native_bonus":
            native_scale = torch.where(info["success"], 1.0, self.args.reward_scale)
            base_reward = base_reward + native_scale * native_reward
        elif self.args.minigrid_reward == "native_no_success":
            # MT15 native success reward is b(T); replace only that component.
            # Preserve every non-success native reward and its historical scaling.
            base_reward = base_reward + self.args.reward_scale * torch.where(
                info["success"], torch.zeros_like(native_reward), native_reward)
        bonus = success * float(self.cfg["r_epstime_scale"])
        efficiency_bonus = getattr(self.args, 'temporal_efficiency_bonus', None)
        if efficiency_bonus is None:
            efficiency_bonus = self.args.update_ddl
        if efficiency_bonus:
            bonus = bonus * (1.0 + (self.real_time2end_init / info["eps_time"].clamp(min=1)).clamp(0, 1))
        info.update(
            native_reward=native_reward,
            reward_without_timeaware=base_reward,
            timeaware_reward_bonus=bonus,
        )
        # Native return remains an evaluation metric; success mode adds none of it.
        return observations, base_reward + bonus, done, info


class MiniGridCNN(nn.Module):
    """Existing MiniGrid CNN backbone with the configured TSIL MLP head."""

    def __init__(self, envs, args, output_dim):
        super().__init__()
        self.image_shape = envs.image_shape
        self.image_dim = envs.image_dim
        self.drop_constant_inputs = bool(args.tsil_drop_constant_inputs)
        if self.drop_constant_inputs and len(envs.unique_task_ids) != 1:
            raise ValueError("Removing task inputs requires a single-task MiniGrid policy")
        self.task_dim = envs.task_observation_embedding_dim
        self.cnn = _cnn(last_activation=True)
        with torch.no_grad():
            feature_dim = self.cnn(torch.zeros(1, *self.image_shape)).shape[-1]
        self.head = MLP(
            input_size=feature_dim + envs.aux_dim + (2 if self.drop_constant_inputs else 3 + self.task_dim),
            hidden_size=args.hidden_size, output_size=output_dim,
            num_hidden_layers=len(args.hidden_size), activation=args.activation,
            use_layernorm=args.use_layernorm, init_std=1.0,
        )

    def forward(self, observation):
        image = observation[..., 3:3 + self.image_dim].reshape(-1, *self.image_shape)
        features = self.cnn(image.float())
        if self.drop_constant_inputs:
            # Raw metadata still serves normalizers, action masks and replay.
            # Neither constant reaches a learned actor/critic layer.
            auxiliary = observation[..., 3 + self.image_dim:-self.task_dim]
            return self.head(torch.cat((features, observation[..., 1:3], auxiliary), dim=-1))
        return self.head(torch.cat((features, observation[..., :3],
                                    observation[..., 3 + self.image_dim:]), dim=-1))


class MiniGridAgent(CategoricalPolicyMixin, Agent):
    def init_policy_net(self, envs, args):
        self.critic = MiniGridCNN(envs, args, self.critic_dim)
        self.actor = MiniGridCNN(envs, args, envs.num_actions)
        self.temporal_critic = None

    def save_checkpoint(self, folder_path, ckpt_name='eps', suffix='', ckpt_path=None,
                        reward_normalizer=None, verbose=False):
        super().save_checkpoint(folder_path, ckpt_name, suffix, ckpt_path, reward_normalizer, verbose)
        env = self.envs
        if env.configuration_bank is not None:
            from projects.TSIL.benchmarks.configurations import write_json
            path = Path(ckpt_path) if ckpt_path is not None else Path(folder_path) / f'{ckpt_name}_{suffix}'
            write_json(str(path) + '.minigrid.json', dict(
                checkpoint=path.name, time_feature_scale=self.time_feature_scale,
                configuration_bank=env.configuration_bank), compact=True)
