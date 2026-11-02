"""Vector adapter and catalogs for project-defined MiniGrid multitask suites.

Core-MT20 selects the smallest practical official registration from each
reward-bearing core family. Variant-MT50 is a strict superset that adds
controlled variants for affinity analysis. Playground is omitted because it
has no task reward or success state. MiniGrid itself does not publish either
suite or official difficulty labels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


MINIGRID_MAX_ACTIONS = 7
MINIGRID_VIEW_SHAPE = (7, 7, 3)


@dataclass(frozen=True)
class MiniGridTaskSpec:
    task_id: int
    family: str
    env_id: str
    max_steps: int
    num_actions: int = MINIGRID_MAX_ACTIONS


@dataclass(frozen=True)
class MiniGridSuiteSpec:
    key: str
    display_name: str
    tasks: tuple[MiniGridTaskSpec, ...]

    @property
    def task_count(self) -> int:
        return len(self.tasks)


def _make_task_specs(rows, start=0):
    return tuple(
        MiniGridTaskSpec(start + offset, *row)
        for offset, row in enumerate(rows)
    )


# Exact MT7 task set used by the official ICLR 2024 MooRE MiniGrid experiment.
MINIGRID_MT7_CNN_TASKS = _make_task_specs((
    ("DoorKey", "MiniGrid-DoorKey-6x6-v0", 360),
    ("DistShift", "MiniGrid-DistShift1-v0", 252),
    ("RedBlueDoors", "MiniGrid-RedBlueDoors-6x6-v0", 720),
    ("LavaGap", "MiniGrid-LavaGapS7-v0", 196),
    ("Memory", "MiniGrid-MemoryS11-v0", 605),
    ("Crossing", "MiniGrid-SimpleCrossingS9N2-v0", 324),
    ("MultiRoom", "MiniGrid-MultiRoom-N2-S4-v0", 40),
))


# One smallest useful registration from each of the 20 reward-bearing families.
MINIGRID_MT20_TASKS = _make_task_specs((
    ("BlockedUnlockPickup", "MiniGrid-BlockedUnlockPickup-v0", 576),
    ("Crossing", "MiniGrid-SimpleCrossingS9N1-v0", 324),
    ("DistShift", "MiniGrid-DistShift1-v0", 252),
    ("DoorKey", "MiniGrid-DoorKey-5x5-v0", 250),
    ("DynamicObstacles", "MiniGrid-Dynamic-Obstacles-5x5-v0", 100, 3),
    ("Empty", "MiniGrid-Empty-Random-5x5-v0", 100),
    ("Fetch", "MiniGrid-Fetch-5x5-N2-v0", 125),
    ("FourRooms", "MiniGrid-FourRooms-v0", 100),
    ("GoToDoor", "MiniGrid-GoToDoor-5x5-v0", 100),
    ("GoToObject", "MiniGrid-GoToObject-6x6-N2-v0", 180),
    ("KeyCorridor", "MiniGrid-KeyCorridorS3R1-v0", 270),
    ("LavaGap", "MiniGrid-LavaGapS5-v0", 100),
    ("LockedRoom", "MiniGrid-LockedRoom-v0", 190),
    ("Memory", "MiniGrid-MemoryS7-v0", 245),
    ("MultiRoom", "MiniGrid-MultiRoom-N2-S4-v0", 40),
    ("ObstructedMaze", "MiniGrid-ObstructedMaze-1Dl-v0", 288),
    ("PutNear", "MiniGrid-PutNear-6x6-N2-v0", 30),
    ("RedBlueDoors", "MiniGrid-RedBlueDoors-6x6-v0", 720),
    ("Unlock", "MiniGrid-Unlock-v0", 288),
    ("UnlockPickup", "MiniGrid-UnlockPickup-v0", 288),
))

# MiniGrid has no official difficulty labels. These 30 variants use compact
# layouts and max_steps <= 576 as a reproducible simple/medium proxy. MT20's
# smallest RedBlueDoors registration is the sole 720-step suite exception.
_MINIGRID_MT50_VARIANTS = _make_task_specs((
    ("Crossing", "MiniGrid-SimpleCrossingS9N2-v0", 324),
    ("Crossing", "MiniGrid-SimpleCrossingS9N3-v0", 324),
    ("Crossing", "MiniGrid-LavaCrossingS9N1-v0", 324),
    ("Crossing", "MiniGrid-LavaCrossingS9N2-v0", 324),
    ("Crossing", "MiniGrid-LavaCrossingS9N3-v0", 324),
    ("DistShift", "MiniGrid-DistShift2-v0", 252),
    ("DoorKey", "MiniGrid-DoorKey-6x6-v0", 360),
    ("DynamicObstacles", "MiniGrid-Dynamic-Obstacles-6x6-v0", 144, 3),
    ("DynamicObstacles", "MiniGrid-Dynamic-Obstacles-Random-5x5-v0", 100, 3),
    ("DynamicObstacles", "MiniGrid-Dynamic-Obstacles-Random-6x6-v0", 144, 3),
    ("Empty", "MiniGrid-Empty-5x5-v0", 100),
    ("Empty", "MiniGrid-Empty-6x6-v0", 144),
    ("Empty", "MiniGrid-Empty-8x8-v0", 256),
    ("Empty", "MiniGrid-Empty-Random-6x6-v0", 144),
    ("Fetch", "MiniGrid-Fetch-6x6-N2-v0", 180),
    ("Fetch", "MiniGrid-Fetch-8x8-N3-v0", 320),
    ("GoToDoor", "MiniGrid-GoToDoor-6x6-v0", 144),
    ("GoToDoor", "MiniGrid-GoToDoor-8x8-v0", 256),
    ("GoToObject", "MiniGrid-GoToObject-8x8-N2-v0", 320),
    ("KeyCorridor", "MiniGrid-KeyCorridorS3R2-v0", 270),
    ("KeyCorridor", "MiniGrid-KeyCorridorS3R3-v0", 270),
    ("KeyCorridor", "MiniGrid-KeyCorridorS4R3-v0", 480),
    ("LavaGap", "MiniGrid-LavaGapS6-v0", 144),
    ("LavaGap", "MiniGrid-LavaGapS7-v0", 196),
    ("Memory", "MiniGrid-MemoryS9-v0", 405),
    ("MultiRoom", "MiniGrid-MultiRoom-N4-S5-v0", 120),
    ("MultiRoom", "MiniGrid-MultiRoom-N6-v0", 120),
    ("ObstructedMaze", "MiniGrid-ObstructedMaze-1Dlh-v0", 288),
    ("ObstructedMaze", "MiniGrid-ObstructedMaze-1Dlhb-v0", 288),
    ("PutNear", "MiniGrid-PutNear-8x8-N3-v0", 40),
), start=len(MINIGRID_MT20_TASKS))

MINIGRID_MT50_TASKS = MINIGRID_MT20_TASKS + _MINIGRID_MT50_VARIANTS
MINIGRID_MT7_CNN = MiniGridSuiteSpec(
    "mt7_cnn", "MiniGrid-MooRE-MT7-CNN", MINIGRID_MT7_CNN_TASKS
)
MINIGRID_MT20 = MiniGridSuiteSpec("mt20", "MiniGrid-Core-MT20", MINIGRID_MT20_TASKS)
MINIGRID_MT50 = MiniGridSuiteSpec("mt50", "MiniGrid-Variant-MT50", MINIGRID_MT50_TASKS)
MINIGRID_SUITES = {
    suite.key: suite
    for suite in (MINIGRID_MT7_CNN, MINIGRID_MT20, MINIGRID_MT50)
}


def get_minigrid_suite(key: str) -> MiniGridSuiteSpec:
    normalized = str(key).lower()
    if normalized not in MINIGRID_SUITES:
        raise ValueError(
            f"Unknown MiniGrid suite {key!r}; choose from {sorted(MINIGRID_SUITES)}"
        )
    return MINIGRID_SUITES[normalized]

# Complete vocabulary for the mission templates used by the selected
# environments in MiniGrid 2.3.1. Unknown words have an explicit stable slot.
_MISSION_VOCABULARY = (
    "<unk>",
    "a",
    "and",
    "at",
    "avoid",
    "ball",
    "blue",
    "box",
    "door",
    "end",
    "fetch",
    "find",
    "from",
    "get",
    "go",
    "goal",
    "green",
    "grey",
    "hallway",
    "key",
    "lava",
    "matching",
    "must",
    "near",
    "object",
    "of",
    "open",
    "opening",
    "pick",
    "purple",
    "put",
    "reach",
    "red",
    "room",
    "rooms",
    "square",
    "the",
    "then",
    "to",
    "traverse",
    "unlock",
    "up",
    "use",
    "yellow",
    "you",
)


class MiniGridMissionEncoder:
    """Losslessly encode the selected MiniGrid mission templates by position."""

    vocabulary = _MISSION_VOCABULARY

    def __init__(self, max_tokens: int = 24):
        if int(max_tokens) <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        self.max_tokens = int(max_tokens)
        self._token_to_index = {
            token: index for index, token in enumerate(self.vocabulary)
        }
        self.output_dim = self.max_tokens * len(self.vocabulary)

    @staticmethod
    def tokenize(mission: str) -> list[str]:
        return re.findall(r"[a-z]+", str(mission).lower())

    def encode(self, mission: str) -> np.ndarray:
        tokens = self.tokenize(mission)
        if len(tokens) > self.max_tokens:
            raise ValueError(
                f"MiniGrid mission has {len(tokens)} tokens, exceeding "
                f"minigrid_mission_max_tokens={self.max_tokens}: {mission!r}"
            )

        unknown_tokens = sorted(
            {token for token in tokens if token not in self._token_to_index}
        )
        if unknown_tokens:
            raise ValueError(
                "MiniGrid mission contains tokens outside the frozen suite "
                f"vocabulary: {unknown_tokens}"
            )

        encoded = np.zeros(
            (self.max_tokens, len(self.vocabulary)), dtype=np.float32
        )
        for position, token in enumerate(tokens):
            token_index = self._token_to_index[token]
            encoded[position, token_index] = 1.0
        return encoded.reshape(-1)


class _ObservationEncoder:
    """Flatten MiniGrid's symbolic view, mission, direction, and task ID."""

    _object_eye = np.eye(11, dtype=np.float32)
    _color_eye = np.eye(6, dtype=np.float32)
    _state_eye = np.eye(3, dtype=np.float32)
    _direction_eye = np.eye(4, dtype=np.float32)

    def __init__(
        self,
        mission_max_tokens: int,
        task_count: int,
        suite_name: str,
    ):
        self.mission = MiniGridMissionEncoder(mission_max_tokens)
        self.task_count = int(task_count)
        self.suite_name = str(suite_name)
        self.image_dim = 7 * 7 * (11 + 6 + 3)
        self.output_dim = (
            self.image_dim
            + self._direction_eye.shape[0]
            + self.mission.output_dim
            + self.task_count
        )

    def encode(self, observation: dict, task_id: int) -> np.ndarray:
        image = np.asarray(observation["image"])
        if image.shape != MINIGRID_VIEW_SHAPE:
            raise ValueError(
                f"{self.suite_name} expects MiniGrid's 7x7 symbolic view, "
                f"got {image.shape}"
            )

        image_features = np.concatenate(
            (
                self._object_eye[image[..., 0]],
                self._color_eye[image[..., 1]],
                self._state_eye[image[..., 2]],
            ),
            axis=-1,
        ).reshape(-1)
        direction = int(observation["direction"])
        if not 0 <= direction < 4:
            raise ValueError(f"Invalid MiniGrid direction: {direction}")
        task_embedding = np.zeros(self.task_count, dtype=np.float32)
        task_embedding[int(task_id)] = 1.0
        return np.concatenate(
            (
                image_features,
                self._direction_eye[direction],
                self.mission.encode(observation["mission"]),
                task_embedding,
            )
        ).astype(np.float32, copy=False)


class _OfficialImageObservationEncoder:
    """Preserve the official compact 3x7x7 image and append only task ID."""

    def __init__(self, task_count: int, suite_name: str):
        self.task_count = int(task_count)
        self.suite_name = str(suite_name)
        self.output_dim = int(np.prod(MINIGRID_VIEW_SHAPE)) + self.task_count

    def encode(self, observation: dict, task_id: int) -> np.ndarray:
        image = np.asarray(observation["image"])
        if image.shape != MINIGRID_VIEW_SHAPE:
            raise ValueError(
                f"{self.suite_name} expects MiniGrid's 7x7 symbolic view, "
                f"got {image.shape}"
            )
        task_embedding = np.zeros(self.task_count, dtype=np.float32)
        task_embedding[int(task_id)] = 1.0
        return np.concatenate(
            (image.transpose(2, 0, 1).reshape(-1), task_embedding)
        ).astype(np.float32, copy=False)


def _load_minigrid(suite: MiniGridSuiteSpec):
    try:
        import gymnasium as gym
        import minigrid
    except ImportError as exc:
        raise ImportError(
            "MiniGrid multitask suites are optional. Install them in the active Conda "
            "environment with: pip install -r "
            "requirements.txt"
        ) from exc

    # MiniGrid 2.3.1 does not auto-register with newer Gymnasium releases.
    sentinel = suite.tasks[0].env_id
    if sentinel not in gym.registry:
        minigrid.register_minigrid_envs()

    missing = [
        spec.env_id
        for spec in suite.tasks
        if spec.env_id not in gym.registry
    ]
    if missing:
        raise RuntimeError(
            "The installed MiniGrid version does not provide the frozen "
            f"{suite.display_name} registry IDs: {missing}. "
            "Install minigrid==2.3.1."
        )
    return gym


class MiniGridVecEnv:
    """Synchronous CPU vector environment matching the shared trainer API."""

    def __init__(
        self,
        tasks: Sequence[int],
        task_counts: Sequence[int],
        seed: int,
        suite: str = "mt20",
        device: torch.device | str = "cpu",
        mission_max_tokens: int = 24,
        seq_length: int = 1,
        include_time_obs: bool = False,
        observation_mode: str = "full_symbolic",
    ):
        self.suite = get_minigrid_suite(suite)
        task_by_id = {spec.task_id: spec for spec in self.suite.tasks}
        task_ids = [int(task_id) for task_id in tasks]
        counts = [int(count) for count in task_counts]
        if len(task_ids) != len(counts):
            raise ValueError("tasks and task_counts must have the same length")
        if not task_ids or any(task_id not in task_by_id for task_id in task_ids):
            raise ValueError(
                f"{self.suite.display_name} task IDs must be in "
                f"[0, {self.suite.task_count - 1}], got {task_ids}"
            )
        if len(set(task_ids)) != len(task_ids):
            raise ValueError(
                f"{self.suite.display_name} tasks must be unique, got {task_ids}"
            )
        if any(count <= 0 for count in counts):
            raise ValueError(
                f"{self.suite.display_name} task counts must be positive, got {counts}"
            )

        gym = _load_minigrid(self.suite)
        self.device = torch.device(device)
        self.suite_name = self.suite.display_name
        self.task_idx = task_ids
        self.task_env_count = counts
        self.num_envs = sum(counts)
        self.num_actions = MINIGRID_MAX_ACTIONS
        self.seq_length = int(seq_length)
        self.include_time_obs = bool(include_time_obs)
        self.observation_mode = str(observation_mode).lower()
        self.ctrl_dt = self.dt = 1.0
        self.cfg = {"r_epstime_scale": 0.0, "r_scene_vel_scale": 0.0}
        self.reward_settings = {"r_success": 1.0}
        self.task_observation_embedding_dim = self.suite.task_count
        self.task_embedding_id_mode = "absolute"
        if self.observation_mode == "full_symbolic":
            self._encoder = _ObservationEncoder(
                mission_max_tokens,
                self.suite.task_count,
                self.suite.display_name,
            )
        elif self.observation_mode == "official_image":
            self._encoder = _OfficialImageObservationEncoder(
                self.suite.task_count,
                self.suite.display_name,
            )
        else:
            raise ValueError(
                "observation_mode must be full_symbolic or official_image, "
                f"got {observation_mode!r}"
            )
        self.num_observations = self._encoder.output_dim + int(self.include_time_obs)
        self.num_states = self.num_observations

        self._envs = []
        env_task_ids = []
        max_steps = []
        action_counts = np.full(
            self.suite.task_count, MINIGRID_MAX_ACTIONS, dtype=np.int64
        )
        selected_specs = [task_by_id[task_id] for task_id in task_ids]
        self.task_metadata = [
            {
                "task_id": spec.task_id,
                "family": spec.family,
                "env_id": spec.env_id,
                "max_steps": spec.max_steps,
                "num_actions": spec.num_actions,
            }
            for spec in selected_specs
        ]
        for task_id, count in zip(task_ids, counts):
            spec = task_by_id[task_id]
            for _ in range(count):
                env = gym.make(spec.env_id)
                declared_actions = int(env.action_space.n)
                if declared_actions != spec.num_actions:
                    env.close()
                    raise RuntimeError(
                        f"{spec.env_id} exposes {declared_actions} actions; "
                        f"{self.suite.display_name} expects {spec.num_actions}."
                    )
                declared_max_steps = int(env.unwrapped.max_steps)
                if declared_max_steps != spec.max_steps:
                    env.close()
                    raise RuntimeError(
                        f"{spec.env_id} exposes max_steps={declared_max_steps}; "
                        f"{self.suite.display_name} expects the official budget "
                        f"{spec.max_steps}."
                    )
                # Keep the official task-completable episode limit separate from
                # PPO's fixed-length rollout horizon.
                env.unwrapped.max_steps = spec.max_steps
                self._envs.append(env)
                env_task_ids.append(task_id)
                max_steps.append(spec.max_steps)
            action_counts[task_id] = spec.num_actions

        self.task_indices = torch.tensor(
            env_task_ids, dtype=torch.long, device=self.device
        )
        self.extras = {"task_indices": self.task_indices}
        self.unique_task_ids = torch.tensor(
            sorted(task_ids), dtype=torch.long, device=self.device
        )
        self._max_steps_by_env = np.asarray(max_steps, dtype=np.int64)
        self.max_eps_time = float(self._max_steps_by_env.max())
        self._episode_steps = np.zeros(self.num_envs, dtype=np.int64)
        self._needs_reset = np.zeros(self.num_envs, dtype=np.bool_)
        self._initial_seeds = [int(seed) + env_id for env_id in range(self.num_envs)]
        self._has_seeded = np.zeros(self.num_envs, dtype=np.bool_)

        action_mask = np.zeros(
            (self.suite.task_count, MINIGRID_MAX_ACTIONS), dtype=np.bool_
        )
        for task_id in range(self.suite.task_count):
            action_mask[task_id, : action_counts[task_id]] = True
        self._action_mask = torch.tensor(
            action_mask, dtype=torch.bool, device=self.device
        )

        observation_high = 10.0 if self.observation_mode == "official_image" else 1.0
        self.obs_space = self.observation_space = gym.spaces.Box(
            low=0.0,
            high=observation_high,
            shape=(self.num_observations,),
            dtype=np.float32,
        )
        self.state_space = gym.spaces.Box(
            low=0.0,
            high=observation_high,
            shape=(self.num_states,),
            dtype=np.float32,
        )
        # The shared storage only uses this shape. Actual validation is handled
        # by step(), because the underlying environments are Discrete.
        self.act_space = self.action_space = gym.spaces.Box(
            low=0.0,
            high=float(MINIGRID_MAX_ACTIONS - 1),
            shape=(1,),
            dtype=np.float32,
        )

    def add_timeaware_obs(self, observations):
        prefix = ["time_elapsed_fraction"] if self.include_time_obs else []
        return prefix + list(observations)

    def _get_avg_time2end_upper(self) -> float:
        return float(self._max_steps_by_env.mean())

    def action_mask_from_task_embedding(
        self, task_embedding: torch.Tensor
    ) -> torch.Tensor:
        task_ids = task_embedding.argmax(dim=-1).long()
        return self._action_mask.to(task_embedding.device)[task_ids]

    def action_mask_for_task_ids(self, task_ids: torch.Tensor) -> torch.Tensor:
        return self._action_mask.to(task_ids.device)[task_ids.long()]

    def reset_all(self):
        """Compatibility no-op; reset() performs the synchronized reset."""

    def _reset_one(self, env_id: int):
        seed = self._initial_seeds[env_id] if not self._has_seeded[env_id] else None
        observation, _ = self._envs[env_id].reset(seed=seed)
        self._has_seeded[env_id] = True
        self._episode_steps[env_id] = 0
        self._needs_reset[env_id] = False
        return observation

    def _encode_batch(self, observations) -> torch.Tensor:
        encoded = np.stack(
            [
                self._encoder.encode(observation, int(task_id))
                for observation, task_id in zip(
                    observations, self.task_indices.detach().cpu().tolist()
                )
            ],
            axis=0,
        )
        if self.include_time_obs:
            elapsed = (
                self._episode_steps / self._max_steps_by_env
            ).astype(np.float32)[:, None]
            task_dim = self.task_observation_embedding_dim
            encoded = np.concatenate(
                (elapsed, encoded[:, :-task_dim], encoded[:, -task_dim:]),
                axis=1,
            )
        return torch.as_tensor(encoded, dtype=torch.float32, device=self.device)

    def reset(self):
        observations = [self._reset_one(env_id) for env_id in range(self.num_envs)]
        encoded = self._encode_batch(observations)
        return {"obs": encoded, "states": encoded.clone()}

    def _discrete_actions(self, actions) -> np.ndarray:
        action_tensor = torch.as_tensor(actions).detach()
        if action_tensor.ndim == 2 and action_tensor.shape[1] == self.num_actions:
            # Supports the shared random-policy rollout path, which emits
            # per-action random scores rather than sampled discrete indices.
            mask = self.action_mask_for_task_ids(self.task_indices).to(
                action_tensor.device
            )
            action_tensor = action_tensor.masked_fill(~mask, -torch.inf).argmax(dim=1)
        elif action_tensor.ndim == 2 and action_tensor.shape[1] == 1:
            action_tensor = action_tensor[:, 0]
        elif action_tensor.ndim != 1:
            raise ValueError(
                "MiniGrid actions must have shape [num_envs], [num_envs, 1], "
                f"or [num_envs, 7], got {tuple(action_tensor.shape)}"
            )
        if action_tensor.shape[0] != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} actions, got {action_tensor.shape[0]}"
            )

        discrete = action_tensor.to(dtype=torch.long, device="cpu").numpy()
        task_ids = self.task_indices.detach().cpu().numpy()
        valid_counts = self._action_mask.sum(dim=1).detach().cpu().numpy()
        invalid = (discrete < 0) | (discrete >= valid_counts[task_ids])
        if invalid.any():
            first = int(np.flatnonzero(invalid)[0])
            raise ValueError(
                f"Action {int(discrete[first])} is invalid for task "
                f"{int(task_ids[first])} in {self.suite.display_name}."
            )
        return discrete

    def step(self, actions):
        discrete_actions = self._discrete_actions(actions)
        observations = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=np.bool_)
        successes = np.zeros(self.num_envs, dtype=np.bool_)
        terminations = np.zeros(self.num_envs, dtype=np.bool_)
        timeouts = np.zeros(self.num_envs, dtype=np.bool_)
        episode_times = np.zeros(self.num_envs, dtype=np.float32)

        for env_id, action in enumerate(discrete_actions):
            if self._needs_reset[env_id]:
                observations.append(self._reset_one(env_id))
                continue

            observation, reward, terminated, truncated, _ = self._envs[env_id].step(
                int(action)
            )
            self._episode_steps[env_id] += 1
            done = bool(terminated or truncated)
            rewards[env_id] = float(reward)
            dones[env_id] = done
            if done:
                terminations[env_id] = bool(terminated)
                successes[env_id] = bool(terminated and float(reward) > 0.0)
                timeouts[env_id] = bool(truncated and not terminated)
                episode_times[env_id] = float(self._episode_steps[env_id])
                self._needs_reset[env_id] = True
            observations.append(observation)

        encoded = self._encode_batch(observations)
        info = {
            "success": torch.as_tensor(successes, device=self.device),
            # MiniGrid's official reward is sparse (success reward, plus a
            # failure penalty in environments such as DynamicObstacles).
            # Expose the absent dense component explicitly so the shared
            # logger does not infer one by subtracting a unit success bonus.
            "dense_reward": torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            ),
            "time_outs": torch.as_tensor(timeouts, device=self.device),
            "terminated": torch.as_tensor(terminations, device=self.device),
            "eps_time": torch.as_tensor(
                episode_times, dtype=torch.float32, device=self.device
            ),
        }
        return (
            {"obs": encoded, "states": encoded.clone()},
            torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
            torch.as_tensor(dones, device=self.device),
            info,
        )

    def close(self):
        for env in self._envs:
            env.close()


__all__ = [
    "MINIGRID_MT7_CNN",
    "MINIGRID_MT7_CNN_TASKS",
    "MINIGRID_MT20",
    "MINIGRID_MT20_TASKS",
    "MINIGRID_MT50",
    "MINIGRID_MT50_TASKS",
    "MINIGRID_SUITES",
    "MiniGridVecEnv",
    "get_minigrid_suite",
]
