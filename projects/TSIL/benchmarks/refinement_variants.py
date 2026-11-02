"""Second selection pilot: change geometry/start pose, retain rewards and limits."""
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid
from minigrid.envs import UnlockEnv

from projects.TSIL.benchmarks.selection_variants import (
    CompactBlockedUnlockPickupEnv, register_selection_variants,
)


class SizedUnlockEnv(UnlockEnv):
    def __init__(self, room_size=7, max_steps=288, **kwargs):
        RoomGrid.__init__(self, mission_space=MissionSpace(self._gen_mission),
            num_rows=1, num_cols=2, room_size=room_size, max_steps=max_steps, **kwargs)


class EasyStartBlockedUnlockPickupEnv(CompactBlockedUnlockPickupEnv):
    """Face the blocking ball from an empty adjacent cell; all puzzle steps remain."""
    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        ball = next((x, y) for x in range(width) for y in range(height)
                    if self.grid.get(x, y) is not None and self.grid.get(x, y).type == 'ball')
        poses = []
        for direction, (dx, dy) in enumerate(((1, 0), (0, 1), (-1, 0), (0, -1))):
            x, y = ball[0] - dx, ball[1] - dy
            if 0 < x < self.room_size - 1 and 0 < y < height - 1 and self.grid.get(x, y) is None:
                poses.append(((x, y), direction))
        if not poses:
            raise RuntimeError('Blocking ball has no empty adjacent start')
        self.agent_pos, self.agent_dir = poses[self._rand_int(0, len(poses))]


def register_refinement_variants():
    import gymnasium as gym
    register_selection_variants()
    entries = {
        'TSIL-DoorKey-S7-H360-v0': ('minigrid.envs:DoorKeyEnv', dict(size=7, max_steps=360)),
        'TSIL-Unlock-S7-H288-v0': (__name__ + ':SizedUnlockEnv', {}),
        'TSIL-BlockedUnlockPickup-S4-EasyStart-H576-v0': (
            __name__ + ':EasyStartBlockedUnlockPickupEnv', {}),
        'TSIL-RedBlueDoors-S5-H720-v0': ('minigrid.envs:RedBlueDoorEnv', dict(size=5, max_steps=720)),
    }
    for name, (entry_point, kwargs) in entries.items():
        if name not in gym.registry:
            gym.register(name, entry_point=entry_point, kwargs=kwargs)
