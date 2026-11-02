"""Versioned T10/T11 layout candidates; retain native pickup objectives."""
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Door, Key, Wall
from minigrid.envs import BlockedUnlockPickupEnv, KeyCorridorEnv
from minigrid.minigrid_env import MiniGridEnv

from projects.TSIL.benchmarks.control_variants import VariablePoseBlockedUnlockPickupEnv
from projects.TSIL.benchmarks.gentle_variants import (
    FacingBallBlockedUnlockPickupEnv, register_gentle_variants,
)


class RoomierFacingBallBlockedUnlockPickupEnv(FacingBallBlockedUnlockPickupEnv):
    """Keep the blocked unlock-and-pickup sequence; vary a roomier layout."""

    def __init__(self, **kwargs):
        super().__init__(room_size=5, max_steps=576, **kwargs)


class LongerGentleKeyCorridorEnv(KeyCorridorEnv):
    """Native key/door/ball sequence with five passage lengths and safe starts."""

    def __init__(self, **kwargs):
        MiniGridEnv.__init__(
            self, mission_space=MissionSpace(lambda: 'pick up the purple ball'),
            width=13, height=3, max_steps=270, **kwargs,
        )

    def _gen_grid(self, width, height):
        extra = self._rand_int(1, 6)
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        for x in range(6 + extra, width):
            self.grid.set(x, 1, Wall())
        self.put_obj(Key('yellow'), 1, 1)
        self.put_obj(Door('blue'), 2, 1)
        self.put_obj(Door('yellow', is_locked=True), 4 + extra, 1)
        self.obj = Ball('purple')
        self.put_obj(self.obj, 5 + extra, 1)
        self.agent_pos = (self._rand_int(3, 4 + extra), 1)
        self.agent_dir = self._rand_int(0, 4)
        self.mission = 'pick up the purple ball'


def register_repair_variants():
    import gymnasium as gym
    register_gentle_variants()
    entries = (
        ('TSIL-BlockedUnlockPickup-S5-FacingBall-H576-v0', RoomierFacingBallBlockedUnlockPickupEnv),
        ('TSIL-KeyCorridor-GentleLong-H270-v0', LongerGentleKeyCorridorEnv),
    )
    for name, cls in entries:
        if name not in gym.registry:
            gym.register(name, entry_point=__name__ + ':' + cls.__name__)


class NearDoorBlockedUnlockPickupEnv(FacingBallBlockedUnlockPickupEnv):
    """Blocked pickup with a nearby box and a shorter retry horizon."""

    def __init__(self, **kwargs):
        super().__init__(room_size=5, max_steps=180, **kwargs)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        self.grid.set(*self.obj.cur_pos, None)
        room = self.get_room(1, 0)
        x = room.top[0] + 1
        candidates = [(x, y) for y in range(room.top[1] + 1, room.top[1] + room.size[1] - 1)
                      if self.grid.get(x, y) is None]
        self.put_obj(self.obj, *candidates[self._rand_int(0, len(candidates))])


class AdjacentKeyCorridorEnv(KeyCorridorEnv):
    """Larger key room and corridor; the start is next to the required key."""

    def __init__(self, **kwargs):
        MiniGridEnv.__init__(
            self, mission_space=MissionSpace(lambda: 'pick up the purple ball'),
            width=12, height=5, max_steps=120, **kwargs,
        )

    def _gen_grid(self, width, height):
        extra = self._rand_int(0, 2)
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        for x in range(4, width - 1):
            for y in (1, 3):
                self.grid.set(x, y, Wall())
        for x in range(10 + extra, width - 1):
            self.grid.set(x, 2, Wall())
        self.put_obj(Door('blue', is_open=True), 4, 2)
        self.put_obj(Door('yellow', is_locked=True), 8 + extra, 2)
        self.obj = Ball('purple')
        self.put_obj(self.obj, 9 + extra, 2)
        key_pos = (self._rand_int(1, 4), self._rand_int(1, 4))
        self.put_obj(Key('yellow'), *key_pos)
        starts = [(key_pos[0] + dx, key_pos[1] + dy)
                  for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1))
                  if 1 <= key_pos[0] + dx <= 3 and 1 <= key_pos[1] + dy <= 3]
        self.agent_pos = starts[self._rand_int(0, len(starts))]
        toward_key = next(d for d, (dx, dy) in enumerate(((1, 0), (0, 1), (-1, 0), (0, -1)))
                          if (self.agent_pos[0] + dx, self.agent_pos[1] + dy) == key_pos)
        self.agent_dir = (toward_key + self._rand_int(0, 2)) % 4
        self.mission = 'pick up the purple ball'


def register_repair_v2_variants():
    import gymnasium as gym
    entries = (
        ('TSIL-BlockedUnlockPickup-S5-NearDoor-H180-v0', NearDoorBlockedUnlockPickupEnv),
        ('TSIL-KeyCorridor-AdjacentKey-H120-v0', AdjacentKeyCorridorEnv),
    )
    for name, cls in entries:
        if name not in gym.registry:
            gym.register(name, entry_point=__name__ + ':' + cls.__name__)


class CompactNearFaceBlockedUnlockPickupEnv(VariablePoseBlockedUnlockPickupEnv):
    """Retain the compact blocker puzzle; vary a facing or one-turn start."""

    def __init__(self, **kwargs):
        super().__init__(room_size=4, max_steps=120, **kwargs)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        ball = next((x, y) for x in range(width) for y in range(height)
                    if self.grid.get(x, y) is not None and self.grid.get(x, y).type == 'ball')
        directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
        facing = next(d for d, (dx, dy) in enumerate(directions)
                      if (self.agent_pos[0] + dx, self.agent_pos[1] + dy) == ball)
        self.agent_dir = (facing + self._rand_int(0, 2)) % 4


class ShortLockedKeyCorridorEnv(KeyCorridorEnv):
    """A large key room with a short, still-required locked corridor."""

    def __init__(self, **kwargs):
        MiniGridEnv.__init__(
            self, mission_space=MissionSpace(lambda: 'pick up the purple ball'),
            width=9, height=5, max_steps=80, **kwargs,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        for x in range(4, width - 1):
            for y in (1, 3):
                self.grid.set(x, y, Wall())
        self.grid.set(7, 2, Wall())
        self.put_obj(Door('blue', is_open=True), 4, 2)
        self.put_obj(Door('yellow', is_locked=True), 5, 2)
        self.obj = Ball('purple')
        self.put_obj(self.obj, 6, 2)
        key_pos = (self._rand_int(1, 4), self._rand_int(1, 4))
        self.put_obj(Key('yellow'), *key_pos)
        starts = [(key_pos[0] + dx, key_pos[1] + dy)
                  for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1))
                  if 1 <= key_pos[0] + dx <= 3 and 1 <= key_pos[1] + dy <= 3]
        self.agent_pos = starts[self._rand_int(0, len(starts))]
        self.agent_dir = self._rand_int(0, 4)
        self.mission = 'pick up the purple ball'


def register_repair_v3_variants():
    import gymnasium as gym
    entries = (
        ('TSIL-BlockedUnlockPickup-S4-NearFace-H120-v0', CompactNearFaceBlockedUnlockPickupEnv),
        ('TSIL-KeyCorridor-ShortLocked-H80-v0', ShortLockedKeyCorridorEnv),
    )
    for name, cls in entries:
        if name not in gym.registry:
            gym.register(name, entry_point=__name__ + ':' + cls.__name__)


class NearKeyBlockedUnlockPickupEnv(BlockedUnlockPickupEnv):
    """Two rooms, mandatory blocker, and a key beside the door approach."""
    def __init__(self, **kwargs):
        MiniGridEnv.__init__(self, mission_space=MissionSpace(lambda: 'pick up the purple box'),
                            width=9, height=5, max_steps=120, **kwargs)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        for y in (1, 3):
            self.grid.set(4, y, Wall())
        self.put_obj(Door('yellow', is_locked=True), 4, 2)
        self.put_obj(Ball('blue'), 3, 2)
        self.put_obj(Key('yellow'), 3, (1, 3)[self._rand_int(0, 2)])
        self.obj = Box('purple')
        self.put_obj(self.obj, self._rand_int(5, 8), self._rand_int(1, 4))
        self.agent_pos = ((2, 2), (2, 1), (2, 3), (1, 2))[self._rand_int(0, 4)]
        self.agent_dir = self._rand_int(0, 4)
        self.mission = 'pick up the purple box'


class NearDoorKeyCorridorEnv(KeyCorridorEnv):
    """Retain a larger key room, but keep the key at its corridor entrance."""
    def __init__(self, **kwargs):
        MiniGridEnv.__init__(self, mission_space=MissionSpace(lambda: 'pick up the purple ball'),
                            width=11, height=5, max_steps=80, **kwargs)

    def _gen_grid(self, width, height):
        extra = self._rand_int(0, 3)
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        for x in range(4, width-1):
            for y in (1, 3): self.grid.set(x, y, Wall())
        for x in range(7+extra, width-1): self.grid.set(x, 2, Wall())
        self.put_obj(Door('blue', is_open=True), 4, 2)
        self.put_obj(Door('yellow', is_locked=True), 5+extra, 2)
        self.obj = Ball('purple'); self.put_obj(self.obj, 6+extra, 2)
        key_pos = (3, self._rand_int(1, 4)); self.put_obj(Key('yellow'), *key_pos)
        starts = [(key_pos[0]+dx, key_pos[1]+dy) for dx,dy in ((1,0),(0,1),(-1,0),(0,-1))
                  if 1 <= key_pos[0]+dx <= 3 and 1 <= key_pos[1]+dy <= 3]
        self.agent_pos = starts[self._rand_int(0, len(starts))]
        self.agent_dir = self._rand_int(0, 4)
        self.mission = 'pick up the purple ball'


def register_repair_v4_variants():
    import gymnasium as gym
    for name, cls in (
        ('TSIL-BlockedUnlockPickup-NearKey-H120-v0', NearKeyBlockedUnlockPickupEnv),
        ('TSIL-KeyCorridor-NearDoorKey-H80-v0', NearDoorKeyCorridorEnv),
    ):
        if name not in gym.registry:
            gym.register(name, entry_point=__name__ + ':' + cls.__name__)
