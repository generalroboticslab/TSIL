"""Compact task layouts with native rewards and actions."""
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid
from minigrid.core.world_object import Goal
from minigrid.envs import EmptyEnv, LavaGapEnv, UnlockPickupEnv, BlockedUnlockPickupEnv


class EmptyStartGoalEnv(EmptyEnv):
    """Sample distinct start and goal cells, with the native random heading."""
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.place_obj(Goal())
        self.place_agent()
        self.mission = self._gen_mission()


class LavaGapStartEnv(LavaGapEnv):
    """Random safe starts strictly before the lava wall; crossing is still required."""
    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        self.place_agent(top=(1, 1), size=(int(self.gap_pos[0]) - 1, height - 2))


class CompactUnlockPickupEnv(UnlockPickupEnv):
    def __init__(self, room_size=4, max_steps=288, **kwargs):
        RoomGrid.__init__(self, mission_space=MissionSpace(
            mission_func=self._gen_mission, ordered_placeholders=[COLOR_NAMES]),
            num_rows=1, num_cols=2, room_size=room_size, max_steps=max_steps, **kwargs)


class CompactBlockedUnlockPickupEnv(BlockedUnlockPickupEnv):
    def __init__(self, room_size=4, max_steps=576, **kwargs):
        RoomGrid.__init__(self, mission_space=MissionSpace(
            mission_func=self._gen_mission, ordered_placeholders=[COLOR_NAMES, ['box', 'key']]),
            num_rows=1, num_cols=2, room_size=room_size, max_steps=max_steps, **kwargs)


def register_selection_variants():
    import gymnasium as gym
    import minigrid
    # Our first suite entry is custom, so the shared loader's sentinel is insufficient.
    if 'MiniGrid-Empty-6x6-v0' not in gym.registry:
        minigrid.register_minigrid_envs()
    entries = {
        'TSIL-Empty-StartGoal-6x6-v0': ('EmptyStartGoalEnv', dict(size=6, max_steps=144)),
        'TSIL-LavaGap-Start-S7-v0': ('LavaGapStartEnv', dict(size=7, max_steps=196)),
        'TSIL-UnlockPickup-S4-H288-v0': ('CompactUnlockPickupEnv', dict()),
        'TSIL-BlockedUnlockPickup-S4-H576-v0': ('CompactBlockedUnlockPickupEnv', dict()),
    }
    for name, (cls, kwargs) in entries.items():
        if name not in gym.registry:
            gym.register(name, entry_point=__name__ + ':' + cls, kwargs=kwargs)
    native = {
        'TSIL-MultiRoom-N3-S4-H120-v0': ('MultiRoomEnv', dict(
            minNumRooms=3, maxNumRooms=3, maxRoomSize=4, max_steps=120)),
        'TSIL-RedBlueDoors-S4-H720-v0': ('RedBlueDoorEnv', dict(size=4, max_steps=720)),
    }
    for name, (cls, kwargs) in native.items():
        if name not in gym.registry:
            gym.register(name, entry_point='minigrid.envs:' + cls, kwargs=kwargs)
