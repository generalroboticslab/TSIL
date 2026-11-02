"""Two control-diverse variants with a fixed semantic palette."""
from minigrid.envs import KeyCorridorEnv

from projects.TSIL.benchmarks.refinement_variants import (
    EasyStartBlockedUnlockPickupEnv, register_refinement_variants,
)


def fixed_palette(env):
    for cell in env.grid.grid:
        if cell is None:
            continue
        if cell.type == 'key' or (cell.type == 'door' and cell.is_locked):
            cell.color = 'yellow'
        elif cell.type == 'door' or cell.type == 'ball':
            cell.color = 'blue'
    env.obj.color = 'purple'
    env.mission = 'pick up the purple ' + env.obj.type


class VariablePoseBlockedUnlockPickupEnv(EasyStartBlockedUnlockPickupEnv):
    """Retain the adjacent-ball start, vary heading and all right-room box cells."""
    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        self.grid.set(*self.obj.cur_pos, None)
        room = self.get_room(1, 0)
        candidates = [(x, y) for x in range(room.top[0] + 1, room.top[0] + room.size[0] - 1)
                      for y in range(room.top[1] + 1, room.top[1] + room.size[1] - 1)
                      if self.grid.get(x, y) is None]
        self.put_obj(self.obj, *candidates[self._rand_int(0, len(candidates))])
        self.agent_dir = self._rand_int(0, 4)
        fixed_palette(self)


class FixedColorKeyCorridorEnv(KeyCorridorEnv):
    def __init__(self, **kwargs):
        super().__init__(room_size=4, num_rows=1, max_steps=270, **kwargs)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        fixed_palette(self)


def register_control_variants():
    import gymnasium as gym
    register_refinement_variants()
    for name, cls in (
        ('TSIL-BlockedUnlockPickup-S4-VariablePose-FixedColors-H576-v0', 'VariablePoseBlockedUnlockPickupEnv'),
        ('TSIL-KeyCorridor-S4R1-FixedColors-H270-v0', 'FixedColorKeyCorridorEnv'),
    ):
        if name not in gym.registry:
            gym.register(name, entry_point=__name__ + ':' + cls)
