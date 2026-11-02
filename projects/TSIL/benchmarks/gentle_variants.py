"""Gentle T10/T11 probes, separate from the retained MT15 bank."""
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Key, Ball, Door, Wall
from minigrid.minigrid_env import MiniGridEnv
from minigrid.envs import KeyCorridorEnv
from projects.TSIL.benchmarks.control_variants import VariablePoseBlockedUnlockPickupEnv


class FacingBallBlockedUnlockPickupEnv(VariablePoseBlockedUnlockPickupEnv):
    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        for direction,(dx,dy) in enumerate(((1,0),(0,1),(-1,0),(0,-1))):
            obj = self.grid.get(self.agent_pos[0]+dx,self.agent_pos[1]+dy)
            if obj is not None and obj.type == 'ball':
                self.agent_dir = direction
                return
        raise RuntimeError('Expected an adjacent blocking ball')


class GentleKeyCorridorEnv(KeyCorridorEnv):
    """Keep one-cell side rooms; add only 1 or 2 middle-passage floor cells.

    Left key -- ordinary door -- agent passage -- locked door -- target ball.
    Native KeyCorridor pickup success/termination and H=270 are unchanged.
    """
    def __init__(self, **kwargs):
        MiniGridEnv.__init__(self, mission_space=MissionSpace(lambda:'pick up the purple ball'),
                            width=9, height=3, max_steps=270, **kwargs)

    def _gen_grid(self, width, height):
        extra = self._rand_int(1,3)
        self.grid = Grid(width,height)
        self.grid.wall_rect(0,0,width,height)
        # Unused trailing canvas stays wall; active right boundary depends on extra.
        for x in range(6+extra,width): self.grid.set(x,1,Wall())
        self.put_obj(Key('yellow'),1,1)
        self.put_obj(Door('blue'),2,1)
        self.put_obj(Door('yellow',is_locked=True),4+extra,1)
        self.obj = Ball('purple')
        self.put_obj(self.obj,5+extra,1)
        self.agent_pos = (self._rand_int(3,4+extra),1)
        self.agent_dir = self._rand_int(0,4)
        self.mission = 'pick up the purple ball'


def register_gentle_variants():
    import gymnasium as gym
    for name,cls in (
        ('TSIL-BlockedUnlockPickup-S4-FacingBall-VariableBox-H576-v0','FacingBallBlockedUnlockPickupEnv'),
        ('TSIL-KeyCorridor-GentlePassage-H270-v0','GentleKeyCorridorEnv'),
    ):
        if name not in gym.registry: gym.register(name,entry_point=__name__+':'+cls)
