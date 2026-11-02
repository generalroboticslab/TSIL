"""Remove nuisance recoloring without changing targets, key matches or dynamics."""
import gymnasium as gym

PALETTE = ('red', 'green', 'blue', 'purple', 'yellow', 'grey')


def canonicalize(env, family):
    cells = [((x,y), env.grid.get(x,y)) for x in range(env.width) for y in range(env.height)
             if env.grid.get(x,y) is not None]
    if family in {'GoToObject', 'GoToDoor', 'Fetch'}:
        # Group equal type/color labels. In Fetch, two identical objects can both
        # satisfy the mission; never turn that into a single acceptable target.
        groups = {}
        for pos,obj in cells:
            if obj.type in ('key','ball','box','door'):
                groups.setdefault((obj.type,obj.color), []).append((pos,obj))
        target = (env.targetType, env.targetColor) if family == 'Fetch' else None
        ordered = sorted(groups, key=lambda k:(k[0], min(p for p,_ in groups[k])))
        for i,key in enumerate(ordered):
            for _,obj in groups[key]: obj.color = PALETTE[i]
        if family == 'Fetch':
            env.targetColor = groups[target][0][1].color
            env.mission = 'fetch a %s %s' % (env.targetColor,env.targetType)
        else:
            obj = env.grid.get(*env.target_pos)
            env.target_color = obj.color
            env.mission = 'go to the %s %s' % (obj.color,obj.type)
        return
    # Red and blue are intrinsic semantic roles; do not remap their order.
    if family == 'RedBlueDoors':
        return
    for _,obj in cells:
        if obj.type == 'key' or (obj.type == 'door' and obj.is_locked): obj.color = 'yellow'
        elif obj.type in ('door','ball'): obj.color = 'blue'
        elif obj.type == 'box': obj.color = 'purple'
    if family in {'UnlockPickup','BlockedUnlockPickup','KeyCorridor'}:
        env.obj.color = 'purple'
        env.mission = 'pick up the purple ' + env.obj.type
    elif family == 'DoorKey':
        env.mission = 'use the key to open the door and then get to the goal'
    elif family == 'Unlock':
        env.mission = 'open the door'


class FixedPalette(gym.Wrapper):
    def __init__(self, env, family):
        super().__init__(env)
        self.family = family

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        canonicalize(self.unwrapped, self.family)
        return self.unwrapped.gen_obs(), info


def make_fixed_palette(base_id, family, **kwargs):
    return FixedPalette(gym.make(base_id, **kwargs), family)
