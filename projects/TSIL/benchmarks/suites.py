"""TSIL MiniGrid task selections and versioned configuration suites."""

from projects.TSIL.benchmarks.minigrid_adapter import (
    MINIGRID_SUITES, MiniGridSuiteSpec, MiniGridTaskSpec, get_minigrid_suite,
)


MT7_LARGE = MiniGridSuiteSpec("tsil_mt7_large", "TSIL-MiniGrid-MT7-Large", (
    MiniGridTaskSpec(0, "DoorKey", "MiniGrid-DoorKey-8x8-v0", 640),
    MiniGridTaskSpec(1, "DistShift", "MiniGrid-DistShift1-v0", 252),
    MiniGridTaskSpec(2, "RedBlueDoors", "MiniGrid-RedBlueDoors-8x8-v0", 1280),
    MiniGridTaskSpec(3, "LavaGap", "MiniGrid-LavaGapS7-v0", 196),
    MiniGridTaskSpec(4, "Memory", "MiniGrid-MemoryS11-v0", 605),
    MiniGridTaskSpec(5, "Crossing", "MiniGrid-SimpleCrossingS11N5-v0", 484),
    MiniGridTaskSpec(6, "MultiRoom", "MiniGrid-MultiRoom-N6-v0", 120),
))

# Ten native variants, with constant missions and no disappearing-cue task.
# Retained MT7 variants keep their task IDs for bank/result correspondence.
MT10_FF = MiniGridSuiteSpec("tsil_mt10_ff", "TSIL-MiniGrid-MT10-FF", (
    MiniGridTaskSpec(0, "DoorKey", "MiniGrid-DoorKey-6x6-v0", 360),
    MiniGridTaskSpec(1, "DistShift", "MiniGrid-DistShift1-v0", 252),
    MiniGridTaskSpec(2, "Unlock", "MiniGrid-Unlock-v0", 288),
    MiniGridTaskSpec(3, "LavaGap", "MiniGrid-LavaGapS7-v0", 196),
    MiniGridTaskSpec(4, "DoorKey", "MiniGrid-DoorKey-8x8-v0", 640),
    MiniGridTaskSpec(5, "Crossing", "MiniGrid-SimpleCrossingS9N2-v0", 324),
    MiniGridTaskSpec(6, "MultiRoom", "MiniGrid-MultiRoom-N2-S4-v0", 40),
    MiniGridTaskSpec(7, "Crossing", "MiniGrid-SimpleCrossingS11N5-v0", 484),
    MiniGridTaskSpec(8, "Crossing", "MiniGrid-LavaCrossingS9N2-v0", 324),
    MiniGridTaskSpec(9, "MultiRoom", "MiniGrid-MultiRoom-N4-S5-v0", 120),
))


MT15_FF = MiniGridSuiteSpec("tsil_mt15_ff", "TSIL-MiniGrid-MT15-FF", tuple(
    MiniGridTaskSpec(i, *row) for i, row in enumerate([
        ("Empty", "MiniGrid-Empty-Random-5x5-v0", 100),
        ("DistShift", "MiniGrid-DistShift1-v0", 252),
        ("DoorKey", "MiniGrid-DoorKey-5x5-v0", 250),
        ("Crossing", "MiniGrid-SimpleCrossingS9N1-v0", 324),
        ("LavaGap", "MiniGrid-LavaGapS5-v0", 100),
        ("MultiRoom", "MiniGrid-MultiRoom-N2-S4-v0", 40),
        ("DynamicObstacles", "MiniGrid-Dynamic-Obstacles-5x5-v0", 100, 3),
        ("FourRooms", "MiniGrid-FourRooms-v0", 100),
        ("Unlock", "MiniGrid-Unlock-v0", 288),
        ("UnlockPickup", "MiniGrid-UnlockPickup-v0", 288),
        ("BlockedUnlockPickup", "MiniGrid-BlockedUnlockPickup-v0", 576),
        ("KeyCorridor", "MiniGrid-KeyCorridorS3R1-v0", 270),
        ("RedBlueDoors", "TSIL-RedBlueDoors-S4-H720-v0", 720),
        ("Fetch", "MiniGrid-Fetch-5x5-N2-v0", 125),
        ("GoToDoor", "MiniGrid-GoToDoor-5x5-v0", 100),
    ])
))


MT15_SELECTION = MiniGridSuiteSpec('tsil_mt15_selection', 'TSIL-MiniGrid-MT15-Selection', tuple(
    MiniGridTaskSpec(i, *row) for i, row in enumerate([
        ('Empty', 'TSIL-Empty-StartGoal-6x6-v0', 144),
        ('GoToObject', 'MiniGrid-GoToObject-6x6-N2-v0', 180),
        ('DoorKey', 'MiniGrid-DoorKey-6x6-v0', 360),
        ('Crossing', 'MiniGrid-SimpleCrossingS9N2-v0', 324),
        ('LavaGap', 'TSIL-LavaGap-Start-S7-v0', 196),
        ('MultiRoom', 'TSIL-MultiRoom-N3-S4-H120-v0', 120),
        ('DynamicObstacles', 'MiniGrid-Dynamic-Obstacles-Random-6x6-v0', 144, 3),
        ('FourRooms', 'MiniGrid-FourRooms-v0', 100),
        ('Unlock', 'MiniGrid-Unlock-v0', 288),
        ('UnlockPickup', 'TSIL-UnlockPickup-S4-H288-v0', 288),
        ('BlockedUnlockPickup', 'TSIL-BlockedUnlockPickup-S4-H576-v0', 576),
        ('KeyCorridor', 'MiniGrid-KeyCorridorS3R1-v0', 270),
        ('RedBlueDoors', 'TSIL-RedBlueDoors-S4-H720-v0', 720),
        ('Fetch', 'MiniGrid-Fetch-5x5-N2-v0', 125),
        ('GoToDoor', 'MiniGrid-GoToDoor-5x5-v0', 100),
    ])
))


_REFINEMENTS = {
    2: MiniGridTaskSpec(2, 'DoorKey', 'TSIL-DoorKey-S7-H360-v0', 360),
    8: MiniGridTaskSpec(8, 'Unlock', 'TSIL-Unlock-S7-H288-v0', 288),
    10: MiniGridTaskSpec(10, 'BlockedUnlockPickup', 'TSIL-BlockedUnlockPickup-S4-EasyStart-H576-v0', 576),
    12: MiniGridTaskSpec(12, 'RedBlueDoors', 'TSIL-RedBlueDoors-S5-H720-v0', 720),
}
MT15_REFINEMENT = MiniGridSuiteSpec('tsil_mt15_refinement', 'TSIL-MiniGrid-MT15-Refinement',
    tuple(_REFINEMENTS.get(i, spec) for i, spec in enumerate(MT15_SELECTION.tasks)))


_CONTROL_VARIANTS = {
    10: MiniGridTaskSpec(10, 'BlockedUnlockPickup', 'TSIL-BlockedUnlockPickup-S4-VariablePose-FixedColors-H576-v0', 576),
    11: MiniGridTaskSpec(11, 'KeyCorridor', 'TSIL-KeyCorridor-S4R1-FixedColors-H270-v0', 270),
}
MT15_CONTROL = MiniGridSuiteSpec('tsil_mt15_control', 'TSIL-MiniGrid-MT15-Control',
    tuple(_CONTROL_VARIANTS.get(i, spec) for i, spec in enumerate(MT15_REFINEMENT.tasks)))


MT15_FIXED = MiniGridSuiteSpec('tsil_mt15_fixed', 'TSIL-MiniGrid-MT15-FixedPalette', tuple(
    MiniGridTaskSpec(i, spec.family, 'TSIL-MT15-Fixed-T%02d-v0' % i, spec.max_steps, spec.num_actions)
    for i,spec in enumerate(MT15_REFINEMENT.tasks)))
_GENTLE_VARIANTS = {
    10: MiniGridTaskSpec(10, 'BlockedUnlockPickup', 'TSIL-BlockedUnlockPickup-S4-FacingBall-VariableBox-H576-v0',576),
    11: MiniGridTaskSpec(11, 'KeyCorridor', 'TSIL-KeyCorridor-GentlePassage-H270-v0',270),
}
MT15_GENTLE = MiniGridSuiteSpec('tsil_mt15_gentle', 'TSIL-MiniGrid-MT15-Gentle',
    tuple(_GENTLE_VARIANTS.get(i,spec) for i,spec in enumerate(MT15_FIXED.tasks)))


_REPAIR_V1_VARIANTS = {
    10: MiniGridTaskSpec(10, 'BlockedUnlockPickup',
        'TSIL-BlockedUnlockPickup-S5-FacingBall-H576-v0', 576),
    11: MiniGridTaskSpec(11, 'KeyCorridor',
        'TSIL-KeyCorridor-GentleLong-H270-v0', 270),
}
MT15_REPAIR_V1 = MiniGridSuiteSpec('tsil_mt15_repair_v1',
    'TSIL-MiniGrid-MT15-RepairV1',
    tuple(_REPAIR_V1_VARIANTS.get(i, spec) for i, spec in enumerate(MT15_FIXED.tasks)))


_REPAIR_V2_VARIANTS = {
    10: MiniGridTaskSpec(10, 'BlockedUnlockPickup',
        'TSIL-BlockedUnlockPickup-S5-NearDoor-H180-v0', 180),
    11: MiniGridTaskSpec(11, 'KeyCorridor',
        'TSIL-KeyCorridor-AdjacentKey-H120-v0', 120),
}
MT15_REPAIR_V2 = MiniGridSuiteSpec('tsil_mt15_repair_v2',
    'TSIL-MiniGrid-MT15-RepairV2',
    tuple(_REPAIR_V2_VARIANTS.get(i, spec) for i, spec in enumerate(MT15_FIXED.tasks)))


_REPAIR_V3_VARIANTS = {
    10: MiniGridTaskSpec(10, 'BlockedUnlockPickup',
        'TSIL-BlockedUnlockPickup-S4-NearFace-H120-v0', 120),
    11: MiniGridTaskSpec(11, 'KeyCorridor',
        'TSIL-KeyCorridor-ShortLocked-H80-v0', 80),
}
MT15_REPAIR_V3 = MiniGridSuiteSpec('tsil_mt15_repair_v3',
    'TSIL-MiniGrid-MT15-RepairV3',
    tuple(_REPAIR_V3_VARIANTS.get(i, spec) for i, spec in enumerate(MT15_FIXED.tasks)))


_REPAIR_V4_VARIANTS = {
    10: MiniGridTaskSpec(10, 'BlockedUnlockPickup', 'TSIL-BlockedUnlockPickup-NearKey-H120-v0', 120),
    11: MiniGridTaskSpec(11, 'KeyCorridor', 'TSIL-KeyCorridor-NearDoorKey-H80-v0', 80),
}
MT15_REPAIR_V4 = MiniGridSuiteSpec('tsil_mt15_repair_v4', 'TSIL-MiniGrid-MT15-RepairV4',
    tuple(_REPAIR_V4_VARIANTS.get(i, spec) for i, spec in enumerate(MT15_FIXED.tasks)))


def get_tsil_minigrid_suite(key):
    if key == 'mt15_repair_v4':
        get_tsil_minigrid_suite('mt15_fixed')
        from projects.TSIL.benchmarks.repair_variants import register_repair_v4_variants
        register_repair_v4_variants()
        MINIGRID_SUITES.setdefault(MT15_REPAIR_V4.key, MT15_REPAIR_V4)
        return MT15_REPAIR_V4
    if key == 'mt15_repair_v3':
        get_tsil_minigrid_suite('mt15_fixed')
        from projects.TSIL.benchmarks.repair_variants import register_repair_v3_variants
        register_repair_v3_variants()
        MINIGRID_SUITES.setdefault(MT15_REPAIR_V3.key, MT15_REPAIR_V3)
        return MT15_REPAIR_V3
    if key == 'mt15_repair_v2':
        get_tsil_minigrid_suite('mt15_fixed')
        from projects.TSIL.benchmarks.repair_variants import register_repair_v2_variants
        register_repair_v2_variants()
        MINIGRID_SUITES.setdefault(MT15_REPAIR_V2.key, MT15_REPAIR_V2)
        return MT15_REPAIR_V2
    if key == 'mt15_repair_v1':
        get_tsil_minigrid_suite('mt15_fixed')
        from projects.TSIL.benchmarks.repair_variants import register_repair_variants
        register_repair_variants()
        MINIGRID_SUITES.setdefault(MT15_REPAIR_V1.key, MT15_REPAIR_V1)
        return MT15_REPAIR_V1
    if key in {'mt15_fixed','mt15_gentle'}:
        import gymnasium as gym
        from projects.TSIL.benchmarks.refinement_variants import register_refinement_variants
        register_refinement_variants()
        for spec,base in zip(MT15_FIXED.tasks,MT15_REFINEMENT.tasks):
            if spec.env_id not in gym.registry:
                gym.register(spec.env_id,entry_point='projects.TSIL.benchmarks.fixed_palette:make_fixed_palette',
                             kwargs=dict(base_id=base.env_id,family=base.family))
        result = MT15_FIXED
        if key == 'mt15_gentle':
            from projects.TSIL.benchmarks.gentle_variants import register_gentle_variants
            register_gentle_variants()
            result = MT15_GENTLE
        MINIGRID_SUITES.setdefault(result.key,result)
        return result
    if key == 'mt15_control':
        from projects.TSIL.benchmarks.control_variants import register_control_variants
        register_control_variants()
        MINIGRID_SUITES.setdefault(MT15_CONTROL.key, MT15_CONTROL)
        return MT15_CONTROL
    if key == 'mt15_refinement':
        from projects.TSIL.benchmarks.refinement_variants import register_refinement_variants
        register_refinement_variants()
        MINIGRID_SUITES.setdefault(MT15_REFINEMENT.key, MT15_REFINEMENT)
        return MT15_REFINEMENT
    if key == 'mt15_selection':
        from projects.TSIL.benchmarks.selection_variants import register_selection_variants
        register_selection_variants()
        MINIGRID_SUITES.setdefault(MT15_SELECTION.key, MT15_SELECTION)
        return MT15_SELECTION
    if key == "mt15_ff":
        import gymnasium as gym
        custom_id = "TSIL-RedBlueDoors-S4-H720-v0"
        if custom_id not in gym.registry:
            gym.register(custom_id, entry_point="minigrid.envs:RedBlueDoorEnv",
                         kwargs=dict(size=4, max_steps=720))
        MINIGRID_SUITES.setdefault(MT15_FF.key, MT15_FF)
        return MT15_FF
    if key == "mt10_ff":
        MINIGRID_SUITES.setdefault(MT10_FF.key, MT10_FF)
        return MT10_FF
    if key == "mt7_large":
        # The shared vector adapter resolves suites by key. Add only our unique
        # key at runtime, retaining every existing suite and registration.
        MINIGRID_SUITES.setdefault(MT7_LARGE.key, MT7_LARGE)
        return MT7_LARGE
    return get_minigrid_suite("mt7_cnn" if key == "mt7" else key)
