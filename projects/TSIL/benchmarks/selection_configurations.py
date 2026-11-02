"""Outcome-blind selection of distinct configurations with covered validation strata.

Coverage means observed spatial bins, headings, missions and object/door categories,
plus coordinate-wise training bounds. It is not a proof of joint-support coverage
or a claim that every unseen topology has the same difficulty.
"""
import numpy as np

from projects.TSIL.benchmarks.configurations import describe, fingerprint
from projects.TSIL.benchmarks.suites import get_tsil_minigrid_suite


def coverage_features(descriptor):
    from minigrid.core.constants import OBJECT_TO_IDX
    grid = np.asarray(descriptor['grid'])
    width, height = grid.shape[:2]
    features = {('heading', descriptor['agent_dir'])}
    coordinates = {}

    def position(name, x, y):
        # 3x3 coarse spatial strata: prevent unseen edge/room regions in validation.
        features.add((name, min(2, int(3*x/width)), min(2, int(3*y/height))))
        coordinates.setdefault(name, []).append((int(x), int(y)))

    position('start', *descriptor['agent_pos'])
    if 'mission' in descriptor:
        features.add(('mission', descriptor['mission']))
    objects = {OBJECT_TO_IDX[k]: k for k in ('door', 'key', 'ball', 'box', 'goal')}
    for index, name in objects.items():
        cells = np.argwhere(grid[:, :, 0] == index)
        features.add(('count', name, len(cells)))
        for x, y in cells:
            position(name, x, y)
            features.add(('object', name, int(grid[x,y,1]), int(grid[x,y,2])))
    # Lava wall/opening position and internal wall density describe obstacle strata.
    for name in ('lava', 'wall'):
        cells = np.argwhere(grid[:, :, 0] == OBJECT_TO_IDX[name])
        interior = [(x,y) for x,y in cells if 0 < x < width-1 and 0 < y < height-1]
        bins = {(min(2,int(3*x/width)), min(2,int(3*y/height))) for x,y in interior}
        features.update((name, *xy) for xy in bins)
        features.add(('density', name, len(interior)//8))
    return features, coordinates


def training_envelope(records):
    union, bounds = set(), {}
    for record in records:
        features, positions = coverage_features(record['descriptor'])
        union.update(features)
        for name, points in positions.items():
            bounds.setdefault(name, []).extend(points)
    bounds = {name: (np.min(points, axis=0), np.max(points, axis=0)) for name,points in bounds.items()}
    return union, bounds


def covered(record, envelope):
    features, positions = coverage_features(record['descriptor'])
    union, bounds = envelope
    return features <= union and all(name in bounds and all(
        np.all(np.asarray(point) >= bounds[name][0]) and np.all(np.asarray(point) <= bounds[name][1])
        for point in points) for name,points in positions.items())


def build_selection_bank(count=32, seed=42, tasks=None, pool_size=512, suite='mt15_selection'):
    from projects.TSIL.benchmarks.minigrid_adapter import _load_minigrid
    if count < 1 or pool_size < 2*count:
        raise ValueError('Need positive count and a candidate pool at least twice that count')
    suite_name = suite
    suite = get_tsil_minigrid_suite(suite_name)
    gym = _load_minigrid(suite)
    tasks = list(range(suite.task_count)) if tasks is None else list(tasks)
    if not tasks or len(set(tasks)) != len(tasks) or any(t not in range(15) for t in tasks):
        raise ValueError('Expected unique task IDs 0..14')
    bank = dict(format_version=1, suite=suite_name, seed=seed,
                requested_per_split=count, coverage_rule='observed_strata_and_coordinate_bounds_v1', tasks={})
    for task in tasks:
        spec = suite.tasks[task]
        env = gym.make(spec.env_id)
        records, seen = [], set()
        try:
            for offset in range(pool_size * 40):
                config_seed = seed + task * 100000 + offset
                env.reset(seed=config_seed)
                raw = env.unwrapped
                if raw.max_steps != spec.max_steps:
                    raise RuntimeError(f'{spec.env_id}: unexpected native horizon {raw.max_steps}')
                descriptor = describe(raw, spec.family)
                identity = fingerprint(spec.env_id, descriptor)
                if identity in seen:
                    continue
                seen.add(identity)
                records.append(dict(id=identity, seed=config_seed, descriptor=descriptor,
                                    target=float(spec.max_steps), success_count=0, best_success_steps=None))
                if len(records) == pool_size:
                    break
        finally:
            env.close()
        if len(records) < 2*count:
            raise RuntimeError(f'{spec.family}: insufficient distinct configuration support ({len(records)})')
        rng = np.random.RandomState(seed + task)
        rng.shuffle(records)
        # Cover as many input strata as possible without consulting any policy result.
        features = [coverage_features(r['descriptor'])[0] for r in records]
        union, selected = set(), []
        available = list(range(len(records)))
        for _ in range(count):
            index = max(available, key=lambda i: len(features[i] - union))
            selected.append(records[index])
            union.update(features[index])
            available.remove(index)
        envelope = training_envelope(selected)
        validation = [records[i] for i in available if covered(records[i], envelope)][:count]
        if len(validation) != count:
            raise RuntimeError(f'{spec.family}: only {len(validation)} covered validation configs; enlarge pool')
        bank['tasks'][str(task)] = dict(
            family=spec.family, env_id=spec.env_id, horizon=spec.max_steps,
            train=selected, heldout=validation, candidate_count=len(records),
            coverage=dict(all_heldout_covered=True, distinct_identities=2*count,
                training_strata=len(envelope[0]), coordinate_bounds={
                    name: [low.tolist(),high.tolist()] for name,(low,high) in envelope[1].items()}))
    return bank
