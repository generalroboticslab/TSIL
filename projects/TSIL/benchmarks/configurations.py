"""Fixed MiniGrid configurations and a task-local categorical target lookup."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from projects.TSIL.benchmarks.suites import get_tsil_minigrid_suite


def write_json(path, data, compact=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=None if compact else 2) + '\n')
    temporary.replace(path)


def describe(env, family):
    """Initial global metadata for target lookup only, never CNN input.

    Memory identity excludes cue/choice identities: those vary at every reset.
    Grid channels are compared categorically, not as numeric object-ID distances.
    """
    grid = env.grid.encode().copy()
    if family == 'Memory':
        from minigrid.core.constants import OBJECT_TO_IDX
        mask = np.isin(grid[:, :, 0], [OBJECT_TO_IDX['key'], OBJECT_TO_IDX['ball']])
        grid[:, :, 0][mask] = OBJECT_TO_IDX['key']
    descriptor = dict(grid=grid.tolist(), agent_pos=list(map(int, env.agent_pos)),
                      agent_dir=int(env.agent_dir))
    if family in {'Fetch', 'GoToDoor', 'GoToObject'}:
        descriptor['mission'] = env.mission
    return descriptor


def fingerprint(env_id, descriptor):
    payload = json.dumps([env_id, descriptor], sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()


def build_bank(count=32, seed=42, tasks=None, suite='mt7'):
    if suite in {'mt15_fixed','mt15_gentle'}:
        from projects.TSIL.benchmarks.retained_configurations import build_retained_bank
        return build_retained_bank(suite=suite,seed=seed,count=count,tasks=tasks)
    if suite in {'mt15_selection', 'mt15_refinement', 'mt15_control', 'mt15_repair_v1', 'mt15_repair_v2', 'mt15_repair_v3', 'mt15_repair_v4'}:
        from projects.TSIL.benchmarks.selection_configurations import build_selection_bank
        return build_selection_bank(count=count, seed=seed, tasks=tasks, suite=suite)
    from projects.TSIL.benchmarks.minigrid_adapter import _load_minigrid
    if count < 1:
        raise ValueError('Configuration count must be positive')
    suite_name = suite
    if suite_name not in {'mt7', 'mt10_ff', 'mt15_ff'}:
        raise ValueError('Fixed banks support mt7, mt10_ff and mt15_ff')
    suite = get_tsil_minigrid_suite(suite_name)
    gym = _load_minigrid(suite)
    tasks = list(range(suite.task_count)) if tasks is None else list(tasks)
    if not tasks or len(set(tasks)) != len(tasks) or any(t not in range(suite.task_count) for t in tasks):
        raise ValueError('Expected unique task IDs in the selected suite')
    bank = dict(format_version=1, suite=suite_name, seed=seed, requested_per_split=count,
                memory_rule='Fixed geometry/start; resample all four cue/choice variants', tasks={})
    # Exact finite supports of the native MT7 registrations (Memory ignores cue).
    finite_support = {'MiniGrid-DistShift1-v0': 1, 'MiniGrid-LavaGapS7-v0': 15,
                      'MiniGrid-MemoryS11-v0': 8, 'MiniGrid-LavaGapS5-v0': 3,
                      'MiniGrid-Empty-Random-5x5-v0': 32,
                      'MiniGrid-DoorKey-5x5-v0': 48,
                      'MiniGrid-SimpleCrossingS9N1-v0': 42,
                      'MiniGrid-Dynamic-Obstacles-5x5-v0': 21}
    for task in tasks:
        spec = suite.tasks[task]
        wanted = min(2 * count, finite_support.get(spec.env_id, 2 * count))
        env = gym.make(spec.env_id)
        records, seen = [], set()
        try:
            for offset in range(20000):
                config_seed = seed + task * 100000 + offset
                env.reset(seed=config_seed)
                descriptor = describe(env.unwrapped, spec.family)
                identity = fingerprint(spec.env_id, descriptor)
                if identity in seen:
                    continue
                seen.add(identity)
                records.append(dict(id=identity, seed=config_seed, descriptor=descriptor,
                                    target=float(spec.max_steps), success_count=0,
                                    best_success_steps=None))
                if len(records) == wanted:
                    break
            if len(records) != wanted:
                raise RuntimeError(f'{spec.family}: found only {len(records)}/{wanted} configurations')
        finally:
            env.close()
        # Split real configuration identities, not merely different RNG seeds.
        rng = np.random.RandomState(seed + task)
        rng.shuffle(records)
        train_count = min(count, (len(records) + 1) // 2)
        bank['tasks'][str(task)] = dict(
            family=spec.family, env_id=spec.env_id, horizon=spec.max_steps,
            train=records[:train_count], heldout=records[train_count:],
            finite_support=finite_support.get(spec.env_id),
        )
    return bank


def configuration_distance(left, right):
    a, b = np.asarray(left['grid']), np.asarray(right['grid'])
    if a.shape != b.shape:
        raise ValueError('Neighbors must belong to the same task/variant and grid shape')
    layout = np.any(a != b, axis=-1).mean()
    pose = np.abs(np.asarray(left['agent_pos']) - right['agent_pos']) / np.asarray(a.shape[:2])
    direction = float(left['agent_dir'] != right['agent_dir'])
    mission = float(left.get('mission') != right.get('mission'))
    return float(layout + pose.sum() + .25 * direction + mission)


def infer_target(query, task_bank, k=5):
    """Use successful training labels only; return both target and provenance."""
    if k < 1:
        raise ValueError('k must be positive')
    candidates = [r for r in task_bank['train'] if r['success_count'] > 0]
    if not candidates:
        return float(task_bank['horizon']), dict(neighbors=[], fallback='no_successful_training_configs')
    ranked = sorted(((configuration_distance(query['descriptor'], r['descriptor']), r)
                     for r in candidates), key=lambda pair: (pair[0], pair[1]['id']))[:k]
    target = float(np.clip(np.mean([r['target'] for _, r in ranked]), 1, task_bank['horizon']))
    return target, dict(neighbors=[dict(id=r['id'], distance=d, target=r['target']) for d, r in ranked],
                        fallback=None)


def vary_memory_cue(env, variant):
    """Preserve native geometry/start while preventing a worker-fixed answer."""
    from minigrid.core.world_object import Ball, Key
    middle, x = env.height // 2, env.width - 2
    cue = [Key, Ball][variant // 2]
    choices = [Key, Ball] if variant % 2 == 0 else [Ball, Key]
    env.grid.set(1, middle - 1, cue('green'))
    for y, obj in zip([middle - 2, middle + 2], choices):
        env.grid.set(x, y, obj('green'))
    upper, lower = (x, middle - 1), (x, middle + 1)
    env.success_pos, env.failure_pos = (upper, lower) if cue == choices[0] else (lower, upper)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--suite', choices=['mt7', 'mt10_ff', 'mt15_ff', 'mt15_selection', 'mt15_refinement', 'mt15_control', 'mt15_fixed', 'mt15_gentle', 'mt15_repair_v1', 'mt15_repair_v2', 'mt15_repair_v3', 'mt15_repair_v4'], default='mt7')
    parser.add_argument('--tasks',type=int,nargs='+')
    options = parser.parse_args()
    if options.output.exists():
        parser.error('Use a new bank path; existing banks are immutable inputs')
    result = build_bank(options.count, options.seed, tasks=options.tasks, suite=options.suite)
    write_json(options.output, result)
    print({t['env_id']: dict(train=len(t['train']), heldout=len(t['heldout'])) for t in result['tasks'].values()})
