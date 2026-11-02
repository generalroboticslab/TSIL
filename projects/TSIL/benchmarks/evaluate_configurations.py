"""Evaluate a checkpoint's own configuration bank, with no target-label updates."""
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from core.evaluation.metrics import normalized_completion_time
from projects.TSIL.benchmarks.configurations import infer_target
from projects.TSIL.benchmarks.minigrid import TSILMiniGridEnv, MiniGridAgent
from projects.TSIL.training import TSILArgs


def evaluate_configurations(run_dir, checkpoint='best_suc_tail', episodes_per_config=4, seed=1042, device='cuda', total_episodes=None, configuration_slots=32):
    if episodes_per_config < 1:
        raise ValueError('episodes_per_config must be positive')
    run_dir = Path(run_dir)
    path = run_dir / 'checkpoints' / ('eps_' + checkpoint)
    saved = json.loads(Path(str(path) + '.minigrid.json').read_text())
    if saved['checkpoint'] != path.name:
        raise ValueError('Checkpoint and configuration-bank sidecar do not match')
    bank = saved['configuration_bank']
    source_bank = json.loads(json.dumps(bank))
    if total_episodes is not None:
        if total_episodes < configuration_slots or configuration_slots < 1:
            raise ValueError("Episode budget must cover all configuration slots")
        for task in bank['tasks'].values():
            selected = task['heldout'] or task['train']
            task['heldout'] = [dict(selected[i % len(selected)]) for i in range(configuration_slots)]

    config = json.loads((run_dir / 'config.json').read_text())
    fields_allowed = {field.name for field in fields(TSILArgs)}
    all_results = {}
    for split in (['heldout'] if total_episodes is not None else ['train', 'heldout']):
        tasks = [t for t in config['tasks'] if bank['tasks'][str(t)][split]]
        if not tasks:
            all_results[split] = {}
            continue
        args = TSILArgs(**{k: v for k, v in config.items() if k in fields_allowed})
        args.tasks = tasks
        args.task_counts = [len(bank['tasks'][str(t)][split]) for t in tasks]
        args.seed, args.deterministic = seed, False
        env = TSILMiniGridEnv(args, device, configuration_bank=bank, configuration_split=split)
        # Shared checkpoints must preserve their training time-feature scale on task subsets.
        env.max_eps_time = saved['time_feature_scale']
        env.configuration_split = 'evaluation'
        try:
            metadata = []
            for i, (task, record) in enumerate(zip(env.task_indices.cpu().tolist(), env.configuration_records)):
                if split == 'train':
                    target, source = record['target'], dict(neighbors=[], fallback=None, source='saved_training_target')
                else:
                    target, source = infer_target(record, source_bank['tasks'][str(task)], k=5)
                env.obs_range['time2end'][1][i] = target
                env.obs_range['time2end'][0][i] = -target
                metadata.append(dict(config_id=record['id'], seed=record['seed'], task_id=task,
                                     target=target, target_source=source, episodes=[]))
            quotas = [episodes_per_config] * len(metadata)
            if total_episodes is not None:
                for task in tasks:
                    ids = [i for i, row in enumerate(metadata) if row['task_id'] == task]
                    base, extra = divmod(total_episodes, len(ids))
                    for j, i in enumerate(ids): quotas[i] = base + int(j < extra)
            for row, quota in zip(metadata, quotas):
                row['requested_episodes'] = quota
            agent = MiniGridAgent(env, args, device=device)
            agent.load_checkpoint(str(path), map_location=device, strict=True)
            agent.set_mode('eval')
            torch.manual_seed(seed)
            np.random.seed(seed)
            obs = env.reset()['obs']
            returns = np.zeros(env.num_envs)
            lengths = np.zeros(env.num_envs, dtype=int)
            with torch.no_grad():
                for _ in range(max(quotas) * (int(env.episode_limits.max()) + 1) + 2):
                    valid = ~env._needs_reset.copy()
                    actions, _ = agent.get_action_and_value(obs, action_only=True, update=False)
                    observation, _, done, info = env.step(actions)
                    obs = observation['obs']
                    returns += info['native_reward'].cpu().numpy()
                    lengths += valid.astype(int)
                    for i in done.nonzero(as_tuple=True)[0].cpu().tolist():
                        if len(metadata[i]['episodes']) < quotas[i]:
                            metadata[i]['episodes'].append(dict(success=bool(info['success'][i]),
                                native_return=float(returns[i]), length=int(lengths[i]),
                                horizon=int(env.episode_limits[i].item())))
                        returns[i] = lengths[i] = 0
                    if all(len(row['episodes']) == quotas[i] for i, row in enumerate(metadata)):
                        break
            if not all(len(row['episodes']) == quotas[i] for i, row in enumerate(metadata)):
                raise RuntimeError('Configuration evaluation did not finish')
            all_results[split] = {}
            for task in tasks:
                rows = [r for r in metadata if r['task_id'] == task]
                episodes = [e for r in rows for e in r['episodes']]
                grouped = {}
                for row in rows:
                    grouped.setdefault(row['config_id'], []).extend(row['episodes'])
                config_success = [np.mean([e['success'] for e in group]) for group in grouped.values()]
                config_completion = [float(normalized_completion_time(
                    [e['success'] for e in group], [e['length'] for e in group],
                    [e['horizon'] for e in group]).mean()) for group in grouped.values()]
                lengths_success = [e['length'] for e in episodes if e['success']]
                completion = normalized_completion_time(
                    [e['success'] for e in episodes],
                    [e['length'] for e in episodes],
                    [e['horizon'] for e in episodes],
                )
                has_disjoint_support = bool(source_bank['tasks'][str(task)]['heldout'])
                all_results[split][str(task)] = dict(
                    success_rate=float(np.mean(config_success)),
                    mean_return=float(np.mean([e['native_return'] for e in episodes])),
                    mean_episode_length=float(np.mean([e['length'] for e in episodes])),
                    successful_episode_length=float(np.mean(lengths_success)) if lengths_success else None,
                    normalized_completion_time=float(np.mean(config_completion)),
                    configurations=rows, episodes=episodes,
                    unique_configurations=len({r['config_id'] for r in rows}),
                    configuration_slots=len(rows),
                    repeated_configuration_slots=len(rows) - len({r['config_id'] for r in rows}),
                    configuration_provenance=(
                        'disjoint_evaluation_bank'
                        if split == 'heldout' and has_disjoint_support
                        else 'repeated_finite_support'
                    ))
        finally:
            env.close()
    # A fixed task without a disjoint split is explicitly labeled, not invented.
    primary = {str(t): all_results['heldout'].get(str(t)) or all_results.get('train', {}).get(str(t))
               for t in config['tasks']}
    # Exact budgets may allocate different quotas (e.g. 62/63 for 2,000/32).
    # Do not report the unused episodes_per_config default as the executed quota.
    uniform_quota = episodes_per_config
    if total_episodes is not None:
        uniform_quota = (total_episodes // configuration_slots
                         if total_episodes % configuration_slots == 0 else None)
    return dict(checkpoint=checkpoint, protocol='fixed_configurations_knn5', seed=seed,
        stochastic=True, episodes_per_configuration=uniform_quota,
        episode_budget_mode='exact_per_task' if total_episodes is not None else 'per_configuration',
        total_episodes_per_task=total_episodes, seen=all_results.get('train', {}), heldout=all_results['heldout'], per_task=primary,
        success_rate=float(np.mean([r['success_rate'] for r in primary.values()])),
        mean_return=float(np.mean([r['mean_return'] for r in primary.values()])),
        normalized_completion_time=float(np.mean([
            r['normalized_completion_time'] for r in primary.values()
        ])),
        evaluation_label='evaluation',
        tasks_with_disjoint_configuration_support=[
            int(t) for t in primary if source_bank['tasks'][t]['heldout']
        ],
        tasks_with_repeated_finite_support=[
            int(t) for t in primary if not source_bank['tasks'][t]['heldout']
        ],
        target_bank_source=str(path) + '.minigrid.json')
