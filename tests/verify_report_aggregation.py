"""Regression checks for task-balanced reports, axes, coverage and table outputs."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from projects.TSIL.figures import main_method_results as figures
from projects.TSIL.reports import paper_metrics as tables
from projects.TSIL.reports import stability_summary as stress
from projects.TSIL.reports import run_results
from core.training.logger import TrainingLogger


def observations():
    rows = []
    for method, task, values in [('A', 1, [1, 1, 1]), ('A', 2, [0]),
                                 ('B', 1, [.6]), ('B', 2, [.6])]:
        for seed, value in enumerate(values):
            rows.append(dict(method=method, label=method, task_id=task, seed=seed,
                             tail_best_success=value, auc_success=value,
                             steps_to_threshold=100, selected_success_eps_time=20,
                             success_episodes=100 * value, success_rate=value))
    return rows


def main():
    # MiniGrid uses decision steps, MetaWorld retains physical time as metadata.
    for backend, expected in [('minigrid', 600), ('metaworld', 10)]:
        metrics = SimpleNamespace(args=SimpleNamespace(benchmark_backend=backend, dt=1/60),
                                  global_update_iter=1, global_step=600, global_episodes=2)
        logged = TrainingLogger._base_metrics(None, metrics)
        assert logged['misc/steps'] == 600
        assert logged['misc/interaction_time'] == expected
    rows = observations()
    bars, _, n = figures._bar_values(rows, ['A', 'B'], 'tail_best_success', 'sem')
    summary = tables._summary_rows(rows, ['A', 'B'], ['A', 'B'], .8)
    assert np.allclose(bars, [.5, .6])
    assert [r['last10_best_success_mean'] for r in summary] == [.5, .6]
    assert [r['auc_success_mean'] for r in summary] == [.5, .6]
    assert n == [2, 2] and all(r['_success_n'] == 2 for r in summary)
    numeric = tables._numeric_summary_rows(rows, ['A', 'B'], ['A', 'B'], ['auc_success'])
    assert [r['auc_success_mean'] for r in numeric] == [.5, .6]
    # Metrics are measured before seed averaging, including nonlinear peak metrics.
    curve = lambda y: (np.array([0., 100.]), np.array(y, dtype=float))
    groups = {1: [curve([0, 1]), curve([1, 0])], 2: [curve([0, 0])]}
    assert stress.task_scalar_stats(groups, lambda c: max(c[1]))[0] == .5
    assert np.allclose(stress.average_task_curves(groups)[1], .25)

    args = SimpleNamespace(train_root=Path('/unused'), benchmark='mt15', task_ids=[1, 2],
                           x_key='misc/steps', min_target_steps_by_experiment={})
    def fake_curves(root, benchmark, experiment, stage, task_ids, *rest):
        return ([curve([1, 1])] if task_ids == [1] else []), int(task_ids == [1])
    with patch.object(stress, 'STRESS_EXPERIMENTS', [('noise', 'scratch', 'Noise', 'Noise', [(None, '0')])]), \
         patch.object(stress, 'METHODS', [('TSIL', 'TSIL', 'blue')]), \
         patch.object(stress, 'load_metric_curves', side_effect=fake_curves):
        partial, _, missing = stress.collect_stress(args)
        assert partial[0]['status'] == 'partial'
        assert partial[0]['task_count'] == 1 and '/T2:' in missing[0]
        assert stress.missing_table_cells(partial, [1, 2])
        complete = dict(partial[0], status='ok', task_ids='[1, 2]', task_count=2)
        assert not stress.missing_table_cells([complete], [1, 2])
        assert stress.missing_table_cells([], [1, 2])
        assert stress.missing_table_cells([dict(complete, task_ids='')], [1, 2])

    # Rendering both panels must use the supplied step axis for either benchmark.
    settings = SimpleNamespace(methods=['A'], task_ids=[1], bar_error='sem',
                               x_key='misc/steps', figure_dir=Path('/unused'), format='pdf')
    captured = []
    def capture(fig, path):
        captured.append(fig)
    with patch.object(figures, '_aggregate_method_curve', return_value=(np.array([600., 1200.]), np.array([.5, .6]), np.zeros(2))), \
         patch.object(figures, '_save', side_effect=capture):
        figures.plot_main_training_curves(settings, {}, ['blue'], ['A'])
    for axis in captured[0].axes:
        assert axis.get_xlabel() == 'Environment steps'
        assert np.allclose(axis.lines[0].get_xdata(), [600, 1200])
    plt.close(captured[0])

    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp)
        args = SimpleNamespace(write_table=True, write_signal_table=True, write_sil_table=True,
                               table_dir=output, table_prefix='metrics', methods=['A', 'B'],
                               legends=['A', 'B'], threshold=.8)
        signal = [dict(r, **{m: .5 for m in tables.SIGNAL_METRICS}) for r in rows]
        replay = [dict(r, archive_best_eps_time=20, positive_gap_nll_50pct=.5) for r in rows]
        with patch.object(tables, 'parse_args', return_value=args), \
             patch.object(tables, '_collect_run_rows', return_value=rows), \
             patch.object(tables, '_collect_signal_rows', return_value=signal), \
             patch.object(tables, '_collect_sil_rows', return_value=replay):
            tables.main()
        files = list(output.rglob('*.*'))
        assert len(files) == 9, files
        assert 'auc_success_mean' in (output/'summary_csv/metrics_summary.csv').read_text()
        assert 'fast_success_mean' in (output/'summary_csv/metrics_signals_summary.csv').read_text()
        assert 'archive_best_eps_time_mean' in (output/'summary_csv/metrics_replay_summary.csv').read_text()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        runs = []
        for index, task in enumerate([1, 2, 1, 2]):
            run = root / str(index)
            run.mkdir()
            config = dict(tasks=[task], seed=0, method_name='TSIL', benchmark_backend='minigrid',
                          episodeLength=120, dt=1/60, num_envs=1, num_steps=1,
                          total_timesteps=200)
            (run/'config.json').write_text(json.dumps(config))
            records = [dict(iteration=step, **{'misc/steps':step, 'reward/success':.5,
                       'misc/success_episodes':step/2, 'signal/success_eps_time':30})
                       for step in [100, 200]]
            (run/'plot_metrics_history.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
            runs.append(run)
        try:
            run_results.load_runs(runs[:3], ['A', 'A', 'B'])
        except ValueError as exc:
            assert 'same tasks' in str(exc)
        else:
            raise AssertionError('Mismatched tasks were accepted')
        run_results.load_runs(runs, ['A', 'A', 'B', 'B'])
        try:
            run_results.load_runs([runs[0], runs[2]], ['A', 'A'])
        except ValueError as exc:
            assert 'Duplicate task/seed' in str(exc)
        else:
            raise AssertionError('Duplicate seed measurements were accepted')
        evaluation = dict(checkpoint='last', per_task={'1':dict(success_rate=.5, episodes=[{}]*32)})
        (runs[0]/'minigrid_evaluation.json').write_text(json.dumps(evaluation))
        run_results.generate(SimpleNamespace(run_dirs=runs[:2], labels=['A', 'A'],
                                            output=root/'partial', checkpoint='last'))
        manifest = json.loads((root/'partial/sources.json').read_text())
        assert manifest['evaluations_missing'] == 1
        assert manifest['missing_evaluations'][0]['task'] == 2
        assert not manifest['evaluation_figure_complete']
        assert not (root/'partial/figures/evaluation_success.pdf').exists()
        assert (root/'partial/tables/evaluation_runs.csv').exists()
    print('PASS: seed-to-task means, nonlinear metrics, missing-task coverage, step axes, three distinct table sets')


if __name__ == '__main__':
    main()
