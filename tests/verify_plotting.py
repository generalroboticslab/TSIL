"""Check report selection, metric units, and output-format contracts on CPU."""
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from core.plotting.primitives import _save_and_show
from core.plotting.train_data import _success_eps_time_fill_value
from projects.TSIL.reports.run_results import load_runs
from projects.TSIL.reports.curve_metrics import _curve_metrics


def main():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        runs = []
        for index in range(2):
            run = root / str(index)
            run.mkdir()
            config = dict(tasks=[10], method_name="TSIL", seed=index,
                          benchmark_backend="minigrid", episodeLength=120, dt=1/60,
                          num_envs=32, num_steps=32, total_timesteps=1024,
                          sil_freeze_replay_gap=bool(index))
            (run / "config.json").write_text(json.dumps(config))
            (run / "plot_metrics_history.jsonl").write_text(json.dumps({"misc/steps":1024})+'\n')
            assert _success_eps_time_fill_value(str(run)) == 120
            runs.append(run)
        for selection in [runs, [runs[0], runs[0]]]:
            try:
                load_runs(selection)
            except ValueError:
                pass
            else:
                raise AssertionError("Mixed protocols or duplicate runs must not pool silently")
        records, backend = load_runs(runs, ["recomputed", "frozen"])
        assert len(records) == 2 and backend == "minigrid"
        config = json.loads((runs[0] / "config.json").read_text())
        config['benchmark_backend'] = 'metaworld'
        (runs[0] / "config.json").write_text(json.dumps(config))
        assert _success_eps_time_fill_value(str(runs[0])) == 2
        try:
            load_runs(runs, ["A", "B"])
        except ValueError:
            pass
        else:
            raise AssertionError("Different benchmark units must not pool")
        curve = (np.array([0., 100.]), np.array([0., 1.]))
        metrics = _curve_metrics(curve, start_at_zero=True)
        assert np.isclose(metrics['auc_success'], 0.5)
        fig, ax = plt.subplots()
        ax.plot(*curve)
        path = root / "check.pdf"
        _save_and_show(fig, str(path), show=False)
        assert path.read_bytes().startswith(b'%PDF')
        assert not path.with_suffix('.png').exists()
        plt.close(fig)
    print('PASS: run selection, duplicate/protocol guards, time units, AUC, PDF-only output')


if __name__ == '__main__':
    main()
