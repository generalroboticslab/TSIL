# Figures and tables

Run commands from `TSIL/`. `RUN_DIR` is a completed training folder containing
`config.json` and `plot_metrics_history.jsonl`.

## Training and evaluation reports

```bash
python -m projects.TSIL.plot runs --run-dirs RUN_DIR --labels TSIL --output results/TSIL/paper_artifacts/my_report
```

Pass multiple run folders and labels to compare methods. Reuse a label to
combine seeds. Generate separate reports for MetaWorld and MiniGrid, using
a new output folder for each report.

Outputs include PDF figures and CSV/Markdown tables for success, AUC,
completion time and success count. Training curves use environment steps.
Metrics are averaged across seeds within each task, then equally across tasks.
Completion time uses seconds for MetaWorld and decision steps for MiniGrid.

Evaluation figures require evaluations for all selected runs. Available
measurements and missing evaluations are recorded in the tables and
`sources.json`. Add `--checkpoint last` to use final-checkpoint evaluations.

## Learning signals and replay

These plots require the corresponding histories in `RUN_DIR/trajectories/`:

```bash
python -m projects.TSIL.plot --task LearningSignalCombined --signal_history RUN_DIR/trajectories/training_signal_history.jsonl --legends TSIL --save_path results/TSIL/paper_artifacts/signals.pdf
python -m projects.TSIL.plot --task SilRevisitMechanism --signal_history RUN_DIR/trajectories/training_signal_history.jsonl --legends TSIL --save_path results/TSIL/paper_artifacts/replay.pdf
```

Other views: `LearningSignalMap`, `LearningSignalSummary`,
`LearningSignalSurface`, and `SilRevisitOutcome`.

## Disturbances and sweeps

Use the report command above with a distinct label for each method/condition.
For grouped experiment folders, additional commands are available:

```bash
python -m projects.TSIL.reports.stability_summary --help
python -m projects.TSIL.reports.paper_metrics --help
python -m projects.TSIL.figures.main_method_results --help
```

These commands read `<benchmark>/<experiment>/<stage>/T<task>/<method>/<timestamp>/<run>/`
under `--train-root`. The stability report also accepts `--input-table` with a
saved summary CSV; `--strict` rejects missing results.
