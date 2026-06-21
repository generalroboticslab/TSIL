"""Synthetic adaptive temporal-target learning-signal figure helper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.TSIL.figures.signal_landscape import plot_learning_signal_combined


METHODS = (
    ("PPO_ATTL", "ATTL"),
    ("PPO_FTTL", "FTTL"),
    ("PPO_IHD2S", "D2S IH"),
    ("PPO_IHSC", "Step-cost IH"),
    ("PPO", "IH"),
)


def _episode_records(method_idx: int, sample_idx: int, num_episodes: int, seed: int):
    rng = np.random.default_rng(seed + 97 * method_idx + 1009 * sample_idx)
    method_shift = np.linspace(0.46, -0.34, len(METHODS))[method_idx]
    time_bias = np.linspace(-0.13, 0.15, len(METHODS))[method_idx]
    records = []
    for idx in range(num_episodes):
        frac = idx / max(num_episodes - 1, 1)
        phase = int(np.floor(frac * 4.0))
        phase = min(phase, 3)
        phase_center = (phase + 0.5) / 4.0

        eps_time_norm = np.clip(
            rng.beta(1.7 + 0.25 * phase, 2.4 - 0.18 * method_idx)
            + time_bias
            + 0.10 * np.sin(2.0 * np.pi * (frac + 0.13 * method_idx)),
            0.015,
            0.985,
        )
        dense_return = np.clip(
            0.20
            + 0.66 * rng.beta(1.3 + 0.18 * method_idx, 1.8)
            + 0.18 * phase_center
            - 0.24 * eps_time_norm
            + rng.normal(0.0, 0.055),
            0.0,
            1.0,
        )
        success_score = (
            1.35 * dense_return
            - 1.05 * eps_time_norm
            + method_shift
            + 0.26 * np.sin(2.0 * np.pi * frac + 0.6 * method_idx)
            + rng.normal(0.0, 0.22)
        )
        success = bool(success_score > 0.18)
        if dense_return > 0.88 and eps_time_norm < 0.28:
            success = True
        if dense_return < 0.18 and eps_time_norm > 0.72:
            success = False

        mass = rng.gamma(1.7, 0.9) * (
            0.42 + 1.45 * float(success) * (1.0 - eps_time_norm) + 0.55 * dense_return
        )
        records.append({
            "iteration": idx + 1 + sample_idx * (num_episodes + 10),
            "episodes": idx + 1,
            "episode_id": idx + 1,
            "eps_time": float(100.0 * eps_time_norm),
            "max_eps_time": 100.0,
            "dense_return": float(dense_return),
            "positive_adv_mass_update": float(max(mass, 1e-5)),
            "success": success,
            "episode_step_count": 2,
        })
    return records


def _write_history(root: Path, task_id: str, method: str, sample_idx: int, records):
    traj_dir = root / f"T{task_id}" / method / f"synthetic_run{sample_idx}" / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    history_path = traj_dir / "training_signal_history.jsonl"
    episode_path = traj_dir / "training_episode_signal_history.jsonl"
    history_path.write_text(json.dumps({
        "iteration": int(records[-1]["iteration"]) if records else 0,
        "episode_history_file": episode_path.name,
    }) + "\n")
    episode_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return history_path


def generate_synthetic_histories(output_root: Path, task_id: str, num_episodes: int, summary_samples: int, seed: int):
    map_histories = []
    map_labels = []
    summary_histories = []
    summary_labels = []
    for method_idx, (method, label) in enumerate(METHODS):
        records = _episode_records(method_idx, 0, num_episodes, seed)
        map_histories.append(_write_history(output_root / "map", task_id, method, 0, records))
        map_labels.append(label)
        for sample_idx in range(summary_samples):
            summary_task = str(sample_idx)
            sample_records = _episode_records(method_idx, sample_idx, num_episodes, seed + 13)
            summary_histories.append(_write_history(output_root / "summary", summary_task, method, sample_idx, sample_records))
            summary_labels.append(label)
    manifest = {
        "map_histories": [str(path) for path in map_histories],
        "map_labels": map_labels,
        "summary_histories": [str(path) for path in summary_histories],
        "summary_labels": summary_labels,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path, manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a fast synthetic group2 learning-signal plot.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--task-id", type=str, default="28")
    parser.add_argument("--num-episodes", type=int, default=3600)
    parser.add_argument("--summary-samples", type=int, default=3)
    parser.add_argument("--num-bins", type=int, default=12)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--last-frac", type=float, default=1.0)
    parser.add_argument("--summary-last-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260528)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path, manifest = generate_synthetic_histories(
        output_root=args.output_root,
        task_id=args.task_id,
        num_episodes=args.num_episodes,
        summary_samples=args.summary_samples,
        seed=args.seed,
    )
    plot_learning_signal_combined(
        signal_history_path=manifest["map_histories"],
        labels=manifest["map_labels"],
        save_path=str(args.save_path),
        show=False,
        num_bins=args.num_bins,
        last_frac=args.last_frac,
        min_count=args.min_count,
        summary_signal_history_path=manifest["summary_histories"],
        summary_labels=manifest["summary_labels"],
        summary_last_frac=args.summary_last_frac,
    )
    print(f"[Synthetic data] {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
