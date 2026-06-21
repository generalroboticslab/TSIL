"""Trajectory replay plotting helpers used by core evaluation."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from matplotlib import pyplot as plt


def plot_replay_results(
    task_id: int,
    bucket_label: str,
    rank: int,
    record: Dict[str, Any],
    replay: Dict[str, Any],
    save_path: str,
) -> None:
    train_reward = np.asarray(record["trajectory"]["reward_raw"], dtype=np.float32)
    replay_reward = np.asarray(replay["reward"], dtype=np.float32)
    train_cumulative = np.cumsum(train_reward)
    replay_cumulative = np.asarray(replay["cumulative_reward"], dtype=np.float32)
    action = np.asarray(record["trajectory"]["action"], dtype=np.float32)
    steps = np.arange(len(train_reward))
    replay_steps = np.arange(len(replay_reward))

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    ax = axes.ravel()

    ax[0].plot(steps, train_reward, label="Train", linewidth=2)
    ax[0].plot(replay_steps, replay_reward, label="Replay", linewidth=2, linestyle="--")
    ax[0].set_title("Step Reward")
    ax[0].set_xlabel("Step")
    ax[0].set_ylabel("Reward")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(steps, train_cumulative, label="Train", linewidth=2)
    ax[1].plot(replay_steps, replay_cumulative, label="Replay", linewidth=2, linestyle="--")
    ax[1].set_title("Cumulative Reward")
    ax[1].set_xlabel("Step")
    ax[1].set_ylabel("Return")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    for dim in range(action.shape[1]):
        ax[2].plot(np.arange(action.shape[0]), action[:, dim], linewidth=1.5, label=f"a{dim}")
    ax[2].set_title("Action Trace")
    ax[2].set_xlabel("Step")
    ax[2].set_ylabel("Action")
    ax[2].grid(True, alpha=0.3)
    if action.shape[1] <= 6:
        ax[2].legend(ncol=2)

    scene_linvel = np.asarray(replay["scene_linvel"], dtype=np.float32)
    scene_linvel_lim = np.asarray(replay["scene_linvel_lim"], dtype=np.float32)
    ax[3].plot(replay_steps, scene_linvel, label="Scene Instability", linewidth=2)
    ax[3].plot(replay_steps, scene_linvel_lim, label="Threshold", linewidth=2, linestyle="--")
    ax[3].set_title("Instability Replay Trace")
    ax[3].set_xlabel("Step")
    ax[3].set_ylabel("Value")
    ax[3].legend()
    ax[3].grid(True, alpha=0.3)

    ax[4].plot(replay_steps, replay["real_cur_time"], label="Elapsed", linewidth=2)
    ax[4].plot(replay_steps, replay["real_time2end"], label="GT Time to End", linewidth=2)
    ax[4].plot(replay_steps, replay["observed_time2end"], label="Observed Time to End", linewidth=2, linestyle="--")
    ax[4].set_title("Time Trace")
    ax[4].set_xlabel("Step")
    ax[4].set_ylabel("Time (s)")
    ax[4].legend()
    ax[4].grid(True, alpha=0.3)

    summary = record["summary"]
    summary_lines = [
        f"Task: {task_id}",
        f"Bucket: {bucket_label}",
        f"Rank: {rank}",
        f"Train success: {bool(summary['success'])}",
        f"Replay success: {replay['final_success']}",
        f"Train return: {float(summary['episode_return_raw']):.3f}",
        f"Replay return: {float(replay['episode_return_raw']):.3f}",
        f"Train length: {int(summary['episode_length'])}",
        f"Replay length: {int(replay['episode_length'])}",
        f"Train eps_time: {float(summary['eps_time']):.3f}",
        f"Train eps_time_p: {float(summary['eps_time_p']):.3f}",
        f"Train max inst: {float(summary['eps_max_scevel']):.3f}",
        f"Train sum inst: {float(summary['eps_sum_inst']):.3f}",
    ]
    ax[5].axis("off")
    ax[5].text(0.0, 1.0, "\n".join(summary_lines), va="top", ha="left", fontsize=11)

    fig.suptitle(f"Trajectory Replay | Task {task_id} | {bucket_label} | Rank {rank}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_compare_results(
    task_id: int,
    rank: int,
    compare_payloads: Dict[str, Dict[str, Any]],
    save_path: str,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    bucket_names = list(compare_payloads.keys())

    for col, bucket_name in enumerate(bucket_names):
        record = compare_payloads[bucket_name]["record"]
        replay = compare_payloads[bucket_name]["replay"]
        summary = record["summary"]
        steps = np.arange(len(record["trajectory"]["reward_raw"]))
        replay_steps = np.arange(len(replay["reward"]))

        axes[0, col].plot(steps, record["trajectory"]["reward_raw"], label="Train", linewidth=2)
        axes[0, col].plot(replay_steps, replay["reward"], label="Replay", linewidth=2, linestyle="--")
        axes[0, col].set_title(f"{bucket_name} Step Reward")
        axes[0, col].legend()
        axes[0, col].grid(True, alpha=0.3)

        axes[1, col].plot(replay_steps, replay["scene_linvel"], label="Scene Instability", linewidth=2)
        axes[1, col].plot(replay_steps, replay["scene_linvel_lim"], label="Threshold", linewidth=2, linestyle="--")
        axes[1, col].set_title(f"{bucket_name} Instability")
        axes[1, col].legend()
        axes[1, col].grid(True, alpha=0.3)

        axes[2, col].axis("off")
        axes[2, col].text(
            0.0,
            1.0,
            "\n".join([
                f"Bucket: {bucket_name}",
                f"Rank: {rank}",
                f"Train success: {bool(summary['success'])}",
                f"Replay success: {replay['final_success']}",
                f"Train return: {float(summary['episode_return_raw']):.3f}",
                f"Replay return: {float(replay['episode_return_raw']):.3f}",
                f"Train length: {int(summary['episode_length'])}",
                f"Replay length: {int(replay['episode_length'])}",
            ]),
            va="top",
            ha="left",
            fontsize=11,
        )

    fig.suptitle(f"Trajectory Compare | Task {task_id} | Rank {rank}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
