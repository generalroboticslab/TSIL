"""Diagnostic success-state validation for MTBench Franka tasks.

This script checks the task success predicates, not policy performance.  It
creates the normal training environment, lets reset settle, then asks two
questions:

1. Does the reset state already report success?
2. If the task state is placed at an oracle goal state, does compute_reward()
   report success?
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Sequence

import isaacgym  # noqa: F401 - IsaacGym must be imported before torch.
import torch

from MTBench.isaacgymenvs.tasks.franka.vec_task.franka_base import TASK_IDX_TO_NAME
from projects.TSIL.diagnostics import reset_validation as reset_debug


def _task_indices(env) -> torch.Tensor:
    if hasattr(env, "extras") and "task_indices" in env.extras:
        return env.extras["task_indices"].to(env.device)
    return env.task_indices.to(env.device)


def _task_env_ids(env, task_id: int) -> torch.Tensor:
    return (_task_indices(env) == task_id).nonzero(as_tuple=False).flatten()


def _set_task_tensor(kwargs: Dict[str, Any], key: str, value: torch.Tensor) -> None:
    kwargs[key] = value.detach().clone()


def _set_open_gripper_dofs(env) -> None:
    franka_dof_ids = env.franka_dof_idx.view(env.num_envs, -1)
    finger_ids = franka_dof_ids[:, -2:].flatten()
    env._dof_state[finger_ids, 0] = 0.04
    env._dof_state[finger_ids, 1] = 0.0


def _set_fingers_around(env, env_ids: torch.Tensor, center: torch.Tensor, gap: float = 0.04) -> None:
    left = center.detach().clone()
    right = center.detach().clone()
    left[:, 1] += gap / 2.0
    right[:, 1] -= gap / 2.0
    env.world_states["eef_lf_pos"][env_ids] = left
    env.world_states["eef_rf_pos"][env_ids] = right


def _apply_generic_oracle(env) -> None:
    target = env.target_pos.detach().clone()
    env.obj_pos[:] = target
    env.obj_init_pos[:] = target
    env.obj_init_pos[:, 2] -= 0.10
    if env.tcp_init is not None:
        env.tcp_init[:] = target + torch.tensor([0.15, 0.0, 0.0], device=env.device)
    _set_fingers_around(env, torch.arange(env.num_envs, device=env.device), target)
    _set_open_gripper_dofs(env)


def _apply_task_overrides(env, task_ids: Sequence[int]) -> Dict[int, str]:
    notes: Dict[int, str] = {}
    for task_id in task_ids:
        env_ids = _task_env_ids(env, task_id)
        if env_ids.numel() == 0:
            continue
        task_name = TASK_IDX_TO_NAME[task_id]
        target = env.target_pos[env_ids].detach().clone()
        kwargs = env.specialized_kwargs.get(task_name, {})

        if task_name == "assembly":
            center = target.detach().clone()
            center[:, 2] -= 0.01
            _set_task_tensor(kwargs, "round_nut_center_pos", center)
            quat = torch.zeros((env_ids.numel(), 4), device=env.device)
            quat[:, 3] = 1.0
            _set_task_tensor(kwargs, "round_nut_center_quat", quat)
            env.obj_pos[env_ids] = center
            _set_fingers_around(env, env_ids, center)
            notes[task_id] = "round_nut_center placed just below peg target"

        elif task_name == "box_close":
            _set_task_tensor(kwargs, "lid_base_pos", target)
            _set_task_tensor(kwargs, "obj_rot", torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=env.device).repeat(env_ids.numel(), 1))
            notes[task_id] = "lid point placed at target"

        elif task_name == "disassemble":
            _set_task_tensor(kwargs, "round_nut_center_pos", target)
            env.obj_pos[env_ids] = target
            env.obj_pos[env_ids, 2] += 0.01
            quat = torch.zeros((env_ids.numel(), 4), device=env.device)
            quat[:, 3] = 1.0
            _set_task_tensor(kwargs, "round_nut_quat", quat)
            notes[task_id] = "observed nut point placed above target z"

        elif task_name == "hammer":
            hammer_pos = target.detach().clone()
            hammer_pos[:, 0] -= 0.07
            hammer_pos[:, 1] += 0.16
            env.obj_pos[env_ids] = hammer_pos
            env.obj_init_pos[env_ids] = hammer_pos
            env.obj_init_pos[env_ids, 2] -= 0.10
            _set_fingers_around(env, env_ids, hammer_pos)
            _set_task_tensor(kwargs, "hammer_block_dof_pos", torch.full((env_ids.numel(), 1), 0.10, device=env.device))
            quat = torch.tensor([[0.0, 0.0, -0.7068252, 0.7073883]], device=env.device).repeat(env_ids.numel(), 1)
            _set_task_tensor(kwargs, "hammer_rot", quat)
            notes[task_id] = "hammer head placed on nail and block DOF pressed"

        elif task_name == "lever_pull":
            _set_task_tensor(kwargs, "lever_dof_pos", torch.full((env_ids.numel(), 1), -math.pi / 2.0, device=env.device))
            _set_task_tensor(kwargs, "lever_pos_init", env.obj_init_pos[env_ids].detach().clone())
            notes[task_id] = "lever DOF set to vertical success angle"

        elif task_name == "peg_insert_side":
            _set_task_tensor(kwargs, "peg_head_pos", target)
            init = target.detach().clone()
            init[:, 0] -= 0.10
            _set_task_tensor(kwargs, "peg_head_pos_init", init)
            notes[task_id] = "peg_head_pos placed at target"

        elif task_name == "stick_push":
            stick = target.detach().clone()
            stick[:, 2] += 0.02
            env.obj_pos[env_ids] = stick
            env.obj_init_pos[env_ids] = stick
            env.obj_init_pos[env_ids, 2] -= 0.10
            _set_fingers_around(env, env_ids, stick)
            _set_task_tensor(kwargs, "stick_init_pos", env.obj_init_pos[env_ids].detach().clone())
            _set_task_tensor(kwargs, "thermos_pos", target)
            _set_task_tensor(kwargs, "thermos_dof_pos", torch.zeros((env_ids.numel(), 2), device=env.device))
            notes[task_id] = "stick lifted in gripper and thermos center placed at target"

        elif task_name == "stick_pull":
            stick = target.detach().clone()
            stick[:, 2] += 0.03
            handle = target.detach().clone()
            end = handle.detach().clone()
            end[:, 1] -= 0.01
            env.obj_pos[env_ids] = stick
            env.obj_init_pos[env_ids] = stick
            env.obj_init_pos[env_ids, 2] -= 0.10
            _set_fingers_around(env, env_ids, stick)
            _set_task_tensor(kwargs, "stick_init_pos", env.obj_init_pos[env_ids].detach().clone())
            _set_task_tensor(kwargs, "thermos_pos", target)
            _set_task_tensor(kwargs, "thermos_dof_pos", torch.zeros((env_ids.numel(), 2), device=env.device))
            _set_task_tensor(kwargs, "thermos_insertion_pos", handle)
            _set_task_tensor(kwargs, "thermos_insertion_pos_init", handle)
            _set_task_tensor(kwargs, "stick_end_pos", end)
            notes[task_id] = "stick held and inserted into handle at target"

    return notes


def _compute_reward(env) -> None:
    env.reset_buf.zero_()
    env.success_buf.zero_()
    env.progress_buf.fill_(2)
    env.actions = reset_debug._zero_actions(env).to(env.device)
    env.compute_task_reward(env.actions)


def _settle(env, steps: int) -> None:
    for _ in range(steps):
        env.step(reset_debug._zero_actions(env))


def _collect_rows(env, task_ids: Sequence[int], oracle_notes: Dict[int, str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    task_index = _task_indices(env)
    for task_id in task_ids:
        env_ids = (task_index == task_id).nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            continue
        success = env.success_buf[env_ids].detach().clone().bool()
        reset = env.reset_buf[env_ids].detach().clone().bool()
        reward = env.rew_buf[env_ids].detach().clone()
        obj_dist = torch.norm(env.obj_pos[env_ids] - env.target_pos[env_ids], dim=-1)
        rows.append(
            {
                "task_id": task_id,
                "task_name": TASK_IDX_TO_NAME[task_id],
                "env_count": int(env_ids.numel()),
                "success_count": int(success.sum().item()),
                "reset_count": int(reset.sum().item()),
                "max_reward": float(reward.max().item()),
                "min_obj_target_dist": float(obj_dist.min().item()),
                "max_obj_target_dist": float(obj_dist.max().item()),
                "oracle_note": oracle_notes.get(task_id, "obj_pos/tcp placed at target"),
            }
        )
    return rows


def _rows_to_issues(rows: Sequence[Dict[str, Any]], phase: str) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for row in rows:
        if phase == "reset" and row["success_count"] > 0:
            issues.append(
                {
                    "phase": phase,
                    "task_id": row["task_id"],
                    "task_name": row["task_name"],
                    "kind": "success_at_reset",
                    "value": row["success_count"],
                    "threshold": 0,
                    "detail": f"{row['success_count']}/{row['env_count']} envs reported success before oracle goal injection",
                }
            )
        if phase == "oracle" and row["success_count"] != row["env_count"]:
            issues.append(
                {
                    "phase": phase,
                    "task_id": row["task_id"],
                    "task_name": row["task_name"],
                    "kind": "oracle_goal_not_successful",
                    "value": row["success_count"],
                    "threshold": row["env_count"],
                    "detail": f"{row['success_count']}/{row['env_count']} envs reported success after oracle goal injection",
                }
            )
    return issues


def _print_summary(title: str, rows: Sequence[Dict[str, Any]]) -> None:
    print(f"\n{title}")
    reset_debug._print_rows(
        ["tid", "task", "envs", "succ", "reset", "max_rew", "obj_d_min", "obj_d_max", "note"],
        [
            [
                row["task_id"],
                row["task_name"],
                row["env_count"],
                row["success_count"],
                row["reset_count"],
                reset_debug._compact_float(row["max_reward"]),
                reset_debug._compact_float(row["min_obj_target_dist"]),
                reset_debug._compact_float(row["max_obj_target_dist"]),
                row["oracle_note"],
            ]
            for row in rows
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="all", help="Task ids, ranges, comma list, or 'all'.")
    parser.add_argument("--envs-per-task", type=int, default=1)
    parser.add_argument("--episode-length", type=int, default=80)
    parser.add_argument("--settle-steps", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=reset_debug._str2bool, default=False)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--env-name", default=reset_debug.DEFAULT_ENV_NAME)
    parser.add_argument("--fixed", type=reset_debug._str2bool, default=True)
    parser.add_argument("--same-init-config-per-task", type=reset_debug._str2bool, default=False)
    parser.add_argument("--termination-on-success", type=reset_debug._str2bool, default=True)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--init-curri-ratio", type=float, default=1.0)
    parser.add_argument("--task-embedding-dim", type=int, default=len(TASK_IDX_TO_NAME))
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--fail-on-issue", action="store_true")
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    cli.contact_report = False
    task_ids = reset_debug._parse_tasks(cli.tasks)
    args = reset_debug._build_args(cli, task_ids)

    print(
        "Success validation config: "
        f"tasks={task_ids}, envs={args.num_envs}, settle_steps={cli.settle_steps}, "
        f"fixed={args.fixed}, device={args.sim_device}"
    )

    env = reset_debug._make_env(args)
    try:
        env.reset_all()
        _settle(env, cli.settle_steps)
        reset_rows = _collect_rows(env, task_ids, {})
        reset_issues = _rows_to_issues(reset_rows, "reset")

        _apply_generic_oracle(env)
        oracle_notes = _apply_task_overrides(env, task_ids)
        _compute_reward(env)
        oracle_rows = _collect_rows(env, task_ids, oracle_notes)
        oracle_issues = _rows_to_issues(oracle_rows, "oracle")
    finally:
        if hasattr(env, "close"):
            env.close()

    _print_summary("Reset-state success check", reset_rows)
    _print_summary("Oracle-goal success check", oracle_rows)

    issues = reset_issues + oracle_issues
    if issues:
        print("\nIssues:")
        for issue in issues:
            print(
                f"- phase={issue['phase']} T{issue['task_id']} {issue['task_name']}: "
                f"{issue['kind']} value={issue['value']} threshold={issue['threshold']} ({issue['detail']})"
            )
    else:
        print("\nNo success-state validation issues found.")

    if cli.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(cli.output_json)), exist_ok=True)
        with open(cli.output_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": {
                        "tasks": task_ids,
                        "task_counts": args.task_counts,
                        "envs": args.num_envs,
                        "episode_length": cli.episode_length,
                        "settle_steps": cli.settle_steps,
                        "fixed": args.fixed,
                        "same_init_config_per_task": args.same_init_config_per_task,
                        "termination_on_success": args.termination_on_success,
                        "seed": args.seed,
                        "beta": args.beta,
                        "sim_device": args.sim_device,
                    },
                    "reset_rows": reset_rows,
                    "oracle_rows": oracle_rows,
                    "issues": issues,
                },
                f,
                indent=2,
            )
        print(f"\nWrote JSON report to {cli.output_json}")

    return 1 if issues and cli.fail_on_issue else 0


if __name__ == "__main__":
    raise SystemExit(main())
