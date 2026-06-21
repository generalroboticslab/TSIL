"""Lightweight reset validation for MTBench Franka tasks.

This script is intentionally diagnostic-only: it creates the same IsaacGym
environment stack used by training, runs a few reset cycles with simple actions,
and reports suspicious reset behavior without modifying task code.

Example:
    PYTHONPATH="$(pwd)" \
    conda run -n tsil python -m projects.TSIL.diagnostics.reset_validation \
        --tasks 28 --envs-per-task 16 --episodes 3 --episode-length 30

All tasks, small sweep:
    PYTHONPATH="$(pwd)" \
    conda run -n tsil python -m projects.TSIL.diagnostics.reset_validation \
        --tasks all --envs-per-task 2 --episodes 3 --episode-length 20
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import isaacgym  # noqa: F401 - IsaacGym must be imported before torch.
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from core.training.args import Args
from MTBench import isaacgymenvs
from MTBench.isaacgymenvs.tasks.franka.vec_task import task_fns
from MTBench.isaacgymenvs.tasks.franka.vec_task.franka_base import TASK_IDX_TO_NAME


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HYDRA_CONFIG_DIR = os.path.join(REPO_ROOT, "MTBench", "isaacgymenvs", "cfg")
DEFAULT_ENV_NAME = Args().env_name


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def _parse_tasks(raw: str) -> List[int]:
    raw = raw.strip()
    if raw.lower() == "all":
        return _runnable_task_ids()

    task_ids: List[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise argparse.ArgumentTypeError(f"invalid task range: {chunk}")
            task_ids.extend(range(start, end + 1))
        else:
            task_ids.append(int(chunk))

    runnable = set(_runnable_task_ids())
    unknown = sorted(set(task_ids) - set(TASK_IDX_TO_NAME))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown task ids: {unknown}")
    missing_impl = sorted(set(task_ids) - runnable)
    if missing_impl:
        names = [TASK_IDX_TO_NAME[tid] for tid in missing_impl]
        raise argparse.ArgumentTypeError(f"task ids are mapped but not exported by task_fns: {list(zip(missing_impl, names))}")
    if len(set(task_ids)) != len(task_ids):
        raise argparse.ArgumentTypeError(f"duplicate task ids: {task_ids}")
    return task_ids


def _runnable_task_ids() -> List[int]:
    return sorted(tid for tid, name in TASK_IDX_TO_NAME.items() if hasattr(task_fns, name))


def _compact_float(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e3:
        return f"{value:.2e}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _compact_vec(values: torch.Tensor) -> str:
    return "[" + ",".join(_compact_float(float(v)) for v in values.detach().cpu().tolist()) + "]"


def _compact_list(values: Sequence[float]) -> str:
    return "[" + ",".join(_compact_float(float(v)) for v in values) + "]"


def _print_rows(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    widths = [len(str(h)) for h in headers]
    for row in rows:
        widths = [max(width, len(str(value))) for width, value in zip(widths, row)]
    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * width for width in widths]))
    for row in rows:
        print(fmt.format(*row))


def _build_args(cli: argparse.Namespace, task_ids: Sequence[int]) -> Args:
    args = Args()
    args.tasks = list(task_ids)
    args.task_counts = [cli.envs_per_task for _ in task_ids]
    args.num_envs = sum(args.task_counts)
    args.env_name = cli.env_name
    args.episodeLength = cli.episode_length
    args.fixed = cli.fixed
    args.same_init_config_per_task = cli.same_init_config_per_task
    args.termination_on_success = cli.termination_on_success
    args.reward_scale = cli.reward_scale
    args.seed = cli.seed
    args.beta = cli.beta
    args.sim_device = cli.sim_device
    args.graphics_device_id = -1
    args.headless = True
    args.rendering = False
    args.force_render = False
    args.saving = False
    args.wandb = False
    args.quiet = True
    args.init_curri_ratio = cli.init_curri_ratio
    args.task_embedding_dim = max(cli.task_embedding_dim, max(task_ids) + 1)

    pipeline = "gpu" if "cuda" in cli.sim_device else "cpu"
    overrides = [
        f"task_id={list(task_ids)}",
        f"task_counts={args.task_counts}",
        f"task.env.episodeLength={args.episodeLength}",
        f"+task.env.taskEmbeddingDim={args.task_embedding_dim}",
        f"fixed={args.fixed}",
        f"same_init_config_per_task={args.same_init_config_per_task}",
        f"termination_on_success={args.termination_on_success}",
        f"reward_scale={args.reward_scale}",
        f"pipeline={pipeline}",
        f"sim_device={args.sim_device}",
    ]
    if cli.contact_report:
        # The MTBench MetaWorld config disables PhysX contact collection for
        # speed. Enable last-substep contacts only for this diagnostic path.
        overrides.append("task.sim.physx.contact_collection=1")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=HYDRA_CONFIG_DIR):
        cfg = compose(config_name="custom_config", overrides=overrides)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    for key, value in cfg_dict.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def _make_env(args: Args):
    return isaacgymenvs.make(
        seed=args.seed,
        task=args.env_name,
        num_envs=args.num_envs,
        sim_device=args.sim_device,
        rl_device=args.sim_device,
        graphics_device_id=args.graphics_device_id,
        headless=True,
        force_render=False,
        custom_args=args,
        tasks=args.tasks,
        taskEnvCount=args.task_counts,
        cfg=args,
    )


def _all_env_ids(env) -> torch.Tensor:
    return torch.arange(env.num_envs, device=env.device, dtype=torch.long)


def _zero_actions(env) -> torch.Tensor:
    neutral = 0.5 if getattr(env, "use_beta", False) else 0.0
    return torch.full((env.num_envs, env.num_actions), neutral, device=env.rl_device)


def _random_actions(env, scale: float) -> torch.Tensor:
    actions = torch.empty((env.num_envs, env.num_actions), device=env.rl_device).uniform_(-scale, scale)
    if getattr(env, "use_beta", False):
        return torch.clamp(actions + 0.5, 0.0, 1.0)
    return actions


def _actions(env, cli: argparse.Namespace) -> torch.Tensor:
    if cli.action_mode == "random":
        return _random_actions(env, cli.random_action_scale)
    return _zero_actions(env)


def _settle_once(env, cli: argparse.Namespace) -> None:
    if cli.settle_mode == "simulate":
        env.gym.simulate(env.sim)
        if env.device == "cpu":
            env.gym.fetch_results(env.sim, True)
        if env.force_render:
            env.render()
        return

    env.step(_actions(env, cli))


def _refresh_obs(env, reset_ids: torch.Tensor | None = None) -> None:
    if reset_ids is None:
        env.compute_observations()
    else:
        env.compute_observations(reset_ids=reset_ids)


def _per_env_root_speeds(env) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_lin = torch.zeros(env.num_envs, device=env.device)
    max_ang = torch.zeros(env.num_envs, device=env.device)
    max_lin_actor = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    root_state = env._root_state
    for env_id in range(env.num_envs):
        start = int(env.franka_actor_idx[env_id].item())
        count = int(env.env_actor_count[env_id].item())
        state = root_state[start : start + count]
        if state.numel() == 0:
            continue
        lin_speed = torch.norm(state[:, 7:10], dim=-1)
        max_lin[env_id], max_lin_actor[env_id] = lin_speed.max(dim=0)
        max_ang[env_id] = torch.norm(state[:, 10:13], dim=-1).max()
    return max_lin, max_ang, max_lin_actor


def _per_env_dof_velocity(env) -> Tuple[torch.Tensor, torch.Tensor]:
    max_abs_vel = torch.zeros(env.num_envs, device=env.device)
    max_dof_local = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    dof_state = env._dof_state
    for env_id in range(env.num_envs):
        start = int(env.franka_dof_start_idx[env_id].item())
        count = int(env.env_dof_count[env_id].item())
        state = dof_state[start : start + count]
        if state.numel() == 0:
            continue
        max_abs_vel[env_id], max_dof_local[env_id] = state[:, 1].abs().max(dim=0)
    return max_abs_vel, max_dof_local


def _task_indices(env) -> torch.Tensor:
    if hasattr(env, "extras") and "task_indices" in env.extras:
        return env.extras["task_indices"].to(env.device)
    if hasattr(env, "task_indices"):
        return env.task_indices.to(env.device)
    return torch.zeros(env.num_envs, dtype=torch.long, device=env.device)


def _collect_env_metrics(env, obj_ref: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
    _refresh_obs(env)
    root_lin, root_ang, root_actor = _per_env_root_speeds(env)
    dof_vel, dof_local = _per_env_dof_velocity(env)
    obj_disp = torch.zeros(env.num_envs, device=env.device)
    if obj_ref is not None and hasattr(env, "obj_pos"):
        obj_disp = torch.norm(env.obj_pos - obj_ref, dim=-1)
    obj_pos = getattr(env, "obj_pos", torch.zeros((env.num_envs, 3), device=env.device))
    target_pos = getattr(env, "target_pos", torch.zeros((env.num_envs, 3), device=env.device))

    finite = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for tensor_name in ("_root_state", "_dof_state", "obj_pos", "target_pos"):
        tensor = getattr(env, tensor_name, None)
        if tensor is not None:
            tensor_finite = torch.isfinite(tensor)
            if tensor.shape[0] == env.num_envs:
                finite = finite & tensor_finite.reshape(env.num_envs, -1).all(dim=1)
            elif not bool(tensor_finite.all().item()):
                finite = finite & False

    return {
        "task": _task_indices(env),
        "progress": env.progress_buf.detach().clone(),
        "done": env.reset_buf.detach().clone().bool(),
        "success": env.success_buf.detach().clone().bool(),
        "timeout": getattr(env, "timeout_buf", torch.zeros_like(env.reset_buf)).detach().clone().bool(),
        "root_lin": root_lin.detach().clone(),
        "root_ang": root_ang.detach().clone(),
        "root_actor": root_actor.detach().clone(),
        "dof_vel": dof_vel.detach().clone(),
        "dof_local": dof_local.detach().clone(),
        "obj_disp": obj_disp.detach().clone(),
        "obj_pos": obj_pos.detach().clone(),
        "target_pos": target_pos.detach().clone(),
        "finite": finite.detach().clone(),
    }


def _vec3_tuple(value: Any) -> List[float] | None:
    if value is None:
        return None
    if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z"):
        return [float(value.x), float(value.y), float(value.z)]
    try:
        if len(value) >= 3:
            return [float(value[0]), float(value[1]), float(value[2])]
    except Exception:
        return None
    return None


def _contact_attr(contact: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(contact, name)
    except Exception:
        pass
    try:
        return contact[name]
    except Exception:
        return default


def _contact_float(contact: Any, name: str) -> float | None:
    value = _contact_attr(contact, name, None)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _body_label_cache(env, env_id: int) -> Tuple[int, List[Dict[str, Any]]]:
    if not hasattr(env, "_reset_validation_body_label_cache"):
        env._reset_validation_body_label_cache = {}

    cache = env._reset_validation_body_label_cache
    if env_id in cache:
        return cache[env_id]

    env_ptr = env.envs[env_id]
    global_start = int(env.franka_rigid_body_start_idx[env_id].item())
    labels: List[Dict[str, Any]] = []
    local_body = 0
    for actor_slot in range(env.gym.get_actor_count(env_ptr)):
        actor_handle = env.gym.get_actor_handle(env_ptr, actor_slot)
        actor_name = env.gym.get_actor_name(env_ptr, actor_handle)
        body_names = env.gym.get_actor_rigid_body_names(env_ptr, actor_handle)
        for actor_body_idx, body_name in enumerate(body_names):
            labels.append(
                {
                    "actor_slot": actor_slot,
                    "actor_name": actor_name,
                    "actor_body_idx": actor_body_idx,
                    "body_name": body_name,
                    "local_body": local_body,
                    "global_body": global_start + local_body,
                }
            )
            local_body += 1

    cache[env_id] = (global_start, labels)
    return cache[env_id]


def _unknown_body_label(body_index: int) -> Dict[str, Any]:
    actor_name = "ground" if body_index < 0 else "unknown"
    body_name = "ground" if body_index < 0 else f"body_{body_index}"
    return {
        "actor_slot": -1,
        "actor_name": actor_name,
        "actor_body_idx": -1,
        "body_name": body_name,
        "local_body": body_index,
        "global_body": body_index,
    }


def _contact_body_label(labels: Sequence[Dict[str, Any]], global_start: int, body_index: int) -> Dict[str, Any]:
    if 0 <= body_index < len(labels):
        return labels[body_index]
    local_index = body_index - global_start
    if 0 <= local_index < len(labels):
        return labels[local_index]
    return _unknown_body_label(body_index)


def _label_text(label: Dict[str, Any]) -> str:
    return f"{label['actor_name']}/{label['body_name']}"


def _contact_actor_filters(raw: str) -> List[str]:
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def _label_matches(label: Dict[str, Any], filters: Sequence[str]) -> bool:
    if not filters:
        return True
    haystack = f"{label['actor_name']}/{label['body_name']}".lower()
    return any(item in haystack for item in filters)


def _contact_matches_pair(
    label0: Dict[str, Any],
    label1: Dict[str, Any],
    filters: Sequence[str],
) -> bool:
    return _label_matches(label0, filters) or _label_matches(label1, filters)


def _contact_pair_rows(
    env,
    env_id: int,
    global_start: int,
    labels: Sequence[Dict[str, Any]],
    filters: Sequence[str],
    max_pairs: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        contacts = env.gym.get_env_rigid_contacts(env.envs[env_id])
    except Exception as exc:
        return [{"error": f"get_env_rigid_contacts failed: {exc}"}]

    for contact in contacts:
        body0 = int(_contact_attr(contact, "body0", -999999))
        body1 = int(_contact_attr(contact, "body1", -999999))
        label0 = _contact_body_label(labels, global_start, body0)
        label1 = _contact_body_label(labels, global_start, body1)
        if not _contact_matches_pair(label0, label1, filters):
            continue
        row = {
            "body0": body0,
            "body1": body1,
            "label0": _label_text(label0),
            "label1": _label_text(label1),
            "lambda": _contact_float(contact, "lambda"),
            "min_dist": _contact_float(contact, "min_dist"),
            "initial_overlap": _contact_float(contact, "initial_overlap"),
            "normal": _vec3_tuple(_contact_attr(contact, "normal", None)),
        }
        rows.append(row)
        if len(rows) >= max_pairs:
            break
    return rows


def _contact_force_rows(
    env,
    global_start: int,
    labels: Sequence[Dict[str, Any]],
    filters: Sequence[str],
    top_k: int,
    min_force: float,
) -> List[Dict[str, Any]]:
    if env._contact_forces is None or len(labels) == 0:
        return []
    forces = env._contact_forces[global_start : global_start + len(labels)].detach()
    norms = torch.norm(forces, dim=-1)
    if norms.numel() == 0:
        return []

    sorted_ids = torch.argsort(norms, descending=True)
    rows: List[Dict[str, Any]] = []
    for local_id_t in sorted_ids:
        local_id = int(local_id_t.item())
        norm = float(norms[local_id].item())
        if norm < min_force:
            break
        label = labels[local_id]
        if not _label_matches(label, filters):
            continue
        rows.append(
            {
                "label": _label_text(label),
                "local_body": int(label["local_body"]),
                "global_body": int(label["global_body"]),
                "force": [float(v) for v in forces[local_id].detach().cpu().tolist()],
                "force_norm": norm,
            }
        )
        if len(rows) >= top_k:
            break
    return rows


def _collect_contact_reports(
    env,
    cli: argparse.Namespace,
    check: str,
    cycle: int,
    phase: str,
    env_ids: torch.Tensor | None = None,
) -> List[Dict[str, Any]]:
    if not cli.contact_report:
        return []

    filters = _contact_actor_filters(cli.contact_actors)
    if env_ids is None:
        env_ids = _all_env_ids(env)
    env_ids = env_ids.detach().flatten().to(env.device)
    if env_ids.numel() == 0:
        return []

    env_ids = env_ids[: cli.contact_max_envs]
    task_indices = _task_indices(env)
    reports: List[Dict[str, Any]] = []
    for env_id_t in env_ids:
        env_id = int(env_id_t.item())
        task_id = int(task_indices[env_id].item())
        global_start, labels = _body_label_cache(env, env_id)
        force_rows = _contact_force_rows(
            env,
            global_start,
            labels,
            filters,
            cli.contact_top_k,
            cli.contact_min_force,
        )
        pair_rows = _contact_pair_rows(
            env,
            env_id,
            global_start,
            labels,
            filters,
            cli.contact_max_pairs,
        )
        if not force_rows and not pair_rows:
            continue
        reports.append(
            {
                "check": check,
                "cycle": cycle,
                "phase": phase,
                "env_id": env_id,
                "task_id": task_id,
                "task_name": TASK_IDX_TO_NAME.get(task_id, str(task_id)),
                "actor_filters": filters,
                "top_net_forces": force_rows,
                "contact_pairs": pair_rows,
            }
        )
    return reports


def _print_contact_reports(reports: Sequence[Dict[str, Any]]) -> None:
    if not reports:
        return
    print("\nContact diagnostics:")
    for report in reports:
        print(
            f"- {report['check']} cycle={report['cycle']} phase={report['phase']} "
            f"env={report['env_id']} T{report['task_id']} {report['task_name']}"
        )
        if report["top_net_forces"]:
            force_parts = []
            for row in report["top_net_forces"]:
                force_parts.append(
                    f"{row['label']} |F|={_compact_float(row['force_norm'])} "
                    f"F={_compact_list(row['force'])}"
                )
            print("  net: " + "; ".join(force_parts))
        if report["contact_pairs"]:
            pair_parts = []
            for row in report["contact_pairs"]:
                if "error" in row:
                    pair_parts.append(row["error"])
                    continue
                details = []
                if row.get("lambda") is not None:
                    details.append(f"lambda={_compact_float(row['lambda'])}")
                if row.get("initial_overlap") is not None:
                    details.append(f"overlap={_compact_float(row['initial_overlap'])}")
                if row.get("min_dist") is not None:
                    details.append(f"min_dist={_compact_float(row['min_dist'])}")
                if row.get("normal") is not None:
                    details.append(f"n={_compact_list(row['normal'])}")
                suffix = " " + " ".join(details) if details else ""
                pair_parts.append(f"{row['label0']} <-> {row['label1']}{suffix}")
            print("  pairs: " + "; ".join(pair_parts))


def _merge_window_metrics(current: Dict[str, torch.Tensor] | None, new: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if current is None:
        return {key: value.detach().clone() for key, value in new.items()}

    merged = {key: value.detach().clone() for key, value in current.items()}
    for key in ("root_lin", "root_ang", "dof_vel", "obj_disp"):
        better = new[key] > merged[key]
        merged[key][better] = new[key][better]
        if key == "root_lin":
            merged["root_actor"][better] = new["root_actor"][better]
        if key == "dof_vel":
            merged["dof_local"][better] = new["dof_local"][better]
    for key in ("done", "success", "timeout"):
        merged[key] = merged[key] | new[key]
    merged["finite"] = merged["finite"] & new["finite"]
    merged["progress"] = new["progress"].detach().clone()
    return merged


def _max_for_task(metrics: Dict[str, torch.Tensor], task_id: int, key: str) -> float:
    mask = metrics["task"] == task_id
    if not bool(mask.any().item()):
        return 0.0
    return float(metrics[key][mask].max().item())


def _count_for_task(metrics: Dict[str, torch.Tensor], task_id: int, key: str) -> int:
    mask = metrics["task"] == task_id
    if not bool(mask.any().item()):
        return 0
    return int(metrics[key][mask].sum().item())


def _count_progress_le(metrics: Dict[str, torch.Tensor], task_id: int, max_progress: int) -> int:
    mask = (metrics["task"] == task_id) & metrics["done"] & (metrics["progress"] <= max_progress)
    return int(mask.sum().item())


def _example_for_task(
    metrics: Dict[str, torch.Tensor],
    task_id: int,
    value_key: str,
    index_key: str | None = None,
) -> str:
    mask = metrics["task"] == task_id
    if not bool(mask.any().item()):
        return "example unavailable"
    env_ids = mask.nonzero(as_tuple=False).flatten()
    local_argmax = torch.argmax(metrics[value_key][mask])
    env_id = int(env_ids[local_argmax].item())
    parts = [f"example_env={env_id}"]
    if index_key is not None:
        parts.append(f"{index_key}={int(metrics[index_key][env_id].item())}")
    if "obj_pos" in metrics:
        parts.append(f"obj_pos={_compact_vec(metrics['obj_pos'][env_id])}")
    if "target_pos" in metrics:
        parts.append(f"target_pos={_compact_vec(metrics['target_pos'][env_id])}")
    return ", ".join(parts)


def _record_summary(
    summaries: List[Dict[str, Any]],
    check: str,
    cycle: int,
    phase: str,
    task_ids: Sequence[int],
    metrics: Dict[str, torch.Tensor],
) -> None:
    for task_id in task_ids:
        mask = metrics["task"] == task_id
        if not bool(mask.any().item()):
            continue
        summaries.append(
            {
                "check": check,
                "cycle": cycle,
                "phase": phase,
                "task_id": task_id,
                "task_name": TASK_IDX_TO_NAME[task_id],
                "num_envs": int(mask.sum().item()),
                "max_progress": int(metrics["progress"][mask].max().item()),
                "done_count": int(metrics["done"][mask].sum().item()),
                "success_count": int(metrics["success"][mask].sum().item()),
                "timeout_count": int(metrics["timeout"][mask].sum().item()),
                "nonfinite_count": int((~metrics["finite"][mask]).sum().item()),
                "max_root_linvel": _max_for_task(metrics, task_id, "root_lin"),
                "max_root_angvel": _max_for_task(metrics, task_id, "root_ang"),
                "max_dof_absvel": _max_for_task(metrics, task_id, "dof_vel"),
                "max_obj_disp": _max_for_task(metrics, task_id, "obj_disp"),
            }
        )


def _add_issue(
    issues: List[Dict[str, Any]],
    *,
    check: str,
    cycle: int,
    phase: str = "",
    task_id: int,
    kind: str,
    value: float | int,
    threshold: float | int,
    detail: str,
) -> None:
    issues.append(
        {
            "check": check,
            "cycle": cycle,
            "phase": phase,
            "task_id": task_id,
            "task_name": TASK_IDX_TO_NAME[task_id],
            "kind": kind,
            "value": value,
            "threshold": threshold,
            "detail": detail,
        }
    )


def _check_thresholds(
    cli: argparse.Namespace,
    issues: List[Dict[str, Any]],
    check: str,
    cycle: int,
    task_ids: Sequence[int],
    metrics: Dict[str, torch.Tensor],
    *,
    phase: str = "",
    require_progress_zero: bool = False,
    flag_done_at_progress_le: int | None = None,
) -> None:
    for task_id in task_ids:
        max_root = _max_for_task(metrics, task_id, "root_lin")
        max_dof = _max_for_task(metrics, task_id, "dof_vel")
        max_disp = _max_for_task(metrics, task_id, "obj_disp")
        nonfinite = _count_for_task({"task": metrics["task"], "finite_bad": ~metrics["finite"]}, task_id, "finite_bad")

        if max_root > cli.max_reset_root_linvel:
            example = _example_for_task(metrics, task_id, "root_lin", "root_actor")
            _add_issue(
                issues,
                check=check,
                cycle=cycle,
                phase=phase,
                task_id=task_id,
                kind="root_linvel_high",
                value=max_root,
                threshold=cli.max_reset_root_linvel,
                detail=f"actor root linear velocity is high right after reset or settle; {example}",
            )
        if max_dof > cli.max_reset_dof_absvel:
            example = _example_for_task(metrics, task_id, "dof_vel", "dof_local")
            _add_issue(
                issues,
                check=check,
                cycle=cycle,
                phase=phase,
                task_id=task_id,
                kind="dof_velocity_high",
                value=max_dof,
                threshold=cli.max_reset_dof_absvel,
                detail=f"joint velocity is high right after reset or settle; {example}",
            )
        if max_disp > cli.max_settle_obj_disp:
            example = _example_for_task(metrics, task_id, "obj_disp")
            _add_issue(
                issues,
                check=check,
                cycle=cycle,
                phase=phase,
                task_id=task_id,
                kind="object_drift_high",
                value=max_disp,
                threshold=cli.max_settle_obj_disp,
                detail=f"primary object moved too much during the post-reset settle window; {example}",
            )
        if nonfinite:
            _add_issue(
                issues,
                check=check,
                cycle=cycle,
                phase=phase,
                task_id=task_id,
                kind="nonfinite_state",
                value=nonfinite,
                threshold=0,
                detail="state tensor contains NaN or Inf",
            )

        if require_progress_zero:
            max_progress = _max_for_task(metrics, task_id, "progress")
            if max_progress != 0:
                _add_issue(
                    issues,
                    check=check,
                    cycle=cycle,
                    phase=phase,
                    task_id=task_id,
                    kind="progress_not_zero_after_reset",
                    value=max_progress,
                    threshold=0,
                    detail="progress_buf should be zero immediately after reset_idx/reset_all",
                )

        if flag_done_at_progress_le is not None:
            early_done = _count_progress_le(metrics, task_id, flag_done_at_progress_le)
            if early_done:
                _add_issue(
                    issues,
                    check=check,
                    cycle=cycle,
                    phase=phase,
                    task_id=task_id,
                    kind="done_immediately_after_reset",
                    value=early_done,
                    threshold=0,
                    detail=f"reset_buf became true at progress <= {flag_done_at_progress_le}",
                )


def _forced_reset_check(
    env,
    cli: argparse.Namespace,
    task_ids: Sequence[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    summaries: List[Dict[str, Any]] = []
    issues: List[Dict[str, Any]] = []
    contact_reports: List[Dict[str, Any]] = []
    env_ids = _all_env_ids(env)

    for cycle in range(cli.episodes):
        env.reset_idx(env_ids)
        _refresh_obs(env, env_ids)
        reset_metrics = _collect_env_metrics(env)
        contact_reports.extend(_collect_contact_reports(env, cli, "forced", cycle, "after_reset", env_ids))
        _record_summary(summaries, "forced", cycle, "after_reset", task_ids, reset_metrics)
        _check_thresholds(
            cli,
            issues,
            "forced",
            cycle,
            task_ids,
            reset_metrics,
            phase="after_reset",
            require_progress_zero=True,
        )

        obj_ref = env.obj_pos.detach().clone() if hasattr(env, "obj_pos") else None
        first_step_metrics = None
        window_metrics = None
        final_step_metrics = None
        for step_idx in range(cli.settle_steps):
            _settle_once(env, cli)
            step_metrics = _collect_env_metrics(env, obj_ref=obj_ref)
            window_metrics = _merge_window_metrics(window_metrics, step_metrics)
            final_step_metrics = step_metrics
            if step_idx == 0:
                first_step_metrics = step_metrics
                contact_reports.extend(_collect_contact_reports(env, cli, "forced", cycle, "first_step", env_ids))

        if first_step_metrics is not None:
            _check_thresholds(
                cli,
                issues,
                "forced",
                cycle,
                task_ids,
                first_step_metrics,
                phase="first_step",
                flag_done_at_progress_le=1,
            )

        if window_metrics is not None:
            _record_summary(summaries, "forced", cycle, "settle_window", task_ids, window_metrics)
            _check_thresholds(cli, issues, "forced", cycle, task_ids, window_metrics, phase="settle_window")

        if final_step_metrics is not None:
            contact_reports.extend(_collect_contact_reports(env, cli, "forced", cycle, "settle_final", env_ids))
            _record_summary(summaries, "forced", cycle, "settle_final", task_ids, final_step_metrics)
            _check_thresholds(cli, issues, "forced", cycle, task_ids, final_step_metrics, phase="settle_final")

    return summaries, issues, contact_reports


def _timeout_reset_check(
    env,
    cli: argparse.Namespace,
    task_ids: Sequence[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    summaries: List[Dict[str, Any]] = []
    issues: List[Dict[str, Any]] = []
    contact_reports: List[Dict[str, Any]] = []
    env_ids = _all_env_ids(env)
    max_rollout_steps = cli.episode_length + cli.timeout_slack_steps

    for cycle in range(cli.episodes):
        env.reset_idx(env_ids)
        _refresh_obs(env, env_ids)
        start_metrics = _collect_env_metrics(env)
        contact_reports.extend(_collect_contact_reports(env, cli, "timeout", cycle, "after_reset", env_ids))
        _record_summary(summaries, "timeout", cycle, "after_reset", task_ids, start_metrics)
        _check_thresholds(
            cli,
            issues,
            "timeout",
            cycle,
            task_ids,
            start_metrics,
            phase="after_reset",
            require_progress_zero=True,
        )

        done_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        terminal_progress = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        terminal_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        terminal_timeout = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        for local_step in range(1, max_rollout_steps + 1):
            _, _, done, extras = env.step(_actions(env, cli))
            done = done.bool().to(env.device)
            newly_done = done & ~done_seen
            if bool(newly_done.any().item()):
                done_seen |= newly_done
                terminal_progress[newly_done] = env.progress_buf[newly_done].long()
                terminal_success[newly_done] = env.success_buf[newly_done].bool()
                timeout_tensor = extras.get("time_outs", torch.zeros_like(done)).bool().to(env.device)
                terminal_timeout[newly_done] = timeout_tensor[newly_done]
            if bool(done_seen.all().item()):
                break

        terminal_metrics = _collect_env_metrics(env)
        _record_summary(summaries, "timeout", cycle, "terminal", task_ids, terminal_metrics)

        for task_id in task_ids:
            task_mask = _task_indices(env) == task_id
            seen_mask = task_mask & done_seen
            missing = int((task_mask & ~done_seen).sum().item())
            if missing:
                _add_issue(
                    issues,
                    check="timeout",
                    cycle=cycle,
                    phase="terminal",
                    task_id=task_id,
                    kind="episode_did_not_finish",
                    value=missing,
                    threshold=0,
                    detail=f"envs did not finish within {max_rollout_steps} steps",
                )

            if bool(seen_mask.any().item()):
                min_progress = int(terminal_progress[seen_mask].min().item())
                max_progress = int(terminal_progress[seen_mask].max().item())
                early = int((terminal_progress[seen_mask] <= 1).sum().item())
                before_horizon = int((terminal_progress[seen_mask] < cli.episode_length - 1).sum().item())
                if early:
                    _add_issue(
                        issues,
                        check="timeout",
                        cycle=cycle,
                        phase="terminal",
                        task_id=task_id,
                        kind="terminal_at_progress_le_1",
                        value=early,
                        threshold=0,
                        detail="episode ended immediately after reset",
                    )
                if before_horizon:
                    _add_issue(
                        issues,
                        check="timeout",
                        cycle=cycle,
                        phase="terminal",
                        task_id=task_id,
                        kind="terminal_before_horizon",
                        value=before_horizon,
                        threshold=0,
                        detail=(
                            f"terminal progress range {min_progress}-{max_progress}; "
                            f"horizon terminal progress is normally {cli.episode_length - 1}"
                        ),
                    )

        done_ids = done_seen.nonzero(as_tuple=False).flatten()
        if len(done_ids) == 0:
            continue

        # In this env stack, the step after a terminal signal simulates once,
        # then post_physics_step resets done envs and computes reward on the
        # freshly reset state. A clean reset therefore returns progress_buf == 0.
        env.step(_actions(env, cli))
        followup_metrics = _collect_env_metrics(env)
        contact_reports.extend(_collect_contact_reports(env, cli, "timeout", cycle, "post_reset_probe", done_ids))
        _record_summary(summaries, "timeout", cycle, "post_reset_probe", task_ids, followup_metrics)

        for task_id in task_ids:
            task_mask = followup_metrics["task"] == task_id
            mask = task_mask & done_seen
            if not bool(mask.any().item()):
                continue
            max_progress = int(followup_metrics["progress"][mask].max().item())
            min_progress = int(followup_metrics["progress"][mask].min().item())
            if min_progress != 0 or max_progress != 0:
                _add_issue(
                    issues,
                    check="timeout",
                    cycle=cycle,
                    phase="post_reset_probe",
                    task_id=task_id,
                    kind="progress_dirty_after_timeout_reset",
                    value=max_progress,
                    threshold=0,
                    detail=f"post-reset probe progress range is {min_progress}-{max_progress}, expected 0",
                )
            immediate_done = int((followup_metrics["done"][mask] & (followup_metrics["progress"][mask] == 0)).sum().item())
            if immediate_done:
                _add_issue(
                    issues,
                    check="timeout",
                    cycle=cycle,
                    phase="post_reset_probe",
                    task_id=task_id,
                    kind="post_timeout_reset_done_immediately",
                    value=immediate_done,
                    threshold=0,
                    detail="env reset after timeout but reset_buf became true again on the probe step",
                )

        _check_thresholds(cli, issues, "timeout", cycle, task_ids, followup_metrics, phase="post_reset_probe")

        obj_ref = env.obj_pos.detach().clone() if hasattr(env, "obj_pos") else None
        first_step_metrics = None
        window_metrics = None
        final_step_metrics = None
        for step_idx in range(cli.settle_steps):
            _settle_once(env, cli)
            step_metrics = _collect_env_metrics(env, obj_ref=obj_ref)
            window_metrics = _merge_window_metrics(window_metrics, step_metrics)
            final_step_metrics = step_metrics
            if step_idx == 0:
                first_step_metrics = step_metrics
                contact_reports.extend(_collect_contact_reports(env, cli, "timeout", cycle, "post_reset_first_step", done_ids))

        if first_step_metrics is not None:
            _check_thresholds(
                cli,
                issues,
                "timeout",
                cycle,
                task_ids,
                first_step_metrics,
                phase="post_reset_first_step",
                flag_done_at_progress_le=1,
            )

        if window_metrics is not None:
            _record_summary(summaries, "timeout", cycle, "post_reset_settle_window", task_ids, window_metrics)
            _check_thresholds(cli, issues, "timeout", cycle, task_ids, window_metrics, phase="post_reset_settle_window")

        if final_step_metrics is not None:
            contact_reports.extend(_collect_contact_reports(env, cli, "timeout", cycle, "post_reset_settle_final", done_ids))
            _record_summary(summaries, "timeout", cycle, "post_reset_settle_final", task_ids, final_step_metrics)
            _check_thresholds(cli, issues, "timeout", cycle, task_ids, final_step_metrics, phase="post_reset_settle_final")

    return summaries, issues, contact_reports


def _summarize_by_task(task_ids: Sequence[int], summaries: Sequence[Dict[str, Any]], issues: Sequence[Dict[str, Any]]) -> List[List[Any]]:
    grouped: Dict[Tuple[str, int], Dict[str, Any]] = {}
    issue_counts: Dict[Tuple[str, int], int] = defaultdict(int)
    for issue in issues:
        issue_counts[(issue["check"], issue["task_id"])] += 1

    for item in summaries:
        key = (item["check"], item["task_id"])
        if key not in grouped:
            grouped[key] = {
                "check": item["check"],
                "task_id": item["task_id"],
                "task_name": item["task_name"],
                "cycles": set(),
                "max_root_linvel": 0.0,
                "max_dof_absvel": 0.0,
                "max_obj_disp": 0.0,
                "done_count": 0,
                "success_count": 0,
                "timeout_count": 0,
                "nonfinite_count": 0,
            }
        group = grouped[key]
        group["cycles"].add(item["cycle"])
        for metric in ("max_root_linvel", "max_dof_absvel", "max_obj_disp"):
            group[metric] = max(float(group[metric]), float(item[metric]))
        for count_key in ("done_count", "success_count", "timeout_count", "nonfinite_count"):
            group[count_key] += int(item[count_key])

    rows: List[List[Any]] = []
    for check in ("forced", "timeout"):
        for task_id in task_ids:
            group = grouped.get((check, task_id))
            if group is None:
                continue
            rows.append(
                [
                    check,
                    task_id,
                    group["task_name"],
                    len(group["cycles"]),
                    issue_counts[(check, task_id)],
                    _compact_float(group["max_root_linvel"]),
                    _compact_float(group["max_dof_absvel"]),
                    _compact_float(group["max_obj_disp"]),
                    group["done_count"],
                    group["success_count"],
                    group["timeout_count"],
                    group["nonfinite_count"],
                ]
            )
    return rows


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _parse_checks(raw: str) -> List[str]:
    checks = [part.strip().lower() for part in raw.split(",") if part.strip()]
    valid = {"forced", "timeout"}
    unknown = sorted(set(checks) - valid)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown checks: {unknown}")
    if not checks:
        raise argparse.ArgumentTypeError("at least one check is required")
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="28", help="Task ids, ranges, comma list, or 'all'. Example: 28 or 0-49 or 0,28,49")
    parser.add_argument("--envs-per-task", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--episode-length", type=int, default=30)
    parser.add_argument("--checks", type=_parse_checks, default=_parse_checks("forced,timeout"))
    parser.add_argument("--settle-steps", type=int, default=3)
    parser.add_argument(
        "--settle-mode",
        choices=["step", "simulate"],
        default="step",
        help="Use normal env.step actions during settle, or only advance PhysX with the reset targets already installed.",
    )
    parser.add_argument("--timeout-slack-steps", type=int, default=5)
    parser.add_argument("--action-mode", choices=["zero", "random"], default="zero")
    parser.add_argument("--random-action-scale", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--beta",
        type=_str2bool,
        default=False,
        help="Whether actions are encoded in beta-policy [0,1] space before env conversion.",
    )
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME)
    parser.add_argument("--fixed", type=_str2bool, default=True)
    parser.add_argument("--same-init-config-per-task", type=_str2bool, default=False)
    parser.add_argument("--termination-on-success", type=_str2bool, default=True)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--init-curri-ratio", type=float, default=1.0)
    parser.add_argument("--task-embedding-dim", type=int, default=len(TASK_IDX_TO_NAME))
    parser.add_argument("--max-reset-root-linvel", type=float, default=0.50)
    parser.add_argument("--max-reset-dof-absvel", type=float, default=1.0)
    parser.add_argument("--max-settle-obj-disp", type=float, default=0.03)
    parser.add_argument("--contact-report", action="store_true", help="Print and record reset-time contact diagnostics.")
    parser.add_argument(
        "--contact-actors",
        default="peg_block,obj,block",
        help="Comma-separated actor/body substrings to include in contact diagnostics.",
    )
    parser.add_argument("--contact-top-k", type=int, default=6)
    parser.add_argument("--contact-max-envs", type=int, default=2)
    parser.add_argument("--contact-max-pairs", type=int, default=12)
    parser.add_argument("--contact-min-force", type=float, default=1e-6)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--fail-on-issue", action="store_true")
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    task_ids = _parse_tasks(cli.tasks)
    args = _build_args(cli, task_ids)

    print(
        "Reset validation config: "
        f"tasks={task_ids}, envs={args.num_envs}, episodes={cli.episodes}, "
        f"episodeLength={cli.episode_length}, checks={','.join(cli.checks)}, "
        f"fixed={args.fixed}, same_init={args.same_init_config_per_task}, "
        f"device={args.sim_device}"
    )

    env = _make_env(args)
    summaries: List[Dict[str, Any]] = []
    issues: List[Dict[str, Any]] = []
    contact_reports: List[Dict[str, Any]] = []
    try:
        env.reset_all()
        if "forced" in cli.checks:
            forced_summaries, forced_issues, forced_contacts = _forced_reset_check(env, cli, task_ids)
            summaries.extend(forced_summaries)
            issues.extend(forced_issues)
            contact_reports.extend(forced_contacts)
        if "timeout" in cli.checks:
            timeout_summaries, timeout_issues, timeout_contacts = _timeout_reset_check(env, cli, task_ids)
            summaries.extend(timeout_summaries)
            issues.extend(timeout_issues)
            contact_reports.extend(timeout_contacts)
    finally:
        if hasattr(env, "close"):
            env.close()

    rows = _summarize_by_task(task_ids, summaries, issues)
    print()
    _print_rows(
        [
            "check",
            "tid",
            "task",
            "cycles",
            "issues",
            "max_root_v",
            "max_dof_v",
            "max_obj_d",
            "done",
            "succ",
            "tout",
            "bad",
        ],
        rows,
    )

    if issues:
        print("\nIssues:")
        for issue in issues[:50]:
            value = issue["value"]
            if isinstance(value, float):
                value = _compact_float(value)
            print(
                f"- {issue['check']} cycle={issue['cycle']} "
                f"phase={issue.get('phase', '')} T{issue['task_id']} {issue['task_name']}: {issue['kind']} "
                f"value={value} threshold={issue['threshold']} ({issue['detail']})"
            )
        if len(issues) > 50:
            print(f"- ... {len(issues) - 50} more issues omitted from console")
    else:
        print("\nNo reset validation issues exceeded the configured thresholds.")

    _print_contact_reports(contact_reports)

    if cli.output_json:
        _write_json(
            cli.output_json,
            {
                "config": {
                    "tasks": task_ids,
                    "task_counts": args.task_counts,
                    "episodes": cli.episodes,
                    "episode_length": cli.episode_length,
                    "checks": cli.checks,
                    "settle_steps": cli.settle_steps,
                    "settle_mode": cli.settle_mode,
                    "action_mode": cli.action_mode,
                    "fixed": args.fixed,
                    "same_init_config_per_task": args.same_init_config_per_task,
                    "termination_on_success": args.termination_on_success,
                    "seed": args.seed,
                    "beta": args.beta,
                    "sim_device": args.sim_device,
                },
                "summaries": summaries,
                "issues": issues,
                "contact_reports": contact_reports,
            },
        )
        print(f"\nWrote JSON report to {cli.output_json}")

    return 1 if issues and cli.fail_on_issue else 0


if __name__ == "__main__":
    raise SystemExit(main())
