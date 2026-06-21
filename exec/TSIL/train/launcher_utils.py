#!/usr/bin/env python3
"""Shared TSIL Hydra launcher helpers."""

from __future__ import annotations

import datetime as dt
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from math import ceil
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from hydra import compose, initialize_config_dir
from hydra._internal.core_plugins.basic_sweeper import BasicSweeper
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import DictConfig, ListConfig, OmegaConf

from projects.TSIL.ckpt_layout import benchmark_task_dir_name


TRAIN_DIR = Path(__file__).resolve().parent
CONF_DIR = TRAIN_DIR / "conf"
OWNED_TRAIN_ARG_NAMES = {
    "method_suffix",
    "result_task_dir_name",
    "script_time",
    "seed",
    "task_name",
    "tasks",
}


def compose_benchmark_config(overrides: Iterable[str] | None = None) -> DictConfig:
    """Compose the benchmark Hydra config outside the decorated app for tests/tooling."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        return compose(config_name="config", overrides=list(overrides or []))


def hydra_job_num() -> int:
    try:
        value = OmegaConf.select(HydraConfig.get(), "job.num", default=0)
    except ValueError:
        return 0
    if value in (None, "???"):
        return 0
    return int(value)


def hydra_output_dir(repo_root: Path) -> Path:
    try:
        output_dir = OmegaConf.select(HydraConfig.get(), "runtime.output_dir")
    except ValueError:
        output_dir = None
    return Path(output_dir or repo_root / "results" / "TSIL" / "hydra_res" / "manual").resolve()


def candidate_gpus(cfg: Mapping[str, Any]) -> list[str]:
    launch_cfg = cfg.get("launch") or {}
    return [str(gpu) for gpu in (launch_cfg.get("gpus") or [])]


def jobs_per_gpu(cfg: Mapping[str, Any], gpu_count: int) -> int:
    if gpu_count <= 0:
        return 0
    launch_cfg = cfg.get("launch") or {}
    return max(ceil(int(launch_cfg.get("n_jobs", 1)) / gpu_count), 1)


def display_path(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except (OSError, ValueError):
        return str(path)


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds_int = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{seconds_int:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{seconds_int:02d}s"


def total_jobs_from_multirun(sweep_dir: Path) -> int | None:
    multirun_path = sweep_dir / "multirun.yaml"
    if not multirun_path.is_file():
        return None
    try:
        multirun_cfg = OmegaConf.load(multirun_path)
        task_overrides = OmegaConf.select(multirun_cfg, "hydra.overrides.task", default=[])
        override_args = list(OmegaConf.to_container(task_overrides, resolve=False) or [])
        overrides = OverridesParser.create().parse_overrides(override_args)
        batches = BasicSweeper.split_arguments(overrides, max_batch_size=None)
    except Exception:
        return None
    return sum(len(batch) for batch in batches)


def job_label(job_num: int, total_jobs: int | None) -> str:
    return str(job_num) if total_jobs is None else f"{job_num}/{total_jobs}"


def _stringify(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def render_arg_map(arg_map: Mapping[str, Any] | None) -> list[str]:
    if not arg_map:
        return []

    rendered: list[str] = []
    for key, value in arg_map.items():
        if value is None:
            if key == "target_kl":
                rendered.append(f"--{key}")
            continue
        rendered.append(f"--{key}")
        if isinstance(value, (list, tuple, ListConfig)):
            rendered.extend(_stringify(item) for item in value)
        else:
            rendered.append(_stringify(value))
    return rendered


def _tail_text(path: Path, max_lines: int = 80) -> str:
    try:
        return "".join(path.read_text(errors="replace").splitlines(keepends=True)[-max_lines:])
    except OSError:
        return ""


def write_failure_summary(
    output_dir: Path,
    job_num: int,
    gpu_id: str | None,
    selectors: str,
    command: list[str],
    log_path: Path,
    returncode: int,
) -> None:
    gpu_label = gpu_id if gpu_id is not None else "inherit"
    summary = (
        f"FAIL job={job_num} gpu={gpu_label} returncode={returncode}\n"
        f"selectors={selectors}\n"
        f"log={log_path}\n"
        f"command={command_string(command)}\n\n"
        f"--- tail train.log ---\n{_tail_text(log_path)}"
    )
    (output_dir / "error.txt").write_text(summary)
    with (output_dir.parent / "failures.txt").open("a") as failures:
        failures.write(f"job={job_num} gpu={gpu_label} returncode={returncode} log={log_path}\n")


def _append_gpu_status(sweep_dir: Path, message: str) -> None:
    try:
        with (sweep_dir / "gpu_status.txt").open("a") as status_file:
            status_file.write(f"{dt.datetime.now().isoformat(timespec='seconds')} {message}\n")
    except OSError:
        pass


@contextmanager
def acquire_gpu_slot(
    sweep_dir: Path,
    gpu_ids: Iterable[str] | None,
    jobs_per_gpu: int,
    wait_seconds: float,
    job_num: int,
    log_path: Path,
):
    gpu_ids = [str(gpu_id) for gpu_id in (gpu_ids or [])]
    if not gpu_ids or jobs_per_gpu <= 0:
        yield None, None
        return

    slot_dir = sweep_dir / "gpu_slots"
    slot_dir.mkdir(parents=True, exist_ok=True)
    start = job_num % len(gpu_ids)
    ordered_gpus = gpu_ids[start:] + gpu_ids[:start]
    gpu_id = None
    slot_path = None
    slot_index = None
    while slot_path is None:
        for candidate_gpu in ordered_gpus:
            for slot in range(jobs_per_gpu):
                candidate = slot_dir / f"gpu{candidate_gpu}_slot{slot}.lock"
                try:
                    fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    continue
                with os.fdopen(fd, "w") as lock_file:
                    lock_file.write(
                        f"pid={os.getpid()}\njob={job_num}\ngpu={candidate_gpu}\n"
                        f"slot={slot}\nlog={log_path}\n"
                        f"started_at={dt.datetime.now().isoformat(timespec='seconds')}\n"
                    )
                gpu_id = candidate_gpu
                slot_path = candidate
                slot_index = slot
                _append_gpu_status(sweep_dir, f"ACQUIRE job={job_num} gpu={gpu_id} slot={slot} log={log_path}")
                break
            if slot_path is not None:
                break
        if slot_path is None:
            time.sleep(wait_seconds)

    try:
        yield gpu_id, slot_path
    finally:
        _append_gpu_status(sweep_dir, f"RELEASE job={job_num} gpu={gpu_id} slot={slot_index} log={log_path}")
        try:
            slot_path.unlink()
        except OSError:
            pass


def _method_components(cfg: Mapping[str, Any]) -> list[str]:
    components = cfg.get("method_components") or ["sil", "temporal"]
    return [str(component) for component in components]


def _method_metadata_order(cfg: Mapping[str, Any]) -> list[str]:
    configured = cfg.get("method_metadata_order")
    if configured:
        return [str(component) for component in configured]
    return list(reversed(_method_components(cfg)))


def _mapping_value(mapping: Mapping[str, Any] | None, benchmark_id: str) -> Any:
    if not mapping:
        return None
    if benchmark_id in mapping:
        return mapping[benchmark_id]
    return mapping.get("default")


def _component_metadata_value(
    component_cfg: Mapping[str, Any],
    benchmark_id: str,
    scalar_key: str,
    mapping_key: str,
) -> Any:
    mapped_value = _mapping_value(component_cfg.get(mapping_key), benchmark_id)
    if mapped_value is not None:
        return mapped_value
    return component_cfg.get(scalar_key)


def _component_items_in_order(
    cfg: Mapping[str, Any],
    component_order: Iterable[str],
) -> list[tuple[str, Mapping[str, Any]]]:
    method_cfg = cfg["method"]
    return [(component_name, method_cfg[component_name]) for component_name in component_order]


def resolve_task_id(cfg: Mapping[str, Any]) -> int:
    explicit = cfg.get("task_id")
    if explicit is None:
        raise ValueError("task_id must be specified explicitly for benchmark launches")
    return int(explicit)


def resolve_seed(cfg: Mapping[str, Any]) -> int:
    explicit = cfg.get("seed")
    if explicit is None:
        raise ValueError("seed must be specified explicitly for benchmark launches")
    return int(explicit)


def resolve_script_time(cfg: Mapping[str, Any]) -> str:
    explicit = cfg.get("script_time")
    if explicit:
        return str(explicit)
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def task_name_for(cfg: Mapping[str, Any], task_id: int) -> str:
    return f'{cfg["benchmark"]["task_name_prefix"]}{task_id}'


def _value_lookup_key(value: Any) -> str:
    if isinstance(value, float):
        return format(value, "g")
    return str(value)


def _canonical_numeric_key(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None

    try:
        decimal_value = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None

    normalized = format(decimal_value, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def _lookup_token(mapping: Mapping[str, Any] | None, value: Any) -> str | None:
    if not mapping:
        return None
    value_key = _value_lookup_key(value)
    if value_key in mapping:
        return str(mapping[value_key])
    alt_key = str(value)
    if alt_key in mapping:
        return str(mapping[alt_key])

    canonical_value_key = _canonical_numeric_key(value)
    if canonical_value_key is None:
        return None

    for mapping_key, token in mapping.items():
        if _canonical_numeric_key(mapping_key) == canonical_value_key:
            return str(token)
    return None


def _base_method_suffix(cfg: Mapping[str, Any]) -> str | None:
    benchmark_cfg = cfg["benchmark"]
    experiment_cfg = cfg["experiment"]
    component_cfgs = _component_items_in_order(cfg, _method_metadata_order(cfg))

    parts: list[str] = []
    for _, component_cfg in component_cfgs:
        token = _component_metadata_value(
            component_cfg,
            benchmark_cfg["id"],
            "method_suffix_token",
            "method_suffix_token_by_benchmark",
        )
        if token:
            parts.append(str(token))

    append = experiment_cfg.get("method_suffix_append")
    if append:
        parts.append(str(append))
    return "_".join(parts) if parts else None


def _reward_scale_from_args(train_arg_map: Mapping[str, Any]) -> Any:
    success_reward = train_arg_map.get("successRewardScale")
    if success_reward not in (None, 0, 0.0, "0", "0.0"):
        return success_reward

    eps_time_scale = train_arg_map.get("epstimeRewardScale")
    if isinstance(eps_time_scale, str):
        stripped = eps_time_scale.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            values = [value.strip() for value in stripped[1:-1].split(",") if value.strip()]
            if values:
                return values[0]
    if isinstance(eps_time_scale, (list, tuple, ListConfig)) and eps_time_scale:
        return eps_time_scale[0]
    return eps_time_scale


def build_method_suffix(cfg: Mapping[str, Any], train_arg_map: Mapping[str, Any]) -> str | None:
    experiment_cfg = cfg["experiment"]
    experiment_id = str(experiment_cfg["id"])
    base_suffix = _base_method_suffix(cfg)

    if experiment_id == "sweep_tsil_coef":
        token = _lookup_token(experiment_cfg.get("value_suffixes", {}).get("sil_coef"), train_arg_map.get("sil_coef"))
        if token is None:
            raise ValueError(
                "sweep_tsil_coef requires sil_coef to match experiment.value_suffixes.sil_coef"
            )
        tw_token = _component_metadata_value(
            cfg["method"]["temporal"],
            cfg["benchmark"]["id"],
            "method_suffix_token",
            "method_suffix_token_by_benchmark",
        )
        prefix_parts = [str(tw_token)] if tw_token else []
        sil_mode = str(cfg["method"]["sil"].get("args", {}).get("sil_mode", "sil"))
        prefix_parts.append("BC" if sil_mode == "bc" else "SIL")
        return f'{"_".join(prefix_parts)}_{cfg["method"]["sil"]["id"]}_{token}'

    if experiment_id == "sweep_suc_rew":
        reward_scale = _reward_scale_from_args(train_arg_map)
        token = _lookup_token(experiment_cfg.get("value_suffixes", {}).get("reward_scale"), reward_scale)
        if token is None:
            raise ValueError(
                "sweep_suc_rew requires a reward scale matching experiment.value_suffixes.reward_scale"
            )
        return f"{base_suffix}_MaxR_{token}" if base_suffix else f"MaxR_{token}"

    if experiment_cfg.get("value_suffixes"):
        suffix_parts = [base_suffix] if base_suffix else []
        for arg_name, mapping in experiment_cfg.get("value_suffixes", {}).items():
            token = _lookup_token(mapping, train_arg_map.get(str(arg_name)))
            if token is None:
                raise ValueError(
                    f"{experiment_id} requires {arg_name} to match experiment.value_suffixes.{arg_name}"
                )
            suffix_parts.append(token)
        return "_".join(suffix_parts) if suffix_parts else None

    override = experiment_cfg.get("method_suffix_override")
    if override:
        return str(override)
    return base_suffix


def build_description(cfg: Mapping[str, Any], train_arg_map: Mapping[str, Any]) -> str:
    benchmark_cfg = cfg["benchmark"]
    experiment_cfg = cfg["experiment"]
    component_cfgs = _component_items_in_order(cfg, _method_metadata_order(cfg))

    fragments = [
        str(fragment)
        for _, component_cfg in component_cfgs
        for fragment in [
            _component_metadata_value(
                component_cfg,
                benchmark_cfg["id"],
                "description_fragment",
                "description_fragment_by_benchmark",
            )
        ]
        if fragment
    ]
    description = str(benchmark_cfg["description_prefix"])
    if fragments:
        description = f"{description} {' and '.join(fragments)}"

    experiment_id = str(experiment_cfg["id"])
    if experiment_id == "sweep_tsil_coef":
        description = (
            f"{description} with replay coefficient "
            f'{_stringify(train_arg_map.get("sil_coef"))} in {cfg["method"]["sil"]["id"]} mode'
        )
    elif experiment_id == "sweep_suc_rew":
        description = (
            f"{description} with reward scale "
            f'{_stringify(_reward_scale_from_args(train_arg_map))}'
        )
    else:
        experiment_suffix = experiment_cfg.get("description_suffix")
        if experiment_suffix:
            description = f"{description} with {experiment_suffix}"

    return description


def _merge_arg_maps(target: dict[str, Any], arg_map: Mapping[str, Any] | None) -> None:
    if not arg_map:
        return
    for key, value in arg_map.items():
        target[str(key)] = value


def _resolve_force_args(cfg: Mapping[str, Any]) -> dict[str, Any]:
    force_args = dict(cfg.get("force_args") or {})
    legacy_args = dict(cfg.get("extra_train_args") or {})
    if force_args and legacy_args:
        raise ValueError("Use either force_args or extra_train_args, not both")
    return force_args or legacy_args


def _validate_force_args(force_args: Mapping[str, Any] | None) -> None:
    if not force_args:
        return
    owned = sorted(name for name in OWNED_TRAIN_ARG_NAMES if name in force_args)
    if owned:
        joined = ", ".join(owned)
        raise ValueError(
            f"force_args may not override launcher-owned training args: {joined}"
        )


def build_training_arg_map(
    cfg: Mapping[str, Any],
    task_id: int | None = None,
    seed: int | None = None,
    script_time: str | None = None,
) -> dict[str, Any]:
    task_id = resolve_task_id(cfg) if task_id is None else int(task_id)
    seed = resolve_seed(cfg) if seed is None else int(seed)
    script_time = resolve_script_time(cfg) if script_time is None else str(script_time)

    force_args = _resolve_force_args(cfg)
    _validate_force_args(force_args)

    arg_map: dict[str, Any] = {}
    _merge_arg_maps(arg_map, cfg["train"].get("args"))
    _merge_arg_maps(arg_map, cfg["benchmark"].get("args"))
    _merge_arg_maps(arg_map, cfg["experiment"].get("args"))
    _merge_arg_maps(arg_map, cfg["training_stage"].get("args"))
    for _, component_cfg in _component_items_in_order(cfg, _method_components(cfg)):
        _merge_arg_maps(arg_map, component_cfg.get("args"))
    _merge_arg_maps(arg_map, force_args)

    method_suffix = build_method_suffix(cfg, arg_map)
    if method_suffix:
        arg_map["method_suffix"] = method_suffix

    arg_map["task_name"] = task_name_for(cfg, task_id)
    arg_map["script_time"] = script_time
    arg_map["tasks"] = task_id
    arg_map["seed"] = seed

    if (
        arg_map.get("result_benchmark_name") is not None
        and arg_map.get("result_experiment_name") is not None
        and arg_map.get("result_training_stage_name") is not None
    ):
        arg_map["result_task_dir_name"] = benchmark_task_dir_name(task_id)

    return arg_map


def build_training_argv(
    cfg: Mapping[str, Any],
    task_id: int | None = None,
    seed: int | None = None,
    script_time: str | None = None,
) -> list[str]:
    return render_arg_map(build_training_arg_map(cfg, task_id=task_id, seed=seed, script_time=script_time))


def build_training_command(
    cfg: Mapping[str, Any],
    task_id: int | None = None,
    seed: int | None = None,
    script_time: str | None = None,
    python_executable: str | None = None,
) -> list[str]:
    python_bin = sys.executable if python_executable is None else str(python_executable)
    return [
        python_bin,
        "-u",
        "-m",
        str(cfg["train"]["python_module"]),
        *build_training_argv(cfg, task_id=task_id, seed=seed, script_time=script_time),
    ]


def summarize_selection(cfg: Mapping[str, Any], task_id: int | None = None, seed: int | None = None) -> str:
    task_id = resolve_task_id(cfg) if task_id is None else int(task_id)
    seed = resolve_seed(cfg) if seed is None else int(seed)
    components = " ".join(
        f"{component_name}={cfg['method'][component_name]['id']}"
        for component_name in _method_components(cfg)
    )
    return (
        f"benchmark={cfg['benchmark']['id']} experiment={cfg['experiment']['id']} "
        f"training_stage={cfg['training_stage']['id']} task_id={task_id} seed={seed} {components}"
    )


def command_string(command: Iterable[str]) -> str:
    return shlex.join(list(command))


def cloned_cfg_with_updates(cfg: Mapping[str, Any], updates: Mapping[str, Any]) -> DictConfig:
    cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    return OmegaConf.merge(cfg_copy, OmegaConf.create(updates))
