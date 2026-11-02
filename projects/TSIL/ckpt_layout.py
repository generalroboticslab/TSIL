"""TSIL checkpoint/result-layout helpers."""

from __future__ import annotations

import json
import os
import re
from typing import List, Optional


BENCHMARKS = {"mt01", "mt15"}
DEFAULT_TRAINING_STAGE = "scratch"
DEFAULT_EXPERIMENT = "compare_temporal"
_TIMESTAMP_RE = re.compile(r"(20\d{6})(?:[_-]?(\d{6}))?")


def normalize_benchmark(benchmark: str) -> str:
    benchmark = str(benchmark).strip().lower()
    if benchmark not in BENCHMARKS:
        raise ValueError(
            f"Unsupported benchmark '{benchmark}'. "
            f"Expected one of {sorted(BENCHMARKS)}."
        )
    return benchmark


def normalize_experiment(experiment: Optional[str]) -> str:
    experiment = "" if experiment is None else str(experiment).strip()
    return experiment or DEFAULT_EXPERIMENT


def benchmark_task_dir_name(task_id: int) -> str:
    return f"T{int(task_id)}"


def benchmark_task_dir(
    train_res_dir: str,
    benchmark: str,
    experiment: str,
    train_stage: str,
    task_id: int,
) -> str:
    return os.path.join(
        train_res_dir,
        normalize_benchmark(benchmark),
        normalize_experiment(experiment),
        str(train_stage),
        benchmark_task_dir_name(task_id),
    )


def benchmark_task_path_label(
    benchmark: str,
    experiment: str,
    train_stage: str,
    task_id: int,
) -> str:
    return os.path.join(
        normalize_benchmark(benchmark),
        normalize_experiment(experiment),
        str(train_stage),
        benchmark_task_dir_name(task_id),
    )


def discover_benchmark_task_methods(
    train_res_dir: str,
    benchmark: str,
    task_ids: List[int],
    train_stage: str = DEFAULT_TRAINING_STAGE,
    methods: Optional[List[str]] = None,
    experiment: str = DEFAULT_EXPERIMENT,
) -> dict:
    """Discover run folders for a benchmark/task grid."""
    benchmark = normalize_benchmark(benchmark)
    experiment = normalize_experiment(experiment)
    result = {}

    if methods is None:
        for tid in task_ids:
            task_dir = benchmark_task_dir(
                train_res_dir,
                benchmark,
                experiment,
                train_stage,
                tid,
            )
            if os.path.isdir(task_dir):
                methods = sorted(
                    name
                    for name in os.listdir(task_dir)
                    if os.path.isdir(os.path.join(task_dir, name))
                )
                break
        if methods is None:
            raise FileNotFoundError(
                "Could not auto-discover methods under benchmark "
                f"'{benchmark}', experiment '{experiment}', and stage '{train_stage}'."
            )

    for tid in task_ids:
        result[tid] = {}
        task_dir = benchmark_task_dir(
            train_res_dir,
            benchmark,
            experiment,
            train_stage,
            tid,
        )
        for method in methods:
            method_dir = os.path.join(task_dir, method)
            if not os.path.isdir(method_dir):
                result[tid][method] = []
                continue
            latest = _latest_script_time_dir(method_dir)
            result[tid][method] = (
                [] if latest is None else _run_folders_in_dir(latest)
            )

    return result


def _path_recency_key(path: str):
    names = [os.path.basename(path)]
    try:
        names.extend(os.listdir(path))
    except OSError:
        pass
    timestamps = [
        (match.group(1), match.group(2) or "")
        for name in names
        for match in _TIMESTAMP_RE.finditer(name)
    ]
    timestamp = max(timestamps) if timestamps else ("", "")
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    return timestamp + (mtime, os.path.basename(path))


def _immediate_subdirs(root_dir: str) -> List[str]:
    return sorted(
        (
            os.path.join(root_dir, name)
            for name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, name))
        ),
        key=_path_recency_key,
    )


def _timestamp_key_from_text(text: str) -> Optional[tuple[str, str]]:
    matches = [
        (match.group(1), match.group(2) or "")
        for match in _TIMESTAMP_RE.finditer(str(text))
    ]
    return max(matches) if matches else None


def _max_script_time_key() -> Optional[tuple[str, str]]:
    value = os.environ.get("TEMPORALSIL_MAX_SCRIPT_TIME", "").strip()
    return _timestamp_key_from_text(value) if value else None


def _filter_script_time_dirs(script_time_dirs: List[str]) -> List[str]:
    max_key = _max_script_time_key()
    if max_key is None:
        return script_time_dirs
    return [
        path for path in script_time_dirs
        if (key := _timestamp_key_from_text(os.path.basename(path))) is not None
        and key <= max_key
    ]


def latest_script_time_dir(method_dir: str) -> Optional[str]:
    script_time_dirs = _filter_script_time_dirs(_immediate_subdirs(method_dir))
    if not script_time_dirs:
        return None
    if _allow_incomplete_results():
        return script_time_dirs[-1]
    for script_time_dir in reversed(script_time_dirs):
        run_folders = _run_folders_in_dir(script_time_dir)
        if run_folders and all(
            _run_folder_complete(run_folder) for run_folder in run_folders
        ):
            return script_time_dir
    return None


def _latest_script_time_dir(method_dir: str) -> Optional[str]:
    return latest_script_time_dir(method_dir)


def _run_folders_in_dir(root_dir: str) -> List[str]:
    return [
        path for path in _immediate_subdirs(root_dir)
        if os.path.isfile(os.path.join(path, "config.json"))
    ]


def _allow_incomplete_results() -> bool:
    return os.environ.get("TIMEAWARE_ALLOW_INCOMPLETE_RESULTS", "").lower() in {
        "1",
        "true",
        "yes",
    }


def _run_folder_complete(run_folder: str) -> bool:
    try:
        with open(os.path.join(run_folder, "config.json"), "r") as file_obj:
            config = json.load(file_obj)
        batch_size = int(
            config.get("batch_size") or int(config["num_envs"]) * int(config["num_steps"])
        )
        total_timesteps = config.get("total_timesteps")
        if total_timesteps is None and config.get("total_iters") is not None:
            total_timesteps = int(config["total_iters"]) * batch_size
        expected_steps = max((int(total_timesteps) // batch_size) * batch_size, batch_size)
        with open(os.path.join(run_folder, "plot_metrics_history.jsonl"), "r") as file_obj:
            last_line = ""
            for line in file_obj:
                if line.strip():
                    last_line = line
        last_record = json.loads(last_line)
        last_steps = last_record.get("misc/steps", last_record.get("steps"))
        return last_steps is not None and float(last_steps) >= expected_steps
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
