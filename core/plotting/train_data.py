"""Training-run discovery, history loading, caching, and curve interpolation."""
from __future__ import annotations

import json
import os
import re
from typing import List, Optional

import numpy as np


def _wandb_project_path(project: str, entity: Optional[str] = None) -> str:
    return f"{entity}/{project}" if entity else project


# ═══════════════════════════════════════════════════════════════════════
#  Run-folder discovery & time filtering
# ═══════════════════════════════════════════════════════════════════════

def filter_run_folders_by_time(
    run_folders: List[str],
    start_time: str = None,
    end_time: str = None,
) -> List[str]:
    """Keep only run folders whose YYYYMMDD_HHMMSS prefix falls in [start, end]."""
    if start_time is None and end_time is None:
        return run_folders

    def _normalise(ts: str, is_end: bool = False) -> str:
        ts = ts.replace("-", "").replace(":", "").replace(" ", "_")
        if len(ts) == 8:
            return ts + ("_235959" if is_end else "_000000")
        return ts

    lo = _normalise(start_time, is_end=False) if start_time else "00000000_000000"
    hi = _normalise(end_time, is_end=True) if end_time else "99999999_999999"

    filtered = []
    for folder in run_folders:
        basename = os.path.basename(folder)
        if (
            len(basename) >= 15
            and basename[8] == "_"
            and basename[:8].isdigit()
            and basename[9:15].isdigit()
        ):
            ts = basename[:15]
        else:
            filtered.append(folder)
            continue
        if lo <= ts <= hi:
            filtered.append(folder)

    if len(filtered) < len(run_folders):
        print(
            f"  Time filter [{start_time} → {end_time}]: "
            f"kept {len(filtered)}/{len(run_folders)} runs"
        )
    return filtered


def _immediate_subdirs(root_dir: str) -> List[str]:
    return sorted(
        os.path.join(root_dir, name)
        for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name))
    )


def _run_folders_in_dir(root_dir: str) -> List[str]:
    return [
        path for path in _immediate_subdirs(root_dir)
        if os.path.isfile(os.path.join(path, "config.json"))
    ]


def _normalize_method_name(method_name: str) -> str:
    mapping = {
        "": "Vanilla",
        "none": "Vanilla",
        "fh": "FH",
        "pt": "Vanilla",
    }
    return mapping.get(method_name.lower(), method_name)


def _strip_stage_tokens(name_tail: str) -> str:
    tail = name_tail
    for stage in ("PT", "FT", "Stu"):
        if tail == stage:
            tail = ""
        elif tail.startswith(f"{stage}_"):
            tail = tail[len(stage) + 1:]
        elif tail.endswith(f"_{stage}"):
            tail = tail[: -(len(stage) + 1)]
        elif tail.startswith(stage):
            tail = tail[len(stage) :]
        elif tail.endswith(stage):
            tail = tail[: -len(stage)]
    return tail.strip("_")


def infer_train_group_path(eval_uni_name: str) -> Optional[str]:
    """Best-effort mapping from legacy short names to project train-result groups."""
    if os.sep in eval_uni_name:
        return None

    match = re.fullmatch(r"MT1ddl_T(\d+)", eval_uni_name)
    if match:
        return os.path.join(f"MT1_T{match.group(1)}", "PreT", "DDL")

    match = re.fullmatch(r"(MT1_T\d+)_([A-Za-z0-9_]+)", eval_uni_name)
    if match:
        return os.path.join(
            match.group(1),
            "PreT",
            _normalize_method_name(match.group(2)),
        )

    match = re.fullmatch(r"(MT1_T\d+)", eval_uni_name)
    if match:
        return os.path.join(match.group(1), "PreT", "Vanilla")

    match = re.fullmatch(r"(MT\d+)(.*)", eval_uni_name)
    if not match:
        return None

    task_name = match.group(1)
    tail = match.group(2).lstrip("_")
    train_stage = "PreT"
    if "Stu" in tail:
        train_stage = "Stu"
    elif "FT" in tail or "Rcritic" in tail:
        train_stage = "FT"

    method_name = _normalize_method_name(_strip_stage_tokens(tail))
    return os.path.join(task_name, train_stage, method_name)


def discover_train_run_folders_from_dir(
    group_dir: str,
    start_time: str = None,
    end_time: str = None,
    relative_hint: str = None,
) -> List[str]:
    """Resolve a project train-result directory to the leaf run folders it contains."""
    if not os.path.isdir(group_dir):
        raise FileNotFoundError(f"Train results directory not found: {group_dir}")

    if os.path.isfile(os.path.join(group_dir, "config.json")):
        run_folders = [group_dir]
    else:
        run_folders = _run_folders_in_dir(group_dir)

    rel = relative_hint or group_dir
    path_parts = [p for p in os.path.normpath(rel).split(os.sep) if p and p != "."]

    if not run_folders and len(path_parts) >= 3:
        for script_time_dir in _immediate_subdirs(group_dir):
            run_folders.extend(_run_folders_in_dir(script_time_dir))

    if not run_folders:
        expected = "task/train_stage/method[/script_time]"
        raise FileNotFoundError(
            f"No run folders found under {group_dir}. "
            f"Expected evalUniName like {expected}."
        )
    return filter_run_folders_by_time(sorted(run_folders), start_time, end_time)


def discover_train_run_folders(
    train_res_dir: str,
    eval_uni_name: str,
    start_time: str = None,
    end_time: str = None,
) -> List[str]:
    """Resolve *eval_uni_name* against the project train-result root and return leaf run folders."""
    rel_path = os.path.normpath(eval_uni_name.strip())
    if not rel_path or rel_path == ".":
        raise ValueError("evalUniName must not be empty")

    group_dir = os.path.join(train_res_dir, rel_path)
    if not os.path.isdir(group_dir):
        inferred_rel_path = infer_train_group_path(rel_path)
        if inferred_rel_path is not None:
            inferred_dir = os.path.join(train_res_dir, inferred_rel_path)
            if os.path.isdir(inferred_dir):
                print(
                    f"Resolved legacy evalUniName '{eval_uni_name}' "
                    f"-> '{inferred_rel_path}'"
                )
                group_dir = inferred_dir
                rel_path = inferred_rel_path

    return discover_train_run_folders_from_dir(
        group_dir,
        start_time=start_time,
        end_time=end_time,
        relative_hint=rel_path,
    )


# ═══════════════════════════════════════════════════════════════════════
#  WandB data fetching & caching
# ═══════════════════════════════════════════════════════════════════════

def _wandb_cache_path(run_folder: str, x_key: str, metric_key: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", f"{x_key}__{metric_key}")
    cache_dir = os.path.join(run_folder, "wandb_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{safe}.npz")


def _load_wandb_cache(cache_path: str):
    if not os.path.isfile(cache_path):
        return None
    try:
        data = np.load(cache_path)
        x, y = data["x"], data["y"]
        if len(x) > 0:
            return x, y
    except Exception:
        pass
    return None


def _save_wandb_cache(cache_path: str, x: np.ndarray, y: np.ndarray):
    np.savez_compressed(cache_path, x=x, y=y)


def _plot_metric_candidates(metric_key: str) -> list[str]:
    candidates = [metric_key]
    aliases = {
        "signal/fast_success_rate": "signal/sil_fast_success_rate",
        "signal/fast_success_count": "signal/sil_fast_success_count",
        "signal/fast_revisit_gap_steps": "signal/sil_revisit_gap_steps",
        "signal/first_fast_revisit_steps": "signal/sil_first_revisit_steps",
    }
    if metric_key in aliases:
        candidates.append(aliases[metric_key])
    if "/" not in metric_key:
        candidates.extend([
            f"signal/{metric_key}",
            f"reward/{metric_key}",
            f"train/{metric_key}",
            f"misc/{metric_key}",
        ])
    return candidates


def _plot_x_candidates(x_key: str) -> list[str]:
    candidates = [x_key]
    aliases = {
        "iteration": ("misc/iterations",),
        "misc/iterations": ("iteration",),
        "steps": ("misc/steps",),
        "misc/steps": ("steps",),
        "interaction_time": ("misc/interaction_time",),
        "misc/interaction_time": ("interaction_time",),
    }
    for alias in aliases.get(x_key, ()):
        if alias not in candidates:
            candidates.append(alias)
    return candidates


def _success_eps_time_fill_value(run_folder: str) -> Optional[float]:
    try:
        with open(os.path.join(run_folder, "config.json"), "r") as f:
            config = json.load(f)
        return float(config["episodeLength"]) * _run_ctrl_dt(config)
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _run_ctrl_dt(config: dict) -> float:
    return float(config["dt"]) * float(config.get("control_freq_inv", 1))


def _run_ctrl_dt_from_folder(run_folder: str) -> Optional[float]:
    try:
        with open(os.path.join(run_folder, "config.json"), "r") as f:
            config = json.load(f)
        return _run_ctrl_dt(config)
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _derive_interaction_time(record: dict, ctrl_dt: Optional[float]) -> Optional[float]:
    if ctrl_dt is None:
        return None
    step_val = next(
        (record.get(key) for key in ("misc/steps", "steps") if record.get(key) is not None),
        None,
    )
    return None if step_val is None else float(step_val) * ctrl_dt


def _load_local_plot_metric_history(
    run_folder: str,
    x_key: str,
    metric_key: str,
) -> Optional[tuple]:
    history_path = os.path.join(run_folder, "plot_metrics_history.jsonl")
    if not os.path.isfile(history_path):
        return None

    x_candidates = _plot_x_candidates(x_key)
    y_candidates = _plot_metric_candidates(metric_key)
    derive_interaction_time = x_key in {"interaction_time", "misc/interaction_time"}
    ctrl_dt = _run_ctrl_dt_from_folder(run_folder) if derive_interaction_time else None
    fill_null_y = (
        _success_eps_time_fill_value(run_folder)
        if metric_key.split("/")[-1] == "success_eps_time" else None
    )
    xs, ys = [], []
    try:
        with open(history_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                x_val = next(
                    (record.get(key) for key in x_candidates if record.get(key) is not None),
                    None,
                )
                if x_val is None and derive_interaction_time:
                    x_val = _derive_interaction_time(record, ctrl_dt)
                y_key = next((key for key in y_candidates if key in record), None)
                y_val = record.get(y_key) if y_key is not None else None
                if y_val is None and y_key is not None and fill_null_y is not None:
                    y_val = fill_null_y
                if x_val is None or y_val is None:
                    continue
                xs.append(float(x_val))
                ys.append(float(y_val))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None

    if not xs:
        return None

    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not mask.any():
        return None
    order = np.argsort(x_arr[mask])
    return x_arr[mask][order], y_arr[mask][order]


def _download_wandb_run(
    run,
    x_key: str,
    metric_key: str,
    cache_path: Optional[str] = None,
):
    """Download a single wandb run's training curve via ``scan_history``."""
    try:
        history = run.scan_history()
        x_vals, y_vals = [], []
        last_x = None
        for row in history:
            if x_key in row and row[x_key] is not None:
                last_x = row[x_key]
            if metric_key in row and row[metric_key] is not None:
                xv = row.get(x_key) if row.get(x_key) is not None else last_x
                if xv is not None:
                    x_vals.append(xv)
                    y_vals.append(row[metric_key])
        x_arr = np.array(x_vals, dtype=float)
        y_arr = np.array(y_vals, dtype=float)

        if len(x_arr) == 0:
            return None

        if cache_path is not None:
            _save_wandb_cache(cache_path, x_arr, y_arr)
        return x_arr, y_arr
    except Exception as e:
        print(f"    Warning: download failed for {run.name}: {e}")
        return None


def _load_curve_from_wandb(
    run_folder: str,
    project: str,
    entity: Optional[str],
    x_key: str,
    metric_key: str,
    api=None,
    use_cache: bool = True,
):
    """Load a single run's wandb curve, with local .npz caching."""
    from core.common.io import read_json

    cache_file = _wandb_cache_path(run_folder, x_key, metric_key)
    if use_cache:
        cached = _load_wandb_cache(cache_file)
        if cached is not None:
            print(
                f"    [cache hit] {os.path.basename(run_folder)} "
                f"({len(cached[0])} pts)"
            )
            return cached

    config_path = os.path.join(run_folder, "config.json")
    if not os.path.isfile(config_path):
        return None
    config = read_json(config_path)
    wandb_run_id = config.get("wandb_run_id")
    wandb_run_path = config.get("wandb_run_path")
    if isinstance(wandb_run_path, (list, tuple)):
        wandb_run_path = "/".join(str(part) for part in wandb_run_path)
    run_name = config.get("run_name")
    if wandb_run_id is None and run_name is None:
        return None

    if api is None:
        import wandb

        api = wandb.Api()

    run = None
    if wandb_run_id:
        candidate_paths = []
        if wandb_run_path:
            candidate_paths.append(str(wandb_run_path))
        candidate_paths.append(f"{_wandb_project_path(project, entity)}/{wandb_run_id}")
        for candidate_path in candidate_paths:
            try:
                run = api.run(candidate_path)
                break
            except Exception:
                pass
    if run is None and run_name:
        for r in api.runs(
            _wandb_project_path(project, entity),
            filters={"display_name": run_name},
        ):
            run = r
            break
    if run is None:
        print(f"    Could not find wandb run for {os.path.basename(run_folder)}")
        return None

    result = _download_wandb_run(
        run,
        x_key,
        metric_key,
        cache_path=cache_file,
    )
    if result is not None:
        print(f"    [downloaded] {run.name} ({len(result[0])} pts) → cached")
    else:
        print(f"    Warning: no data points in {metric_key} for {os.path.basename(run_folder)}")
    return result


def _load_run_curve(
    run_folder: str,
    project: str,
    entity: Optional[str],
    x_key: str,
    metric_key: str,
    api=None,
    use_cache: bool = True,
):
    """Load one curve from local plot history, old signal history, then WandB."""
    result = _load_local_plot_metric_history(run_folder, x_key, metric_key)
    if result is None and metric_key.startswith("signal/"):
        result = _load_local_signal_history(run_folder, metric_key.split("/", 1)[1])
    if result is not None:
        return result, api
    if metric_key.split("/")[-1] == "success_episodes":
        return None, api

    if api is None:
        import wandb
        api = wandb.Api()
    
    result = _load_curve_from_wandb(
        run_folder,
        project,
        entity,
        x_key,
        metric_key,
        api=api,
        use_cache=use_cache,
    )
    return result, api


def load_curves_from_wandb_run_name(
    project: str,
    entity: Optional[str],
    run_name_patterns: List[str],
    metric_key: str = "reward/success",
    x_key: str = "misc/s_steps",
) -> dict:
    """Load training curves from wandb runs matching name patterns."""
    import wandb

    api = wandb.Api()
    results = {}
    for pattern in run_name_patterns:
        runs_data = []
        runs = api.runs(
            _wandb_project_path(project, entity),
            filters={"display_name": {"$regex": pattern}},
        )
        for run in runs:
            result = _download_wandb_run(run, x_key, metric_key)
            if result is not None:
                runs_data.append(result)
                print(f"  Loaded run: {run.name} ({len(result[0])} pts)")
        results[pattern] = runs_data
        print(f"Pattern '{pattern}': Found {len(runs_data)} runs")
    return results


def load_curves_from_run_dirs(
    train_res_dirs: List[str],
    project: str = "TSIL",
    entity: Optional[str] = None,
    metric_key: str = "reward/success",
    x_key: str = "misc/steps",
    use_scan_history: bool = False,
    start_time: str = None,
    end_time: str = None,
    use_cache: bool = True,
) -> dict:
    """Load training curves from run dirs, using local history then WandB."""
    api = None

    results = {}
    for train_dir in train_res_dirs:
        dir_name = os.path.basename(train_dir)
        runs_data = []

        if not os.path.exists(train_dir):
            print(f"Warning: Directory {train_dir} does not exist")
            continue

        try:
            run_folders = discover_train_run_folders_from_dir(
                train_dir,
                start_time=start_time,
                end_time=end_time,
            )
        except FileNotFoundError as exc:
            print(f"Warning: {exc}")
            continue

        for run_folder in run_folders:
            result, api = _load_run_curve(
                run_folder,
                project,
                entity,
                x_key,
                metric_key,
                api,
                use_cache,
            )
            if result is not None:
                runs_data.append(result)

        results[dir_name] = runs_data
        print(f"Directory '{dir_name}': Loaded {len(runs_data)} runs")
    return results


# ═══════════════════════════════════════════════════════════════════════
#  Signal processing
# ═══════════════════════════════════════════════════════════════════════

def smooth_curve(y: np.ndarray, window_size: int) -> np.ndarray:
    """Moving-average smoothing with reflected-edge padding."""
    if window_size <= 1:
        return y
    if window_size % 2 == 0:
        window_size += 1
    half_window = window_size // 2
    y_padded = np.pad(
        y,
        (half_window, half_window),
        "constant",
        constant_values=(y[0], y[-1]),
    )
    kernel = np.ones(window_size) / window_size
    return np.convolve(y_padded, kernel, mode="valid")


def interpolate_to_common_x(
    runs_data: List[tuple],
    num_points: int = 1000,
) -> tuple:
    """Interpolate multiple runs to a common x-axis for averaging."""
    if len(runs_data) == 0:
        return None, None
    x_min = max(run[0].min() for run in runs_data)
    x_max = min(run[0].max() for run in runs_data)
    common_x = np.linspace(x_min, x_max, num_points)

    interpolated_ys = []
    for x_vals, y_vals in runs_data:
        idx = np.argsort(x_vals)
        xs, ys = x_vals[idx], y_vals[idx]
        ux, _ = np.unique(xs, return_index=True)
        uy = np.array([ys[xs == x].mean() for x in ux])
        interpolated_ys.append(np.interp(common_x, ux, uy))
    return common_x, np.array(interpolated_ys)


# ═══════════════════════════════════════════════════════════════════════
#  MT1 task/method discovery and loaders
# ═══════════════════════════════════════════════════════════════════════

def _latest_script_time_dir(method_dir: str) -> Optional[str]:
    """Return the latest (lexicographically largest) script_time sub-dir."""
    subdirs = _immediate_subdirs(method_dir)
    if not subdirs:
        return None
    return sorted(subdirs)[-1]


def discover_mt1_task_methods(
    train_res_dir: str,
    task_ids: List[int],
    train_stage: str = "PreT",
    methods: Optional[List[str]] = None,
) -> dict:
    """Discover run folders for each (task, method) pair."""
    result = {}

    if methods is None:
        for tid in task_ids:
            stage_dir = os.path.join(train_res_dir, f"MT1_T{tid}", train_stage)
            if os.path.isdir(stage_dir):
                methods = sorted(
                    name
                    for name in os.listdir(stage_dir)
                    if os.path.isdir(os.path.join(stage_dir, name))
                )
                break
        if methods is None:
            raise FileNotFoundError("Could not auto-discover methods")

    for tid in task_ids:
        result[tid] = {}
        for method in methods:
            method_dir = os.path.join(
                train_res_dir,
                f"MT1_T{tid}",
                train_stage,
                method,
            )
            if not os.path.isdir(method_dir):
                result[tid][method] = []
                continue
            latest = _latest_script_time_dir(method_dir)
            if latest is None:
                result[tid][method] = []
                continue
            result[tid][method] = _run_folders_in_dir(latest)

    return result


def load_mt1_curves_by_task_method(
    train_res_dir: str,
    task_ids: List[int],
    train_stage: str = "PreT",
    methods: Optional[List[str]] = None,
    project: str = "TSIL",
    entity: Optional[str] = None,
    metric_key: str = "reward/success",
    x_key: str = "misc/iterations",
    num_interp_points: int = 500,
    smoothing_window: int = 1,
    use_cache: bool = True,
) -> dict:
    """Load MT1 training curves by task and method."""
    api = None

    task_methods = discover_mt1_task_methods(
        train_res_dir,
        task_ids,
        train_stage,
        methods,
    )
    discovered_methods = (
        list(next(iter(task_methods.values())).keys())
        if task_methods
        else (methods or [])
    )

    grid_data: dict = {"__methods__": discovered_methods}

    for tid in task_ids:
        grid_data[tid] = {}
        for method in discovered_methods:
            run_folders = task_methods.get(tid, {}).get(method, [])
            if not run_folders:
                print(f"  MT1_T{tid}/{method}: no run folders found")
                grid_data[tid][method] = None
                continue

            runs_data = []
            for rf in run_folders:
                result, api = _load_run_curve(
                    rf,
                    project,
                    entity,
                    x_key,
                    metric_key,
                    api,
                    use_cache,
                )
                if result is not None:
                    runs_data.append(result)

            if not runs_data:
                grid_data[tid][method] = None
                continue

            common_x, interp_ys = interpolate_to_common_x(
                runs_data,
                num_interp_points,
            )
            if common_x is None:
                grid_data[tid][method] = None
                continue

            if smoothing_window > 1:
                interp_ys = np.array(
                    [smooth_curve(y, smoothing_window) for y in interp_ys]
                )

            mean_y = np.mean(interp_ys, axis=0)
            std_y = np.zeros_like(interp_ys[0]) if len(interp_ys) <= 1 else np.std(interp_ys, axis=0, ddof=1)
            grid_data[tid][method] = (common_x, mean_y, std_y, len(runs_data))
            print(
                f"  MT1_T{tid}/{method}: {len(runs_data)} run(s), "
                f"final={mean_y[-1]:.4f}±{std_y[-1]:.4f}"
            )

    return grid_data


def _resolve_signal_history_path(run_folder: str) -> Optional[str]:
    default_path = os.path.join(
        run_folder,
        "trajectories",
        "training_signal_history.jsonl",
    )
    if os.path.isfile(default_path):
        return default_path

    meta_path = os.path.join(run_folder, "trajectories", "meta_data.json")
    if not os.path.isfile(meta_path):
        return None

    try:
        with open(meta_path, "r") as f:
            meta_data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    history_relpath = meta_data.get("training_signal_metrics", {}).get("history_file")
    if not history_relpath:
        return None
    history_path = os.path.join(run_folder, "trajectories", history_relpath)
    return history_path if os.path.isfile(history_path) else None


def _load_local_signal_history(
    run_folder: str,
    metric_key: str,
    x_key: str = "iteration",
) -> Optional[tuple]:
    local_plot_history = _load_local_plot_metric_history(run_folder, x_key, metric_key)
    if local_plot_history is not None:
        return local_plot_history

    history_path = _resolve_signal_history_path(run_folder)
    if history_path is None:
        return None

    xs = []
    ys = []
    try:
        with open(history_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                x_val = record.get(x_key)
                y_val = _resolve_signal_metric_value(record, metric_key)
                if (
                    y_val is None
                    and metric_key == "success_eps_time"
                    and any(key in record for key in ("success_eps_time", "success_used_time"))
                ):
                    y_val = _success_eps_time_fill_value(run_folder)
                if x_val is None or y_val is None:
                    continue
                xs.append(float(x_val))
                ys.append(float(y_val))
    except (OSError, json.JSONDecodeError, ValueError):
        return None

    if len(xs) == 0:
        return None

    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not mask.any():
        return None

    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    order = np.argsort(x_arr)
    return x_arr[order], y_arr[order]


def _resolve_signal_metric_value(record: dict, metric_key: str) -> Optional[float]:
    value = record.get(metric_key)
    if value is not None:
        return float(value)

    top_dense_return_aliases = (
        "top10pct_Rdense_sr",
        "top_10perc_Rdense_sr",
        "dense_tail_purity_10",
    )
    if metric_key in top_dense_return_aliases:
        for alias in top_dense_return_aliases:
            value = record.get(alias)
            if value is not None:
                return float(value)
        return None

    if metric_key == "success_eps_time":
        value = record.get("success_used_time")
        return None if value is None else float(value)

    return None


def load_mt1_signal_curves_by_task_method(
    train_res_dir: str,
    task_ids: List[int],
    train_stage: str = "PreT",
    methods: Optional[List[str]] = None,
    metric_key: str = "top10pct_Rdense_sr",
    num_interp_points: int = 500,
    smoothing_window: int = 1,
) -> dict:
    """Load MT1 signal curves by task and method."""
    task_methods = discover_mt1_task_methods(
        train_res_dir,
        task_ids,
        train_stage,
        methods,
    )
    discovered_methods = (
        list(next(iter(task_methods.values())).keys())
        if task_methods
        else (methods or [])
    )

    grid_data: dict = {"__methods__": discovered_methods}

    for tid in task_ids:
        grid_data[tid] = {}
        for method in discovered_methods:
            run_folders = task_methods.get(tid, {}).get(method, [])
            if not run_folders:
                print(f"  MT1_T{tid}/{method}: no run folders found")
                grid_data[tid][method] = None
                continue

            runs_data = []
            for rf in run_folders:
                result = _load_local_signal_history(rf, metric_key)
                if result is not None:
                    runs_data.append(result)

            if not runs_data:
                print(f"  MT1_T{tid}/{method}: no local history for {metric_key}")
                grid_data[tid][method] = None
                continue

            common_x, interp_ys = interpolate_to_common_x(
                runs_data,
                num_interp_points,
            )
            if common_x is None:
                grid_data[tid][method] = None
                continue

            if smoothing_window > 1:
                interp_ys = np.array(
                    [smooth_curve(y, smoothing_window) for y in interp_ys]
                )

            mean_y = np.mean(interp_ys, axis=0)
            std_y = np.zeros_like(interp_ys[0]) if len(interp_ys) <= 1 else np.std(interp_ys, axis=0, ddof=1)
            grid_data[tid][method] = (common_x, mean_y, std_y, len(runs_data))
            print(
                f"  MT1_T{tid}/{method}: {len(runs_data)} run(s), "
                f"final={mean_y[-1]:.4f}±{std_y[-1]:.4f}"
            )

    return grid_data
