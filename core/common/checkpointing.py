"""Checkpoint path resolution and filesystem helpers."""

import json
import os
import shutil
from pathlib import Path


def _is_checkpoint_run_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def _checkpoint_candidate_to_run_dir(path: Path) -> Path:
    if _is_checkpoint_run_dir(path):
        return path.resolve()
    if path.is_file() and path.parent.name == "checkpoints":
        run_dir = path.parent.parent
        if _is_checkpoint_run_dir(run_dir):
            return run_dir.resolve()
    return None


def infer_task_name_from_run_dir(result_dir, run_dir):
    """Infer task_name from a resolved run folder under result_dir."""
    result_root = Path(result_dir).expanduser().resolve()
    run_path = Path(run_dir).expanduser().resolve()

    config_path = run_path / "config.json"
    if config_path.is_file():
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                task_name = json.load(handle).get("task_name")
        except (OSError, json.JSONDecodeError):
            task_name = None
        if task_name:
            return task_name

    try:
        rel_path = run_path.relative_to(result_root)
    except ValueError as exc:
        raise ValueError(
            f"Run directory '{run_dir}' is not inside result_dir '{result_root}'."
        ) from exc

    if len(rel_path.parts) < 1:
        raise ValueError(
            f"Unable to infer task_name from run directory '{run_dir}'."
        )

    return rel_path.parts[0]


def infer_task_name_from_checkpoint(result_dir, checkpoint):
    """Infer task name from a checkpoint spec when possible."""
    try:
        run_dir = resolve_checkpoint_run_dir(result_dir, None, checkpoint)
        return infer_task_name_from_run_dir(result_dir, run_dir)
    except (FileNotFoundError, FileExistsError, ValueError):
        pass

    checkpoint_path = Path(checkpoint).expanduser()
    if len(checkpoint_path.parts) > 1:
        return checkpoint_path.parts[0]

    checkpoint_parts = str(checkpoint).split("_")
    for stage_name in ("PreT", "FT", "TW", "Stu"):
        if stage_name in checkpoint_parts:
            stage_idx = checkpoint_parts.index(stage_name)
            if stage_idx > 1:
                return "_".join(checkpoint_parts[1:stage_idx])
            break

    raise ValueError(
        f"Unable to infer task_name from checkpoint '{checkpoint}'. Please pass --task_name explicitly."
    )


def resolve_checkpoint_run_dir(result_dir, task_name, checkpoint):
    """Resolve a checkpoint run folder from a path-like spec or bare run folder name."""
    result_root = Path(result_dir).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser()
    task_roots = []
    if task_name is not None:
        task_roots.append(result_root / task_name)

    unique_task_roots = []
    seen_roots = set()
    for task_root in task_roots:
        if task_root in seen_roots:
            continue
        seen_roots.add(task_root)
        unique_task_roots.append(task_root)

    candidate_dirs = []
    if checkpoint_path.is_absolute():
        candidate_dirs.append(checkpoint_path)
    else:
        candidate_dirs.append(result_root / checkpoint_path)
        for task_root in unique_task_roots:
            candidate_dirs.append(task_root / checkpoint_path)

    for candidate_dir in candidate_dirs:
        run_dir = _checkpoint_candidate_to_run_dir(candidate_dir)
        if run_dir is not None:
            return str(run_dir)

    search_roots = [task_root for task_root in unique_task_roots if task_root.is_dir()]
    if result_root not in search_roots:
        search_roots.append(result_root)

    suffix_parts = checkpoint_path.parts
    matches = []
    for search_root in search_roots:
        for path in search_root.rglob(checkpoint_path.name):
            if path.parts[-len(suffix_parts):] != suffix_parts:
                continue
            run_dir = _checkpoint_candidate_to_run_dir(path)
            if run_dir is not None:
                matches.append(run_dir)
            elif _is_checkpoint_run_dir(path):
                matches.append(path.resolve())
    matches = sorted(set(matches))

    if len(matches) == 1:
        return str(matches[0].resolve())
    if len(matches) > 1:
        rel_matches = [str(path.resolve().relative_to(result_root)) for path in matches]
        if len(search_roots) == 1 and search_roots[0] != result_root:
            scope = str(search_roots[0].resolve().relative_to(result_root))
        else:
            scope = str(result_root)
        raise FileExistsError(
            f"Checkpoint '{checkpoint}' is ambiguous under '{scope}'. Matches: {rel_matches}"
        )

    if task_name is None:
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint}' could not be found under '{result_root}'. "
            "Pass a fuller run-folder path or verify the run name."
        )
    raise FileNotFoundError(
        f"Checkpoint '{checkpoint}' could not be found for task_name '{task_name}' under '{result_root}'. "
        "Pass the full run-folder path or verify the task_name."
    )


def check_file_exist(file_path):
    if os.path.exists(file_path):
        response = input(f"Find existing dir/file {file_path}! Whether remove or not (y/n):")
        if response == 'y' or response == 'Y':
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
        else:
            raise Exception("Give up this evaluation because of exsiting file.")


__all__ = [
    "infer_task_name_from_checkpoint",
    "infer_task_name_from_run_dir",
    "resolve_checkpoint_run_dir",
    "check_file_exist",
]
