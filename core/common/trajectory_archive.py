import json
import os
from typing import Any, Callable, Dict, List

import h5py
import numpy as np


def _to_h5_compatible(value: Any) -> Any:
    if isinstance(value, str):
        return np.bytes_(value)
    if isinstance(value, (list, tuple)):
        value = np.asarray(value)
    if isinstance(value, np.ndarray) and value.dtype.kind in {"U", "O"}:
        return value.astype("S")
    return value


def _save_h5_group(group: h5py.Group, data: Dict[str, Any]) -> None:
    for key, value in data.items():
        if isinstance(value, dict):
            subgroup = group.create_group(key)
            _save_h5_group(subgroup, value)
            continue

        value = _to_h5_compatible(value)
        if isinstance(value, np.ndarray) and value.ndim > 0:
            group.create_dataset(key, data=value, compression="gzip")
        else:
            group.create_dataset(key, data=value)


def save_episode_record_h5(data: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as h5_file:
        _save_h5_group(h5_file, data)


def _load_h5_node(node: Any) -> Any:
    if isinstance(node, h5py.Group):
        return {key: _load_h5_node(node[key]) for key in node.keys()}

    value = node[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind == "S":
        return value.astype(str)
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def load_episode_record_h5(path: str) -> Dict[str, Any]:
    with h5py.File(path, "r") as h5_file:
        return _load_h5_node(h5_file)


def _validate_archive_index(index: Dict[str, Any]) -> Dict[str, Any]:
    for task_key, task_data in index.get("tasks", {}).items():
        if "reward_topk" in task_data:
            raise ValueError(
                "Unsupported trajectory archive format: found legacy 'reward_topk' bucket "
                f"for task {task_key}. Re-run training/evaluation with a new archive that uses 'stage_high_returns'."
            )
    return index


def load_archive_index(source_dir: str) -> Dict[str, Any]:
    index_path = os.path.join(source_dir, "archive_index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Archive index not found: {index_path}")
    with open(index_path, "r") as file_obj:
        return _validate_archive_index(json.load(file_obj))


class PerTaskTrajectoryArchive:
    """Small disk-only archive for env0 visualization/debug replay.

    This is intentionally separate from the SIL in-memory archive.  It stores
    at most ten indexed entries per task: first success, one fast success per
    training stage, one high-return trajectory per stage, and one last
    trajectory per stage.
    """

    BUCKETS = ("first_success", "stage_fast_successes", "stage_high_returns", "last")

    def __init__(self, trajectory_dir: str, saving: bool = True, stage_targets=None):
        self.trajectory_dir = trajectory_dir
        self.stage_targets = tuple(float(value) for value in (stage_targets or (0.25, 0.50, 1.00)))
        self.saving = bool(saving)
        self.archive_dir = os.path.join(trajectory_dir, "episode_archives")
        self.index_path = os.path.join(trajectory_dir, "archive_index.json")
        self._entry_counter = 0
        self.dirty = False
        self.index = {
            "topk": len(self.stage_targets),
            "stage_targets": list(self.stage_targets),
            "entry_counter": self._entry_counter,
            "tasks": {},
        }

        if self.saving:
            os.makedirs(self.archive_dir, exist_ok=True)
        if self.saving and os.path.exists(self.index_path):
            with open(self.index_path, "r") as file_obj:
                self.index = _validate_archive_index(json.load(file_obj))
            self._entry_counter = int(self.index.get("entry_counter", 0))
            self.index["stage_targets"] = list(self.stage_targets)
            self.index["topk"] = len(self.stage_targets)

    def _task_data(self, task_id: int) -> Dict[str, List[Dict[str, Any]]]:
        return self.index["tasks"].setdefault(str(int(task_id)), {})

    def _task_bucket(self, task_id: int, bucket: str) -> List[Dict[str, Any]]:
        return self._task_data(task_id).setdefault(bucket, [])

    def _make_entry_path(self, task_id: int) -> str:
        rel_path = os.path.join(
            "episode_archives",
            f"task_{int(task_id):02d}",
            f"episode_{self._entry_counter:08d}.h5",
        )
        self._entry_counter += 1
        self.index["entry_counter"] = self._entry_counter
        self.dirty = True
        return rel_path

    def _materialize_candidate(
        self,
        candidate: Dict[str, Any],
        materialize_record_fn: Callable[[Dict[str, Any]], Any],
    ) -> str:
        if candidate.get("path"):
            return candidate["path"]

        rel_path = self._make_entry_path(int(candidate["task_id"]))
        materialized = materialize_record_fn(candidate)
        disk_record = materialized[0] if isinstance(materialized, tuple) and len(materialized) == 2 else materialized
        if self.saving:
            save_episode_record_h5(disk_record, os.path.join(self.trajectory_dir, rel_path))
        candidate["path"] = rel_path
        return rel_path

    def _summary_progress(self, summary: Dict[str, Any]) -> float:
        progress = summary.get("training_progress", summary.get("progress", 1.0))
        try:
            progress = float(progress)
        except (TypeError, ValueError):
            progress = 1.0
        return min(max(progress, 0.0), 1.0)

    def _stage_index(self, summary: Dict[str, Any]) -> int:
        if "archive_stage_index" in summary:
            return int(summary["archive_stage_index"])
        progress = self._summary_progress(summary)
        for stage_idx, target in enumerate(self.stage_targets):
            if progress <= target + 1e-8:
                return stage_idx
        return len(self.stage_targets) - 1

    def _stage_label(self, stage_idx: int) -> str:
        return f"{int(round(self.stage_targets[int(stage_idx)] * 100)):03d}pct"

    def _tag_candidate_stage(self, candidate: Dict[str, Any]) -> int:
        summary = candidate["summary"]
        stage_idx = self._stage_index(summary)
        summary["archive_stage_index"] = int(stage_idx)
        summary["archive_stage_target"] = float(self.stage_targets[stage_idx])
        summary["archive_stage_label"] = self._stage_label(stage_idx)
        return stage_idx

    def _all_referenced_paths(self) -> set:
        paths = set()
        for task_data in self.index.get("tasks", {}).values():
            for entries in task_data.values():
                paths.update(entry.get("path") for entry in entries if entry.get("path"))
        return paths

    def _cleanup_unreferenced_paths(self, maybe_removed_paths) -> None:
        referenced_paths = self._all_referenced_paths()
        for rel_path in maybe_removed_paths:
            if not rel_path or rel_path in referenced_paths:
                continue
            abs_path = os.path.join(self.trajectory_dir, rel_path)
            if self.saving and os.path.exists(abs_path):
                os.remove(abs_path)

    @staticmethod
    def _entries_equivalent(old_entries: List[Dict[str, Any]], new_entries: List[Dict[str, Any]]) -> bool:
        if len(old_entries) != len(new_entries):
            return False
        for old_entry, new_entry in zip(old_entries, new_entries):
            if old_entry.get("path") != new_entry.get("path"):
                return False
            if float(old_entry.get("score", 0.0)) != float(new_entry.get("score", 0.0)):
                return False
            if old_entry.get("summary") != new_entry.get("summary"):
                return False
        return True

    def _candidate_entry(self, candidate: Dict[str, Any], score: float) -> Dict[str, Any]:
        return {
            "path": candidate["path"],
            "score": float(score),
            "summary": dict(candidate["summary"]),
        }

    def _set_bucket_entries(self, task_id: int, bucket: str, new_entries: List[Dict[str, Any]]) -> bool:
        entries = self._task_bucket(task_id, bucket)
        if self._entries_equivalent(entries, new_entries):
            return False
        removed_paths = [entry.get("path") for entry in entries]
        entries[:] = new_entries
        self._cleanup_unreferenced_paths(removed_paths)
        self.dirty = True
        return True

    def _update_stage_bucket(
        self,
        task_id: int,
        bucket: str,
        candidate: Dict[str, Any],
        score: float,
        materialize_record_fn: Callable[[Dict[str, Any]], Any],
        prefer_lowest: bool = False,
    ) -> bool:
        stage_idx = self._stage_index(candidate["summary"])
        entries = list(self._task_bucket(task_id, bucket))
        existing_idx = None
        for idx, entry in enumerate(entries):
            if self._stage_index(entry.get("summary", {})) == stage_idx:
                existing_idx = idx
                break

        if existing_idx is not None:
            old_score = float(entries[existing_idx].get("score", 0.0))
            if prefer_lowest and float(score) >= old_score:
                return False
            if not prefer_lowest and float(score) <= old_score:
                return False
            del entries[existing_idx]

        self._materialize_candidate(candidate, materialize_record_fn)
        entries.append(self._candidate_entry(candidate, float(score)))
        entries.sort(key=lambda entry: self._stage_index(entry.get("summary", {})))
        return self._set_bucket_entries(task_id, bucket, entries)

    def consider_env0_debug_episode(
        self,
        candidate: Dict[str, Any],
        materialize_record_fn: Callable[[Dict[str, Any]], Any],
    ) -> bool:
        if int(candidate.get("env_id", -1)) != 0 or candidate.get("snapshot") is None:
            return False

        task_id = int(candidate["task_id"])
        summary = candidate["summary"]
        self._tag_candidate_stage(candidate)

        updated = False
        if bool(summary.get("success", False)) and not self._task_bucket(task_id, "first_success"):
            self._materialize_candidate(candidate, materialize_record_fn)
            entry = self._candidate_entry(candidate, float(summary.get("global_episodes", 0)))
            updated = self._set_bucket_entries(task_id, "first_success", [entry]) or updated

        updated = self._update_stage_bucket(
            task_id,
            "stage_high_returns",
            candidate,
            float(summary.get("episode_return_raw", 0.0)),
            materialize_record_fn,
        ) or updated

        if bool(summary.get("success", False)):
            updated = self._update_stage_bucket(
                task_id,
                "stage_fast_successes",
                candidate,
                float(summary.get("eps_time", float("inf"))),
                materialize_record_fn,
                prefer_lowest=True,
            ) or updated

        updated = self._update_stage_bucket(
            task_id,
            "last",
            candidate,
            float(summary.get("global_episodes", 0)),
            materialize_record_fn,
        ) or updated

        return updated

    def save_index(self) -> None:
        if not self.saving:
            self.dirty = False
            return
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, "w") as file_obj:
            json.dump(self.index, file_obj, indent=2)
        self.dirty = False

    def get_bucket_entry(self, task_id: int, bucket: str, rank: int = 1) -> Dict[str, Any]:
        entries = self.index.get("tasks", {}).get(str(int(task_id)), {}).get(bucket, [])
        if rank < 1 or rank > len(entries):
            return None
        return entries[rank - 1]

    def num_tasks_with_success(self, task_ids=None) -> int:
        if task_ids is None:
            task_keys = list(self.index.get("tasks", {}).keys())
        else:
            task_keys = [str(int(task_id)) for task_id in task_ids]
        return sum(
            1
            for task_key in task_keys
            if self.index.get("tasks", {}).get(task_key, {}).get("first_success")
        )
