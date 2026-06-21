"""Common shared helpers."""


def __getattr__(name):
    if name in ("read_json", "save_json", "write_csv_line"):
        from core.common import io as _io
        return getattr(_io, name)
    if name in (
        "check_file_exist",
        "infer_task_name_from_checkpoint",
        "infer_task_name_from_run_dir",
        "resolve_checkpoint_run_dir",
    ):
        from core.common import checkpointing as _ckpt
        return getattr(_ckpt, name)
    if name in ("to_numpy", "replace_nan", "get_args_attr", "weighted_average"):
        from core.common import tensor as _tensor
        return getattr(_tensor, name)
    if name in (
        "PerTaskTrajectoryArchive",
        "load_archive_index",
        "load_episode_record_h5",
        "save_episode_record_h5",
    ):
        from core.common import trajectory_archive as _archive
        return getattr(_archive, name)
    if name == "_format_time":
        from core.common import time as _time
        return getattr(_time, name)
    if name == "build_task_id_lookup":
        from core.common.task_layout import build_task_id_lookup
        return build_task_id_lookup
    if name == "tf_utils":
        from core.common import tf_utils as _tf
        return _tf
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "read_json",
    "save_json",
    "write_csv_line",
    "check_file_exist",
    "infer_task_name_from_checkpoint",
    "infer_task_name_from_run_dir",
    "resolve_checkpoint_run_dir",
    "to_numpy",
    "replace_nan",
    "get_args_attr",
    "weighted_average",
    "PerTaskTrajectoryArchive",
    "load_archive_index",
    "load_episode_record_h5",
    "save_episode_record_h5",
    "_format_time",
    "build_task_id_lookup",
    "tf_utils",
]
