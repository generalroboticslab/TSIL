"""Trajectory replay and replay-frame export helpers."""


def __getattr__(name):
    if name == "TrajectoryReplayer":
        from core.evaluation.replay.trajectory_replay import TrajectoryReplayer
        return TrajectoryReplayer
    if name in ("remove_border_black", "process_path"):
        from core.evaluation.replay import background as _background
        return getattr(_background, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TrajectoryReplayer",
    "remove_border_black",
    "process_path",
]
