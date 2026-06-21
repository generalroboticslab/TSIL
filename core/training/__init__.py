"""Shared training package.

Components are imported from their sub-modules directly (e.g.
``from core.training.storage import RolloutStorage`` and
``from core.training.metrics import MetricTracker``).  Eager imports
of the heavy trainer module are avoided to prevent isaacgym/torch
import-order issues.
"""

__all__ = [
    "Args",
    "parse_args",
    "RolloutStorage",
    "MetricTracker",
    "TsilTrajectoryMemory",
    "RolloutCollector",
    "PolicyUpdater",
    "TsilReplayLoss",
    "TrainingLogger",
    "WandbLogger",
    "OnPolicyTrainer",
    "run_training",
]


def __getattr__(name):
    """Lazy import to avoid importing isaacgym at package-load time."""
    if name in ("Args", "parse_args"):
        from core.training.args import Args, parse_args
        return Args if name == "Args" else parse_args
    if name in ("RolloutStorage", "MetricTracker", "TsilTrajectoryMemory"):
        from core.training.algo.tsil.memory import TsilTrajectoryMemory
        from core.training.metrics import MetricTracker
        from core.training.storage import RolloutStorage
        return {"RolloutStorage": RolloutStorage, "MetricTracker": MetricTracker,
                "TsilTrajectoryMemory": TsilTrajectoryMemory}[name]
    if name == "RolloutCollector":
        from core.training.rollout import RolloutCollector
        return RolloutCollector
    if name in ("PolicyUpdater", "TsilReplayLoss"):
        from core.training.algo import PolicyUpdater, TsilReplayLoss
        return PolicyUpdater if name == "PolicyUpdater" else TsilReplayLoss
    if name in ("TrainingLogger", "WandbLogger"):
        from core.training.logger import TrainingLogger, WandbLogger
        return TrainingLogger if name == "TrainingLogger" else WandbLogger
    if name in ("OnPolicyTrainer", "run_training"):
        from core.training.trainer import OnPolicyTrainer, run_training
        if name == "OnPolicyTrainer":
            return OnPolicyTrainer
        return run_training
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
