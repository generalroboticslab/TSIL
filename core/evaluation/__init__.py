"""Shared evaluation package."""


def __getattr__(name):
    if name == "EvalMetrics":
        from core.evaluation.metrics import EvalMetrics
        return EvalMetrics
    if name == "TrajectoryReplayer":
        from core.evaluation.replay import TrajectoryReplayer
        return TrajectoryReplayer
    if name in ("BaseEvaluator", "run_evaluation"):
        from core.evaluation import evaluator
        return getattr(evaluator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EvalMetrics",
    "TrajectoryReplayer",
    "BaseEvaluator",
    "run_evaluation",
]
