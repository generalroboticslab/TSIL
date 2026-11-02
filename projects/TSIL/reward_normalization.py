"""Stable project imports for the shared reward normalizer."""
from core.agents.reward_normalization import PerEnvironmentReturnRewardNormalizer


class StdOnlyReturnRewardNormalizer(PerEnvironmentReturnRewardNormalizer):
    """Compatibility class for MiniGrid std-only checkpoints and callers."""
    def __init__(self, *args, **kwargs):
        kwargs['scale_mode'] = 'std_only'
        super().__init__(*args, **kwargs)


__all__ = ['PerEnvironmentReturnRewardNormalizer', 'StdOnlyReturnRewardNormalizer']
