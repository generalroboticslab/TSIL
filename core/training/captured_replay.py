"""Replay a saved aligned update without a live trajectory archive or resampling."""
import copy

import torch

from core.training.algo.tsil.loss import TsilReplayLoss
from core.training.algo.tsil.sampling import _TsilMemorySamplingMixin
from core.training.update_events import restore_rng


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(to_device(item, device) for item in value)
    return copy.deepcopy(value)


class CapturedAlignedReplay(TsilReplayLoss, _TsilMemorySamplingMixin):
    """Use the production losses and slice selection with a recorded dataset/order."""

    def __init__(self, args, device, dtype, snapshot):
        super().__init__(args, device, dtype)
        if snapshot.get("capture_format_version") != 2 or "aligned_replay" not in snapshot:
            raise ValueError("An aligned replay event capture is required")
        self.record = snapshot["aligned_replay"]
        self.dataset = to_device(self.record["dataset"], device)
        self.epoch_index = 0
        self.context = to_device(self.record["context"], device)
        self.context.update(sample_batch_fn=self._next_update_scope_replay_slice,
                            _update_event_recorder=self)

    def update_diagnostics(self, agent, replay_context, rollout_batch, updater, global_update_iter):
        # Archive diagnostics were recorded before the real optimizer loop.
        self._full_actor_diagnostics = self.record["full_actor_diagnostics"]

    def prepared(self, replay_module, replay_context):
        # Outcome-gradient diagnostics above this hook may consume randomness.
        restore_rng(self.record["rng_before_epochs"])

    def reset_epoch_sampler(self, replay_context, num_ppo_minibatches, agent=None):
        if self.epoch_index >= len(self.record["epochs"]):
            raise ValueError("Counterfactual extends beyond the recorded replay epochs")
        record = self.record["epochs"][self.epoch_index]
        assert record["epoch"] == self.epoch_index
        self.epoch_index += 1
        replay_context.update(to_device(record["context"], self.device))
        if record["has_batch"]:
            if self.dataset is None:
                raise ValueError("Captured replay epoch is missing its dataset")
            replay_context["_sil_replay_batch"] = self.dataset
        else:
            replay_context.pop("_sil_replay_batch", None)
            replay_context.pop("_sil_replay_order", None)
        # Dataset construction and randperm consumed RNG in the original run.
        restore_rng(record["rng"])

    def replay_epoch(self, epoch, replay_context):
        pass
