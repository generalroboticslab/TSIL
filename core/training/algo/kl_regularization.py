"""Optional smooth trust-region penalty relative to rollout log-probabilities."""
import torch


class SuccessActivatedPenalty:
    """Latch soft regularization on after the observed success threshold is met."""

    def __init__(self, threshold=0.0):
        self.threshold = float(threshold)
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("KL activation success must be between zero and one")
        self.active = self.threshold == 0.0

    def coefficient(self, configured_coefficient, current_success):
        coefficient = float(configured_coefficient)
        if coefficient <= 0.0:
            return 0.0
        if not self.active:
            if current_success is None:
                raise ValueError("Success-activated KL requires the current training success")
            if float(current_success) >= self.threshold:
                self.active = True
        return coefficient if self.active else 0.0

    def state_dict(self):
        return {"active": self.active}

    def load_state_dict(self, state):
        if not isinstance(state["active"], bool):
            raise ValueError("KL activation state must be a boolean")
        self.active = state["active"] or self.threshold == 0.0


def rollout_kl_penalty(logratio, valid, coefficient):
    """Sample estimate of KL(old || current), with its gradient retained.

    Rollout actions are sampled from the old policy. The expm1 control variate
    has zero expectation under that policy and gives a nonnegative per-sample
    estimator. This is a penalty, not a strict bound or a rollback rule.
    """
    if coefficient <= 0:
        return logratio.new_zeros(())
    values = logratio[valid]
    return float(coefficient) * (torch.expm1(values) - values).mean()
