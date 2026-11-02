"""Pure helpers for temporal deadline termination."""

import torch


def temporal_deadline_limit(real_time2end_init, grace_frac=0.0, min_seconds=0.0):
    """Return the hard deadline limit in seconds for each environment."""
    grace = max(float(grace_frac), 0.0)
    floor = max(float(min_seconds), 0.0)
    limit = real_time2end_init * (1.0 + grace)
    if floor > 0.0:
        limit = torch.clamp(limit, min=floor)
    return limit


def temporal_deadline_expired(real_timecur, real_time2end_init, grace_frac=0.0, min_seconds=0.0):
    """Return environments whose elapsed time has reached the deadline."""
    return real_timecur >= temporal_deadline_limit(
        real_time2end_init,
        grace_frac=grace_frac,
        min_seconds=min_seconds,
    )


def apply_temporal_deadline_reset(
    reset_buf,
    success_buf,
    real_timecur,
    real_time2end_init,
    grace_frac=0.0,
    min_seconds=0.0,
):
    """OR non-success deadline expirations into reset_buf."""
    expired = temporal_deadline_expired(
        real_timecur,
        real_time2end_init,
        grace_frac=grace_frac,
        min_seconds=min_seconds,
    ).to(device=reset_buf.device)
    deadline_done = expired & ~success_buf.to(device=reset_buf.device, dtype=torch.bool)
    updated_reset = torch.where(deadline_done, torch.ones_like(reset_buf), reset_buf)
    return updated_reset, deadline_done


def temporal_deadline_timeout_mask(max_horizon_timeout, deadline_done, terminal=True):
    """Apply deadline terminal-vs-timeout semantics to a max-horizon timeout mask."""
    deadline_done = deadline_done.to(device=max_horizon_timeout.device, dtype=torch.bool)
    if bool(terminal):
        return max_horizon_timeout.to(dtype=torch.bool) & ~deadline_done
    return max_horizon_timeout.to(dtype=torch.bool) | deadline_done
