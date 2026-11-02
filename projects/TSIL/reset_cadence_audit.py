"""Opt-in diagnostic for Meta-World's scene-wide reset simulation step.

The Meta-World reset path advances the complete Isaac Gym scene once whenever
at least one lane resets. With thousands of asynchronous training lanes this
happens on almost every policy step. A synchronized, one-episode evaluation
does not have that cadence before its first completion. This helper makes the
missing refresh step explicit so the two regimes can be compared. It is never
installed by normal training or evaluation.
"""


def install_reset_cadence_diagnostic(env):
    """Ensure one reset-style scene refresh occurs after every policy step.

    If a lane is already pending reset, ``env.post_physics_step`` supplies the
    refresh through ``reset_idx``. Otherwise this wrapper supplies exactly one
    equivalent scene-wide simulation step before normal post-step processing.
    The episode progress counter is deliberately not advanced, matching the
    existing reset path.
    """
    original_post_physics_step = env.post_physics_step
    counts = {"forced_refresh_steps": 0, "natural_reset_steps": 0}

    def post_physics_step_with_training_reset_cadence():
        has_pending_reset = bool(env.reset_buf.any().item())
        if has_pending_reset:
            counts["natural_reset_steps"] += 1
        else:
            env.gym.simulate(env.sim)
            env.gym.fetch_results(env.sim, True)
            counts["forced_refresh_steps"] += 1
        return original_post_physics_step()

    env.post_physics_step = post_physics_step_with_training_reset_cadence
    return counts
