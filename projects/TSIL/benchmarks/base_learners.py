"""MiniGrid counterparts of the documented Meta-World R6Label and S settings.

Only the learner package transfers: MiniGrid keeps its CNN, categorical actions,
missions, native horizons and IH timeout bootstrap. There are no robot-velocity
fields to retain/remove. Correct full task-label preprocessing already exists.
"""

BASE_LEARNERS = {
    'R6Label': dict(ppo_boundary_mode='legacy', reward_normalizer_mode='legacy_task',
                    reward_normalizer_scale_mode='std_magnitude_floor', tsil_drop_constant_inputs=False),
    'R2-std-only': dict(ppo_boundary_mode='exclude_value', reward_normalizer_mode='per_environment',
                       reward_normalizer_scale_mode='std_only', tsil_drop_constant_inputs=True),
}
