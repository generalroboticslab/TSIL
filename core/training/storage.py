"""Rollout storage buffers.

RolloutStorage is the explicit shared data bus between the rollout collector,
policy updater, and the trainer orchestrator.

"""

import torch


class RolloutStorage:
    """Container for rollout tensor buffers and current environment state."""

    def __init__(self, num_steps, num_envs, obs_shape, state_shape, act_shape,
                 num_cost, device, dtype, use_lstm=False, rollout_agent=None):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.dtype = dtype

        # ------------------------------------------------------------------
        # Rollout data buffers (written by RolloutCollector, read by updaters)
        # ------------------------------------------------------------------
        self.obs = torch.zeros((num_steps, num_envs) + obs_shape, dtype=dtype, device=device)
        self.states = torch.zeros((num_steps, num_envs) + state_shape, dtype=dtype, device=device)
        self.actions = torch.zeros((num_steps, num_envs) + act_shape, dtype=dtype, device=device)
        self.logprobs = torch.zeros((num_steps, num_envs), dtype=dtype, device=device)
        self.rewards = torch.zeros((num_steps, num_envs), dtype=dtype, device=device)
        self.dones = torch.zeros((num_steps, num_envs), dtype=dtype, device=device)
        self.timeouts = torch.zeros((num_steps, num_envs), dtype=dtype, device=device)
        self.values = torch.zeros((num_steps, num_envs), dtype=dtype, device=device)

        # Cost buffers
        self.costs = torch.zeros((num_steps, num_envs, num_cost), dtype=dtype, device=device)
        self.values_c = torch.zeros((num_steps, num_envs, num_cost), dtype=dtype, device=device)

        # Time-awareness buffer
        self.values_t = torch.zeros((num_steps, num_envs), dtype=dtype, device=device)

        # Episode tracking
        self.step_episode_ids = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)

        # ------------------------------------------------------------------
        # Current env state (updated every env step by the rollout collector)
        # ------------------------------------------------------------------
        self.next_obs = None
        self.next_state = None
        self.next_done = torch.zeros(num_envs, device=device)
        self.next_timeout = torch.zeros(num_envs, device=device)
        self.current_episode_ids = torch.arange(num_envs, dtype=torch.long, device=device)
        self.next_episode_uid = int(num_envs)

        # ------------------------------------------------------------------
        # LSTM state (optional)
        # ------------------------------------------------------------------
        self.next_lstm_state = None
        self.lstm_state_storage = None
        if use_lstm and rollout_agent is not None:
            num_layers_crt = rollout_agent.crt_lstm.num_layers
            num_layers_act = rollout_agent.act_lstm.num_layers
            hidden_size_crt = rollout_agent.crt_lstm.hidden_size
            hidden_size_act = rollout_agent.act_lstm.hidden_size
            self.next_lstm_state = (
                torch.zeros(num_layers_crt, num_envs, hidden_size_crt, dtype=dtype, device=device),
                torch.zeros(num_layers_crt, num_envs, hidden_size_crt, dtype=dtype, device=device),
                torch.zeros(num_layers_act, num_envs, hidden_size_act, dtype=dtype, device=device),
                torch.zeros(num_layers_act, num_envs, hidden_size_act, dtype=dtype, device=device),
            )
            self.lstm_state_storage = tuple(
                torch.zeros((num_steps,) + lstm_state.shape, dtype=dtype, device=device)
                for lstm_state in self.next_lstm_state
            )

        # Rollout-level bookkeeping set by the collector each rollout
        self.rollout_completed_episode_success = {}
        self.rollout_completed_episode_time = {}
        self.rollout_completed_episode_dense_return = {}
        self.rollout_completed_episode_task_id = {}
        self.rollout_completed_episode_id_by_env = {}

    def init_from_env_reset(self, envs):
        """Initialize current state from environment reset."""
        envs.reset_all()
        next_obs_dict = envs.reset()
        self.next_obs = torch.Tensor(next_obs_dict["obs"]).to(self.device)
        self.next_state = torch.Tensor(next_obs_dict["states"]).to(self.device)
__all__ = ["RolloutStorage"]
