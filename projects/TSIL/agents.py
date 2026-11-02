"""Single-task TSIL policies with constant inputs removed from the networks."""

import torch

from core.agents.agent import Agent


def validate_constant_input_config(args):
    if (getattr(args, "tsil_drop_critic_velocity_inputs", False)
            and not getattr(args, "tsil_drop_constant_inputs", False)):
        raise ValueError("tsil_drop_critic_velocity_inputs requires tsil_drop_constant_inputs")
    if getattr(args, "project_name", "TSIL") != "TSIL":
        raise ValueError("tsil_drop_constant_inputs is specific to the TSIL project")
    if len(set(args.tasks)) != 1:
        raise ValueError("Removing task inputs requires a single-task policy")
    if getattr(args, "benchmark_backend", "metaworld") != "metaworld":
        raise ValueError("Constant-input removal currently supports MetaWorld observations")
    if any(getattr(args, name, False) for name in ["beta", "use_lstm", "use_cost"]):
        raise ValueError("Constant-input removal currently supports feedforward Gaussian policies without costs")
    if getattr(args, "ratio_range", None) is not None or getattr(args, "warmup_rand", False):
        raise ValueError("Removing the speed input requires a fixed speed ratio")
    if getattr(args, "goal_speed", None) not in {None, 1.0} or getattr(args, "goal_time", None) is not None:
        raise ValueError("Constant-input removal requires the training speed ratio of one")


class SingleTaskAgent(Agent):
    """Keep raw observations/metadata intact; project normalized network inputs.

    Raw fields remain [speed_ratio, elapsed, target, physical..., task_one_hot].
    The actor/critic receive [elapsed/max_time, target/max_time, normalized physical...].
    The saved config selects this same class during evaluation and checkpoint load.
    """

    def __init__(self, envs, args, num_actions=None, device=None):
        validate_constant_input_config(args)
        if len(envs.unique_task_ids) != 1:
            raise ValueError("Removing task inputs requires a single-task environment")
        if envs.add_timeaware_obs([]) != ["time_ratio", "time_cur", "ddl"]:
            raise ValueError("Constant-input removal requires the standard three temporal fields")
        cfg = getattr(envs, "cfg", {})
        if cfg.get("warmup_rand", False) or cfg.get("budget_portion") is not None:
            raise ValueError("Constant-input removal requires a fixed, unstaged speed ratio")
        self.removed_critic_fields = []
        self.critic_input_indices = list(range(1, envs.num_states - 50))
        if getattr(args, "tsil_drop_critic_velocity_inputs", False):
            slices = getattr(envs, "state_slice", {})
            masked = set(getattr(envs, "maskout_names", []))
            removed = set()
            for name in ("sce_linvel", "lim_linvel"):
                field = slices.get(name)
                if (not isinstance(field, slice) or field.step not in (None, 1)
                        or field.start is None or field.stop is None
                        or field.stop != field.start + 1
                        or not 3 <= field.start < envs.num_states - 50):
                    raise ValueError(f"Missing or invalid critic field layout: {name}")
                if name not in masked:
                    raise ValueError(f"Cannot remove unmasked critic field: {name}")
                removed.add(field.start)
                self.removed_critic_fields.append(name)
            if len(removed) != 2:
                raise ValueError("Critic velocity fields must have distinct coordinates")
            self.critic_input_indices = [i for i in self.critic_input_indices if i not in removed]
        super().__init__(envs, args, num_actions=num_actions, device=device)
        # Derived from environment metadata; not a checkpoint parameter. Old
        # checkpoints keep their original keys and input dimensions.
        self.register_buffer("_critic_input_indices", torch.tensor(
            self.critic_input_indices, dtype=torch.long, device=self.device), persistent=False)

    def network_input_dims(self):
        if self.task_embedding_dim != 50:
            raise ValueError("Constant-input removal requires the MetaWorld 50-entry task embedding")
        return self.obs_dim - 51, len(self.critic_input_indices)

    def preprocess_obs(self, obs, update=False):
        normalized = super().preprocess_obs(obs, update=update)
        return normalized[..., 1:self.obs_dim - self.task_embedding_dim]

    def preprocess_state(self, state, update=False):
        normalized = super().preprocess_state(state, update=update)
        if self.removed_critic_fields:
            projected = normalized.index_select(-1, self._critic_input_indices)
        else:
            projected = normalized[..., 1:self.state_dim - self.task_embedding_dim]
        return projected

    def _task_ids_from_state(self, state):
        if not self.pertask_norm:
            return None
        return self.unique_task_ids[0].expand(state.shape[:-1])
