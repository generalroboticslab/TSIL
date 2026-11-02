"""Categorical policy and image encoder shared by TSIL MiniGrid runs."""
import torch
from torch import nn
from torch.distributions import Categorical

class CategoricalPolicyMixin:
    """Replace the continuous policy distribution with a masked categorical."""

    is_categorical_policy = True

    def _actor_output_dim(self):
        return int(self.envs.num_actions)

    def _init_actor_params(self):
        return None

    def init_policy_net(self, envs, args):
        super().init_policy_net(envs, args)
        # Agent creates this Normal-policy parameter. It is not part of a
        # categorical policy and must not appear in PolicyGradEx parameters.
        if hasattr(self, "actor_logstd"):
            del self.actor_logstd

    def build_distribution(self, raw_obs, update=False, validate_args=False):
        obs = self.preprocess_obs(raw_obs, update=update)
        logits = self._actor_forward(obs)
        _, task_embedding = self._split_obs(obs)
        if task_embedding is None:
            raise ValueError("MiniGrid categorical policies require a task embedding.")
        valid_actions = self.envs.action_mask_from_task_embedding(task_embedding)
        logits = logits.masked_fill(~valid_actions, torch.finfo(logits.dtype).min)
        distribution = Categorical(
            logits=logits,
            validate_args=validate_args,
        )
        return distribution, distribution.probs

    @staticmethod
    def _action_indices(action):
        action = torch.as_tensor(action)
        if action.ndim > 1 and action.shape[-1] == 1:
            action = action.squeeze(-1)
        return action.long()

    def get_logprob(self, raw_obs, action, update=False):
        distribution, _ = self.build_distribution(
            raw_obs, update=update, validate_args=False
        )
        return distribution.log_prob(self._action_indices(action))

    def get_action_and_value(
        self,
        raw_obs,
        raw_state=None,
        action=None,
        action_only=False,
        denorm=True,
        update=False,
    ):
        self.probs, action_probabilities = self.build_distribution(
            raw_obs, update=update
        )
        if action is None:
            action_indices = (
                action_probabilities.argmax(dim=-1)
                if self.deterministic
                else self.probs.sample()
            )
        else:
            action_indices = self._action_indices(action)
        action_column = action_indices.unsqueeze(-1)
        if action_only:
            return action_column, self.probs

        logprob = self.probs.log_prob(action_indices)
        entropy = self.probs.entropy()
        self.prob_entropy = entropy
        return (
            action_column,
            action_probabilities,
            logprob,
            entropy,
            *self.get_value(raw_state, denorm=denorm, update=update),
        )

def _cnn(last_activation: bool) -> nn.Sequential:
    layers = [
        nn.Conv2d(3, 16, (2, 2)),
        nn.ReLU(),
        nn.Conv2d(16, 32, (2, 2)),
        nn.ReLU(),
        nn.Conv2d(32, 64, (2, 2)),
    ]
    if last_activation:
        layers.append(nn.Tanh())
    layers.append(nn.Flatten())
    return nn.Sequential(*layers)
