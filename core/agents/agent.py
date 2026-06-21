import os
import torch
import torch.nn as nn
from torch.distributions import Normal, Beta

from copy import deepcopy
from core.agents.utils import (
    MLP,
    PerTaskRunningMeanStd,
    get_activation,
    save_checkpoint,
)
from core.common.tensor import get_args_attr


class Agent(nn.Module):
    def __init__(self, envs, args, num_actions=None, device=None):
        super().__init__()
        self.args = args
        self.envs = envs
        self.envs.agent = self
        self.device = torch.device(device if device is not None else getattr(envs, "device", "cpu"))
        self.tensor_dtype = torch.float32
        self.activation = get_activation(get_args_attr(args, 'activation', 'tanh'))
        self.use_layernorm = get_args_attr(args, 'use_layernorm', True)
        self.deterministic = args.deterministic
        self.hidden_size = args.hidden_size
        self.obs_dim = envs.num_observations
        self.state_dim = envs.num_states

        self.use_cost = get_args_attr(args, "use_cost", False)
        self.use_timeawareness = get_args_attr(args, 'use_timeawareness', False)
        self.num_costs = get_args_attr(args, "num_cost", 2)

        self.critic_dim = 1 if not self.use_timeawareness else 2 # value and time2end value
        self.action_logits_num = envs.num_actions * 2 if num_actions is None else num_actions * 2
        
        # Layer Number
        self.num_hidden_layer = len(self.hidden_size) if type(self.hidden_size) in [list, tuple] else 3

        # Per-task normalisation config
        self.pertask_norm = get_args_attr(args, 'pertask_norm', False)
        unique_task_ids = getattr(envs, 'unique_task_ids', None)
        self.unique_task_ids = (
            torch.tensor([0], dtype=torch.long, device=self.device)
            if unique_task_ids is None
            else torch.as_tensor(unique_task_ids, dtype=torch.long, device=self.device)
        )
        self.task_embedding_dim = get_args_attr(args, 'task_embedding_dim', 0)
        if self.task_embedding_dim == 0:
            # Auto-detect: number of unique tasks = size of one-hot embedding
            self.task_embedding_dim = len(self.unique_task_ids)
        self.public_time_obs_dim = len(envs.add_timeaware_obs([])) if hasattr(envs, "add_timeaware_obs") else 0
        self.time_feature_scale = max(float(getattr(envs, "max_eps_time", 1.0)), 1e-8)

        self.init_preprocess_net(envs, args)
        self.init_policy_net(envs, args)
        self.init_costs_net()
            
        self.init_critic_params = deepcopy(self.critic.state_dict())
        self.to(device=self.device, dtype=self.tensor_dtype)


    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        try:
            self.device = next(self.parameters()).device
        except StopIteration:
            pass
        return module


    # ------------------------------------------------------------------ #
    #  Network Initialization
    # ------------------------------------------------------------------ #

    def init_preprocess_net(self, envs, args):
        # Body dims: exclude task embedding and public time features from RMS normalisation.
        norm_obs_dim   = self.obs_dim   - self.task_embedding_dim - self.public_time_obs_dim
        norm_state_dim = self.state_dim - self.task_embedding_dim - self.public_time_obs_dim
        task_ids = self.unique_task_ids if self.pertask_norm else None

        # Input Normalization Layer (clip=False: match MTBench, no hard-clamp on normalized obs)
        if get_args_attr(args, 'norm_obs', False):
            self.obs_normalizer   = PerTaskRunningMeanStd(norm_obs_dim, task_ids, clip=False, dtype=self.tensor_dtype, device=self.device)
            self.state_normalizer = PerTaskRunningMeanStd(norm_state_dim, task_ids, clip=False, dtype=self.tensor_dtype, device=self.device)
        
        # Value Normalization Layer
        if get_args_attr(args, 'norm_value', False):
            self.value_normalizer = PerTaskRunningMeanStd(1, task_ids, dtype=self.tensor_dtype, device=self.device)

        if self.use_timeawareness:
            self.value_normalizer_t = PerTaskRunningMeanStd(1, task_ids, dtype=self.tensor_dtype, device=self.device)

    def init_policy_net(self, envs, args):
        # Use MLP for the critic and actor
        activation = get_args_attr(args, 'activation', 'tanh')
        use_layernorm = get_args_attr(args, 'use_layernorm', True)
        self.critic = MLP(
            input_size=self.state_dim,
            hidden_size=self.hidden_size,
            output_size=self.critic_dim,
            num_hidden_layers=self.num_hidden_layer,
            activation=activation,
            use_layernorm=use_layernorm,
            init_std=1.0,
        ).to(self.device)
        self.actor = MLP( # The output is the mean only
            input_size=self.obs_dim,
            hidden_size=self.hidden_size,
            output_size=self.action_logits_num // 2,
            num_hidden_layers=self.num_hidden_layer,
            activation=activation,
            use_layernorm=use_layernorm,
            init_std=0.01,
        ).to(self.device)
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_logits_num // 2))


    def _split_obs(self, x):
        """Split observation tensor into normalized body and task embedding."""
        if self.task_embedding_dim <= 0:
            return x, None
        split_idx = self.obs_dim - self.task_embedding_dim
        return x[..., :split_idx], x[..., split_idx:self.obs_dim]


    def _split_state(self, x):
        """Split state tensor into normalized body and task embedding."""
        if self.task_embedding_dim <= 0:
            return x, None
        split_idx = self.state_dim - self.task_embedding_dim
        return x[..., :split_idx], x[..., split_idx:self.state_dim]


    def init_costs_net(self):
        if not self.use_cost:
            return
        
        activation = get_args_attr(self.args, 'activation', 'tanh')
        use_layernorm = get_args_attr(self.args, 'use_layernorm', True)
        self.critic_t = MLP(
            input_size=self.state_dim,
            hidden_size=self.hidden_size,
            output_size=1,
            num_hidden_layers=self.num_hidden_layer,
            activation=activation,
            use_layernorm=use_layernorm,
            init_std=1.0,
            output_softplus=True
        ).to(self.device)

        self.critic_inst = MLP(
            input_size=self.state_dim,
            hidden_size=self.hidden_size,
            output_size=1,
            num_hidden_layers=self.num_hidden_layer,
            activation=activation,
            use_layernorm=use_layernorm,
            init_std=1.0,
            output_softplus=True
        ).to(self.device)


    # ------------------------------------------------------------------ #
    #  Value Functions
    # ------------------------------------------------------------------ #

    def get_value(self, raw_state, denorm=True, update=False):
        """
        Get the value estimate for a given state.
        
        Args:
            raw_state: The raw state input
            denorm: If True and norm_value is enabled, return denormalized value.
                   Use denorm=True during rollout/GAE, denorm=False during loss computation.
        """
        x = self.preprocess_state(raw_state, update=update)
        value = self.critic(x)

        if self.use_timeawareness: value, value_t = torch.chunk(value, 2, dim=-1)
        else: value_t = torch.zeros_like(value)

        # We must ensure all value prediction is (N, ) shape
        value = value.flatten()
        value_t = value_t.flatten()
            
        # Denormalize value for rollout/GAE computation.
        task_ids = self._task_ids_from_state(x)
        if denorm and get_args_attr(self.args, 'norm_value', False):
            value = self.normalize_value(value, denorm=denorm, task_ids=task_ids)
            if self.use_timeawareness:
                value_t = self.normalize_value_t(value_t, denorm=denorm, task_ids=task_ids)
        
        return value, self.get_cost_value(x), value_t
    

    def get_cost_value(self, x):
        ### ! We use x after the preprocess ###
        value_c = torch.cat([self.critic_t(x), self.critic_inst(x)], dim=-1) if self.use_cost else torch.zeros(x.shape[0], 2, device=self.device)
        if self.num_costs == 1:
            value_c = value_c.flatten()
        return value_c


    # ------------------------------------------------------------------ #
    #  Policy Distribution Helpers
    # ------------------------------------------------------------------ #

    def _actor_forward(self, obs):
        return self.actor(obs)


    def _build_normal_distribution_from_mean(self, action_mean, validate_args=False, action_logstd=None):
        if action_logstd is None:
            action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        return Normal(action_mean, action_std, validate_args=validate_args)


    def _build_beta_distribution_from_logits(self, action_logits, validate_args=False):
        action_logalpha, action_logbeta = torch.chunk(action_logits, 2, dim=-1)
        action_alpha = torch.exp(action_logalpha)
        action_beta = torch.exp(action_logbeta)
        return Beta(action_alpha, action_beta, validate_args=validate_args)


    def _build_distribution_from_actor_output(self, actor_output, validate_args=False):
        probs = self._build_normal_distribution_from_mean(actor_output, validate_args=validate_args)
        return probs, actor_output


    def build_distribution(self, raw_obs, update=False, validate_args=False):
        obs = self.preprocess_obs(raw_obs, update=update)
        actor_output = self._actor_forward(obs)
        return self._build_distribution_from_actor_output(actor_output, validate_args=validate_args)


    def get_logprob(self, raw_obs, action, update=False):
        """Actor-only log-probability path for analysis tools such as PolicyGradEx."""
        probs, _ = self.build_distribution(raw_obs, update=update, validate_args=False)
        return probs.log_prob(action).sum(1)


    def get_action_and_value(self, raw_obs, raw_state=None, action=None, action_only=False, denorm=True, update=False):
        """
        Get action, log probability, entropy, and value.
        
        Args:
            denorm: If True, return denormalized value (for rollout/GAE).
                   If False, return normalized value (for loss computation).
        """
        self.probs, action_mean = self.build_distribution(raw_obs, update=update)
        probs = self.probs
        if action is None:
            action = probs.mean if self.deterministic else probs.sample()
        if action_only:
            return action, probs
        
        logprob = probs.log_prob(action).sum(1)
        self.prob_entropy = probs.entropy() # Record the current probs for logging
        entropy = self.prob_entropy.sum(1)
        # if torch.isnan(logprob).any() or torch.isinf(logprob).any():
        #     print("logprob has inf or nan")
        #     import ipdb; ipdb.set_trace()

        return action, action_mean, logprob, entropy, *self.get_value(raw_state, denorm=denorm, update=update)
    

    # ------------------------------------------------------------------ #
    #  Preprocessing / Normalization
    # ------------------------------------------------------------------ #

    def preprocess_obs(self, obs, update=False):
        obs = self.normalize_obs(obs, update=update)
        return obs
    

    def preprocess_state(self, state, update=False):
        state = self.normalize_state(state, update=update)
        return state
    
    
    def normalize_obs(self, obs, denorm=False, update=False):
        obs_body, task_emb = self._split_obs(obs)
        time_obs, obs_body = self._split_public_time_obs(obs_body)
        time_obs = self._scale_public_time_obs(time_obs, denorm=denorm)
        if not get_args_attr(self.args, 'norm_obs', False):
            obs_body = torch.cat([time_obs, obs_body], dim=-1) if time_obs is not None else obs_body
            return torch.cat([obs_body, task_emb], dim=-1) if task_emb is not None else obs_body
        # Pass actual task_ids for per-task stats; None falls back to task-agnostic (fast path)
        task_ids = self._task_ids_from_emb(task_emb) if (self.pertask_norm and task_emb is not None) else None
        obs_body = self.obs_normalizer(obs_body, task_ids=task_ids, denorm=denorm, update=update)
        obs_body = torch.cat([time_obs, obs_body], dim=-1) if time_obs is not None else obs_body
        return torch.cat([obs_body, task_emb], dim=-1) if task_emb is not None else obs_body
    

    def normalize_state(self, state, denorm=False, update=False):
        state_body, task_emb = self._split_state(state)
        time_obs, state_body = self._split_public_time_obs(state_body)
        time_obs = self._scale_public_time_obs(time_obs, denorm=denorm)
        if not get_args_attr(self.args, 'norm_obs', False):
            state_body = torch.cat([time_obs, state_body], dim=-1) if time_obs is not None else state_body
            return torch.cat([state_body, task_emb], dim=-1) if task_emb is not None else state_body
        task_ids = self._task_ids_from_emb(task_emb) if (self.pertask_norm and task_emb is not None) else None
        state_body = self.state_normalizer(state_body, task_ids=task_ids, denorm=denorm, update=update)
        state_body = torch.cat([time_obs, state_body], dim=-1) if time_obs is not None else state_body
        return torch.cat([state_body, task_emb], dim=-1) if task_emb is not None else state_body
    

    def normalize_value(self, value, denorm=False, update=False, task_ids=None):
        if not get_args_attr(self.args, 'norm_value', False):
            return value
        return self.value_normalizer(value, task_ids=task_ids, denorm=denorm, update=update)
    

    def normalize_value_t(self, value_t, denorm=False, update=False, task_ids=None):
        if not get_args_attr(self.args, 'use_timeawareness', False):
            return value_t
        return self.value_normalizer_t(value_t, task_ids=task_ids, denorm=denorm, update=update)
    

    def _task_ids_from_emb(self, task_emb):
        """Extract actual task IDs from a one-hot task embedding tensor.
        Note: in the future, embeddings may be learned rather than one-hot, so this method provides a single point of change for that logic.

        Args:
            task_emb: ``(B, task_embedding_dim)`` one-hot float tensor, or None.
        Returns:
            1-D long tensor of task IDs, or None.
        """
        if task_emb is None or self.task_embedding_dim == 0:
            return None
        emb_idx = task_emb.argmax(dim=-1).long()
        return self.unique_task_ids.to(task_emb.device)[emb_idx]


    def _split_public_time_obs(self, obs_body):
        time_dim = min(self.public_time_obs_dim, obs_body.shape[-1])
        if time_dim <= 0:
            return None, obs_body
        return obs_body[..., :time_dim], obs_body[..., time_dim:]


    def _scale_public_time_obs(self, time_obs, denorm=False):
        if time_obs is None:
            return None
        scaled = time_obs.clone()
        if scaled.shape[-1] > 1:
            factor = self.time_feature_scale if denorm else (1.0 / self.time_feature_scale)
            scaled[..., 1:] = scaled[..., 1:] * factor
        return scaled


    def _task_ids_from_state(self, state):
        """Extract task IDs from a preprocessed state tensor when enabled."""
        if not self.pertask_norm or self.task_embedding_dim <= 0:
            return None
        return self._task_ids_from_emb(state[..., -self.task_embedding_dim:])

    
    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #

    def named_policy_parameters(self):
        """Yield trainable parameters that affect action log-probability.

        This gives analyzer-style tools a structural way to get policy params
        without probing a sample-dependent execution path.
        """
        policy_prefixes = ("actor", "actor_", "act_lstm", "_actor_heads")
        for name, param in self.named_parameters():
            if param.requires_grad and name.startswith(policy_prefixes):
                yield name, param


    def set_mode(self, mode='train'):
        if mode == 'train': 
            self.train()
        elif mode == 'eval': 
            self.eval()
    

    # ------------------------------------------------------------------ #
    #  Checkpointing
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, folder_path, ckpt_name="eps", suffix="", ckpt_path=None, reward_normalizer=None, verbose=False):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if ckpt_path is None:
            ckpt_path = "{}/{}_{}".format(folder_path, ckpt_name, suffix)
        if verbose:
            print('Saving models to {}'.format(ckpt_path))
        torch.save(self.state_dict(), ckpt_path)
        
        if reward_normalizer is not None:
            save_checkpoint(reward_normalizer, folder_path, ckpt_name="rew_norm_eps", suffix=suffix, verbose=verbose)


    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False, map_location='cuda:0', reset_critic=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location=map_location)
            self.load_state_dict(checkpoint, strict=False)
            
            if reset_critic:
                self.critic.load_state_dict(self.init_critic_params)
                if get_args_attr(self.args, 'norm_value', False):
                    self.value_normalizer.reset()
                    if self.use_timeawareness:
                        self.value_normalizer_t.reset()

            # try:
            #     self.load_state_dict(checkpoint, strict=False)
            # except RuntimeError as e:
            #     for name, param in checkpoint.items():
            #         if len(param.shape) == 2:
            #             ckpt_dim = param.shape[1]
            #             model_dim = self.state_dict()[name].shape[1]
            #             if ckpt_dim != model_dim:
            #                 self.state_dict()[name][:, :ckpt_dim].copy_(param)
            #                 print(f"############ Waring: The checkpoint is not strictly loaded. The ckpt input_dim is {ckpt_dim} while the current model is {model_dim} ###########")
            #             else:
            #                 self.state_dict()[name].copy_(param)
            #         elif len(param.shape) == 1:
            #             ckpt_dim = param.shape[0]
            #             model_dim = self.state_dict()[name].shape[0]
            #             if ckpt_dim != model_dim:
            #                 self.state_dict()[name][:ckpt_dim].copy_(param)
            #                 print(f"############ Waring: The checkpoint is not strictly loaded. The ckpt input_dim is {ckpt_dim} while the current model is {model_dim} ###########")
            #             else:
            #                 self.state_dict()[name].copy_(param)
            #         else:
            #             self.state_dict()[name].copy_(param)
            
            if getattr(self.args, 'freeze', False):
                # freeze the model parameters apart from the last layer
                last_layer_num = list(self.state_dict().keys())[-1].split('.')[-2] # Example: "actor.mlp.7.weight"
                for name, param in self.named_parameters():
                    if last_layer_num not in name:
                        param.requires_grad = False
                        print(name, "is frozen", "shape", param.shape)
        
        else:
            print("No checkpoint found at {}".format(ckpt_path))

        if evaluate: self.set_mode('eval')
        else: self.set_mode('train')



class BetaAgent(Agent):
    def __init__(self, envs, args, num_actions=None, device=None):
        super().__init__(envs, args, num_actions=num_actions, device=device)
        
    # ------------------------------------------------------------------ #
    #  Network Initialization
    # ------------------------------------------------------------------ #

    def init_policy_net(self, envs, args):
        # Use MLP for the critic and actor
        activation = get_args_attr(args, 'activation', 'tanh')
        use_layernorm = get_args_attr(args, 'use_layernorm', True)
        self.critic = MLP(
            input_size=self.state_dim,
            hidden_size=self.hidden_size,
            output_size=self.critic_dim,
            num_hidden_layers=self.num_hidden_layer,
            activation=activation,
            use_layernorm=use_layernorm,
            init_std=1.0,
        ).to(self.device)
        self.actor = MLP( # The output is the alpha and beta for the Beta distribution
            input_size=self.obs_dim,
            hidden_size=self.hidden_size,
            output_size=self.action_logits_num,
            num_hidden_layers=self.num_hidden_layer,
            activation=activation,
            use_layernorm=use_layernorm,
            init_std=0.01,
        ).to(self.device)
    

    # ------------------------------------------------------------------ #
    #  Policy Distribution Helpers
    # ------------------------------------------------------------------ #

    def _build_distribution_from_actor_output(self, actor_output, validate_args=False):
        probs = self._build_beta_distribution_from_logits(actor_output, validate_args=validate_args)
        return probs, probs.mean


    def get_logprob(self, raw_obs, action, update=False):
        probs, _ = self.build_distribution(raw_obs, update=update, validate_args=False)
        return probs.log_prob(action).sum(-1)


    def get_action_and_value(self, raw_obs, raw_state=None, action=None, action_only=False, denorm=True, update=False):
        """
        Get action, log probability, entropy, and value.
        
        Args:
            denorm: If True, return denormalized value (for rollout/GAE).
                   If False, return normalized value (for loss computation).
        """
        self.probs, action_mean = self.build_distribution(raw_obs, update=update)
        probs = self.probs
        if action is None:
            action = probs.mean if self.deterministic else probs.sample()
        if action_only: # Only return the action and probs for evaluation
            return action, probs

        logprob = probs.log_prob(action).sum(-1) # log_prb means prob density not mass! This could be larger than 1
        self.prob_entropy = probs.entropy() # Record the current probs for logging to avoid repeated computation
        entropy = self.prob_entropy.sum(-1)
        
        return action, action_mean, logprob, entropy, *self.get_value(raw_state, denorm=denorm, update=update)
    

    def logprob_saliency(self, raw_obs, raw_state=None):
        with torch.enable_grad():
            obs = self.preprocess_obs(raw_obs)
            obs.requires_grad_(True)
            actor_output = self._actor_forward(obs)
            probs, _ = self._build_distribution_from_actor_output(actor_output)
            action = probs.mean if self.deterministic else probs.rsample()

            logprob = probs.log_prob(action).sum(-1) # action logprob; we do not compute per action grad but treat action as a whole

            self.actor.zero_grad(set_to_none=True)
            if obs.grad is not None:
                obs.grad.zero_()
            grad_out = torch.ones_like(logprob)
            logprob.backward(grad_out)
            grad_obs = obs.grad.detach()
        
        grad_obs = grad_obs.abs()
        grad_obs = grad_obs / (grad_obs.max(dim=-1, keepdim=True)[0] + 1e-10) # Normalize to [0, 1]

        return action, grad_obs


class LSTMAgent(Agent):
    def __init__(self, envs, args, num_actions=None, device=None):
        if not isinstance(args.lstm_hidden_size, int):
            raise TypeError(f"lstm_hidden_size must be an integer, got {args.lstm_hidden_size}")
        super().__init__(envs, args, num_actions=num_actions, device=device)
        self.lstm_hidden_size = args.lstm_hidden_size
        
    
    # ------------------------------------------------------------------ #
    #  Network Initialization
    # ------------------------------------------------------------------ #

    def init_policy_net(self, envs, args):
        self.crt_lstm = nn.LSTM(
            input_size=self.state_dim,
            hidden_size=args.lstm_hidden_size,
            num_layers=1
        ).to(self.device)
        self.act_lstm = nn.LSTM(
            input_size=self.obs_dim,
            hidden_size=args.lstm_hidden_size,
            num_layers=1
        ).to(self.device)

        # Use MLP for the critic and actor
        activation = get_args_attr(args, 'activation', 'tanh')
        use_layernorm = get_args_attr(args, 'use_layernorm', True)
        self.critic = MLP(
            input_size=args.lstm_hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.critic_dim,
            num_hidden_layers=self.num_hidden_layer,
            activation=activation,
            use_layernorm=use_layernorm,
            init_std=1.0,
        ).to(self.device)
        self.actor = MLP( # The output is the mean only
            input_size=args.lstm_hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.action_logits_num,
            num_hidden_layers=self.num_hidden_layer,
            activation=activation,
            use_layernorm=use_layernorm,
            init_std=0.01,
        ).to(self.device)
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_logits_num // 2))


    # ------------------------------------------------------------------ #
    #  LSTM Helpers
    # ------------------------------------------------------------------ #

    def lstm_fw(self, lstm, x, lstm_state, done):
        """
        lstm_state: (hidden, cell)
        """
        obs_ft = x

        # LSTM forward
        batch_size = lstm_state[0].shape[1]
        # batch_first cannot process parallel computation (start from each sequence).
        # Sequence len is 1 when doing roll-out and will be 32 during training.
        obs_ft = obs_ft.reshape((-1, batch_size, lstm.input_size)) 
        done = done.reshape((-1, batch_size))
        new_ft = []
        for ft, d in zip(obs_ft, done):
            ft, lstm_state = lstm(
                ft.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_ft.append(ft)
        # torch.cat(new_ft) shape: [seq_len, batch_size, hidden_size]
        # After flatten(start_dim=0, end_dim=1): [seq_len * batch_size, hidden_size]
        new_ft = torch.flatten(torch.cat(new_ft), start_dim=0, end_dim=1)
        return new_ft, lstm_state

    
    # ------------------------------------------------------------------ #
    #  Value Functions
    # ------------------------------------------------------------------ #

    def get_value(self, raw_state, lstm_state, done, denorm=True, update=False):
        """
        Get the value estimate for a given state with LSTM.
        
        Args:
            raw_state: The raw state input
            lstm_state: The LSTM hidden state
            done: Done flags for resetting LSTM state
            denorm: If True and norm_value is enabled, return denormalized value.
                   Use denorm=True during rollout/GAE, denorm=False during loss computation.
        """
        x = self.preprocess_state(raw_state, update=update)
        crt_lstm_state = lstm_state[:2]
        crt_obs_ft, crt_lstm_state = self.lstm_fw(self.crt_lstm, x, crt_lstm_state, done)
        lstm_state = crt_lstm_state + lstm_state[2:]  # combine the lstm states to one tuple
        value = self.critic(crt_obs_ft)
        if self.use_timeawareness:
            value, value_t = torch.chunk(value, 2, dim=-1)
        else:
            value_t = torch.zeros_like(value)
        
        # We must ensure all value prediction is (N, ) shape
        value = value.flatten()
        value_t = value_t.flatten()
        
        # Denormalize value for rollout/GAE computation.
        if denorm and get_args_attr(self.args, 'norm_value', False):
            value = self.normalize_value(value, denorm=True)
            if self.use_timeawareness:
                value_t = self.normalize_value_t(value_t, denorm=True)
        
        return value, lstm_state, self.get_cost_value(x), value_t


    # ------------------------------------------------------------------ #
    #  Policy Distribution Helpers
    # ------------------------------------------------------------------ #

    def get_logprob(self, raw_obs, action, lstm_state, done, update=False):
        probs, _, _ = self.build_distribution(
            raw_obs, lstm_state, done, update=update, validate_args=False)
        return probs.log_prob(action).sum(1)


    def build_distribution(self, raw_obs, lstm_state, done, update=False, validate_args=False):
        obs = self.preprocess_obs(raw_obs, update=update)
        act_lstm_state = lstm_state[2:]
        act_obs_ft, act_lstm_state = self.lstm_fw(self.act_lstm, obs, act_lstm_state, done)
        actor_output = self.actor(act_obs_ft)
        probs = self._build_beta_distribution_from_logits(actor_output, validate_args=validate_args)
        return probs, probs.mean, act_lstm_state


    def get_action_and_value(self, raw_obs, lstm_state, done, raw_state=None, action=None, action_only=False, denorm=True, update=False):
        """
        Get action, log probability, entropy, and value with LSTM.
        
        Args:
            denorm: If True, return denormalized value (for rollout/GAE).
                   If False, return normalized value (for loss computation).
        """
        crt_lstm_state = lstm_state[:2]

        self.probs, action_mean, act_lstm_state = self.build_distribution(
            raw_obs, lstm_state, done, update=update)
        # If the model of action is complicated, creating the distribution will take more time.
        probs = self.probs
        lstm_state = crt_lstm_state + act_lstm_state # combine the lstm states to one tuple
        if action is None:
            action = probs.mean if self.deterministic else probs.sample()
        if action_only: # Only return the action and probs for evaluation
            return action, probs, lstm_state

        logprob = probs.log_prob(action).sum(1) # log_prb means prob density not mass! This could be larger than 1
        self.prob_entropy = probs.entropy() # Record the current probs for logging to avoid repeated computation
        entropy = self.prob_entropy.sum(1)
        
        return action, action_mean, logprob, entropy, *self.get_value(raw_state, lstm_state, done, denorm=denorm, update=update)

# ============================ Agent Factory ============================

def get_agent(envs, args, device=None):
    if device is None:
        device = getattr(envs, "device", None) or getattr(args, "sim_device", "cpu")
    agent = None

    # Beta
    if args.beta:
        if args.use_lstm:
            agent = LSTMAgent(envs, args, device=device).to(device)
        else:
            agent = BetaAgent(envs, args, device=device).to(device)
    # Normal
    else:
        agent = Agent(envs, args, device=device).to(device)
        
    return agent
    

if __name__ == "__main__":
    class TestModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor_std = nn.Parameter(torch.zeros(1, 2), requires_grad=False)
