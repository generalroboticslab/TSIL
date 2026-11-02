"""Independent norm clipping for agents with disjoint actor and critic networks.

This changes clipping only: losses, optimizer ownership, and Adam state stay
unchanged. Shared or unclassified trainable parameters must not be silently
clipped twice or assigned to an arbitrary group.
"""

from torch.nn.utils import clip_grad_norm_


def actor_critic_parameter_groups(agent):
    """Partition the main optimizer's parameters and reject unsupported networks."""
    actor_ids = {id(parameter) for _, parameter in agent.named_policy_parameters()}
    actor, critic = [], []
    seen = set()
    for name, parameter in agent.named_ppo_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise ValueError(f"Aliased optimizer parameter: {name}")
        seen.add(id(parameter))
        is_critic = name.startswith(
            ("critic", "crt_lstm", "temporal_critic", "_critic_heads")
        )
        if id(parameter) in actor_ids:
            if is_critic:
                raise ValueError(f"Shared actor/critic parameter: {name}")
            actor.append(parameter)
        elif is_critic:
            critic.append(parameter)
        else:
            raise ValueError(f"Unclassified optimizer parameter: {name}")
    if not actor or not critic or not actor_ids.issubset(seen):
        raise ValueError(
            "Separate clipping requires complete, disjoint actor/critic groups"
        )
    return actor, critic


def clip_actor_critic_gradients(agent, max_norm):
    """Clip each group to max_norm, preserving relative gradients within it."""
    actor, critic = actor_critic_parameter_groups(agent)
    return clip_grad_norm_(actor, max_norm), clip_grad_norm_(critic, max_norm)
