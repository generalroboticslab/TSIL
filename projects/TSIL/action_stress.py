"""Environment-side action replacement and opt-in mismatch controls."""
import torch


def corrupt_actions(
    actions,
    probability,
    num_actions=None,
    generator=None,
    mode="uniform_replacement",
    strength=1.0,
):
    if not 0 <= probability <= 1:
        raise ValueError("probability must lie in [0, 1]")
    if not 0 <= strength <= 1:
        raise ValueError("strength must lie in [0, 1]")
    if mode not in {"uniform_replacement", "additive_uniform", "categorical_mixture"}:
        raise ValueError("Unknown action corruption mode")
    if probability == 0:
        return actions, torch.zeros(actions.shape[0], dtype=torch.bool, device=actions.device)
    event = torch.rand(actions.shape[0], device=actions.device, generator=generator) < probability

    if mode == "additive_uniform":
        if num_actions is not None:
            raise ValueError("additive_uniform requires continuous actions")
        noise = torch.rand(
            actions.shape, device=actions.device, generator=generator, dtype=actions.dtype,
        ) * 2 - 1
        disturbed = torch.clamp(actions + strength * noise, -1, 1)
        expanded = event.reshape((-1,) + (1,) * (actions.ndim - 1))
        return torch.where(expanded, disturbed, actions), event

    if mode == "categorical_mixture":
        if num_actions is None:
            raise ValueError("categorical_mixture requires a discrete action count")
        choose_random = (
            torch.rand(actions.shape[0], device=actions.device, generator=generator) < strength
        )
        mask = event & choose_random
        random = torch.randint(
            num_actions, actions.shape, device=actions.device, generator=generator,
        ).to(actions.dtype)
        expanded = mask.reshape((-1,) + (1,) * (actions.ndim - 1))
        return torch.where(expanded, random, actions), mask

    if num_actions is not None:
        random = torch.randint(
            num_actions, actions.shape, device=actions.device, generator=generator,
        ).to(actions.dtype)
    else:
        random = torch.rand(
            actions.shape, device=actions.device, generator=generator, dtype=actions.dtype,
        ) * 2 - 1
    expanded = event.reshape((-1,) + (1,) * (actions.ndim - 1))
    return torch.where(expanded, random, actions), event


def storage_and_execution_actions(
    commanded_actions,
    probability,
    num_actions=None,
    generator=None,
    store_executed=False,
    mode="uniform_replacement",
    strength=1.0,
):
    """Return PPO-buffer actions, environment actions, and replacement mask.

    Normal actuator noise stores the policy command so its old log-probability
    remains correct. ``store_executed`` exists only to reproduce the explicitly
    labelled invalid-likelihood negative control.
    """
    executed, mask = corrupt_actions(
        commanded_actions, probability, num_actions, generator, mode, strength,
    )
    stored = executed if store_executed else commanded_actions
    return stored, executed, mask
