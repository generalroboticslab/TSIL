"""Bounded, count-balanced selection from currently positive trajectory steps."""
import os
import time
import torch


def least_selected_indices(positive, counts, limit, generator):
    """Select without replacement within each row; ties use a dedicated RNG."""
    limit = min(max(int(limit), 0), positive.shape[1])
    # Double precision preserves the integer count ordering even for long runs.
    scores = counts.double() + torch.rand(counts.shape, device=counts.device,
                                         dtype=torch.float64, generator=generator)
    scores = scores.masked_fill(~positive, float("inf"))
    indices = scores.topk(limit, dim=1, largest=False).indices
    valid = positive.gather(1, indices)
    rows = torch.arange(len(positive), device=positive.device)[:, None].expand_as(indices)
    return rows[valid], indices[valid], valid.sum(1)


def gap_proportional_indices(gaps, limit, generator):
    """Weighted sampling without replacement per trajectory, before loss caps.

    Exponential races produce a weighted random ordering: each successive draw
    is proportional to the remaining positive gaps. No importance correction;
    this intentionally changes replay emphasis, not the loss denominator.
    """
    limit = min(max(int(limit), 0), gaps.shape[1])
    positive = gaps > 0
    uniform = torch.rand(gaps.shape, device=gaps.device, dtype=torch.float64,
                         generator=generator).clamp_min(torch.finfo(torch.float64).tiny)
    # Log-space races avoid underflow for very small positive gaps.
    scores = torch.log(-torch.log(uniform)) - torch.log(gaps.double().clamp_min(torch.finfo(torch.float64).tiny))
    scores = scores.masked_fill(~positive, float("inf"))
    indices = scores.topk(limit, dim=1, largest=False).indices
    valid = positive.gather(1, indices)
    rows = torch.arange(len(gaps), device=gaps.device)[:, None].expand_as(indices)
    return rows[valid], indices[valid], valid.sum(1)


def global_gap_proportional_indices(gaps, limit, generator, alpha=1.0):
    """Draw positive-gap transitions globally without replacement.

    The input rows identify retained trajectories, but they impose no sampling
    quota.  Exponential races produce a weighted ordering over the flattened
    positive-transition pool.  No importance weights are returned because the
    prioritized replay objective is intentional.
    """
    flat = gaps.reshape(-1).double().clamp_min(0)
    positive = flat > 0
    count = min(max(int(limit), 0), int(positive.sum().item()))
    counts = torch.zeros(gaps.shape[0], dtype=torch.long, device=gaps.device)
    if count == 0:
        empty = torch.empty(0, dtype=torch.long, device=gaps.device)
        return empty, empty, counts
    priorities = flat.pow(float(alpha))
    uniform = torch.rand(
        flat.shape, device=gaps.device, dtype=torch.float64, generator=generator,
    ).clamp_min(torch.finfo(torch.float64).tiny)
    scores = torch.log(-torch.log(uniform)) - torch.log(
        priorities.clamp_min(torch.finfo(torch.float64).tiny)
    )
    scores.masked_fill_(~positive, float("inf"))
    selected = scores.topk(count, largest=False).indices
    rows = torch.div(selected, gaps.shape[1], rounding_mode="floor")
    steps = selected % gaps.shape[1]
    counts = torch.bincount(rows, minlength=gaps.shape[0])
    return rows, steps, counts


def replacement_indices(gaps, limit, generator, beta=0.0, alpha=1.0):
    """Per-trajectory draws with replacement; one dataset-wide IS scale.

    The target is uniform over each row's positive candidates. The maximum IS
    weight spans ALL such candidates, not just the sampled minibatch. Beta zero
    takes exactly the same random draws as beta one.
    """
    positive = gaps > 0
    counts = positive.sum(1).clamp(max=max(int(limit), 0))
    active_rows = torch.nonzero(counts > 0, as_tuple=False).view(-1)
    empty = torch.empty(0, dtype=torch.long, device=gaps.device)
    if not active_rows.numel():
        return empty, empty, counts, gaps.new_empty(0), gaps.new_empty(0)
    active = gaps[active_rows].double().clamp_min(0)
    priorities = (active / active.max(1, keepdim=True).values).pow(alpha)
    probs = priorities / priorities.sum(1, keepdim=True)
    indices = torch.multinomial(probs, int(counts.max()), replacement=True, generator=generator)
    valid = torch.arange(indices.shape[1], device=gaps.device)[None, :] < counts[active_rows, None]
    rows = active_rows[:, None].expand_as(indices)[valid]
    steps = indices[valid]
    sampled_p = probs.gather(1, indices)[valid]
    npositive = (active > 0).sum(1)
    log_ratio = -(npositive.double().log()[:, None] + probs.clamp_min(torch.finfo(torch.float64).tiny).log()) * beta
    maximum = log_ratio.masked_fill(active <= 0, -float('inf')).max()
    weights = (log_ratio.gather(1, indices)[valid] - maximum).exp()
    return rows, steps, counts, weights.to(gaps.dtype), sampled_p.to(gaps.dtype)


@torch.no_grad()
def screen_and_select(memory, env_ids, slot_ids, returns, task_ids, schedule, context):
    agent = context.get("_sil_value_agent")
    if agent is None:
        raise ValueError("Positive replay selection requires the current value agent")
    started = time.perf_counter()
    lengths = memory._archive_length[env_ids, slot_ids]
    width = returns.shape[1]
    positive = torch.zeros_like(returns, dtype=torch.bool)
    selection = getattr(memory.args, "sil_transition_selection", "window")
    replacement = selection == "gap_proportional_replacement"
    global_priority = selection == "global_gap_proportional_positive"
    gap_priority = replacement or global_priority or selection == "gap_proportional_positive"
    gaps = torch.zeros_like(returns) if gap_priority else None
    chunk = max(int(getattr(memory.args, "sil_microbatch_size", 0)), 1)
    if chunk == 1:
        chunk = 20480
    deadline = memory._deadline_for_batch(env_ids, task_ids, schedule) if schedule is not None else None
    for start in range(0, returns.numel(), chunk):
        flat = torch.arange(start, min(start + chunk, returns.numel()), device=memory.device)
        rows, steps = flat // width, flat % width
        valid = steps < lengths[rows]
        rows, steps = rows[valid], steps[valid]
        if not rows.numel():
            continue
        states = memory._archive_state[env_ids[rows], slot_ids[rows], steps].to(memory.dtype).clone()
        if deadline is not None:
            memory._relabel_public_deadline(states, deadline[rows])
        values = agent.get_value(states, denorm=False, update=False)[0].view(-1)
        targets = returns[rows, steps].to(memory.dtype)
        if memory.args.norm_value:
            targets = agent.normalize_value(targets, denorm=False, update=False,
                task_ids=task_ids[rows] if memory.args.pertask_norm else None)
        if not torch.isfinite(values).all() or not torch.isfinite(targets).all():
            raise FloatingPointError("Nonfinite value/target in positive replay screening")
        positive[rows, steps] = targets > values
        if gap_priority:
            gaps[rows, steps] = (targets - values).clamp_min(0)
    if not hasattr(memory, "_positive_selection_rng"):
        memory._positive_selection_rng = torch.Generator(device=memory.device)
        memory._positive_selection_rng.manual_seed(int(getattr(memory.args, "seed", 0)) + 17041)
    # Diagnostic callers can inspect selection without changing counts or RNG.
    committing = bool(context.get("_sil_building_replay_batch", False))
    rng_state = memory._positive_selection_rng.get_state() if not committing else None
    counts = memory._archive_selection_count[env_ids, slot_ids]
    if global_priority:
        budget = (
            int(memory._archive_num_envs)
            * int(memory.args.sil_replay_trajectories_per_env)
            * int(memory.args.sil_replay_transitions_per_trajectory)
        )
        rows, steps, selected_counts = global_gap_proportional_indices(
            gaps, budget, memory._positive_selection_rng,
            alpha=memory.args.sil_per_alpha,
        )
    elif replacement:
        rows, steps, selected_counts, weights, probabilities = replacement_indices(
            gaps, memory.args.sil_replay_transitions_per_trajectory, memory._positive_selection_rng,
            beta=memory.args.sil_per_beta, alpha=memory.args.sil_per_alpha)
        context['_replacement_per_fields'] = dict(sil_importance_weights=weights,
                                                  sil_sample_probabilities=probabilities)
    elif gap_priority:
        rows, steps, selected_counts = gap_proportional_indices(gaps,
            memory.args.sil_replay_transitions_per_trajectory, memory._positive_selection_rng)
    else:
        rows, steps, selected_counts = least_selected_indices(positive, counts,
            memory.args.sil_replay_transitions_per_trajectory, memory._positive_selection_rng)
    prior = counts[rows, steps]
    if committing:
        if replacement:
            memory._archive_selection_count.index_put_((env_ids[rows], slot_ids[rows], steps),
                torch.ones_like(steps, dtype=memory._archive_selection_count.dtype), accumulate=True)
        else:
            memory._archive_selection_count[env_ids[rows], slot_ids[rows], steps] += 1
    else:
        memory._positive_selection_rng.set_state(rng_state)
    stats = dict(replay_screened_candidates=float(lengths.sum().item()),
        replay_screen_positive_count=float(positive.sum().item()),
        replay_screen_selected_count=float(rows.numel()),
        replay_screen_selected_envs=float(env_ids[rows].unique().numel()),
        replay_screen_global=float(global_priority),
        replay_screen_budget=float(budget if global_priority else 0),
        replay_screen_duplicate_frac=float(
            1.0 - int(torch.unique(rows * width + steps).numel()) / int(rows.numel())
            if rows.numel() else 0.0
        ),
        replay_screen_new_selection_frac=float((prior == 0).float().mean().item()) if rows.numel() else 0.,
        replay_screen_prior_selection_mean=float(prior.float().mean().item()) if rows.numel() else 0.,
        replay_screen_seconds=time.perf_counter() - started)
    if committing:
        if replacement:
            stats.update(replay_per_beta=float(memory.args.sil_per_beta),
                         replay_importance_mean=float(weights.mean()) if weights.numel() else 0.,
                         replay_importance_min=float(weights.min()) if weights.numel() else 0.)
        context.setdefault("stats", {}).update(stats)
        folder = getattr(memory.args, "instance_dir", None)
        if folder:
            import json
            with open(os.path.join(folder, "positive_selection.jsonl"), "a") as f:
                f.write(json.dumps(dict(global_step=context.get("global_step"), **stats)) + "\n")
            # Sampler state only. Full training resume also requires the matching
            # archive and simulator state; an ordinary policy checkpoint is insufficient.
            path = os.path.join(folder, "positive_sampler_state.pt")
            torch.save(sampler_state(memory), path + ".tmp")
            os.replace(path + ".tmp", path)
    return rows, steps, selected_counts, torch.zeros_like(lengths, dtype=torch.bool)


def sampler_state(memory):
    return {"counts": memory._archive_selection_count.detach().cpu().clone(),
            "rng": memory._positive_selection_rng.get_state().clone(),
            "global_step": memory._archive_global_step.detach().cpu().clone(),
            "length": memory._archive_length.detach().cpu().clone(),
            "valid": memory._archive_valid.detach().cpu().clone()}


def load_sampler_state(memory, state):
    for name in ("global_step", "length", "valid"):
        if not torch.equal(getattr(memory, "_archive_" + name).cpu(), state[name]):
            raise ValueError("Sampler state requires matching archived trajectory identities")
    if memory._archive_selection_count.shape != state["counts"].shape:
        raise ValueError("Sampler count shape does not match archive")
    memory._archive_selection_count.copy_(state["counts"])
    memory._positive_selection_rng = torch.Generator(device=memory.device)
    memory._positive_selection_rng.set_state(state["rng"])
