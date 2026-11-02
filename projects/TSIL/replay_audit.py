"""Read-only A/C replay capture. No RNG draws, normalizer updates, or loss edits."""
import json
from pathlib import Path
import torch


def _destination(args, context):
    iteration = int(context["audit_iteration"])
    interval = int(args.sil_audit_interval)
    if not interval or (iteration > 1 and iteration % interval):
        return None
    if context.get("_ac_audit_recorded") == iteration:
        return None
    folder = Path(args.instance_dir) / "replay_audit"
    folder.mkdir(parents=True, exist_ok=True)
    context["_ac_audit_recorded"] = iteration
    return folder / ("iter%06d" % iteration)


def _json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def record_empty(args, context):
    path = _destination(args, context)
    if path is not None:
        _json(path.with_suffix('.json'), dict(iteration=context['audit_iteration'],
              steps=context['global_step'], empty=True, selected_transitions=0))


@torch.no_grad()
def capture(helper, agent, batch, context, normalized_value, raw_gap, weights):
    path = _destination(helper.args, context)
    if path is None:
        return
    task_ids = batch['task_ids']
    observed = batch['observed_returns'].detach()
    normalize_ids = task_ids if helper.args.pertask_norm else None
    if helper.args.norm_value:
        observed_norm = agent.normalize_value(observed, denorm=False, update=False, task_ids=normalize_ids)
        value = agent.normalize_value(normalized_value.detach(), denorm=True, update=False, task_ids=normalize_ids)
    else:
        observed_norm, value = observed, normalized_value.detach()
    observed_gap = observed_norm.view(-1) - normalized_value.detach().view(-1)
    trajectory_weights = batch.get('sil_weights', torch.ones_like(raw_gap))
    no_boot_weights = helper._normalize_actor_gap_weights(observed_gap.clamp_min(0) * trajectory_weights)
    success = batch['sil_success_mask'].bool()
    n = success.numel()
    summary = dict(iteration=context['audit_iteration'], steps=context['global_step'], empty=False,
                   selected_transitions=n, success_fraction=success.float().mean().item(),
                   audit_scope='first joint replay slice of selected PPO updates',
                   return_units='reward-normalized discounted returns; V denormalized from value statistics',
                   gap_units='value-normalized differences',
                   counterfactual='same samples and V, observed rewards only; full-slice gap std recomputed')
    for label, mask in [('success', success), ('failure', ~success)]:
        def mean(x):
            return x[mask].float().mean().item() if bool(mask.any()) else None
        summary[label] = dict(count=int(mask.sum()), target=mean(batch['returns']),
            observed=mean(observed), bootstrap=mean(batch['returns']-observed), value=mean(value),
            signed_gap=mean(raw_gap), gate_fraction=mean(raw_gap > 0),
            observed_only_gate_fraction=mean(observed_gap > 0),
            bootstrap_only_gate_fraction=mean((raw_gap > 0) & (observed_gap <= 0)),
            actor_weight_mean=mean(weights), observed_only_actor_weight_mean=mean(no_boot_weights))
    # Preserve every scalar in the logical slice, plus a bounded, stratified state sample.
    # Deterministic spacing within each outcome avoids changing the training RNG stream.
    picks=[]
    quota=max(int(helper.args.sil_audit_sample_size)//2, 1)
    for mask in (success, ~success):
        indices=mask.nonzero().view(-1)
        if indices.numel():
            positions=torch.linspace(0,indices.numel()-1,min(quota,indices.numel()),device=indices.device).long()
            picks.append(indices[positions])
    indices=torch.cat(picks)
    scalars={k:batch[k].detach().cpu().clone() for k in ['returns','observed_returns','sil_weights',
             'sil_success_mask','sil_return_fill_mask','task_ids','env_ids','slot_ids','step_indices','trajectory_lengths'] if k in batch}
    scalars.update(value=value.cpu().clone(), normalized_value=normalized_value.detach().cpu().clone(),
                   signed_gap=raw_gap.detach().cpu().clone(), actor_weights=weights.detach().cpu().clone(),
                   observed_only_gap=observed_gap.cpu(), observed_only_actor_weights=no_boot_weights.cpu())
    payload=dict(summary=summary, scalars=scalars, sample_indices=indices.cpu(),
        sample={k:batch[k][indices].detach().cpu().clone() for k in ['obs','state','action']},
        agent_state={k:v.detach().cpu().clone() for k,v in agent.state_dict().items()},
        reward_normalizer_state={k:v.detach().cpu().clone() for k,v in context['reward_normalizer'].state_dict().items()})
    temporary=path.with_suffix('.pt.tmp');torch.save(payload,temporary);temporary.replace(path.with_suffix('.pt'))
    _json(path.with_suffix('.json'), summary)
