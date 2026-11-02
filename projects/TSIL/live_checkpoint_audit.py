"""Evaluate a saved policy inside its still-live Meta-World training simulator."""

import hashlib
import json
from pathlib import Path

import torch

from projects.TSIL.benchmarks.metaworld_targets import load_target_bank


def tensor_sha256(value):
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def tensor_summary(value, include_first=False):
    value = value.detach()
    flat = value.reshape(-1)
    result = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "all_finite": bool(torch.isfinite(value).all().item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "mean": float(flat.float().mean().item()),
        "std": float(flat.float().std(unbiased=False).item()),
        "sha256": tensor_sha256(value),
    }
    if include_first:
        result["first"] = value[0].detach().cpu().tolist()
    return result


def module_state_sha256(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def verify_checkpoint_matches_agent(agent, checkpoint_path):
    saved = torch.load(checkpoint_path, map_location="cpu")
    live = agent.state_dict()
    missing = sorted(set(live) - set(saved))
    extra = sorted(set(saved) - set(live))
    changed = sorted(
        name for name in set(live) & set(saved)
        if not torch.equal(live[name].detach().cpu(), saved[name].detach().cpu())
    )
    return {
        "matches": not missing and not extra and not changed,
        "missing_keys": missing,
        "extra_keys": extra,
        "changed_keys": changed,
    }


def assign_exact_checkpoint_bank(env, bank):
    """Restore each live lane's checkpoint configuration and exact deadline."""
    features = torch.tensor(bank["features"], device=env.device, dtype=env.last_rand_vecs.dtype)
    deadlines = torch.tensor(
        bank["deadline_seconds"], device=env.device,
        dtype=env.obs_range["time2end"][1].dtype,
    )
    task_ids = torch.tensor(bank["task_ids"], device=env.device, dtype=torch.long)
    if features.shape != env.last_rand_vecs.shape:
        raise ValueError(
            f"Live environment and target-bank configurations differ: "
            f"{tuple(env.last_rand_vecs.shape)} vs {tuple(features.shape)}"
        )
    if not torch.equal(task_ids, env.task_indices.long()):
        raise ValueError("Live environment and target-bank task layouts differ")
    env.last_rand_vecs.copy_(features)
    env.obs_range["time2end"][0].copy_(-deadlines)
    env.obs_range["time2end"][1].copy_(deadlines)
    env.reset_all()
    if not torch.equal(env.last_rand_vecs, features):
        raise RuntimeError("Live reset changed checkpoint configurations")
    if not torch.allclose(env.real_time2end_init, deadlines, rtol=0, atol=1e-7):
        raise RuntimeError("Live reset did not apply exact checkpoint deadlines")
    return deadlines


def run_live_checkpoint_audit(trainer, checkpoint_name="last"):
    """Score one clean episode per lane without recreating the simulator."""
    args, env, agent = trainer.args, trainer.envs, trainer.agent
    checkpoint = Path(args.checkpoint_dir) / f"eps_{checkpoint_name}"
    output = Path(args.trajectory_dir) / f"inprocess_checkpoint_audit_{checkpoint_name}.json"
    bank = load_target_bank(checkpoint)
    checkpoint_comparison = verify_checkpoint_matches_agent(agent, checkpoint)
    if not checkpoint_comparison["matches"]:
        raise RuntimeError(f"Final checkpoint differs from live agent: {checkpoint_comparison}")

    # Preserve the long-running simulator and its explicit physical properties,
    # while disabling observation/action noise exactly as in checkpoint eval.
    env.training = False
    env.add_obs_noise = False
    env.add_act_noise = False
    env.cfg["apply_noise_eval"] = False
    agent.set_mode("eval")
    agent.deterministic = False

    deadlines = assign_exact_checkpoint_bank(env, bank)
    initial = env.reset()
    initial_obs = initial["obs"].to(trainer.device).clone()
    initial_state = initial["states"].to(trainer.device).clone()
    initial_policy_goal = env.states["target_pos"].detach().clone()
    obs = initial_obs
    normalizer = getattr(agent, "obs_normalizer", None)
    normalizer_before = module_state_sha256(normalizer) if normalizer is not None else None
    normalized_obs = agent.preprocess_obs(obs, update=False)
    normalizer_after = module_state_sha256(normalizer) if normalizer is not None else None

    action_seed = int(args.inprocess_audit_action_seed)
    torch.manual_seed(action_seed)
    torch.cuda.manual_seed_all(action_seed)
    active = torch.ones(env.num_envs, dtype=torch.bool, device=trainer.device)
    success = torch.zeros(env.num_envs, dtype=torch.bool, device=trainer.device)
    episode_time = torch.zeros(env.num_envs, dtype=torch.float32, device=trainer.device)
    episode_limit = torch.zeros_like(episode_time)
    first_action = None
    steps = 0
    control_steps_before = int(env.control_steps)
    with torch.no_grad():
        while bool(active.any().item()):
            action, _ = agent.get_action_and_value(obs, action_only=True)
            if first_action is None:
                first_action = action.detach().clone()
            next_obs, _, done, infos = env.step(action)
            terminal = done.to(trainer.device).bool() & active
            if terminal.any():
                success[terminal] = infos["success"][terminal].to(trainer.device).bool()
                episode_time[terminal] = infos["eps_time"][terminal].to(trainer.device).float()
                episode_limit[terminal] = infos["eps_time_limit"][terminal].to(trainer.device).float()
                active[terminal] = False
            obs = next_obs["obs"].to(trainer.device)
            steps += 1
            if steps > int(env.max_episode_length) + 2:
                raise RuntimeError(f"Live audit did not terminate every lane: {int(active.sum())} remain")

    completion = torch.where(
        success,
        (episode_time / episode_limit.clamp_min(1e-8)).clamp(max=1.0),
        torch.ones_like(episode_time),
    )
    payload = {
        "protocol": "same_process_live_training_simulator",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "target_bank_sha256": bank["sha256"],
        "checkpoint_matches_live_agent": checkpoint_comparison,
        "task": int(env.task_indices[0].item()),
        "num_envs": int(env.num_envs),
        "episodes": int(env.num_envs),
        "successes": int(success.sum().item()),
        "success_rate": float(success.float().mean().item()),
        "normalized_completion_time": float(completion.mean().item()),
        "action_mode": "stochastic",
        "evaluation_action_seed": action_seed,
        "noise_free": True,
        "simulator_control_steps_before_audit": control_steps_before,
        "audit_steps": steps,
        "configuration": tensor_summary(env.last_rand_vecs, include_first=True),
        "deadline_seconds": tensor_summary(deadlines, include_first=True),
        "world_object_start": tensor_summary(env.obj_init_pos, include_first=True),
        "world_goal": tensor_summary(env.target_pos, include_first=True),
        "policy_goal": tensor_summary(initial_policy_goal, include_first=True),
        "raw_observation": tensor_summary(initial_obs, include_first=True),
        "raw_state": tensor_summary(initial_state, include_first=True),
        "normalized_observation": tensor_summary(normalized_obs, include_first=True),
        "first_action": tensor_summary(first_action, include_first=True),
        "normalizer_sha256_before": normalizer_before,
        "normalizer_sha256_after": normalizer_after,
        "normalizer_frozen": normalizer_before == normalizer_after,
        "success_by_lane": success.detach().cpu().tolist(),
        "episode_time_by_lane": episode_time.detach().cpu().tolist(),
        "episode_limit_by_lane": episode_limit.detach().cpu().tolist(),
    }
    tensor_path = output.with_name(output.stem + "_tensors.pt")
    torch.save({
        "raw_observation": initial_obs.detach().cpu(),
        "normalized_observation": normalized_obs.detach().cpu(),
        "raw_state": initial_state.detach().cpu(),
        "first_action": first_action.detach().cpu(),
        "configuration": env.last_rand_vecs.detach().cpu(),
        "deadline_seconds": deadlines.detach().cpu(),
    }, tensor_path)
    payload["tensor_artifact"] = str(tensor_path)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        "Live checkpoint audit: "
        f"{payload['successes']}/{payload['episodes']} "
        f"({100 * payload['success_rate']:.3f}%) -> {output}",
        flush=True,
    )
    return payload
