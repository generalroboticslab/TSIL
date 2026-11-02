"""One-time MiniGrid D2S critic reset, synchronized with the reward switch."""
import json
from pathlib import Path
import torch


def maybe_reset_critic(trainer, global_step):
    args = trainer.args
    threshold = float(args.hard_clip) * args.total_timesteps
    if getattr(trainer, '_d2s_critic_reset_step', None) is not None or global_step < threshold:
        return
    from projects.TSIL.benchmarks.minigrid import MiniGridCNN
    # Recreate the original critic initialization without perturbing action RNG.
    # Keep parameter identities so optimizer groups and rollout references remain valid.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(args.seed) + 7919)
        fresh = MiniGridCNN(trainer.envs, args, trainer.agent.critic_dim)
    trainer.agent.critic.load_state_dict(fresh.state_dict())
    for parameter in trainer.agent.critic.parameters():
        trainer.optimizer.state.pop(parameter, None)
    trainer._d2s_critic_reset_step = int(global_step)
    event = dict(nominal_step=int(global_step), threshold=threshold,
                 critic_reset=True, critic_optimizer_state_reset=True,
                 actor_preserved=True, normalizers_preserved=True)
    if getattr(args, 'instance_dir', None):
        Path(args.instance_dir, 'd2s_switch.json').write_text(json.dumps(event, indent=2)+'\n')
    print('D2S critic reset:', event, flush=True)
