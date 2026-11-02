"""Check the notebook's frozen actor gap and current-critic loss contract."""
import json
from pathlib import Path

notebook = json.loads((Path(__file__).resolve().parents[1] / 'notebooks/TSIL_demo.ipynb').read_text())
namespace = {}
for index in (3, 5, 11, 13):
    exec(compile(''.join(notebook['cells'][index]['source']), f'cell_{index}', 'exec'), namespace)
torch, nn = namespace['torch'], namespace['nn']


class TestPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(5))
        self.value = nn.Parameter(torch.tensor(0.0))

    def dist_value(self, obs):
        return namespace['Categorical'](logits=self.logits.expand(len(obs), -1)), self.value.expand(len(obs))


cfg = namespace['DemoConfig']()
model = TestPolicy()
batch = {'obs': torch.zeros(3, 4), 'actions': torch.tensor([0, 1, 2]),
         'returns': torch.tensor([1.0, 2.0, 3.0]), 'frozen_gaps': torch.tensor([1.0, 2.0, 3.0]),
         'weights': torch.tensor([1.0, 1.5, 2.0])}
original = {key: value.clone() for key, value in batch.items()}
loss, _ = namespace['compute_tsil_replay_loss'](model, batch, cfg)
before_actor, before_critic = torch.autograd.grad(loss, (model.logits, model.value))
with torch.no_grad():
    model.value.fill_(10.0)  # Close every current-critic gate without changing the policy.
loss, _ = namespace['compute_tsil_replay_loss'](model, batch, cfg)
after_actor, after_critic = torch.autograd.grad(loss, (model.logits, model.value))
assert torch.equal(before_actor, after_actor)
assert before_critic.abs() > 0 and after_critic == 0
assert all(torch.equal(value, original[key]) for key, value in batch.items())
assert cfg.update_epochs == 3
assert not hasattr(cfg, "sil_epochs")
print('PASS: actor gap stays frozen; current-critic residual changes; replay data is unchanged')

# A fixed rollout makes the number of optimizer steps comparable across methods.
# This catches an independent replay-epoch loop that would add replay-only steps.
step_counts = []
original_adam = torch.optim.Adam
original_collect = namespace['collect_rollouts']
original_model = namespace['ActorCritic']


class RecordingAdam(original_adam):
    def step(self, *args, **kwargs):
        step_counts.append(1)
        return super().step(*args, **kwargs)


def fixed_rollout(model, cfg, method, target):
    states = [namespace['np'].array([0, 0, i, target], dtype='float32') for i in range(8)]
    rewards = [0.0] * 7 + [1.0]
    steps = [(s, 0, -1.6, 0.0, rewards[i], i == 7, False, 0.0) for i, s in enumerate(states)]
    episode = dict(obs=states, actions=[0] * 8, rewards=rewards, base_rewards=rewards,
                   success=True, eps_time=8, dense_hits=0, terminated=True,
                   truncated=False, terminal_next_obs=states[-1], terminal_bootstrap_value=0.0)
    return steps, [episode]


def zero_value_model():
    model = original_model()
    nn.init.zeros_(model.v.weight)
    nn.init.zeros_(model.v.bias)
    return model


try:
    torch.optim.Adam = RecordingAdam
    namespace['collect_rollouts'] = fixed_rollout
    namespace['ActorCritic'] = zero_value_model
    tiny = namespace['DemoConfig'](iters=1, minibatch_size=4)
    counts = []
    for method in ('PPO', 'PPO+TSIL'):
        step_counts.clear()
        result = namespace['train'](method, 0, tiny)
        counts.append(len(step_counts))
        if method == 'PPO+TSIL':
            assert result['history'][0]['replay_positive_frac'] > 0
    assert counts == [6, 6], counts  # Three epochs, two PPO minibatches each.
finally:
    torch.optim.Adam = original_adam
    namespace['collect_rollouts'] = original_collect
    namespace['ActorCritic'] = original_model
print('PASS: PPO and PPO+TSIL each perform three joint epochs; no extra replay steps')

# Match benchmark fallback ranking: observed rewards decide archive admission;
# critic bootstraps are only used when replay targets are prepared.
def episode(length, reward, bootstrap=0.0, truncated=False):
    return dict(obs=[namespace['np'].zeros(4, dtype='float32') for _ in range(length)],
                actions=[0] * length, rewards=[reward] * length, base_rewards=[reward] * length,
                success=False, eps_time=length, terminated=False, truncated=truncated,
                terminal_next_obs=namespace['np'].zeros(4, dtype='float32'),
                terminal_bootstrap_value=bootstrap)

rank_cfg = namespace['DemoConfig'](sil_topk=1, gamma=.9)
memory = namespace['TsilTrajectoryMemory'](rank_cfg)
assert memory.add(episode(1, 2, truncated=True), 1)
assert not memory.add(episode(1, 0, bootstrap=10, truncated=True), 2)
assert memory.records[0]['return_score'] == 2
weight_cfg = namespace['DemoConfig'](sil_topk=2)
memory = namespace['TsilTrajectoryMemory'](weight_cfg)
assert memory.add(episode(1, 1), 1)
assert memory.add(episode(4, 1), 2)
batch = memory.sample_trajectory_batch(10, TestPolicy())
assert torch.allclose(torch.sort(batch['weights']).values,
                      torch.tensor([.625, .625, .625, .625, 2.5]))
print('PASS: observed-return archive ranking and inverse-count trajectory weights')
