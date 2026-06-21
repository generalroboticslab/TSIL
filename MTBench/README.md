# MTBench Environment Subset

This repository vendors the MTBench task and asset code needed by the
TSIL release. The public TSIL training loop uses its own
PPO/TSIL implementation under `core/` and only relies on MTBench for Isaac Gym
environment construction.

Included here:

- `assets/`: robot, object, texture, and scene assets.
- `isaacgymenvs/tasks/`: Meta-World and related Isaac Gym task definitions.
- `isaacgymenvs/cfg/task/`: task-level Hydra configs used to instantiate
  environments.
- `isaacgymenvs/utils/`: environment utility code used by the task classes.

Not included in this public TSIL branch:

- The legacy MTBench experiment shell scripts.
- The legacy MTBench training stack.
- The legacy third-party training backend bundled by upstream MTBench.

Use the top-level TSIL README for installation, training, evaluation,
and plotting commands.

Original MTBench project: https://github.com/Viraj-Joshi/MTBench
