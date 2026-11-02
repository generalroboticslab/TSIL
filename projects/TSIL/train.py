"""TSIL training entrypoint."""

from core.training.trainer import run_training
from projects.TSIL.training import TSILTrainer, parse_args


def main():
    run_training(trainer_cls=TSILTrainer, args_fn=parse_args)


if __name__ == "__main__":
    main()
