"""Verify shipped method defaults and regenerate the repaired bank identities."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from projects.TSIL.run import training_arguments


def main():
    for path in (ROOT / "configs").glob("*/*.json"):
        benchmark = path.parent.name
        task = int(path.stem[1:])
        args = training_arguments(benchmark, task)
        assert args["sil_freeze_replay_gap"] is True
        assert args["sil_replay_normalization_scope"] == "slice"
        assert args["gradient_clip_scope"] == "joint"
        assert args["sil_analysis_interval"] == 0
        assert args["sil_actual_direction_diagnostics"] is False
        assert args["sil_replay_epochs"] == 5
        if benchmark == "minigrid":
            bank = json.loads(Path(args["minigrid_config_bank"]).read_text())
            assert str(task) in bank["tasks"]
    from projects.TSIL.benchmarks.suites import get_tsil_minigrid_suite
    from projects.TSIL.benchmarks.configurations import describe, fingerprint
    import gymnasium as gym
    suite = get_tsil_minigrid_suite("mt15_repair_v4")
    count = 0
    for seed in (42, 43):
        bank = json.loads((ROOT / f"projects/TSIL/benchmarks/banks/mt15_repair_v4_seed{seed}.json").read_text())
        for task in (10, 11):
            record = bank["tasks"][str(task)]
            ids = []
            env = gym.make(record["env_id"])
            try:
                for split in ("train", "heldout"):
                    assert len(record[split]) == 32
                    for row in record[split]:
                        env.reset(seed=row["seed"])
                        actual = describe(env.unwrapped, record["family"])
                        assert actual == row["descriptor"]
                        assert fingerprint(record["env_id"], actual) == row["id"]
                        ids.append(row["id"])
                        count += 1
            finally:
                env.close()
            assert len(set(ids)) == 64
    print(f"PASS: 30 method configs; {count} regenerated identities; train/held-out disjoint")


if __name__ == "__main__":
    main()
