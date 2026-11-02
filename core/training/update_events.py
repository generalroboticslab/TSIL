"""Diagnostic snapshots around large policy updates; never alters learning."""
from collections import deque
import copy
import json
from pathlib import Path
import random

import numpy as np
import torch


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


class UpdateEventDiagnostics:
    def __init__(self, args):
        self.args = args
        self.enabled = bool(getattr(args, "update_event_capture", False))
        self.ring = deque(maxlen=3)
        self.peak_success = 0.0
        self.captures = 0
        self.last_capture = -10
        self.current = None

    def before(self, trainer, batch, replay_context=None):
        self.current = None
        self.peak_success = max(self.peak_success, float(trainer.metrics.cur_success_rate))
        threshold = float(getattr(self.args, "update_event_peak_threshold", 0.2))
        if not self.enabled or self.captures >= 4 or self.peak_success < threshold:
            return
        if trainer.args.use_sil:
            if (not bool(getattr(self.args, "sil_aligned_replay", False))
                    or int(getattr(self.args, "sil_separate_updates", 0)) > 0
                    or bool(getattr(self.args, "use_lstm", False))
                    or replay_context is None):
                raise ValueError("Replay event capture requires aligned joint replay, a context, and no LSTM")
        snapshot = {
            "step": int(trainer.metrics.global_step),
            "iteration": int(trainer.metrics.global_update_iter),
            "success": float(trainer.metrics.cur_success_rate),
            "agent": cpu_copy(trainer.agent.state_dict()),
            "optimizer": cpu_copy(trainer.optimizer.state_dict()),
            "reward_normalizer": cpu_copy(trainer.reward_normalizer.state_dict()),
            "batch": cpu_copy(batch),
            "rng": rng_state(),
            "cur_ent": float(trainer.metrics.cur_ent),
            "schedule_context": cpu_copy(trainer._sil_schedule_context()),
        }
        if hasattr(trainer.policy_updater, "update_state_dict"):
            snapshot["updater_state"] = cpu_copy(trainer.policy_updater.update_state_dict())
        if trainer.args.use_sil:
            snapshot["capture_format_version"] = 2
            snapshot["aligned_replay"] = {"dataset": None, "epochs": []}
            replay_context["_update_event_recorder"] = self
        self.current = snapshot
        self.ring.append(snapshot)

    def prepared(self, replay_module, replay_context):
        """Record the state after diagnostics, before shuffling the first epoch."""
        replay = self.current["aligned_replay"]
        replay["context"] = cpu_copy({key: replay_context[key] for key in (
            "stats", "has_tsil_memory", "success_env_ids", "fallback_env_ids",
            "current_success_rate", "global_step",
        ) if key in replay_context})
        replay["full_actor_diagnostics"] = bool(getattr(replay_module, "_full_actor_diagnostics", False))
        replay["rng_before_epochs"] = rng_state()

    def replay_epoch(self, epoch, replay_context):
        """Save the selected dataset once and each actual permutation, without sampling."""
        replay = self.current["aligned_replay"]
        dataset = replay_context.get("_sil_aligned_replay_dataset")
        if dataset is not None and replay["dataset"] is None:
            replay["dataset"] = cpu_copy(dataset)
        fields = ("_sil_batch_size", "_sil_num_minibatches", "_sil_replay_epoch_index",
                  "_sil_replay_slice_index", "_sil_replay_num_slices", "_sil_replay_order", "stats")
        replay["epochs"].append({
            "epoch": int(epoch), "has_batch": replay_context.get("_sil_replay_batch") is not None,
            "context": cpu_copy({key: replay_context[key] for key in fields if key in replay_context}),
            "rng": rng_state(),
        })

    def after(self, trainer):
        if not self.ring or self.captures >= 4:
            return
        iteration = int(trainer.metrics.global_update_iter)
        rows = getattr(trainer.policy_updater, "last_update_trace", [])
        max_kl = max((row["approx_kl"] for row in rows), default=0.0)
        if iteration - self.last_capture < 5 or max_kl < float(getattr(self.args, "update_event_kl_threshold", 0.1)):
            return
        root = Path(self.args.instance_dir) / "update_events" / f"event_{self.captures:02d}_iter{iteration:04d}"
        root.mkdir(parents=True, exist_ok=False)
        for snapshot in self.ring:
            torch.save(snapshot, root / f"pre_iter{snapshot['iteration']:04d}.pt")
        torch.save(cpu_copy(trainer.agent.state_dict()), root / "post_agent.pt")
        torch.save(cpu_copy(trainer.optimizer.state_dict()), root / "post_optimizer.pt")
        (root / "manifest.json").write_text(json.dumps({
            "iteration": iteration, "step": int(trainer.metrics.global_step), "max_kl": max_kl,
            "success": float(trainer.metrics.cur_success_rate), "trace": rows,
            "scope": ("Exact PPO inputs plus selected aligned replay dataset and epoch permutations when enabled; "
                      "simulator state and the trajectory archive are not captured. Counterfactuals cannot "
                      "extend past the recorded epochs of an interrupted update.")
        }, indent=2) + "\n")
        print(f"[update_event] saved {root} max_kl={max_kl:.5f}", flush=True)
        self.captures += 1
        self.last_capture = iteration
