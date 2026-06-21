#!/usr/bin/env python3
"""Hydra-first TSIL benchmark launcher entrypoint."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from launcher_utils import (  # noqa: E402
    acquire_gpu_slot,
    build_training_command,
    candidate_gpus,
    command_string,
    display_path,
    format_duration,
    hydra_job_num,
    hydra_output_dir,
    job_label,
    jobs_per_gpu,
    summarize_selection,
    total_jobs_from_multirun,
    write_failure_summary,
)


def run_single(cfg: DictConfig) -> int:
    job_started_at = time.perf_counter()
    if cfg.get("print_config", False):
        print(OmegaConf.to_yaml(cfg, resolve=True))

    command = build_training_command(cfg, python_executable=cfg.get("python_executable"))
    gpu_ids = candidate_gpus(cfg)
    job_num = hydra_job_num()
    output_dir = hydra_output_dir(REPO_ROOT)
    sweep_dir = output_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    command_path = output_dir / "command.txt"
    command_path.write_text(command_string(command) + "\n")
    selectors = summarize_selection(cfg)
    log_label = display_path(log_path, REPO_ROOT)
    total_jobs = total_jobs_from_multirun(sweep_dir)
    job_status_label = job_label(job_num, total_jobs)
    if total_jobs is not None:
        try:
            (sweep_dir / "total_runs.txt").write_text(f"{total_jobs}\n")
        except OSError:
            pass

    print(
        f"QUEUE job={job_status_label} gpus={','.join(gpu_ids) if gpu_ids else 'inherit'} "
        f"{selectors} log: {log_label}",
        flush=True,
    )

    if cfg.get("dry_run", False):
        log_path.write_text(f"DRY RUN\n{command_string(command)}\n")
        elapsed = format_duration(time.perf_counter() - job_started_at)
        print(
            f"DONE  job={job_status_label} returncode=0 elapsed={elapsed} dry_run=true log: {log_label}",
            flush=True,
        )
        return 0

    with acquire_gpu_slot(
        sweep_dir=sweep_dir,
        gpu_ids=gpu_ids,
        jobs_per_gpu=jobs_per_gpu(cfg, len(gpu_ids)),
        wait_seconds=5.0,
        job_num=job_num,
        log_path=log_path,
    ) as (gpu_id, slot_path):
        env = os.environ.copy()
        env["TIMEAWARE_SWEEP"] = "1"
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
        print(
            f"START job={job_status_label} gpu={gpu_id if gpu_id is not None else 'inherit'} "
            f"slot={slot_path.name if slot_path else 'none'} log: {log_label}",
            flush=True,
        )
        with log_path.open("w") as log_file:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
    status = "DONE " if result.returncode == 0 else "FAIL "
    elapsed = format_duration(time.perf_counter() - job_started_at)
    print(
        f"{status} job={job_status_label} gpu={gpu_id if gpu_id is not None else 'inherit'} "
        f"returncode={result.returncode} elapsed={elapsed} log: {log_label}",
        flush=True,
    )
    if result.returncode != 0:
        write_failure_summary(
            output_dir=output_dir,
            job_num=job_num,
            gpu_id=gpu_id,
            selectors=selectors,
            command=command,
            log_path=log_path,
            returncode=result.returncode,
        )
        raise SystemExit(int(result.returncode))
    return 0


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> int:
    return run_single(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
