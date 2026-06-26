#!/usr/bin/env bash

entrypoints_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
launcher="${entrypoints_dir}/../launcher.py"

launch() {
  local n_jobs="${N_JOBS:-1}"
  local gpus="${GPUS:-}"
  local script_time="${SCRIPT_TIME:-$(date +%Y%m%d_%H%M%S)}"
  local repo_dir
  repo_dir=$(cd -- "${entrypoints_dir}/../../../.." && pwd)

  export PYTHONPATH="${repo_dir}:${PYTHONPATH:-}"

  local gpu_override=()
  if [[ -n "${gpus}" ]]; then
    if [[ "${gpus}" != \[* ]]; then
      gpus="[${gpus}]"
    fi
    gpu_override=("launch.gpus=${gpus}")
  fi

  python -u "${launcher}" -m \
    hydra/launcher=joblib \
    hydra.launcher.n_jobs="${n_jobs}" \
    launch.n_jobs="${n_jobs}" \
    "${gpu_override[@]}" \
    "script_time='${script_time}'" \
    "$@"
}
