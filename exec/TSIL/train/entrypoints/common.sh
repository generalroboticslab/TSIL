#!/usr/bin/env bash

entrypoints_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
launcher="${entrypoints_dir}/../launcher.py"

launch() {
  local n_jobs="${N_JOBS:-8}"
  local script_time="${SCRIPT_TIME:-$(date +%Y%m%d_%H%M%S)}"
  local repo_dir
  repo_dir=$(cd -- "${entrypoints_dir}/../../../.." && pwd)

  export PYTHONPATH="${repo_dir}:${PYTHONPATH:-}"

  python -u "${launcher}" -m \
    hydra/launcher=joblib \
    hydra.launcher.n_jobs="${n_jobs}" \
    launch.n_jobs="${n_jobs}" \
    "script_time='${script_time}'" \
    "$@"
}
