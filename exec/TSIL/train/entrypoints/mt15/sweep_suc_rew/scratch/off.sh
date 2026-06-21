#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${script_dir}/../../../common.sh"
source "${script_dir}/../../task_ids.sh"

launch \
  benchmark=mt15 \
  experiment=sweep_suc_rew \
  training_stage=scratch \
  method/sil=off \
  method/temporal=ih \
  task_id="${TASK_IDS}" \
  seed=42 \
  +force_args.successRewardScale=10,100,1000,10000,100000,1000000 \
  "$@"
