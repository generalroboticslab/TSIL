#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${script_dir}/../../../common.sh"
source "${script_dir}/../../task_ids.sh"

launch \
  benchmark=mt01 \
  experiment=compare_tsil \
  training_stage=scratch \
  method/sil=tsil_notrain,sil_trans,tsil \
  method/temporal=attl \
  task_id="${TASK_IDS}" \
  seed=42,43,44 \
  +force_args.total_timesteps=30000000 \
  "$@"
