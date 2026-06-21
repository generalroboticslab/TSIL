#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${script_dir}/../../../common.sh"
source "${script_dir}/../../task_ids.sh"

launch \
  benchmark=mt15 \
  experiment=sweep_lr \
  training_stage=scratch \
  method/sil=tsil_notrain,sil_trans,tsil \
  method/temporal=attl \
  task_id="${TASK_IDS}" \
  seed=42,43,44 \
  +force_args.lr=0.001,0.005,0.01 \
  +force_args.total_timesteps=30000000 \
  "$@"
