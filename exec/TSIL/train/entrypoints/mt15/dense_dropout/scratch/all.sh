#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${script_dir}/../../../common.sh"
source "${script_dir}/../../task_ids.sh"

launch \
  benchmark=mt15 \
  experiment=dense_dropout \
  training_stage=scratch \
  method/sil=tsil_notrain,sil_trans,tsil \
  method/temporal=attl \
  task_id="${TASK_IDS}" \
  seed=42,43,44 \
  +force_args.dense_reward_dropout_p=0.4,0.6,0.8 \
  +force_args.dense_reward_dropout_mode=episode \
  +force_args.total_timesteps=50000000 \
  "$@"
