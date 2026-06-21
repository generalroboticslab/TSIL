#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${script_dir}/../../../common.sh"
source "${script_dir}/../../task_ids.sh"

launch \
  benchmark=mt15 \
  experiment=policy_grad_noise \
  training_stage=scratch \
  method/sil=tsil_notrain,sil_trans,tsil \
  method/temporal=attl \
  task_id="${TASK_IDS}" \
  seed=42,43,44 \
  +force_args.policy_grad_noise_scale=5,10,20 \
  +force_args.target_kl=null \
  +force_args.max_grad_norm=1e9 \
  +force_args.total_timesteps=30000000 \
  "$@"
