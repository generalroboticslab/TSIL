#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${script_dir}/../../../common.sh"
source "${script_dir}/../../task_ids.sh"

launch \
  benchmark=mt01 \
  experiment=sweep_suc_rew \
  training_stage=scratch \
  method/sil=off \
  method/temporal=ih \
  task_id="${TASK_IDS}" \
  seed=42,43,44 \
  +force_args.successRewardScale=10,100,1000,10000,100000,1000000 \
  +force_args.same_init_config_per_task=False \
  +force_args.total_timesteps=30000000 \
  "$@"

launch \
  benchmark=mt01 \
  experiment=sweep_suc_rew \
  training_stage=scratch \
  method/sil=off \
  method/temporal=fttl \
  task_id="${TASK_IDS}" \
  seed=42,43,44 \
  '+force_args.epstimeRewardScale=[10,10],[100,100],[1000,1000],[10000,10000],[100000,100000],[1000000,1000000]' \
  +force_args.same_init_config_per_task=False \
  +force_args.total_timesteps=30000000 \
  "$@"

launch \
  benchmark=mt01 \
  experiment=sweep_suc_rew \
  training_stage=scratch \
  method/sil=tsil \
  method/temporal=attl \
  task_id="${TASK_IDS}" \
  seed=42,43,44 \
  '+force_args.epstimeRewardScale=[10,10],[100,100],[1000,1000],[10000,10000],[100000,100000],[1000000,1000000]' \
  +force_args.same_init_config_per_task=False \
  +force_args.total_timesteps=30000000 \
  "$@"
