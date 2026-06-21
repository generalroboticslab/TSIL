#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${script_dir}/../../../common.sh"
source "${script_dir}/../../task_ids.sh"

launch \
  benchmark=mt01 \
  experiment=sweep_tsil_coef \
  training_stage=scratch \
  method/sil=bc \
  method/temporal=fttl \
  task_id="${TASK_IDS}" \
  seed=42 \
  +force_args.sil_coef=0.001,0.01,0.1,1,10 \
  "$@"
