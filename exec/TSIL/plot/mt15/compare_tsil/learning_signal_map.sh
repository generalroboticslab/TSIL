#!/bin/bash
# Plot per-run learning signal maps for compare_tsil/all.sh.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/../task_ids.sh"
source "${script_dir}/../../common.sh"

METHODS=(
	PPO_ATTL_TSIL_NOTRAIN
	PPO_ATTL_SIL_TRANS
	PPO_ATTL_TSIL
)
LEGENDS=("ATTL" "ATTL+SIL" "TSIL")

init_plot_args "${BASH_SOURCE[0]}" "$@"
run_learning_signal_maps
