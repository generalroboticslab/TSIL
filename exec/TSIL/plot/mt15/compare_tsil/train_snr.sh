#!/bin/bash
# Plot local training signal metrics for compare_tsil/all.sh.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/../task_ids.sh"
source "${script_dir}/../../common.sh"

ENABLE_PVALUE_ANALYSIS=true
METHODS=(
	PPO_ATTL_TSIL_NOTRAIN
	PPO_ATTL_SIL_TRANS
	PPO_ATTL_TSIL
)
LEGENDS=(
	"ATTL"
	"ATTL+SIL"
	"TSIL"
)

init_plot_args "${BASH_SOURCE[0]}" "$@"
run_signal_metrics
