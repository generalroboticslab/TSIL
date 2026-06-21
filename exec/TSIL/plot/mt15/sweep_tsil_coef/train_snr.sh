#!/bin/bash
# Plot local training signal metrics for sweep_tsil_coef entrypoints.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/../task_ids.sh"
source "${script_dir}/../../common.sh"

ENABLE_PVALUE_ANALYSIS=false
SIL_COEF_LEGENDS=(
	"SIL coef 0.001"
	"SIL coef 0.01"
	"SIL coef 0.1"
	"SIL coef 1"
	"SIL coef 10"
)

run_family() {
	SAVE_GROUP="${1}"
	SAVE_PREFIX=""
	shift
	METHODS=("$@")
	LEGENDS=("${SIL_COEF_LEGENDS[@]}")
	run_signal_metrics
}

init_plot_args "${BASH_SOURCE[0]}" "$@"

run_family "addl_tsil_trans" \
	PPO_ATTL_SIL_tsil_trans_1e-3 \
	PPO_ATTL_SIL_tsil_trans_1e-2 \
	PPO_ATTL_SIL_tsil_trans_1e-1 \
	PPO_ATTL_SIL_tsil_trans_1e0 \
	PPO_ATTL_SIL_tsil_trans_1e1
