#!/bin/bash
# Plot MT15 training curves for sweep_tsil_coef entrypoints.
# Usage:
#   bash exec/TSIL/plot/mt15/sweep_tsil_coef/train_curve.sh
#   bash exec/TSIL/plot/mt15/sweep_tsil_coef/train_curve.sh --refresh
#   bash exec/TSIL/plot/mt15/sweep_tsil_coef/train_curve.sh 0 1 5

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/../task_ids.sh"
source "${script_dir}/../../common.sh"

ENABLE_PVALUE_ANALYSIS=false
CURVE_Y_KEYS=("reward/success" "reward/eps_G" "signal/success_eps_time")
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
	run_train_curves
}

init_plot_args "${BASH_SOURCE[0]}" "$@"

run_family "addl_tsil_trans" \
	PPO_ATTL_SIL_tsil_trans_1e-3 \
	PPO_ATTL_SIL_tsil_trans_1e-2 \
	PPO_ATTL_SIL_tsil_trans_1e-1 \
	PPO_ATTL_SIL_tsil_trans_1e0 \
	PPO_ATTL_SIL_tsil_trans_1e1
