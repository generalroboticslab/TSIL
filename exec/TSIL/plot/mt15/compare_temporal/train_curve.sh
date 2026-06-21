#!/bin/bash
# Plot MT15 training curves for compare_temporal/all.sh.
# Usage:
#   bash exec/TSIL/plot/mt15/compare_temporal/train_curve.sh
#   bash exec/TSIL/plot/mt15/compare_temporal/train_curve.sh --refresh
#   bash exec/TSIL/plot/mt15/compare_temporal/train_curve.sh 0 1 5

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/../task_ids.sh"
source "${script_dir}/../../common.sh"

ENABLE_PVALUE_ANALYSIS=false
METHODS=(
	PPO_ATTL_TSIL_NOTRAIN
	PPO_FTTL
	PPO_IHD2S
	PPO_IHSC
	PPO
)
METHOD_EXPERIMENTS=(
	compare_tsil
	compare_temporal
	compare_temporal
	compare_temporal
	compare_temporal
)
LEGENDS=(
	"ATTL"
	"FTTL"
	"D2S IH"
	"Step-cost IH"
	"IH"
)

init_plot_args "${BASH_SOURCE[0]}" "$@"
run_train_curves
