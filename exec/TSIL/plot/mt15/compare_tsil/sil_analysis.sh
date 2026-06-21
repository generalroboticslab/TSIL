#!/bin/bash
# Plot MT15 SIL replay mechanism diagnostics and direction landscapes.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/../task_ids.sh"
source "${script_dir}/../../common.sh"

ENABLE_PVALUE_ANALYSIS=false
CURVE_Y_KEYS=(
	"train/sil_revisit_nll_topk_mean"
	"train/sil_supervised_nll_topk_mean"
	"train/sil_revisit_logp_topk_mean"
	"train/sil_archive_best_eps_time"
	"signal/fast_success_rate"
	"signal/fast_revisit_gap_steps"
)
CURVE_SUMMARY_Y_KEYS=(
	"train/sil_revisit_nll_topk_mean"
	"train/sil_supervised_nll_topk_mean"
	"signal/fast_success_rate"
)

run_family() {
	METHODS=("${1}_TSIL_NOTRAIN" "${1}_SIL_TRANS" "${1}_TSIL")
	LEGENDS=("ATTL" "ATTL+SIL" "TSIL")
	SAVE_GROUP="${3}"
	SAVE_PREFIX=""
	run_sil_revisit_analysis
	run_train_curves
	run_sil_direction_landscapes
}

init_plot_args "${BASH_SOURCE[0]}" "$@"
run_family PPO_ATTL ATTL attl
