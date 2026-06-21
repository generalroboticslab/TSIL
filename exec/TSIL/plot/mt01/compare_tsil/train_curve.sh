#!/bin/bash
# Plot MT01 training curves for compare_tsil/all.sh.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/../task_ids.sh"
source "${script_dir}/../../common.sh"

ENABLE_PVALUE_ANALYSIS=false

run_family() {
	METHODS=("${1}_TSIL_NOTRAIN" "${1}_SIL_TRANS" "${1}_TSIL")
	LEGENDS=("ATTL" "ATTL+SIL" "TSIL")
	SAVE_GROUP="${3}"
	SAVE_PREFIX=""
	run_train_curves
}

init_plot_args "${BASH_SOURCE[0]}" "$@"
run_family PPO_ATTL ATTL attl
