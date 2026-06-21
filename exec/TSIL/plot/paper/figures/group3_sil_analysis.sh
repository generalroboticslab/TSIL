#!/bin/bash
# Paper group 3: TSIL/SIL report curves and direction-landscape figures.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../../../../.." && pwd)"
BENCHMARK="${BENCHMARK:-mt01}"

case "${BENCHMARK}" in
	mt01) task_file="${script_dir}/../../mt01/task_ids.sh" ;;
	mt15) task_file="${script_dir}/../../mt15/task_ids.sh" ;;
	*) echo "Group 3 supports BENCHMARK=mt01 or mt15." >&2; exit 1 ;;
esac
source "${task_file}"
source "${script_dir}/../../common.sh"

ENABLE_PVALUE_ANALYSIS=false
CURVE_SUCCESS_EPISODES_SUMMARY=false
EXPERIMENT=compare_tsil
TRAINING_STAGE=scratch
CURVE_Y_KEYS=(
	"train/sil_supervised_nll_topk_mean"
	"train/sil_archive_best_eps_time"
)
CURVE_SUMMARY_Y_KEYS=(
	"train/sil_supervised_nll_topk_mean"
	"train/sil_archive_best_eps_time"
)

run_family() {
	local family="$1"
	local out_name="$2"
	local -a original_tasks detail_tasks
	PLOT_ROOT="results/TSIL/paper_artifacts/figures/${BENCHMARK}/group_3_sil_analysis/${out_name}"
	PLOT_DIR="${PLOT_ROOT}/${PLOT_FORMAT}"
	SAVE_PREFIX=""
	METHODS=("${family}_TSIL_NOTRAIN" "${family}_SIL_TRANS" "${family}_TSIL")
	LEGENDS=("ATTL" "ATTL+SIL" "TSIL")

	original_tasks=("${TASKS[@]}")
	if [[ "${#original_tasks[@]}" -eq 0 ]]; then
		read -r -a detail_tasks <<< "${GROUP3_DETAIL_TASKS:-${TASK_IDS[*]}}"
	else
		detail_tasks=("${original_tasks[@]}")
	fi
	TASKS=("${detail_tasks[@]}")
	TASK_ARGS=(--task_ids "${TASKS[@]}")
	run_sil_direction_landscape_pair_grid
	if [[ "${GROUP3_INDIVIDUAL_LANDSCAPES:-false}" == true ]]; then
		run_sil_direction_landscapes
	fi

	TASKS=("${original_tasks[@]}")
	if [[ "${#TASKS[@]}" -eq 0 ]]; then
		TASK_ARGS=(--task_ids "${TASK_IDS[@]}")
	else
		TASK_ARGS=(--task_ids "${TASKS[@]}")
	fi
	run_train_curves
}

init_plot_args "${BASH_SOURCE[0]}" "$@"
cd "${repo_dir}"
run_family PPO_ATTL attl
