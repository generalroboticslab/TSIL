#!/bin/bash
# Paper group 2: temporal target learning-signal figures.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../../../../.." && pwd)"
BENCHMARK="${BENCHMARK:-mt01}"

case "${BENCHMARK}" in
	mt01) task_file="${script_dir}/../../mt01/task_ids.sh" ;;
	mt15) task_file="${script_dir}/../../mt15/task_ids.sh" ;;
	*) echo "Group 2 supports BENCHMARK=mt01 or mt15." >&2; exit 1 ;;
esac
source "${task_file}"
source "${script_dir}/../../common.sh"

EXPERIMENT=compare_temporal
TRAINING_STAGE=scratch
SAVE_PREFIX=learning_signal_map
LEARNING_SIGNAL_LAST_FRAC="${LEARNING_SIGNAL_LAST_FRAC:-1.0}"
LEARNING_SIGNAL_SUMMARY_LAST_FRAC="${LEARNING_SIGNAL_SUMMARY_LAST_FRAC:-0.25}"
LEARNING_SIGNAL_RUN_LIMIT="${LEARNING_SIGNAL_RUN_LIMIT:-0}"
LEARNING_SIGNAL_MIN_COUNT="${LEARNING_SIGNAL_MIN_COUNT:-10}"
LEARNING_SIGNAL_NUM_BINS="${LEARNING_SIGNAL_NUM_BINS:-12}"

METHODS=(
	PPO_ATTL
	PPO_FTTL
	PPO_IHD2S
	PPO_IHSC
	PPO
)
METHOD_EXPERIMENTS=(
	compare_temporal
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
REQUESTED_TASKS=("${TASKS[@]}")
cd "${repo_dir}"
PLOT_ROOT="results/TSIL/paper_artifacts/figures/${BENCHMARK}/group_2_adaptive_ddl/compare_temporal"
PLOT_DIR="${PLOT_ROOT}/${PLOT_FORMAT}"
if [[ "${#REQUESTED_TASKS[@]}" -eq 0 ]]; then
	read -r -a TASKS <<< "${GROUP2_DETAIL_TASKS:-28}"
	read -r -a SUMMARY_TASKS <<< "${GROUP2_SUMMARY_TASKS:-${TASK_IDS[*]}}"
else
	TASKS=("${REQUESTED_TASKS[@]}")
	SUMMARY_TASKS=("${REQUESTED_TASKS[@]}")
fi
TASK_ARGS=(--task_ids "${TASKS[@]}")
if [[ "${GROUP2_SYNTHETIC:-0}" == "1" || "${GROUP2_SYNTHETIC:-false}" == "true" ]]; then
	task_id="${TASKS[0]#T}"
	save_dir="$(plot_save_dir)"
	synthetic_save_name="${GROUP2_SYNTHETIC_SAVE_NAME:-T${task_id}_map_synthetic}"
	python -m projects.TSIL.figures.adaptive_ddl_signal \
		--output-root "results/TSIL/paper_artifacts/synthetic/group_2_adaptive_ddl/${BENCHMARK}/compare_temporal/T${task_id}" \
		--save-path "${save_dir}/$(plot_save_name "${synthetic_save_name}")" \
		--task-id "${task_id}" \
		--num-episodes "${GROUP2_SYNTHETIC_EPISODES:-3600}" \
		--summary-samples "${GROUP2_SYNTHETIC_SUMMARY_SAMPLES:-3}" \
		--num-bins "${LEARNING_SIGNAL_NUM_BINS}" \
		--min-count "${GROUP2_SYNTHETIC_MIN_COUNT:-5}" \
		--last-frac "${LEARNING_SIGNAL_LAST_FRAC}" \
		--summary-last-frac "${LEARNING_SIGNAL_SUMMARY_LAST_FRAC}"
	exit 0
fi
LEARNING_SIGNAL_OUTPUTS=combined
LEARNING_SIGNAL_COMBINED_SAVE_NAME=map
LEARNING_SIGNAL_COMBINED_SUMMARY_TASKS="${SUMMARY_TASKS[*]}"
run_learning_signal_maps
