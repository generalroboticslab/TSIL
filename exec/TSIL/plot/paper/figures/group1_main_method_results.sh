#!/bin/bash
# Paper group 1: motivation and main method result figures.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../../../../.." && pwd)"
BENCHMARK="${BENCHMARK:-mt15}"
X_KEY="${X_KEY:-misc/steps}"
PLOT_FORMAT="${PLOT_FORMAT:-pdf}"
BAR_ERROR="${BAR_ERROR:-sem}"

TASK_ARGS=()
for arg in "$@"; do
	case "${arg}" in
		--png) PLOT_FORMAT=png ;;
		--pdf) PLOT_FORMAT=pdf ;;
		--svg) PLOT_FORMAT=svg ;;
		--steps) X_KEY=misc/steps ;;
		--iterations) X_KEY=misc/iterations ;;
		*) TASK_ARGS+=("${arg}") ;;
	esac
done

case "${BENCHMARK}" in
	mt01) task_file="${script_dir}/../../mt01/task_ids.sh" ;;
	mt15) task_file="${script_dir}/../../mt15/task_ids.sh" ;;
	*) echo "Group 1 supports BENCHMARK=mt01 or mt15." >&2; exit 1 ;;
esac
source "${task_file}"

if [[ "${#TASK_ARGS[@]}" -eq 0 ]]; then
	TASK_ARGS=("${TASK_IDS[@]}")
fi

METHODS=(
	PPO_ATTL_TSIL
	PPO_ATTL_SIL_TRANS
	PPO_ATTL_TSIL_NOTRAIN
	# PPO_ATTL
	PPO_FTTL
	PPO_IHD2S
	PPO_IHSC
	PPO
)
METHOD_EXPERIMENTS=(
	compare_tsil
	compare_tsil
	compare_tsil
	# compare_temporal
	compare_temporal
	compare_temporal
	compare_temporal
	compare_temporal
)
LEGENDS=(
	"TSIL"
	"ATTL+SIL"
	"ATTL"
	"FTTL"
	"D2S IH"
	"Step-cost IH"
	"IH"
)
DISPLAY_LABELS=("TSIL" "ATTL+SIL" "ATTL" "FTTL" "D2S IH" "Step-cost IH" "IH")

cd "${repo_dir}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/timeaware_matplotlib}"
export PYTHONPATH="${repo_dir}:${PYTHONPATH:-}"

python -m projects.TSIL.figures.main_method_results \
	--figure-dir "results/TSIL/paper_artifacts/figures/${BENCHMARK}/group_1_main_method_results/${PLOT_FORMAT}" \
	--benchmark "${BENCHMARK}" \
	--training-stage scratch \
	--task-ids "${TASK_ARGS[@]}" \
	--format "${PLOT_FORMAT}" \
	--x-key "${X_KEY}" \
	--bar-error "${BAR_ERROR}" \
	--methods "${METHODS[@]}" \
	--method-experiments "${METHOD_EXPERIMENTS[@]}" \
	--legends "${LEGENDS[@]}" \
	--display-labels "${DISPLAY_LABELS[@]}"
