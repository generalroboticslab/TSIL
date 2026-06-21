#!/usr/bin/env bash
# Paper group 2: temporal target learning-signal tables.

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

TASK_ARGS=("${TASK_IDS[@]}")
if [[ "$#" -gt 0 ]]; then TASK_ARGS=("$@"); fi

cd "${repo_dir}"
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
python -m projects.TSIL.reports.paper_metrics \
	--write-signal-table \
	--table-prefix learning_signal_mass \
	--table-dir "results/TSIL/paper_artifacts/tables/${BENCHMARK}/group_2_adaptive_ddl" \
	--benchmark "${BENCHMARK}" \
	--experiment compare_temporal \
	--training-stage scratch \
	--task-ids "${TASK_ARGS[@]}" \
	--methods "${METHODS[@]}" \
	--method-experiments "${METHOD_EXPERIMENTS[@]}" \
	--legends "${LEGENDS[@]}"
