#!/usr/bin/env bash
# Paper group 1: motivation and main method result tables.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../../../../.." && pwd)"
BENCHMARK="${BENCHMARK:-mt01}"
X_KEY="${X_KEY:-misc/steps}"

case "${BENCHMARK}" in
	mt01) task_file="${script_dir}/../../mt01/task_ids.sh" ;;
	mt15) task_file="${script_dir}/../../mt15/task_ids.sh" ;;
	*) echo "Group 1 supports BENCHMARK=mt01 or mt15." >&2; exit 1 ;;
esac
source "${task_file}"

TASK_ARGS=("${TASK_IDS[@]}")
if [[ "$#" -gt 0 ]]; then TASK_ARGS=("$@"); fi

cd "${repo_dir}"
METHODS=(
	PPO_ATTL_SIL_TRANS
	PPO_ATTL_TSIL
	PPO_ATTL_TSIL_NOTRAIN
	PPO_FTTL
	PPO_IHD2S
	PPO_IHSC
	PPO
)
LEGENDS=(
	"ATTL+SIL"
	"TSIL"
	"ATTL"
	"FTTL"
	"D2S IH"
	"Step-cost IH"
	"IH"
)
METHOD_EXPERIMENTS=(
	compare_tsil
	compare_tsil
	compare_tsil
	compare_temporal
	compare_temporal
	compare_temporal
	compare_temporal
)

python -m projects.TSIL.reports.paper_metrics \
	--write-table \
	--table-prefix main_metrics \
	--table-dir "results/TSIL/paper_artifacts/tables/${BENCHMARK}/group_1_main_method_results" \
	--benchmark "${BENCHMARK}" \
	--experiment compare_temporal \
	--training-stage scratch \
	--x-key "${X_KEY}" \
	--task-ids "${TASK_ARGS[@]}" \
	--methods "${METHODS[@]}" \
	--method-experiments "${METHOD_EXPERIMENTS[@]}" \
	--legends "${LEGENDS[@]}"

python -m projects.TSIL.reports.paper_metrics \
	--write-table \
	--table-prefix reward_sweep_ih_metrics \
	--table-dir "results/TSIL/paper_artifacts/tables/${BENCHMARK}/group_1_main_method_results" \
	--benchmark "${BENCHMARK}" \
	--experiment sweep_suc_rew \
	--training-stage scratch \
	--x-key "${X_KEY}" \
	--task-ids "${TASK_ARGS[@]}" \
	--methods PPO_MaxR_1e1 PPO_MaxR_1e2 PPO_MaxR_1e3 PPO_MaxR_1e4 PPO_MaxR_1e5 PPO_MaxR_1e6 \
	--legends "IH reward 10" "IH reward 100" "IH reward 1000" "IH reward 10000" "IH reward 100000" "IH reward 1000000"
