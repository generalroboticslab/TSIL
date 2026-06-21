#!/usr/bin/env bash
# Paper group 3: TSIL/SIL diagnostic tables.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../../../../.." && pwd)"
BENCHMARK="${BENCHMARK:-mt01}"
X_KEY="${X_KEY:-misc/steps}"

case "${BENCHMARK}" in
	mt01) task_file="${script_dir}/../../mt01/task_ids.sh" ;;
	mt15) task_file="${script_dir}/../../mt15/task_ids.sh" ;;
	*) echo "Group 3 supports BENCHMARK=mt01 or mt15." >&2; exit 1 ;;
esac
source "${task_file}"

TASK_ARGS=("${TASK_IDS[@]}")
if [[ "$#" -gt 0 ]]; then TASK_ARGS=("$@"); fi

cd "${repo_dir}"
python -m projects.TSIL.reports.paper_metrics \
	--write-sil-table \
	--table-prefix addl_sil \
	--table-dir "results/TSIL/paper_artifacts/tables/${BENCHMARK}/group_3_sil_analysis" \
	--benchmark "${BENCHMARK}" \
	--experiment compare_tsil \
	--training-stage scratch \
	--x-key "${X_KEY}" \
	--task-ids "${TASK_ARGS[@]}" \
	--methods PPO_ATTL_TSIL_NOTRAIN PPO_ATTL_SIL_TRANS PPO_ATTL_TSIL \
	--legends "ATTL" "ATTL+SIL" "TSIL"
