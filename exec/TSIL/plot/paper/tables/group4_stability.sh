#!/usr/bin/env bash
# Paper group 4: stability experiment tables.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../../../../.." && pwd)"

BENCHMARK="${BENCHMARK:-mt01}"
X_KEY="${X_KEY:-misc/steps}"
ESTIMATED="${ESTIMATED:-0}"
EXCLUDE_EXPERIMENTS="${EXCLUDE_EXPERIMENTS-}"
MIN_TARGET_STEPS_BY_EXPERIMENT="${MIN_TARGET_STEPS_BY_EXPERIMENT-dense_dropout=50000000}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/timeaware_matplotlib}"

case "${BENCHMARK}" in
	mt01) task_file="${script_dir}/../../mt01/task_ids.sh" ;;
	mt15) task_file="${script_dir}/../../mt15/task_ids.sh" ;;
	*) echo "Unsupported BENCHMARK=${BENCHMARK}" >&2; exit 1 ;;
esac
source "${task_file}"

TASK_ARGS=()
for arg in "$@"; do
	case "${arg}" in
		--estimated) ESTIMATED=1 ;;
		T[0-9]*) TASK_ARGS+=("${arg#T}") ;;
		*) TASK_ARGS+=("${arg}") ;;
	esac
done
if [[ "${#TASK_ARGS[@]}" -eq 0 ]]; then
	TASK_ARGS=("${TASK_IDS[@]}")
fi

strict_args=()
if [[ "${STRICT:-0}" == "1" || "${STRICT:-0}" == "true" ]]; then
	strict_args+=(--strict)
fi

exclude_args=()
if [[ -n "${EXCLUDE_EXPERIMENTS}" ]]; then
	read -r -a exclude_list <<< "${EXCLUDE_EXPERIMENTS}"
	exclude_args+=(--exclude-experiments "${exclude_list[@]}")
fi

min_target_args=()
if [[ -n "${MIN_TARGET_STEPS_BY_EXPERIMENT}" ]]; then
	read -r -a min_target_list <<< "${MIN_TARGET_STEPS_BY_EXPERIMENT}"
	min_target_args+=(--min-target-steps-by-experiment "${min_target_list[@]}")
fi

if [[ "${ESTIMATED}" == "1" || "${ESTIMATED}" == "true" ]]; then
	group_name="group_4_stability_estimates"
	input_args=(--input-table "results/TSIL/paper_artifacts/tables/${BENCHMARK}/${group_name}/stress_sweep_estimate_input.csv")
else
	group_name="group_4_stability_exps"
	input_args=()
fi

cd "${repo_dir}"
python -m projects.TSIL.reports.stability_summary \
	--write-tables \
	--table-dir "results/TSIL/paper_artifacts/tables/${BENCHMARK}/${group_name}" \
	--benchmark "${BENCHMARK}" \
	--task-ids "${TASK_ARGS[@]}" \
	--train-root results/TSIL/train_res \
	--x-key "${X_KEY}" \
	"${input_args[@]}" \
	"${exclude_args[@]}" \
	"${min_target_args[@]}" \
	"${strict_args[@]}"
