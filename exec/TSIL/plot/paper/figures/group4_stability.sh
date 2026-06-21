#!/usr/bin/env bash
# Paper group 4: stability experiment figures.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../../../../.." && pwd)"

BENCHMARK="${BENCHMARK:-mt01}"
X_KEY="${X_KEY:-misc/steps}"
PLOT_FORMAT="${PLOT_FORMAT:-pdf}"
ESTIMATED="${ESTIMATED:-0}"
EXCLUDE_EXPERIMENTS="${EXCLUDE_EXPERIMENTS-}"
MIN_TARGET_STEPS_BY_EXPERIMENT="${MIN_TARGET_STEPS_BY_EXPERIMENT-dense_dropout=50000000}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/timeaware_matplotlib}"

TASK_ARGS=()
for arg in "$@"; do
	case "${arg}" in
		--estimated) ESTIMATED=1 ;;
		--png) PLOT_FORMAT=png ;;
		--pdf) PLOT_FORMAT=pdf ;;
		--svg) PLOT_FORMAT=svg ;;
		--steps) X_KEY=misc/steps ;;
		--iterations) X_KEY=misc/iterations ;;
		T[0-9]*) TASK_ARGS+=("${arg#T}") ;;
		*) TASK_ARGS+=("${arg}") ;;
	esac
done

case "${BENCHMARK}" in
	mt01) task_file="${script_dir}/../../mt01/task_ids.sh" ;;
	mt15) task_file="${script_dir}/../../mt15/task_ids.sh" ;;
	*) echo "Unsupported BENCHMARK=${BENCHMARK}" >&2; exit 1 ;;
esac
source "${task_file}"

if [[ "${#TASK_ARGS[@]}" -eq 0 ]]; then
	TASK_ARGS=("${TASK_IDS[@]}")
fi

if [[ "${ESTIMATED}" == "1" || "${ESTIMATED}" == "true" ]]; then
	group_name="group_4_stability_estimates"
	input_args=(--input-table "results/TSIL/paper_artifacts/tables/${BENCHMARK}/${group_name}/stress_sweep_estimate_input.csv")
else
	group_name="group_4_stability_exps"
	input_args=()
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

figure_dir="results/TSIL/paper_artifacts/figures/${BENCHMARK}/${group_name}/${PLOT_FORMAT}"
figure_dir_abs="${repo_dir}/${figure_dir}"

cd "${repo_dir}"
python -m projects.TSIL.reports.stability_summary \
	--write-figures \
	--figure-dir "${figure_dir}" \
	--benchmark "${BENCHMARK}" \
	--task-ids "${TASK_ARGS[@]}" \
	--train-root results/TSIL/train_res \
	--format "${PLOT_FORMAT}" \
	--x-key "${X_KEY}" \
	"${input_args[@]}" \
	"${exclude_args[@]}" \
	"${min_target_args[@]}"

printf "Generated stability plots:\n"
printf "  %s\n" "${figure_dir_abs}/stress_sweep_robustness.${PLOT_FORMAT}"
