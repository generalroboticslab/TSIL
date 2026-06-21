#!/usr/bin/env bash

DEFAULT_CURVE_Y_KEYS=("reward/success" "reward/eps_G" "signal/success_eps_time")
DEFAULT_CURVE_SUMMARY_Y_KEYS=("reward/success" "signal/success_eps_time")
DEFAULT_SIGNAL_KEYS=("top10pct_Rdense_sr" "succ_posadv_ratio" "succ_posadv_step_frac" "fast_succ_posadv_ratio")

init_plot_args() {
	local script_path="$1"
	shift
	local script_dir benchmark_dir
	script_dir="$(cd -- "$(dirname -- "${script_path}")" && pwd)"
	benchmark_dir="$(basename "$(dirname "${script_dir}")")"
	if [[ "${benchmark_dir}" != mt* ]]; then
		benchmark_dir="$(basename "${script_dir}")"
	fi

	BENCHMARK="${BENCHMARK:-${benchmark_dir}}"
	if [[ "$(basename "${script_dir}")" == mt* ]]; then
		EXPERIMENT="${EXPERIMENT:-compare_temporal}"
	else
		EXPERIMENT="${EXPERIMENT:-$(basename "${script_dir}")}"
	fi
	TRAINING_STAGE="${TRAINING_STAGE:-scratch}"
	PLOT_KIND="${PLOT_KIND:-$(basename "${script_path}" .sh)}"
	PLOT_FORMAT="${PLOT_FORMAT:-pdf}"
	PLOT_WANDB_PROJECT="${PLOT_WANDB_PROJECT:-${WANDB_PROJECT:-TSIL}}"
	export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/timeaware_matplotlib}"

	ENABLE_PVALUE_ANALYSIS="${ENABLE_PVALUE_ANALYSIS:-true}"
	REFRESH_ARGS=()
	TASKS=()
	for arg in "$@"; do
		case "$arg" in
			--refresh|--refresh_wandb) REFRESH_ARGS+=(--refresh_wandb) ;;
			--disable_pvalue_analysis) ENABLE_PVALUE_ANALYSIS=false ;;
			--enable_pvalue_analysis) ENABLE_PVALUE_ANALYSIS=true ;;
			--png) PLOT_FORMAT=png ;;
			--pdf) PLOT_FORMAT=pdf ;;
			--steps) CURVE_X_KEY=steps; SIGNAL_X_KEY=steps ;;
			--episodes) CURVE_X_KEY=misc/episodes; SIGNAL_X_KEY=misc/episodes ;;
			--iterations) CURVE_X_KEY=misc/iterations; SIGNAL_X_KEY=iteration ;;
			*) TASKS+=("$arg") ;;
		esac
	done

	PLOT_ROOT="results/TSIL/plot_res/${BENCHMARK}/${EXPERIMENT}/${TRAINING_STAGE}/${PLOT_KIND}"
	PLOT_DIR="${PLOT_ROOT}/${PLOT_FORMAT}"

	if [[ ${#TASKS[@]} -eq 0 ]]; then
		TASK_ARGS=(--task_ids "${TASK_IDS[@]}")
	else
		TASK_ARGS=(--task_ids "${TASKS[@]}")
	fi
	PVALUE_ARGS=()
	if [[ "${ENABLE_PVALUE_ANALYSIS}" == false ]]; then
		PVALUE_ARGS+=(--disable_pvalue_analysis)
	fi
}

plot_save_dir() {
	local save_dir
	if [[ -n "${SAVE_GROUP:-}" ]]; then
		save_dir="${PLOT_ROOT}/${SAVE_GROUP}/${PLOT_FORMAT}"
	else
		save_dir="${PLOT_DIR}"
	fi
	mkdir -p "${save_dir}"
	printf "%s" "${save_dir}"
}

plot_save_name() {
	local name="$1"
	local prefix="${SAVE_PREFIX-${PLOT_KIND}}"
	if [[ -n "${prefix}" ]]; then
		printf "%s_%s.%s" "${prefix}" "${name}" "${PLOT_FORMAT}"
	else
		printf "%s.%s" "${name}" "${PLOT_FORMAT}"
	fi
}

summary_plot_save_name() {
	printf "summary_%s" "$(plot_save_name "$1")"
}

plot_cmd() {
	local -a method_experiment_args=()
	local -a wandb_args=(--project "${PLOT_WANDB_PROJECT}")
	if declare -p METHOD_EXPERIMENTS >/dev/null 2>&1; then
		method_experiment_args=(--method_experiments "${METHOD_EXPERIMENTS[@]}")
	fi
	if [[ -n "${WANDB_ENTITY:-}" ]]; then
		wandb_args+=(--wandb_entity "${WANDB_ENTITY}")
	fi
	python -m projects.TSIL.plot \
		--benchmark "${BENCHMARK}" \
		--experiment "${EXPERIMENT}" \
		--training-stage "${TRAINING_STAGE}" \
		"${wandb_args[@]}" \
		"${TASK_ARGS[@]}" \
		--methods "${METHODS[@]}" \
		"${method_experiment_args[@]}" \
		--legends "${LEGENDS[@]}" \
		"${PVALUE_ARGS[@]}" \
		"${REFRESH_ARGS[@]}" \
		"$@"
}

method_experiment_for_index() {
	local method_idx="$1"
	if declare -p METHOD_EXPERIMENTS >/dev/null 2>&1; then
		if [[ "${#METHOD_EXPERIMENTS[@]}" -ne "${#METHODS[@]}" ]]; then
			echo "METHOD_EXPERIMENTS must match METHODS length" >&2
			return 1
		fi
		printf "%s" "${METHOD_EXPERIMENTS[$method_idx]}"
	else
		printf "%s" "${EXPERIMENT}"
	fi
}

curve_x_key_for_y() {
	local default_x_key="$1"
	local y_key="$2"
	if [[ "${y_key}" == "misc/success_episodes" ]]; then
		printf "%s" "misc/interaction_time"
	else
		printf "%s" "${default_x_key}"
	fi
}

run_train_curves() {
	local x_key="${CURVE_X_KEY:-misc/iterations}"
	local save_dir y_key y_name summary_y_key y_x_key
	local -a y_keys=("${CURVE_Y_KEYS[@]:-${DEFAULT_CURVE_Y_KEYS[@]}}")
	local -a summary_y_keys
	if declare -p CURVE_SUMMARY_Y_KEYS >/dev/null 2>&1; then
		summary_y_keys=("${CURVE_SUMMARY_Y_KEYS[@]}")
	elif [[ -n "${CURVE_SUMMARY_Y_KEY:-}" ]]; then
		summary_y_keys=("${CURVE_SUMMARY_Y_KEY}")
	else
		summary_y_keys=("${DEFAULT_CURVE_SUMMARY_Y_KEYS[@]}")
	fi
	if [[ "${CURVE_SUCCESS_EPISODES_SUMMARY:-true}" == true ]]; then
		local has_success_episodes=false
		for summary_y_key in "${summary_y_keys[@]}"; do
			if [[ "${summary_y_key}" == "misc/success_episodes" ]]; then
				has_success_episodes=true
				break
			fi
		done
		if [[ "${has_success_episodes}" == false ]]; then
			summary_y_keys+=("misc/success_episodes")
		fi
	fi

	save_dir="$(plot_save_dir)"
	for y_key in "${y_keys[@]}"; do
		y_name="${y_key##*/}"
		y_x_key="$(curve_x_key_for_y "${x_key}" "${y_key}")"
		plot_cmd --task BenchmarkTrainCurve --x_key "${y_x_key}" --y_key "${y_key}" \
			--save_path "${save_dir}/$(plot_save_name "${y_name}")"
	done

	for summary_y_key in "${summary_y_keys[@]}"; do
		y_name="${summary_y_key##*/}"
		y_x_key="$(curve_x_key_for_y "${x_key}" "${summary_y_key}")"
		plot_cmd --task BenchmarkTrainCurveSummary --x_key "${y_x_key}" --y_key "${summary_y_key}" \
			--save_path "${save_dir}/$(summary_plot_save_name "${y_name}")"
	done

	if [[ "${CURVE_PARETO:-false}" == true ]]; then
		plot_cmd --task BenchmarkSuccessSpeedPareto --x_key "${x_key}" \
			--save_path "${save_dir}/$(summary_plot_save_name "success_speed_pareto")"
	fi
}

run_signal_metrics() {
	local x_key="${SIGNAL_X_KEY:-iteration}"
	local save_dir metric_key
	local -a metric_keys=("${SIGNAL_KEYS[@]:-${DEFAULT_SIGNAL_KEYS[@]}}")

	save_dir="$(plot_save_dir)"
	for metric_key in "${metric_keys[@]}"; do
		plot_cmd --task BenchmarkSignalMetricGrid --x_key "${x_key}" --y_key "${metric_key}" \
			--save_path "${save_dir}/$(plot_save_name "${metric_key}")"
		plot_cmd --task BenchmarkSignalMetricSummary --x_key "${x_key}" --y_key "${metric_key}" \
			--save_path "${save_dir}/$(summary_plot_save_name "${metric_key}")"
	done
}

_latest_subdir() {
	local root="$1"
	local common_dir repo_dir
	if [[ ! -d "${root}" ]]; then
		return 0
	fi
	common_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
	repo_dir="$(cd -- "${common_dir}/../../.." && pwd)"
	PYTHONPATH="${repo_dir}:${PYTHONPATH:-}" python - "$root" <<'PY'
import sys
from projects.TSIL.ckpt_layout import latest_script_time_dir

latest = latest_script_time_dir(sys.argv[1])
if latest:
    print(latest)
PY
}

_selected_task_ids() {
	if [[ ${#TASKS[@]} -eq 0 ]]; then
		printf "%s\n" "${TASK_IDS[@]}"
	else
		printf "%s\n" "${TASKS[@]}"
	fi
}

_collect_learning_signal_histories_for_tasks() {
	local run_limit log_prefix task_token task_id method method_experiment method_dir script_time_dir run_dir history_path run_idx method_idx label
	local -n histories_out="$1"
	local -n labels_out="$2"
	run_limit="$3"
	log_prefix="$4"
	shift 4
	histories_out=()
	labels_out=()

	for task_token in "$@"; do
		task_id="${task_token#T}"
		for method_idx in "${!METHODS[@]}"; do
			method="${METHODS[$method_idx]}"
			label="${method}"
			if declare -p LEGENDS >/dev/null 2>&1 && [[ "${method_idx}" -lt "${#LEGENDS[@]}" ]]; then
				label="${LEGENDS[$method_idx]}"
			fi
			method_experiment="$(method_experiment_for_index "${method_idx}")"
			method_dir="results/TSIL/train_res/${BENCHMARK}/${method_experiment}/${TRAINING_STAGE}/T${task_id}/${method}"
			script_time_dir="$(_latest_subdir "${method_dir}")"
			if [[ -z "${script_time_dir}" ]]; then
				echo "[${log_prefix}] No runs for T${task_id}/${method}" >&2
				continue
			fi

			run_idx=0
			while IFS= read -r run_dir; do
				history_path="${run_dir}/trajectories/training_signal_history.jsonl"
				if [[ ! -f "${history_path}" ]]; then
					continue
				fi
				histories_out+=("${history_path}")
				labels_out+=("${label}")
				run_idx=$((run_idx + 1))
				if [[ "${run_limit}" != "0" && "${run_idx}" -ge "${run_limit}" ]]; then
					break
				fi
			done < <(find "${script_time_dir}" -mindepth 1 -maxdepth 1 -type d | sort)

			if [[ "${run_idx}" -eq 0 ]]; then
				echo "[${log_prefix}] No signal history for T${task_id}/${method}" >&2
			fi
		done
	done
}


run_learning_signal_maps() {
	local save_dir cache_dir cache_path run_limit last_frac summary_last_frac plot_last_frac num_bins min_count task_token task_id save_name output_save_name task_name
	local -a histories labels save_names summary_histories summary_labels summary_tasks cmd
	save_dir="$(plot_save_dir)"
	cache_dir="${LEARNING_SIGNAL_COMBINED_CACHE_DIR:-${PLOT_ROOT:-${save_dir%/*}}/values}"
	run_limit="${LEARNING_SIGNAL_RUN_LIMIT:-1}"
	last_frac="${LEARNING_SIGNAL_LAST_FRAC:-0.25}"
	summary_last_frac="${LEARNING_SIGNAL_SUMMARY_LAST_FRAC:-${last_frac}}"
	num_bins="${LEARNING_SIGNAL_NUM_BINS:-12}"
	min_count="${LEARNING_SIGNAL_MIN_COUNT:-5}"
	read -r -a save_names <<< "${LEARNING_SIGNAL_OUTPUTS:-map summary}"

	while IFS= read -r task_token; do
		task_id="${task_token#T}"
		_collect_learning_signal_histories_for_tasks histories labels "${run_limit}" "learning_signal_map" "${task_token}"
		if [[ "${#histories[@]}" -eq 0 ]]; then
			continue
		fi
		for save_name in "${save_names[@]}"; do
			plot_last_frac="${last_frac}"
			output_save_name="${save_name}"
			if [[ "${save_name}" == "summary" ]]; then
				plot_last_frac="${summary_last_frac}"
			elif [[ "${save_name}" == "combined" ]]; then
				output_save_name="${LEARNING_SIGNAL_COMBINED_SAVE_NAME:-combined}"
			fi
			task_name="LearningSignal${save_name^}"
			cmd=(
				python -m projects.TSIL.plot
				--task "${task_name}"
				--signal_history "${histories[@]}"
				--legends "${labels[@]}"
				--signal_num_bins "${num_bins}"
				--signal_last_frac "${plot_last_frac}"
				--signal_min_count "${min_count}"
			)
			if [[ "${save_name}" == "combined" ]]; then
				summary_histories=("${histories[@]}")
				summary_labels=("${labels[@]}")
				if [[ -n "${LEARNING_SIGNAL_COMBINED_SUMMARY_TASKS:-}" ]]; then
					read -r -a summary_tasks <<< "${LEARNING_SIGNAL_COMBINED_SUMMARY_TASKS}"
					_collect_learning_signal_histories_for_tasks summary_histories summary_labels "${run_limit}" "learning_signal_combined" "${summary_tasks[@]}"
					if [[ "${#summary_histories[@]}" -eq 0 ]]; then
						summary_histories=("${histories[@]}")
						summary_labels=("${labels[@]}")
					fi
				fi
				cmd+=(
					--signal_summary_history "${summary_histories[@]}"
					--signal_summary_legends "${summary_labels[@]}"
					--signal_summary_last_frac "${summary_last_frac}"
				)
				cache_path="${cache_dir}/$(plot_save_name "T${task_id}_${output_save_name}")"
				cmd+=(--signal_combined_cache_path "${cache_path%.*}_values.json")
				if [[ "${LEARNING_SIGNAL_COMBINED_REFRESH_CACHE:-0}" == "1" || "${LEARNING_SIGNAL_COMBINED_REFRESH_CACHE:-false}" == "true" ]]; then
					cmd+=(--signal_combined_refresh_cache)
				fi
			fi
			cmd+=(--save_path "${save_dir}/$(plot_save_name "T${task_id}_${output_save_name}")")
			"${cmd[@]}"
		done
	done < <(_selected_task_ids)
}



run_learning_signal_summary() {
	local save_dir run_limit summary_last_frac num_bins min_count task_token task_id method method_experiment method_dir script_time_dir run_dir history_path run_idx method_idx label
	local -a histories labels
	save_dir="$(plot_save_dir)"
	run_limit="${LEARNING_SIGNAL_RUN_LIMIT:-1}"
	summary_last_frac="${LEARNING_SIGNAL_SUMMARY_LAST_FRAC:-${LEARNING_SIGNAL_LAST_FRAC:-0.25}}"
	num_bins="${LEARNING_SIGNAL_NUM_BINS:-12}"
	min_count="${LEARNING_SIGNAL_MIN_COUNT:-5}"

	while IFS= read -r task_token; do
		task_id="${task_token#T}"
		for method_idx in "${!METHODS[@]}"; do
			method="${METHODS[$method_idx]}"
			label="${method}"
			if declare -p LEGENDS >/dev/null 2>&1 && [[ "${method_idx}" -lt "${#LEGENDS[@]}" ]]; then
				label="${LEGENDS[$method_idx]}"
			fi
			method_experiment="$(method_experiment_for_index "${method_idx}")"
			method_dir="results/TSIL/train_res/${BENCHMARK}/${method_experiment}/${TRAINING_STAGE}/T${task_id}/${method}"
			script_time_dir="$(_latest_subdir "${method_dir}")"
			if [[ -z "${script_time_dir}" ]]; then
				echo "[learning_signal_summary] No runs for T${task_id}/${method}" >&2
				continue
			fi

			run_idx=0
			while IFS= read -r run_dir; do
				history_path="${run_dir}/trajectories/training_signal_history.jsonl"
				if [[ ! -f "${history_path}" ]]; then
					continue
				fi
				histories+=("${history_path}")
				labels+=("${label}")
				run_idx=$((run_idx + 1))
				if [[ "${run_limit}" != "0" && "${run_idx}" -ge "${run_limit}" ]]; then
					break
				fi
			done < <(find "${script_time_dir}" -mindepth 1 -maxdepth 1 -type d | sort)

			if [[ "${run_idx}" -eq 0 ]]; then
				echo "[learning_signal_summary] No signal history for T${task_id}/${method}" >&2
			fi
		done
	done < <(_selected_task_ids)

	if [[ "${#histories[@]}" -eq 0 ]]; then
		return 0
	fi
	python -m projects.TSIL.plot \
		--task LearningSignalSummary \
		--signal_history "${histories[@]}" \
		--legends "${labels[@]}" \
		--signal_num_bins "${num_bins}" \
		--signal_last_frac "${summary_last_frac}" \
		--signal_min_count "${min_count}" \
		--save_path "${save_dir}/$(plot_save_name "${LEARNING_SIGNAL_SUMMARY_NAME:-summary}")"
}

run_sil_revisit_analysis() {
	local save_dir train_root run_limit task_token task_id method method_dir script_time_dir
	local run_dir run_idx method_idx label history_path save_name
	local -a histories labels
	save_dir="$(plot_save_dir)"
	train_root="results/TSIL/train_res/${BENCHMARK}/${EXPERIMENT}/${TRAINING_STAGE}"
	run_limit="${SIL_REVISIT_RUN_LIMIT:-0}"

	while IFS= read -r task_token; do
		task_id="${task_token#T}"
		histories=()
		labels=()
		for method_idx in "${!METHODS[@]}"; do
			method="${METHODS[$method_idx]}"
			label="${method}"
			if declare -p LEGENDS >/dev/null 2>&1 && [[ "${method_idx}" -lt "${#LEGENDS[@]}" ]]; then
				label="${LEGENDS[$method_idx]}"
			fi
			method_dir="${train_root}/T${task_id}/${method}"
			script_time_dir="$(_latest_subdir "${method_dir}")"
			if [[ -z "${script_time_dir}" ]]; then
				echo "[sil_revisit] No runs for T${task_id}/${method}" >&2
				continue
			fi

			run_idx=0
			while IFS= read -r run_dir; do
				history_path="${run_dir}/trajectories/training_signal_history.jsonl"
				if [[ -f "${history_path}" ]]; then
					histories+=("${history_path}")
					labels+=("${label}")
					run_idx=$((run_idx + 1))
				fi
				if [[ "${run_limit}" != "0" && "${run_idx}" -ge "${run_limit}" ]]; then
					break
				fi
			done < <(find "${script_time_dir}" -mindepth 1 -maxdepth 1 -type d | sort)

			if [[ "${run_idx}" -eq 0 ]]; then
				echo "[sil_revisit] No signal history for T${task_id}/${method}" >&2
			fi
		done
		if [[ "${#histories[@]}" -eq 0 ]]; then
			continue
		fi
		for save_name in mechanism outcome; do
			python -m projects.TSIL.plot \
				--task "SilRevisit${save_name^}" \
				--signal_history "${histories[@]}" \
				--legends "${labels[@]}" \
				--save_path "${save_dir}/$(plot_save_name "T${task_id}_sil_revisit_${save_name}")"
		done
	done < <(_selected_task_ids)
}

run_policy_direction_landscapes() {
	local save_dir idx json_path base_name
	save_dir="$(plot_save_dir)"
	idx=0
	for json_path in "$@"; do
		if [[ ! -f "${json_path}" ]]; then
			echo "[policy_direction_landscape] Missing file: ${json_path}" >&2
			continue
		fi
		base_name="$(basename "${json_path}")"
		base_name="${base_name%.*}"
		python -m projects.TSIL.plot \
			--task PolicyDirectionLandscape \
			--policy_landscape "${json_path}" \
			--save_path "${save_dir}/$(plot_save_name "${idx}_${base_name}")"
		idx=$((idx + 1))
	done
	if [[ "${idx}" -eq 0 ]]; then
		echo "Usage: provide one or more policy-landscape JSON files." >&2
		exit 1
	fi
}

sil_landscape_title() {
	case "$1" in
		*_TSIL) printf "%s" "TSIL" ;;
		*_SIL_TRANS) printf "%s" "SIL" ;;
		*) printf "%s" "$1" ;;
	esac
}

sil_landscape_legend_args() {
	case "$1" in
		*_SIL_TRANS) printf "%s" "--hide_legend" ;;
	esac
}

run_sil_direction_landscapes() {
	local save_dir train_root run_limit task_token task_id method method_dir script_time_dir
	local run_dir run_idx save_name n idx plot_title
	local -a json_paths selected_paths selected_labels legend_args
	save_dir="$(plot_save_dir)"
	train_root="results/TSIL/train_res/${BENCHMARK}/${EXPERIMENT}/${TRAINING_STAGE}"
	run_limit="${SIL_ANALYSIS_RUN_LIMIT:-1}"

	while IFS= read -r task_token; do
		task_id="${task_token#T}"
		for method in "${METHODS[@]}"; do
			if [[ "${method}" == *"_NOTRAIN"* ]]; then
				continue
			fi
			method_dir="${train_root}/T${task_id}/${method}"
			script_time_dir="$(_latest_subdir "${method_dir}")"
			if [[ -z "${script_time_dir}" ]]; then
				continue
			fi
			run_idx=0
			while IFS= read -r run_dir; do
				mapfile -t json_paths < <(find "${run_dir}/trajectories" -maxdepth 1 -name 'sil_direction_landscape_iter*.json' 2>/dev/null | sort)
				n="${#json_paths[@]}"
				if [[ "${n}" -gt 0 ]]; then
					save_name="T${task_id}_${method}_run${run_idx}_sil_direction_landscape"
					plot_title="$(sil_landscape_title "${method}")"
					read -r -a legend_args <<< "$(sil_landscape_legend_args "${method}")"
					selected_paths=()
					selected_labels=("25% steps" "50% steps" "75% steps" "100% steps")
					idx=$(( (n - 1) * 25 / 100 ))
					selected_paths+=("${json_paths[$idx]}")
					idx=$(( (n - 1) * 50 / 100 ))
					selected_paths+=("${json_paths[$idx]}")
					idx=$(( (n - 1) * 75 / 100 ))
					selected_paths+=("${json_paths[$idx]}")
					selected_paths+=("${json_paths[$((n - 1))]}")
					python -m projects.TSIL.plot \
						--task PolicyDirectionLandscape \
						--policy_landscape "${selected_paths[@]}" \
						--legends "${selected_labels[@]}" \
						--plot_title "${plot_title}" \
						"${legend_args[@]}" \
						--save_path "${save_dir}/$(plot_save_name "${save_name}")"
				fi
				run_idx=$((run_idx + 1))
				if [[ "${run_limit}" != "0" && "${run_idx}" -ge "${run_limit}" ]]; then
					break
				fi
			done < <(find "${script_time_dir}" -mindepth 1 -maxdepth 1 -type d | sort)
		done
	done < <(_selected_task_ids)
}


run_sil_direction_landscape_pair_grid() {
	local save_dir train_root run_limit task_token task_id suffix candidate method method_dir script_time_dir
	local run_dir run_idx n idx save_name group_count
	local -a json_paths selected_paths selected_labels combined_paths method_names run_dirs
	save_dir="$(plot_save_dir)"
	train_root="results/TSIL/train_res/${BENCHMARK}/${EXPERIMENT}/${TRAINING_STAGE}"
	run_limit="${SIL_ANALYSIS_RUN_LIMIT:-1}"
	selected_labels=("25% steps" "50% steps" "75% steps" "100% steps")

	while IFS= read -r task_token; do
		task_id="${task_token#T}"
		run_idx=0
		while true; do
			combined_paths=()
			method_names=()
			for suffix in "_TSIL" "_SIL_TRANS"; do
				method=""
				for candidate in "${METHODS[@]}"; do
					if [[ "${candidate}" == *"${suffix}" ]]; then
						method="${candidate}"
						break
					fi
				done
				if [[ -z "${method}" ]]; then
					continue
				fi
				method_dir="${train_root}/T${task_id}/${method}"
				script_time_dir="$(_latest_subdir "${method_dir}")"
				if [[ -z "${script_time_dir}" ]]; then
					continue
				fi
				mapfile -t run_dirs < <(find "${script_time_dir}" -mindepth 1 -maxdepth 1 -type d | sort)
				if [[ "${run_idx}" -ge "${#run_dirs[@]}" ]]; then
					continue
				fi
				run_dir="${run_dirs[$run_idx]}"
				mapfile -t json_paths < <(find "${run_dir}/trajectories" -maxdepth 1 -name 'sil_direction_landscape_iter*.json' 2>/dev/null | sort)
				n="${#json_paths[@]}"
				if [[ "${n}" -eq 0 ]]; then
					continue
				fi
				selected_paths=()
				idx=$(( (n - 1) * 25 / 100 ))
				selected_paths+=("${json_paths[$idx]}")
				idx=$(( (n - 1) * 50 / 100 ))
				selected_paths+=("${json_paths[$idx]}")
				idx=$(( (n - 1) * 75 / 100 ))
				selected_paths+=("${json_paths[$idx]}")
				selected_paths+=("${json_paths[$((n - 1))]}")
				combined_paths+=("${selected_paths[@]}")
				method_names+=("${method}")
			done
			group_count="${#method_names[@]}"
			if [[ "${group_count}" -eq 0 ]]; then
				break
			elif [[ "${group_count}" -eq 1 ]]; then
				save_name="T${task_id}_${method_names[0]}_only_run${run_idx}_sil_direction_landscape"
			else
				save_name="T${task_id}_${method_names[0]}_vs_${method_names[1]}_run${run_idx}_sil_direction_landscape"
			fi
			python -m projects.TSIL.plot \
				--task PolicyDirectionLandscapeGroupGrid \
				--policy_landscape "${combined_paths[@]}" \
				--legends "${selected_labels[@]}" \
				--policy_landscape_group_size 4 \
				--save_path "${save_dir}/$(plot_save_name "${save_name}")"
			run_idx=$((run_idx + 1))
			if [[ "${run_limit}" != "0" && "${run_idx}" -ge "${run_limit}" ]]; then
				break
			fi
		done
	done < <(_selected_task_ids)
}
