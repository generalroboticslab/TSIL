#!/usr/bin/env bash

stability_config() {
	local experiment="$1"
	case "${experiment}" in
		dense_dropout)
			TRAINING_STAGE="${TRAINING_STAGE:-scratch}"
			STABILITY_VIEW="${STABILITY_VIEW:-sweep}"
			STABILITY_BASELINE_FAMILY="${STABILITY_BASELINE_FAMILY:-TSIL_NOTRAIN}"
			STABILITY_BASELINE_LABEL="${STABILITY_BASELINE_LABEL:-ATTL}"
			STABILITY_TOKEN="${STABILITY_TOKEN:-Drop0.6}"
			STABILITY_TOKENS=(${STABILITY_TOKENS:-Drop0.4 Drop0.6 Drop0.8})
			STABILITY_TOKEN_LABELS=("drop 0.4" "drop 0.6" "drop 0.8")
			declare -p CURVE_Y_KEYS >/dev/null 2>&1 || CURVE_Y_KEYS=(reward/success reward/eps_G train/replay_dataset_trajectories)
			declare -p CURVE_SUMMARY_Y_KEYS >/dev/null 2>&1 || CURVE_SUMMARY_Y_KEYS=(reward/success)
			;;
		policy_grad_noise)
			TRAINING_STAGE="${TRAINING_STAGE:-scratch}"
			STABILITY_VIEW="${STABILITY_VIEW:-sweep}"
			STABILITY_BASELINE_FAMILY="${STABILITY_BASELINE_FAMILY:-TSIL_NOTRAIN}"
			STABILITY_BASELINE_LABEL="${STABILITY_BASELINE_LABEL:-ATTL}"
			STABILITY_TOKEN="${STABILITY_TOKEN:-PGradN20}"
			STABILITY_TOKENS=(${STABILITY_TOKENS:-PGradN5 PGradN10 PGradN20})
			STABILITY_TOKEN_LABELS=("grad 5" "grad 10" "grad 20")
			declare -p CURVE_Y_KEYS >/dev/null 2>&1 || CURVE_Y_KEYS=(reward/success reward/eps_G train/approx_kl train/policy_grad_noise_scale train/replay_dataset_trajectories)
			declare -p CURVE_SUMMARY_Y_KEYS >/dev/null 2>&1 || CURVE_SUMMARY_Y_KEYS=(reward/success)
			;;
		sweep_clip)
			TRAINING_STAGE="${TRAINING_STAGE:-scratch}"
			STABILITY_VIEW="${STABILITY_VIEW:-sweep}"
			STABILITY_BASELINE_FAMILY="${STABILITY_BASELINE_FAMILY:-TSIL_NOTRAIN}"
			STABILITY_BASELINE_LABEL="${STABILITY_BASELINE_LABEL:-ATTL}"
			STABILITY_TOKEN="${STABILITY_TOKEN:-Clip0.5}"
			STABILITY_TOKENS=(${STABILITY_TOKENS:-Clip0.3 Clip0.5 Clip0.7})
			STABILITY_TOKEN_LABELS=("clip 0.3" "clip 0.5" "clip 0.7")
			declare -p CURVE_Y_KEYS >/dev/null 2>&1 || CURVE_Y_KEYS=(reward/success reward/eps_G train/approx_kl train/replay_dataset_trajectories)
			declare -p CURVE_SUMMARY_Y_KEYS >/dev/null 2>&1 || CURVE_SUMMARY_Y_KEYS=(reward/success)
			;;
		sweep_lr)
			TRAINING_STAGE="${TRAINING_STAGE:-scratch}"
			STABILITY_VIEW="${STABILITY_VIEW:-sweep}"
			STABILITY_BASELINE_FAMILY="${STABILITY_BASELINE_FAMILY:-TSIL_NOTRAIN}"
			STABILITY_BASELINE_LABEL="${STABILITY_BASELINE_LABEL:-ATTL}"
			STABILITY_TOKEN="${STABILITY_TOKEN:-LR5e-3}"
			STABILITY_TOKENS=(${STABILITY_TOKENS:-LR1e-3 LR5e-3 LR1e-2})
			STABILITY_TOKEN_LABELS=("lr 1e-3" "lr 5e-3" "lr 1e-2")
			declare -p CURVE_Y_KEYS >/dev/null 2>&1 || CURVE_Y_KEYS=(reward/success reward/eps_G train/approx_kl train/replay_dataset_trajectories)
			declare -p CURVE_SUMMARY_Y_KEYS >/dev/null 2>&1 || CURVE_SUMMARY_Y_KEYS=(reward/success)
			;;
		*)
			echo "[stability] Unsupported experiment: ${experiment}" >&2
			return 1
			;;
	esac
}

stability_method_name() {
	local family="$1"
	local token="$2"
	local base="${STABILITY_BASE:-PPO_ATTL}"
	if [[ "${family}" == off ]]; then
		printf "%s_%s" "${base}" "${token}"
	else
		printf "%s_%s_%s" "${base}" "${family}" "${token}"
	fi
}

stability_build_methods() {
	local view="${STABILITY_VIEW:-methods}"
	local family="${STABILITY_FAMILY:-TSIL}"
	local baseline_family="${STABILITY_BASELINE_FAMILY:-TSIL_NOTRAIN}"
	local baseline_label="${STABILITY_BASELINE_LABEL:-ATTL}"
	local token="${STABILITY_TOKEN}"
	local idx
	METHODS=()
	LEGENDS=()

	if [[ "${view}" == sweep ]]; then
		for idx in "${!STABILITY_TOKENS[@]}"; do
			METHODS+=("$(stability_method_name "${family}" "${STABILITY_TOKENS[$idx]}")")
			LEGENDS+=("${STABILITY_TOKEN_LABELS[$idx]} ${family}")
		done
		SAVE_GROUP="${SAVE_GROUP:-sweep_${family}}"
		return
	fi

	METHODS=(
		"$(stability_method_name "${baseline_family}" "${token}")"
		"$(stability_method_name SIL_TRANS "${token}")"
		"$(stability_method_name TSIL "${token}")"
	)
	LEGENDS=(
		"${baseline_label}"
		"ATTL+SIL"
		"TSIL"
	)
	SAVE_GROUP="${SAVE_GROUP:-methods_${token}}"
}

run_stability_train_curves() {
	local script_path="$1"
	shift
	local script_dir experiment
	script_dir="$(cd -- "$(dirname -- "${script_path}")" && pwd)"
	experiment="${EXPERIMENT:-$(basename "${script_dir}")}"
	EXPERIMENT="${experiment}"

	stability_config "${experiment}"
	source "${script_dir}/../task_ids.sh"
	source "${script_dir}/../../common.sh"

	ENABLE_PVALUE_ANALYSIS="${ENABLE_PVALUE_ANALYSIS:-false}"
	stability_build_methods
	init_plot_args "${script_path}" "$@"
	run_train_curves
}
