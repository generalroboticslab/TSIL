#!/bin/bash
# Plot per-run learning signal maps for compare_temporal/all.sh.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/../task_ids.sh"
source "${script_dir}/../../common.sh"

METHODS=(
	PPO_ATTL_TSIL_NOTRAIN
	PPO_FTTL
	PPO_IHD2S
	PPO_IHSC
	PPO
)
METHOD_EXPERIMENTS=(
	compare_tsil
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
LEARNING_SIGNAL_LAST_FRAC="${LEARNING_SIGNAL_LAST_FRAC:-1.0}"
LEARNING_SIGNAL_SUMMARY_LAST_FRAC="${LEARNING_SIGNAL_SUMMARY_LAST_FRAC:-0.25}"
LEARNING_SIGNAL_RUN_LIMIT="${LEARNING_SIGNAL_RUN_LIMIT:-0}"

init_plot_args "${BASH_SOURCE[0]}" "$@"
run_learning_signal_maps
