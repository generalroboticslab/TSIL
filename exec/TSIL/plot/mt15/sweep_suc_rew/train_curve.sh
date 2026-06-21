#!/bin/bash
# Plot MT15 training curves for sweep_suc_rew/all.sh.
# Usage:
#   bash exec/TSIL/plot/mt15/sweep_suc_rew/train_curve.sh
#   bash exec/TSIL/plot/mt15/sweep_suc_rew/train_curve.sh --refresh
#   bash exec/TSIL/plot/mt15/sweep_suc_rew/train_curve.sh 0 1 5

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/../task_ids.sh"
source "${script_dir}/../../common.sh"

ENABLE_PVALUE_ANALYSIS=false
CURVE_Y_KEYS=("reward/success" "reward/eps_G" "signal/success_eps_time")
SUCCESS_REWARD_SCALE_LEGENDS=(
	"Success reward 10"
	"Success reward 100"
	"Success reward 1000"
	"Success reward 10000"
	"Success reward 100000"
	"Success reward 1000000"
)
EPSTIME_REWARD_SCALE_LEGENDS=(
	"Time reward 10"
	"Time reward 100"
	"Time reward 1000"
	"Time reward 10000"
	"Time reward 100000"
	"Time reward 1000000"
)

init_plot_args "${BASH_SOURCE[0]}" "$@"

METHODS=(
	PPO_MaxR_1e1
	PPO_MaxR_1e2
	PPO_MaxR_1e3
	PPO_MaxR_1e4
	PPO_MaxR_1e5
	PPO_MaxR_1e6
)
LEGENDS=("${SUCCESS_REWARD_SCALE_LEGENDS[@]}")
SAVE_GROUP="ih"
SAVE_PREFIX=""
run_train_curves

METHODS=(
	PPO_ATTL_TSIL_NOTRAIN_MaxR_1e1
	PPO_ATTL_TSIL_NOTRAIN_MaxR_1e2
	PPO_ATTL_TSIL_NOTRAIN_MaxR_1e3
	PPO_ATTL_TSIL_NOTRAIN_MaxR_1e4
	PPO_ATTL_TSIL_NOTRAIN_MaxR_1e5
	PPO_ATTL_TSIL_NOTRAIN_MaxR_1e6
)
LEGENDS=("${EPSTIME_REWARD_SCALE_LEGENDS[@]}")
SAVE_GROUP="attl"
SAVE_PREFIX=""
run_train_curves

METHODS=(
	PPO_ATTL_TSIL_TRANS_MaxR_1e1
	PPO_ATTL_TSIL_TRANS_MaxR_1e2
	PPO_ATTL_TSIL_TRANS_MaxR_1e3
	PPO_ATTL_TSIL_TRANS_MaxR_1e4
	PPO_ATTL_TSIL_TRANS_MaxR_1e5
	PPO_ATTL_TSIL_TRANS_MaxR_1e6
)
LEGENDS=("${EPSTIME_REWARD_SCALE_LEGENDS[@]}")
SAVE_GROUP="addl_sil"
SAVE_PREFIX=""
run_train_curves

METHODS=(
	PPO_FTTL_MaxR_1e1
	PPO_FTTL_MaxR_1e2
	PPO_FTTL_MaxR_1e3
	PPO_FTTL_MaxR_1e4
	PPO_FTTL_MaxR_1e5
	PPO_FTTL_MaxR_1e6
)
LEGENDS=("${EPSTIME_REWARD_SCALE_LEGENDS[@]}")
SAVE_GROUP="fttl"
SAVE_PREFIX=""
run_train_curves
