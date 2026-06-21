#!/bin/bash
set -euo pipefail

frames_dir="${1:-${FRAMES_DIR:-}}"
if [ -z "${frames_dir}" ]; then
	echo "Usage: bash exec/TSIL/eval/remove_black_background.sh <frames_dir> [output_dir]" >&2
	echo "Env override: FRAMES_DIR=<frames_dir>" >&2
	exit 1
fi
if [ $# -gt 0 ]; then
	shift
fi

output_dir="${1:-${frames_dir}_transparent}"
threshold="${BLACK_BG_THRESHOLD:-8}"

python -m core.evaluation.replay.background \
	"${frames_dir}" \
	--output "${output_dir}" \
	--threshold "${threshold}"
