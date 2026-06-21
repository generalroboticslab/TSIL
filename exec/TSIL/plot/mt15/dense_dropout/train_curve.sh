#!/bin/bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/../../stability_common.sh"
run_stability_train_curves "${BASH_SOURCE[0]}" "$@"
