#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/wsl_env.sh"

cd "${PURE_SPIN_V12_REPO_ROOT}"
python SSM-Models/pure_spin_ssm_v1_2/verify_wsl_environment.py \
  --prepare-data \
  --output SSM-Models/pure_spin_ssm_v1_2/artifacts/wsl_environment.json
python -m pytest -q SSM-Models/pure_spin_ssm_v1_2
