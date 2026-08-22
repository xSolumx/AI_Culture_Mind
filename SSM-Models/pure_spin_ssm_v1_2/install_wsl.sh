#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PURE_SPIN_V12_VENV="${PURE_SPIN_V12_VENV:-/home/local/.venvs/pure-spin-v12-torch210-cu126}"
export PURE_SPIN_V12_LARGE_ROOT="${PURE_SPIN_V12_LARGE_ROOT:-/mnt/e/AI_Culture_Mind_Large/pure_spin_ssm_v1_2}"
if [[ ! -d /mnt/e || ! -w /mnt/e ]]; then
  echo "E: is not mounted read/write at /mnt/e; refusing to cache on C:" >&2
  exit 1
fi
export PIP_CACHE_DIR="${PURE_SPIN_V12_PIP_CACHE_DIR:-${PURE_SPIN_V12_LARGE_ROOT}/pip}"
mkdir -p "${PIP_CACHE_DIR}"
if [[ ! -x "${PURE_SPIN_V12_VENV}/bin/python" ]]; then
  python3 -m venv "${PURE_SPIN_V12_VENV}"
fi
# shellcheck disable=SC1091
source "${PURE_SPIN_V12_VENV}/bin/activate"

python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r "${SCRIPT_DIR}/requirements-wsl.txt"

export CUDA_HOME="${PURE_SPIN_V12_CUDA_ROOT:-/usr/local/cuda-12.6}"
if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]] || \
   ! "${CUDA_HOME}/bin/nvcc" --version | grep -q 'release 12.6'; then
  echo "CUDA 12.6 nvcc is required at ${CUDA_HOME}/bin/nvcc." >&2
  echo "Install NVIDIA's WSL repo package cuda-nvcc-12-6=12.6.85-1." >&2
  exit 1
fi
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# Install the exact official binary tuple only when its exercised Mamba-2
# route is absent. --no-deps prevents newer, unused package metadata from
# replacing the pinned PyTorch/CUDA runtime.
if ! python - <<'PY'
import importlib.metadata
from mamba_ssm import Mamba2
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

assert importlib.metadata.version("mamba-ssm") == "2.3.2.post1"
assert importlib.metadata.version("causal-conv1d") == "1.7.0"
assert Mamba2 is not None
assert mamba_split_conv1d_scan_combined is not None
PY
then
  python -m pip install --no-deps \
    'https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1%2Bcu12torch2.10cxx11abiTRUE-cp310-cp310-linux_x86_64.whl' \
    'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.10cxx11abiTRUE-cp310-cp310-linux_x86_64.whl'
fi

python - <<'PY'
import torch
from mamba_ssm import Mamba2
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

assert torch.__version__ == "2.10.0+cu126"
assert torch.cuda.is_available()
assert Mamba2 is not None
assert mamba_split_conv1d_scan_combined is not None
print("validated official fused Mamba-2 environment")
PY

# wsl_env.sh makes Ninja, CUDA_HOME, the extension cache, and data cache part
# of the executable environment contract. The raw extension is built and all
# expected exported symbols are checked here rather than on a later benchmark.
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/wsl_env.sh"
cd "${PURE_SPIN_V12_REPO_ROOT}"
python SSM-Models/pure_spin_ssm_v1_2/verify_wsl_environment.py \
  --prepare-data \
  --output SSM-Models/pure_spin_ssm_v1_2/artifacts/wsl_environment.json
