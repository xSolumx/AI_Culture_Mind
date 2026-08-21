#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-wsl.txt

# Install the exact official binary tuple without allowing unrelated Mamba
# backend dependencies to replace the pinned PyTorch/CUDA runtime.
python -m pip install --no-deps \
  'https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1%2Bcu13torch2.10cxx11abiTRUE-cp310-cp310-linux_x86_64.whl' \
  'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu13torch2.10cxx11abiTRUE-cp310-cp310-linux_x86_64.whl'

python - <<'PY'
import torch
from mamba_ssm import Mamba2
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

assert torch.__version__ == "2.10.0+cu130"
assert torch.cuda.is_available()
assert Mamba2 is not None
assert mamba_split_conv1d_scan_combined is not None
print("validated official fused Mamba-2 environment")
PY
