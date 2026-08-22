#!/usr/bin/env bash
set -euo pipefail

# One-time root bootstrap for the minimal CUDA 12.6 compiler. This deliberately
# installs no Linux GPU driver; WSL uses the host Windows NVIDIA driver.
if [[ "${EUID}" -ne 0 ]]; then
  echo "run as WSL root: wsl.exe -d Ubuntu -u root -- bash bootstrap_cuda126_wsl.sh" >&2
  exit 1
fi

TEMP_DIR="$(mktemp -d)"
trap 'rm -rf -- "${TEMP_DIR}"' EXIT
wget -q \
  https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb \
  -O "${TEMP_DIR}/cuda-keyring.deb"
dpkg -i "${TEMP_DIR}/cuda-keyring.deb"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  cuda-nvcc-12-6=12.6.85-1

/usr/local/cuda-12.6/bin/nvcc --version
du -sh /usr/local/cuda-12.6
