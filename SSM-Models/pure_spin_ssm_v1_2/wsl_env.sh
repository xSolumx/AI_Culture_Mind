#!/usr/bin/env bash

# Source this file before every v1.2 CUDA test or benchmark. It makes the
# pinned WSL environment, rather than Windows Python/CUDA, authoritative.

PURE_SPIN_V12_VENV="${PURE_SPIN_V12_VENV:-/home/local/.venvs/pure-spin-v12-torch210-cu126}"
if [[ ! -x "${PURE_SPIN_V12_VENV}/bin/python" ]]; then
  echo "missing Pure Spin v1.2 virtual environment: ${PURE_SPIN_V12_VENV}" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "${PURE_SPIN_V12_VENV}/bin/activate"

# Native Python environments stay in WSL's ext4 filesystem: compiled wheels,
# symlinks, and JIT imports are materially faster and more reliable there than
# on DrvFS. Only large weights and download caches live on E:.
export PURE_SPIN_V12_LARGE_ROOT="${PURE_SPIN_V12_LARGE_ROOT:-/mnt/e/AI_Culture_Mind_Large/pure_spin_ssm_v1_2}"
if [[ ! -d /mnt/e || ! -w /mnt/e ]]; then
  echo "E: is not mounted read/write at /mnt/e; refusing to cache on C:" >&2
  return 1 2>/dev/null || exit 1
fi

PURE_SPIN_V12_CUDA_ROOT="${PURE_SPIN_V12_CUDA_ROOT:-/usr/local/cuda-12.6}"
if [[ ! -x "${PURE_SPIN_V12_CUDA_ROOT}/bin/nvcc" ]]; then
  echo "missing pinned CUDA compiler: ${PURE_SPIN_V12_CUDA_ROOT}/bin/nvcc" >&2
  return 1 2>/dev/null || exit 1
fi

export CUDA_HOME="${PURE_SPIN_V12_CUDA_ROOT}"
export PATH="${VIRTUAL_ENV}/bin:${CUDA_HOME}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PURE_SPIN_V12_LOCAL_CACHE_ROOT="${PURE_SPIN_V12_LOCAL_CACHE_ROOT:-${HOME}/.cache/pure_spin_ssm_v1_2}"
export PIP_CACHE_DIR="${PURE_SPIN_V12_PIP_CACHE_DIR:-${PURE_SPIN_V12_LARGE_ROOT}/pip}"
export TORCH_EXTENSIONS_DIR="${PURE_SPIN_V12_EXTENSIONS_DIR:-${PURE_SPIN_V12_LOCAL_CACHE_ROOT}/torch_extensions/torch210_cu126}"
export TORCHINDUCTOR_CACHE_DIR="${PURE_SPIN_V12_TORCHINDUCTOR_CACHE_DIR:-${PURE_SPIN_V12_LOCAL_CACHE_ROOT}/torch_inductor}"
export TRITON_CACHE_DIR="${PURE_SPIN_V12_TRITON_CACHE_DIR:-${PURE_SPIN_V12_LOCAL_CACHE_ROOT}/triton}"
export CUDA_CACHE_PATH="${PURE_SPIN_V12_CUDA_CACHE_PATH:-${PURE_SPIN_V12_LOCAL_CACHE_ROOT}/cuda_cache}"
export HF_HOME="${PURE_SPIN_V12_HF_HOME:-${PURE_SPIN_V12_LARGE_ROOT}/huggingface}"
export HF_HUB_CACHE="${PURE_SPIN_V12_HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${PURE_SPIN_V12_HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export PURE_SPIN_V12_DATA_CACHE="${PURE_SPIN_V12_DATA_CACHE:-${PURE_SPIN_V12_LARGE_ROOT}/data}"

PURE_SPIN_V12_PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PURE_SPIN_V12_REPO_ROOT="$(cd "${PURE_SPIN_V12_PACKAGE_DIR}/../.." && pwd)"
export PYTHONPATH="${PURE_SPIN_V12_REPO_ROOT}/SSM-Models${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p \
  "${PIP_CACHE_DIR}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${CUDA_CACHE_PATH}" \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${PURE_SPIN_V12_DATA_CACHE}"
