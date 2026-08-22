#!/usr/bin/env bash
set -euo pipefail

# llama.cpp is deliberately compiled on WSL's native ext4 filesystem. Only
# large GGUF files and their download cache belong on the slower E: mount.
LLAMA_CPP_COMMIT="${LLAMA_CPP_COMMIT:-86722900390abf479fd9719eda12c299a2b25bbf}"
LLAMA_CPP_SRC="${LLAMA_CPP_SRC:-/home/local/src/llama.cpp}"
LLAMA_CPP_BUILD="${LLAMA_CPP_BUILD:-${LLAMA_CPP_SRC}/build-cu126}"
LLAMA_CPP_BUILD_VENV="${LLAMA_CPP_BUILD_VENV:-/home/local/.venvs/llama-cpp-build}"
PURE_SPIN_V12_VENV="${PURE_SPIN_V12_VENV:-/home/local/.venvs/pure-spin-v12-torch210-cu126}"
GGUF_ROOT="${GGUF_ROOT:-/mnt/e/AI_Culture_Mind_Large/models/gguf}"

if [[ ! -d /mnt/e || ! -w /mnt/e ]]; then
  echo "E: is not mounted read/write at /mnt/e; refusing a C:-backed model cache" >&2
  exit 1
fi
mkdir -p "${GGUF_ROOT}" "$(dirname "${LLAMA_CPP_SRC}")" "${HOME}/.local/bin"

if [[ ! -x "${LLAMA_CPP_BUILD_VENV}/bin/python" ]]; then
  python3 -m venv "${LLAMA_CPP_BUILD_VENV}"
fi
"${LLAMA_CPP_BUILD_VENV}/bin/python" -m pip install --no-cache-dir \
  cmake==4.4.2 ninja==1.13.0

if [[ ! -d "${LLAMA_CPP_SRC}/.git" ]]; then
  git clone https://github.com/ggml-org/llama.cpp.git "${LLAMA_CPP_SRC}"
fi
if ! git -C "${LLAMA_CPP_SRC}" diff --quiet || \
   ! git -C "${LLAMA_CPP_SRC}" diff --cached --quiet; then
  echo "llama.cpp source tree is dirty; refusing to overwrite local work" >&2
  exit 1
fi
git -C "${LLAMA_CPP_SRC}" fetch origin "${LLAMA_CPP_COMMIT}" --depth 1
git -C "${LLAMA_CPP_SRC}" checkout --detach "${LLAMA_CPP_COMMIT}"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]] || \
   ! "${CUDA_HOME}/bin/nvcc" --version | grep -q 'release 12.6'; then
  echo "CUDA 12.6 nvcc is required at ${CUDA_HOME}/bin/nvcc" >&2
  exit 1
fi
CUBLAS_ROOT="$("${PURE_SPIN_V12_VENV}/bin/python" - <<'PY'
from pathlib import Path
import nvidia

print(Path(nvidia.__path__[0]) / "cublas")
PY
)"
for library in libcublas libcublasLt; do
  if [[ ! -e "${CUBLAS_ROOT}/lib/${library}.so" ]]; then
    ln -s "${CUBLAS_ROOT}/lib/${library}.so.12" \
      "${CUBLAS_ROOT}/lib/${library}.so"
  fi
done

# Do not inherit Windows PATH entries or a /mnt/c working directory. CMake's
# dependency discovery otherwise crosses the Plan 9 bridge and becomes
# needlessly slow even though the source and build trees are on ext4.
export PATH="${LLAMA_CPP_BUILD_VENV}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUBLAS_ROOT}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
cd "${LLAMA_CPP_SRC}"

cmake -S . -B "${LLAMA_CPP_BUILD}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DCUDAToolkit_ROOT="${CUDA_HOME}" \
  -DCMAKE_LIBRARY_PATH="${CUBLAS_ROOT}/lib" \
  -DCMAKE_CUDA_FLAGS="-I${CUBLAS_ROOT}/include" \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_CCACHE=OFF \
  -DLLAMA_CURL=OFF
cmake --build "${LLAMA_CPP_BUILD}" \
  --target llama-cli llama-bench llama-quantize --parallel "$(nproc)"

for command in llama-cli llama-bench llama-quantize; do
  ln -sfn "${LLAMA_CPP_BUILD}/bin/${command}" "${HOME}/.local/bin/${command}"
done

"${HOME}/.local/bin/llama-cli" --version
"${HOME}/.local/bin/llama-cli" --list-devices
printf 'GGUF root: %s\n' "${GGUF_ROOT}"
