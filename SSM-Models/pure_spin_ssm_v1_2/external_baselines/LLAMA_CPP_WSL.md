# llama.cpp on the constrained local rig

This is an inference/reference tier, not a way to make 7B or 52B systems into
parameter-matched v1.2 training baselines. It exists to test quantized and
CPU/GPU-offloaded official checkpoints on the only local machine: an RTX 2070
SUPER 8 GB, i7-9700K, and 24 GB DDR4.

## Storage and build contract

- source: `/home/local/src/llama.cpp` (WSL ext4, physically on C:);
- build: `/home/local/src/llama.cpp/build-cu126` (WSL ext4, physically on C:);
- command links: `/home/local/.local/bin/llama-{cli,bench,quantize}`;
- build-tool venv: `/home/local/.venvs/llama-cpp-build`;
- GGUF weights and download cache: `/mnt/e/AI_Culture_Mind_Large/models/gguf`.

Compilation does not use `/mnt/c` or `/mnt/e`. WSL's mounted-drive bridge is
appropriate for large persistent model files, but is much slower for compiler
metadata traffic. The installer also replaces the inherited mixed Windows/WSL
`PATH` with a Linux-only path during configuration and compilation.

The build is pinned to llama.cpp commit
`86722900390abf479fd9719eda12c299a2b25bbf`, CUDA 12.6, and compute capability
75. The build reuses the cuBLAS headers and libraries already owned by the
Torch cu126 environment rather than installing a duplicate SDK.
`GGML_NATIVE=OFF` avoids an irreproducible host-specific CPU build. NCCL is
not required for the single-GPU machine. CURL/HTTPS server support is disabled
because local file inference is the intended scope.

```bash
cd /mnt/c/Users/HaydenLocal/Programming/AI_Culture_Mind/SSM-Models/pure_spin_ssm_v1_2
bash external_baselines/install_llama_cpp_wsl.sh
llama-cli --list-devices
```

## Honest measurement protocol

Start with a quantization that leaves enough of the 8 GB VRAM free for runtime
state; do not assume that a file smaller than 8 GB fits entirely on the GPU.
Measure multiple `--n-gpu-layers` values, including zero and the largest stable
partial offload. Record prompt-processing throughput, generation throughput,
peak host RAM, peak VRAM, context length, quantization, and exact GGUF hash.

Architecture recognition, successful conversion, and CUDA device visibility
do **not** prove that a Mamba/Jamba checkpoint has an optimized GPU path. Each
candidate earns a result row only after a real GGUF loads, produces a checked
output, and completes a repeated `llama-bench` run without paging or OOM.

No large weight was downloaded during environment setup. The pinned metadata
audit in this directory is intentionally weight-free.
