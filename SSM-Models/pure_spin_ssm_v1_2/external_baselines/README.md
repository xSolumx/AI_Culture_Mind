# External baseline programme

This directory turns the requested Falcon-Mamba, Mamba-3, GKA, GDN, Mamba-2,
and Jamba comparisons into a fail-closed source and measurement contract.
`manifest.py` pins every official model revision and implementation source;
`audit.py` verifies the pins and records the local hardware without downloading
weights. Large snapshots and metadata live under
`/mnt/e/AI_Culture_Mind_Large/pure_spin_ssm_v1_2`, never in Git or on C:.

There are two valid comparison tiers:

1. **Matched Shakespeare training:** Pure Spin v1.2, Mamba-2, Mamba-3 SISO,
   Mamba-3 MIMO, a small GKA layer, and a small GDN layer are initialized from
   scratch and trained on identical byte batches. Parameter count, optimizer,
   precision, steps, validation windows, and hardware must match.
2. **Native checkpoint inference:** official pretrained checkpoints keep their
   own tokenizers. Compare tokens/s, memory, latency, and a shared downstream
   evaluation. Their pretrained losses must not be placed beside the tiny
   from-scratch Shakespeare loss.

The RTX 2070 SUPER has 8 GB and compute capability 7.5. The official Mamba-3
187M checkpoints are storage-feasible, but MIMO needs an isolated TileLang
environment. Falcon-Mamba-7B, the 8B GKA/GDN checkpoints, and original Jamba
do not fit in native BF16. Jamba2-3B is close enough to require a measured
quantized/offload feasibility test before it earns a throughput row.

Run the source audit after `wsl_env.sh`:

```bash
source wsl_env.sh
python external_baselines/audit.py --live \
  --output artifacts/external_baseline_audit.json
```

The next implementation gate is a separate Mamba-3/GKA/GDN environment. It
must not upgrade Torch, Triton, or Mamba inside the frozen v1.2 Mamba-2 venv.

Quantized/offloaded checkpoint experiments use the pinned CUDA 7.5 llama.cpp
build described in [`LLAMA_CPP_WSL.md`](LLAMA_CPP_WSL.md). Its source, build,
and binaries stay on WSL ext4; only GGUF weights and download caches use E:.
That tier is for feasible local inference evidence, not matched training.
