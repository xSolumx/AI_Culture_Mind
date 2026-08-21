# Pure Spin SSM v1.2

This folder is the isolated implementation and evidence boundary for Pure Spin
SSM v1.2. It does not inherit empirical claims from older rotor, Spin(8), or
synthetic-memory experiments.

## Architecture

Each causal block contains:

1. RMS normalization and a gated input projection;
2. causal depthwise convolution for local token mixing;
3. the maintained bounded Spin(8) recurrence with shared `8v/8+/8-` triality
   action and a fixed-size recurrent cache;
4. the fused controller-to-28-factor-to-recurrence CUDA lowering when available;
5. gated residual readout and a SwiGLU channel mixer.

The recurrence is not Mamba-2 under different notation. Its state transition is
a selective contractive affine Spin(8) action. Local convolution and SwiGLU are
included because the old pure recurrence lacked the local/channel mixing needed
for a credible language-model comparison.

## Required comparison

`benchmark.py` trains both candidates on the identical immutable UTF-8 byte
stream from `Salesforce/wikitext`, configuration `wikitext-2-raw-v1`. It records
stream hashes, parameter counts, bits per byte, throughput, peak CUDA memory,
software versions, and the exact GPU. The default comparison uses a 128-wide
Pure Spin model and a 144-wide Mamba-2 model; it constructs both candidates up
front and refuses the run unless trainable parameter counts differ by at most
five percent.

The baseline imports `mamba_ssm.Mamba2` with `use_mem_eff_path=True`. The run
fails closed when the official fused SSD kernel is unavailable; Transformers'
reference implementation is never relabeled as fused.

Run under WSL/Linux because the official Mamba package requires Linux and an
NVIDIA CUDA environment:

```bash
python3 -m venv ~/.venvs/pure-spin-v12
source ~/.venvs/pure-spin-v12/bin/activate
cd /mnt/c/Users/HaydenLocal/Programming/AI_Culture_Mind/SSM-Models
cd pure_spin_ssm_v1_2
bash install_wsl.sh
cd ..
PYTHONPATH=. python -m pytest -q pure_spin_ssm_v1_2/test_model.py
cd pure_spin_ssm_v1_2
PYTHONPATH=.. python benchmark.py --steps 300 --batch-size 8 --sequence-length 256
```

The validated local tuple is Python 3.10, Torch 2.10.0+cu130, Triton 3.6.0,
official `mamba_ssm` 2.3.2.post1, and official `causal-conv1d` 1.7.0.
`install_wsl.sh` installs and probes that exact tuple. Its wheel URLs encode the
CUDA, Torch, ABI, Python, OS, and architecture match explicitly; `--no-deps`
prevents backend packages not exercised by this benchmark from replacing the
pinned runtime. The install probe then imports the exact fused SSD symbol.
Set `HF_HOME` to a writable task-specific cache when the default WSL cache is
unavailable.

## Claim ledger

- Algebraic Spin(8) action, bounded recurrence, and scan identities are inherited
  only from their maintained exact/unit-tested modules.
- Shape, causality, gradient-finiteness, and fallback-refusal are unit tests.
- WikiText losses and throughput are empirical properties of a recorded run.
- No quality, speed, Tensor-Core, or scaling advantage is claimed before a
  complete artifact exists for both candidates on the same environment.
- Parameter matching is measured before training and the harness fails closed
  when the raw trainable counts differ by more than five percent.

## Raw CUDA comparison

`csrc/spin_scan_cuda.cu` is a raw CUDA forward kernel for the materialized
shared-action recurrence. One block owns one `(batch, channel, representation)`
state, keeps its eight coordinates in shared memory, and scans the complete
sequence in one launch. `benchmark_raw_cuda.py` compares it with the maintained
Triton scalar kernel and PyTorch oracle using synchronized median latency and
explicit numerical parity.

This kernel is intentionally not used for training yet: it has no backward and
refuses gradient-bearing inputs. It measures recurrence lowering, not
end-to-end Pure Spin versus Mamba-2 performance.

The first RTX 2070 SUPER comparison is reported in
[`RAW_CUDA_RESULTS.md`](RAW_CUDA_RESULTS.md): after current-stream integration
and CUDA-event timing, raw CUDA took 145.1 microseconds versus Triton's 175.5
microseconds in FP32, and 121.6 versus 183.2 microseconds in FP16, on the
recorded shape. It is a one-shape result, not a general speed claim, and the
scalar 8-by-8 kernel is not a Tensor-Core implementation.

See [`REUSE_ATLAS.md`](REUSE_ATLAS.md) for the repository-wide component audit
and the exact mechanism/claim boundary for every reused subsystem.

The completed three-seed natural-data result and its negative findings are in
[`NATURAL_DATA_RESULTS.md`](NATURAL_DATA_RESULTS.md). The current model learns
WikiText bytes but does not beat the official fused Mamba-2 control in quality,
training speed, or peak training memory. Its clearly measured architectural
advantage is much smaller streaming state; the incremental wrapper needed to
realize that advantage end to end remains open.
