# Three-way benchmark: local Spinor model, Mamba-2, and Mamba-3

Run artifact: [`results/benchmark_three_way_100.json`](results/benchmark_three_way_100.json)

This is a matched short training run, not a claim of model-scale superiority. All
three models were trained for 100 updates on the same sampled byte-level
WikiText-2 batches for each seed, with the model constructed only after the
Python/NumPy/PyTorch/CUDA seed was set. Each run saw 409,600 training tokens.
The parameter counts are within 3.3% of one another.

| Model | Parameters | Validation bits/byte (mean ± SD) | Throughput tokens/s (mean ± SD) | Peak CUDA MiB (mean ± SD) |
| --- | ---: | ---: | ---: | ---: |
| Spinor isotypic delta (local) | 674,322 | 3.396 ± 0.021 | 20,318 ± 269 | 2,285 ± 3 |
| Mamba-2 (`transformers`) | 688,220 | **2.551 ± 0.011** | 15,674 ± 195 | 3,147 ± 3 |
| Mamba-3 reference | 696,376 | 3.171 ± 0.040 | **22,938 ± 190** | **463 ± 3** |

Lower bits/byte is better; higher throughput is better. On this short run the
local model is 29.6% faster than the Mamba-2 implementation, but 11.4% slower
than the tensor-only Mamba-3 reference. Mamba-2 has the best validation loss;
the local model trails it by 0.845 bits/byte and Mamba-3 by 0.224 bits/byte.
The local model uses 27.4% less peak memory than Mamba-2.

## Implementation boundary

The local model uses the complete Cl(3) isotypic channel mixer and a tensor CUDA
associative rotor-affine scan. Mamba-3 was implemented locally as a readable
pure-PyTorch reference with exponential-trapezoidal state updates, rotary state
projections, and rank-2 MIMO. Its sequential recurrence was replaced by an
exact affine Hillis--Steele scan and checked against the token-by-token form to
about `4e-7` absolute error in FP32. This is not the official fused Mamba-3
kernel. Mamba-2 likewise reports the Transformers fallback path because the
optional fused CUDA extensions are unavailable in this Windows environment.

The official Mamba repository documents Mamba-3 as a source-tree implementation
with Linux and CUDA build requirements, and exposes both Mamba-2 and Mamba-3
modules ([official repository](https://github.com/state-spaces/mamba)). The
Mamba-3 paper describes the exponential-trapezoidal, complex/rotary, and MIMO
changes ([Lahoti et al., 2026](https://arxiv.org/abs/2603.15569)); the Mamba-2
baseline is the SSD-based selective SSM ([Dao and Gu, 2024](https://arxiv.org/abs/2405.21060)).

The benchmark should therefore be read as a reproducible local systems result:
it compares matched parameter budgets and training work, while making the
backend asymmetry explicit. Longer training, multiple data regimes, and the
official fused Mamba-3 kernels are still required for a scientific architecture
claim.
