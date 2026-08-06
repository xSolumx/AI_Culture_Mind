# Corrected benchmark result

Run artifact: `results/benchmark_corrected_300.json`

Hardware: NVIDIA GeForce RTX 2070 SUPER, Windows, CUDA tensor execution.
Dataset: WikiText-2 raw UTF-8 bytes, vocabulary 256. Each run used 300 updates,
batch 16, sequence length 256, AdamW, and the same sampled batches per seed.
The default static trace was used for the fixed benchmark shape.

## Integrity

The harness seeds Python, NumPy, PyTorch, and CUDA before constructing each
model through a factory. This repairs the earlier defect where models were
constructed before seeding. The result is therefore suitable for this short
two-seed comparison; older benchmark JSON files are not multi-seed evidence.

The Mamba control is Transformers' pure-PyTorch fallback because the fused
`mamba_ssm` extension is unavailable on Windows. The Spinor path is tensor-only
CUDA code with an exact differentiable Hillis--Steele affine scan, not a fused
custom CUDA extension.

## Results

| seed | model | parameters | final bits/byte | final loss | tokens/s | peak MiB |
|---:|---|---:|---:|---:|---:|---:|
| 0 | Spinor isotypic delta | 674,322 | 3.0537 | 2.1167 | 21,301 | 1,678 |
| 0 | Mamba-2 reference | 688,220 | 2.2856 | 1.5843 | 16,599 | 3,148 |
| 1 | Spinor isotypic delta | 674,322 | 3.0654 | 2.1248 | 21,935 | 1,683 |
| 1 | Mamba-2 reference | 688,220 | 2.2631 | 1.5687 | 16,836 | 3,153 |

Mean throughput is approximately 1.29× higher for Spinor; mean bits/byte is
approximately 0.785 worse. This is a short mechanism benchmark, not a claim
of language-model superiority or production-kernel speed.

## Interpretation

The decoder bottleneck and compact four-coefficient rotor representation close
the prior compute mismatch enough for the tensor implementation to beat the
local pure-PyTorch Mamba reference in throughput. The result does not establish
that the geometry helps language modeling: the Mamba control learns a materially
lower loss under the same short budget. A Linux fused Mamba kernel and a
work-efficient fused rotor scan would be a different comparison.

The next fair gates are five or more corrected seeds, longer training, a real
tokenizer/corpus, and synthetic A5/Q8/copy tasks that test the mechanisms the
local archive actually validated.
