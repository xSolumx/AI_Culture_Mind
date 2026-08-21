# Reuse atlas and design decisions

This inventory records which existing repository components are reused by Pure
Spin SSM v1.2 and which claims are deliberately not inherited.

| Source | Reused mechanism | Boundary |
|---|---|---|
| `pure_spin8_ssm.torch_backend` | bounded affine Spin(8) recurrence, full `8v/8+/8-` cache, triality readout | no inherited language-quality claim |
| `pure_spin8_ssm.factorized_scan` | fused learned controller, 28 ordered plane factors, recurrent forward/backward | CUDA FP32 training path; not automatically Tensor-Core execution |
| `pure_spin8_ssm.continuous_scan` | PyTorch oracle, Triton scalar recurrence, FP16 Tensor-Core inference path | materialized-action comparison only |
| `pure_rotor_ssm.torch_backend` | stable exponential-chart lessons, contractive drive construction, causal test contracts | Cl(3,0) implementation is not copied into the Spin(8) model |
| `benchmark_pure_rotor_vs_mamba2.py` | immutable byte-stream construction, deterministic batches, checkpoint/report discipline | Transformers Mamba fallback is rejected |
| `Spin8-SSM-Benchmark/vision_benchmark.py` | continuous-input parameter-accounting lessons | old partial benchmark logs are not v1.2 results |
| isotypic-to-silicon compiler | hardware-aware backend selection and honest schedule metadata | exact representation decomposition does not imply model quality |
| gathered-memory benchmarks | synchronized timing, hashes, matched-control reporting | separate memory mechanism, not included in v1.2 yet |

New v1.2 components are the causal depthwise local mixer, gated Spin readout,
SwiGLU channel mixer, strict official-Mamba adapter, raw CUDA recurrence, and
self-contained natural-data harness.

The official baseline follows the maintained
[`state-spaces/mamba`](https://github.com/state-spaces/mamba) implementation and
its `Mamba2(..., use_mem_eff_path=True)` fused SSD route. The architecture is
described in [Transformers are SSMs](https://arxiv.org/abs/2405.21060). Natural
data comes from the versioned
[`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext)
dataset rather than generated sequence tasks.
