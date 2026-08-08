# Spin8 selective rotor SSM benchmark

This is an isolated, reproducible benchmark built from the strongest findings
in the local Spin(8) research archive:

- a stable input-selective affine recurrence;
- norm-preserving multivector rotor transport;
- the complete eight-kernel Spin(3) isotypic commutant (not only grade mixing);
- invariant gates with independent bounded erase/write controls;
- an equivariant 160-scalar decoder bottleneck so the dominant vocabulary
  projection has the same width as the Mamba-2 control;
- exact scan/recurrent equivalence in the underlying rotor-affine update;
- separate erase/decay and write gates inspired by gated delta-rule memory;
- an untied decoder with explicit logit-temperature control (v2 uses unit
  scaling after the quality audit);
- an optional identity-initialized depthwise causal mixer with exact streaming
  history, retained as an ablation until it clears the quality gate;
- an optional zero-initialized direct decoder ensemble, retained only for
  ablation because its short quality screen did not beat v2;
- an exact differentiable logarithmic-depth associative scan for the affine
  recurrence.

The model is compared with Hugging Face Transformers' `Mamba2ForCausalLM` and a
local pure-PyTorch Mamba-3 reference, using the same byte vocabulary, WikiText-2
bytes, nearly identical parameter counts, optimizer, sequence length, steps,
and seeds. The benchmark does not claim that geometric structure improves
language modeling until a matched experiment demonstrates it.

## Run

From this directory:

```powershell
python -m py_compile spinor_delta_ssm.py benchmark.py
python benchmark.py --steps 300 --seeds 0,1 --output results/benchmark.json
```

Use `--steps 20 --seeds 0` for a smoke run. The benchmark uses byte-level
WikiText-2 so the comparator does not receive a tokenizer or vocabulary-size
advantage. The Mamba-2 implementation falls back to Transformers' reference
PyTorch path because the fused `mamba_ssm` extension is not available on the
current Windows environment. The rotor model uses the exact differentiable
associative scan, not a Python token loop. It is CUDA-safe tensor code, not a
fused custom C++/CUDA extension; this Windows result must not be conflated with
the Linux fused `mamba_ssm` kernel.

The default trace is static at the requested batch/sequence shape and is used
only for throughput measurement. Pass `--no-jit` when testing arbitrary shapes
or when you need the untraced eager execution path.

The completed three-way short run is documented in
[`RESULTS_THREE_WAY_100.md`](RESULTS_THREE_WAY_100.md), with raw data in
[`results/benchmark_three_way_100.json`](results/benchmark_three_way_100.json).
The Mamba-3 reference uses an exact affine scan for its trapezoidal recurrence;
it is intentionally labeled separately from the official fused implementation.

The decoder-scaling quality iteration is documented in
[`RESULTS_QUALITY_V2_300.md`](RESULTS_QUALITY_V2_300.md), with the validated
artifact in [`results/quality_v2_300.json`](results/quality_v2_300.json). Use
`quality_benchmark.py` when you want to measure the local model without
starting competing baselines. Validate a saved report before interpreting its
aggregate with `python validate_quality.py results/quality_v2_300.json`; this
rejects duplicate seeds, malformed hashes, metric/formula mismatches, and
configuration drift.

## Vision benchmark

`vision_benchmark.py` compares the local rotor model with Transformers'
`Mamba2Model` and the pure-PyTorch Mamba-3 reference on identical continuous
4x4 image-patch sequences. It supports CIFAR-100, CIFAR-10, and padded MNIST;
all models use the same class-token pooling, AdamW schedule, deterministic
batch order, and test protocol. The Mamba-2 vocabulary embedding is removed
because continuous patch inputs bypass it, so parameter counts and gradients
describe the actual classifier.

Example screening run:

```powershell
python vision_benchmark.py --datasets cifar100,cifar10,mnist --steps 100 --seeds 0,1 --output results/vision_100.json
```

This is a short matched screen, not a converged image-classification claim;
use more steps and independent seeds for publication-quality comparisons.

The scan is an exact Hillis--Steele reference with `O(L log L)` work. The local
overhaul identifies a work-efficient fused scan as the production optimization
still missing; no speed claim here assumes that kernel exists.

## Scientific boundary

This is a language-model benchmark, not a proof of Spin(8) triality and not a
claim of state-of-the-art performance. The local archive's exact geometric
theorems motivate the recurrence and stability constraints; they do not imply
language-quality gains. Results should be reported with parameter counts,
training tokens, wall time, peak memory, validation loss, and bits per byte.

## External references

- Mamba-2 / structured state-space duality: Dao & Gu, arXiv:2405.21060.
- Gated DeltaNet / delta-rule memory: Yang et al., arXiv:2412.06464.
- Mamba-3 / complex state updates and MIMO: Lahoti et al., arXiv:2603.15569.
- Open-source reference implementation: `state-spaces/mamba`.

## Local research boundary

The archive establishes exact algebraic identities and controlled synthetic
mechanism results, not a language-model advantage. The Spin(8) paired audit
found no stable raw positive-spin advantage over generic SO(8), while the
Q8/A5 studies support exact/retracted noncommutative actions, quality-gated
decoding, and long-horizon controls. Those findings motivate this architecture
and its ablations; they do not transfer automatically to WikiText.
