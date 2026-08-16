# Pure Rotor SSM versus Mamba-2 benchmark

## Status

Last reconciled: **2026-08-16T16:05:27+02:00**.

Implemented runner. A one-step CUDA smoke artifact is available at
[`artifacts/pure_rotor_vs_mamba2_smoke_seed0.json`](artifacts/pure_rotor_vs_mamba2_smoke_seed0.json);
it establishes only that all three paths run on the same cached byte stream.
This document and that artifact must not be read as a quality or systems result.

The runner compares the maintained `pure_rotor_ssm.GASSMLanguageModel` against
Hugging Face Transformers' `Mamba2ForCausalLM`, plus a parameter-identical
Pure Rotor identity-transport ablation. All candidates use raw WikiText-2 UTF-8
bytes, the same random windows per seed, fixed validation windows, AdamW, and
the same number of sampled byte-tokens.

The default architecture is deliberately parameter-near:

| Candidate | Parameters |
|---|---:|
| Pure Rotor / identity-transport | 141,796 |
| Mamba-2 | 145,016 |

This is a 2.2% raw parameter gap. It is not a matched recurrent-state-size
claim: Mamba-2 also carries convolution and selective SSM state. Nor is the
identity ablation an effective-capacity match: fixing `max_rotor_angle=0`
disables its 6,060 rotor-controller parameters. The report records both raw
counts and this disabled count. Therefore that ablation tests whether the
rotor path is useful in the same architecture; it does not isolate transport
from every allocation-of-capacity alternative.

Both configured candidates tie their input and output byte embeddings; the
runner verifies and records that fact rather than assuming it from a library
default.

## Run

From `SSM-Models`:

```powershell
python -m pytest test_pure_rotor_vs_mamba2_benchmark.py -q
python benchmark_pure_rotor_vs_mamba2.py --steps 20 --validation-batches 1 --seeds 0 --offline --output experiments/artifacts/pure_rotor_vs_mamba2_smoke.json
python benchmark_pure_rotor_vs_mamba2.py --steps 1000 --seeds 0,1,2 --checkpoint-directory checkpoints/pure_rotor_vs_mamba2 --output experiments/artifacts/pure_rotor_vs_mamba2_1000.json
```

`--rotor-scan-mode schur_parallel` is a separate execution-path comparison.
Do not compare its wall time with a fused Mamba kernel: the runner uses the
Transformers Mamba-2 path and records whether `mamba_ssm` is importable, while
making no fused-kernel claim.

Use `--offline` for a reproducible cached-data run; it fails immediately if
WikiText-2 is absent instead of hanging on an unavailable download. It reads
the cached Arrow shards directly; use `--dataset-cache-directory` when the
Hugging Face datasets cache is not in its default location.

## Acceptance criteria

- Validate three untouched seeds before describing a mean quality difference.
- Report validation bits per byte, throughput, peak CUDA memory, parameter
  count, data hashes, and the Mamba execution backend.
- Require Pure Rotor to beat the identity-transport ablation before
  attributing any result to rotor transport, while retaining the explicit
  effective-capacity caveat above.
- A short smoke run establishes only that the benchmark executes.
