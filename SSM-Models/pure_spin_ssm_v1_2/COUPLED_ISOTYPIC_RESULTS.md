# Coupled isotypic recurrence results

**Closed:** 2026-08-22

## Result

The exact compiler and its CUDA training backend are accepted as implemented
research machinery. Learned recurrent multiplicity transport is **not**
promoted into the maintained Pure Spin v1.2 architecture.

The candidate used the exact two-sided affine transition

\[
H_t=\operatorname{diag}(s_t)Q_tH_{t-1}R_t^T+D_t,
\]

with one shared Spin action `R_t` and a token-conditioned `Q_t in SO(2)` on the
two-copy multiplicity factor. The baseline fixed `Q_t=I` while retaining the
same shared-action recurrence. Both used the new `raw_cuda_coupled` backend.

| seed | shared identity bpb | shared orthogonal bpb | improvement |
|---:|---:|---:|---:|
| 149 | 2.684138 | **2.683578** | +0.000560 |
| 151 | 2.690222 | **2.676683** | +0.013538 |
| 157 | **2.681593** | 2.702099 | -0.020506 |
| mean | **2.685317** | 2.687453 | -0.002136 |

Positive improvement is identity minus orthogonal. The candidate won two of
three seeds and stayed within the allowed single-seed regression bound, but it
failed the frozen mean-improvement requirement of `+0.0100` bpb. Its observed
mean effect was unfavorable. The conditional established-v1.2 Stage B was
therefore not authorized and was not run.

## Commissioning exclusion and initialization audit

An earlier seed-109 commissioning artifact is retained but excluded. Although
the optional controller was explicitly zeroed in the block constructor, the
model-wide language initializer later randomized every `Linear` again.
Constructing the extra layer also shifted the RNG stream for later common
parameters. Unequal initial validation losses exposed both problems before the
valid gate was accepted.

The repaired implementation marks identity-start controllers for RNG-free zero
initialization and constructs optional layers without advancing the outer RNG.
A regression test now requires common state tensors and initial logits to be
bitwise equal. In all three valid artifacts, baseline and candidate initial
validation loss and step-1 training loss match exactly.

This audit also changes the interpretation of the earlier readout-only
`orthogonal_query` artifacts: they remain valid records of the executed models,
and their failure never justified a default change, but they were not the
claimed exactly paired identity-start ablation. They should not be cited as a
strict falsification without a fresh preregistered replay. The maintained
`multiplicity_router=none` default is unchanged.

## What is proved, measured, and still open

Proved by construction and replayed tests:

- chronological transition composition is associative;
- recurrent and logarithmic-depth prefix scans agree in outputs and gradients;
- the linear operator stays contractive under the existing retention bound;
- the CUDA lowering agrees with the semantic oracle in outputs and all public
  gradients for 3, 6, 15, and 28 factors;
- the complete WSL/CUDA v1.2 suite passes 46 tests.

Measured only on the frozen small Shakespeare campaign:

- learned recurrent `SO(2)` mixing did not improve mean validation quality;
- embedded sequential timers are diagnostic, not an order-balanced speed
  comparison.

Still open:

- fused chunk-level composition of the two-sided affine transition;
- higher multiplicity and structured non-orthogonal contractions, which need
  new gates rather than reinterpretation of this failed candidate.

The lower-parameter shared-action identity question was subsequently closed
negative: it lost all three frozen seeds by 0.02415 mean bpb, so its conditional
speed gate was not run. See `SHARED_ACTION_COMPRESSION_RESULTS.md`.

The machine-readable decision is
[`artifacts/coupled_isotypic_stage_a_summary.json`](artifacts/coupled_isotypic_stage_a_summary.json).
The exact algebra and frozen rules are in
[`COUPLED_ISOTYPIC_PREREGISTRATION.md`](COUPLED_ISOTYPIC_PREREGISTRATION.md).
