# G15B-D Phase-1 results

**Frozen protocol:**
[`G15BD_RESIDUAL_DELTA_PROTOCOL_2026-08-26.md`](G15BD_RESIDUAL_DELTA_PROTOCOL_2026-08-26.md)

**Passed Phase-0 qualification:**
[`G15BD_PHASE0_QUALIFICATION_RESULTS.md`](G15BD_PHASE0_QUALIFICATION_RESULTS.md)

**Exact smoke artifact:**
[`artifacts/g15bd_phase1_smoke_sm75_2026-08-26.json`](artifacts/g15bd_phase1_smoke_sm75_2026-08-26.json)

**SHA-256:**
`b96e8e0cf936af6c828c5ec643484c677a556c6e9f687bc453a70ab653e6464e`

**Execution commit:** `644330c61bf894ed8ff379872d97d5c591c8555c`

**Status:** clean exact-SM75 smoke completed; frozen quality cohort remains
unexecuted

## Bottom line

G15B-D completed its clean exact-SM75 Phase-1 execution smoke. Source binding,
the sealed Phase-0 artifact, matched initialization, optimizer assignment,
finite nonzero gradients, learned-checkpoint reconstruction, data separation,
and every declared numerical boundary pass.

The smoke intentionally records formal `passed=false` and
`eligible_for_promotion=false`. Its decision is:

> smoke execution completed; run the frozen quality cohort

This is execution and profiling evidence only. Four updates and 3,840 training
tokens per arm cannot establish learning quality, overwrite ability, causal use,
natural-text transfer, efficiency, scaling, geometry value, or model promotion.
The semantic residual-delta `D` arm is matched to product `P` in parameters and
recurrent state, but is not compute- or temporary-memory-matched.

## Exact execution

The evidentiary artifact ran from a clean tree under WSL2 on an NVIDIA GeForce
RTX 2070 SUPER with exact compute capability `(7,5)`, Python 3.11.16, PyTorch
2.9.0+cu128, and CUDA 12.8.

The smoke used seed 29 and one batch-two update at each of lengths 128, 256,
512, and 1,024. Each arm therefore received:

- 4 optimizer updates;
- 3,840 training tokens;
- 16 capped standard evaluation decisions;
- 16 capped intervention decisions.

These caps qualify code paths and artifact construction; they are not the
frozen quality cohort.

## Matched contracts

Both arms have:

- 67,033 total and active parameters;
- 4,864 FP32 recurrent-state bytes per sequence;
- identical initialized parameter SHA-256 values;
- the same declared optimizer partition with all 25 trainable tensors assigned;
- finite nonzero gradients;
- exact learned-checkpoint reconstruction;
- zero train/evaluation batch-hash intersection.

The preflight also confirms unchanged sealed Phase-0 core sources. Effective
erase/write residuals are exactly zero for both arms, and the `D` coupled-edit
check passes.

## Numerical execution boundaries

Every declared learned numerical boundary passes and all compared predictions
are exact.

| Arm | Chunk logit | Chunk state | Masked-step logit | Masked-step state |
|---|---:|---:|---:|---:|
| `P` | `2.384e-7` | `1.192e-7` | `3.576e-7` | `1.490e-7` |
| `D` | `2.384e-7` | `1.192e-7` | `3.576e-7` | `8.941e-8` |

The frozen bounds were about `3.086e-3` for logits and `1.953125e-3` for
state, so these smoke residuals remain comfortably inside the declared
execution tolerances. This validates execution parity, not task quality.

## Profiling observations

| Arm | Mean synchronized step | Evaluation wall time | Peak CUDA allocation |
|---|---:|---:|---:|
| `P` | `0.1022907` s | `2.0444` s | `448,071,680` bytes |
| `D` | `0.2244913` s | `50.6322` s | `4,705,855,488` bytes |

These are raw smoke measurements on unequal semantic implementations. `D` is
not compute- or temporary-memory-matched to `P`; the smoke did not preregister
an efficiency gate, use enough repetitions, or isolate kernel costs. The table
therefore identifies a serious implementation-cost diagnostic for the quality
run, not an architecture-level speed or memory verdict.

## Decision boundary

The smoke authorizes execution of the already frozen clean-SM75 quality cohort.
It does not pass the quality cohort and it does not make either arm eligible for
promotion. Until the full cohort is sealed, record no G15B-D learning, causal-
use, ordinary-text, robustness, scaling, efficiency, or geometry conclusion.
