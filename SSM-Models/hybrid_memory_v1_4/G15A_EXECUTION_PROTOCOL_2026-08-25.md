# G15A operational execution protocol

**Frozen:** 2026-08-25, after implementation-integrity review and before any
G15A runner output, smoke metric, calibration result, or quality outcome was
produced or inspected

**Purpose:** operationalize the already frozen G15A mechanism/observability
gate without weakening the original preregistration or its two prospective
amendments.

This document fills the execution variables that were absent from
[`G15_SPIN_DIRAC_PREREGISTRATION.md`](G15_SPIN_DIRAC_PREREGISTRATION.md). It
does not replace the original promotion margins, integrity requirements, or
nonclaims.

## Mandatory primary arms

| Arm | Transport | Second sector |
|---|---|---|
| `I` | identity | identity copy |
| `I+C` | identity | fixed Clifford map |
| `C` | fixed `SO(2)^4` | fixed Clifford map |
| `S` | full factorized Spin(8) | fixed Clifford map |

All arms use one head, the `equivariant_scalar` edit law, bounded values,
strict retention below one, the same 64-scalar matrix state, and exactly the
same trainable parameter tensors. Inactive coordinates are masked, not
removed. `S-broken` and `S+identity-read` remain conditional controls and are
not substituted into this four-arm gate.

## Two deliberately separated workloads

### A. Oracle-controlled supplied-coordinate tracking

This workload isolates the transport mechanism. Exact carrier controls hold
address, value, retention, erase, and write fixed. Only two equivariant scalar
calibration gains are trained, for 100 updates with AdamW, learning rate
`0.05`, and no weight decay. The gains cannot mix carrier coordinates or
encode an action class.

Every episode has one initial rank-one write, supplied bounded action
coordinates, filler, and one final scored read. The fixed vector key is
`e_0`. Classes are:

- eight off-torus Spin(7)-stabilizer planes `(1,2)`, `(1,3)`, `(1,4)`,
  `(1,5)`, `(2,4)`, `(2,5)`, `(3,6)`, and `(3,7)`;
- identity; and
- the central `2*pi` element in plane `(0,1)`, composed from 26 increments of
  `2*pi/26 < 0.25`.

Each off-torus action is composed from four increments of `0.20`. The action
planes fix `e_0`, preventing a changed final query from leaking the class. The
input token sequence is identical across all ten classes. The fixed `T^4`
arm sees all eight off-torus classes as identity but retains the central plane.

The ten deterministic coordinate classes are the complete finite evaluation
support. Accuracy is macro accuracy over those ten classes; the report also
records off-torus and central-pair accuracy. The configured count of 80 is
eight balanced repetitions for percentage accounting, not 80 independent
coordinate schedules.

The oracle ladder separately checks one-hot read, repeated-key overwrite,
collision/old-value removal, and an orthogonal query in float64. All residuals
must be below `1e-10`.

For every arm and seed, predictions after the learned scalar calibration are
replayed under a fixed proper Spin(8) inner conjugation. Carrier controls and
every vector/positive/negative action are transformed consistently. Maximum
float64 prediction residual must be at most `1e-9`. For `I`, the identity-copy
sector transforms as another positive-spinor copy rather than as `8_s-`.

This is oracle mechanism evidence. It does not establish that a token model
learns coordinates, addresses, or edit gates.

### B. Learned final-only no-symmetry retrieval

This workload tests whether `S` harms an ordinary task where transport should
be irrelevant. A one-block `HybridMemoryLM` learns eight delayed value classes
from token inputs. The value token occurs only at position zero; filler follows;
the final query marker is the only scored position. Supplied coordinates are
identically zero in every arm and never enter the residual stream.

The common model has:

- `model_dim=32`, one SpinDirac head, one SpinDirac layer;
- no attention, local convolution, dropout, or hidden coordinate injection;
- tied query/key projection and tied token/output embeddings;
- the frozen bounded scalar edit law;
- exactly matched parameter tensors and FP32 state;
- `HarmonicMuonAdamW`, learning rate `0.003`, weight decay `0.01`;
- 300 updates, batch size 16, sequence length 64;
- final-query cross entropy plus commissioning coefficient `0.25` for
  write-event, address alignment, erase suppression, and retention; and
- gradient clipping at norm 1.

Evaluation uses 80 balanced examples at lengths 64, 256, and 1,024, processed
in microbatches of eight. These are repetitions over a finite eight-class
token support, not a natural-language distribution.

## Seeds, pairing, and machine

The quality seeds are exactly `2131`, `2137`, and `2141`. Model initialization
and every task seed are deterministic hashes of the model seed and semantic
event name. Each arm receives the same batches in the same order.

Quality mode is FP32 and must start from a clean committed worktree. The local
evidentiary execution is restricted to the RTX 2070 SUPER with compute
capability exactly 7.5. It records commit, source/protocol hashes, schedule
hashes, parameters, state bytes, optimizer partition, synchronized step time,
peak CUDA allocation, checkpoints, and environment.

Smoke mode uses seed 17, four updates, four calibration steps, batch size four,
and ten evaluation examples. Smoke output is non-evidentiary and cannot alter
the quality protocol.

## Binding aggregation and promotion rule

For each seed separately:

1. `S` macro symmetry accuracy across lengths 64, 256, and 1,024 must exceed
   each of `I`, `I+C`, and `C` by at least `0.02`.
2. `S` macro learned no-symmetry accuracy across the same lengths must be no
   more than `0.01` below the best `I`, `I+C`, or `C` comparator.
3. Every arm's trained-calibrator inner-conjugation residual must be at most
   `1e-9`.
4. The float64 oracle semantic ladder must pass.

All conditions must hold in all three seeds. There is no mean-over-seeds
rescue. Failure stops Spin promotion. Passing authorizes, but does not itself
pass, the conditional `S-broken` and `S+identity-read` diagnostics.

## Retry and claim policy

- Non-finite loss, gradient, output, or artifact values fail the run.
- Out-of-memory or interruption may resume only from a recorded run boundary;
  hyperparameters and data seeds may not change.
- Unsupported native kernels and compatibility fallbacks are ineligible.
- External GDN2 and Mamba-3 comparisons remain outside G15A until the frozen
  parameter and synchronized-compute matchers exist.
- A pass is still only an oracle-controlled symmetry mechanism plus a finite
  learned no-symmetry retrieval result. It is not G15B associative-memory,
  G15C natural-text/long-recall, G15D scaling, or a fused-kernel result.
