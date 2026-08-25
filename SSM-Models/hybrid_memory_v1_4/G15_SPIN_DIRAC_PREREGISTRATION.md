# G15 Spin-Dirac content-addressed memory preregistration

**Frozen:** 2026-08-25, before any G15 training outcome is inspected
**Status at freeze:** algebraic/unit contracts pass; no trained Spin-Dirac
checkpoint or model-quality result exists

## Question

Once addressing and independent erase/write are held fixed, does shared
Spin(8) transport with a fixed Clifford/triality readout improve a task that
actually contains the corresponding symmetry? Does it retain that benefit
without harming generic associative recall or ordinary natural text?

This is deliberately not the old `structured_spin8` claim. The candidate in
[`spin_dirac_memory.py`](spin_dirac_memory.py) stores one content-addressed
matrix `8_v -> 8_s+` per head, applies a two-sided Spin transport, performs a
decoupled fast-weight edit, and maps the read into `8_s-` with the fixed
Clifford tensor. The state width therefore buys associations; Spin geometry is
restricted to transport and an exact intertwiner.

## Frozen ablations

The primary within-family comparison keeps state shape, trainable parameter
tensors, shell, optimizer, tokenizer, data order, and target tokens identical:

| Arm | Transport | Second read sector |
|---|---|---|
| I | identity | identity copy |
| C | four commuting SO(2) planes | Clifford map |
| S | full factorized Spin(8) | Clifford map |

The identity and commuting arms retain the 28-coordinate controller tensor;
inactive coordinates are masked rather than removed. This keeps parameter
counts exact while exposing unused-parameter counts separately. A fourth
identity-transport/Clifford-read arm may be run to isolate the readout, but it
cannot replace I, C, or S and is not part of the primary promotion decision.

The external memory-law controls are semantic GDN2 with `key_dim=value_dim=8`
per head and official source-current Mamba-3 SISO. They enter a quality table
only after a separate shell matcher brings total trainable parameters within
1% and measured training-step FLOPs or synchronized step time within 5%.
Unsupported kernels and compatibility fallbacks are ineligible.

## Integrity gates

Before training:

1. recurrent and two-sided parallel scans agree in float64;
2. arbitrary chunk replay and token stepping agree with full recurrence;
3. masked tokens freeze both matrix and convolution caches and emit zero mixer
   update;
4. the fixed Clifford read is equivariant under the shared vector/positive/
   negative Spin(8) action;
5. gradients reach address, value, decay, erase, write, coordinate, residual,
   and output paths in the full LM;
6. the full LM preserves neutral identity transport and the configured gate
   biases after shell initialization;
7. every optimizer parameter appears exactly once, with erase/write/decay/
   coordinate controllers in the declared control group;
8. artifacts record commit, source hashes, seeds, target hashes, exact state
   bytes, trainable and used parameters, peak CUDA allocation, and synchronized
   timing.

Failure of any integrity gate stops the cohort and is not a model result.

## Learning ladder

### G15A: mechanism and observability

- oracle one-hot write/read, overwrite, collision, and orthogonal-query tests;
- final-only delayed retrieval at lengths 64/256/1,024;
- direct checks that the coordinate controller changes the delayed read, not
  only the same-token output;
- central-sign-sensitive and supplied-coordinate Spin(8) tracking, plus a
  matched task with no Spin symmetry.

At least three fresh seeds are required. S must beat I and C on the
symmetry-aligned task in all three seeds by at least 2 percentage points, while
not losing more than 1 point on the no-symmetry control. Otherwise the Spin
transport claim fails.

### G15B: generic associative memory

- MQAR, last-write-wins overwrite, repeated-key collision, selective copy, and
  exact-distance needles;
- variable numbers of writes, delays, query positions, and final-only reads;
- no train/eval routing switch;
- identical commissioning-loss coefficients and labels across arms.

S must not trail the best I/C arm by more than 1 point on any three-seed mean.
This gate can establish compatibility, not Spin necessity.

### G15C: ordinary natural text

- at least three fresh model seeds;
- identical lossless tokenizer, training bytes, target tokens, optimizer
  budget, context curriculum `256 -> 512 -> 1,024 -> 2,048 -> 4,096`, and
  attention schedule;
- ordinary next-token loss reported separately from any commissioned memory
  objective;
- held-out BPRB plus counterfactual span/fact recall at distances beyond the
  actual attention window.

Promotion requires S to improve held-out BPRB by at least 0.01 versus both I
and C without a recall regression, or to improve the preregistered long-range
recall measure by at least 0.02 nats while staying within 0.01 BPRB. One metric
cannot compensate for failure of the other.

### G15D: parameter/compute scaling

Repeat the surviving arms at matched state/parameter budgets corresponding to
contexts 256, 512, 1,024, 2,048, and 4,096. Report quality versus parameters,
state bytes, training tokens, synchronized time, and peak CUDA memory. A trend
from unmatched widths or unsupported upstream kernels is inadmissible.

## Explicit nonclaims

Passing algebraic contracts does not establish learnability. Passing a
symmetry-aligned task does not establish generic language advantage. Triality
does not add raw state capacity. The Clifford read is an exact equivariant map,
not a Dirac differential operator. Until G15C and G15D pass, `spin_dirac`
remains an experimental candidate and the default model plan does not change.
