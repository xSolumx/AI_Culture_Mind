# Q8 spinor center-fidelity experiment: prospective contract

Date fixed: 2026-08-03, after the algebraic center-fidelity certificate and
before any learned write-free Q8 center experiment with the new
`pure_quaternion_spinor` family.

The project had already identified the Q8 sandwich-kernel limitation in its
earlier review. The prospective novelty is the learned pure-spinor fix, not the
known diagnosis. The original selective `quaternion_even` ladder is reported
as prior evidence, not renamed: it did not beat GA sandwich reliably at long
length and is confounded by writes and decay.

## Question

Does a left spinor recurrence learn central-sign-sensitive composition that a
rotor sandwich recurrence cannot represent?

This is narrower and stronger than “which noncommutative family gets better
accuracy?” The two actions have different kernels:

```text
spinor:    s -> q s                 faithful on Q8
sandwich:  v -> q v q^-1            factors through Q8/{+-1}=V4
```

## Frozen families

All learned families use eight real state values per channel, no writes,
decay, residual transition, or token-conditioned decoder shortcut, a nonzero
initial orbit state, endpoint loss, and exact recurrent streaming.

1. `pure_quaternion_spinor` at `2*pi`: one shared unit quaternion
   left-multiplies two quaternionic spinors; three token action parameters per
   channel.
2. `pure_ga_rotor` at `2*pi`: the primary equal-chart Cl(3) sandwich control;
   the same three token action parameters per channel.
3. `pure_ga_rotor` at `2.2`: the established A5 optimization cap, retained as a
   legacy convergence control. It cannot reach the exact pi rotation needed
   even for the V4 quotient and is not the primary capacity comparison.
4. `pure_householder4_shared`: capable generic control using four learned
   reflections in O(4), shared across the two four-real state blocks. This is
   the parameter-richer apples-to-apples alternative to quaternion left action.
5. `pure_householder`: the older two-reflection O(8) plane-rotation family,
   retained as an explicitly underparameterized control. It cannot realize a
   pure-quaternion left action: for such an action `rank(I-A)=4`, which is its
   minimum reflection length.
6. exact regular permutation action: nonlearned discrete ceiling.

The two Householder charts each use 16 raw action coordinates per
token/channel (`4x4` versus `2x8`), so their capacity contrast is not a raw
parameter-count contrast. Both are parameter-richer than the three-coordinate
quaternion chart; the report will state those counts rather than claiming
parameter matching where none exists.

The primary spinor and sandwich comparison uses the same `2*pi` chart. The
usual principal chart cannot place both signs of a pure-imaginary quaternion
in the learned Spin(3) token family; `2.2` cannot even realize the quotient's
pi rotations exactly. Chart width is therefore controlled explicitly rather
than discovered as a confound after training.

## Data and falsifier

Use the four-token inverse-paired alphabet `{i,-i,j,-j}`. The exact audit found
that this walk is bipartite: every fixed odd length reaches only
`{+i,-i,+j,-j}`, while every fixed even length reaches only
`{+1,-1,+k,-k}`. The augmented-chain SLEM is therefore exactly one, endpoint
entropy at a fixed length approaches only two bits, and a powers-of-two gate
would silently test only half the group.

Freeze the 2,000-step, 512,000-label curriculum as:

1. 250 batches at L1;
2. 250 batches at L2;
3. 250 batches alternating L3/L4 (125 each);
4. 250 batches alternating L7/L8 (125 each);
5. 1,000 batches alternating L15/L16 (500 each).

This is a depth curriculum while preserving both parity cosets after the first
two stages. The exact audit, including token-contrast rather than only
common-mode action gradients, is frozen in `q8_endpoint_credit_audit.json`.

Evaluation is balanced over pairs of words `(w, w*i*i)`. Their endpoints differ
by central `-1`, but their sandwich actions are identical. Report both ordinary
eight-class accuracy and central-pair accuracy. Every evaluation tier contains
matched odd/even base lengths; no even-only mean may hide the bipartite
coverage defect or the paired falsifier.

## Gates

- The exact sandwich oracle must score exactly at the representational ceiling
  on balanced central pairs, establishing that optimizer failure is not the
  explanation.
- The exact spinor oracle must score 100% with positive state margin.
- A learned family passes only if 10/10 seeds reach >=99% central-pair accuracy
  at every base length L15--L32, every matched odd/even base pair
  `(16k-1,16k)` for `k=3..16`, and the long base pairs L4095/L4096 and
  L16383/L16384, with both `(w,w*i*i)` members evaluated and full/chunk/token
  state parity preserved.
- The sandwich family is expected to fail by theorem; any apparent pass is an
  implementation leak and invalidates the harness until explained.
- The capable four-reflection Householder row distinguishes “spinor fidelity”
  from “any faithful generic orthogonal action.”
  Spinor-specific advantage requires either more reliable optimization or a
  better parameter/accuracy frontier, not merely representational sufficiency.

## Claim tiers fixed before training

1. **Kernel theorem—already proved:** sandwich collapses central sign; spinor
   left action does not. No learned accuracy is needed for this statement.
2. **Capacity result:** exact or learned spinor exceeds the balanced
   central-pair ceiling that both sandwich charts cannot exceed. This says the
   state action can express the missing distinction.
3. **Optimization result:** spinor reaches the full gate reliably across 10/10
   SGD seeds without a new convergence pathology. This may fail even though
   tier 2 is true and must be reported separately.

No Q8 outcome alone licenses a Spin(8), triality, language, or general sequence
model claim.

## Scope

A pass would establish that state representation and group-center kernel
matter on a controlled finite task. It would not establish a language benefit,
Spin(8) triality, or superiority over generic orthogonal transitions. It is a
necessary adversarial bridge before those larger claims.
