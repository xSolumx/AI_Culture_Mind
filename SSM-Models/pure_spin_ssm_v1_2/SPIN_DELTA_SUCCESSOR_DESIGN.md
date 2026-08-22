# Spin-Delta successor design

**Status:** semantic compiler and v1.2 model path implemented on 2026-08-22;
correctness gates pass, but this is not yet a promoted natural-data model.

## Reassessed objective

The target is no longer to make every Spin extension pass. It is to close the
validated quality gap to fused Mamba-2 while preserving the properties that
v1.2 actually earns: exact Spin(8) transport, a tiny streaming cache, bounded
dynamics, and a raw-CUDA path suitable for the RTX 2070 SUPER.

The evidence separates two roles that v1.2 currently conflates:

1. **transport heads** provide independently learned Spin trajectories;
2. **memory slots** provide content-addressed overwrite and retrieval.

Maintained v1.2 has two independent transport heads but only one implicit slot
per head. Its update is leaky accumulation, not a delta memory. v1.3 has real
erase/write/query memory, but dense exceptional transport did not help generic
text. The successor should combine the demonstrated pieces, not the failed
ones.

## State factorization

Let `g=1,...,G` index independent transport heads, `a=1,...,m` index memory
slots, and `r` index `8v`, `8+`, and `8-`. The state is

```text
H[g,a,r] in V_r.
```

The first candidate fixes `G=2`, preserving the two independently controlled
Spin actions that beat shared-action compression, and `m=2`, introducing one
new addressable slot per head. Its recurrent Spin state is 96 scalars per
layer, 384 across four layers. Including the existing 1,536-scalar convolution
history gives 1,920 streaming scalars—about 41.9 times smaller than the 80,384
scalars recorded for the matched Mamba-2 reference.

## Transition

Each head has its own Spin element, shared only across that head's memory
slots:

```text
transported[g,a,r] = rho_r(g_t[g]) H[g,a,r].
```

Within a head, a rank-one safe delta update acts on the slot axis:

```text
L[g] = s[g] (I - beta[g] e[g] e[g]^T),
H'[g,:,r] = L[g] transported[g,:,r] + w[g] z[g,r]^T,
y[g,r] = q[g]^T H'[g,:,r].
```

Here `e` and `w` are normalized two-slot erase/write keys, `q` is a bounded
affine read probe, `0 <= beta <= 1`, and `0 < s < 1`. The implemented first
gate uses independent erase and write directions. This changes no stability
claim: `I-beta e e^T` still has spectral norm at most one, while the write is
an affine drive and does not enter the homogeneous operator.

This is a deliberate correction to the paper design, not scope drift. Exact
zero erase, global boundedness, smooth parameterization, and a nonzero
first-order erase gradient cannot all hold simultaneously: a differentiable
map into `[0,1]` attaining zero is at a local minimum and therefore has zero
derivative. A hard-clamp subgradient could hide that incompatibility but would
introduce a dead negative half-line. The implemented smooth staged embedding
instead starts with `e=(0,1)`, safely erasing only the empty auxiliary slot,
and `beta=1/2`. Write direction, erase direction, and query are first-order
live; erase strength becomes live after the erase direction moves.

The readout concatenates the two queried 24-dimensional triality values and
uses the maintained direction interface initially. Triality invariants remain
an isolated optional control because their previous mean effect was too small.

## Exact compiler structure

For each transport head and triality sector, the local linear map is

```text
A[g,r] = L[g] tensor rho_r(g_t[g]).
```

This is an ordinary affine map on `R^m tensor V_r`; chronological composition
is exactly associative after flattening. Crucially, the two independent Spin
actions live in separate transport heads. Slot mixing never assumes that the
two head actions are shared, avoiding the quality loss established by the
shared-action compression gate.

The existing repo already supplies most machinery:

- v1.3 rank-one delta transition and query semantics;
- v1.2 independent Spin controllers and subgroup ladder;
- the coupled-isotypic `2x2` CUDA forward/backward within one shared-action
  slot bank;
- the isotypic forward schedule and guarded reconstruction lessons;
- exact affine composition and full-gradient semantic oracles.

The required new lowering adds an outer transport-head grid dimension rather
than inventing another recurrence kernel from scratch: one warp owns one
`(batch, head)` two-slot bank across the three triality sectors.

## Exact baseline embedding

The candidate must reduce algebraically to maintained v1.2 at initialization:

- slot zero contains the maintained learned initial state;
- slot one starts at zero;
- the initial bounded probe is `(1,1)`, which reads slot zero because slot one
  is exactly empty;
- the initial erase direction is `(0,1)` with strength `1/2`;
- maintained drive writes with `w=(1,0)` into slot zero;
- every new controller is zero initialized under an RNG fork;
- both independent Spin controllers and every common model tensor are copied
  exactly.

Every common parameter must be bitwise identical. The two recurrences are the
same map over the reals. The generic two-slot float32 contraction may differ
from the scalar maintained path by a final-rounding ulp because it sums an
extra exact-zero term; the gate records and bounds that numerical residual
instead of mislabeling it as an architectural difference.

## Acceptance sequence

1. Semantic recurrent equation and affine composition associativity.
2. Algebraically exact baseline embedding; immediate nonzero gradients into
   write address, erase direction, and query; and a perturbation test proving
   the staged erase-strength path becomes live.
3. Recurrent/parallel output, state, and full-gradient parity in float64.
4. Boundedness and long-sequence finite-state falsifiers.
5. Raw-CUDA output and full-model gradient parity at factor counts 3/6/15/28.
6. A frozen three-seed, 300-step Shakespeare quality gate against unchanged
   maintained v1.2.
7. Only after a quality pass: order-balanced complete-step timing and memory.
8. Only after both gates: longer-horizon Shakespeare plus matched fused
   Mamba-2 confirmation.

No Tensor-Core or Mamba-superiority claim is attached to the design. The first
question is narrower and decisive: does explicit addressable overwrite recover
materially more than the hundredth-bit effects of transport, readout, and
retention modifications?

## Implemented evidence

The implementation is split so that the mathematical oracle is independent of
the language-model wrapper:

- `spin_delta_scan.py` defines the affine transition, chronological
  composition, sequential recurrence, parallel Hillis-Steele prefix scan,
  safe rank-one left map, drive routing, and query contraction;
- `model.py` exposes `recurrence="spin_delta"` with two slots and the scan
  modes `delta_recurrent` and `delta_parallel`;
- `test_spin_delta_scan.py` checks associativity, output and full-gradient
  scan parity in float64, contraction, routing, exact one-step embedding, and
  the staged gradient structure;
- `test_model.py` checks paired parameter identity, near-bitwise float32 model
  embedding, full-model output/gradient parity, causal masking behavior, and a
  384-token finite-state falsifier.

The raw-CUDA lowering is now closed. The compiler flattens the independent
`(batch, transport_head)` grid onto the existing two-copy coupled-isotypic
kernel. Each launch unit therefore sees one head's two slots and one shared
head action; no action or gradient is shared between heads. This reuses the
audited singular-safe forward/backward instead of cloning CUDA code.

Semantic output, state, and full-gradient parity pass at factor counts
3/6/15/28, and full language-model gradient parity passes for Spin(3), Spin(4),
Spin(6), and Spin(8) layers. The complete WSL/cu126 suite now passes 90 tests.
The frozen Shakespeare gate was specified in
`SPIN_DELTA_PREREGISTRATION.md`. The corrected cohort has now closed
negatively: 1/3 wins,
`-0.02520726` mean bpb, and `-0.05938958` worst improvement. Spin-Delta is not
promoted and no speed gate is authorized. See `SPIN_DELTA_RESULTS.md`. The
compiler remains supported research infrastructure, separated from the failed
model claim.

On the pinned WSL Python 3.10 / Torch 2.10.0+cu126 environment, a deliberately
non-evidentiary two-update
Tiny Shakespeare smoke (`B=1`, `L=32`, `d_model=16`, one Spin(3) layer) also
completed forward, backward, optimizer update, validation, and JSON provenance
on the RTX 2070 SUPER. Its loss and throughput are commissioning diagnostics,
not a quality result or a comparison.
