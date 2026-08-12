# Spin(8), Spin(9), and hierarchical memory results

- **Date:** 2026-08-10
- **Programme:** Triality memory and Intertwiner SchurScans
- **Status:** exact algebra plus frozen 10-seed quality and three-process CUDA results
- **Protocols:**
  [`HIERARCHICAL_MATCHED_RETRIEVAL_PREREGISTRATION.md`](HIERARCHICAL_MATCHED_RETRIEVAL_PREREGISTRATION.md)
  and
  [`COMOVING_FUSED_DELTA_PREREGISTRATION.md`](COMOVING_FUSED_DELTA_PREREGISTRATION.md)

> **Task B completion addendum.** A later prospective paired-action replication
> on untouched seeds `20`--`29` closes the independent-action direct/delta row
> `10/10`; see
> [`TASK_B_PAIRED_ACTION_REPLICATION_RESULTS.md`](TASK_B_PAIRED_ACTION_REPLICATION_RESULTS.md).
> The preceding strict replay failed because its historical metric-only source
> did not retain the learned parameters, and that failure remains recorded.

> **Large-slot implementation addendum.** The first next experiment below was
> subsequently completed under new frozen protocols. The
> [`LARGE_SLOT_SEMANTIC_HIERARCHY_RESULTS.md`](LARGE_SLOT_SEMANTIC_HIERARCHY_RESULTS.md)
> campaign uses 64 overlapping keys, shared-three-view and independent-router
> controls, actual gathered state, and a one-kernel fused inference recurrence.
> This document remains authoritative for the earlier eight-slot, Spin(9), and
> transported-FLA cohorts.

## Verdict

The best supported memory architecture is no longer “triality instead of
DeltaRule.” It is:

1. a discrete or sparse hierarchical router;
2. an exact addressed memory behind that router;
3. an optional equivariant Spin(8) or Spin(9) representation prior when the
   task genuinely contains cross-view actions;
4. a stable co-moving compiler into official fused DeltaRule kernels when
   supplied value transport is present.

Full Spin(8) triality still has **no matched-state overwrite-capacity
advantage**. Spin(9) supplies a new exact `9 -> 16` Clifford bind/unbind and a
natural nine-dimensional coarse index, but it also has **no free same-width
capacity advantage**. The positive empirical result is routing: a two-slot
block selector substantially reduces soft interference, and hard routing makes
direct slots and DeltaRule exactly coincide on this synthetic world.

The positive systems result is stronger. A stable inverse-frame compiler now
turns the full noncommuting transported recurrence into an official FLA
DeltaRule call. At length 4,096 it is `5.20x` faster forward and `5.30x` faster
forward+backward than direct slots on the local RTX 2070 SUPER, while using
`153.98 MiB` rather than `575.51 MiB` of incremental training allocation. It
does this with 128 logical recurrent scalars, not the 64-scalar matched budget.

## What full Spin(8) can and cannot buy

The vector and two chiral spinor representations `8v`, `8+`, and `8-` share a
unique triality coupling. For a supplied unit key, the binding operator is
orthogonal, so one pair binds and unbinds exactly. Slotwise, however, this is a
change of gauge: an `H x 8` triality-bound memory has the same `8H` scalar
state, isolation rank, and retrieval capacity as `H` direct eight-vector
slots. Exact linear isolation still requires at most `K <= H` independent
multiplicity codes. Raw superposition in one eight-dimensional representation
does not evade that bound.

The defensible full-Spin(8) benefits are different:

- one action parameterization can move all three views consistently;
- one router can be shared across vector, positive-spinor, and
  negative-spinor memories;
- held-out cross-view action completion can be sample-efficient when the
  symmetry is correct for the task;
- if three views would otherwise select disjoint physical blocks, a shared
  selected-block set has an ideal payload-read ceiling of `3x` less than three
  independent selections. This bandwidth ceiling has not yet been measured by
  a gathered kernel.

These are prior, equivariance, and systems-sharing advantages—not extra
overwrite slots.

## Spin(9) boundary

The maintained real Spin(9) Clifford system has nine symmetric involutions
`P_i` on a 16-dimensional spin module with

\[
P_iP_j+P_jP_i=2\delta_{ij}I.
\]

For `a in R^9`, define `D(a)=sum_i a_i P_i`. Then

\[
D(a)^2=\lVert a\rVert^2I.
\]

Thus a unit nine-vector gives the exact reversible binding
`psi -> D(a) psi`. The new executable diagnostic verifies:

| Gate | Maximum error |
|---|---:|
| Single-pair bind/unbind | `2.22e-16` |
| Spin(9) equivariance | `6.66e-16` |
| Hopf norm identity | `2.22e-16` |
| Dynamic bound/direct gauge | `9.99e-16` |
| Dynamic unbound/direct state | `8.33e-16` |

The wrong-key term is not small: `D(a)D(b)psi` preserves `||psi||` for unit
keys. Raw superposition therefore retains full-norm crosstalk. Slotwise Spin(9)
binding is again an orthogonal gauge of direct 16-vector slots; the diagnostic
compares 64 scalars with 64 scalars and finds no capacity difference.

The quadratic map

\[
h_i(\psi)=\psi^T P_i\psi,
\qquad \lVert h(\psi)\rVert=\lVert\psi\rVert^2,
\]

is more promising as a **coarse route** than as memory. It maps a 16-spinor to
nine equivariant scores but discards the seven-dimensional Hopf fiber, so it
cannot replace fine spinor storage. Representation-theoretically, restricting
the Spin(9) spinor to Spin(8) couples the two chiral eight-dimensional modules;
this motivates a cross-chiral sample-efficiency test against a generic `O(16)`
control. It does not prove such an advantage. For background on the Spin(9)
spin representation and octonionic Hopf geometry, see
[Parton and Piccinni, *The Role of Spin(9) in Octonionic Geometry*](https://arxiv.org/abs/1810.06288).

## NSA-inspired same-router experiment

The transferable idea from
[Native Sparse Attention](https://arxiv.org/abs/2502.11089) is hierarchical
addressing: use a compressed/coarse branch to choose contiguous fine blocks,
retain a local window, and reuse the compression attention scores rather than
train a separate nondifferentiable selector. NSA retains token KV history and
performs query-dependent attention; it is not a fixed-state associative scan.

The local ablation applies that routing principle to the existing 384-parameter
categorical encoder. Direct slots and DeltaRule receive the same route, and all
rows retain 64 recurrent scalars:

- `dense_soft`: access 8 slots, ideal float32 payload `256 B`;
- `block_top1`: access one fixed two-slot block, `64 B`;
- `hard_top1`: access one slot, `32 B`.

There is no auxiliary selector, selector loss, extra parameter, or extra
training example.

### Frozen primary result

Ten independent seed processes produced 240 cells spanning overwrite depth 16,
stream lengths 256/1,024/4,096, perturbations 0/0.1/0.2, and transport off/on.
Every implementation gate passed.

| Variant | Mean cosine across cells | Worst cell mean cosine |
|---|---:|---:|
| Direct dense-soft | `0.792656` | `0.492006` |
| Delta dense-soft | `0.894518` | `0.529438` |
| Direct block-top1 | `0.929427` | `0.660465` |
| Delta block-top1 | **`0.974789`** | **`0.793887`** |
| Direct hard-top1 | `1.000000` | `1.000000` |
| Delta hard-top1 | `1.000000` | `1.000000` |

Relative to dense soft routing, block selection improves mean cosine by
`+0.136770` for direct slots and `+0.080271` for DeltaRule. The long-stream
cold cohort is the clearest mechanism result:

| Variant | Cold mean cosine |
|---|---:|
| Direct dense-soft | `0.463083` |
| Delta dense-soft | `0.732468` |
| Direct block-top1 | `0.813502` |
| Delta block-top1 | `0.934741` |
| Both hard-top1 rows | `1.000000` |

Hard direct and hard Delta predictions agree within `2.46e-14`; oracle rows
are exact; transformed exact-route triality/direct replay agrees within
`1.33e-14`; and chunk/recurrent Delta states agree within `2.46e-14`.

The exact hard result is deliberately narrow. The synthetic nuisance is
orthogonal to the semantic-center span, the learned classifier separates all
held-out aliases, and corruption through `0.20` does not flip its argmax. This
does not establish real-data robustness.

### Adversarial selector frontier

The prospectively frozen post-primary screen adds route perturbations
`0.30/0.50/0.75/1.00`, ten seeds, overwrite and length-256 streams, and both
transport conditions. It maps the expected discontinuous failure:

| Perturbation | Direct dense | Delta dense | Direct block | Delta block | Hard direct/delta |
|---:|---:|---:|---:|---:|---:|
| `0.30` | `0.740948` | `0.853046` | `0.944659` | `0.975917` | `1.000000` |
| `0.50` | `0.655194` | `0.697640` | `0.897183` | `0.937562` | `1.000000` |
| `0.75` | `0.552452` | `0.537227` | `0.763425` | `0.773752` | `0.989667` |
| `1.00` | `0.460371` | `0.410200` | `0.488967` | `0.477463` | `0.555369` |

Hard routing is best until decision flips become common, then fails sharply.
The two-slot block route degrades more gradually but cannot rescue fully
corrupted coarse selection. The screen is synthetic selector corruption, not a
natural-noise estimate.

## Stable co-moving compiler

The native transported DeltaRule is

\[
S_t=A_tS_{t-1}R_t^T+k_tv_t^T,
\qquad A_t=I-\beta_tk_tk_t^T.
\]

For invertible `R_t`, let `P_t=R_t...R_1` and
`Sbar_t=S_tP_t^{-T}`. Then

\[
\bar S_t=A_t\bar S_{t-1}+k_t(P_t^{-1}v_t)^T,
\qquad y_t=P_t(q_t^T\bar S_t).
\]

This is an ordinary transport-free DeltaRule plus an action-prefix scan and two
frame changes. It is exact for general invertible actions; the orthogonal
formula is the special case `P_t^{-1}=P_t^T`.

The initial naive fp16 transpose implementation failed honestly at length
4,096 with `0.11816` relative error. Rounded fp16 actions are not exactly
orthogonal. The corrected implementation accumulates `P_t` in float32, solves
`P_t x_t=v_t` in float32, casts only `x_t` into FLA's required fp16 kernel, and
applies the read frame in float32. Against a float32 native reference at length
4,096, local inverse-frame error is `4.17e-7`, official FLA error is
`1.56e-3`, and native fp16 error is `1.65e-3`.

The float64 exact gate independently gives `1.22e-15` forward relative error
and at most `3.05e-15` relative gradient error over values, keys, queries,
gates, and constrained action coordinates. A separate ambient-coordinate test
also verifies the general invertible formula.

### Frozen CUDA result

The official environment is WSL2, RTX 2070 SUPER, PyTorch `2.13.0+cu130`,
Triton `3.7.1`, `flash-linear-attention 0.5.2`, and `fla-core 0.5.2`. Three
disjoint tuning processes froze implementations; three new processes each ran
100 forward and 100 forward+backward samples. Medians below are medians of the
three process medians and include prefix scan, solve, both frame changes,
DeltaRule, loss, and backward.

| Length | Row | State | Forward ms | Forward+backward ms | Training allocation MiB |
|---:|---|---:|---:|---:|---:|
| 256 | Direct slots | 64 | `2.746` | `7.777` | `35.45` |
|  | Native local Delta | 64 | `7.432` | `24.644` | `16.01` |
|  | Co-moving FLA chunk | 128 | `2.584` | `7.868` | `19.63` |
|  | Co-moving FLA recurrent | 128 | **`1.950`** | **`6.445`** | `19.63` |
| 1,024 | Direct slots | 64 | `7.422` | `24.790` | `143.38` |
|  | Native local Delta | 64 | `9.162` | `39.038` | `63.75` |
|  | Co-moving FLA chunk | 128 | `3.214` | `10.603` | `38.48` |
|  | Co-moving FLA recurrent | 128 | **`2.707`** | **`9.294`** | `38.48` |
| 4,096 | Direct slots | 64 | `29.229` | `98.320` | `575.51` |
|  | Native local Delta | 64 | `18.993` | `122.294` | `257.72` |
|  | Co-moving local Delta | 128 | `19.950` | `80.310` | `271.70` |
|  | Co-moving FLA chunk | 128 | `5.639` | **`18.543`** | `153.98` |
|  | Co-moving FLA recurrent | 128 | **`5.620`** | `19.278` | `153.98` |

At length 4,096, the best fused row is `5.20x` faster forward and `5.30x`
faster forward+backward than direct slots. Against the native local Delta
reference it is `3.38x` and `6.60x` faster. Maximum forward relative error over
all compiled rows is `0.002048`; all gradients are present and finite.

This is a genuine full-transport systems result, but not a state-matched win.
The cumulative `8 x 8` action makes the logical state 128 scalars. The result
also does not establish triality-specific capacity: the compiler applies to
any invertible value transport.

## Recommended architecture and next falsifiers

The strongest next implementation is a hierarchical hot/cold memory:

1. a recent exact local window for immediate overwrite;
2. a coarse shared router, optionally constrained by Spin(8) triality or the
   Spin(9) Hopf scores when cross-view symmetry is real;
3. top-block gathering into fine direct/Delta memories;
4. a stable co-moving fused DeltaRule global summary;
5. a learned gate across local, selected fine, and global outputs.

The next experiments should be, in order:

1. **Completed in the later large-slot campaign:** replace the orthogonal-
   nuisance alias world with overlapping semantic noise, increase slot/block
   count, and measure a real gathered kernel rather than ideal bytes. The
   frozen follow-up fixes the physical block layout; learning the layout itself
   remains open;
2. compare one shared three-view router with three independent routers on
   held-out Spin(8) action words; the independent-action direct/delta replay is
   now prospectively complete;
3. test Spin(9) cross-chiral routing/sample efficiency against parameter- and
   state-matched generic `O(16)` and direct controls;
4. only then scale to language or vision sequences against current fused
   Gated DeltaNet/DeltaProduct and sparse-attention baselines.

The unrestricted Dirac--Gram global inequality is not a prerequisite. It is a
separate sensing/design theorem. It becomes relevant only if a memory claim
uses that global information optimum as a certified objective; it is not needed
for the gauge theorem, Clifford binding, hierarchical routing result, or fused
compiler.

## Artifacts

- `artifacts/hierarchical_matched_retrieval_seed0.json` through `seed9.json`:
  independent primary reliability processes;
- `artifacts/hierarchical_matched_retrieval_seeds0_9.json`: validated primary
  aggregate;
- `artifacts/hierarchical_matched_retrieval_adversarial_seed0.json` through
  `seed9.json`: independent stronger-corruption processes;
- `artifacts/hierarchical_matched_retrieval_adversarial_seeds0_9.json`:
  validated adversarial aggregate;
- `artifacts/spin9_clifford_memory_boundary_20260810.json`: Spin(9) exact
  boundary diagnostic;
- `artifacts/comoving_fla_tuning_run1_20260810.json` through `run3`: disjoint
  implementation-selection processes;
- `artifacts/comoving_fla_frozen_selection_rtx2070s_20260810.json`: frozen
  median-of-process-medians selection;
- `artifacts/comoving_fla_frozen_run1_20260810.json` through `run3`: independent
  measurement processes;
- `artifacts/comoving_fla_frozen_aggregate_20260810.json`: validated CUDA
  aggregate with all raw samples and memory measurements.
