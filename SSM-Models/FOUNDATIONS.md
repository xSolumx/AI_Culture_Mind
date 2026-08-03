# Selective rotor SSM: mathematical contract

This note defines the maintained model in `ga_ssm.py` and
`rotor_ssm_torch.py`. It separates properties that follow from the equations
from hypotheses that still require empirical validation.

## State transition

For channel `c`, the state at token `t` is a multivector
`h[t,c]` in the Euclidean Clifford algebra `Cl(3, 0)`. The layer computes

```text
g_t       = invariant_features(x_t)
Delta_t   = Delta_min + softplus(W_delta g_t + b_delta)
lambda_c  = lambda_min + softplus(rho_c)
d_t,c     = exp(-Delta_t,c lambda_c)
B_t,c     = equivariant_bivector(x_t,c) tanh(W_R g_t + b_R)
q_t,c     = exp(-bounded(B_t,c) / 2)
u_t,c     = sqrt(1 - d_t,c^2) GradeLinear(x_t)_c
h_t,c     = d_t,c q_t,c h_(t-1,c) reverse(q_t,c) + u_t,c
```

`q` is a unit even multivector (a rotor), so conjugation by `q` rotates a
multivector without changing its coefficient norm. `Delta_min > 0` and
`lambda_min > 0` imply the uniform bound

```text
0 < d_t,c <= exp(-Delta_min lambda_min) < 1.
```

Consequently, with `d_max = exp(-Delta_min lambda_min)`,

```text
||h_t|| <= d_max ||h_(t-1)|| + ||u_t||.
```

For bounded drives this gives a bounded state by the geometric-series bound
`sup ||h|| <= ||h_0|| + sup ||u|| / (1 - d_max)`. This is a real stability
guarantee, although a bound close to one can still produce a very long and
numerically demanding memory.

The decay controller and rotor controller are initialized to zero. Decay
rates are chosen so zero-control channels have log-spaced half-lives from 4 to
2,048 tokens. Rotors start exactly at identity, but the bivector exponential
uses its correct small-angle limit, so the rotor controller has a nonzero
gradient at initialization.

## Why parallel training and recurrent streaming are the same model

Represent a token transition by the triple `T = (d, q, u)` acting as

```text
T(h) = d q h reverse(q) + u.
```

Applying `T_a` and then `T_b` gives the closed composition

```text
T_b compose T_a = (
    d_b d_a,
    q_b q_a,
    u_b + d_b q_b u_a reverse(q_b)
).
```

This operation is associative because it is ordinary function composition
and rotor multiplication is associative. JAX therefore computes every prefix
with `lax.associative_scan` during training. Recurrent inference uses the same
equation with one fixed-size state per layer. Tests compare full parallel,
arbitrarily chunked, and token-by-token execution at both state and logit
level.

For a model with `L` layers and `C` multivector channels, streaming cache size
is exactly `L * C * 8` scalars per sequence, independent of context length.
The current implementation still performs ordinary vocabulary decoding, so
generation cost is constant in past context length but not constant in
vocabulary size.

## Spin(3) equivariance

For a fixed frame rotor `s`, transform every multivector as
`x' = s x reverse(s)`. The maintained block commutes with this action:

- `GradeLinear` preserves grades and shares scalar channel weights across all
  coordinates within a grade;
- control networks see only scalar/pseudoscalar coefficients and vector/
  bivector norms, which are invariant under proper 3D rotations;
- the predicted bivector and its exponential transform by conjugation;
- RMS normalization, residual addition, and invariant gating commute with the
  same action.

Induction through the recurrence therefore gives
`h'_t = s h_t reverse(s)`. The test suite verifies this numerically for the
full SSM block, not only for isolated algebra helpers. This is Spin(3)
equivariance under proper Euclidean rotations; it is not a claim of reflection,
Lorentz, or arbitrary Clifford-group equivariance.

### Complete isotypic mixing, not merely grade mixing

The earlier implementation correctly preserved equivariance, but it did not
span every equivariant linear map. Under proper rotor conjugation,

```text
Cl(3,0) = 1 + 3 + 3 + 1.
```

Scalar and pseudoscalar are equivalent trivial representations; vector and
Hodge-dual bivector are equivalent standard three-dimensional
representations. `GradeLinear` independently mixed the four grades and omitted
all intertwiners between these equivalent copies. For `C` input and `D` output
channels it therefore contained `4CD` weights, while the complete Spin(3)
commutant contains `8CD`.

`GALib.Spin3IsotypicLinear` and `schur_scan.Spin3IsotypicLinear` now implement
the complete map. The frozen audit in
`experiments/SPIN3_ISOTYPIC_SCHUR_SCAN_RESULTS.md` numerically recovers the
eight-dimensional centralizer, proves the old rank-four restriction, and gives
an exact Hodge-copy witness that the old family cannot express at any depth.

The same decomposition suggests a representation-factored SSM. For real-type
irreps, transitions of the form

```text
direct_sum_lambda (M_t,lambda tensor rho_lambda(g_t))
```

remain closed under ordered composition, allowing complete multiplicity-space
mixing, group-valued phase, affine writes, associative training scans, and
fixed-state streaming simultaneously. `schur_scan.py` implements the Cl(3)
reference and verifies float64 parallel/recurrent parity below `9e-16`. For
general real representations, Schur's division algebra may be real, complex,
or quaternionic; the implemented Cl(3) sectors are real type.

## Controlled local-GPU evidence

`train_rotor_ssm_torch.py` compares the selective rotor model against an
identity-rotation ablation. Both variants have 22,968 parameters, the same
initialization seed, byte data, batches, optimizer, sequence length, and
training budget. Only `max_rotor_angle` changes. The final protocol used an
RTX 2070 SUPER, WikiText-2 UTF-8 bytes, 300 steps, context 64, batch 32, and
three seeds.

| Seed | Selective rotor loss | Identity loss | Identity minus rotor |
|---:|---:|---:|---:|
| 0 | 2.814146 | 2.807372 | -0.006774 |
| 1 | 2.728127 | 2.822437 | +0.094310 |
| 2 | 2.724192 | 2.839966 | +0.115774 |
| Mean | 2.755489 | 2.823258 | +0.067770 |

The mean advantage is 0.09777 bits/byte, or 2.40% of mean identity loss.
Rotors win two of three seeds; seed 0 is a narrow loss. Learned mean rotor
angles are nonzero and controller weight norms move away from zero, confirming
that the winning models actually use the transition. These short runs are a
promising mechanism-level result, not evidence of state-of-the-art language
modeling or a statistically established breakthrough.

The exact reports, including dataset SHA-256 hashes, loss samples, timings,
memory use, and transition diagnostics, are in `experiments/final_seed*_300.json`.

## Search, compile, retract

The finite-group experiments expose a distinction that the unconstrained SSM
equation alone does not enforce. Keeping every `R_t` inside `Spin(n)` preserves
norm and scan associativity, but it does not guarantee that the collection of
token transitions realizes one coherent algebra. Small mixed-relation errors
can accumulate indefinitely while every individual rotor remains unit length.

`representation_retraction.py` demonstrates a three-phase remedy:

1. **Search:** train independent token tangent parameters in the ambient spin
   group, retaining ordinary optimizer flexibility.
2. **Compile:** once the learned family approaches a stable finite action,
   recover exact irreducible candidates from the group's commuting regular
   actions and select the nearest candidate jointly.
3. **Retract:** after every later ambient optimizer step, project all token
   actions through one shared conjugation tangent. This preserves the complete
   relation table, rather than normalizing tokens independently.

For a known finite group, the compiler needs no character table or supplied
low-dimensional representation matrices. A generic symmetric right-regular
operator commutes with the left-regular group action, so each of its generic
dimension-`d` eigenspaces supplies an exact invariant `d`-dimensional action.
The learned family selects among these discrete candidates and fixes the
global basis.

The ten-seed A5 result in
`experiments/SELF_COMPILING_RETRACTION_RESULTS.md` validates this mechanism
through L4096. Its remaining supervision is substantial: the Cayley table and
token-to-element mapping are known. For language or a general selective SSM,
the analogue must discover approximate word equivalences or latent operator
relations before compilation. The portable principle is therefore not “use
A5”; it is **let optimization search broadly, then compile a discovered
algebraic subsystem and keep subsequent learning on its joint manifold**.

`latent_group_discovery.py` removes the explicit table input under exact prefix
supervision, but those densely observed labels are informationally equivalent
to the table once all edges are covered. It treats prefix classes as anonymous automaton states, infers one
permutation per token from adjacent labeled transitions, closes those
permutations into a finite group, and reconstructs multiplication in an
arbitrary base-state gauge. The regular-representation compiler then proceeds
unchanged. `experiments/LATENT_CAYLEY_RETRACTION_RESULTS.md` verifies this
table-blind route through L16384 in ten seeds. The remaining supervision gap is
now precise: ordinary sequences do not provide exact latent-state labels for
every prefix, so future compilation must infer equivalence from partial,
endpoint-only, or noisy evidence.

### Reverse-edge-cover identifiability

The partial-supervision extension isolates a small theorem behind the recovery
procedure. Let `T_a` be a deterministic permutation transition and let
`iota(a)` be a fixed-point-free inverse-token involution. It induces an
involution on directed edges,

```text
(s, a) <-> (T_a(s), iota(a)).
```

If the observed edge set intersects every two-edge orbit and the token
involution `iota` is uniquely identifiable, the complete transition family is
forced: every missing direction is the inverse of its observed partner. For
`|S|` states and `|A|` tokens this needs at least `|S||A|/2` directed edges
when `iota` is known. When it is unknown, a small number of bidirectional
calibration pairs can identify the shared involution; the tokens must be solved
jointly, not completed as independent permutations.

For the four-token A5 action, the lower cover has 120 edges. Enumerating the
three token perfect matchings and propagating each across the entire action
family recovers the exact action in 1,000/1,000 randomly oriented reverse covers at
**120/240 edges (exactly 50%)**. However, an exact 2-SAT audit constructs
adversarial orientations for which either wrong token matching is also
feasible. The learner safely refuses both as ambiguous. Exact half is therefore
a generic sampled-mask result, not universal identifiability.

One bidirectional calibration pair resolves the ambiguity in the worst case:
the true matching gains positive two-step identity support, both wrong
matchings have zero, and the second token pair is forced. Thus 121/240 is the
worst-case-safe threshold for this matching protocol. The conservatively
preregistered GPU cohort had already started at 122 edges before the 120/121
distinction was established.

Equal-budget uniform random masks recover 0/1,000 at 120, 121, and 122 edges.
Even when granted the true inverse pairing after the fact, the random 120-edge
masks leave 42--76 directed edges underdetermined because some reverse pairs
are completely hidden. Thus the result comes from global family consistency
plus coverage, not from “half the entries” alone. This is a sharp result for
**structured reversible missingness**, not a claim that arbitrary
half-observed Cayley actions are identifiable.

The same audit deliberately varies six base states and all 24 token closure
orders. After removing the base-state coset, all 144 recovered gauges are exact
post-hoc isomorphisms and all 24 closure-order compilers retain machine-precision
invariance and homomorphism. This tests gauge robustness directly rather than
mistaking a deterministic enumeration for ten independently chosen gauges.

### Endpoint mixing barrier

Endpoint labels are weaker than prefix-state traces, but two distinct problems
must be separated. An active endpoint membership-query compiler reconstructs
the anonymous regular action exactly from 1,148 labels; fixed-length neural
training can still fail because the generator random walk has nearly mixed.
For the A5 sampler, mean information between one token position and the final
state falls from 2 bits at L1 to 0.00128 bits at L16. Batch action gradients at
identity remain nonzero but become directionally incoherent. A frozen
short-to-long endpoint curriculum restores coherent early signal and passes
the complete dense/long gate in all ten seeds. This is a first-order
optimization barrier, not an impossibility theorem for endpoint learning.
The causal control is sharper than “show short examples”: an
`L8 -> L1 -> L16 -> L2 -> L4` permutation fits the isolated short blocks but
never forms a faithful representation. The supported mechanism is incremental
depth continuation through intermediate composition scales.
The compiler itself can now be derived from the learned endpoint manifold:
all ten seeds recover an A5-isomorphic multiplication table at step 850 using
16,384 endpoint examples already consumed by training and zero additional
queries. This still assumes exact anonymous endpoint classes and group order;
it is not unsupervised algebra discovery.

### Sandwich actions lose the spin-group center

Rotor conjugation factors through `Spin(n)/{+1,-1}` because `R` and `-R` act
identically in `R h reverse(R)`, even on a full multivector. A left spinor
action does not: central `-1` acts as `-I`. Q8 makes the distinction exact.
Quaternion conjugation has only four distinct actions and cannot distinguish
`i^2=-1` from identity, while quaternion left multiplication gives eight
distinct norm-preserving spinor states. This provides a structural falsifier
for sandwich recurrences and a concrete reason to pursue chiral-spinor states
before escalating to Spin(8) triality.

For the alphabet `{+-i,+-j}`, fixed word parity reaches only four of the eight
Q8 elements. The learned falsifier must therefore use matched odd/even lengths.
Its generic orthogonal control also needs four shared O(4) reflections:
quaternion left multiplication has `rank(I-A)=4`, so the earlier two-reflection
plane-rotation chart cannot serve as a capable Q8 baseline even though it has
the same 16 raw action coordinates per token/channel.

In the first controlled seed, the left-spinor model alone retains 100% central
pair accuracy through L16384. Three channels independently approach a faithful
Q8 action while one remains nuisance; a single joint frame retraction over all
four tokens and all channels repairs the slack without touching the decoder,
reducing homomorphism RMS to `9.98e-8` while preserving 100% accuracy. This is
the concrete form of the project principle: optimize freely in the tangent
chart, then retract the complete action family onto one shared representation
manifold.

Raw discovery is not perfectly reliable: it passes 8/10 on fresh seeds 10--19.
A decoder gate fixed on earlier seeds and using only pre-retraction distance to
the Q8 manifold raises the complete pipeline to 10/10 on those untouched seeds.
All 460 dense and 40 long validation cells are 100%, while the exact retracted
operators remain below `1.7e-7` homomorphism RMS. This separates representation
discovery, algebraic compilation, and decoder observability into explicit,
falsifiable stages.

### State geometry determines a congruence lattice, not one cardinality

A decoder-free recurrent-state corpus can support several exact finite actions
at once. If `~` is a partition of reachable states and every token respects it,
then `~` is a right congruence: `x ~ y` implies `x a ~ y a` for every token
`a`. Congruences are partially ordered by refinement. Coarser partitions are
quotient automata, so transition closure alone cannot identify which quotient
is the intended latent state.

This is observed directly in two Spin(8)-Q8 seeds. Independent clustering
recovers both an eight-state regular Q8 action and a two-state action. The
two-state partition is a balanced 4-to-1 coarsening with purity 1.0, and its
map intertwines all token transitions exactly in two disjoint corpora. It is
not total word-length parity: one inverse-generator pair is in the kernel and
the other maps to the nontrivial element, realizing an index-two character
`Q8 / C4 ~= C2`.

The historical cardinality-selection rule was not “choose the largest stable
clustering.” It accepted the largest reproducible K-means candidate only when
every other candidate found by that scan was certified to be its surjective
homomorphic quotient. In a finite deterministic action, the discrete partition
retains every state already separated by that particular metric fit;
incomparable discovered candidates require refusal. This replaced an arbitrary
Euclidean separation floor with a stronger within-scan algebraic certificate,
but it did not enumerate the complete congruence lattice. Exhaustive enumeration
of the recovered Q8 action later found block counts `{1:1, 2:3, 4:1, 8:1}` in
all nine seeds and established the observation-free identifiability boundary.

## Spin(8) triality memory theorem and implementation

The experimental Spin(8) branch uses the unique equivariant map from a
positive and negative chiral spinor to the vector representation. For a unit
positive key, the induced map from negative spinor to vector is orthogonal, so
single-pair binding is exactly invertible.

Raw superposition does not provide high capacity: every wrong-key term has
full norm. Multiplicity codes expose the exact law. With H channels and K code
columns, cross terms are weighted only by code inner products. Orthonormal
columns give exact retrieval for K at most H. Unit-norm tight frames attain the
classical frame-potential lower bound (K-H)/H on average squared interference
when K exceeds H.

An addressed dynamic form retains scan closure:

[
M_t[h] = r_t[h] V_t M_{t-1}[h] + B_t[h].
]

All retention vectors are diagonal in one fixed multiplicity basis. Transition
composition multiplies retentions and Spin(8) actions and rotates the earlier
drive before adding the later drive. The implementation supports exact hard
slot overwrite, shared Spin(8) transport, logarithmic-depth prefix evaluation,
and constant 8H recurrent state.

The rank-deficient completion experiment separately verifies the sample-
efficiency value of symmetry: the full equivariant bilinear tensor space is
one-dimensional, and that invariant family extrapolates where generic fitted
tensor and MLP families fail.

## What remains unproven

- Natural language has no supplied 3D geometric frame. Spin(3) equivariance is
  therefore an architectural symmetry and regularizer, not yet an identified
  linguistic symmetry.
- The recurrent linear map is channel-diagonal; channel interaction happens
  in controls, input projections, and feed-forward layers. A structured
  multi-channel rotor operator is an important future ablation. SchurScan is
  now the constructive candidate for that ablation.
- The complementary drive scale `sqrt(1-d^2)` couples long retention to weak
  writing. It is a stationary-variance convention, not required for BIBO
  stability. An independently bounded write gate is the next optimization
  falsifier.
- Three seeds, 300 updates, and context 64 are far too small for scaling-law,
  long-context retrieval, or downstream-quality claims.
- The PyTorch reference uses an explicit loop. Production throughput needs a
  fused or compiled selective scan and long-sequence numerical testing.
- Floating-point prefix trees and sequential scans are mathematically equal
  but cannot be bitwise equal because floating-point arithmetic is not
  associative.

This architecture should be treated as a falsifiable research program: retain
the identity, scalar selective-SSM, parameter-matched non-geometric, and
attention baselines; increase seeds and budgets; then test memory retrieval and
language quality before increasing model size.
