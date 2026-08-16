# Selective rotor SSM: mathematical contract

This note begins with the contract for the maintained model in
`pure_rotor_ssm/jax_backend.py` and `pure_rotor_ssm/torch_backend.py`.
`ga_ssm.py` and `rotor_ssm_torch.py` remain compatibility/training entry points
and resolve their maintained model names to that package. Later sections
preserve the broader experimental research ledger and the superseded pre-pure
contract; they are historical evidence, not claims about the rewritten model.

In the recurrence sections, an "exact" statement means finite real arithmetic
with a unit rotor and finite inputs/parameters. Floating-point implementations
approximate that contract and have the explicit qualifications below.

> **Cross-program boundary, 2026-08-10.** The transported hierarchical-memory
> result in `Spin8-Triality-Research` is not an implementation claim about this
> selective Cl(3,0) rotor SSM. Its co-moving compiler applies to a separate
> invertible value-transport recurrence; integrating it here would require a
> new equivalence proof and matched model experiment.

> **Maintained-family boundary, 2026-08-16.** This document remains the
> contract for Pure Rotor v2.1. The new `pure_spin8_ssm/` v1.0 family has a
> separate 24-scalar faithful triality cache, implementation contract, and
> checkpoint schema. Its addition does not alter the equations or load format
> below. See `pure_spin8_ssm/CONTRACT.md` and
> `experiments/PURE_SPIN8_VS_MAMBA2_RESULTS.md`.

## Canonical pure state transition

Each layer state is a full multivector
\(h_{t,c}\in\mathrm{Cl}(3,0)\), stored in the coefficient order
`[1,e1,e2,e3,e12,e13,e23,e123]`. For channel \(c\),

\[
\begin{aligned}
g_t &= \bigl[s,p,\lVert v\rVert^2,\lVert *B\rVert^2,
v\mathbin{\cdot} *B\bigr](x_t),\\
\Delta_{t,c} &= \Delta_{\min}+\operatorname{softplus}((W_\Delta g_t+b_\Delta)_c),\\
\lambda_c &= \lambda_{\min}+\operatorname{softplus}(\rho_c),\\
d_{t,c} &= \exp(-\Delta_{t,c}\lambda_c),\\
q_{t,c} &= \operatorname{RotorChart}(x_t,g_t,c),\\
w_{t,c} &= \operatorname{sigmoid}((W_wg_t+b_w)_c),\\
z_{t,c} &= \frac{P(x_t)_c}{\sqrt{1+\lVert P(x_t)_c\rVert^2}},\\
h_{t,c} &= d_{t,c}q_{t,c}h_{t-1,c}\widetilde q_{t,c}
 +(1-d_{t,c})w_{t,c}z_{t,c}.
\end{aligned}
\]

`P` and the rotor source are complete Spin(3)-isotypic linear maps. They mix
scalar with pseudoscalar and vector with Hodge-dual bivector, spanning the full
linear commutant rather than the old grade-diagonal half-family. The five
control features are polynomial invariants with finite derivatives at zero.
The rotor chart radially maps a predicted bivector of norm (r) to angle
`max_rotor_angle * tanh(r)` and evaluates its zero limit analytically.

For finite real controls and positive floors, (0<d_{t,c}<1),
(0<w_{t,c}<1), (\lVert z_{t,c}\rVert<1), and (q_{t,c}) is unit. Rotor
conjugation is orthogonal on the complete eight-dimensional coefficient
space. Hence

\[
\lVert h_{t,c}\rVert
\le d_{t,c}\lVert h_{t-1,c}\rVert+(1-d_{t,c}),
\qquad
\lVert h_{t,c}\rVert\le\max(\lVert h_{0,c}\rVert,1).
\]

This is a hard input-independent state bound in real arithmetic, not a
stationary-variance heuristic. It holds for arbitrarily large finite additive
input features because the projected candidate is smoothly bounded before it
is written. It does not bound residual block activations or parameter
gradients, and floating-point roundoff can exceed the unit ball by tolerance.

## Associative training and recurrent streaming

Write a transition as \(T=(d,q,b)\), where
\(T(h)=dqh\widetilde q+b\). Chronological composition is

\[
T_b\circ T_a=(d_bd_a,q_bq_a,b_b+d_bq_bb_a\widetilde q_b).
\]

It is associative over exact multivector arithmetic because it is function
composition. JAX uses `lax.associative_scan`; PyTorch uses an autograd-safe,
vectorized Hillis--Steele scan with logarithmic launch depth. Both retain a
sequential scan as the semantic oracle and token-streaming path. Floating-point
grouping makes the two orders numerically close rather than bitwise identical.

### PyTorch Schur-factored execution path (2026-08-16)

`scan_mode="schur_parallel"` expresses the same per-channel transition in the
two trivial plus two standard three-dimensional Spin(3) isotypic coordinates.
It composes the trivial scales, active scales, and active `3 x 3` maps in the
same chronological affine order as the direct rotor scan, then unpacks full-GA
states. The mode is opt-in and PyTorch-only: direct sandwich, scan/recurrent,
cache, CUDA, and first-order-gradient float64 checks establish semantic parity,
not bitwise equality or a throughput guarantee. It does not introduce
cross-channel transport; state transport remains channel-diagonal.

One layer caches exactly `(batch, channels, 8)` numbers, so an (L)-layer,
(C)-channel model stores (8LC) recurrent numbers per sequence regardless
of context length. A false/padded token is the exact identity transition
`(1, identity_rotor, 0)` and leaves the cache unchanged. PyTorch training
caches retain autograd history unless detached; the constant-state statement
describes inference state, not an untruncated training graph.

## Equivariance, full-GA closure, and numerical boundary

The complete deterministic block commutes with simultaneous conjugation by a
fixed Spin(3) rotor. Complete isotypic maps are intertwiners, controls use only
invariants, normalization has one scalar gain per multivector channel, and the
rotor source transforms covariantly. Training dropout uses one Bernoulli mask
per complete multivector, shared across its eight blade coordinates, so it too
commutes with the action for a coupled mask. The claim is proper-rotation
Spin(3) equivariance, not reflection, Lorentz, or arbitrary Clifford-group
equivariance.

Rotors and their products remain in the four-dimensional even subalgebra, but
states and drives occupy the full eight-dimensional algebra. An even state is
preserved only when every drive is even. JAX's algebraically specialized
sandwich rotates the vector and Hodge-dual-bivector copies directly. PyTorch
keeps that implementation as an oracle but dispatches CUDA to two dense
geometric products because they benchmark faster there. Tests compare both on
arbitrary full-GA values.

The strict inequalities and exact associativity above are real-arithmetic
statements. Finite precision introduces rotor norm drift, possible decay
rounding to zero or one, and scan-order differences. Nonfinite values
propagate. The state Jacobian with respect to the previous state has operator
norm (d_t), so pure recurrent gradients can vanish over long horizons even
while forward states remain bounded. JAX and PyTorch use the same equations,
basis order, optimized rotor formulas, mask semantics, and cache contract, but
their RNGs and default initializers differ; independently initialized models
are not expected to have equal logits.

## Superseded pre-pure state transition (historical)

The following three sections record the model audited on 2026-08-06 before the
pure rewrite. They are retained so the audit and frozen artifacts remain
interpretable. Their `sqrt(1-d^2)` write rule, grade-only maps, nonsmooth norm
features, and coordinatewise dropout no longer describe the maintained model.

For channel \(c\), the state \(h_{t,c}\) after token \(t\) is a multivector in
the Euclidean Clifford algebra \(\mathrm{Cl}(3,0)\). Given token features
\(x_t\), the layer computes

\[
\begin{aligned}
g_t &= \operatorname{InvariantFeatures}(x_t),\\
\Delta_{t,c} &= \Delta_{\min}
  +\operatorname{softplus}((W_\Delta g_t+b_\Delta)_c),\\
\lambda_c &= \lambda_{\min}+\operatorname{softplus}(\rho_c),\\
d_{t,c} &= \exp(-\Delta_{t,c}\lambda_c),\\
B_{t,c} &= \operatorname{EquivariantBivector}(x_t,c)
  \tanh((W_Rg_t+b_R)_c),\\
q_{t,c} &= \exp\!\left(-\frac12
  \operatorname{Bounded}(B_{t,c})\right),\\
u_{t,c} &= \sqrt{1-d_{t,c}^2}\,
  \operatorname{GradeLinear}(x_t)_c,\\
h_{t,c} &= d_{t,c}q_{t,c}h_{t-1,c}\widetilde{q}_{t,c}+u_{t,c}.
\end{aligned}
\]

Here `Bounded` is the radial chart implemented by `rotor_from_bivector`: for
\(r=\lVert B\rVert\), it replaces \(B\) by
\(\theta_{\max}\tanh(r)B/r\), with value zero at \(r=0\).

Here \(\widetilde q\) denotes Clifford reversal. In exact arithmetic,
\(q_{t,c}\) is a unit even multivector—a rotor—so the sandwich map
\(h\mapsto q_{t,c}h\widetilde{q}_{t,c}\) preserves the Euclidean coefficient
norm on the full algebra. The strict floors \(\Delta_{\min}>0\) and
\(\lambda_{\min}>0\) imply, for finite real controls,

\[
0<d_{t,c}\leq
d_{\max}:=\exp(-\Delta_{\min}\lambda_{\min})<1.
\]

Consequently,

\[
\lVert h_{t,c}\rVert
\leq d_{\max}\lVert h_{t-1,c}\rVert+\lVert u_{t,c}\rVert.
\]

If the drive is uniformly bounded by \(U\), iteration of this inequality gives

\[
\lVert h_{t,c}\rVert
\leq d_{\max}^{\,t}\lVert h_{0,c}\rVert
+\frac{1-d_{\max}^{\,t}}{1-d_{\max}}U.
\]

This is a genuine bounded-state guarantee for a bound \(U\) on the actual drive
\(u_{t,c}\), not merely on the raw token features. Bounded features imply such a
bound only after fixing finite projection weights. The guarantee does not
promise easy numerical conditioning when \(d_{\max}\) is extremely close to
one; in that regime the model deliberately retains a very long memory.

In floating point, exponentiation can underflow to zero or round to one, and
composed rotors drift slightly from unit norm. The default float32 floors keep
the one-step upper bound representably below one, but reduced precision need
not. The displayed strict inequalities and norm preservation are therefore
mathematical guarantees, with numerical error measured rather than assumed.

The decay controller and rotor controller are initialized to zero. Decay
rates are chosen so zero-control channels have log-spaced half-lives from 4 to
2,048 tokens. Rotors start exactly at identity, but the bivector exponential
uses its correct analytic small-angle limit, so the rotor controller has a
finite, generically nonzero gradient at initialization. The separately
exported raw quaternion normalizer is singular at a zero parameter vector; the
implementations use an identity fallback there, not a continuous extension of
normalization.

## Superseded scan discussion (historical)

Represent a token transition by \(T=(d,q,u)\), acting on a state as

\[
T(h)=dqh\widetilde{q}+u.
\]

Applying \(T_a\) first and \(T_b\) second gives another transition of the same
form:

\[
T_b\circ T_a=
\left(
d_bd_a,
q_bq_a,
u_b+d_bq_bu_a\widetilde{q}_b
\right).
\]

This operation is associative in exact arithmetic because it is ordinary
function composition and rotor multiplication is associative. JAX therefore
computes every prefix with `lax.associative_scan` during training. Recurrent
inference uses the same equation with one fixed-size state per layer. Tests
compare full parallel, arbitrarily chunked, and token-by-token execution at
both state and logit level in deterministic/evaluation mode. Different
floating-point grouping produces close, not identical, values.

For a model with \(L\) layers and \(C\) multivector channels, the streaming
cache contains exactly \(8LC\) stored numeric state scalars per sequence,
independent of context length. In PyTorch training, retaining a cache with its
autograd history is not constant-memory; inference must use `no_grad` or a
detached cache for the storage claim to describe the complete live state.
The current implementation still performs ordinary vocabulary decoding, so
generation cost is constant in past context length but not constant in
vocabulary size.

## Superseded equivariance discussion (historical)

For a fixed frame rotor `s`, transform every multivector as
`x' = s x reverse(s)`. With dropout disabled (or in evaluation mode), the
then-maintained block commuted with this action:

- `GradeLinear` preserves grades and shares scalar channel weights across all
  coordinates within a grade;
- control networks see only scalar/pseudoscalar coefficients and vector/
  bivector norms, which are invariant under proper 3D rotations;
- the predicted bivector and its exponential transform by conjugation;
- RMS normalization, residual addition, and invariant gating commute with the
  same action.

That implementation's elementwise training dropout was not equivariant: a coordinatewise
mask does not commute with a general rotation. The numerical block test runs
with dropout disabled and establishes deterministic block equivariance, not
training-time stochastic equivariance.

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
an exact Hodge-copy witness that a stack of the old *linear* family cannot
express at any depth.

This limitation is historical. The canonical v2 package uses
`Spin3IsotypicLinear` throughout its transition and feed-forward paths;
`GradeLinear` is now only a compatibility alias for that complete map.

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

## Pure v2.0.0 implementation evidence

This section is now historical. The maintained v2.1.0 model preserves the same
bounded recurrence but expands the default physical rotor chart from `pi/2` to
the full open `pi` range. The change was frozen before the transport-ablation
cohort because every v2.0 layer's p95 angle saturated the old cap.

The first rewritten checkpoint used 32 channels, four layers, context 256,
batch 60, and 407,840 parameters. On an RTX 2070 SUPER, 500 steps took 309.7
seconds and peaked at 4,109 MiB of allocated CUDA tensor memory. Fixed-window
validation loss fell from 5.5430 to 1.7616 nats (2.5414 bits/byte). The
checkpoint reloads strictly, and all four rotor controllers are active.

At batch eight and context 256, the parallel path measured 67.29 ms inference
and 198.26 ms forward/backward versus 752.40 ms and 1,689.22 ms for the
recurrent oracle. This is single-device systems and trainability evidence, not
a matched quality comparison. The full protocol, hashes, checkpoint digest,
timings, and limitations are in
`experiments/PURE_ROTOR_SSM_V2_RESULTS.md`.

## Pure v2.1.0 transport evidence

The v2.1 large-model seed-0 retrain used the same 407,840-parameter, C32/L4,
context-256 configuration and changed only the physical rotor chart limit from
`pi/2` to `pi`. It reached 1.760117 validation nats versus 1.761575 for v2.0,
with the same 4,109 MiB peak allocation. The 0.001458-nat difference is one
seed and is not an established quality improvement. All four p95 rotor angles
again approached the chart boundary (3.136--3.140 radians), so the expanded
range is used but does not remove saturation.

The preregistered v2.1 transport ladder then ran 105 prediction trainings and
70 memory trainings over five paired seeds. At C8/L2, rotor confirmation loss
was 2.430973 nats versus 2.451324 for retrained identity, a paired improvement
of 0.020351 nats with five wins. Clamping the trained rotor to identity raised
loss by 0.153236 nats on average and time-shuffling actions raised it by
0.177262, establishing that the learned ordered action matters to this model.

That result is not a unique noncommutative or Clifford advantage. At the same
state size, commuting complex phases reached 2.422885 and quaternion left
action reached 2.406740; generic SO(8) reached 2.324552 using substantially
more parameters. At the nearest effective-parameter match, quaternion and
complex phases remained better than the rotor. At matched measured CUDA time,
a launch-efficient C60 identity model reached 2.023329 versus the C8 rotor's
2.430973, so rotor compute efficiency fails on this eager PyTorch/RTX 2070
SUPER implementation.

Memory claims also fail. Rotor associative-recall means were below identity at
every registered length and decayed toward chance at length 512. The Q8 task
remained at chance-scale accuracy for every family, with no consistent rotor
extrapolation benefit. Consequently v2.1 supports a narrow prediction benefit
over identity, not better memory, not better compute efficiency, and not a
claim that Cl(3,0) rotors are the best stable transition. See
`experiments/PURE_V2_1_TRANSPORT_ABLATION_RESULTS.md` and the raw aggregate
`experiments/pure_v2.1.0_transport_ablation.json`.

## Controlled local-GPU evidence (superseded architecture)

> **Legacy evidence only.** These frozen runs evaluated the pre-pure
> `sqrt(1-d^2)`/grade-linear architecture. Their observations remain valid for
> those artifacts but do not measure the rewritten bounded-write,
> complete-isotypic model. New quality comparisons are required.

`train_rotor_ssm_torch.py` compares the selective rotor model against an
identity-rotation ablation. Both variants have 22,968 nominal parameters, the same
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
that the rotor path is active. This is not a causal attribution of the loss
difference; that would require post-training interventions or a stronger
ablation. The identity variant also contains nominally counted rotor parameters
whose functional gradients are zero when `max_rotor_angle=0`, so equal raw
parameter count is not equal effective transition capacity. These short runs
are a promising mechanism-level result, not evidence of state-of-the-art
language modeling or a statistically established advance.

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

For the real-type finite-group irreps used in these experiments, the compiler
needs no character table or supplied low-dimensional representation matrices.
A generic symmetric right-regular operator commutes with the left-regular group
action, so each appropriately split dimension-`d` eigenspace supplies an exact
invariant `d`-dimensional action. Complex- and quaternionic-type real irreps
have additional Schur degeneracies and are not covered by that unqualified
eigenspace statement.
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
the complete dense/long gate in all ten seeds. Together these controls are
evidence for a first-order optimization barrier, not an impossibility theorem
for endpoint learning or proof that no other mechanism contributes.
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

The later frozen `2.A5` pilot tests the same center issue without the Q8 word-
parity confound. Training omits `a,a`, while paired evaluation contrasts
`a,a=z` against `b,b^-1=e` inside shared contexts whose post-relation A5
projections agree. Across seeds 0--2, the direct Spin quaternion product scan
has 100% target-versus-central-partner margin at L16, L64, and L128. Its mean
exact binary accuracy is 99.76%, 79.60%, and 62.20%; Pure Rotor v2.1,
identity transport, and Transformers Mamba-2 all approach chance on the long
center metrics. The exact-table and float64 quaternion oracles are perfect,
while the projective oracle is exactly 50% on the binary center.

This is empirical evidence that direct Spin composition supplies information
discarded by sandwich transport. It is not a theorem about Mamba-2, a broad
model-quality result, or a reason to rewrite the maintained v2.1 recurrence.
The reusable PyTorch primitive therefore lives in the explicitly experimental
`pure_rotor_ssm/spin_scan.py`; the authoritative measurements and artifact hash
are in `experiments/PURE_ROTOR_2A5_CENTER_PILOT300_RESULTS.md`.

An evaluation-only follow-up then applies a deterministic input-only selector
to the frozen schedules. The shortest locally reduced identity/center pair
absent from every seed has length 11. Without retraining, the same Spin
checkpoints remain 100% exact at L16, retain 100% central margin through L128,
and average 59.68% exact L128 accuracy. The other candidates remain near chance
on long center metrics. This reduces the local-`a,a`-memorization explanation,
but is post-pilot exploratory evidence rather than a preregistered relation-
family result; see `experiments/PURE_ROTOR_2A5_UNSEEN_RELATION_RESULTS.md`.

### Signed rigid path development and local transition identification

The experimental `pure_rotor_ssm/motor_scan.py` replaces a single quaternion
state by a unit dual quaternion `m=q_r+epsilon q_d`. The unit/Study constraints
are `||q_r||=1` and `<q_r,q_d>=0`; the translation is
`2 q_d conjugate(q_r)`. Multiplication is associative and gives the semidirect
law

```text
(q,t) (r,u) = (q r, t + R(q) u).
```

Thus `m` and `-m` preserve distinct recurrent states while inducing the same
physical `SE(3)` action. The direct-product ablation
`Spin(3) x (R^3,+)` preserves the same central sign but omits the `R(q)u`
coupling. The numerical implementation audit establishes matrix equivalence,
Study constraints, scan/cache/gradient parity, and central action blindness
through L4096. It does not establish a speed advantage; eager 4 by 4 matrices
are faster on the recorded GPU.

The rigid `2.A5` task adds body-frame translation tokens while withholding
`a^2`, `b^3`, and `(ab)^5`. Its quotient oracle is physically perfect but
exactly loses the paired sign bit. Blind 300-step training fails for the motor
classifier and all parameter-near learned controls. A 49-parameter direct-
product scan retains sign but fails translation, confirming that center
tracking alone is not enough.

With every-prefix signed pose supervision, however, a right-composed token
increment is observable locally:

```text
q_delta = conjugate(q_prev) q_next,
t_delta = R(q_prev)^T (t_next - t_prev).
```

Averaging these legal prefix differences by token identifies the seven motors
without reading a held-out relation. Across three conjugated generator
coordinates and three independently sampled legal schedules, all nine runs
recover 100% joint signed pose and paired double-cover pose on all 162 splits
through L128. This is an exact formula plus a replicated finite empirical
identification result. It is not end-to-end learning, and it depends on dense
adjacent pose targets; noisy, continuous, and final-only identification remain
open. See `experiments/SPIN_MOTOR_RIGID_2A5_RESULTS.md`.

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

The experimental Spin(8) branch uses the equivariant map from a positive and
negative chiral spinor to the vector representation, unique up to an overall
scalar (then fixed by the implementation's normalization). For a unit positive
key, the induced map from negative spinor to vector is orthogonal, so
single-pair binding is exactly invertible when that key and normalization are
known.

Raw superposition does not provide high capacity: before multiplicity
weighting, every wrong-key bind--unbind map preserves the norm of the stored
value rather than attenuating it by a factor such as \(1/8\). Multiplicity
codes expose the exact law. With \(H\) channels and \(K\) code columns, cross
terms are weighted only by code inner products. Orthonormal columns give exact
retrieval for \(K\leq H\). For \(K>H\), unit-norm tight frames attain the
classical frame-potential lower bound \((K-H)/H\) on average squared code
correlation. This equals expected retrieval MSE for independent, zero-mean
isotropic values; it is not a deterministic minimum for every correlated or
adversarial collection of stored values.

An addressed dynamic form retains scan closure:

\[
M_t[h] = r_t[h] V_t M_{t-1}[h] + B_t[h].
\]

All retention vectors are diagonal in one fixed multiplicity basis. Transition
composition multiplies retentions and Spin(8) actions and rotates the earlier
drive before adding the later drive. The implementation supports exact hard
slot overwrite, shared Spin(8) transport, logarithmic-depth prefix evaluation,
and constant 8H recurrent state.

The rank-deficient completion experiment separately provides controlled
numerical evidence for the sample-
efficiency value of symmetry: the full equivariant bilinear tensor space is
one-dimensional, and that invariant family extrapolates where generic fitted
tensor and MLP families fail.

## Implementation edge conditions and backend boundary

- Empty sequences are outside the maintained scan API. The scan functions and
  PyTorch recurrent layer reject them; callers must supply at least one token.
- The per-token rotor is even, but the state and drive are full
  \(\mathrm{Cl}(3,0)\) multivectors. Even-subalgebra closure holds only when the
  initial state and every drive are even. Learned language embeddings are not
  restricted to that subalgebra.
- The maintained controls use squared vector norm, squared dual-bivector norm,
  and their dot product. These polynomial features have finite derivatives at
  zero. `grade_invariants` remains exported only as a compatibility helper and
  retains the old norm nondifferentiability.
- The state-to-state Jacobian of one recurrence step has operator norm \(d_t\)
  in exact arithmetic, so gradients through a long pure state path contract by
  at most the product of decays and may vanish. Residual paths mitigate this at
  block level. BIBO state stability is not a general bound on gradients with
  respect to every learned parameter.
- JAX and PyTorch implement the same algebra and recurrence, but use different
  parameter initializers, execution groupings, RNG systems, and some output
  dtype conventions. Mathematical/core-primitive parity does not imply that
  separately initialized full models have equal logits.
- Nonfinite features or parameters propagate; the guarantees assume finite
  quantities and do not constitute NaN/Inf recovery behavior.

## What remains unproven

- Natural language has no supplied 3D geometric frame. Spin(3) equivariance is
  therefore an architectural symmetry and regularizer, not yet an identified
  linguistic symmetry.
- The recurrent linear map is channel-diagonal; channel interaction happens
  in controls, input projections, and feed-forward layers. A structured
  multi-channel rotor operator is an important future ablation. The current
  `schur_parallel` implementation is only a per-channel coordinate
  factorization of the existing transport, not that multi-channel operator.
- The new `(1-d) * sigmoid(write) * bounded(candidate)` rule proves a hard
  state bound but couples fast writing to forgetting. Whether that tradeoff is
  better than an independently gated bounded drive is an empirical question.
- Three seeds, 300 updates, and context 64 are far too small for scaling-law,
  long-context retrieval, or downstream-quality claims.
- The PyTorch training path uses a differentiable Hillis--Steele scan with
  logarithmic launch depth and `O(L log L)` work. A fused `O(L)` work-efficient
  CUDA kernel remains an optimization opportunity; correctness does not depend
  on an experimental compiler-only scan primitive.
- Floating-point prefix trees and sequential scans implement the same exact
  recurrence but are not guaranteed, and generally should not be expected, to
  be bitwise equal because floating-point arithmetic is not associative.

This architecture should be treated as a falsifiable research program: retain
the identity, scalar selective-SSM, parameter-matched non-geometric, and
attention baselines; increase seeds and budgets; then test memory retrieval and
language quality before increasing model size.
