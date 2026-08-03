# Selective Spin(8) triality SSM research program

## Research question

Can a token-selective, noncommutative orthogonal recurrence learn ordered
sequence structure more efficiently than parameter- and state-matched
commutative recurrences, and does Spin(8)'s chiral-spinor representation add
useful capacity beyond low-dimensional complex, quaternion, and Cl(3,0)
actions?

The constructive core is now implemented. `spin8_triality.py` supplies exact
vector and two chiral 8D representations driven by one 28D bivector, verifies
triality, and exposes associative affine scans plus constant recurrent state.
`pure_spin8_positive` supplies the write-free positive-chiral mechanism row.
The broader selective language-model comparison and invariant triality-coupled
ablation remain future experiments; the maintained production-scale rotor
language model still uses Cl(3,0).

The strongest new result is no longer just a supplied algebra check. The
implemented infinitesimal equivariance system verifies that the triality map in
Hom(S+ tensor S-, V) is one-dimensional. Under a deliberately rank-16-of-64
observation design, a one-parameter invariant completion model generalizes at
cosine 1.0 in 3/3 seeds, while generic 512-parameter bilinear and 608-parameter
MLP models fit training below 1.2e-6 MSE and fail off-support. This isolates
symmetry-driven law identification from interpolation. See the corresponding
triality-identifiability result document under experiments.

The memory claim has also been corrected and strengthened. A single 8D
triality vector is an exact single-pair store, not a high-capacity VSA.
Orthogonal multiplicity codes provide exact K-at-most-H retrieval; unit-norm
tight frames attain the frame-potential lower bound beyond that rank horizon.
An addressed overwrite recurrence then combines exact slot replacement,
shared Spin(8) transport, logarithmic-depth affine scanning, and constant 8H
recurrent state. All float64 mechanism errors are below 2.3e-15. Addresses,
query keys, and actions remain supplied. See the coded-memory result document
under experiments.

The supplied-action limitation has now been removed in a controlled synthetic
gate. Ten hidden noncommuting four-token Spin(8) families were observed only
through five columns of their vector and positive-chiral actions; the complete
negative action and three columns of each visible action were withheld.
Unconstrained and independently optimized SO(8) controls fit every visible
endpoint but fail the hidden representation and long composition. Joint
diagonal-triality retraction passes 10/10 seeds, recovers hidden tangent
coordinates above 0.999999999996 cosine, and retains at least 0.999999995
cosine through length 2048. See the blind-shared-action result document under
experiments.

The action and address compilers have subsequently been trained together. A
single run receives partial vector/positive action columns, rank-2 negative
endpoints, and fresh continuous aliases without logical key IDs. Joint
triality and balanced routing pass all dense L16--2048 gates in 10/10 seeds.
An independent 84-coordinate action family fits the same supplied evidence but
retains exactly 21 unobserved tangent dimensions: held-out negative-action
cosine is only 0.864--0.930 and direct L2048 retrieval is 0.408--0.546. The
shared 28-coordinate family recovers the hidden tangent above
0.999999999999995 cosine. This is a partial-observation relational-completion
advantage, not a generic memory advantage over a direct oracle.

An exact triangular triality coupling is now implemented in
`spin8_triality_lift.py`. Two independent chiral-spinor affine streams are
scanned first, their `S+ x S- -> V` binding is evaluated pointwise, and the
bound vector drives a second affine scan. A homogeneous 81D lift proves
single-scan closure, while the practical staged implementation retains only 24
streaming scalars and matches direct recurrence to `4.27e-14`. Two-way binding
feedback is not claimed: its generic polynomial degree grows without bound and
breaks fixed-dimensional affine closure. See
`experiments/SPIN8_TRIANGULAR_TRIALITY_LIFT_RESULTS.md`.

The first mechanistic endpoint result is also complete. On Q8, unconstrained
positive-chiral token tangents learn behaviorally correct short paths but have
large raw homomorphism error and eventually drift. A compiler using only
uniform random token strings, recurrent states, and the model's own anonymous
endpoint predictions recovers a regular eight-state action, derives its exact
representation, and jointly retracts the whole token family. It passes seed 19
and untouched seeds 20--28 through L16,384 without a supplied table or
post-compilation gradients. Quaternion and Householder baselines show that
blind table recovery is not unique to Spin(8); the distinctive observed
property is Spin(8)'s rank-eight exact endpoint section versus their rank-four
minimal sections. See
`experiments/SPIN8_TABLE_BLIND_COMPILER_RESULTS.md` and
`experiments/TABLE_BLIND_FAMILY_BASELINES_RESULTS.md`.

## Relation to the 2025-2026 frontier

The defensible current trend is toward more expressive recurrent transitions,
not a field-wide consensus around rotors. DeltaProduct studies products of
Householder transitions; NeurIPS 2025 structured sparse transitions target
finite-state tracking; 2026 prefix-scannable theory emphasizes associativity;
and M2RNN investigates nonlinear matrix-valued state. These support testing
noncommutative dynamics while also supplying serious non-Clifford baselines:

- [DeltaProduct](https://arxiv.org/abs/2502.10297)
- [Structured Sparse Transition Matrices](https://papers.neurips.cc/paper_files/paper/2025/file/77b830c18836a9b2e1395a4936dd687a-Paper-Conference.pdf)
- [Sequential-Parallel Duality in Prefix Scannable Models](https://arxiv.org/abs/2506.10918)
- [M2RNN](https://arxiv.org/abs/2603.14360)

Parallel prefix scans require associative composition, not commutativity. That
observation is central to this project. It does not imply that diagonal SSMs
are dead, that all dense transitions are impractical, or that geometric
algebra is the canonical representation. Those are empirical questions.

## Proposed Spin(8) transition

Use an eight-real-dimensional positive-chirality spinor state per channel,
not a 256-dimensional full Cl(8,0) multivector:

```text
B_t       = sum_(i<j) beta_t,ij G+_ij       # 28 fixed 8x8 generators
Q_t       = exp(B_t) in SO(8)               # chiral spin representation
d_t       = exp(-Delta_t lambda)
h_t       = d_t Q_t h_(t-1) + u_t           # h_t in R^8
```

The generator controller emits 28 bivector coefficients per channel. The
fixed skew-symmetric matrices `G+_ij` realize the positive chiral-spinor
representation. A negative-chirality model uses the corresponding `G-_ij`.

Crucial representation-theory correction: a single real eight-dimensional
half-spin representation does not define a smaller transition family than a
general `SO(8)` action. Triality permutes the vector and two half-spin
representations; their derived 28-dimensional Lie-algebra representations are
related by outer automorphisms of `so(8)`. At group level the kernels differ,
but each image is an orthogonal eight-dimensional rotation group. Therefore a
"dense SO(8) versus chiral Spin(8)" comparison tests parameterization,
optimization, and numerical construction—not raw transition expressivity.

The kernel difference is nevertheless observable when the task uses central
elements. A rotor sandwich factors through the corresponding orthogonal
quotient, while a chiral-spinor action may retain a central sign erased by the
vector action. The Q8 center-fidelity gate in
`experiments/SPINOR_CENTER_FIDELITY_GATE.md` is the minimal example:
quaternion conjugation collapses eight elements to four, but left spinor
multiplication remains faithful.

The generic baseline has now been executed rather than left as a caveat. The
positive and standard SO(8) generator bases have an exactly orthogonal 28D
coefficient map. Plain SGD preserves mapped actions to `3.5e-18` and logits to
`1.1e-16`; AdamW breaks the functional equivalence because its coordinatewise
moments do not commute with that basis rotation. In fresh seeds 60--64, both
charts fit the endpoint curriculum in 5/5 and fail the raw dense gate in 0/5.
This falsifies a single-stream capacity advantage and moves the distinctive
Spin(8) hypothesis to center kernels and coupled triality.

The affine transition tuple `T=(d,Q,u)` composes as

```text
T_b compose T_a = (d_b d_a, Q_b Q_a, u_b + d_b Q_b u_a).
```

It is therefore closed and associative over exact arithmetic. Training can use
a parallel prefix scan; inference carries eight real state values per channel.
Floating-point tree and recurrent evaluation must be compared within measured
tolerances rather than described as bit-exact.

## Triality extension

The high-risk, potentially distinctive variant maintains three eight-real
states transformed by the same bivector controller:

```text
v_t  in V       # vector representation
s+_t in S+      # positive chiral spinor
s-_t in S-      # negative chiral spinor
```

Spin(8) triality relates these three representations. A later model may use a
Spin(8)-invariant trilinear interaction among `V`, `S+`, and `S-` to control
writes or gates. That coupling must be isolated as a separate ablation; it
must not be silently added to the basic chiral-spinor comparison.

The algebra gate now makes a scan-safe coupling precise. With the fixed
Clifford maps `rho_i: S+ -> S-`, token-side features can generate coupled
drives

```text
u-_t       = rho(a_t) p_t
u+_t       = rho(a_t)^T m_t
uV_t[i]    = m_t^T rho_i p_t
```

for token projections `a_t in V`, `p_t in S+`, and `m_t in S-`. These three
writes are linked by the same invariant triality tensor verified by the
algebra harness, yet remain functions of the current token only. The recurrent
transition is still affine on the direct sum `V + S+ + S-`, so associative
prefix composition survives unchanged.

By contrast, feeding the recurrent states themselves through those bilinear
maps produces a nonlinear recurrence whose fixed-dimensional affine tuples do
not close under composition. That version would sacrifice the exact prefix
scan unless the state is lifted to a larger closed feature algebra. The first
triality ablation must therefore compare independent token drives against the
coupled token-drive construction above; recurrent-state bilinear coupling is a
separate, higher-cost research branch.

## The representation ladder implemented now

`recurrence_families_torch.py` gives every family eight real state values per
channel and identically shaped parameter tensors. All rotation controllers
start at identity, so every model has the same parameters and computes the
same function before training.

| Family | Eight-real state interpretation | Selective action | Commutative action |
|---|---|---|---|
| `real_selective` | 8 real scalars | positive component decay | yes |
| `complex_unitary` | 4 complex scalars | four U(1) phases | yes |
| `quaternion_even` | 2 quaternions | unit-quaternion left action | no |
| `ga_rotor_selective` | one Cl(3,0) multivector | rotor sandwich | no |
| `ga_rotor_grade_decay` | one Cl(3,0) multivector | rotor sandwich and four grade decays | no |
| `hybrid_complex_ga` | two complex and two Cl(3,0) channels at the default width | direct-sum phase/rotor action | partly |
| `ga_rotor_static` | one Cl(3,0) multivector | decay/write only; rotor is static | rotor repeats |

The write-free mechanistic harness also exposes
`pure_quaternion_spinor`: two quaternionic spinors per channel transformed by
one shared unit-quaternion left action. It has the same three-coordinate token
chart and eight-real state width as `pure_ga_rotor`, making the Q8
center-fidelity comparison parameter matched.

This ladder separates four questions:

1. Does phase memory beat positive real decay?
2. Does noncommutative composition beat commuting phases?
3. Does the full multivector grade structure help beyond quaternion spinor-like
   left action?
4. Does token-selective rotation beat a learned but token-independent rotor?

The future Spin(8) chiral family must use the same eight-real state contract.
Its larger 28-coordinate controller should be matched either by a common
controller trunk with equal total trainable parameters or by an explicit
parameter-versus-quality Pareto curve.

## Annotation acceptance contract: recurrent state

Every model exposes a tuple containing one `(batch, channels, 8)` recurrent
state per layer. The harness rejects an implementation unless these agree:

- one full sequence;
- arbitrary sequence chunks with carried states;
- one-token calls with carried states.

The cache is `layers * channels * 8` real scalars per sequence and never grows
with context. This directly prevents generation from recomputing its complete
history.

## Experimental tasks

### Stage 1: ordered group products

The implemented harness supports Q8, D4, S3, and A5. Q8 inputs are drawn
from `[1,i,j,k,-1,-i,-j,-k]`; every position is supervised with the ordered
prefix product and the final position measures full-sequence tracking. The
earlier dense-prefix ladder remains a debugging baseline, while the endpoint
mixing result now supplies a controlled short-to-long endpoint curriculum.
Since `i*j=k` but `j*i=-k`, a successful solution must distinguish operation
order.

Q8 must include central-sign pairs such as the empty word versus `i*i`. A pure
Cl(3) sandwich recurrence is incapable of separating these states because both
act as identity after conjugation. A left quaternionic spinor sees
`i*i=-1` as a sign flip. This turns Q8 from merely a quaternion-friendly task
into an architectural kernel falsifier.

Follow with:

- products in non-abelian dihedral and symmetric groups;
- continuously sampled SO(3) and SO(8) operation composition;
- held-out sequence lengths longer than training lengths;
- corrupted-operation and inverse-cancellation tests.

No single group should determine the conclusion: Q8 naturally favors
quaternions and is a diagnostic, not a neutral final benchmark.

A5 is the decisive structural group in the current ladder. It is non-solvable,
so fixed-depth input-dependent complex diagonal SSMs cannot exactly track it at
finite precision, while its faithful icosahedral rotation representation lies
in `SO(3)`. Use a small generator alphabet rather than exposing all 60 elements
as input tokens. Also apply the held-out ordered-pair split; ordinary random
words do not exclude local-transition memorization.

### Stage 2: selective memory

- order-sensitive associative recall;
- induction heads and variable-spacing copy tasks;
- nested operation scopes;
- state tracking under distractors;
- evaluation from 64 through at least 4,096 tokens.

### Stage 3: language

Run byte- and subword-level modeling with fixed tokenization, data hashes,
training tokens, parameter budgets, and validation windows. Report bits per
byte or token NLL together with throughput and cache size. Do not advance a
family based only on training loss.

## Required baselines for Spin(8)

In addition to the seven implemented families:

- a direct skew/Cayley/exponential `SO(8)` parameterization, understood as an
  optimization and kernel baseline for a chiral Spin(8) parameterization;
- an unconstrained but contractively parameterized dense 8x8 recurrence;
- a low-rank stable recurrence;
- the fastest diagonal complex SSM available at the same state and parameter
  budget.

These distinguish a benefit from noncommutativity or orthogonality. A lone
chiral Spin(8) action cannot establish a representation-specific expressivity
benefit over SO(8); a genuinely triality-specific claim requires coupled
vector, positive-spinor, and negative-spinor states plus an isolated invariant
coupling ablation.

## Rotor construction ablation

Generating `Q_t` may dominate runtime. Compare:

1. batched 8x8 matrix exponential;
2. a Cayley transform with a stable linear solve;
3. products of token-selective Givens rotations;
4. a structured Lie-product approximation.

Measure orthogonality error, parallel/recurrent drift, backward stability,
tokens/second, and peak memory. A 16x16 tensor-core tile is not itself proof of
high utilization; actual kernels must be profiled on target hardware.

The Cayley row has a topology caveat: finite skew parameters cannot map to an
orthogonal action with eigenvalue `-1`. It is therefore a fair local chart and
speed baseline near identity, but not an exact center-fidelity baseline when a
token itself must realize a nontrivial central sign. In a plane its angle is
`2 atan(lambda/2)`, approaching `pi` only at infinite tangent magnitude. Keep
this capacity boundary separate from optimization and kernel-speed findings.

## Additional geometric proposals: triage

### Grade-preserving multi-decay: implemented trial

Let `P_k` project onto grade `k` and define
`D_t = sum_k d_t^(k) P_k`. Rotor conjugation preserves grades, so `D_t`
commutes with the rotor action. The transition

```text
h_t = D_t Ad(R_t) h_(t-1) + u_t
```

remains closed under composition:

```text
(D_b,R_b,u_b) compose (D_a,R_a,u_a)
  = (D_b D_a, R_b R_a, u_b + D_b Ad(R_b) u_a).
```

This is implemented as `ga_rotor_grade_decay`, using separate token-selective
decays for grades 0, 1, 2, and 3 while retaining the same eight-real state and
parameter tensors as every other ladder family. Its value remains an empirical
question. The proposed semantic labels for grades (facts, directions,
temporary relations) are hypotheses, not mathematical consequences.

### Pseudo-Euclidean boosts: interesting but unstable by default

Cl(2,1) or Cl(3,1) boosts preserve an indefinite quadratic form, not the
ordinary Euclidean coefficient norm used by optimization. A boost of rapidity
`eta` can have Euclidean operator norm proportional to `exp(|eta|)`. Stability
therefore requires a constraint such as `d_t exp(|eta_t|) < 1`, not merely
`d_t < 1`. General pseudo-Euclidean bivectors can contain elliptic,
hyperbolic, and parabolic parts, so the single `cosh/sinh` expression applies
only to appropriate simple bivectors. Keep this for hierarchy-specific tests
after the Euclidean ladder is understood.

### Motor transitions: domain-specific, retain a write path

Projective GA motors or dual quaternions give associative rigid screw motions
and may suit trajectories, robotics, or spatial event streams. A pure motor
sandwich does not replace a general memory write: zero multivector state stays
zero, and a state constrained to one rigid-motion orbit cannot store arbitrary
token content. A credible model would transport a geometrically meaningful
state with a motor and retain a separate equivariant drive, then compare it on
spatial tasks. It is not presently justified as the default text recurrence.

## Evidence thresholds

Treat Spin(8) as supported only if it:

- passes algebra, Lie-commutator, orthogonality, causality, gradient, and
  streaming-equivalence tests;
- improves the noncommutative suite across at least five seeds and multiple
  groups, not only Q8;
- generalizes to longer sequences better than both complex and dense
  orthogonal baselines;
- retains an advantage on a language or selective-recall task;
- reports its controller parameter and compute premium;
- remains numerically stable at the longest tested context.

The result should be published even if negative. The purpose of the ladder is
to determine which algebraic property matters, not to protect the rotor
hypothesis.

## Implementation sequence

The Q8 bridge is now passed prospectively. On fresh seeds 10--19, raw spinor
training passes 8/10, while joint Q8-family retraction plus the previously
frozen label-free manifold-distance decoder gate passes 10/10: all 460 dense
and 40 long central-pair cells are 100%, with homomorphism RMS below `1.7e-7`.
This validates center-faithful spinor state as a real mechanism and validates
the discovery/retraction/observability pipeline. It does not validate Spin(8)
itself; the next gate starts at the generator algebra.

1. Construct and test the 28 positive- and negative-chirality 8x8 generators.
2. Verify `G_ij^T=-G_ij` and the so(8) commutation relations.
3. Implement recurrent Spin(8) action and an explicit affine-composition oracle.
4. Add full/chunk/token parity and long-prefix drift tests.
5. Add the Spin(8) family to the existing Q8 harness without changing its
   surrounding model or batches.
6. Benchmark exponential/Cayley/Givens construction before long training.
7. Run the multi-group, multi-seed suite and decide whether triality coupling is
   justified.

### 2026-08-03 gate status

Steps 1--4 now pass. The fixed octonion construction gives zero residual for
the Clifford relations, the full `so(8)` commutator table, and the triality
equivariance tensor; the recurrent affine reference scan agrees with serial
execution to `8.9e-16` in float64 and keeps a constant cache.

Step 5 now passes an untouched reliability cohort, with an important method
correction. The first frozen-decoder rank-four retraction passed 8/9 fresh
seeds; seed 4 showed that a shortest-word canonical section can occupy the
wrong decoder gauge. The final path-section compiler estimates all eight
endpoint states from disjoint L15/L16 path ensembles, projects the full Q8
regular family without a rank threshold, and transports the observer on the
reachable section. It passes 9/9 untouched seeds 10--18 at every dense and
L16,384 gate, with float32 homomorphism RMS below `7.4e-7` and no post-compile
gradient steps. It is still table-aware, not a blind discovery engine. See
`experiments/SPIN8_TRIALITY_ALGEBRA_RESULTS.md`,
`experiments/SPIN8_Q8_JOINT_RETRACTION_RESULTS.md`, and
`experiments/SPIN8_Q8_PATH_SECTION_VALIDATION_RESULTS.md`.

The table-aware limitation has since been reduced in two stages. A
decoder-labeled table-blind compiler passes 9/9 fresh seeds. The stricter
state-only compiler clusters recurrent states without logits or target labels,
but its prospectively frozen reliability gate fails at 7/9 because two seeds
fall below an arbitrary Euclidean separation floor. Structural follow-up finds
that both rejected states still support an exact replicated Q8 action at
`k=8`, alongside an exact `Q8/C4 ~= C2` quotient at `k=2`. The current frontier
was therefore no longer table recovery but metric action selection: infer the
largest stable action found by the scan and certify every other candidate as its quotient
before exposing the observer. See
`experiments/SPIN8_STATE_ONLY_COMPILER_RESULTS.md` and
`experiments/SPIN8_STATE_QUOTIENT_LATTICE_RESULTS.md`.

That frozen behavioral gate passed prospectively on Q8. The historically named
finest-congruence compiler scans `k=2..12`, selects `k=8` in all untouched
seeds 49--57 without passing state cardinality to K-means, and proves every
other action found by that scan is its exact quotient in two corpora. Seven seeds
also expose `Q8/C4 ~= C2`; three would fail the old separation floor but pass
the new algebraic certificate. All nine are 100% through L16384 with
homomorphism RMS below `7.2e-7`. This is decoder-free, cardinality-selected
metric compilation from endpoint-supervised states, not unsupervised learning.
The subsequent exact 4,140-partition audit finds the complete Q8 lattice
`{1:1, 2:3, 4:1, 8:1}` in all nine seeds and withdraws the earlier uniqueness
interpretation. See
`experiments/SPIN8_FINEST_CONGRUENCE_RESULTS.md`.

Two later oracle-removal gates isolate a broader shared-family principle.
First, joint diagonal-triality retraction recovers a completely hidden chiral
action from partial vector/positive endpoints in 10/10 seeds, while separately
orthogonal actions retain representation-specific slack. Second, joint
Sinkhorn retraction learns collision-free latent addresses from single-key
episodes in 10/10 seeds and transfers exactly to unseen mixed-key sequences
through length 2048. Independently normalized address rows fit training below
`5.6e-18` yet collide and fail in every seed; untrained joint rows pass 0/10.

The direct-slot joint control also passes 10/10, so the address result supports
joint relational normalization rather than a triality capacity advantage.
See `experiments/SPIN8_BLIND_SHARED_ACTION_RESULTS.md`,
`experiments/SPIN8_LATENT_ADDRESS_THEOREM.md`, and
`experiments/SPIN8_LEARNED_ADDRESS_RESULTS.md`.

The subsequent continuous-alias gate removes the logical key ID. Separate
linear write/query encoders see only fresh 24D aliases; training contains no
mixed-key recurrence. Joint unlabeled marginal balance passes 10/10 seeds on
untouched radius-0.35 aliases through length 2048, whereas independently
perfect encoders collide and fail 0/10. Direct slots again match triality
exactly. Oracle-projected delta memory passes 10/10, but learned delta keys and
additive fast weights pass 0/10 for distinct reasons (key-geometry error versus
missing overwrite correction). See
`experiments/SPIN8_CONTINUOUS_ALIAS_THEOREM.md` and
`experiments/SPIN8_CONTINUOUS_ALIAS_RESULTS.md`.

The combined blind-action/continuous-alias gate then trains both mechanisms in
one optimization run. The shared family passes 10/10; a parameter-richer
independent family fits all visible columns and calibration endpoints but fails
on the held-out negative subspace and direct long-horizon retrieval in 10/10.
The binding-mode independent control remains behaviorally exact because it
bypasses its learned negative action, while its triality residual reaches
0.721. This is retained as an explicit warning that retrieval alone cannot
certify latent geometric coherence. See
`experiments/SPIN8_BLIND_ALIAS_ACTION_PREREGISTRATION.md` and
`experiments/SPIN8_BLIND_ALIAS_ACTION_RESULTS.md`.

The next gate removed the remaining visible-matrix-column oracle. An exhaustive
differential audit found a sharp boundary: five generic transformed-state
pairs spanning at least two triality views have rank 28, whereas four such
pairs or five pairs in a single view have rank 25. The four-probe nullspace was
exponentiated into an exact alternative shared action that preserves all
visible endpoints while changing the hidden chiral action. Under the frozen
`(1 vector, 4 positive, 0 negative)` design, the shared family recovers the
entire unseen negative action and remains correct through length 2048 in 10/10
seeds. An independent 84-parameter family fits the same endpoints but retains
55 tangent directions of slack and collapses under long composition. The
combined preregistered causal-margin gate remains a documented 0/10 because its
one-step `0.05` margin was too strong, even though its long-horizon margin is
approximately one in every seed. See
`experiments/SPIN8_FIVE_PROBE_PREREGISTRATION.md` and
`experiments/SPIN8_FIVE_PROBE_RESULTS.md`.

The active triality-sensing gate is now complete. The endpoint information
matrix is exactly independent of the unknown action under left-invariant
coordinates, so local adaptive querying offers no advantage over a universal
design. Exhaustive numerical optimization selects a permutation of `(2,2,1)`
in every seed, with recurring `det(I)=81/1024` and `trace(I^-1)=43`. A hard
learned selector finds an identifying multiview sensor in all ten untouched
seeds but reaches the frozen near-oracle design gate in only 6/10 because four
seeds freeze into imbalanced allocation basins. It nevertheless beats matched
random sensing in all ten noisy tests and passes the practical recovery gate in
9/10; seed 11 misses only the frozen `0.02` oracle-gap threshold. See
`experiments/SPIN8_ACTIVE_SENSING_PREREGISTRATION.md` and
`experiments/SPIN8_ACTIVE_SENSING_RESULTS.md`.

The joint query-family continuation gate is now complete on untouched seeds
20--29. Soft learning followed by exhaustive joint retraction over all 243 view
assignments and continuous vector polish reaches the balanced optimum in 10/10
seeds, versus 6/10 for a fresh straight-through hard baseline. The frozen
conditioning-reliability gate therefore passes. The stricter requirement that
joint retraction strictly beat independent argmax in 8/10 seeds remains an
honest 4/10 failure: independent argmax was already optimal in six ceiling
cases. Joint retraction repaired all four actual opportunities and harmed none,
but that post-hoc 4/4 decomposition does not replace the preregistered gate.

The invariant audit also separates theorem from numerical recurrence. For any
unit triality query, `J^T J` is a rank-seven orthogonal projector onto the
tangent directions that move the queried vector. Consequently every
five-query information matrix has exact `trace(I)=35`; identifiability and
conditioning concern how those five fixed-trace projectors cover the 28
directions. The balanced sensor obeys the prospectively frozen factorization

\[
\chi_I(\lambda)=\frac{1}{1024}(\lambda-1)^4
(\lambda^2-3\lambda+1)
(2\lambda^2-6\lambda+3)^4
(2\lambda^2-4\lambda+1)^4
(2\lambda^3-8\lambda^2+6\lambda-1)^2,
\]

where adjacent factors are multiplied. It replicated in all ten fresh seeds
and implies `det(I)=81/1024`, `trace(I)=35`, and `trace(I^-1)=43`. The trace
identity is proved; the exact spectrum and determinant optimum are
prospectively replicated numerical results pending a symbolic global proof.
See `experiments/SPIN8_JOINT_SENSOR_RETRACTION_PREREGISTRATION.md` and
`experiments/SPIN8_JOINT_SENSOR_RETRACTION_RESULTS.md`.

The first part of that proof is now complete. Fixing the singleton query by
`Spin(8)` gauge identifies the other four probes with a four-frame in the common
eight-real representation of its `Spin(7)` stabilizer. The unique invariant
Cayley four-form supplies the remaining orbit coordinate `c` after the Gram
matrix is fixed. Exact symbolic elimination over the unit circle gives

\[
\det I_c=\frac{(1-c^2)^3(9-c^2)^2}{1024}.
\]

The derivative with respect to `c^2` is strictly negative on `[0,1)`. Hence the
Cayley-null orbit `c=0` is globally D-optimal among orthonormal balanced
sensors, with determinant `81/1024`; the calibrated Cayley endpoints
`c=+/-1` have rank 25. The maintained certificate proves the complete
one-parameter characteristic polynomial in exact rational arithmetic and
checks infinitesimal invariance under all 21 `Spin(7)` stabilizer generators.
This exceptional-geometric interpretation is consistent with the Cayley form's
unit comass and `Spin(7)` stabilizer established through triality by
[Katz and Shnider](https://arxiv.org/abs/0801.0283).

The global extension was attacked rather than assumed. Fresh seeds 30--39
reproduced the exact partition values `1/32`, `1/16`, `135/2048`, and
`81/1024` in every seed, with single-view rank 25. Across 10,000 random
balanced frames, none beat its row-orthonormal QR completion. Thirty-two
gradient adversaries explicitly maximizing the nonorthogonal advantage all
converged to equality, with maximum residual determinant advantage `5.42e-16`.
Two load-bearing one-correlation slices of the QR inequality are additionally
proved by exact factorization. These results do not turn the unrestricted QR
inequality into a theorem: a general invariant-polynomial or sum-of-squares
certificate is still required. See
`experiments/SPIN8_CAYLEY_SPECTRUM_PREREGISTRATION.md` and
`experiments/SPIN8_CAYLEY_SPECTRUM_RESULTS.md`.

The next mathematical target is therefore sharply isolated: write the balanced
determinant in terms of the four-frame Gram matrix and Cayley invariant, prove
that row-orthonormal completion cannot reduce it, then establish the exact
upper bounds for the other four allocation partitions. Only those two steps
would promote `81/1024` from an orthonormal-orbit theorem plus exhaustive
falsification to a global five-query theorem.
