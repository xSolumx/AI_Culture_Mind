# Validity audit of `FOUNDATIONS.md`

Date: 2026-08-06

> **Historical audit boundary.** This document audits the pre-pure maintained
> implementation as it existed immediately before the same-day rewrite. The
> observations below remain the correction record for that code and its frozen
> artifacts; they are not a validity audit of `pure_rotor_ssm/`. The current
> contract and its new tests are documented in `FOUNDATIONS.md` and
> `pure_rotor_ssm/CONTRACT.md`.

> **Documentation reconciliation, 2026-08-16T16:05:27+02:00.** This historical
> verdict and its frozen measurements were retained unchanged. Current work
> adds a tested PyTorch-only Schur execution path and a one-step Pure Rotor /
> identity / Mamba-2 smoke artifact; neither updates the audit's empirical
> verdict or establishes model superiority. See
> `experiments/PURE_ROTOR_VS_MAMBA2_BENCHMARK.md`.

> **Later status.** The 2026-08-10 Spin(8)/Spin(9) hierarchical-memory report
> closes separate routing and delta-transport engineering gates. It neither
> changes this historical validity verdict nor establishes that the maintained
> rotor model implements the new memory compiler.

> **Separate later model.** `pure_spin8_ssm/` was added at v1.0 on 2026-08-16
> alongside Pure Rotor, with independent tests and checkpoints. It was not the
> subject of this historical audit. Its current boundary is recorded in
> `pure_spin8_ssm/CONTRACT.md`; v1.1 adds the separately tested frozen-token
> compiler and a separately frozen noisy continuous-router cohort. Neither
> alters this historical audit's verdict; current evidence and limitations are
> recorded in `experiments/PURE_SPIN8_CONTINUOUS_OBSERVATION_RESULTS.md`.

## Verdict

The central mathematical object is valid: `ga_ssm.py` and
`rotor_ssm_torch.py` implement a causal, channel-diagonal, noncommutative
rotor-affine recurrence with an exact associative composition law, fixed-size
recurrent state, and a real-arithmetic BIBO bound. The original note was not
fully valid as an implementation contract. It omitted floating-point and
autograd assumptions, overstated training-time equivariance, described
nominal rather than effective parameter matching, and blurred the maintained
Cl(3) model with later research modules.

One direct contradiction was found in code: the JAX bivector exponential had
the correct value at the identity but a NaN Jacobian there. Thus the old claim
that both maintained controllers receive a nonzero identity-tangent gradient
was false for JAX. The chart is now evaluated by a smooth small-angle expansion
in both backends, and the zero-input Jacobians agree exactly in float64.

The corrected note is defensible if its uses of “exact” are read with the new
finite-real-arithmetic qualification. Its empirical sections remain evidence
from small controlled experiments, not architectural theorems or language-
modeling superiority claims.

## Classification

- **E — exact:** follows from the stated equations over finite real numbers.
- **A — assumption-dependent:** correct only with assumptions now made explicit.
- **N — numerical:** verified to a tolerance, not an exact implementation fact.
- **H — empirical hypothesis/result:** supported by frozen reports but not a
  mathematical consequence of the recurrence.
- **C — corrected:** the prior wording or implementation was false or too broad.

## Line-by-line adjudication of the corrected note

| Note lines | Class | Adjudication |
|---:|:---:|---|
| 3–12 | C | The opening formerly implied that all 400+ lines described only the two maintained files. Later SchurScan, finite-group, Q8, and Spin(8) sections are a research ledger over separate modules. |
| 16–35 | E | The displayed transition matches both maintained implementations: invariant controls, positive step/rate parameterization, radial bivector chart, complementary write, and rotor sandwich. `EquivariantBivector` is concretely `GradeLinear(...)[...,4:7]` times an invariant scalar strength. |
| 37–39 | E | This is the actual chart: if `r=||B||`, the physical bivector angle is `theta_max*tanh(r)` in the direction of `B`, with analytic zero limit. |
| 41–50 | E/A | Unit even rotors act orthogonally by conjugation on Cl(3,0), so coefficient norm is preserved. Strict decay follows for finite real controls and positive floors. It is not literally strict in every floating dtype: float16 evaluates `exp(-1e-6)` as `1.0`, while sufficiently large finite exponents can underflow to zero. |
| 52–65 | E | The one-step inequality and geometric-series BIBO bound are correct. They require a uniform bound on the actual drive, not just an assertion that tokens are bounded. |
| 67–77 | A/N | With fixed finite projection weights, bounded inputs give bounded drive. Numerical rotor drift and decay rounding are implementation effects, so the guarantee is an exact-model contract plus measured floating error. |
| 79–86 | C/E | Half-life initialization is correct for the zero controllers. The JAX identity-tangent gradient was previously NaN and is now fixed. Raw four-parameter normalization has no continuous value at the zero vector; the new identity fallback is a deliberate discrete convention. |
| 90–106 | E | The transition-composition formula is correct in chronological order: `q_b q_a` and `u_b + d_b Ad(q_b)u_a`. No commutativity assumption is used. |
| 108–114 | E/N | Associativity is exact because this is function composition. JAX's prefix tree and the recurrence use different floating groupings; parity is approximate. The current test covers full, chunked, and token streaming in evaluation mode. |
| 116–123 | E/A | Each layer caches one `(C,8)` state, hence `8LC` numeric scalars per sequence. PyTorch autograd history grows with context unless streaming inference uses `no_grad` or detaches the cache. Vocabulary projection remains `O(V)` per generated token. |
| 127–148 | C/E/N | The deterministic block is Spin(3)-equivariant. `GradeLinear`, invariant controls, rotor prediction, RMS norm, gates, and residuals commute with proper rotor conjugation. Elementwise training dropout does not: measured maximum errors were `2.24` (JAX) and `1.45` (PyTorch) under the same masks. The test only establishes the dropout-free/evaluation claim. |
| 152–176 | E/C | `Cl(3,0)` under proper conjugation is two trivial plus two standard 3D copies. The linear commutant has `8CD` rather than `4CD` weights. `Spin3IsotypicLinear` realizes it, but neither maintained SSM block currently instantiates that layer. “At any depth” is valid for compositions of the old linear family, not an unrestricted claim about arbitrary nonlinear networks. |
| 178–190 | E/N | The real-type Schur-factored transition family is closed under composition. The checked report records `8.88e-16` float64 scan/recurrent error. Complex- and quaternionic-type multiplicity algebras remain unimplemented. |
| 194–220 | H/C | All losses, parameter counts, hashes, hardware, and protocol fields agree with the three checked JSON reports. The mean difference is `0.0677699` nats = `0.09777` bits/byte = `2.40%` of identity loss. The identity variant has the same raw count but dead rotor-path parameters, and nonzero learned angles show activity rather than causal benefit. Three paired seeds do not establish significance. |
| 224–249 | E/A | Independent unit rotors need not form one coherent token algebra. The search/compile/retract construction is valid for the tested real-type finite-group irreps. The earlier generic finite-group eigenspace wording omitted Schur-type degeneracies. |
| 251–270 | H/A | The A5 and table-blind results agree with their frozen reports. Their supervision remains strong: the first supplies the Cayley structure, while the second supplies exact prefix classes dense enough to reconstruct it. Neither is unsupervised language discovery. |
| 274–318 | E/H | Given a known inverse-token involution and a reverse-edge cover, missing directions are forced. The 120/240 sampled success, adversarial ambiguity, 121/240 protocol threshold, uniform-mask failures, and 144 gauge/order checks agree with the exact-half result and theorem reports. The 121 threshold is specific to the four-token matching protocol, not a universal finite-group theorem. |
| 322–341 | H/C | The information, gradient, curriculum, and endpoint-manifold numbers match the reports. They support an optimization-barrier interpretation but do not uniquely prove it or rule out other mechanisms. Exact anonymous endpoint classes and group order remain supplied. |
| 345–359 | E | Sandwich actions quotient the spin center because `Ad(R)=Ad(-R)`. Q8 conjugation has four actions, left multiplication has eight, fixed word parity reaches four states, and `rank(I-A)=4` rules out a two-reflection capable baseline for the faithful generator. |
| 361–376 | H | The reported seed/cohort, retraction, accuracy, and homomorphism figures are artifact-consistent. They are controlled finite-group results, not evidence that a left-spinor language model is superior. |
| 380–404 | E/H | The right-congruence/quotient-automaton statement is exact. The Q8 two- and eight-state observations and `{1:1,2:3,4:1,8:1}` lattice histogram are empirical/exact-enumeration results for the recovered actions. Metric clustering does not enumerate all congruences by itself. |
| 408–420 | C/E | The triality bilinear map is unique only up to scale before normalization. With the chosen normalization and a known unit key, single-pair inversion is exact. The tight-frame interference bound `(K-H)/H` follows from the frame-potential inequality after subtracting self terms and averaging over `K` items. |
| 422–438 | E/H | The shared-action addressed recurrence is closed under affine composition because diagonal multiplicity retentions commute with the shared 8D action. Hard overwrite and `8H` numeric state are exact API properties. Rank-deficient extrapolation is controlled numerical evidence, not a theorem that symmetry always improves sample efficiency. |
| 440–462 | A/C | These were missing implementation boundaries: nonempty sequences, full-GA state versus even rotors, nondifferentiable grade norms at zero, contracting state Jacobians, distinct backend initialization/RNG/dtype behavior, and propagation of nonfinite values. |
| 464–489 | H/A | The remaining-open list is honest after qualifying the “stationary variance” convention and changing “cannot be bitwise equal” to “not guaranteed.” The proposed baselines and larger runs are future validation, not current evidence. |

## Derivations of the structural claims

### 1. Rotor norm preservation

Write a Cl(3,0) rotor as `q = a + B`, where `B` is bivector-valued and the
exponential chart gives `q reverse(q)=1`. Rotor conjugation preserves each
grade and restricts to an orthogonal 3D rotation on vectors. In three Euclidean
dimensions the bivector grade is its Hodge-dual standard copy, while scalar and
pseudoscalar coefficients are fixed. Therefore, for coefficient norm

```text
||h||_coef^2 = h_0^2 + ||h_1||^2 + ||h_2||^2 + h_3^2,
```

each summand is preserved and `||q h reverse(q)||_coef = ||h||_coef`. This is a
full-GA statement; it does not require the state itself to be even.

### 2. Strict contraction and bounded input

For finite real controls,

```text
Delta_t >= Delta_min > 0
lambda_c >= lambda_min > 0
d_t = exp(-Delta_t lambda_c) <= exp(-Delta_min lambda_min) = d_max < 1.
```

Using the rotor isometry and the triangle inequality,

```text
||h_t|| <= d_max ||h_(t-1)|| + ||u_t||.
```

If `||u_t|| <= U`, induction gives

```text
||h_t|| <= d_max^t ||h_0|| + U sum_(k=0)^(t-1) d_max^k
        = d_max^t ||h_0|| + U(1-d_max^t)/(1-d_max).
```

The targeted persistent-drive test used `d=.999`, `U=.1`, and 20,000 steps.
The state reached `99.9999997959` against the limiting bound `100`, with only
`7.96e-13` maximum numerical overshoot of the finite-time formula.

### 3. Ordered transition composition

For `T_a(h)=d_a Ad(q_a)h+u_a` and `T_b` applied later,

```text
T_b(T_a(h))
= d_b d_a Ad(q_b q_a)h + u_b + d_b Ad(q_b)u_a.
```

This proves closure and the implemented order. Associativity follows from
function composition, not from commuting token actions. Float64 three-way
composition differed by `7.77e-16`; float32 differed by `2.98e-7`, directly
exhibiting floating non-associativity without invalidating the exact law.

### 4. State and gradient stability are different claims

Because controls depend on the current input rather than the previous state,

```text
d h_t / d h_(t-1) = d_t Ad(q_t),
```

whose exact operator norm is `d_t`. A state-path gradient over `k` steps is at
most the product of those decays and can vanish for long horizons. Parameter
gradients also include sums of drive/controller derivatives and are not bounded
merely by the BIBO state proof. The residual block supplies a separate identity
gradient path, but that is an architectural mitigation rather than a theorem of
well-conditioned training.

## Numerical and edge-case results

All targeted checks used the current checkout after the small-angle correction.

| Check | Result |
|---|---:|
| JAX/PyTorch float64 geometric product maximum difference | `8.88e-16` |
| JAX/PyTorch float64 rotor maximum difference | `2.78e-16` |
| JAX/PyTorch float64 sandwich maximum difference | `1.55e-15` |
| Rotor sandwich coefficient-norm error | JAX `8.88e-16`; PyTorch `1.33e-15` |
| Zero-bivector Jacobian | finite in both; backend difference `0` |
| Initialized JAX rotor-controller gradient | finite, L1 `5.129864` |
| Float32 transition associativity residual | `2.98e-7` |
| Float64 transition associativity residual | `7.77e-16` |
| JAX parallel/recurrent maximum difference, L4096 | `2.86e-6` |
| Even-input sandwich odd-grade leakage | exactly `0` in the tested tensor path |
| Float16 `exp(-1e-6)` | `1.0` (strict contraction rounded away) |
| Empty scans | explicit `ValueError` after correction |
| Full/chunk/token state-cache equivalence | passes in maintained JAX and PyTorch tests |

### Zero norms

- `rotor_from_bivector(0)` is now the identity with analytic tangent
  `-(theta_max/2) I` in bivector coordinates.
- `normalized_rotor(0)` cannot be continuously defined. Both maintained
  utilities now choose the identity instead of returning the zero
  multivector, so the function's advertised unit-output postcondition holds.
- Vector and bivector norms in `grade_invariants` are genuinely
  nondifferentiable at zero. PyTorch currently selects a zero subgradient;
  JAX can produce NaNs when differentiating the raw norm at that point. This
  was documented rather than silently changing trained feature semantics from
  norms to squared norms.

### Full GA versus even closure

Rotors occupy components `[1,e12,e13,e23]`, and rotor products remain even.
The recurrent state is nevertheless eight-dimensional. Arbitrary embeddings
and `GradeLinear` drives contain scalar, vector, bivector, and pseudoscalar
components. If the initial state and drives are even, the recurrence stays
even; the maintained language model does not impose those premises.

### Streaming semantics

The cache is the final post-transition SSM state for each layer, before that
layer's residual/FFN output is discarded. Passing it into the next chunk is
semantically sufficient because normalization and feed-forward operations are
token-local. The equivalence claim assumes evaluation mode. Training dropout
changes masks across calls, and an attached PyTorch autograd graph makes live
memory grow unless the cache is detached.

### JAX/PyTorch parity

The algebra, bounded rotor chart, decay/write equations, recurrence, residual
block, and tied decoder are mathematically aligned. Full models are not
drop-in numerical replicas: Flax and PyTorch use different parameter layouts
and initializers; JAX defaults to a parallel tree while PyTorch loops; dropout
RNGs differ; and JAX explicitly returns float32 logits. Primitive/recurrence
parity is established, while equality of separately initialized full-model
logits is neither expected nor claimed.

## Associated SpinorModel boundary

The historical `SpinorModel/geometric_layers.py` uses the same Cl(3,0) basis
and a correct geometric-product table, but its arbitrary left-multivector
operators, componentwise GELU, and ordinary LayerNorm do not satisfy the
Spin(3)-equivariant contract in `FOUNDATIONS.md`. It is a separate historical
Transformer path.

`SpinorModel/overhauled` is a second persistent rotor-affine implementation,
but it is not equation-identical to the maintained pair named by the note. Its
drive is

```text
(1-d) * sigmoid(write_control) * candidate,
```

rather than `sqrt(1-d^2) * projected_input`; it also implements identity
padding transitions and an O(L log L)-work Hillis-Steele reference scan. Its
tests support its own convex-recursion bound and streaming contract. Results
from one write convention must not be cited as direct evidence for the other.

## Artifact audit and limits

The checked reports support the values repeated by the note:

- `experiments/final_seed0_300.json` through `final_seed2_300.json` contain the
  stated losses, 22,968 raw parameters, RTX 2070 SUPER device, data hashes, and
  transition diagnostics.
- `experiments/SPIN3_ISOTYPIC_SCHUR_SCAN_RESULTS.md` records centralizer rank
  8, old-family rank 4, and `8.88e-16` scan parity.
- `experiments/SELF_COMPILING_RETRACTION_RESULTS.md`,
  `LATENT_CAYLEY_RETRACTION_RESULTS.md`, `INVERSE_COVER_EXACT_HALF_RESULT.md`,
  endpoint reports, Q8 reports, congruence reports, and Spin(8) reports contain
  the cohort counts and thresholds summarized in the note.

This audit checked those claims against code, reports, JSON artifacts, and the
focused test suites. It did not rerun every expensive historical GPU cohort
from raw data. Artifact consistency is therefore not independent replication.
The structural derivations and new targeted numerical tests are independently
recomputed in the current checkout.

## Narrow corrections made

1. Replaced the JAX zero-norm bivector chart with an analytic small-angle
   evaluation and mirrored it in PyTorch.
2. Made raw zero quaternion parameters fall back to the identity rotor instead
   of the zero multivector.
3. Added zero-Jacobian, zero-normalization, and empty-sequence regression tests.
4. Made empty scan rejection explicit rather than leaking an `IndexError`.
5. Corrected `FOUNDATIONS.md` qualifications without rewriting or deleting its
   historical research ledger.
