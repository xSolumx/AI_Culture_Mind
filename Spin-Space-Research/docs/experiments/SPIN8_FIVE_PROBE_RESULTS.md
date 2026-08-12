# Spin(8) Five-Probe Identifiability Results

**Date:** 2026-08-03
**Preregistration:** `SPIN8_FIVE_PROBE_PREREGISTRATION.md`
**Raw artifact:** `spin8_five_probe_seeds0_9.json`
**Artifact SHA-256:** `75a79d8189aabf1a9ba414d0dea259f3656f1e82f9db38c4905725d3a95909cf`

## Result in one sentence

Five generic transformed-state probes spanning two Spin(8) triality views were
necessary and sufficient for local identification of a shared 28-dimensional
token action in this design, and SGD recovered the completely unobserved third
action through length 2,048 in 10/10 seeds; four probes retained an exact
three-dimensional stabilizer and independently normalized actions retained 55
unconstrained directions.

## Frozen gate outcomes

| Gate | Outcome |
|---|---:|
| Differential rank | **10/10 pass** |
| Exact four-probe ambiguity witness | **10/10 pass** |
| Every learned family fits visible endpoints | **10/10 pass** |
| Five-mixed-probe hidden completion | **10/10 pass** |
| Preregistered combined causal-margin gate | **0/10 fail** |

The last row is not a contradiction. The combined gate required both a `0.05`
one-step hidden-action margin and a `0.05` length-2,048 margin over every
underidentified control. The observed one-step margins were only
`0.0112–0.0289`, because four-probe and single-view solutions can be locally
close to the teacher. The length-2,048 margins were `0.9837–1.0335`: their
unfixed stabilizers become decisive under long noncommuting composition. The
pre-registered combined gate therefore remains recorded as failed rather than
being relaxed after inspection.

## Sharp differential boundary

The exhaustive allocation audit at the identity reproduced:

| Total generic probes | Rank, one triality view | Rank, at least two views | Residual stabilizer dimension, mixed view |
|---:|---:|---:|---:|
| 1 | 7 | — | 21 |
| 2 | 13 | 14 | 14 |
| 3 | 18 | 20 | 8 |
| 4 | 22 | 25 | 3 |
| 5 | 25 | 28 | 0 |

For one representation, `k` generic fixed vectors leave an `SO(8-k)`
stabilizer, hence

\[
\operatorname{rank}=28-\binom{8-k}{2}.
\]

Across distinct triality views, the observed nullities follow

\[
21,14,8,3,0,
\]

matching the connected-stabilizer chain

\[
\operatorname{Spin}(7)\supset G_2\supset SU(3)\supset SU(2)\supset\{e\}.
\]

The critical frozen allocations were exact in all ten seeds:

| Family | Allocation `(V,S+,S-)` | Rank | Parameter space | Slack |
|---|---:|---:|---:|---:|
| shared four mixed | `(1,3,0)` | 25 | 28 | 3 |
| shared five mixed | `(1,4,0)` | 28 | 28 | 0 |
| shared five single | `(5,0,0)` | 25 | 28 | 3 |
| independent five mixed | `(1,4,0)` | 29 | 84 | 55 |

Thus the gain is neither the raw number of observed vectors nor parameter
count. It is joint retraction of evidence from distinct triality views onto one
shared action family.

## Constructive necessity, not optimizer folklore

For each seed, the harness extracted a nonzero tangent from the exact
three-dimensional kernel of the four-probe Jacobian and exponentiated it. Right
composition by this stabilizer produced a second valid shared Spin(8) family
with:

- maximum visible endpoint difference at most `8.88e-16`;
- hidden negative-spinor mean cosine only `0.87897–0.88157`;
- triality-equivariance error at floating-point precision.

This proves that four probes cannot identify the hidden action for the frozen
design, independently of whether an optimizer happens to choose a favorable
representative.

## Optimization and generalization

All four learned variants fit their supplied evidence:

| Variant | Worst visible MSE over ten seeds |
|---|---:|
| shared five mixed | `2.45e-13` |
| shared four mixed | `1.18e-13` |
| shared five single | `3.53e-14` |
| independent five mixed | `3.94e-14` |

Only the identifiable shared family recovered the withheld `S-` action:

| Variant | Hidden `S-` one-step mean cosine range | Worst-representation L2,048 mean cosine range |
|---|---:|---:|
| shared five mixed | `0.999999999966–0.99999999999998` | `0.999999845–0.9999999999` |
| shared four mixed | `0.97107–0.98877` | `-0.04526–0.00238` |
| shared five single | `0.96297–0.98839` | `-0.04933–-0.00199` |
| independent five mixed | `0.86742–0.90483` | `-0.05966–0.01630` |

The controls are important: all of them interpolate the visible endpoint data,
and two appear quite good under a one-step cosine. Long noncommuting words
amplify their unidentifiable stabilizer component until their orientation is
effectively unrelated to the oracle. The identifiable family remains correct
through every dense length from 16 to 2,048.

## Numerical contracts

- shared-five triality-equivariance maximum error: `1.44e-15`;
- maximum parallel-prefix/recurrent error over every learned family: `1.29e-14`;
- maximum absolute log-norm drift through length 2,048: `4.21e-13`;
- minimum shared-five dense mean cosine: `0.9999998448`.

## What is established

1. Five generic state/action examples across at least two triality views are a
   sharp local identifiability boundary for this real Spin(8) realization.
2. Triality sharing completes an entirely unobserved chiral action from those
   five examples; independent action families cannot.
3. Four probes are exactly ambiguous, with a constructible `SU(2)` tangent
   stabilizer rather than merely a poorly optimized solution.
4. One-step interpolation is an unsafe mechanism metric. Long noncommuting
   composition exposes geometrically small but structurally unconstrained
   errors.

## What is not established

- The experiment establishes generic local identifiability. It does not yet
  eliminate every possible global discrete ambiguity.
- The five-probe number is scoped to three eight-dimensional triality
  representations and generic probes; it is not a universal sample-complexity
  theorem for arbitrary groups or observations.
- The preregistered combined causal-margin gate failed, even though its
  long-horizon component separated every seed.
- Semantic alias inference, learned probe selection, and language modeling were
  not tested here.

### Exact global addendum (2026-08-06)

A subsequent deterministic certificate closes the global-discrete gap for one
explicit `(1,4,0)` tuple. The triality closure of
`(e0^V; e0^+, e1^+, e2^+, e4^+)` has exact rank `(8,8,8)` with basis
determinant `-1` in every representation, so its full global stabilizer is
trivial. Removing `e4^+` leaves three exact generators closing as `su(2)`.

See [the exact global five-probe theorem](SPIN8_GLOBAL_FIVE_PROBE_THEOREM.md).
The subsequent
[continuous orbit theorem](SPIN8_CONTINUOUS_PROBE_ORBIT_THEOREM.md) uses exact
allocation-specific invariants and the compact principal-orbit theorem to
prove universal four-probe insufficiency and generic global identifiability in
every mixed five-probe allocation. The original numerical cohort remains
evidence rather than part of that exact proof.

## Next gate

The strongest next experiment is **active triality sensing**: let a model choose
which representation and probe state to query under a fixed five-query budget,
without knowing the teacher action. Compare a learned information-gain policy
against random mixed probes, single-view probes, and an oracle D-optimal design.
The scientific question becomes whether a scan-compatible system can discover
the minimal identifying experiment, not merely solve it after the probes are
supplied.
