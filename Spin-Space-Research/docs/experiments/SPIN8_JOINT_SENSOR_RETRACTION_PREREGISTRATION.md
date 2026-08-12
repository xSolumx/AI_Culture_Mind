# Spin(8) Joint Sensor Retraction Preregistration

**Frozen:** 2026-08-03, before any joint-continuation result or seed 20-29
sensor was evaluated.

## Questions

1. Can soft continuation followed by joint family retraction eliminate the
   discrete allocation traps that limited hard straight-through sensing to
   6/10 strict D-optimal passes?
2. Does the balanced five-query oracle obey one exact information-spectrum law
   across a fresh cohort?

## Part I: joint late retraction

The previous hard selector rounded every query independently throughout
optimization. Four of ten policies found rank-28 designs but froze into
imbalanced allocations. The proposed continuation maintains one soft
representation distribution and one candidate vector per representation for
each of five queries. It optimizes only the same regularized D-optimal
objective.

At the end, all `3^5 = 243` physical hard assignments are evaluated using the
learned vector bank. The complete five-query assignment is selected jointly by
unregularized hard-design log determinant. This is a family retraction, not
five independent argmax operations.

No balance, diversity, rank, allocation, or oracle-vector target is allowed in
the training loss or retraction rule.

## Frozen variants

| Variant | Description |
|---|---|
| `hard_straight_through` | previous hard-forward/soft-backward policy, fresh seed |
| `soft_independent_argmax` | soft continuation, then independent query-wise argmax |
| `soft_joint_retracted` | same soft continuation, joint best hard assignment over 243 possibilities |
| `soft_joint_polished` | joint assignment followed by equal-budget continuous vector optimization with views frozen |
| `oracle_doptimal` | all 21 allocations, four continuous restarts each |
| `random_mixed` | matched random identifiable sensor |

The independent and joint rows share the identical learned soft checkpoint and
vector bank. The polished row may change vectors but never its jointly selected
views.

## Frozen cohort and optimization

- development seed: 1, used only to verify implementation and set the soft
  temperature schedule;
- untouched reliability seeds: 20-29;
- five physical queries for every reported sensor;
- CPU float64 initialization, CUDA float64 optimization;
- unchanged D-optimal ridge `1e-7` during continuation;
- hard unregularized information metrics after retraction;
- noisy endpoint standard deviation `1e-3`;
- dense composition lengths `16,32,64,128,256,512,1024,2048`.

## Frozen joint-retraction gates

### A. Retraction validity

For every seed:

- joint selection evaluates exactly 243 assignments;
- selected views contain at least two triality representations;
- selected hard design has rank 28;
- the selected assignment has maximal log determinant among all 243 hard
  assignments formed from that learned vector bank.

### B. Conditioning reliability

At least 8/10 untouched seeds must satisfy, after hard-view continuous polish:

- allocation is a permutation of `(2,2,1)`;
- log-determinant gap to oracle at most `0.10`;
- `trace(I^-1)` no more than 10% above oracle.

The polished joint pass count must exceed the fresh hard straight-through pass
count by at least two seeds. Otherwise the intervention has not resolved the
identified failure mode.

### C. Causal retraction comparison

In at least 8/10 seeds, joint retraction before polish must strictly improve
both log determinant and `trace(I^-1)` over independent argmax from the same
soft checkpoint. This isolates the retraction rule from continuous training.

### D. Noisy action recovery

At least 8/10 polished joint sensors must:

- beat matched random sensing in least-observed-view one-step cosine;
- beat matched random sensing at worst-representation length-2,048 cosine;
- remain within `0.02` of the oracle at length 2,048;
- preserve triality equivariance below `1e-8`, scan parity below `1e-9`, and
  absolute log-norm drift below `1e-5`.

## Part II: exact spectral signature

For any unit query in any one of the three triality representations, define

\[
P_{r,x}=J_{r,x}^{\mathsf T}J_{r,x}.
\]

The first analytic claim is that `P_{r,x}` is a rank-seven orthogonal
projector. Therefore every five-query information matrix has

\[
\operatorname{tr}(I)=5\cdot7=35,
\]

independently of design quality.

The preceding oracle cohort suggested the following exact characteristic
polynomial for the balanced optimum:

\[
\chi_I(\lambda)=\frac{1}{1024}
(\lambda-1)^4
(\lambda^2-3\lambda+1)
(2\lambda^2-6\lambda+3)^4
(2\lambda^2-4\lambda+1)^4
(2\lambda^3-8\lambda^2+6\lambda-1)^2.
\]

If exact, it implies

\[
\det(I)=\frac{81}{1024},\qquad
\operatorname{tr}(I)=35,\qquad
\operatorname{tr}(I^{-1})=43.
\]

This factorization was discovered after the seed 10-19 active-sensing cohort
and is therefore prospectively tested only on seeds 20-29.

## Frozen spectral gates

### E. Single-query projector theorem

Across all three representations and 100 deterministic random unit probes:

- `||P^2-P||_max < 1e-12`;
- rank exactly 7;
- trace error from 7 below `1e-12`.

### F. Fresh-oracle polynomial replication

For every seed 20-29 oracle optimum:

- allocation is a permutation of `(2,2,1)`;
- maximum absolute evaluation of the exact factor polynomial on any
  information eigenvalue below `1e-10`;
- relative characteristic-coefficient error below `1e-10`;
- determinant error from `81/1024` below `1e-10`;
- trace error from 35 below `1e-10`;
- inverse-trace error from 43 below `1e-10`.

## Interpretation boundaries

- Passing E proves `trace(I)=35` for every five-query design.
- Passing F would establish a reproducible exact algebraic signature for the
  numerical oracle family. It would not by itself prove that no other physical
  design has larger determinant.
- Passing A-D would show that joint late retraction repairs the observed
  optimization trap without being told the balanced allocation.
- Exhaustively choosing among 243 assignments is feasible here because the
  query budget is five. A scalable relaxation remains necessary for large
  query families.
- No language-model or semantic-memory claim is authorized by this gate.
