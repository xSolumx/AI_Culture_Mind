# Spin(8) Dirac--Gram Proof Gate Preregistration

**Frozen:** 2026-08-03, after the exact Cayley-spectrum theorem and after an
exploratory search exposed the stronger Gram-volume inequality, but before the
fresh random and gradient-adversarial cohorts specified below.

## Why this gate exists

The Cayley-spectrum theorem proves D-optimality only after the four balanced
probes have been made orthonormal. The remaining question is whether a general
unit four-frame can gain information by retaining correlations.

The old target was merely

\[
\det I(X)\leq \det I(Q),
\]

where `G=XX^T` and `Q=G^{-1/2}X` is any row-orthonormal completion. Exploratory
work exposed a sharper candidate:

\[
\boxed{\det I(X)\leq \det(G)^3\det I(Q).}
\tag{DG}
\]

For unit rows, Hadamard's inequality gives `det(G)<=1`, so `(DG)` implies the
old QR lemma and makes equality possible only on the orthonormal locus.

This is a conjecture until an analytic certificate covers the full six-
correlation Gram domain. Numerical success must not be called a proof.

## Invariant form

Let `Phi(X)` be the Spin(7)-invariant Cayley four-form and
`Delta=det(G)`. Row whitening preserves the oriented four-plane, hence its
normalized Cayley coordinate

\[
c=\frac{\Phi(X)}{\sqrt{\Delta}}.
\]

The proved orthonormal theorem gives

\[
\det I(Q)=\frac{(1-c^2)^3(9-c^2)^2}{1024}.
\]

Consequently `(DG)` is equivalent, for `Delta>0`, to

\[
1024\,\Delta^2\det I(X)
\leq
(\Delta-\Phi^2)^3(9\Delta-\Phi^2)^2.
\tag{IG}
\]

This polynomial-looking invariant inequality is the primary theorem target.

## Exact analytic requirements

The harness must prove in exact arithmetic that the strengthened inequality
holds on both previously derived one-correlation slices. After removing the
manifest nonnegative factors `u(1-u)^3(1-z)^3`, the residual bivariate
polynomials must be represented in the degree `(2,2)` Bernstein basis on the
unit square. Every Bernstein coefficient must be strictly positive. This is a
complete proof on those slices, not sampling.

The harness must also verify exactly that:

- every single-query information block is a rank-seven orthogonal projector;
- same-view projector overlap is `1+6<x,y>^2`;
- different-view projector overlap is the constant `7/4`.

These identities are structural diagnostics. They do not by themselves prove
`(DG)`.

## Fresh falsifiers

The exploratory frames and optimizer states used to discover `(DG)` are not
part of the confirmatory cohort.

### Random Gram-volume attack

- deterministic seed: `20260805`;
- at least `1,000,000` full-rank unit four-frames;
- dense float64 evaluation in bounded chunks;
- failure: any log-ratio
  `logdet(I(X))-logdet(I(Q))-3 logdet(G) > 1e-9`.

The report must preserve the largest observed ratio and the corresponding Gram
determinant. A pass is counterexample-search evidence only.

### Gradient adversary

- deterministic seed: `20260806`;
- at least 64 independent starts;
- at least 2,000 Adam steps;
- objective: maximize the same regularized log-ratio;
- final unregularized failure tolerance: `1e-8`.

Convergence toward the orthonormal equality manifold is evidence for the
conjecture, not proof.

### Whitening-flow attack

On the product of four unit spheres, define the projected negative
frame-potential flow

\[
\dot x_i=-\sum_{j\ne i}\langle x_i,x_j\rangle x_j
+x_i\sum_{j\ne i}\langle x_i,x_j\rangle^2.
\]

The proposed Lyapunov statement is

\[
\frac{d}{dt}\log\det I(X(t))\geq0,
\]

with equality on the orthonormal locus. The fresh harness will attack the
normalized derivative over random frames and by gradient minimization. Any
negative derivative below `-1e-8` falsifies this route. A numerical pass does
not prove global monotonicity.

## Explicitly rejected shortcut

A simple eigenvalue-majorization proof is not admissible: exploratory testing
already found many frames for which the QR spectrum does not majorize the raw
spectrum. Likewise, the approximate-design Kiefer--Wolfowitz sensitivity
certificate does not apply to this exact five-probe problem and is violated by
the candidate design. Neither failed route may be reused as if proved.

## Promotion rule

The unrestricted global theorem may be claimed only after one of:

1. an exact factorization or sum-of-squares certificate of `(IG)` over the
   complete feasible Gram--Cayley domain; or
2. an exact proof that the whitening flow is globally determinant-monotone and
   converges to the orthonormal orbit.

Until then, the exact result remains two analytic slices plus fresh global
falsification evidence.
