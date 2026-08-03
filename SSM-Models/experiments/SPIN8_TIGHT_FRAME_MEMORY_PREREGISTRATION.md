# Spin(8) tight-frame memory preregistration

Date frozen: 2026-08-03, after the coded-memory gate and before tight-frame
results.

## Question

When there are more associations than multiplicity channels, can the code
geometry attain the smallest average crosstalk permitted by linear algebra
without changing the Spin(8) triality primitive or the associative scan?

## Bound

Let (C\in\mathbb R^{H\times K}) have unit-norm columns and let
(G=C^TC).  For independent isotropic values, the query-averaged expected
relative squared error is

[
\frac{1}{K}\sum_q\sum_{k\ne q}G_{qk}^2
=\frac{\lVert G\rVert_F^2-K}{K}.
]

Since ({\rm rank}(G)\le H) and ({\rm tr}(G)=K),

[
\lVert G\rVert_F^2\ge K^2/H,
qquad
\mathbb E[\mathrm{relative\ squared\ error}]
\ge \max(0,(K-H)/H).
]

Unit-norm tight frames attain equality.  This is the Welch frame-potential
bound, not a new theorem.

## Constructions

- (K\le H): orthonormal Walsh columns, exact retrieval.
- power-of-two (K>H): the first (H) rows of a (K\times K) Walsh matrix,
  normalized to unit columns.
- (K=H+1): a regular simplex frame, which also distributes interference
  uniformly with pairwise inner product (-1/H).

## Frozen gates

- empirical mean relative squared error within 15% of
  (max(0,(K-H)/H)) for every nonzero tight-frame cell;
- maximum exact relative error below (10^{-10}) for (K\le H);
- for every tested (K>H>1), tight-frame error is lower than the matched
  random-code error;
- the existing Spin(8) transport and parallel/recurrent gates remain passed.

## Interpretation boundary

Passing establishes optimal average linear crosstalk for this coded triality
memory under isotropic values.  It does not exceed the information bound,
guarantee worst-case error, learn the code assignments, or make the
construction novel by itself.  Its research value is the compatibility of the
optimal frame code with shared Spin(8) transport and an exact associative
state-space scan.
