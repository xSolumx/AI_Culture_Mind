# Spin(8) coded-memory preregistration

Date frozen: 2026-08-03, before running the coded-memory harness.

## Correction being tested

For a unit positive-chiral key (p), triality binding defines an orthogonal
map (M(p):S^-\to V).  A single association is therefore exactly invertible,
but a raw superposition is

[
M(p_q)^T\sum_k M(p_k)n_k
= n_q + \sum_{k\ne q}M(p_q)^TM(p_k)n_k.
]

Every wrong-key operator is orthogonal.  Its contribution has full norm; the
usual random-vector inner-product factor (1/8) is not a crosstalk attenuation
factor here.  Raw one-channel binding is consequently a single-slot primitive,
not a high-capacity associative memory.

## Coded multiplicity construction

Use (H) vector multiplicity channels and a unit code column (c_k\in
\mathbb R^H) for association (k):

[
m_h=\sum_k c_{hk}M(p_k)n_k,qquad
\widehat n_q=\sum_h c_{hq}M(p_q)^Tm_h.
]

Then

[
\widehat n_q=n_q+\sum_{k\ne q}
\langle c_q,c_k\rangle M(p_q)^TM(p_k)n_k.
]

Predictions frozen before execution:

1. orthonormal code columns give exact retrieval for every value and key when
   (K\le H);
2. no linear code matrix with (H<K) can give exact retrieval for all
   associations, because its Gram matrix has rank at most (H);
3. normalized random codes have mean squared relative error approximately
   ((K-1)/H) for independent random values;
4. a shared Spin(8) action transports keys, values, and coded memory
   equivariantly without changing retrieval error;
5. the selective affine transition
   (m_t=d_tV_tm_{t-1}+b_t) remains associative and its parallel prefix scan
   matches recurrent execution.

## Evaluation

- float64 exact tests for (H,K\in\{1,2,4,8,16,32});
- 256 random trials for each random-code cell;
- report raw one-channel, random-code, and orthogonal-code errors separately;
- test a random shared Spin(8) transport action;
- test recurrent/parallel equality for a length-64 selective affine memory
  sequence;
- retain the full raw JSON report.

## Frozen gates

- exact orthogonal-code retrieval: maximum relative error below (10^{-10})
  for every (K\le H) cell;
- equivariant transport: maximum discrepancy below (10^{-10});
- parallel/recurrent parity: maximum discrepancy below (10^{-10});
- random-code empirical mean squared error must be within 20% relative error
  of ((K-1)/H) whenever that prediction is nonzero.

## Interpretation boundary

Passing proves an exact, scan-compatible coded associative memory with capacity
linear in multiplicity width.  It does not prove compression beyond the state
dimension, learned addressing, language-model utility, or novelty relative to
all associative-memory literature.  For (K\le H), exact cancellation spends
one independent multiplicity degree of freedom per arbitrary 8D value.
