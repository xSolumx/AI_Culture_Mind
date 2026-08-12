# Spin(8) triality identifiability preregistration

Date frozen: 2026-08-03, before the invariant-space audit or training.

## Question

Does imposing the exact shared Spin(8) symmetry reduce cross-representation
binding from a 512-parameter interpolation problem to an identifiable
one-parameter law under deliberately incomplete observations?

## Algebra gate

Numerically construct the infinitesimal equivariance constraints on

```text
T in Hom(S+ tensor S-, V).
```

Across all 28 generators, the nullspace must have dimension one and its unit
basis must have absolute cosine at least `1-1e-10` with the fixed octonionic
triality tensor.

## Rank-deficient training split

- positive source state is exactly basis spinor `e0`;
- negative source state is one of `e0..e3`;
- training words are powers of generator 0 with lengths 1--4;
- therefore at most 16 distinct observation cells exist;
- verify bilinear feature rank is strictly below 64 before training;
- evaluation uses continuously perturbed source caps and held-out generator 2
  or mixed generators at lengths 8, 32, 128, and 512.

## Families

1. fixed exact triality;
2. one learned scalar multiplying the exact invariant tensor;
3. unconstrained 512-parameter bilinear tensor;
4. 608-parameter MLP.

## Frozen interpretation

- The invariant model passes only if every one of three seeds has mean cosine
  at least `0.9999` in every test cell.
- The unconstrained models must be allowed to fit the observed cells. A failure
  off-orbit is meaningful only if final training MSE is below `1e-5`.
- The hypothesis is supported strongly if the invariant model passes 3/3 while
  an unconstrained family fits training but has a worst-cell mean cosine below
  `0.95` in at least 2/3 seeds.
- If the generic bilinear family also generalizes, then the incomplete cells
  plus optimizer bias were sufficient and the claimed sample-efficiency
  advantage is not established.
- This remains a supplied-action mechanistic task, not a language claim.
