# Two-Edge Shared-Grid Reconstruction Preregistration

**Frozen:** 2026-08-06, before shared-grid determinant evaluation

## Correction to the cost model

The amplitude theorem listed `61,321` points for eight separate anisotropic
sector grids. That count is mathematically correct but computationally wasteful.
At one parameter point, the same eight quotient-representative direct
determinants recover all eight Walsh coefficients simultaneously. Sector
evaluations must therefore be shared.

The componentwise maximum of the eight proved residual degree bounds is

```text
(4,4,4,4,4,4).
```

One common five-node tensor grid has only

```text
5^6 = 15,625 points.
```

It reconstructs every sector and must additionally verify that coefficients
above each sector's tighter individual bound vanish exactly. This reduces the
one-grid direct determinant count from `490,568` to `125,000`.

## Frozen protocol

1. Use two disjoint five-node rational-circle grids, `alpha` and `beta`.
2. At every point, evaluate exactly eight quotient-representative `28 x 28`
   information determinants.
3. Recover all eight Walsh sectors from the same Hadamard transform.
4. Divide only by globally proved amplitudes: the character monomial and the
   common `s^6` factor. Do not divide by slice-suggested endpoint factors.
5. Split each grid into 25 tiles indexed by the first two axes. Each tile has
   `5^4=625` points and `5,000` direct determinant evaluations.
6. Save and hash every tile before assembly.
7. Reconstruct all eight coefficient maps exactly.
8. Require exact zero for every coefficient outside each sector's preproved
   anisotropic degree bound.
9. Require complete coefficient-map equality between grids, not merely equal
   hashes.
10. Require at least 32 fresh off-grid points. At each point compare all eight
    reconstructed sectors against the same eight new direct determinants.

## Prospective structural audit

After the frozen reconstruction passes, and only then:

- factor each sector over `QQ`;
- test the endpoint factors suggested by the old slice atlas;
- test whether late-coordinate dependence lies in nested Cholesky boundary
  ideals, as proved for sector `110101`;
- report counterexamples as readily as confirmations.

The exact `110101` result motivated these questions but cannot be used as a
license to divide the other sectors before reconstruction.

## Interpretation boundaries

- Matching coefficient maps prove exact reconstruction under the already
  proved degree ceiling.
- Factorization of sectors is not positivity of orientation margins.
- Failure of one tile or process is not mathematical evidence.
- Only a completed two-grid comparison plus fresh holdouts can promote the
  all-sector polynomial record.
- The unrestricted family still contains the final residual `h`; this campaign
  remains scoped to the preregistered `h=0` bridge.
