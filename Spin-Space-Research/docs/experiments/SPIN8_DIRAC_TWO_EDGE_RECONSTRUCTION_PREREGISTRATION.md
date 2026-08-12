# Two-Edge Sector Reconstruction Preregistration

**Frozen:** 2026-08-06, before full-grid evaluation

## First target

Reconstruct sector `110101` first. The exact chart-sign and boundary-rank
certificate gives its residual polynomial the conservative multidegree bound

```text
(3,3,3,3,3,3)
```

in variable order `(a2,d2,e2,g2,i2,c2)`. Its proof-safe tensor grid therefore
has `4^6 = 4096` points. This sector was selected because it has the smallest
formal grid, not because a favorable full-grid result was inspected.

## Frozen protocol

1. Use the globally proved amplitude
   `s^6 * a*A*d*D*E*g*G*I*c`.
2. Evaluate one exact Walsh coefficient from eight quotient-representative
   direct `28 x 28` determinants at each point.
3. Split each grid into four leading-axis slabs and save every slab before
   assembly.
4. Reconstruct all coefficients exactly on grid `alpha`.
5. Repeat on a disjoint grid `beta`.
6. Require complete coefficient-map equality, not hash equality alone.
7. Require at least 32 fresh exact off-grid points, each checked against all
   relevant direct orientation determinants.

## Interpretation

- One-grid reconstruction under the proved degree ceiling is algebraically
  sufficient to identify the polynomial.
- The second grid and holdouts are independent implementation safeguards.
- Vanishing high-degree coefficients may tighten this sector's degree, but do
  not tighten any other sector by analogy.
- A pass proves one residual sector polynomial. It does not prove positivity
  of the eight orientation margins or the two-edge theorem.

## Crash boundary

No process evaluates more than one 1024-point slab. Completed slabs are
content-addressed and reusable. A crash or timeout cannot be interpreted as a
mathematical result.
