# Cayley-Null Edge-Family Protocol History

This record separates prospective decisions from post-result hardening.  The
ordering is preserved in the execution transcript but was not sealed by
independent timestamped commits.

## Exploratory version 0

- Used four interpolation nodes per squared coordinate.
- Stopped without promotion when the formal degree audit required five nodes.
- Produced no theorem artifact.

## Prospective version 1

- Declared conservative multidegrees `(4,4,4,4)` and `(3,3,3,4)`.
- Required five nodes per coordinate on two disjoint rational grids.
- Required an exact two-character Walsh reduction, 256 signed off-grid
  determinant checks, and native Bernstein nonnegativity.
- Ran before the successful exact reconstruction and positivity result.

## Verification amendment version 2

Added after the successful certificate in response to independent review:

- records Bernstein zero indices and minimum positive coefficients in the raw
  artifact;
- reconstructs Bernstein arrays from stored monomial maps in the lightweight
  artifact verifier;
- compares the two stored coefficient maps directly and compares complete
  stored symmetry/divisibility certificates with freshly recomputed records;
- recomputes all 256 exact holdout determinants in that verifier;
- removes the test's silent pass when the artifact is absent;
- mechanically checks common adjoint conjugacy across `V`, `S+`, and `S-`;
- states the boundary-divisibility proof in the quotient by the four circle
  relations, verifies both analytic branches at all four boundaries, and
  derives the conservative post-division multidegrees from exact block rank.

Version 2 does not rerun either 625-node interpolation grid inside the unit
test.  The full command-line replay does so and remains the proof-bearing
reconstruction path.
