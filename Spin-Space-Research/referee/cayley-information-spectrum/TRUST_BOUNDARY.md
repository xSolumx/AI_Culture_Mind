# Trust boundary

This package separates four logically different layers.

## A. Ordinary mathematical input

The package uses the classical facts that a unit vector in the eight-
dimensional triality representations has `Spin(7)` stabilizer and that the
`Spin(7)` action on oriented four-planes has a one-dimensional Cayley orbit
coordinate. The latter orbit description is recorded explicitly in the
discussion of the Cayley hyperbolic plane in Berndt--Tamaru and is rooted in
the classical Cayley calibration of Harvey--Lawson.

These are cited inputs, not discoveries of this repository.

## B. Project-specific structural input

The full repository constructs rational generator matrices for the three
triality views and checks:

- common Lie brackets and triality equivariance;
- the exact `28 x 28` information family;
- fixed support components of dimensions `8+8+8+4`;
- the four displayed characteristic polynomials;
- a signed-permutation conjugacy of the twin eight-dimensional blocks;
- principal and endpoint isotropy restriction ranks used in the `2+2` flag
  quotient.

Those checks are replayed by the maintained SymPy/FLINT harnesses. The minimal
verifier does **not** reconstruct them.

## C. Independently reconstructed algebra

[`verify.py`](verify.py) uses no project code and no CAS. Given the four block
laws, it recomputes with exact rational arithmetic:

- every coefficient of the degree-28 characteristic polynomial;
- the determinant and first two direct moments;
- both inverse moments and their derivative identities;
- the positive Bernstein coefficients used for the second inverse moment;
- endpoint factors, rank, and equal first-order slopes.

It then compares the complete reconstruction with the stored coefficient
artifact byte-for-structure, rather than trusting a stored `passed` flag.

## D. What remains a referee obligation

A referee should still inspect:

1. whether the chosen rational generator matrices realize the claimed common
   `Spin(8)` normalization;
2. whether the information-projector definition matches the intended design
   problem;
3. whether the global balanced-flag normal form follows from the cited orbit
   description and the stated isotropy argument;
4. whether the manuscript's terminology—especially “complete orthonormal
   balanced family”—matches that normal form exactly.

The SHA-256 manifests prove file identity only. Exact arithmetic rules out
rounding error inside the verified identities; it does not make an incorrect
modeling assumption true.

## Upstream verifier disclosure

The maintained criteria script imports the block formulas from the maintained
block module. The FLINT cross-check changes the arithmetic backend for selected
polynomial operations but imports the same rational coefficient maps. Neither
is described as a wholly independent reconstruction from first principles.
This compact verifier is independent only from the block-laws-to-spectrum
stage onward, and says so explicitly.
