# Spin(8) Q8 path-section compiler: untouched validation cohort

Status: frozen before training Spin(8) seeds 10--18.

Date: 2026-08-03

Seeds 0--9 are excluded: seed 0 selected the original orbit retraction, seeds
1--9 formed its fresh cohort, and seed 4 selected the path-section refinement.
Seeds 10--18 are untouched for this Spin(8) compiler.

## Fixed protocol

- unchanged positive-chiral four-channel 2,000-step Q8 curriculum;
- unchanged path-section calibration: 32 x 512 paths at each of L15 and L16,
  with the fixed disjoint calibration seed formula;
- unchanged full regular Gram-commutant projection;
- unchanged real Schur logarithm with paired `-1` modes;
- unchanged minimum-change observer transport;
- no target one-hot labels or gradient steps after compilation;
- unchanged dense and long evaluation generators.

The compiler must not be tuned or skipped per seed. A seed whose raw model is
below 99% on calibration counts as a pipeline failure rather than being
silently excluded.

## Frozen reporting and gates

Apply every per-seed gate from
`SPIN8_Q8_PATH_SECTION_COMPILER_PREREGISTRATION.md`. Also report:

- raw versus compiled dense accuracy;
- calibration endpoint counts;
- centroid projection spectra and exact-section condition number;
- observer displacement;
- raw 28D tangent parameter norms per token/channel, to audit large-angle
  exponential aliasing;
- all failed seeds individually.

The validation supports reliability at 8/9 passes and uniform reliability only
at 9/9. The original frozen-decoder 8/9 cohort remains a separate result.
