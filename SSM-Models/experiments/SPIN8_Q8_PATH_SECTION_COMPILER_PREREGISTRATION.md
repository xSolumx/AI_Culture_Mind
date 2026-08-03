# Spin(8) Q8 path-section compiler: seed-4 diagnostic

Status: frozen after canonical observer transport failed and before collecting
the calibration paths.

Date: 2026-08-03

This is a post-cohort seed-4 diagnostic, not fresh validation.

## Diagnosed failure

Raw seed 4 is 100% correct at L15/L16, but its arbitrary shortest-word
canonical states decode at only 50%. Transporting those canonical logits
therefore preserves the wrong path gauge exactly. An exact group state needs a
representative section estimated where the learned model is behaviorally valid.

## Frozen calibration protocol

- Generate 32 held-out batches of 512 paths at L15 and another 32 at L16.
- Use deterministic seed base `10,500,000 + 10,000 * model_seed`.
- L15 and L16 are both required because the generator alphabet is bipartite;
  together they cover all eight Q8 endpoints.
- Group paths by their table-computed final element and average the raw final
  recurrent states, yielding one `32 x 8` centroid matrix.
- Calibration must cover every state and raw final accuracy must be reported.
- Evaluation uses the pre-existing dense and long generators with different
  seeds.

## Frozen compiler

1. Apply the full regular-orbit Gram-commutant projection to the eight endpoint
   centroids; no rank threshold is used.
2. Map the exact regular token family through the real positive-spinor
   logarithm.
3. Set the initial state to the projected identity centroid.
4. Transport the observer by the minimum-change pseudoinverse formula from raw
   centroid logits to the exact orbit.
5. Use no target one-hot labels, gradient steps, decoder loss, or evaluation
   data during compilation. The Q8 table and endpoint grouping remain explicit
   oracle inputs.

## Diagnostic gate

- calibration covers 8/8 endpoints and raw final accuracy is at least 99%;
- exact action reconstruction and homomorphism RMS are each `<= 1e-5`;
- centroid-logit transport maximum is `<= 1e-5`;
- dense and long member/joint minima are each at least 99%;
- streaming thresholds remain unchanged.

A seed-4 pass validates the path-section mechanism only. Untouched seeds are
required before replacing the original 8/9 reliability result.
