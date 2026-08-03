# Spin(8) Q8 orbit-family retraction: prospective protocol

Status: frozen before applying the retraction to the seed-0 checkpoint.

Date: 2026-08-03

## Observation motivating the intervention

The first `pure_spin8_positive` Q8 curriculum run is perfect through base
length 32 but fails at longer lengths. Its learned canonical-orbit matrices
have four dominant singular values and four much smaller values in the three
useful channels. This is consistent with a learned faithful four-real
quaternionic orbit embedded in the eight-real chiral state, plus an
underconstrained complement.

This observation selects the retraction method. It does not set its gate.

## Frozen family-level construction

For each channel:

1. Generate the eight canonical learned Q8 states from the learned initial
   state and the complete four-token action family.
2. Pair each element with its central negative and form the four signed
   differences for `(1, i, j, k)`.
3. Apply one rectangular polar decomposition to these four differences,
   producing one shared orthonormal `8 x 4` active frame `E`.
4. Derive every token action from the exact Q8 right-multiplication law on
   that same frame:

   ```text
   A_t = E R_t E^T + (I - E E^T).
   ```

5. Express each exact `A_t` through the fixed positive-chiral Spin(8) Lie
   basis and replace the full token-action family jointly.
6. Replace the initial orbit state with the identity column of the shared
   frame. Keep the trained decoder completely frozen.

No token is normalized, rounded, or fitted independently. No target label is
used to tune the decoder. The Q8 table determines relations only after the
unconstrained tangent training has finished.

## Frozen gates

- every retracted action is orthogonal with determinant `+1`;
- token-action reconstruction through the Spin(8) exponential has maximum
  absolute residual `<= 1e-5` in the float32 checkpoint;
- full-space homomorphism RMS is `<= 1e-5`;
- central-pair member and joint accuracy are each at least `99%` at every
  preregistered base length from 15 through 256;
- streaming-state residual is `<= 1e-5` and streaming-logit residual is
  `<= 1e-4`;
- per-channel orbit singular values, frame projection displacement, and action
  displacement are reported whether the gate passes or fails.

This seed-0 run is a mechanistic smoke test, not a reliability claim. A pass
justifies a separately frozen fresh-seed cohort; a failure falsifies this
specific orbit-first projection without falsifying chiral Spin(8) capacity.
