# Spin(8) Q8 regular-orbit retraction: seed-4 diagnostic protocol

Status: frozen after the affine-orbit diagnostic failed and before applying
this construction to seed 4.

Date: 2026-08-03

This is post-cohort exploratory diagnosis. It cannot change the original 8/9
prospective result and needs untouched seeds for any reliability claim.

## Hypothesis

Seed 4's learned decoder may use Q8 components outside the single faithful 4D
irrep. A rank-four retraction, even with a fixed mean, can discard nontrivial
one-dimensional components and therefore change decoder behavior. The full
8D real regular representation contains every Q8 irrep present in the regular
action and requires no data-dependent rank cutoff.

## Frozen construction

For each channel let `H` be the `8 x 8` matrix of all learned canonical states
and let `R_g` be the exact right-regular permutation action.

1. Form the learned orbit Gram matrix `G = H^T H`.
2. Project it onto the exact regular-action commutant:

   ```text
   Gbar = mean_g R_g^T G R_g.
   ```

3. Take the positive-semidefinite square root `P = sqrt(Gbar)`.
4. Fit one orthogonal Procrustes factor `O` minimizing `||H - O P||_F`.
5. Define the exact target orbit and complete token family by

   ```text
   Hbar = O P
   A_t  = O R_t O^T.
   ```

6. Express every `A_t` through the fixed positive-spinor Lie basis, set the
   initial state to the identity column of `Hbar`, and keep the decoder frozen.

This uses all states and all tokens jointly. It introduces no singular-value
threshold, channel selection, label optimization, or independent token
normalization. The known Q8 table remains an explicit compiler oracle.

## Diagnostic gate

Use the same algebra, dense, long, and streaming thresholds as the prior two
retractions. Report the commutant residual, target-orbit projection RMS,
projected Gram eigenvalues, and decoder behavior. A seed-4 pass is evidence for
the diagnosis only; it is not fresh validation.
