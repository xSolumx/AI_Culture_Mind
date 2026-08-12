# Spin(8) Q8 exact-family observer transport: seed-4 diagnostic

Status: frozen after both frozen-decoder seed-4 refinements failed and before
transporting the observer.

Date: 2026-08-03

This is post-cohort exploratory diagnosis. It cannot alter the original 8/9
reliability result and needs untouched seeds for a new reliability claim.

## Hypothesis

An exact state-space realization requires both the transition family and its
observer. A nonlinear orbit projection is not a pure global basis change, so
`W_new = W_old U^T` is generally invalid. The correct transport is determined
on the reachable canonical subspace.

## Frozen construction

Use the threshold-free full regular-orbit retraction to obtain exact
`X_new in R^(32 x 8)` from the raw learned canonical orbit
`X_old in R^(32 x 8)`. Keep the output bias and logit scale fixed. Set

```text
Y_old = W_old X_old
W_new = W_old + (Y_old - W_old X_new) pinv(X_new).
```

This is the minimum-Frobenius-change linear observer satisfying
`W_new X_new = Y_old` when `X_new` has full column rank. It uses the old model's
logits as a teacher but no target class labels, loss optimization, or gradient
steps. The exact token actions and initial orbit are not modified after the
regular retraction.

## Diagnostic gate

- exact-family algebra and streaming gates remain unchanged;
- `rank(X_new) = 8`;
- canonical pre-scale logit transport maximum residual `<= 1e-5`;
- dense and long member/joint minima each at least `99%`;
- report observer displacement RMS and condition number of `X_new`.

A pass establishes a behavior-preserving compiler for this observed seed. It
does not establish table-blind discovery or fresh-seed reliability.
