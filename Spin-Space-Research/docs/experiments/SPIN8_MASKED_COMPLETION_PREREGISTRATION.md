# Spin(8) masked cross-representation completion preregistration

Date frozen: 2026-08-03, before training.

## Mechanism

For a unit positive spinor key `s+`, triality defines an orthogonal map

```text
C_s+(s-)[i] = s-^T rho[i] s+.
```

Its adjoint should recover the negative spinor from the key and bound vector.
This experiment masks `S-` and asks models to complete it from `(S+,V)` after
shared Spin(8) transport.

## Data split

- Freeze three random shared Spin(8) generator actions.
- Train only on words of lengths 1--8 over generators 0 and 1.
- Select hyperparameters without generator 2.
- Test on words composed exclusively from held-out generator 2 and on mixed
  three-generator words at lengths 8, 32, 128, and 512.
- Initial positive keys are continuously sampled from a narrow cap around one
  fixed basis spinor. Initial negative values are sampled from narrow caps
  around four fixed basis spinors. This prevents rotationally invariant initial
  data from making the held-out-generator split distributionally vacuous.
  Targets are the transported negative spinors.

## Families

1. fixed exact triality tensor, no learned binding parameters;
2. learned unconstrained 512-parameter bilinear tensor;
3. approximately parameter-matched two-layer MLP on concatenated `(S+,V)`;
4. zero and direct linear sanity baselines in evaluation.

The action transport is an oracle shared across families. This isolates the
cross-representation completion mechanism; it is not yet a learned recurrent
controller comparison.

## Frozen gates

- Exact triality: maximum MSE `<=1e-12` and cosine `>=0.999999` at every split
  and length.
- Learned family reliability: report every seed; no pass-rate claim below
  three seeds.
- A learned bilinear pass requires cosine `>=0.99` at every held-out-generator
  and mixed length plus tensor cosine `>=0.99` with the exact triality tensor.
- MLP is a serious baseline, not expected to fail by definition; report its
  complete distribution.
- No attention-replacement or language-model claim follows from a pass.
